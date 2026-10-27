# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full multimodal model vs HF fp32 `EmbeddingGemma2Model` + sentence-transformers pooling, real weights.

Inputs: reference/full_model_inputs.py (text incl. multilingual and a 2080-token document, images,
images with captions, an 8-frame video with text, speech clips with and without text, two mixed
items). Each item goes through the public API (`EmbeddingGemma2.encode`, eager) and through HF
with the same processor output.

    test_input_ids_match_sentence_transformers  wrapper ids == ids ST 6.2.0 built (sha256, CPU only)
    test_merge_matches_hf                       on-device merged inputs_embeds vs HF, row by row
    test_full_model_item                        per item: cosine(TT, HF) >= 0.99 at 768 and MRL 256
    test_batched_mixed_modalities               one encode() call over a heterogeneous list
                                                (batches: 3 texts, 2 image+caption items, 2 audio items)
    test_retrieval                              texts x images, texts x audios: TT argmax == HF argmax
    test_no_host_fallback_in_full_forward       no torch / host transfer / ttnn fallback in one
                                                image + audio + text forward
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from torch.overrides import TorchFunctionMode

import ttnn
from models.demos.blackhole.embedding_gemma2.reference import full_model_inputs as F
from models.demos.blackhole.embedding_gemma2.tests.common import COSINE_EMBEDDING, log_result

DOC_DIR = Path(__file__).resolve().parents[2] / "doc" / "full_model"
MRL_DIM = 256


def all_items() -> dict:
    out = F.items()
    out.update(F.IMAGE_RETRIEVAL_TEXTS)
    out.update(F.AUDIO_RETRIEVAL_TEXTS)
    return out


ITEMS = (
    "text_query",
    "text_doc",
    "text_de",
    "text_ja",
    "text_es",
    "text_long",
    "image_house",
    "image_pajamas",
    "image_dog",
    "image_handwriting",
    "image_house_caption",
    "image_dog_caption",
    "image_pajamas_caption",
    "video_text",
    "audio_speech2s",
    "audio_speech10s",
    "audio_tone3s",
    "audio_speech2s_text",
    "audio_speech10s_text",
    "mixed_image_audio",
    "mixed_two_images",
)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1))


# ---------------------------------------------------------------------- fixtures (imported by the
# other test_full_model_*.py modules)
@pytest.fixture(scope="session")
def full_items():
    return all_items()


@pytest.fixture(scope="session")
def hf_full(eg2_snapshot):
    from models.demos.blackhole.embedding_gemma2.reference.hf_full_reference import load_reference_full_model

    return load_reference_full_model(eg2_snapshot)


@pytest.fixture(scope="session")
def hf_processor(eg2_snapshot):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(eg2_snapshot))


@pytest.fixture(scope="session")
def hf_embedding(hf_full, hf_processor, full_items):
    """hf_embedding(name, truncate_dim=None) -> [d] HF fp32 ST embedding (cached)."""
    from models.demos.blackhole.embedding_gemma2.reference.hf_full_reference import embed, truncate
    from models.demos.blackhole.embedding_gemma2.tt.embedding_model import process_item

    cache = {}

    def get(name, truncate_dim=None):
        if name not in cache:
            cache[name] = embed(hf_full, process_item(hf_processor, full_items[name]))[0]
        return truncate(cache[name], truncate_dim)

    return get


@pytest.fixture(scope="module")
def full_model(eg2_device, eg2_snapshot):
    from models.demos.blackhole.embedding_gemma2.tt.embedding_model import EmbeddingGemma2

    return EmbeddingGemma2.from_pretrained(eg2_device, str(eg2_snapshot), max_seq_len=8192)


# ---------------------------------------------------------------------- tests
def test_input_ids_match_sentence_transformers(hf_processor, full_items):
    """The wrapper's formatting gives the token ids sentence-transformers 6.2.0 gave (recorded sha256)."""
    from models.demos.blackhole.embedding_gemma2.tt.embedding_model import process_item

    st = json.loads((DOC_DIR / "st_reference.json").read_text())["items"]
    bad = []
    for name, rec in st.items():
        ids = process_item(hf_processor, full_items[name])["input_ids"][0].to(torch.int64)
        sha = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        if sha != rec["input_ids_sha256"] or ids.numel() != rec["tokens"]:
            bad.append((name, ids.numel(), rec["tokens"]))
    logger.info(f"input ids checked against ST for {len(st)} items; mismatches: {bad}")
    assert not bad


@pytest.mark.parametrize("name", ["mixed_image_audio", "mixed_two_images", "video_text"])
def test_merge_matches_hf(name, full_model, hf_full, hf_processor, full_items):
    """The on-device merge output (inputs_embeds entering the text encoder) vs HF's merged
    inputs_embeds, row by row: text rows equal up to bf16, every soft-token row sits at its HF
    position (a misplaced row would pair unrelated tokens and pull the mean row cosine down)."""
    from models.demos.blackhole.embedding_gemma2.reference.hf_full_reference import model_inputs
    from models.demos.blackhole.embedding_gemma2.tt.embedding_model import process_item

    feats = process_item(hf_processor, full_items[name])
    seen = {}

    def grab(_mod, _args, kwargs):
        seen["embeds"] = kwargs["inputs_embeds"].detach().clone()

    h = hf_full.language_model.register_forward_pre_hook(grab, with_kwargs=True)
    try:
        with torch.no_grad():
            hf_full(**model_inputs(feats))
    finally:
        h.remove()
    ref = seen["embeds"][0].float()  # [L, 512]
    m = full_model
    p = m.process(full_items[name])
    host = m.prepare_batch([p])
    x = m.to_device(host)
    m.prepare(host.key, x)
    merged = m.model.merge(x)
    tt = ttnn.to_torch(merged).float().reshape(host.key.seq_len, -1)[: p.length]
    ttnn.deallocate(merged)
    ids = p.input_ids
    ok = True
    text_rows = torch.ones_like(ids, dtype=torch.bool)
    for modality, tok in m.token_ids.items():
        rows = ids == tok
        text_rows &= ~rows
        if rows.any():
            c = torch.nn.functional.cosine_similarity(tt[rows], ref[rows], dim=-1)
            ok &= log_result(
                "merge_row_cosine_mean", float(c.mean()), 0.99, item=name, modality=modality, rows=int(rows.sum())
            )
            log_result("merge_row_cosine_min", float(c.min()), 0.0, item=name, modality=modality)
    d = float((tt[text_rows] - ref[text_rows]).abs().max() / ref[text_rows].abs().max())
    ok &= log_result("merge_text_rows_rel_max_diff", -d, -0.01, item=name)  # bf16 rounding only
    assert ok


@pytest.mark.parametrize("name", ITEMS)
def test_full_model_item(name, full_model, hf_embedding, full_items):
    tt = full_model.encode([full_items[name]])[0]
    tt_mrl = full_model.encode([full_items[name]], truncate_dim=MRL_DIM)[0]
    ref = hf_embedding(name)
    ref_mrl = hf_embedding(name, MRL_DIM)
    p = full_model.process(full_items[name])
    key = full_model.batch_key([p])
    assert abs(float(tt.norm()) - 1.0) < 1e-3 and abs(float(tt_mrl.norm()) - 1.0) < 1e-3
    ok = log_result("full_model_cosine", cos(tt, ref), COSINE_EMBEDDING, item=name, tokens=p.length, key=str(key))
    ok &= log_result("full_model_cosine_mrl256", cos(tt_mrl, ref_mrl), COSINE_EMBEDDING, item=name, key=str(key))
    assert ok


def test_batched_mixed_modalities(full_model, hf_embedding, full_items):
    """One encode() over a heterogeneous list; same-structure items share a forward (batch > 1)."""
    names = [
        "text_query",
        "image_house_caption",
        "audio_speech2s_text",
        "text_de",
        "image_pajamas_caption",
        "audio_speech10s_text",
        "text_doc",
        "mixed_image_audio",
    ]
    _, batches = full_model.plan([full_items[n] for n in names])
    keys = [str(full_model.prepare_batch([full_model.process(full_items[names[i]]) for i in b]).key) for b in batches]
    logger.info(f"batches {[[names[i] for i in b] for b in batches]} keys {keys}")
    assert max(len(b) for b in batches) >= 2
    out = full_model.encode([full_items[n] for n in names])
    ok = True
    for i, n in enumerate(names):
        ok &= log_result("full_model_batched_cosine", cos(out[i], hf_embedding(n)), COSINE_EMBEDDING, item=n)
    assert ok


def _matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.float() @ b.float().T


@pytest.mark.parametrize("kind", ["image", "audio"])
def test_retrieval(kind, full_model, hf_embedding, full_items):
    """Rows = captions (text-only), columns = media-only items: TT and HF pick the same column per row."""
    caps = F.IMAGE_RETRIEVAL_TEXTS if kind == "image" else F.AUDIO_RETRIEVAL_TEXTS
    media = [f"{kind}_{k.removeprefix('caption_')}" for k in caps]
    rows, cols = list(caps), media
    tt_r = full_model.encode([full_items[n] for n in rows])
    tt_c = full_model.encode([full_items[n] for n in cols])
    hf_r = torch.stack([hf_embedding(n) for n in rows])
    hf_c = torch.stack([hf_embedding(n) for n in cols])
    s_tt, s_hf = _matrix(tt_r, tt_c), _matrix(hf_r, hf_c)
    arg_tt, arg_hf = s_tt.argmax(-1), s_hf.argmax(-1)
    top2 = s_hf.topk(2, dim=-1).values
    result = {
        "rows": rows,
        "columns": cols,
        "tt": [[round(float(v), 5) for v in r] for r in s_tt],
        "hf_fp32": [[round(float(v), 5) for v in r] for r in s_hf],
        "argmax_tt": [cols[i] for i in arg_tt],
        "argmax_hf": [cols[i] for i in arg_hf],
        "hf_top1_minus_top2": [round(float(t[0] - t[1]), 5) for t in top2],
        "max_abs_diff": round(float((s_tt - s_hf).abs().max()), 6),
        "agree": bool(torch.equal(arg_tt, arg_hf)),
    }
    logger.info(f"retrieval {kind}: {json.dumps(result)}")
    out_dir = os.environ.get("EMBEDDING_GEMMA2_ARTIFACTS")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"retrieval_{kind}.json").write_text(json.dumps(result, indent=1) + "\n")
    log_result("retrieval_argmax_agree", float(result["agree"]), 1.0, kind=kind, max_abs_diff=result["max_abs_diff"])
    assert result["agree"], f"argmax differs: TT {result['argmax_tt']} HF {result['argmax_hf']}"


# ---------------------------------------------------------------------- fallback audit
_HOST_ENTRY_POINTS = (
    "to_torch",
    "from_torch",
    "as_tensor",
    "to_device",
    "from_device",
    "copy_host_to_device_tensor",
    "copy_device_to_host_tensor",
)


class _RecordTorch(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append(getattr(func, "__qualname__", repr(func)))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("name", ["mixed_image_audio", "video_text"])
def test_no_host_fallback_in_full_forward(name, eg2_device, full_model, full_items, monkeypatch):
    """Towers + merge + text encoder + pooling in one forward: no torch call, no host transfer,
    no ttnn op falling back to torch. The Tracy report of one image item (test_full_model_perf.py)
    is the device-side evidence."""
    m = full_model
    host = m.prepare_batch([m.process(full_items[name])])
    x = m.to_device(host)
    m.prepare(host.key, x)
    ttnn.deallocate(m.forward(x, host.key))  # compile pass
    ttnn.synchronize_device(eg2_device)

    def _forbidden(n):
        def raiser(*_a, **_k):
            raise AssertionError(f"host transfer ttnn.{n} called inside the forward")

        return raiser

    originals = {n: getattr(ttnn, n) for n in _HOST_ENTRY_POINTS}
    for n in _HOST_ENTRY_POINTS:
        monkeypatch.setattr(ttnn, n, _forbidden(n))
    rec = _RecordTorch()
    with ttnn.manage_config("throw_exception_on_fallback", True), rec:
        out = m.forward(x, host.key)
    for n, f in originals.items():
        monkeypatch.setattr(ttnn, n, f)
    ttnn.synchronize_device(eg2_device)
    logger.info(f"torch calls inside the full forward ({host.key}): {len(rec.calls)} {sorted(set(rec.calls))[:20]}")
    assert rec.calls == [], f"torch functions called inside the forward: {sorted(set(rec.calls))}"
    assert torch.isfinite(ttnn.to_torch(out).float()).all()
