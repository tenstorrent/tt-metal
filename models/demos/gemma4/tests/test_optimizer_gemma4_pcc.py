# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model correctness gate for optimizer runs on Gemma 4 (the gemma4 demo code, any variant in HF_MODEL).

The Gemma 4 counterpart of models/tt_transformers/tests/test_optimizer_pcc.py, with the same checks
and the same printed lines, so an optimizer reads both gates the same way:

1. PCC of the logits at every teacher-forced position -- the prefill's last position, then
   FORCED_TOKENS decode steps -- against the Hugging Face bf16 reference, held to the floor
   PCC_THRESHOLD on the worst position ("PCC: x" is the number an optimizer parses).
2. Top-1 / top-5 agreement with the reference's argmax and the mean correlation, held RELATIVE to a
   baseline pinned from the unmodified tree (generated/optimizer_accuracy_baseline_<model dir>.json).

The model is built with gemma4's own create_tt_model, prefilled with ttnn_prefill_forward, and decoded
one teacher-forced token at a time with ttnn_decode_forward, untraced -- the calls
tests/unit/test_model.py::test_full_model_decode makes, extended to every position. Weights come from
the converted-weight store (models/tt_transformers/tests/optimizer_weight_cache.py) through a fresh
per-run TT_CACHE_PATH: gemma4's own warm cache reloads tensorbins straight onto the mesh, which is the
pinned-memory path that stalls on this QB2.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.tt_transformers.tests.optimizer_weight_cache import RunCache

# The checkpoint, stated so an optimizer's roofline can resolve the model from the test it runs.
HF_MODEL_ID = os.environ.get("HF_MODEL_ID") or "google/gemma-4-26B-A4B-it"
MESH_SHAPE = (1, 4)  # QB2: four Blackhole chips in a row, TP=4

# Absolute floor for the worst logits PCC over every teacher-forced position: a model that no longer
# correlates with the reference at all. Not the tt-transformers 0.90: the unmodified tree's worst
# position on this checkpoint is 0.633 (step 79 of 128, 2026-09-28), and gemma4's own full-model
# check passes at 0.84 on one position (tests/pcc_thresholds.json). The pinned top-1/top-5/mean
# checks below are what judge a change; this only refuses a broken model.
PCC_THRESHOLD = 0.50
MAX_SEQ_LEN = 1024
PROMPT_TOKENS = 128
FORCED_TOKENS = 128

# Relative floors against the pinned baseline (same as the tt-transformers gate). Over 129 positions
# one token is 0.78 points, so 1.0 allows a single flip and refuses two.
TOP1_DROP_PTS = float(os.environ.get("PCC_GATE_TOP1_DROP_PTS", "1.0"))
TOP5_DROP_PTS = float(os.environ.get("PCC_GATE_TOP5_DROP_PTS", "1.0"))
MEAN_PCC_DROP = float(os.environ.get("PCC_GATE_MEAN_PCC_DROP", "0.002"))

GENERATED = Path(__file__).resolve().parents[4] / "generated"
CACHE_ROOT = GENERATED / "optimizer_cache"
REFERENCE_ROOT = GENERATED / "optimizer_reference"

# Public-domain English (Declaration of Independence, 1776), the text the tt-transformers gate uses.
TEXT = (
    "When in the Course of human events, it becomes necessary for one people to dissolve the "
    "political bands which have connected them with another, and to assume among the powers of the "
    "earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle "
    "them, a decent respect to the opinions of mankind requires that they should declare the causes "
    "which impel them to the separation. We hold these truths to be self-evident, that all men are "
    "created equal, that they are endowed by their Creator with certain unalienable Rights, that "
    "among these are Life, Liberty and the pursuit of Happiness. That to secure these rights, "
    "Governments are instituted among Men, deriving their just powers from the consent of the "
    "governed, That whenever any Form of Government becomes destructive of these ends, it is the "
    "Right of the People to alter or to abolish it, and to institute new Government, laying its "
    "foundation on such principles and organizing its powers in such form, as to them shall seem "
    "most likely to effect their Safety and Happiness. Prudence, indeed, will dictate that "
    "Governments long established should not be changed for light and transient causes; and "
    "accordingly all experience hath shewn, that mankind are more disposed to suffer, while evils "
    "are sufferable, than to right themselves by abolishing the forms to which they are accustomed. "
    "But when a long train of abuses and usurpations, pursuing invariably the same Object evinces a "
    "design to reduce them under absolute Despotism, it is their right, it is their duty, to throw "
    "off such Government, and to provide new Guards for their future security. Such has been the "
    "patient sufferance of these Colonies; and such is now the necessity which constrains them to "
    "alter their former Systems of Government. The history of the present King of Great Britain is "
    "a history of repeated injuries and usurpations, all having in direct object the establishment "
    "of an absolute Tyranny over these States. To prove this, let Facts be submitted to a candid "
    "world."
)


def _model_path() -> str:
    return os.environ["HF_MODEL"]


def _baseline_path() -> Path:
    return GENERATED / f"optimizer_accuracy_baseline_{Path(_model_path()).name}.json"


def _pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    return float(torch.corrcoef(torch.stack((actual, expected)))[0, 1])


def _agreement(actual: torch.Tensor, expected: torch.Tensor) -> tuple[bool, bool]:
    """(top-1 match, reference argmax within the model's top 5) for one position."""
    ref = int(expected.reshape(-1).argmax())
    top5 = actual.reshape(-1).float().topk(5).indices.tolist()
    return top5[0] == ref, ref in top5


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def encode_text(tokenizer) -> list[int]:
    """TEXT's tokens, starting with BOS: gemma4's tokenizer adds none, and Gemma expects it."""
    tokens = tokenizer.encode(TEXT)
    if tokenizer.bos_token_id is not None and (not tokens or tokens[0] != tokenizer.bos_token_id):
        tokens = [tokenizer.bos_token_id, *tokens]
    return tokens


def _tokens(tokenizer) -> list[int]:
    tokens = encode_text(tokenizer)
    needed = PROMPT_TOKENS + FORCED_TOKENS
    assert len(tokens) >= needed, f"TEXT tokenizes to {len(tokens)} tokens; the gate needs {needed}"
    return tokens[:needed]


def _reference_logits(model_path: str, tokens: list[int]) -> torch.Tensor:
    """HF bf16 logits for these tokens, [len(tokens), vocab], from the on-disk cache when it holds them."""
    key = hashlib.sha256((Path(model_path).name + ":" + ",".join(map(str, tokens))).encode()).hexdigest()[:16]
    path = REFERENCE_ROOT / f"{Path(model_path).name}-{key}.pt"
    if path.is_file():
        return torch.load(path)
    reference = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True).eval()
    with torch.no_grad():
        logits = reference(torch.tensor([tokens], dtype=torch.long)).logits[0].float()
    reference = None
    gc.collect()
    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(logits, path)
    return logits


def _load_baseline() -> dict | None:
    try:
        doc = json.loads(_baseline_path().read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("top1_pct") is not None else None


def _vocab_logits(tensor, vocab_size: int) -> torch.Tensor:
    """A mesh tensor's logits as one host row per position, [positions, vocab]: gathered when vocab-sharded."""
    shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(tensor)]
    if shards[0].shape[-1] >= vocab_size:
        full = shards[0]
    else:
        full = torch.cat(shards, dim=-1)
    return full.reshape(-1, full.shape[-1])[:, :vocab_size]


@pytest.mark.no_reset_default_device
@pytest.mark.timeout(3600)
def test_optimizer_gemma4_pcc(monkeypatch):
    """Teacher-forced prefill + FORCED_TOKENS decode steps against Hugging Face, every layer."""
    from models.demos.gemma4.tt.common import create_tt_model

    model_path = _model_path()
    mesh_device = model = model_args = tt_kv_cache = None
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = RunCache(CACHE_ROOT, "gemma4-pcc-")
    monkeypatch.setenv("TT_CACHE_PATH", cache.path)
    # Under `pytest -s` the first print lands on the node id's line, which contains "pcc"; the
    # optimizer's parser reads any "pcc ... <float>" on a line as a measurement. End that line first.
    print("", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokens = _tokens(tokenizer)
        prompt = torch.tensor([tokens[:PROMPT_TOKENS]], dtype=torch.long)
        forced = tokens[PROMPT_TOKENS:]
        hf_logits = _reference_logits(model_path, tokens)

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*MESH_SHAPE), l1_small_size=24576, num_command_queues=1
        )
        model_args, model, tt_kv_cache, _state_dict = cache.build(
            lambda: create_tt_model(
                mesh_device=mesh_device,
                max_batch_size=1,
                max_seq_len=MAX_SEQ_LEN,
                model_path=model_path,
                create_kv_cache=True,
            )
        )
        _state_dict = None
        gc.collect()
        cache.loaded()
        vocab = model_args.vocab_size
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        t0 = time.perf_counter()
        tokens_tt = ttnn.from_torch(
            prompt.to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        embeds = ttnn.to_layout(
            ttnn.reshape(model.embed_tokens(tokens_tt), (1, 1, PROMPT_TOKENS, model_args.hidden_size)),
            ttnn.TILE_LAYOUT,
        )
        prefill_out = model.ttnn_prefill_forward(
            embeds, page_table=None, kv_cache=tt_kv_cache, input_ids_torch=prompt, embeds_torch=None
        )
        prefill_logits = _vocab_logits(prefill_out, vocab)
        prefill_out.deallocate(True)
        last = prefill_logits[-1] if prefill_logits.shape[0] >= PROMPT_TOKENS else prefill_logits[0]
        prefill_pcc = _pcc(last, hf_logits[PROMPT_TOKENS - 1])
        top1_hits, top5_hits = [], []
        hit1, hit5 = _agreement(last, hf_logits[PROMPT_TOKENS - 1])
        top1_hits.append(hit1)
        top5_hits.append(hit5)

        decode_pccs = []
        for step, token in enumerate(forced):
            position = PROMPT_TOKENS + step
            device_inputs = model.prepare_inputs_decode(torch.tensor([token]), torch.tensor([position]), page_table=None)
            logits, _ = model.ttnn_decode_forward(
                x=device_inputs[0],
                current_pos=device_inputs[1],
                rot_mat_idxs=device_inputs[2],
                page_table=device_inputs[3],
                kv_cache=tt_kv_cache,
            )
            step_logits = _vocab_logits(logits, vocab)[0]
            decode_pccs.append(_pcc(step_logits, hf_logits[position]))
            hit1, hit5 = _agreement(step_logits, hf_logits[position])
            top1_hits.append(hit1)
            top5_hits.append(hit5)
        elapsed = time.perf_counter() - t0

        positions = len(top1_hits)
        top1_pct = 100.0 * sum(top1_hits) / positions
        top5_pct = 100.0 * sum(top5_hits) / positions
        mean_pcc = (prefill_pcc + sum(decode_pccs)) / positions
        worst_pcc = min(prefill_pcc, min(decode_pccs))
        worst_step = min(range(len(decode_pccs)), key=decode_pccs.__getitem__)
        print(
            f"ACCURACY positions={positions} top1_pct={top1_pct:.2f} top5_pct={top5_pct:.2f} "
            f"mean_corr={mean_pcc:.6f} worst_corr={worst_pcc:.6f} worst_step={worst_step} "
            f"prefill_corr={prefill_pcc:.6f} device_seconds={elapsed:.1f} tree={_git_head()}",
            flush=True,
        )

        baseline = _load_baseline()
        pin = os.environ.get("PCC_GATE_PIN_BASELINE") == "1" or baseline is None
        failures = []
        if pin:
            path = _baseline_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "top1_pct": top1_pct,
                        "top5_pct": top5_pct,
                        "mean_corr": mean_pcc,
                        "worst_corr": worst_pcc,
                        "positions": positions,
                        "tree": _git_head(),
                        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    indent=1,
                )
            )
            print(
                f"ACCURACY_BASELINE pinned to {path} from tree {_git_head() or '?'}: "
                f"top1={top1_pct:.2f} top5={top5_pct:.2f} mean_corr={mean_pcc:.6f}. Later runs are held "
                f"to top1 >= base-{TOP1_DROP_PTS}, top5 >= base-{TOP5_DROP_PTS}, mean_corr >= base-{MEAN_PCC_DROP}.",
                flush=True,
            )
        else:
            if top1_pct < baseline["top1_pct"] - TOP1_DROP_PTS:
                failures.append(f"top-1 {top1_pct:.2f}% < baseline {baseline['top1_pct']:.2f}% - {TOP1_DROP_PTS}")
            if top5_pct < baseline["top5_pct"] - TOP5_DROP_PTS:
                failures.append(f"top-5 {top5_pct:.2f}% < baseline {baseline['top5_pct']:.2f}% - {TOP5_DROP_PTS}")
            if mean_pcc < baseline["mean_corr"] - MEAN_PCC_DROP:
                failures.append(
                    f"mean logits correlation {mean_pcc:.6f} < baseline {baseline['mean_corr']:.6f} - {MEAN_PCC_DROP}"
                )
            print(
                f"ACCURACY_VS_BASELINE tree={baseline.get('tree') or '?'} "
                f"top1_delta_pts={top1_pct - baseline['top1_pct']:+.2f} "
                f"top5_delta_pts={top5_pct - baseline['top5_pct']:+.2f} "
                f"mean_corr_delta={mean_pcc - baseline['mean_corr']:+.6f}",
                flush=True,
            )

        if failures:
            print("ACCURACY GATE FAILED: " + "; ".join(failures), flush=True)
            # An optimizer reads the worst "PCC: x" in this output as the verdict; a change that loses
            # tokens against the pinned baseline is a failed verdict, whatever its correlation.
            print("PCC: 0.000000 (accuracy gate failed, see ACCURACY GATE FAILED above)", flush=True)
            assert not failures, "; ".join(failures)

        print(
            f"Prefill PCC: {prefill_pcc:.6f} | Decode PCC (worst of {len(decode_pccs)} steps): "
            f"{min(decode_pccs):.6f} | PCC: {worst_pcc:.6f}",
            flush=True,
        )
        assert worst_pcc >= PCC_THRESHOLD
    finally:
        model = None
        model_args = None
        tt_kv_cache = None
        gc.collect()
        if mesh_device is not None:
            for submesh in list(mesh_device.get_submeshes()):
                if submesh is not mesh_device:
                    ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        cache.cleanup()
