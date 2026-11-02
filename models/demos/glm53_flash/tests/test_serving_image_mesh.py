# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The served image + video path on a (2, 4) block: ``GLM53FlashForConditionalGeneration`` built through
``initialize_vllm_model`` (resident vision tower), warmed up in the plugin's order, then driven with the
plugin-shaped ``prefill_forward`` call (per-row ``pixel_values`` / ``image_grid_thw`` / ``pixel_values_videos`` /
``video_grid_thw`` item lists + placeholder ranges) and checked against the model harness path of
``demo/multimodal_demo.py`` (``image_features`` / ``video_features`` + ``prefill_multimodal`` on the SAME model
instance, another user slot) with the same media and prompt:

1. the demo image (``shapes_hello.png``, 286 tokens, image rows 12..267): one eager device chunk padded to the 512
   bucket -- first token == the harness's, last-token logits PCC > 0.999 (bit-exactness is reported: the harness pads
   to 320 rows, the served chunk to 512), then 8 traced decode steps through ``decode_forward`` == the harness's
   replays of the same trace;
2. the same image behind 288 filler tokens (image rows 299..554, 574 tokens): the served prompt walks TWO
   512-token device chunks, the image straddling the boundary (213 rows in the first chunk, 43 in the second),
   against the harness's single-shot prefill (the 1024 bucket) -- first token equal, logits PCC > 0.999;
3. a synthetic 4-frame clip of the image (``synthetic_clip``: 2 temporal groups, 512 vision tokens in two runs
   with timestamps, 549 tokens) as a video item -- first token equal, logits PCC > 0.999; the tower's features for
   the clip against the HF fp32 reference when ``bringup/references/glm53_flash/MMV_shapes_clip4`` exists.

Layers 0-3 only (``GLM53_MODEL_LAYERS``): the tokens are meaningless, the check is served-path == harness-path.
The 512-token chunk keeps the row slabs at 256 rows and the harness's single shot at 512 (the KDA L1 hazard at 1024-row
slabs on 2 rows, ``test_decode_packed_inputs_mesh.py``);
``max_seq_len`` 1536 keeps the traced ladder at 256 / 512 / 1024 on 2 rows (the 2048 trace = 1024-row slabs).

    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-3 GLM53_MODEL_B=4 GLM53_VLLM_PREFILL_CHUNK=512 GLM53_MODEL_TRACE_MB=512 \\
        TT_VISIBLE_DEVICES=<rows-4-5 block> MESH_DEVICE=P150x8 TT_MESH_GRAPH_DESC_PATH=<p150_x8 descriptor> \\
        pytest models/demos/glm53_flash/tests/test_serving_image_mesh.py -s --timeout 3600

Expected: three "SERVED_IMAGE_RESULT ... first token == harness" lines, decode tokens identical, PCC > 0.999;
``GLM53_VLLM_VISION_RESIDENT=0`` runs the same with the per-request tower (build + release per prefill).
First failure candidates (in order): (a) a ``prefill_streams`` shape / pad mismatch between the 512-bucket chunk
and the layers (``eager_pad_length`` vs ``PrefillTraceInputs``: an assertion in ``FunctionalDecoder.prefill_forward``);
(b) the straddling case's second chunk (``start=512``, 62 tokens padded to 256) disagreeing with the single-shot
prefill in the KDA chunked scan (``next_start`` contract): first token differs, PCC < 0.999 -> compare against
``prefill_chunked`` of the text path with the same boundary before blaming the scatter; (c) the video item's
placeholder range (``mm_placeholder_offsets`` at ``<|begin_of_video|>``) counting 512 rows -- a ValueError from
``feature_rows`` means the processor's frame structure changed; (d) the tower batch of two 1024-patch windows
(``cu_window`` with two windows + a pad window) differing from one-window runs: tower PCC vs HF below the bar.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt import generator_vllm as gv
from models.demos.glm53_flash.tt.generator_vllm import GLM53FlashForConditionalGeneration
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mm_inputs import IMAGE_TOKEN_ID, VIDEO_END_ID, VIDEO_START_ID, feature_rows, mm_items
from models.demos.glm53_flash.tt.multimodal import (
    encode_multimodal,
    encode_multimodal_video,
    image_features,
    prefill_multimodal,
    synthetic_clip,
    video_features,
)

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "4"))  # users per row = max_num_seqs
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
MAX_SEQ_LEN = 1536
CHUNK = 512  # two device chunks inside MAX_SEQ_LEN on any row count up to 16 rows; 256 local rows per rank on 2 rows
# and the harness's single-shot bucket (1024) at 512: ttKDA's SP scan clashes its L1 group summaries with the exclusive
# scan's CBs at 1024 local rows x 2 ranks (the 2026-10-29 block runs; the 8x4's 2048 local rows are unaffected)
N_DECODE = int(os.environ.get("GLM53_MM_DECODE_STEPS", "8"))
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "demo" / "sample_prompts"
IMAGE = Path(os.environ.get("GLM53_MM_IMAGE", SAMPLE_DIR / "shapes_hello.png"))
QUESTION = "What shapes and what word are in this image? Answer in one sentence."
QUESTION_V = "What happens in this clip? Answer in one sentence."
REF_CLIP = Path("bringup/references/glm53_flash/MMV_shapes_clip4")
LOGITS_PCC_BAR = 0.999
FILLER = 288


def _plugin_kwargs(items, prefix_ids: torch.Tensor) -> dict:
    """The plugin's per-row lists for ONE request of ``items`` (``MMItem`` in prompt order) with the placeholder
    ranges vLLM would attach: the image's replacement run, the video's whole ``<|begin_of_video|>`` ..
    ``<|end_of_video|>`` span."""
    flat = prefix_ids.reshape(-1)  # a [1, S] prompt: nonzero on 2-D ids would return (row, col) pairs
    rows = feature_rows(flat, items)
    offs, lens = [], []
    for it, r in zip(items, rows):
        if it.kind == "image":
            offs.append(int(r[0]))
            lens.append(int(r.numel()))
        else:
            s = int(torch.nonzero(flat == VIDEO_START_ID).reshape(-1)[0])
            e = int(torch.nonzero(flat == VIDEO_END_ID).reshape(-1)[0])
            offs.append(s)
            lens.append(e - s + 1)
    return {
        "pixel_values": [[it.pixel_values if it.kind == "image" else None for it in items]],
        "image_grid_thw": [[it.grid_thw[0] if it.kind == "image" else None for it in items]],
        "pixel_values_videos": [[it.pixel_values if it.kind == "video" else None for it in items]],
        "video_grid_thw": [[it.grid_thw[0] if it.kind == "video" else None for it in items]],
        "mm_placeholder_offsets": [offs],
        "mm_placeholder_lengths": [lens],
    }


def _greedy(n):
    return SimpleNamespace(temperature=[0.0] * n, top_k=[1] * n)


def _warm_harness(adapter, n_tokens: int, user: int) -> None:
    """Build the reference path's programs for an ``n_tokens`` prompt BEFORE the trace captures: ``prefill_multimodal``
    runs the whole prompt single-shot (its padded bucket, e.g. 1024 rows for the straddling prompt), a row count the
    served warm-up never runs (the adapter chunks at the device chunk), so without this the first reference call
    compiles under the sealed program cache (EmbeddingsDeviceOperation, 2026-10-28 8x4). Zero features at the image
    rows; slot ``user``'s state is overwritten by its next prompt."""
    model = adapter.model
    n_img = 256
    ids = torch.full((n_tokens,), 1000, dtype=torch.long)
    ids[11 : 11 + n_img] = IMAGE_TOKEN_ID
    streams, logits_last, _ = prefill_multimodal(
        model, ids.reshape(1, -1), [torch.zeros(n_img, model.embedding.hidden)], user=user
    )
    ttnn.deallocate(streams)
    ttnn.deallocate(logits_last)
    adapter._sync()


def _harness(adapter, ids: torch.Tensor, feats: list[torch.Tensor], user: int) -> tuple[int, torch.Tensor, list[int]]:
    """The demo's path on the adapter's model: ``prefill_multimodal`` into slot ``user`` of every row, the first
    token, the last-token logits, then ``N_DECODE`` replays of the adapter's decode trace reading slot ``user``."""
    model = adapter.model
    streams, logits_last, owner_row = prefill_multimodal(model, ids, feats, user=user)
    first = int(model.first_token(logits_last, owner_row))
    logits = adapter._eager_last_logits(logits_last, owner_row).clone()
    ttnn.deallocate(streams)
    ttnn.deallocate(logits_last)
    S = int(ids.numel())
    U = model.total_users
    gen = [first]
    positions = [0] * U
    for _ in range(N_DECODE):
        toks = torch.full((U,), adapter.pad_token_id, dtype=torch.long)
        for r in range(model.mesh.rows):
            toks[r * B + user] = gen[-1]
            positions[r * B + user] = S + len(gen) - 1
        model.load_decode_inputs(toks, positions)
        model.replay(adapter.trace_id)
        ids_tr = adapter._trace_outputs[-1]  # (logits, ids) of the captured step
        gen.append(int(model.ids_to_host(ids_tr)[user]))
    return first, logits, gen[1:]


def _served(adapter, ids: torch.Tensor, items, logical: int) -> tuple[int, torch.Tensor, list[int]]:
    """The plugin-shaped call: ``prefill_forward`` (host logits) for the whole prompt in one engine step, then
    ``N_DECODE`` ``decode_forward`` steps (device greedy)."""
    S = int(ids.numel())
    kw = _plugin_kwargs(items, ids)
    t0 = time.perf_counter()
    logits = adapter.prefill_forward(
        ids.reshape(1, -1).to(torch.int32),
        prompt_lens=[S],
        start_pos=torch.tensor([0]),
        empty_slots=[logical],
        enable_trace=True,
        **kw,
    )
    adapter._sync()
    ttft = time.perf_counter() - t0
    first = int(logits[0, 0].argmax())
    gen = [first]
    n = adapter.slots.n
    for _ in range(N_DECODE):
        toks = torch.zeros(n, 1, dtype=torch.int32)
        pos = torch.full((n,), -1, dtype=torch.long)
        toks[logical, 0] = gen[-1]
        pos[logical] = S + len(gen) - 1
        out = adapter.decode_forward(
            toks,
            pos,
            sampling_params=_greedy(n),
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        gen.append(int(out[logical, 0]))
    logger.info(f"served prefill of {S} tokens ({[it.kind for it in items]}): TTFT {ttft * 1000:.0f} ms (host logits)")
    return first, logits[0, 0].clone(), gen[1:]


def _compare(tag: str, served, harness, tok) -> None:
    s_first, s_logits, s_gen = served
    h_first, h_logits, h_gen = harness
    p = pcc(s_logits, h_logits)
    exact = torch.equal(s_logits, h_logits)
    same = sum(int(a == b) for a, b in zip(s_gen, h_gen))
    logger.info(
        f"SERVED_IMAGE_RESULT {tag}: first token {s_first} {tok.decode([s_first])!r} {'==' if s_first == h_first else '!='} "
        f"harness {h_first} {tok.decode([h_first])!r}; logits PCC {p:.6f} ({'bit-exact' if exact else 'not bit-exact'}); "
        f"decode {same}/{len(h_gen)} tokens identical: served {s_gen} harness {h_gen}"
    )
    assert s_first == h_first, f"{tag}: first token differs ({s_first} vs {h_first})"
    assert p > LOGITS_PCC_BAR, f"{tag}: logits PCC {p}"
    assert s_gen == h_gen, f"{tag}: decode tokens differ"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_served_image_and_video_match_the_harness(mesh_device, monkeypatch):
    from transformers import AutoProcessor

    monkeypatch.setenv("GLM53_VLLM_PREFILL_CHUNK", str(CHUNK))
    monkeypatch.setenv(gv.IMAGES_ENV, "auto")
    snapshot = default_snapshot_dir()
    proc = AutoProcessor.from_pretrained(str(snapshot))
    tok = proc.tokenizer
    t0 = time.perf_counter()
    a = GLM53FlashForConditionalGeneration.initialize_vllm_model(
        SimpleNamespace(_name_or_path=str(snapshot)), mesh_device, max_batch_size=B, max_seq_len=MAX_SEQ_LEN
    )
    assert a.vision is not None and a.geometry.multimodal, "the multimodal class must build the tower"
    logger.info(
        f"adapter built in {time.perf_counter() - t0:.0f} s: tower {'resident' if a.vision.resident else 'per request'}, "
        f"traced lengths {a.geometry.traced_prefill_lengths}, device chunk {a.geometry.device_prefill_chunk}"
    )
    t0 = time.perf_counter()
    a.warmup_model_prefill(enable_trace=False, can_sample_on_device=True)
    a.warmup_model_decode(enable_trace=False, max_batch_size=B, num_blocks=1, can_sample_on_device=True)
    _warm_harness(a, CHUNK + 62, B - 1)  # the reference path's single-shot shape of the straddling prompt
    a.warmup_model_prefill(enable_trace=True, can_sample_on_device=True)
    a.warmup_model_decode(enable_trace=True, max_batch_size=B, num_blocks=1, can_sample_on_device=True)
    logger.info(f"warm-up (text ladder + multimodal + traces) in {time.perf_counter() - t0:.0f} s")
    tower = a.vision.tower if a.vision.resident else None
    harness_user = B - 1

    # 1. the demo image, one device chunk
    ids, pv, grid = encode_multimodal(proc, IMAGE, QUESTION)
    S = int(ids.numel())
    rows = torch.nonzero(ids.reshape(-1) == IMAGE_TOKEN_ID).reshape(-1)
    logger.info(
        f"image prompt: {S} tokens, {int(rows.numel())} image tokens at {int(rows[0])}..{int(rows[-1])}, grid {grid.tolist()}"
    )
    feats = image_features(mesh_device, pv, grid, tower=tower)
    harness = _harness(a, ids, feats, harness_user)
    items = mm_items(_plugin_kwargs_items(pv, grid, None, None), 0)
    served = _served(a, ids, items, 0)
    _compare("image/1 chunk", served, harness, tok)
    a.release_request(0)

    # 2. the image behind filler text: two device chunks, the image straddling the chunk boundary
    filler = torch.tensor(
        tok.encode(" The quick brown fox jumps over the lazy dog." * 100, add_special_tokens=False)[:FILLER]
    )
    ids2 = torch.cat([ids[0, :11], filler, ids[0, 11:]]).reshape(1, -1)
    rows2 = torch.nonzero(ids2.reshape(-1) == IMAGE_TOKEN_ID).reshape(-1)
    assert int(rows2[0]) < CHUNK < int(rows2[-1]), (int(rows2[0]), int(rows2[-1]))
    logger.info(
        f"straddling prompt: {ids2.numel()} tokens, image rows {int(rows2[0])}..{int(rows2[-1])} across the {CHUNK} boundary"
    )
    harness2 = _harness(a, ids2, feats, harness_user)
    served2 = _served(a, ids2, items, 0)
    _compare("image/straddling 2 chunks", served2, harness2, tok)
    a.release_request(0)

    # 3. a synthetic clip as a video item (2 temporal groups)
    clip = synthetic_clip(IMAGE, 4)
    ids3, pvv, vg = encode_multimodal_video(proc, clip, QUESTION_V)
    S3 = int(ids3.numel())
    rows3 = torch.nonzero(ids3.reshape(-1) == IMAGE_TOKEN_ID).reshape(-1)
    logger.info(f"video prompt: {S3} tokens, {int(rows3.numel())} vision tokens, grid {vg.tolist()}")
    t0 = time.perf_counter()
    vfeats = video_features(mesh_device, pvv, vg, tower=tower)
    logger.info(
        f"tower on the clip ({int(pvv.shape[0])} patches, {int(vg[0, 0])} groups): {(time.perf_counter() - t0) * 1000:.0f} ms"
    )
    if (REF_CLIP / "features.pt").exists() and (REF_CLIP / "prompt.json").exists():
        ref = torch.load(REF_CLIP / "features.pt")
        ref_ids = json.loads((REF_CLIP / "prompt.json").read_text())["input_ids"]
        if ref_ids == ids3[0].tolist():
            p32 = pcc(vfeats[0], ref["fp32"][0])
            floor = pcc(ref["bf16_model"][0], ref["fp32"][0])
            logger.info(
                f"clip tower PCC vs HF fp32 {p32:.6f} (HF bf16 floor {floor:.6f}, bar {min(0.995, floor - 0.03):.4f})"
            )
            assert p32 > min(0.995, floor - 0.03)
        else:
            logger.warning(f"{REF_CLIP}: another prompt (ids differ); tower PCC skipped")
    harness3 = _harness(a, ids3, vfeats, harness_user)
    items3 = mm_items(_plugin_kwargs_items(None, None, pvv, vg), 0)
    served3 = _served(a, ids3, items3, 0)
    _compare("video/2 groups", served3, harness3, tok)
    a.release_request(0)
    a.release_persistent_capture()


def _plugin_kwargs_items(pv, grid, pvv, vg) -> dict:
    """The plugin's lists for one request holding one image or one video (no placeholder ranges yet: the test
    attaches them through ``feature_rows`` in ``_plugin_kwargs``)."""
    if pv is not None:
        return {
            "pixel_values": [[pv]],
            "image_grid_thw": [[grid[0]]],
            "pixel_values_videos": [[None]],
            "video_grid_thw": [[None]],
        }
    return {
        "pixel_values": [[None]],
        "image_grid_thw": [[None]],
        "pixel_values_videos": [[pvv]],
        "video_grid_thw": [[vg[0]]],
    }
