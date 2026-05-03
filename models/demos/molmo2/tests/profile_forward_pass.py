# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Wall-clock profiling of the full Molmo2 forward pass for a video test case.

Breaks down every major stage — ViT encode, image pooling (CPU), image projector,
scatter injection (CPU D2H+H2D), prefill mask (CPU), 36 decoder blocks, last-token
slice (CPU D2H+H2D), lm_head, and one traced decode step — using
ttnn.synchronize_device() boundaries for accurate host-side wall time.

Run (two passes: warmup compile + trace capture, then profiled):
    MESH_DEVICE=T3K pytest models/demos/molmo2/tests/profile_forward_pass.py -v -s

Set MOLMO2_VERIF_DIR to point at the dir containing test.jsonl and videos/.
"""

import json
import math
import os
import pathlib
import sys
import time

import pytest
import torch
from loguru import logger

import ttnn

HF_PATH = pathlib.Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
    )
)
VERIF_DIR = pathlib.Path(
    os.environ.get(
        "MOLMO2_VERIF_DIR",
        "/home/ttuser/ssinghal/PR-fix/tt-metal/models/demos/molmo2/verification",
    )
)
VIDEO_DIR = VERIF_DIR / "videos"
TEST_JSONL = VERIF_DIR / "test.jsonl"
WEIGHT_CACHE = pathlib.Path("/tmp/molmo2_weight_cache")


_MESH_SHAPE = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
    "TG": (8, 4),
}.get(os.environ.get("MESH_DEVICE"), (1, 8))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _MESH_SHAPE
    is_single = rows * cols == 1
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    logger.info(f"Opened {rows}×{cols} mesh: {device.get_num_devices()} devices")
    yield device
    ttnn.close_mesh_device(device)
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def tt_model(mesh_device):
    from transformers import AutoModelForImageTextToText

    from models.demos.molmo2.tt.model import TtMolmo2Model
    from models.demos.molmo2.tt.model_config import Molmo2Config
    from models.tt_transformers.tt.ccl import TT_CCL

    sys.path.insert(0, str(HF_PATH))
    logger.info("Loading HF weights (bfloat16)...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        str(HF_PATH), trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    state_dict = hf_model.state_dict()
    del hf_model

    cfg = Molmo2Config(mesh_device=mesh_device)
    ccl = TT_CCL(mesh_device)
    WEIGHT_CACHE.mkdir(parents=True, exist_ok=True)
    model = TtMolmo2Model(
        mesh_device=mesh_device,
        tt_ccl=ccl,
        state_dict=state_dict,
        weight_cache_path=WEIGHT_CACHE,
        dtype=ttnn.bfloat16,
        configuration=cfg,
    )
    del state_dict
    logger.info("TTNN model ready")
    return model, cfg


@pytest.fixture(scope="module")
def processor():
    sys.path.insert(0, str(HF_PATH))
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(HF_PATH), trust_remote_code=True)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def _load_first_video_test() -> tuple:
    """Return (video_path, prompt) for the first test with a local video file."""
    with open(TEST_JSONL) as f:
        for line in f:
            entry = json.loads(line.strip())
            content = entry["messages"][0]["content"]
            prompt, video_url = "", None
            for item in content:
                if item.get("type") == "text":
                    prompt = item["text"]
                elif item.get("type") == "video_url":
                    video_url = item["video_url"]["url"]
            if video_url is None:
                continue
            vpath = VIDEO_DIR / video_url.split("/")[-1]
            if vpath.exists():
                return vpath, prompt
    raise FileNotFoundError(f"No local video files found in {VIDEO_DIR}")


def build_inputs(processor, video_path: pathlib.Path, prompt: str):
    """Run HF processor and return (input_ids, pixel_values, pool_idx, token_type_ids)."""
    sys.path.insert(0, str(HF_PATH))
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}, {"type": "video"}],
        }
    ]
    formatted = processor.apply_chat_template([conversation], tokenize=False, add_generation_prompt=True)
    if isinstance(formatted, list):
        formatted = formatted[0]

    inputs = processor(text=formatted, videos=str(video_path), return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    token_type_ids = inputs.get("token_type_ids")
    pv = inputs.get("pixel_values_videos")
    pool_idx = inputs.get("video_token_pooling")
    if pv is not None:
        pv = pv.float().unsqueeze(0)
        pool_idx = pool_idx.unsqueeze(0)
    return input_ids, pv, pool_idx, token_type_ids


# ---------------------------------------------------------------------------
# Stage-level timing helpers
# ---------------------------------------------------------------------------


def _sync(mesh):
    ttnn.synchronize_device(mesh)


def _ms(t: float) -> float:
    return t * 1000.0


def _from_torch(t, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


# ---------------------------------------------------------------------------
# Core profiling function
# ---------------------------------------------------------------------------


def run_profiled_forward(model, cfg, input_ids, pv, pool_idx, token_type_ids) -> dict:
    """Run one forward pass with per-stage wall-clock timing.

    All device stages are bracketed with ttnn.synchronize_device() so the host
    wall time reflects true device execution time, not just submission latency.
    CPU stages (pooling, scatter, mask, slice) are timed directly.

    Returns a dict of stage names → seconds.
    """
    from models.demos.molmo2.tt.prefill_mask import build_molmo2_prefill_mask
    from models.tt_transformers.tt.common import Mode

    mesh = model.mesh_device
    B, S = input_ids.shape
    timings = {}

    model.reset_kv_cache(user_id=0)

    # ------------------------------------------------------------------ #
    # 1. Embedding
    # ------------------------------------------------------------------ #
    input_ids_tt = _from_torch(input_ids.unsqueeze(0), mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    _sync(mesh)
    t0 = time.perf_counter()
    x_tt = ttnn.embedding(input_ids_tt, model.embedding)
    _sync(mesh)
    timings["1_embedding"] = time.perf_counter() - t0
    ttnn.deallocate(input_ids_tt)

    # ------------------------------------------------------------------ #
    # 2–5. Vision backbone (ViT + pooling + projector + scatter injection)
    # ------------------------------------------------------------------ #
    if pv is not None:
        B_pv, n_crops, n_patches, px_dim = pv.shape
        n_crops_flat = B_pv * n_crops
        pv_4d = pv.reshape(n_crops_flat, 1, n_patches, px_dim).to(torch.bfloat16)
        _MAX_VIT_BATCH = 8

        # --- 2. ViT encode (TTNN, chunked) ---
        _sync(mesh)
        t0 = time.perf_counter()
        vit_chunks = []
        for start in range(0, n_crops_flat, _MAX_VIT_BATCH):
            chunk_tt = _from_torch(pv_4d[start : start + _MAX_VIT_BATCH], mesh)
            feat_tt = model.vit_encoder.forward(chunk_tt)
            ttnn.deallocate(chunk_tt)
            _sync(mesh)
            feat_cpu = ttnn.to_torch(ttnn.get_device_tensors(feat_tt)[0]).float()
            ttnn.deallocate(feat_tt)
            vit_chunks.append(feat_cpu.squeeze(0))
        timings["2_vit_encode"] = time.perf_counter() - t0

        vit_cpu = torch.cat(vit_chunks, dim=0).reshape(B_pv, n_crops, n_patches, 2304)

        # --- 3. Image pooling (TTNN chunked) ---
        _sync(mesh)
        t0 = time.perf_counter()
        pooled_cpu = model._run_chunked_ttnn_pooling(vit_cpu, pool_idx)
        _sync(mesh)
        timings["3_image_pooling_cpu"] = time.perf_counter() - t0

        # --- 4. Image projector (TTNN) ---
        valid_token = (pool_idx[0] >= 0).any(dim=-1)
        pooled_valid = pooled_cpu[0][valid_token]
        pooled_tt = _from_torch(pooled_valid.to(torch.bfloat16).unsqueeze(0).unsqueeze(0), mesh)
        _sync(mesh)
        t0 = time.perf_counter()
        proj_out = model.image_projector.forward(pooled_tt)
        _sync(mesh)
        timings["4_image_projector"] = time.perf_counter() - t0
        ttnn.deallocate(pooled_tt)
        proj_cpu = ttnn.to_torch(ttnn.get_device_tensors(proj_out)[0]).float().squeeze(0).squeeze(0)
        ttnn.deallocate(proj_out)

        # --- 5. Scatter injection: dense zero delta + ttnn.add (no D2H) ---
        t0 = time.perf_counter()
        H = cfg.dim
        is_patch = input_ids.view(-1) == cfg.image_patch_id
        delta = torch.zeros(1, 1, S, H, dtype=torch.bfloat16)
        delta.view(-1, H)[is_patch] = proj_cpu.to(torch.bfloat16)
        delta_tt = _from_torch(delta, mesh)
        x_tt = ttnn.to_layout(x_tt, ttnn.TILE_LAYOUT)
        x_tt = ttnn.reshape(x_tt, [1, 1, S, H])
        x_tt = ttnn.add(x_tt, delta_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(delta_tt)
        _sync(mesh)
        timings["5_scatter_injection"] = time.perf_counter() - t0

        n_crops_out = n_crops
        N_pooled_out = pool_idx.shape[1]
        N_valid = int(valid_token.sum().item())
    else:
        x_tt = ttnn.to_layout(x_tt, ttnn.TILE_LAYOUT)
        for k in ("2_vit_encode", "3_image_pooling_cpu", "4_image_projector", "5_scatter_injection"):
            timings[k] = 0.0
        n_crops_out = 0
        N_pooled_out = 0
        N_valid = 0
        x_tt = ttnn.reshape(x_tt, [1, 1, S, cfg.dim])

    # ------------------------------------------------------------------ #
    # 6. Prefill padding + attention mask build (CPU + H2D upload)
    # ------------------------------------------------------------------ #
    if S <= 8192:
        S_pad = max(256, 1 << math.ceil(math.log2(S)) if S > 1 else 256)
    else:
        S_pad = ((S + 255) // 256) * 256
    pad_len = S_pad - S

    if pad_len > 0:
        x_pad = _from_torch(torch.zeros(1, 1, pad_len, cfg.dim, dtype=torch.bfloat16), mesh)
        x_tt = ttnn.concat([x_tt, x_pad], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_pad)

    attn_mask = None
    if token_type_ids is not None:
        tti_pad = (
            torch.cat([token_type_ids.long(), torch.zeros(B, pad_len, dtype=torch.long)], dim=1)
            if pad_len > 0
            else token_type_ids.long()
        )
        t0 = time.perf_counter()
        attn_mask = build_molmo2_prefill_mask(S_pad, tti_pad, mesh, dtype=ttnn.bfloat8_b)
        _sync(mesh)
        timings["6_prefill_mask"] = time.perf_counter() - t0
    else:
        timings["6_prefill_mask"] = 0.0

    # ------------------------------------------------------------------ #
    # 7. RoPE setup (CPU slice + H2D — cached after first call)
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    rot_mats = model._get_rot_mats_prefill(S_pad)
    _sync(mesh)
    timings["7_rope_setup"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 8. 36 decoder blocks (TTNN)
    # ------------------------------------------------------------------ #
    _sync(mesh)
    t0 = time.perf_counter()
    for layer in model.layers:
        x_tt = layer.forward(x_tt, rot_mats=rot_mats, user_id=0, mode="prefill", attn_mask=attn_mask)
    _sync(mesh)
    timings["8_decoder_blocks"] = time.perf_counter() - t0

    if attn_mask is not None:
        ttnn.deallocate(attn_mask)

    # ------------------------------------------------------------------ #
    # 9. ln_f (TTNN)
    # ------------------------------------------------------------------ #
    _sync(mesh)
    t0 = time.perf_counter()
    x_tt = model.ln_f(x_tt, mode=Mode.PREFILL)
    _sync(mesh)
    timings["9_ln_f"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 10. Last-token slice: ttnn.slice on device — no D2H/H2D
    # ------------------------------------------------------------------ #
    _sync(mesh)
    t0 = time.perf_counter()
    x_last_tt = ttnn.slice(x_tt, (0, 0, S - 1, 0), (1, 1, S, cfg.dim), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x_tt)
    _sync(mesh)
    timings["10_last_token_slice"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 11. lm_head (TTNN)
    # ------------------------------------------------------------------ #
    _sync(mesh)
    t0 = time.perf_counter()
    logits = ttnn.linear(
        x_last_tt,
        model.lm_head,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=cfg.compute_kernel_config_hifi2_fp16,
    )
    _sync(mesh)
    timings["11_lm_head"] = time.perf_counter() - t0
    ttnn.deallocate(x_last_tt)
    logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float().squeeze(0)
    ttnn.deallocate(logits)

    # ------------------------------------------------------------------ #
    # 12. Single traced decode step
    # Capture trace on first call (warmup pass); reuse on profiled pass.
    # The trace encodes: embedding → 36 layers → ln_f → lm_head entirely
    # on device. execute_trace is blocking=True so no extra sync needed.
    # ------------------------------------------------------------------ #
    if model._decode_trace_id is None:
        model._decode_trace_tensors = model._allocate_decode_trace_tensors()
        model._decode_trace_id, model._decode_trace_output = model._capture_decode_trace(model._decode_trace_tensors, S)

    next_token = int(logits_cpu[0, 0].argmax().item())
    _sync(mesh)
    t0 = time.perf_counter()
    model._execute_decode_trace(next_token, S)
    _sync(mesh)
    timings["12_decode_step"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # Summary metadata
    # ------------------------------------------------------------------ #
    timings["_seq_len"] = S
    timings["_S_pad"] = S_pad
    timings["_n_crops"] = n_crops_out
    timings["_N_pooled"] = N_pooled_out
    timings["_N_valid"] = N_valid
    timings["_decode_traced"] = model._decode_trace_id is not None

    return timings


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

_STAGE_META = [
    # (key, label, type)
    ("1_embedding", "Embedding", "TTNN"),
    ("2_vit_encode", "ViT encode (25 blks, all crops)", "TTNN"),
    ("3_image_pooling_cpu", "Image pooling (cross-attn)", "TTNN"),
    ("4_image_projector", "Image projector (SwiGLU)", "TTNN"),
    ("5_scatter_injection", "Scatter inject (delta+ttnn.add)", "TTNN"),
    ("6_prefill_mask", "Prefill mask (img_mm+max+where)", "TTNN"),
    ("7_rope_setup", "RoPE setup (cached=near-zero)", "CPU"),
    ("8_decoder_blocks", "36 decoder blocks", "TTNN"),
    ("9_ln_f", "ln_f", "TTNN"),
    ("10_last_token_slice", "Last-token slice (ttnn.slice)", "TTNN"),
    ("11_lm_head", "lm_head", "TTNN"),
]


def print_timing_table(timings: dict, label: str = ""):
    S = timings["_seq_len"]
    S_pad = timings["_S_pad"]
    n_crops = timings["_n_crops"]
    N_pooled = timings["_N_pooled"]
    N_valid = timings["_N_valid"]

    prefill_keys = [k for k, _, _ in _STAGE_META]
    total_prefill = sum(timings.get(k, 0.0) for k in prefill_keys)
    total_cpu = sum(timings.get(k, 0.0) for k, _, t in _STAGE_META if t == "CPU")
    total_ttnn = total_prefill - total_cpu

    W = 72
    print(f"\n{'='*W}")
    print(f"  Molmo2 Forward Pass Profile  {label}")
    print(f"  S={S} (padded→{S_pad}), n_crops={n_crops}, N_pooled={N_pooled}, N_valid={N_valid}")
    print(f"{'='*W}")
    print(f"  {'Stage':<38} {'ms':>8}  {'%':>6}  {'Type':<5}")
    print(f"  {'-'*62}")

    for key, label_s, kind in _STAGE_META:
        t = timings.get(key, 0.0)
        pct = 100.0 * t / total_prefill if total_prefill > 0 else 0.0
        marker = " ◄" if kind == "CPU" else ""
        print(f"  {label_s:<38} {_ms(t):>8.1f}  {pct:>5.1f}%  {kind:<5}{marker}")

    print(f"  {'-'*62}")
    print(f"  {'TOTAL prefill':<38} {_ms(total_prefill):>8.1f}  {'100.0%':>6}")
    print(f"  {'  TTNN subtotal':<38} {_ms(total_ttnn):>8.1f}  {100*total_ttnn/total_prefill:>5.1f}%")
    print(f"  {'  CPU subtotal (incl. D2H/H2D)':<38} {_ms(total_cpu):>8.1f}  {100*total_cpu/total_prefill:>5.1f}%  ◄")
    t_decode = timings["12_decode_step"]
    traced = timings.get("_decode_traced", False)
    trace_label = "traced" if traced else "eager"
    print(f"  Decode step (1 step, {trace_label}):   {_ms(t_decode):>8.1f} ms  ({1/t_decode:.1f} tok/s)")
    print(f"{'='*W}\n")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_profile_forward_pass(mesh_device, tt_model, processor):
    """Profile the Molmo2 forward pass wall-clock time by stage for a video input."""
    model, cfg = tt_model

    video_path, prompt = _load_first_video_test()
    logger.info(f"Video: {video_path.name}")
    logger.info(f"Prompt: {prompt[:80]!r}")

    logger.info("Preprocessing inputs...")
    input_ids, pv, pool_idx, token_type_ids = build_inputs(processor, video_path, prompt)
    S = input_ids.shape[1]
    n_crops = pv.shape[1] if pv is not None else 0
    logger.info(f"  S={S}, n_crops={n_crops}")

    # ---- Warmup pass (triggers JIT kernel compilation for this S) ----
    logger.info("Warmup pass (JIT compile, not timed)...")
    t0 = time.perf_counter()
    _ = run_profiled_forward(model, cfg, input_ids, pv, pool_idx, token_type_ids)
    logger.info(f"  Warmup done in {time.perf_counter()-t0:.1f}s")

    # ---- Profiled pass 1 (kernels cached, but caches may be cold) ----
    logger.info("Profiled pass 1...")
    t0 = time.perf_counter()
    timings1 = run_profiled_forward(model, cfg, input_ids, pv, pool_idx, token_type_ids)
    logger.info(f"  Pass 1 done in {time.perf_counter()-t0:.1f}s")

    # ---- Profiled pass 2 (steady state — caches warm, no compile overhead) ----
    logger.info("Profiled pass 2 (steady state)...")
    t0 = time.perf_counter()
    timings2 = run_profiled_forward(model, cfg, input_ids, pv, pool_idx, token_type_ids)
    logger.info(f"  Pass 2 done in {time.perf_counter()-t0:.1f}s")

    label = f"({video_path.name[:16]}…)"
    print_timing_table(timings1, label=f"{label} pass 1")
    print_timing_table(timings2, label=f"{label} pass 2 — steady state")

    assert timings2["8_decoder_blocks"] > 0
    assert timings2["12_decode_step"] > 0
