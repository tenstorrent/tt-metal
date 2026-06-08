# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 TTNN end-to-end performance — full breakdown of trace vs host costs.

The companion test `test_perf_ttnn_full_e2e_trace.py` measures only the
device-side trace replay (~83 ms). This test additionally measures every
host-side cost a real pipeline would pay per inference:

  1. Pure trace replay (device-only)         — same as the existing test
  2. Host noise resample (torch.randn)
  3. Host→Device transfer for the noise tensor
  4. Device→Host transfer of the action output
  5. CPU post-processing (slice + bf16→fp32)
  6. Full "realistic" call (noise + trace + readback chained)
  7. Untraced sample_actions (full Python path, no trace cache)

The (6) "realistic full call" is the closest estimate of what a robot
control loop would see between camera frame and emitted actions. The (7)
untraced number shows what we save by capturing the trace.

Skipped if the checkpoint isn't present locally.
"""

import os
import statistics
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
import ttnn

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_libero_upstream"
CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))

NUM_WARMUP = 2
NUM_STEADY_ITERS = 20
NUM_UNTRACED_ITERS = 5  # untraced is ~5-10× slower than trace, keep small
LANG_SEQ_LEN = 256  # tile-aligned
SEED = 0
TRACE_REGION_SIZE = 134_217_728  # 128 MiB — full sample_actions trace ~81 MB
# Production pi0.5 LIBERO passes 3 images to SigLIP — see [[pi05-siglip-bs3-production]].
# (NUM_IMAGE_VIEWS=3 already used in this file's host-preprocessing breakdown below.)
NUM_CAMERAS = int(os.environ.get("PI0_NUM_CAMERAS", "2"))

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_inputs(device, num_cameras: int = NUM_CAMERAS):
    torch.manual_seed(SEED)
    images = [torch.randn(1, 3, 224, 224, dtype=torch.float32) for _ in range(num_cameras)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(num_cameras)]
    lang_tokens = torch.randint(0, 256000, (1, LANG_SEQ_LEN), dtype=torch.int32)
    lang_masks = torch.ones(1, LANG_SEQ_LEN, dtype=torch.bool)

    images_ttnn = [
        ttnn.from_torch(
            im, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for im in images
    ]
    img_masks_ttnn = [
        ttnn.from_torch(
            m.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for m in img_masks
    ]
    lang_tokens_ttnn = ttnn.from_torch(
        lang_tokens.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    lang_masks_ttnn = ttnn.from_torch(
        lang_masks.to(torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return images_ttnn, img_masks_ttnn, lang_tokens_ttnn, lang_masks_ttnn


def _call_sample_actions(model, images_ttnn, img_masks_ttnn, lang_tokens_ttnn, lang_masks_ttnn):
    return model.sample_actions(
        images=images_ttnn,
        img_masks=img_masks_ttnn,
        lang_tokens=lang_tokens_ttnn,
        lang_masks=lang_masks_ttnn,
        state=None,
    )


# -----------------------------------------------------------------------------
# Libero-style preprocessing helpers (mirror models/.../eval/libero_rollout.py)
# -----------------------------------------------------------------------------

NUM_IMAGE_VIEWS = 3  # pi0.5 LIBERO: agentview + wrist + empty (-1 fill)
IMG_SIZE = 224
TOKENIZER_PATH = "/storage/sdawle/pi05_weights/paligemma_tokenizer.model"


def _resize_with_pad_centered(img_hwc_uint8: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Aspect-preserving bilinear resize, centered pad with -1.0. Mirrors
    libero_rollout.Pi0_5LiberoAdapter._resize_with_pad_centered."""
    from PIL import Image

    h, w = img_hwc_uint8.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = (
        np.asarray(
            Image.fromarray(img_hwc_uint8).resize((nw, nh), Image.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )
    resized = resized * 2.0 - 1.0
    out = -np.ones((size, size, 3), dtype=np.float32)
    oy = (size - nh) // 2
    ox = (size - nw) // 2
    out[oy : oy + nh, ox : ox + nw] = resized
    return out


def _image_for_pi05(img_hwc_uint8: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 3, 224, 224) float32 in [-1, 1]."""
    img_pad = _resize_with_pad_centered(img_hwc_uint8)
    chw = np.transpose(img_pad, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).contiguous()


def _make_tokens_random(rng: np.random.Generator):
    """Generate a fake tokenized prompt of length LANG_SEQ_LEN. We don't
    invoke SentencePiece here because the test should run without a tokenizer
    file present — but the cost shape (Python str work + small fill) is small
    enough that this is a faithful overestimate."""
    tokens = rng.integers(0, 256000, size=LANG_SEQ_LEN, dtype=np.int32)
    mask = np.ones(LANG_SEQ_LEN, dtype=bool)
    return (
        torch.from_numpy(tokens).unsqueeze(0),
        torch.from_numpy(mask).unsqueeze(0),
    )


def _make_tokens_sentencepiece(sp, task_desc: str, state_bins: np.ndarray):
    """Real tokenizer path used by the libero adapter."""
    state_str = " ".join(str(int(b)) for b in state_bins)
    full = f"Task: {task_desc}, State: {state_str};\nAction: "
    tokens = sp.encode(full, add_bos=True)
    L = len(tokens)
    if L < LANG_SEQ_LEN:
        mask = [True] * L + [False] * (LANG_SEQ_LEN - L)
        tokens = tokens + [0] * (LANG_SEQ_LEN - L)
    else:
        tokens = tokens[:LANG_SEQ_LEN]
        mask = [True] * LANG_SEQ_LEN
    return (
        torch.tensor(tokens, dtype=torch.int32).unsqueeze(0),
        torch.tensor(mask, dtype=torch.bool).unsqueeze(0),
    )


def _summarize(name: str, samples_ms: List[float]) -> dict:
    return {
        "name": name,
        "avg": statistics.mean(samples_ms),
        "min": min(samples_ms),
        "max": max(samples_ms),
        "std": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "n": len(samples_ms),
    }


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_ttnn_e2e_host_breakdown(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint

    print(f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}")
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    _ah = action_horizon_from_checkpoint(CHECKPOINT_DIR)
    _num_steps = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "10"))
    cfg = Pi0_5ModelConfig(action_horizon=_ah, num_denoising_steps=_num_steps)
    print(f"   action_horizon={_ah}  num_denoising_steps={_num_steps}")
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print(f"✅ Model loaded")

    images_ttnn, img_masks_ttnn, lang_tokens_ttnn, lang_masks_ttnn = _build_inputs(device)
    print(f"   num_cameras={len(images_ttnn)} (SigLIP runs bs={len(images_ttnn)} via concat)")

    ah = cfg.action_horizon
    ah_padded = ((ah + 31) // 32) * 32
    action_dim = cfg.action_dim
    print(f"   action chunk shape: ({ah}, {action_dim}), padded to ({ah_padded}, {action_dim})")

    # --------- warmup / JIT compile ---------
    print(f"\n🔥 Warmup ({NUM_WARMUP} calls) — JIT compile")
    for i in range(NUM_WARMUP):
        with torch.no_grad():
            out = _call_sample_actions(model, images_ttnn, img_masks_ttnn, lang_tokens_ttnn, lang_masks_ttnn)
        ttnn.synchronize_device(device)
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
        print(f"   warmup call {i + 1} done")

    # --------- capture trace (no resample inside trace) ---------
    model.resample_noise = False

    # Pre-stage upstream-compat artifacts (mask + RoPE tables) when
    # PI0_UPSTREAM_MASKS=1 / PI0_SIGLIP_HF=1 is set. Otherwise sample_actions
    # would build them on host and upload inside the captured trace — which
    # trips "Event Synchronization is not supported during trace capture".
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import use_upstream_masks

    if use_upstream_masks():
        num_image_tokens = model.cfg.siglip_config.num_patches if hasattr(model, "cfg") else 256
        prefix_len = num_image_tokens * len(img_masks_ttnn) + LANG_SEQ_LEN
        model.prepare_upstream_artifacts(img_masks_ttnn, lang_masks_ttnn, prefix_len=prefix_len)
        print(f"   pre-staged upstream artifacts (prefix_len={prefix_len})")

    print(f"\n📷 Capturing trace…")
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = _call_sample_actions(model, images_ttnn, img_masks_ttnn, lang_tokens_ttnn, lang_masks_ttnn)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0
    print(f"   trace captured in {capture_ms:.2f} ms")

    # Validate
    actions = ttnn.to_torch(out_trace)
    actions = actions[:, :ah, :action_dim]
    assert actions.shape == (1, ah, action_dim)
    assert torch.isfinite(actions).all(), "trace output contains NaN/Inf"
    print(f"   ✅ output shape {tuple(actions.shape)}, all finite")

    # --------- measurements ---------
    print(f"\n⏱️  Measuring per-stage costs in steady state ({NUM_STEADY_ITERS} iters each)")

    # (1) Pure trace replay
    print(f"\n   [1/7] Pure trace replay (device-only)…")
    trace_only_ms: List[float] = []
    for _ in range(NUM_STEADY_ITERS):
        start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        trace_only_ms.append((time.perf_counter() - start) * 1000.0)

    # (2) Host noise resample (CPU only)
    print(f"   [2/7] Host noise resample (torch.randn)…")
    noise_cpu_ms: List[float] = []
    for _ in range(NUM_STEADY_ITERS):
        start = time.perf_counter()
        _ = torch.randn(1, ah_padded, action_dim, dtype=torch.float32)
        noise_cpu_ms.append((time.perf_counter() - start) * 1000.0)

    # (3) Host→Device transfer for the noise tensor (with cleanup so we don't leak L1)
    print(f"   [3/7] Host→Device noise transfer (ttnn.from_torch)…")
    h2d_noise_ms: List[float] = []
    for _ in range(NUM_STEADY_ITERS):
        noise_torch = torch.randn(1, ah_padded, action_dim, dtype=torch.float32)
        start = time.perf_counter()
        noise_ttnn = ttnn.from_torch(
            noise_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        h2d_noise_ms.append((time.perf_counter() - start) * 1000.0)
        ttnn.deallocate(noise_ttnn)

    # (4) Device→Host transfer of the action output
    print(f"   [4/7] Device→Host actions readback (ttnn.to_torch)…")
    d2h_actions_ms: List[float] = []
    for _ in range(NUM_STEADY_ITERS):
        start = time.perf_counter()
        _ = ttnn.to_torch(out_trace)
        d2h_actions_ms.append((time.perf_counter() - start) * 1000.0)

    # (5) CPU postprocess (slice + dtype convert + numpy)
    print(f"   [5/7] CPU postprocess (slice + bf16→fp32 → numpy)…")
    cpu_post_ms: List[float] = []
    actions_cached = ttnn.to_torch(out_trace)  # not timed
    for _ in range(NUM_STEADY_ITERS):
        start = time.perf_counter()
        a = actions_cached[:, :ah, :action_dim].float().numpy()
        cpu_post_ms.append((time.perf_counter() - start) * 1000.0)

    # (6) Image preprocessing × 2 cameras (resize + normalize + transpose)
    print(f"   [6/11] Image preprocessing × 2 cameras (resize+normalize+transpose)…")
    rng = np.random.default_rng(SEED)
    img_preproc_ms: List[float] = []
    for _ in range(NUM_STEADY_ITERS):
        cam1 = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        cam2 = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        start = time.perf_counter()
        img1 = _image_for_pi05(cam1)
        img2 = _image_for_pi05(cam2)
        img3 = torch.ones_like(img1) * -1.0
        img_preproc_ms.append((time.perf_counter() - start) * 1000.0)

    # (7) Host→Device transfer of all 3 images
    print(f"   [7/11] Host→Device image transfer (× 3 views)…")
    h2d_images_ms: List[float] = []
    cam1 = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    cam2 = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    img1_t = _image_for_pi05(cam1)
    img2_t = _image_for_pi05(cam2)
    img3_t = torch.ones_like(img1_t) * -1.0
    imgs_torch = [img1_t, img2_t, img3_t]
    for _ in range(NUM_STEADY_ITERS):
        start = time.perf_counter()
        imgs_ttnn_local = [
            ttnn.from_torch(
                im,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for im in imgs_torch
        ]
        h2d_images_ms.append((time.perf_counter() - start) * 1000.0)
        for t in imgs_ttnn_local:
            ttnn.deallocate(t)

    # (8) Tokenization — try real SentencePiece if available, else random fallback
    use_sp = os.path.exists(TOKENIZER_PATH)
    if use_sp:
        import sentencepiece

        sp = sentencepiece.SentencePieceProcessor()
        sp.load(TOKENIZER_PATH)
        state_bins = rng.integers(0, 256, size=8, dtype=np.int32)
        task_desc = "pick up the black bowl between the plate and the ramekin and place it on the plate"
        print(f"   [8/11] Tokenization (SentencePiece encode + pad to {LANG_SEQ_LEN})…")
    else:
        print(f"   [8/11] Tokenization (RANDOM fallback — SentencePiece model not found)…")
    tokenize_ms: List[float] = []
    for _ in range(NUM_STEADY_ITERS):
        if use_sp:
            start = time.perf_counter()
            tok, msk = _make_tokens_sentencepiece(sp, task_desc, state_bins)
            tokenize_ms.append((time.perf_counter() - start) * 1000.0)
        else:
            start = time.perf_counter()
            tok, msk = _make_tokens_random(rng)
            tokenize_ms.append((time.perf_counter() - start) * 1000.0)

    # (9) Host→Device transfer of tokens + lang_mask
    print(f"   [9/11] Host→Device tokens + lang_mask…")
    h2d_tokens_ms: List[float] = []
    for _ in range(NUM_STEADY_ITERS):
        if use_sp:
            tok, msk = _make_tokens_sentencepiece(sp, task_desc, state_bins)
        else:
            tok, msk = _make_tokens_random(rng)
        start = time.perf_counter()
        tok_ttnn = ttnn.from_torch(
            tok.to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        msk_ttnn = ttnn.from_torch(
            msk.to(torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        h2d_tokens_ms.append((time.perf_counter() - start) * 1000.0)
        ttnn.deallocate(tok_ttnn)
        ttnn.deallocate(msk_ttnn)

    # (10) Realistic full call (libero-style): preprocess + tokenize + all H2D + trace + D2H + post
    print(f"   [10/11] Realistic libero-style full call (preprocess + H2D + trace + D2H + post)…")
    full_call_ms: List[float] = []
    # Pre-stage a fixed camera image set so each iteration measures the same work
    cam1 = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    cam2 = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    for _ in range(NUM_STEADY_ITERS):
        start = time.perf_counter()
        # Image preprocessing
        img1 = _image_for_pi05(cam1)
        img2 = _image_for_pi05(cam2)
        img3 = torch.ones_like(img1) * -1.0
        # Image H2D
        imgs_ttnn_local = [
            ttnn.from_torch(
                im, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for im in (img1, img2, img3)
        ]
        # Tokenize + H2D
        if use_sp:
            tok, msk = _make_tokens_sentencepiece(sp, task_desc, state_bins)
        else:
            tok, msk = _make_tokens_random(rng)
        tok_ttnn = ttnn.from_torch(tok.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        msk_ttnn = ttnn.from_torch(msk.to(torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # Noise resample + H2D
        noise_torch = torch.randn(1, ah_padded, action_dim, dtype=torch.float32)
        noise_ttnn = ttnn.from_torch(
            noise_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # Trace replay
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        # D2H + slice + cast
        a = ttnn.to_torch(out_trace)[:, :ah, :action_dim].float().numpy()
        full_call_ms.append((time.perf_counter() - start) * 1000.0)
        # Cleanup transient device tensors so we don't leak L1/DRAM across iters
        for t in imgs_ttnn_local:
            ttnn.deallocate(t)
        ttnn.deallocate(tok_ttnn)
        ttnn.deallocate(msk_ttnn)
        ttnn.deallocate(noise_ttnn)

    # (11) Untraced sample_actions for comparison (full Python walk per op)
    print(f"   [11/11] Untraced sample_actions ({NUM_UNTRACED_ITERS} iters, Python dispatch every op)…")
    model.resample_noise = True
    untraced_ms: List[float] = []
    for _ in range(NUM_UNTRACED_ITERS):
        start = time.perf_counter()
        with torch.no_grad():
            out_u = _call_sample_actions(model, images_ttnn, img_masks_ttnn, lang_tokens_ttnn, lang_masks_ttnn)
        ttnn.synchronize_device(device)
        untraced_ms.append((time.perf_counter() - start) * 1000.0)
        if isinstance(out_u, ttnn.Tensor):
            ttnn.deallocate(out_u)

    ttnn.release_trace(device, tid)

    # --------- report ---------
    print("\n" + "=" * 92)
    print("  PI0.5 TTNN END-TO-END HOST + DEVICE BREAKDOWN (steady state)")
    print("=" * 92)
    print(f"   Trace capture (one-time):  {capture_ms:7.2f} ms")
    print("-" * 92)
    header = f"{'Stage':<46} {'avg ms':>9} {'min':>8} {'max':>8} {'std':>7} {'N':>5}"
    print(header)
    print("-" * 92)

    measurements = [
        _summarize("[1] Pure trace replay (device only)", trace_only_ms),
        _summarize("[2] Host noise resample (torch.randn)", noise_cpu_ms),
        _summarize("[3] H→D noise (from_torch)", h2d_noise_ms),
        _summarize("[4] D→H actions (to_torch)", d2h_actions_ms),
        _summarize("[5] CPU postprocess (slice+cast)", cpu_post_ms),
        _summarize("[6] Image preprocess × 2 (resize+norm+CHW)", img_preproc_ms),
        _summarize("[7] H→D images × 3 (from_torch)", h2d_images_ms),
        _summarize("[8] Tokenization" + (" (SentencePiece)" if use_sp else " (random fallback)"), tokenize_ms),
        _summarize("[9] H→D tokens + lang_mask", h2d_tokens_ms),
        _summarize("[10] Libero-style FULL call (chained)", full_call_ms),
        _summarize("[11] Untraced sample_actions (Python)", untraced_ms),
    ]
    for m in measurements:
        print(f"   {m['name']:<46} {m['avg']:>9.2f} {m['min']:>8.2f} {m['max']:>8.2f} {m['std']:>7.2f} {m['n']:>5}")
    print("-" * 92)

    # Throughput summary
    trace_avg = measurements[0]["avg"]
    full_avg = measurements[9]["avg"]  # libero-style full call
    untraced_avg = measurements[10]["avg"]

    def fps(ms):
        return 1000.0 / ms if ms > 0 else 0.0

    print(f"   Pure trace fps:           {fps(trace_avg):7.2f} chunks/s, {fps(trace_avg)*ah:7.2f} actions/s")
    print(f"   Realistic full-call fps:  {fps(full_avg):7.2f} chunks/s, {fps(full_avg)*ah:7.2f} actions/s")
    print(f"   Untraced fps:             {fps(untraced_avg):7.2f} chunks/s, {fps(untraced_avg)*ah:7.2f} actions/s")
    print("-" * 92)
    host_overhead = full_avg - trace_avg
    trace_savings = untraced_avg - trace_avg
    print(f"   Per-call host overhead vs pure trace:    +{host_overhead:6.2f} ms")
    print(f"   Trace saves vs untraced Python dispatch: −{trace_savings:6.2f} ms")
    print("=" * 92)

    assert trace_avg > 0
