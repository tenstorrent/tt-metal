# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 TTNN end-to-end performance — full `sample_actions` captured as a
single TTNN trace.

Trace captures: SigLIP image encoding + VLM prefix prefill + 10-step
denoise loop + action projection. Inputs (images, lang tokens, masks) are
pre-staged once on device and the trace re-references them on every replay.

Steady-state replay latency = real e2e fps for pi0.5 on this hardware.

Skipped if the checkpoint isn't present locally.
"""

import os
import statistics
import time
from pathlib import Path
from typing import List

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_libero_upstream"
CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))


NUM_WARMUP = int(os.environ.get("PI05_TRACE_NUM_WARMUP", "2"))
NUM_ITERS = int(os.environ.get("PI05_TRACE_NUM_ITERS", "20"))
LANG_SEQ_LEN = 256  # tile-aligned
SEED = 0
TRACE_REGION_SIZE = 134_217_728  # 128 MiB — full sample_actions trace ~81 MB
# Production pi0.5 (lerobot/pi05_base + pi05_libero_upstream) declares 3 image
# slots in config.json — base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb. For
# LIBERO (single arm), the right_wrist slot is zero-padded with img_mask=False
# (see eval/libero_rollout.py:336-355) — but SigLIP runs BEFORE the mask is
# consumed (see tt/ttnn_prefix.py:99-100 vs :117+), so the production SigLIP
# batch is 3, not 2. Override with PI0_NUM_CAMERAS=1 or =2 for A/B.
NUM_CAMERAS = int(os.environ.get("PI0_NUM_CAMERAS", "2"))

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_inputs(device, num_cameras: int = NUM_CAMERAS):
    torch.manual_seed(SEED)
    # One (1, 3, 224, 224) tensor per camera. The prefix-embed (ttnn_prefix.py:86-109)
    # detects same-shape inputs and runs SigLIP once at bs=num_cameras via ttnn.concat.
    images = [torch.randn(1, 3, 224, 224, dtype=torch.float32) for _ in range(num_cameras)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(num_cameras)]
    lang_tokens = torch.randint(0, 256000, (1, LANG_SEQ_LEN), dtype=torch.int32)
    lang_masks = torch.ones(1, LANG_SEQ_LEN, dtype=torch.bool)

    # PI0_SIGLIP_USE_FOLD=1 — pre-stack ALL cameras on host into one (N, H, W, 3)
    # NHWC tensor and upload as a single ROW_MAJOR tensor. Eliminates BOTH the slow
    # on-device BCHW→BHWC permute AND the expensive ROW_MAJOR cross-camera concat
    # (which costs ~0.5 ms on device vs ~free on TILE batch dim).
    _use_fold = os.environ.get("PI0_SIGLIP_USE_FOLD", "").lower() in ("1", "true", "yes", "on")
    if _use_fold:
        # Host: list of (1, 3, 224, 224) → permute each → concat along batch → (N, 224, 224, 3)
        # Then pre-reshape to (N, H, W/patch, C*patch) so the device-side reshape
        # before fold (~0.29 ms in TTNN's data-movement reshape) is eliminated.
        # Patch size 14 hardcoded — the perf test only runs at the SigLIP default.
        _PATCH = 14
        stacked_host = torch.cat([im.permute(0, 2, 3, 1).contiguous() for im in images], dim=0)
        N_, H_, W_, C_ = stacked_host.shape
        stacked_host = stacked_host.reshape(N_, H_, W_ // _PATCH, C_ * _PATCH).contiguous()
        stacked_ttnn = ttnn.from_torch(
            stacked_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # embed_prefix expects a list; pass a list of 1 pre-stacked tensor.
        # The model side (see ttnn_prefix.py) detects the pre-stacked case via
        # PI0_SIGLIP_USE_FOLD and shape and bypasses the cross-camera concat.
        images_ttnn = [stacked_ttnn]
    else:
        images_ttnn = [
            ttnn.from_torch(
                im,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for im in images
        ]
    # Pre-convert img_masks to TTNN so embed_images doesn't trigger a host->device
    # transfer during trace capture (which would call Synchronize).
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


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_ttnn_full_e2e_trace(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    action_horizon = action_horizon_from_checkpoint(CHECKPOINT_DIR)
    num_denoising_steps = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "10"))
    print(
        f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}  "
        f"(action_horizon={action_horizon}, num_denoising_steps={num_denoising_steps})"
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig(action_horizon=action_horizon, num_denoising_steps=num_denoising_steps)
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print(f"✅ Model loaded")

    images_ttnn, img_masks_ttnn, lang_tokens_ttnn, lang_masks_ttnn = _build_inputs(device)
    siglip_bs = int(images_ttnn[0].shape[0]) if len(images_ttnn) == 1 else len(images_ttnn)
    print(
        f"   num_cameras={NUM_CAMERAS} (SigLIP runs bs={siglip_bs}{' via host fold' if len(images_ttnn) == 1 and NUM_CAMERAS > 1 else ' via concat'})"
    )

    print(f"\n🔥 Warmup ({NUM_WARMUP} calls) — JIT compile of full sample_actions")
    for i in range(NUM_WARMUP):
        with torch.no_grad():
            out = _call_sample_actions(model, images_ttnn, img_masks_ttnn, lang_tokens_ttnn, lang_masks_ttnn)
        ttnn.synchronize_device(device)
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
        print(f"   warmup call {i + 1} done")

    # Disable per-call noise resampling for trace: a ttnn.from_torch inside
    # sample_actions would require a host→device transfer that trace capture
    # forbids. The captured trace reuses model.x_t_ttnn — refresh it between
    # trace replays by setting model.x_t_ttnn from host before each replay.
    model.resample_noise = False

    # Pre-stage upstream-compat artifacts (mask + RoPE tables) before trace
    # capture. When PI0_UPSTREAM_MASKS=1 or PI0_SIGLIP_HF=1, sample_actions
    # would otherwise build these on host + upload them inside the captured
    # region — which trace capture forbids. With the artifacts cached on the
    # model, sample_actions consumes them by reference and the captured
    # trace replays cleanly. No-op when neither env var is set.
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import use_upstream_masks

    if use_upstream_masks():
        # prefix_len = num_image_tokens · num_cameras + LANG_SEQ_LEN.
        # At num_cameras=2 this is 256·2 + 256 = 768 (was 512 at bs=1).
        num_image_tokens = cfg.siglip_config.num_patches
        prefix_len = num_image_tokens * len(img_masks_ttnn) + LANG_SEQ_LEN
        model.prepare_upstream_artifacts(img_masks_ttnn, lang_masks_ttnn, prefix_len=prefix_len)
        print(f"   pre-staged upstream artifacts (prefix_len={prefix_len})")

    print(f"\n📷 Capturing trace of full sample_actions…")
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = _call_sample_actions(model, images_ttnn, img_masks_ttnn, lang_tokens_ttnn, lang_masks_ttnn)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0
    print(f"   trace captured in {capture_ms:.2f} ms")

    # Validate output of the captured trace.
    actions = ttnn.to_torch(out_trace)
    actions = actions[:, : cfg.action_horizon, : cfg.action_dim]
    assert actions.shape == (1, cfg.action_horizon, cfg.action_dim)
    assert torch.isfinite(actions).all(), "trace output contains NaN/Inf"
    print(f"   ✅ output shape {tuple(actions.shape)}, all finite")

    print(f"\n⏱️  Measuring traced sample_actions ({NUM_ITERS} calls)")
    times_ms: List[float] = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        print(f"   call {i + 1:2d}: {elapsed_ms:7.2f} ms")

    ttnn.release_trace(device, tid)

    avg = statistics.mean(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    sd = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    chunks_per_sec = 1000.0 / avg if avg > 0 else 0.0
    actions_per_sec = chunks_per_sec * cfg.action_horizon

    print("\n" + "=" * 72)
    print(f"  PI0.5 TTNN FULL END-TO-END WITH TRACE ({CHECKPOINT_DIR.name})")
    print("=" * 72)
    print(
        f"   Includes:            SigLIP (bs={siglip_bs}) + VLM prefill + {num_denoising_steps}-step denoise + project"
    )
    print(f"   Trace capture:       {capture_ms:7.2f} ms (one-time)")
    print(f"   Iterations:          {NUM_ITERS} traced replays")
    print("-" * 72)
    print(f"   Per-call avg:        {avg:7.2f} ms")
    print(f"   Per-call min:        {mn:7.2f} ms")
    print(f"   Per-call max:        {mx:7.2f} ms")
    print(f"   Per-call stddev:     {sd:7.2f} ms")
    print("-" * 72)
    print(f"   Chunk throughput:    {chunks_per_sec:7.2f} chunks/s")
    print(f"   Action throughput:   {actions_per_sec:7.2f} actions/s")
    print("=" * 72)
    assert avg > 0
