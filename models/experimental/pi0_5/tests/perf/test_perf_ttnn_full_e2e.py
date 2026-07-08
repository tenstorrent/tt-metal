# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 TTNN end-to-end performance — the full `sample_actions` path.

Includes:
  - SigLIP image encoding
  - VLM prefix prefill (Gemma 2B, 18 layers, KV-cached)
  - 10-step denoise loop (adaRMS expert, 18 layers/step)
  - action projection

Reports:
  - Cold-start latency (1st call, includes JIT compile)
  - Steady-state latency (subsequent calls, same inputs)
  - chunks/s and actions/s

This is the headline number for pi0.5 — fps of full inference on Blackhole.
Skipped if the checkpoint isn't present locally.
"""

import os
import statistics
import time
from pathlib import Path
from typing import List

# Pin chip 9 + auto-source production env (default if not explicitly set).
# setdefault: an explicit shell export still wins. Must run before any
# ttnn / pi0_5 import so modules see the production flags at construction.
os.environ.setdefault("TT_VISIBLE_DEVICES", "9")


def _apply_production_env_defaults():
    """Source _bench_runs/pi05_production.env as DEFAULTS — no manual `source` needed."""
    import re as _re

    root = os.environ.get("TT_METAL_HOME") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), *([os.pardir] * 4))
    )
    envf = os.path.join(root, "_bench_runs", "pi05_production.env")
    if not os.path.exists(envf):
        return
    with open(envf) as f:
        for line in f:
            m = _re.match(r"\s*export\s+([A-Z0-9_]+)=(\S+)", line)
            if m:
                os.environ.setdefault(m.group(1), m.group(2))


_apply_production_env_defaults()

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_libero_upstream"
CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))

NUM_WARMUP = int(os.environ.get("PI05_E2E_NUM_WARMUP", "0"))
NUM_ITERS = int(os.environ.get("PI05_E2E_NUM_ITERS", "1"))
LANG_SEQ_LEN = 256
SEED = 0
TRACE_REGION_SIZE = 80_000_000
# Production pi0.5 LIBERO passes 3 images to SigLIP (base + wrist + zero placeholder
# for the unused right_wrist slot — see [[pi05-siglip-bs3-production]]). Default to
# bs=3 to match real production; override with PI0_NUM_CAMERAS=1/2 for A/B.
NUM_CAMERAS = int(os.environ.get("PI0_NUM_CAMERAS", "2"))

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_inputs(cfg, device, num_cameras: int = NUM_CAMERAS):
    torch.manual_seed(SEED)
    images = [torch.randn(1, 3, 224, 224, dtype=torch.float32) for _ in range(num_cameras)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(num_cameras)]
    lang_tokens = torch.randint(0, 256000, (1, LANG_SEQ_LEN), dtype=torch.int32)
    lang_masks = torch.ones(1, LANG_SEQ_LEN, dtype=torch.bool)

    # PI0_SIGLIP_USE_FOLD=1 — match the production / trace-test image upload path
    # (see test_perf_ttnn_full_e2e_trace.py:61 and [[pi05-siglip-fold-win]]).
    # Pre-stack ALL cameras on host into one (N, H, W, 3) NHWC tensor, then
    # pre-reshape to (N, H, W/patch, C*patch) so the device-side permute /
    # untilize / reshape / concat (~0.89 ms) all disappear from prefix_setup.
    _use_fold = os.environ.get("PI0_SIGLIP_USE_FOLD", "").lower() in ("1", "true", "yes", "on")
    if _use_fold:
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
    return images_ttnn, img_masks, lang_tokens_ttnn, lang_masks_ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_ttnn_full_e2e_fps(device):
    """End-to-end `sample_actions` latency on real pi05_libero_upstream weights."""
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

    images_ttnn, img_masks, lang_tokens_ttnn, lang_masks_ttnn = _build_inputs(cfg, device)
    siglip_bs = int(images_ttnn[0].shape[0]) if len(images_ttnn) == 1 else len(images_ttnn)
    print(
        f"   num_cameras={NUM_CAMERAS} (SigLIP runs bs={siglip_bs}{' via host fold' if len(images_ttnn) == 1 and NUM_CAMERAS > 1 else ' via concat'})"
    )

    # Set NUM_WARMUP=0 to skip the cold-start call entirely (useful when
    # profiling — the per-op CSV will then contain exactly NUM_ITERS
    # inferences, no extra warmup pass).
    cold_ms = float("nan")
    if NUM_WARMUP > 0:
        print(f"\n🔥 Warmup ({NUM_WARMUP} call) — full sample_actions (JIT compile)")
        cold_start = time.perf_counter()
        for _ in range(NUM_WARMUP):
            with torch.no_grad():
                out = model.sample_actions(
                    images=images_ttnn,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens_ttnn,
                    lang_masks=lang_masks_ttnn,
                    state=None,
                )
            ttnn.synchronize_device(device)
        cold_ms = (time.perf_counter() - cold_start) * 1000.0
        print(f"   cold-start full sample_actions: {cold_ms:.2f} ms")

        # Validate the output before continuing.
        actions = ttnn.to_torch(out)
        actions = actions[:, : cfg.action_horizon, : cfg.action_dim]
        assert actions.shape == (1, cfg.action_horizon, cfg.action_dim)
        assert torch.isfinite(actions).all(), "actions contain NaN/Inf"
        print(f"   ✅ output shape {tuple(actions.shape)}, all finite")
    else:
        print(f"\n⚠️  NUM_WARMUP=0 — skipping cold-start (first timed iter will include JIT compile)")

    print(f"\n⏱️  Measuring steady-state ({NUM_ITERS} sample_actions calls)")
    times_ms: List[float] = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.sample_actions(
                images=images_ttnn,
                img_masks=img_masks,
                lang_tokens=lang_tokens_ttnn,
                lang_masks=lang_masks_ttnn,
                state=None,
            )
        ttnn.synchronize_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        print(f"   call {i + 1:2d}: {elapsed_ms:7.2f} ms")

    avg = statistics.mean(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    sd = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    chunks_per_sec = 1000.0 / avg if avg > 0 else 0.0
    actions_per_sec = chunks_per_sec * cfg.action_horizon

    print("\n" + "=" * 72)
    print(f"  PI0.5 TTNN END-TO-END PERFORMANCE ({CHECKPOINT_DIR.name})")
    print("=" * 72)
    print(f"   Denoising steps:     {num_denoising_steps}")
    print(f"   Cold-start (JIT):    {cold_ms:7.2f} ms (one-time)")
    print(f"   Steady-state avg:    {avg:7.2f} ms")
    print(f"   Steady-state min:    {mn:7.2f} ms")
    print(f"   Steady-state max:    {mx:7.2f} ms")
    print(f"   Steady-state stddev: {sd:7.2f} ms")
    print("-" * 72)
    print(f"   Chunk throughput:    {chunks_per_sec:7.2f} chunks/s")
    print(f"   Action throughput:   {actions_per_sec:7.2f} actions/s  ({cfg.action_horizon}/chunk)")
    print("=" * 72)
    assert avg > 0
