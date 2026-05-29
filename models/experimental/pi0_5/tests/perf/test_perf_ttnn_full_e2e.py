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

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_base"
CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))

NUM_WARMUP = 0
NUM_ITERS = 1
LANG_SEQ_LEN = 256
SEED = 0
TRACE_REGION_SIZE = 80_000_000

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_inputs(cfg, device, batch_size: int = 1):
    torch.manual_seed(SEED)
    image = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
    img_mask = torch.ones(batch_size, dtype=torch.bool)
    lang_tokens = torch.randint(0, 256000, (batch_size, LANG_SEQ_LEN), dtype=torch.int32)
    lang_masks = torch.ones(batch_size, LANG_SEQ_LEN, dtype=torch.bool)

    image_ttnn = ttnn.from_torch(
        image,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
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
    return image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_ttnn_full_e2e_fps(device):
    """End-to-end `sample_actions` latency on real pi05_base weights."""
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

    image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn = _build_inputs(cfg, device)

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
                    images=[image_ttnn],
                    img_masks=[img_mask],
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
                images=[image_ttnn],
                img_masks=[img_mask],
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
    print("  PI0.5 TTNN END-TO-END PERFORMANCE (real pi05_base weights)")
    print("=" * 72)
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
