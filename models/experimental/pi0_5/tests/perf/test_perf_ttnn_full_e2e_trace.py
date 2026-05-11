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

import statistics
import time
from pathlib import Path
from typing import List

import pytest
import torch
import ttnn

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_base"

NUM_WARMUP = 2
NUM_ITERS = 20
LANG_SEQ_LEN = 32  # tile-aligned
SEED = 0
TRACE_REGION_SIZE = 134_217_728  # 128 MiB — full sample_actions trace ~81 MB

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_inputs(device, batch_size: int = 1):
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
    # Pre-convert img_mask to TTNN so embed_images doesn't trigger a host->device
    # transfer during trace capture (which would call Synchronize).
    img_mask_ttnn = ttnn.from_torch(
        img_mask.float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
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
    return image_ttnn, img_mask_ttnn, lang_tokens_ttnn, lang_masks_ttnn


def _call_sample_actions(model, image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn):
    return model.sample_actions(
        images=[image_ttnn],
        img_masks=[img_mask],
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

    print(f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}")
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig()
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print(f"✅ Model loaded")

    image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn = _build_inputs(device)

    print(f"\n🔥 Warmup ({NUM_WARMUP} calls) — JIT compile of full sample_actions")
    for i in range(NUM_WARMUP):
        with torch.no_grad():
            out = _call_sample_actions(model, image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn)
        ttnn.synchronize_device(device)
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
        print(f"   warmup call {i + 1} done")

    print(f"\n📷 Capturing trace of full sample_actions…")
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = _call_sample_actions(model, image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn)
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
    print("  PI0.5 TTNN FULL END-TO-END WITH TRACE (real pi05_base weights)")
    print("=" * 72)
    print(f"   Includes:            SigLIP + VLM prefill + 10-step denoise + project")
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
