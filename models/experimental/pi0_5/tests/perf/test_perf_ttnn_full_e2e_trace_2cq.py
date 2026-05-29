# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 TTNN end-to-end performance — full `sample_actions` captured as a single
trace on CQ 0, with CQ 1 uploading the NEXT chunk's inputs (camera frame +
noise) in parallel with the current chunk's compute.

This is the repo-canonical "performant" e2e number (same shape as the
resnet50 / vit / sentence_bert trace+2CQ runners):

  - Trace captures the FULL pipeline: SigLIP image encode + VLM prefix prefill
    + N-step denoise loop + action projection.
  - H2D of the next chunk's inputs is issued on CQ 1, event-overlapped with the
    CQ 0 trace replay — so the steady-state number is max(H2D, compute), and a
    fast-enough H2D is fully hidden behind compute.
  - D2H (reading the action chunk back to host) is INCLUDED in the timed loop
    each chunk (after trace replay + sync), so steady-state is max(H2D_in, compute, D2H_out).

Contrast:
  - test_perf_ttnn_full_e2e_trace.py     — same pipeline, device-compute floor
                                           (execute_trace only, no H2D/D2H).
  - test_perf_ttnn_full_e2e.py           — non-trace, host-dispatch bound.
  - test_perf_ttnn_trace_e2e.py (2CQ)    — same 2CQ pattern but denoise-loop only.

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

NUM_WARMUP = 2
NUM_ITERS = 20
LANG_SEQ_LEN = 256  # tile-aligned
SEED = 0
TRACE_REGION_SIZE = 134_217_728  # 128 MiB — full sample_actions trace ~81 MB

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_inputs(device, batch_size: int = 1):
    """Device-resident input buffers the trace references. These are the targets
    of the per-chunk CQ 1 H2D uploads in the steady-state loop."""
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
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE, "num_command_queues": 2}],
    indirect=True,
)
def test_pi0_5_ttnn_full_e2e_trace_2cq(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN, use_upstream_masks

    action_horizon = action_horizon_from_checkpoint(CHECKPOINT_DIR)
    num_denoising_steps = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "10"))
    print(
        f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}  "
        f"(action_horizon={action_horizon}, num_denoising_steps={num_denoising_steps})"
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig(action_horizon=action_horizon, num_denoising_steps=num_denoising_steps)
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print("✅ Model loaded")

    image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn = _build_inputs(device)

    print(f"\n🔥 Warmup ({NUM_WARMUP} calls) — JIT compile of full sample_actions")
    for i in range(NUM_WARMUP):
        with torch.no_grad():
            out = _call_sample_actions(model, image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn)
        ttnn.synchronize_device(device)
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
        print(f"   warmup call {i + 1} done")

    # Deterministic noise buffer so trace replay reads model.x_t_ttnn (the CQ 1
    # upload target) instead of allocating fresh noise inside the captured region.
    model.resample_noise = False

    # Pre-stage upstream-compat artifacts (mask + RoPE) before capture so
    # sample_actions consumes them by reference (no host->device inside trace).
    if use_upstream_masks():
        prefix_len = 256 + LANG_SEQ_LEN
        model.prepare_upstream_artifacts([img_mask], lang_masks_ttnn, prefix_len=prefix_len)
        print(f"   pre-staged upstream artifacts (prefix_len={prefix_len})")

    print("\n📷 Capturing trace of full sample_actions…")
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = _call_sample_actions(model, image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0
    print(f"   trace captured in {capture_ms:.2f} ms")

    # Validate the captured output ONCE (this D2H is intentionally outside the
    # timed loop — e2e convention excludes readback).
    actions = ttnn.to_torch(out_trace)[:, : cfg.action_horizon, : cfg.action_dim]
    assert actions.shape == (1, cfg.action_horizon, cfg.action_dim)
    assert torch.isfinite(actions).all(), "trace output contains NaN/Inf"
    print(f"   ✅ output shape {tuple(actions.shape)}, all finite")

    # Host-side staging tensors for the per-chunk CQ 1 uploads: the next camera
    # frame (the dominant per-chunk H2D) + the next noise. Built once; values are
    # irrelevant to timing (compute is data-independent) but deterministic.
    x_t = model.x_t_ttnn
    ah_padded = model._action_horizon_padded
    torch.manual_seed(SEED)
    host_imgs: List[ttnn.Tensor] = []
    host_noise: List[ttnn.Tensor] = []
    for _ in range(NUM_ITERS + 1):
        img = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        host_imgs.append(ttnn.from_torch(img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        n = torch.zeros(1, ah_padded, cfg.action_dim, dtype=torch.float32)
        n[:, : cfg.action_horizon, :] = torch.randn(1, cfg.action_horizon, cfg.action_dim)
        host_noise.append(ttnn.from_torch(n, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))

    print(f"\n⏱️  Measuring full e2e trace + 2CQ ({NUM_ITERS} chunks; H2D on CQ1, D2H included)")
    times_ms: List[float] = []
    # Pre-stage chunk 0's inputs on CQ 1.
    ttnn.copy_host_to_device_tensor(host_imgs[0], image_ttnn, cq_id=1)
    ttnn.copy_host_to_device_tensor(host_noise[0], x_t, cq_id=1)
    write_event = ttnn.record_event(device, 1)

    for i in range(NUM_ITERS):
        start = time.perf_counter()
        # CQ 0 waits until CQ 1 finished writing this chunk's inputs, then runs.
        ttnn.wait_for_event(0, write_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        op_event = ttnn.record_event(device, 0)

        # Upload next chunk's frame + noise on CQ 1, overlapped with CQ 0 compute.
        # CQ 1 waits for op_event so it only overwrites the input buffers after
        # the trace has finished consuming them.
        if i + 1 < NUM_ITERS:
            ttnn.wait_for_event(1, op_event)
            ttnn.copy_host_to_device_tensor(host_imgs[i + 1], image_ttnn, cq_id=1)
            ttnn.copy_host_to_device_tensor(host_noise[i + 1], x_t, cq_id=1)
            write_event = ttnn.record_event(device, 1)

        ttnn.synchronize_device(device)
        # D2H: read action chunk back to host (included in per-chunk timing).
        chunk_actions = ttnn.to_torch(out_trace)[:, : cfg.action_horizon, : cfg.action_dim]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        print(f"   chunk {i + 1:2d}: {elapsed_ms:7.2f} ms")

    ttnn.release_trace(device, tid)

    final = chunk_actions
    assert final.shape == (1, cfg.action_horizon, cfg.action_dim)
    assert torch.isfinite(final).all(), "final trace output contains NaN/Inf"

    avg = statistics.mean(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    sd = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    chunks_per_sec = 1000.0 / avg if avg > 0 else 0.0
    actions_per_sec = chunks_per_sec * cfg.action_horizon

    print("\n" + "=" * 72)
    print("  PI0.5 TTNN FULL E2E — TRACE + 2CQ (H2D overlapped on CQ1, D2H included)")
    print("=" * 72)
    print("   Includes:            SigLIP + VLM prefill + denoise + project")
    print(f"   H2D each chunk:      next camera frame + noise on CQ1 (overlapped)")
    print(f"   D2H each chunk:      action chunk readback via to_torch (in timed loop)")
    print(f"   Trace capture:       {capture_ms:7.2f} ms (one-time)")
    print(f"   Iterations:          {NUM_ITERS} chunks")
    print("-" * 72)
    print(f"   Per-chunk avg:       {avg:7.2f} ms")
    print(f"   Per-chunk min:       {mn:7.2f} ms")
    print(f"   Per-chunk max:       {mx:7.2f} ms")
    print(f"   Per-chunk stddev:    {sd:7.2f} ms")
    print("-" * 72)
    print(f"   Chunk throughput:    {chunks_per_sec:7.2f} chunks/s")
    print(f"   Action throughput:   {actions_per_sec:7.2f} actions/s  ({cfg.action_horizon}/chunk)")
    print("=" * 72)
    assert avg > 0
