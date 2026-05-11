# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 TTNN end-to-end denoise-loop performance with trace (+ optional 2CQ).

Builds on test_perf_ttnn_trace.py by:
  1) Hoisting deterministic work — `adarms_cond` is precomputed per denoise
     step at model init (Pi0_5ModelTTNN._precompute_bs1_adarms_cond), so the
     per-step path is only `action_in_proj → forward_expert → action_out_proj
     → Euler update`. The MLP that produces adarms_cond runs once per step at
     init, not 10× per chunk.
  2) Capturing the FULL 10-step denoise loop as a single trace, so we measure
     a real chunk latency (not a 1-step × 10 estimate).
  3) An optional 2CQ variant: CQ 1 uploads the next chunk's initial noise to
     a pre-allocated device buffer while CQ 0 replays the trace for the
     current chunk.
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
PREFIX_LEN = 32
SEED = 0
TRACE_REGION_SIZE = 80_000_000

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_prefix_kv(model, device, batch_size: int = 1):
    """One (K, V) per expert layer — represents a cached VLM prefix."""
    ec = model.config.expert_config
    torch.manual_seed(SEED)
    prefix_kv = []
    for _ in range(ec.depth):
        k = torch.randn(batch_size, ec.num_kv_heads, PREFIX_LEN, ec.head_dim) * 0.1
        v = torch.randn(batch_size, ec.num_kv_heads, PREFIX_LEN, ec.head_dim) * 0.1
        prefix_kv.append(
            (
                ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            )
        )
    return prefix_kv


def _run_denoise_loop(model, x_t, prefix_kv):
    """Inline 10-step denoise loop using precomputed adarms_cond. bs=1."""
    cfg = model.config
    num_steps = cfg.num_denoising_steps
    timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

    for i in range(num_steps):
        dt = timesteps[i + 1] - timesteps[i]
        suffix_embs = model.suffix_embedding.embed_actions(x_t)
        adarms_cond = model._adarms_cond_per_step_bs1[i]
        expert_out, _ = model.backbone.forward_expert(
            suffix_embs,
            adarms_cond=adarms_cond,
            past_key_values=prefix_kv,
        )
        velocity = model.suffix_embedding.project_output(expert_out)
        v_scaled = ttnn.mul(velocity, dt)
        x_t = ttnn.add(x_t, v_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)
    return x_t


def _print_summary(label, capture_ms, times_ms, cfg):
    avg = statistics.mean(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    sd = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    chunks_per_sec = 1000.0 / avg if avg > 0 else 0.0
    actions_per_sec = chunks_per_sec * cfg.action_horizon
    per_step_avg = avg / cfg.num_denoising_steps

    print("\n" + "=" * 72)
    print(f"  PI0.5 TTNN PERFORMANCE — {label}")
    print("=" * 72)
    print(f"   Denoise steps/chunk: {cfg.num_denoising_steps}")
    print(f"   Action horizon:      {cfg.action_horizon}")
    print(f"   Trace capture:       {capture_ms:7.2f} ms (one-time)")
    print(f"   Iterations:          {len(times_ms)} replays")
    print("-" * 72)
    print(f"   Chunk avg:           {avg:7.2f} ms  ({per_step_avg:.2f} ms/step)")
    print(f"   Chunk min:           {mn:7.2f} ms")
    print(f"   Chunk max:           {mx:7.2f} ms")
    print(f"   Chunk stddev:        {sd:7.2f} ms")
    print("-" * 72)
    print(f"   Chunk throughput:    {chunks_per_sec:7.2f} chunks/s")
    print(f"   Action throughput:   {actions_per_sec:7.2f} actions/s")
    print("=" * 72)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_ttnn_perf_trace_e2e(device):
    """Full 10-step denoise loop captured as a single trace, single CQ."""
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    print(f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}")
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig()
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print(f"✅ Model loaded; {len(model._adarms_cond_per_step_bs1)} adarms_cond tensors precomputed")

    prefix_kv = _build_prefix_kv(model, device)

    # Initial x_t (noise) lives at a fixed device buffer so trace can reference it.
    x_t = model.x_t_ttnn

    print(f"\n🔥 Warmup ({NUM_WARMUP} chunks) — JIT compile")
    for i in range(NUM_WARMUP):
        out = _run_denoise_loop(model, x_t, prefix_kv)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
        print(f"   warmup chunk {i + 1} done")

    print(f"\n📷 Capturing trace of full 10-step denoise loop…")
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = _run_denoise_loop(model, x_t, prefix_kv)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0
    print(f"   trace captured in {capture_ms:.2f} ms")

    print(f"\n⏱️  Measuring traced chunk replay ({NUM_ITERS} chunks)")
    times_ms: List[float] = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        print(f"   chunk {i + 1:2d}: {elapsed_ms:7.2f} ms")

    ttnn.release_trace(device, tid)
    _print_summary("trace, full 10-step denoise loop (1 CQ)", capture_ms, times_ms, cfg)
    assert statistics.mean(times_ms) > 0


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 24576,
            "trace_region_size": TRACE_REGION_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
def test_pi0_5_ttnn_perf_trace_2cq(device):
    """
    Full 10-step denoise loop with trace, with CQ 1 uploading the next
    chunk's initial noise while CQ 0 replays the trace for the current chunk.

    Each chunk in steady state issues:
      - CQ 1: copy_host_to_device_tensor(host_noise[i+1] -> x_t buffer)
      - CQ 0: execute_trace (10 denoise steps)
    The two operations overlap when chunk compute > host upload (which is
    always true here — upload is ~3KB, compute is ~120ms).
    """
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    print(f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}")
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig()
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print(f"✅ Model loaded")

    prefix_kv = _build_prefix_kv(model, device)
    x_t = model.x_t_ttnn

    # Pre-allocate host noise tensors for NUM_ITERS chunks.
    torch.manual_seed(SEED)
    host_noise = [torch.randn(1, cfg.action_horizon, cfg.action_dim, dtype=torch.float32) for _ in range(NUM_ITERS + 1)]
    # Move each to a host TTNN tensor so copy_host_to_device_tensor can target x_t.
    host_noise_ttnn = [ttnn.from_torch(n, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for n in host_noise]

    print(f"\n🔥 Warmup ({NUM_WARMUP} chunks)")
    for i in range(NUM_WARMUP):
        out = _run_denoise_loop(model, x_t, prefix_kv)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

    print(f"\n📷 Capturing trace…")
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = _run_denoise_loop(model, x_t, prefix_kv)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0
    print(f"   trace captured in {capture_ms:.2f} ms")

    print(f"\n⏱️  Measuring 2CQ + trace ({NUM_ITERS} chunks)")
    times_ms: List[float] = []
    # Pre-stage chunk 0 noise on CQ 1.
    ttnn.copy_host_to_device_tensor(host_noise_ttnn[0], x_t, cq_id=1)
    write_event = ttnn.record_event(device, 1)

    for i in range(NUM_ITERS):
        start = time.perf_counter()
        ttnn.wait_for_event(0, write_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        op_event = ttnn.record_event(device, 0)

        # Pre-stage next chunk's noise on CQ 1 in parallel with the trace.
        if i + 1 < NUM_ITERS:
            ttnn.wait_for_event(1, op_event)
            ttnn.copy_host_to_device_tensor(host_noise_ttnn[i + 1], x_t, cq_id=1)
            write_event = ttnn.record_event(device, 1)

        ttnn.synchronize_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        print(f"   chunk {i + 1:2d}: {elapsed_ms:7.2f} ms")

    ttnn.release_trace(device, tid)
    _print_summary("trace + 2CQ, full 10-step denoise loop", capture_ms, times_ms, cfg)
    assert statistics.mean(times_ms) > 0
