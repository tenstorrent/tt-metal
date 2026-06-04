# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 TTNN performance with trace capture (action-expert path).

Same code path as `test_perf_ttnn.py` (embed_suffix → forward_expert (adaRMS)
→ project_output) but the steady-state iterations execute a captured trace
rather than re-dispatching every op from Python. This eliminates the
host-side TTNN dispatch overhead from the timing.

Two command queues are *not* used in the hot loop here — there are no
host→device transfers per iteration in this focused test, so 2CQ would
provide no benefit. (For full end-to-end `sample_actions` with per-call
image upload, 2CQ matters; that's the pi0 e2e perf-test pattern.)

Reports:
  - Per-step latency with trace (avg / min / max / stddev), in ms
  - Denoise-step throughput, chunk throughput, action throughput

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

NUM_WARMUP_ITERATIONS = 2
NUM_INFERENCE_ITERATIONS = 20
PREFIX_LEN = 32
SEED = 0
TRACE_REGION_SIZE = 80_000_000

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_inputs(model, device, batch_size: int = 1):
    cfg = model.config
    ec = cfg.expert_config

    torch.manual_seed(SEED)
    # Host-pad to ah_padded — the sharded RMSNorm requires tile-aligned logical
    # shape (cfg.action_horizon isn't tile-aligned in general, e.g. 50 → 64).
    # Same workaround as prof_one_denoise_step.py and test_perf_ttnn_full_e2e_trace_2cq.py.
    ah_padded = model._action_horizon_padded
    noisy_actions = torch.zeros(batch_size, ah_padded, cfg.action_dim, dtype=torch.float32)
    noisy_actions[:, : cfg.action_horizon, :] = torch.randn(batch_size, cfg.action_horizon, cfg.action_dim)
    timestep = torch.tensor([0.5])

    noisy_ttnn = ttnn.from_torch(
        noisy_actions,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    t_ttnn = ttnn.from_torch(
        timestep,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Prefix KV must match the suffix-side dtype (bf8_b) so the keep_padded
    # concat path validates. See note in test_perf_single_expert_block.py and
    # prof_one_denoise_step.py.
    prefix_padded = ((PREFIX_LEN + 31) // 32) * 32
    prefix_kv_cache = []
    for _ in range(ec.depth):
        k = torch.zeros(batch_size, ec.num_kv_heads, prefix_padded, ec.head_dim, dtype=torch.float32)
        v = torch.zeros(batch_size, ec.num_kv_heads, prefix_padded, ec.head_dim, dtype=torch.float32)
        k[:, :, :PREFIX_LEN] = torch.randn(batch_size, ec.num_kv_heads, PREFIX_LEN, ec.head_dim) * 0.1
        v[:, :, :PREFIX_LEN] = torch.randn(batch_size, ec.num_kv_heads, PREFIX_LEN, ec.head_dim) * 0.1
        prefix_kv_cache.append(
            (
                ttnn.from_torch(k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device),
                ttnn.from_torch(v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device),
            )
        )

    return noisy_ttnn, t_ttnn, prefix_kv_cache


def _one_denoise_step(model, noisy_ttnn, t_ttnn, prefix_kv_cache):
    """suffix → forward_expert(adaRMS) → action_out_proj. One full denoise step."""
    suffix_embs, _, _, adarms_cond = model.embed_suffix(None, noisy_ttnn, t_ttnn)
    expert_out, _ = model.backbone.forward_expert(
        suffix_embs,
        adarms_cond=adarms_cond,
        past_key_values=prefix_kv_cache,
    )
    return model.suffix_embedding.project_output(expert_out)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_ttnn_perf_trace(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    print(f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}")
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig()
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print(f"✅ Model loaded ({cfg.expert_config.depth} expert layers)")

    noisy_ttnn, t_ttnn, prefix_kv = _build_inputs(model, device, batch_size=1)

    # Warmup primes JIT compilation; trace can only capture once the graph is hot.
    print(f"\n🔥 Warmup ({NUM_WARMUP_ITERATIONS} iterations) — JIT compile / cache fill")
    for i in range(NUM_WARMUP_ITERATIONS):
        out = _one_denoise_step(model, noisy_ttnn, t_ttnn, prefix_kv)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
        print(f"   warmup iter {i + 1} done")

    # Capture trace. The captured ops will be replayed verbatim.
    print(f"\n📷 Capturing trace…")
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = _one_denoise_step(model, noisy_ttnn, t_ttnn, prefix_kv)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0
    print(f"   trace captured in {capture_ms:.2f} ms")

    print(f"\n⏱️  Measuring traced replay ({NUM_INFERENCE_ITERATIONS} iterations)")
    times_ms: List[float] = []
    for i in range(NUM_INFERENCE_ITERATIONS):
        start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        print(f"   iter {i + 1:2d}: {elapsed_ms:7.2f} ms")

    ttnn.release_trace(device, tid)

    avg = statistics.mean(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    sd = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    steps_per_chunk = cfg.num_denoising_steps
    chunk_ms_est = avg * steps_per_chunk
    chunks_per_sec = 1000.0 / chunk_ms_est if chunk_ms_est > 0 else 0.0
    actions_per_sec = chunks_per_sec * cfg.action_horizon

    print("\n" + "=" * 72)
    print("  PI0.5 TTNN PERFORMANCE WITH TRACE (action-expert path)")
    print("=" * 72)
    print(f"   Expert layers:       {cfg.expert_config.depth}")
    print(f"   Suffix tokens:       {cfg.action_horizon}")
    print(f"   Synthetic prefix:    {PREFIX_LEN} tokens")
    print(f"   Trace capture:       {capture_ms:7.2f} ms (one-time)")
    print(f"   Iterations:          {NUM_INFERENCE_ITERATIONS} replays")
    print("-" * 72)
    print(f"   Per-step avg:        {avg:7.2f} ms")
    print(f"   Per-step min:        {mn:7.2f} ms")
    print(f"   Per-step max:        {mx:7.2f} ms")
    print(f"   Per-step stddev:     {sd:7.2f} ms")
    print(f"   Denoise step rate:   {1000.0 / avg:7.2f} steps/s")
    print("-" * 72)
    print(f"   At {steps_per_chunk} steps / chunk:")
    print(f"     Chunk latency:     {chunk_ms_est:7.2f} ms")
    print(f"     Chunk throughput:  {chunks_per_sec:7.2f} chunks/s")
    print(f"     Action throughput: {actions_per_sec:7.2f} actions/s")
    print("=" * 72)

    assert avg > 0
