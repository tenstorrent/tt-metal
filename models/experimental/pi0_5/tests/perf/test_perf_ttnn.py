# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 TTNN performance test on the real lerobot/pi05_base checkpoint.

Measures steady-state per-denoise-step latency on the action-expert path
(embed_suffix → forward_expert(adaRMS) → project_output) with a synthetic
prefix KV cache. This isolates the pi0.5-specific cost (adaRMS norms +
gated residuals across 18 expert layers + final adaRMS norm) from the
shared VLM prefix, and is a fair proxy for steady-state inference since
the denoise loop dominates total `sample_actions` time.

Reports:
  - Per-step latency (avg / min / max / stddev), in ms
  - Denoise-step throughput (steps / s)
  - Action-chunk throughput @ `num_denoising_steps` per chunk, plus
    per-action throughput @ `action_horizon` actions per chunk.

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
NUM_INFERENCE_ITERATIONS = 10
PREFIX_LEN = 32  # tile-aligned synthetic prefix
SEED = 0

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_inputs(model, device, batch_size: int = 1):
    cfg = model.config
    ec = cfg.expert_config

    torch.manual_seed(SEED)
    noisy_actions = torch.randn(batch_size, cfg.action_horizon, cfg.action_dim)
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

    # Synthetic prefix KV cache: 1 (K, V) pair per expert layer (allocated once).
    prefix_kv_cache = []
    for _ in range(ec.depth):
        k = torch.randn(batch_size, ec.num_kv_heads, PREFIX_LEN, ec.head_dim) * 0.1
        v = torch.randn(batch_size, ec.num_kv_heads, PREFIX_LEN, ec.head_dim) * 0.1
        prefix_kv_cache.append(
            (
                ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            )
        )

    return noisy_ttnn, t_ttnn, prefix_kv_cache


def _one_denoise_step(model, noisy_ttnn, t_ttnn, prefix_kv_cache):
    """suffix → forward_expert (adaRMS) → action_out_proj. One full denoise step."""
    suffix_embs, _, _, adarms_cond = model.embed_suffix(None, noisy_ttnn, t_ttnn)
    expert_out, _ = model.backbone.forward_expert(
        suffix_embs,
        adarms_cond=adarms_cond,
        past_key_values=prefix_kv_cache,
    )
    velocity = model.suffix_embedding.project_output(expert_out)
    # Force completion of the queued op stream so timing is meaningful.
    ttnn.synchronize_device(model.device)
    return velocity


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pi0_5_ttnn_perf(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    print(f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}")
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig()  # default 10 denoise steps, horizon=50, dim=32
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print(f"✅ Model loaded ({cfg.expert_config.depth} expert layers)")

    noisy_ttnn, t_ttnn, prefix_kv = _build_inputs(model, device, batch_size=1)

    print(f"\n🔥 Warmup ({NUM_WARMUP_ITERATIONS} iterations)")
    for i in range(NUM_WARMUP_ITERATIONS):
        _ = _one_denoise_step(model, noisy_ttnn, t_ttnn, prefix_kv)
        print(f"   warmup iter {i + 1} done")

    print(f"\n⏱️  Measuring ({NUM_INFERENCE_ITERATIONS} iterations)")
    times_ms: List[float] = []
    for i in range(NUM_INFERENCE_ITERATIONS):
        start = time.perf_counter()
        _ = _one_denoise_step(model, noisy_ttnn, t_ttnn, prefix_kv)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        print(f"   iter {i + 1:2d}: {elapsed_ms:7.2f} ms")

    avg = statistics.mean(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    sd = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    steps_per_chunk = cfg.num_denoising_steps
    chunk_ms_est = avg * steps_per_chunk  # ignoring prefix prefill cost
    chunks_per_sec = 1000.0 / chunk_ms_est if chunk_ms_est > 0 else 0.0
    actions_per_sec = chunks_per_sec * cfg.action_horizon

    print("\n" + "=" * 72)
    print("  PI0.5 TTNN PERFORMANCE (action-expert path, real pi05_base weights)")
    print("=" * 72)
    print(f"   Expert layers:       {cfg.expert_config.depth}")
    print(f"   Suffix tokens:       {cfg.action_horizon} (no state token)")
    print(f"   Synthetic prefix:    {PREFIX_LEN} tokens")
    print(f"   Iterations:          {NUM_INFERENCE_ITERATIONS} (after {NUM_WARMUP_ITERATIONS} warmup)")
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
    print(f"     Action throughput: {actions_per_sec:7.2f} actions/s  ({cfg.action_horizon}/chunk)")
    print("=" * 72)

    assert avg > 0, "average latency must be > 0"
