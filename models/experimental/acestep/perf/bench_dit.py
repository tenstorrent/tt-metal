# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sweep a single ACE-Step DiT forward across sequence lengths and report latency + regime.

Measures one DiT step (the denoise-loop body) at increasing latent lengths T (T' = T/2 is the
DiT self-attention sequence). Isolates where the model transitions from host-dispatch-bound
(flat latency at short T') to compute/attention-bound (super-linear at long T'), and finds the
practical ceiling. This is a capacity/latency probe, not a PCC test.

Reuses the shared `models.perf.benchmarking_utils.BenchmarkProfiler` (same harness as tt_dit's
performance tests) rather than a bespoke timer.

    export ACESTEP_PIPELINE_DIR=/path/to/acestep_pipeline
    python models/experimental/acestep/perf/bench_dit.py
"""

import statistics
import sys

import torch
import ttnn

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline

PATCH = 2
HIDDEN_LAT = 64
CTX_LAT = 128
ENC = 128  # fixed conditioning (cross-attn KV) length for the sweep
LATENT_HZ = 25
WARMUP = 2
ITERS = 3

# T' = T/2. Default sweep spans the flat/overhead-bound region through the compute-bound region.
T_SWEEP = [2, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096]


def _regime(t_prime, ms_per_tprime_prev, ms):
    if t_prime <= 128:
        return "overhead-bound"
    if t_prime <= 1024:
        return "compute-bound"
    return "attention-bound"


def bench_one(pipe, device, profiler, T, iteration):
    """Time one DiT forward at latent length T. Returns mean seconds, or None on failure."""
    try:
        xt = ttnn.from_torch(
            torch.randn(1, 1, T, HIDDEN_LAT), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        ctx = ttnn.from_torch(
            torch.randn(1, 1, T, CTX_LAT), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        enc = ttnn.from_torch(torch.randn(1, 1, ENC, 2048), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        t_prime = T // PATCH
        cos_tt, sin_tt = pipe._rope_tables(t_prime)
        sliding = pipe._sliding_mask(t_prime)
        t_scalar = torch.tensor([1.0], dtype=torch.float32)

        def step():
            vt = pipe.dit.forward(xt, ctx, t_scalar, t_scalar, cos_tt, sin_tt, enc, sliding_mask=sliding)
            ttnn.synchronize_device(device)
            ttnn.deallocate(vt)

        for _ in range(WARMUP):
            step()
        times = []
        for it in range(ITERS):
            with profiler(f"dit_step_T{T}", iteration=it):
                step()
            times.append(profiler.get_duration(f"dit_step_T{T}", it))

        for t in (xt, ctx, enc):
            ttnn.deallocate(t)
        return statistics.mean(times)
    except Exception as e:
        print(f"  T={T} (T'={T // PATCH}) FAILED: {str(e).splitlines()[0][:100]}")
        return None


def main():
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device, with_vae=False, with_encoders=False)
        profiler = BenchmarkProfiler()
        print(f"{'T':>7} {'Tprime':>7} {'seconds':>9} {'latency':>10} {'regime':>16}   50-step denoise")
        prev = None
        for T in T_SWEEP:
            mean = bench_one(pipe, device, profiler, T, 0)
            if mean is None:
                break
            tp = T // PATCH
            print(
                f"{T:>7} {tp:>7} {T / LATENT_HZ:>8.2f}s {mean * 1000:>8.1f}ms {_regime(tp, prev, mean):>16}   ~{mean * 50:.1f}s"
            )
            prev = mean
            sys.stdout.flush()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
