# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Achieved DRAM bandwidth of streaming ops at the residual-stream size (16384 x 2048), traced.

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/probe_dram_bw.py
"""

import time

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.fused_add_rmsnorm import (
    fused_add_rmsnorm,
    make_add_norm_constants,
)

REPS = 10
M, W = 16384, 2048


def timed(device, fn):
    fn()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [fn() for _ in range(REPS)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        best = min(best, (time.perf_counter() - t) / REPS * 1e6)
    ttnn.release_trace(device, tid)
    for o in outs:
        for t in o if isinstance(o, tuple) else (o,):
            ttnn.deallocate(t)
    return best


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=0, trace_region_size=100_000_000)
    a = ttnn.from_torch(torch.randn(1, 1, M, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(1, 1, M, W), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    consts = make_add_norm_constants(torch.rand(W) + 0.5, 1e-5, device)
    mb = M * W / 1e6
    cases = {
        "clone bf16 (2 B r + 2 B w)": (lambda: ttnn.clone(a), 4 * mb),
        "add bf16+bfp8 -> bf16 (3 B r + 2 B w)": (lambda: ttnn.add(a, b, dtype=ttnn.bfloat16), 5 * mb),
        "fused_add_rmsnorm (3 B r + 3 B w)": (
            lambda: fused_add_rmsnorm(a, b, consts, out_dtype=ttnn.bfloat8_b, sum_dtype=ttnn.bfloat16),
            6.125 * mb,
        ),
        "fused_add_rmsnorm HiFi2": (
            lambda: fused_add_rmsnorm(
                a, b, consts, out_dtype=ttnn.bfloat8_b, sum_dtype=ttnn.bfloat16, fidelity=ttnn.MathFidelity.HiFi2
            ),
            6.125 * mb,
        ),
        "fused_add_rmsnorm data movement only": (
            lambda: fused_add_rmsnorm(
                a, b, consts, out_dtype=ttnn.bfloat8_b, sum_dtype=ttnn.bfloat16, defines=[("DM_ONLY", "1")]
            ),
            6.125 * mb,
        ),
        "fused_add_rmsnorm LoFi": (
            lambda: fused_add_rmsnorm(
                a, b, consts, out_dtype=ttnn.bfloat8_b, sum_dtype=ttnn.bfloat16, fidelity=ttnn.MathFidelity.LoFi
            ),
            6.125 * mb,
        ),
    }
    for name, (fn, mbytes) in cases.items():
        us = timed(device, fn)
        print(f"{name}: {us:7.1f} us  {mbytes / us * 1e3:6.1f} GB/s")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
