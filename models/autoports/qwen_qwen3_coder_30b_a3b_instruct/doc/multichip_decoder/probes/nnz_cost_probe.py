# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What dynamic ``nnz`` costs at the shipped decode shapes.

The design phase priced dynamic mode from a sweep whose baseline was a DRAM-out,
random-weight microbenchmark reading 264.65 us for the E=128 pair, where the
profiled single-chip layer reads 92.07. Only *ratios* were used from it, which
was the right caution -- but the ratio it produced (EP + dynamic nnz = 2.13x
faster than the single-chip pair) did not survive contact with the layer, where
the multichip profile reads 82.65 us for the pair against the single chip's
92.07, i.e. **1.11x**.

This measures the gap directly, at the shapes and memory configs the shipped
decode actually uses: E=32 local experts, M=1, bfloat4_b weights, LoFi, L1
output, stage 02's tuned block widths. One die, because the sparse matmul is
per-die identical and a one-die measurement is representative.

    python nnz_cost_probe.py

Prints ``P|`` lines only. Nothing here passes a wrong ``nnz`` -- the sparsity
always has exactly ``nnz`` non-zeros -- so this run cannot hang.
See ``nnz_hazard_probe.py`` for the deliberate mismatch.
"""
import statistics
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.optimized_decoder import (
    EXPERT_IN0_BLOCK_W_DOWN,
    EXPERT_IN0_BLOCK_W_GATE_UP,
    _expert_compute_kernel_config,
    _tuned_sparse_matmul_config,
)

E, H, I = 32, 2048, 768

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=90_000_000, l1_small_size=32768)
try:
    torch.manual_seed(0)

    def w(shape):
        return ttnn.from_torch(
            torch.randn(*shape) * 0.02,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    gate_up_w = w((1, E, H, 2 * I))
    down_w = w((1, E, I, H))
    x = ttnn.from_torch(
        torch.randn(1, 1, 1, H) * 0.02,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tile = ttnn.Tile([32, 32])
    cc = _expert_compute_kernel_config(mesh)
    gu_pc = _tuned_sparse_matmul_config(1, 2 * I, H, EXPERT_IN0_BLOCK_W_GATE_UP)
    dn_pc = _tuned_sparse_matmul_config(1, H, I, EXPERT_IN0_BLOCK_W_DOWN)

    def sparsity_of(live):
        p = torch.zeros(1, 1, 1, E, dtype=torch.bfloat16)
        p[..., :live] = 1.0
        return ttnn.from_torch(p, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=mesh)

    def timed(fn, iters=30):
        for _ in range(3):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(16):
            ttnn.deallocate(fn())
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.release_trace(mesh, tid)
        return statistics.median(ts) / 16

    for live, nnz, tag in ((8, None, "nnz=None (shipped)"), (8, 8, "nnz=8 exact"), (2, 2, "nnz=2 exact")):
        sp = sparsity_of(live)

        def pair():
            fused = ttnn.sparse_matmul(
                ttnn.reshape(x, (1, 1, 1, H)),
                gate_up_w,
                sparsity=sp,
                nnz=nnz,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=tile,
                program_config=gu_pc,
                compute_kernel_config=cc,
                dtype=ttnn.bfloat16,
            )
            return fused

        gu = timed(pair)
        di = ttnn.reshape(
            ttnn.slice(ttnn.reshape(pair(), (1, E, 2 * I)), [0, 0, 0], [1, E, I]),
            (1, E, 1, I),
        )

        def dn():
            return ttnn.sparse_matmul(
                di,
                down_w,
                sparsity=sp,
                nnz=nnz,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=tile,
                program_config=dn_pc,
                is_input_a_sparse=True,
                is_input_b_sparse=False,
                compute_kernel_config=cc,
                dtype=ttnn.bfloat16,
            )

        d = timed(dn)
        print(f"P|E=32 live={live} {tag:20s} gate_up {gu:7.2f} us  down {d:7.2f} us  pair {gu + d:7.2f} us", flush=True)
        ttnn.deallocate(di)
        ttnn.deallocate(sp)
finally:
    ttnn.close_mesh_device(mesh)
print("P|done")
