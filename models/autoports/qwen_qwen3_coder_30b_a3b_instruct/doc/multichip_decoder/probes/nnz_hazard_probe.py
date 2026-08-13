# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the sparse_matmul nnz-mismatch hazard, on ONE die, under the watcher.

Why this exists: stage 03 decided decode must pass ``nnz=None`` because under
expert parallelism the locally-live expert count is data-dependent. That
decision rests on the claim that a wrong ``nnz`` *hangs the board*, which the
design phase took from a source comment (``sparse_matmul_device_operation.cpp``
205-211, tt-metal #45943) rather than from a measurement. This measures it.

Run with the watcher on, which turns the hang into a loud on-device assert:

    TT_METAL_WATCHER=10 python nnz_hazard_probe.py

One die, no fabric, no CCL -- deliberately, so that if it does hang, exactly one
board needs resetting and the watcher's active-eth kernel-config overflow (see
work_log.md) cannot get in the way.

Prints ``P|`` lines only.
"""
import sys

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.functional_decoder import _sparse_matmul_config

E, H, N, TOPK = 32, 2048, 2048, 8

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
try:
    torch.manual_seed(0)
    weight = ttnn.from_torch(
        torch.randn(1, E, H, N) * 0.02,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x = ttnn.from_torch(
        torch.randn(1, 1, 1, H) * 0.02,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pc = _sparse_matmul_config(1, N, in0_block_w=16)
    tile = ttnn.Tile([32, 32])

    def run(live: int, nnz):
        pattern = torch.zeros(1, 1, 1, E, dtype=torch.bfloat16)
        pattern[..., :live] = 1.0
        sparsity = ttnn.from_torch(pattern, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=mesh)
        out = ttnn.sparse_matmul(
            x,
            weight,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=tile,
            program_config=pc,
            dtype=ttnn.bfloat16,
        )
        ttnn.synchronize_device(mesh)
        t = ttnn.to_torch(out)
        ttnn.deallocate(out)
        ttnn.deallocate(sparsity)
        return t

    # 1. exact nnz -- the shipped single-chip contract. Must succeed.
    t = run(TOPK, TOPK)
    print(f"P|exact nnz=8 with 8 live: ok, out {tuple(t.shape)} finite={bool(torch.isfinite(t).all())}", flush=True)

    # 2. nnz=None with the same pattern -- the shipped multichip contract.
    t = run(TOPK, None)
    print(f"P|nnz=None with 8 live: ok, finite={bool(torch.isfinite(t).all())}", flush=True)

    # 3. nnz=None with an EMPTY window -- the EP case that has no legal exact nnz.
    t = run(0, None)
    print(f"P|nnz=None with 0 live: ok, max|out| = {t.abs().max().item()}", flush=True)

    # 4. THE HAZARD: 2 live entries, nnz still 8. This is what a host-computed
    #    nnz does on a die whose window holds only 2 of the global top-8.
    print("P|about to run nnz=8 against 2 live entries -- expect a watcher assert or a hang", flush=True)
    t = run(2, TOPK)
    print(f"P|MISMATCH nnz=8 vs 2 live: returned without error, finite={bool(torch.isfinite(t).all())}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
print("P|done")
