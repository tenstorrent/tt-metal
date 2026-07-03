# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Placement candidate sweep for the net-new entropy-accept chain over the 256 axis.

sort -> cumsum -> subtract -> le -> scatter operate on a tiny [B, 256] tensor, so the
question is DRAM-vs-L1 placement and preallocated constants, not big-matmul geometry.
This measures the eager device time of the chain under a few placements to justify the
chosen config with before/after numbers.
"""
from __future__ import annotations

import time
import torch
import ttnn
from models.experimental.diffusion_gemma.tt import denoise_loop as DL

CANVAS = 256


def log(m):
    print(m, flush=True)


def repl(mesh):
    return ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None


def _dealloc(o):
    outs = o if isinstance(o, (tuple, list)) else [o]
    for t in outs:
        if hasattr(t, "deallocate"):
            t.deallocate(True)


def timed(name, fn, mesh, reps=20):
    _dealloc(fn())
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        o = fn()
        ttnn.synchronize_device(mesh)
        _dealloc(o)
    dt = (time.perf_counter() - t0) * 1e3 / reps
    log(f"OP {name}: {dt:.4f} ms")
    return dt


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        torch.manual_seed(0)
        ent_t = torch.rand(1, CANVAS).float() * 0.02  # entropy-scale values
        ent_dram = ttnn.from_torch(
            ent_t,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=repl(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ent_l1 = ttnn.to_memory_config(ent_dram, ttnn.L1_MEMORY_CONFIG)

        consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=CANVAS, budget=0.1)

        # A) baseline: DRAM entropy, per-call ttnn.full/zeros_like (original eager)
        timed("accept_DRAM_percall_consts", lambda: DL.entropy_budget_accept(ent_dram, 0.1), mesh)
        # B) DRAM entropy, preallocated (persistent, trace-safe) consts
        timed(
            "accept_DRAM_prealloc_consts",
            lambda: DL.entropy_budget_accept(ent_dram, 0.1, budget_t=consts.budget_t, zeros=consts.accept_zeros),
            mesh,
        )
        # C) L1 entropy, preallocated consts
        timed(
            "accept_L1_prealloc_consts",
            lambda: DL.entropy_budget_accept(ent_l1, 0.1, budget_t=consts.budget_t, zeros=consts.accept_zeros),
            mesh,
        )

        # component ops
        timed("sort_256", lambda: ttnn.sort(ent_dram, dim=-1), mesh)
        sv, si = ttnn.sort(ent_dram, dim=-1)
        timed("cumsum_256", lambda: ttnn.cumsum(sv, dim=-1), mesh)
        sv.deallocate(True)
        si.deallocate(True)
        log("DIAG_DONE")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
