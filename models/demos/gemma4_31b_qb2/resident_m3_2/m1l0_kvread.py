# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""M1L-0 (STEP3_PLAN_v3.1 N3): achieved KV-read bandwidth of one full-attention layer at long context, W chunked
workers on the cores the resident engine leaves free, paged interleaved BF8 KV behind the aliased full view with
random even pages (b -> 2b), per-chunk table reads. Every chip reads its own copy at once (as in the engine).
Run through bench/dflash2/guarded.sh:  python -m models.demos.gemma4_31b_qb2.resident_m3_2.m1l0_kvread --out x.json"""

import argparse
import json
import time

import torch

import ttnn
from models.demos.gemma4_31b_qb2.resident_s2.run_step2 import close_mesh, open_mesh
from models.demos.gemma4_31b_qb2.resident_m3_2 import PKG_DIR
from models.demos.gemma4_31b_qb2.resident_m3_2.common import streamer_cores

BF8 = 1088
DT = 16  # head_dim 512 / 32


def free_cores(mesh, reserve=5):
    """compute cores outside the streamer blocks (row-major), minus `reserve` (the hub and the slices)"""
    _, used = streamer_cores(mesh)
    g = mesh.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x) if (x, y) not in used]
    return cores[reserve:]


def run(mesh, k, v, tab, P, ctx, workers, chunk, split, iters, reps):
    tiles = ctx // 32
    cr = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in workers])
    W = len(workers)

    def cb(idx, pages, page, dt):
        return ttnn.CBDescriptor(total_size=pages * page, core_ranges=cr, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)])

    cbs = [cb(0, chunk * DT, BF8, ttnn.bfloat8_b), cb(1, chunk * DT, BF8, ttnn.bfloat8_b), cb(2, 1, 64, ttnn.uint32),
           cb(3, 1, 64, ttnn.uint32)]
    NCRISC, BRISC = ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0
    NOC0, NOC1 = ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default
    rts = []
    for which in (0, 1) if split else (0,):
        rt = ttnn.RuntimeArgs()
        for w, c in enumerate(workers):
            t0, t1 = w * tiles // W, (w + 1) * tiles // W
            rt[c.x][c.y] = [k.buffer_address(), v.buffer_address(), tab.buffer_address(), t0, t1 - t0, iters, 0, 4 * P]
        rts.append(rt)

    def kern(which, proc, noc):
        return ttnn.KernelDescriptor(
            kernel_source=f"{PKG_DIR}/kernels/m1l0_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cr,
            compile_time_args=[chunk, int(split), which],
            runtime_args=rts[which],
            config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc),
        )

    kernels = [kern(0, BRISC, NOC0), kern(1, NCRISC, NOC1)] if split else [kern(0, NCRISC, NOC1)]
    mesh_pd = ttnn.MeshProgramDescriptor()
    for chip in range(mesh.get_num_devices()):
        coord = ttnn.MeshCoordinate(0, chip)
        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    io = [k, v, tab]
    ttnn.generic_op(io, mesh_pd)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.generic_op(io, mesh_pd)
    ttnn.synchronize_device(mesh)
    per_pass = (time.perf_counter() - t0) / (reps * iters)
    nbytes = 2 * tiles * DT * BF8
    return {"ctx": ctx, "workers": W, "chunk": chunk, "kv_split": split, "ms_per_layer": round(1e3 * per_pass, 4),
            "MB_per_layer": round(nbytes / 1e6, 1), "GBps": round(nbytes / per_pass / 1e9, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctxs", default="32768,131072,262144")
    ap.add_argument("--workers", default="28,56,70")
    ap.add_argument("--chunks", default="4,8")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--reps", type=int, default=4)
    a = ap.parse_args()
    ctxs = [int(c) for c in a.ctxs.split(",")]
    mesh = open_mesh()
    res = {"runs": []}
    try:
        P = max(ctxs) // 128
        n_blocks = P + 16
        zeros = lambda: ttnn.zeros([n_blocks, 4, 128, 256], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                   device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sk, sv = zeros(), zeros()
        k = ttnn.experimental.view(sk, [2 * n_blocks, 1, 128, 512])
        v = ttnn.experimental.view(sv, [2 * n_blocks, 1, 128, 512])
        perm = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(3))[:P]
        tab = ttnn.from_torch((2 * perm).to(torch.int32).reshape(1, P), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cores = free_cores(mesh)
        res["free_cores"] = len(cores)
        for ctx in ctxs:
            for W in [int(w) for w in a.workers.split(",")]:
                for chunk in [int(c) for c in a.chunks.split(",")]:
                    for split in (False, True):
                        row = run(mesh, k, v, tab, P, ctx, cores[:W], chunk, split, a.iters, a.reps)
                        print(json.dumps(row), flush=True)
                        res["runs"].append(row)
                        json.dump(res, open(a.out, "w"), indent=1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
