# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone check of the fp32 ``ttnn.all_gather`` race the device audio tokenizer works around
(tt/audio/tokenizer.py, ``Ops.gather_last``): on a 1x8 Galaxy row, gathering fp32 column shards and consuming the
result at once gave outputs that changed run to run with identical inputs (code-frame flips 12.3 / 6.7 / 7.6 % in three
runs of one 30 s clip, the rows that changed moving between runs); with a ``ttnn.synchronize_device`` after each
fp32 gather, or with the fp32 tensor gathered as two bf16 tensors, every run was the same bit for bit. The companion of
allreduce_fp32_race.py (the ViT's fp32 all-reduce) for one ttnn issue.

The pattern is the tokenizer's column-parallel projection: per-chip fp32 outputs of a matmul (``a @ w_c``, 128
columns per chip), ``ttnn.all_gather(dim=3, cluster_axis=1)`` over the row (DRAM), then a residual add that consumes
the gathered [rows, 1024] at once. Each configuration (``--chains`` projections back to back per run, nothing read in
between, as in one encoder pass: 48 fp32 gathers) runs ``--runs`` times without a sync between the gather and the
add, and once with one (the reference); the final residual is read back from every chip. Reported per configuration:
runs that differ from the synced reference (bitwise, any chip), runs whose chips disagree with each other, and the
largest difference. A clean all-gather gives 0 / 0 everywhere.

Run on one Galaxy row (TT_VISIBLE_DEVICES = the row's 8 chips):
    python models/experimental/mimo_v2_6_flash/tests/perf/allgather_fp32_race.py [--runs 20] [--chains 1,48] \
        [--rows 1536,3072] [--dtypes float32,bfloat16] [--payload 8704]
"""

import argparse

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf.allreduce_fp32_race import TP, chips, fabric_ring_config, per_chip


def run(mesh, a, w, res, dtype, sync, chain):
    """``chain`` gathers back to back, each added to the running residual; the final residual from every chip."""
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    y = res
    for _ in range(chain):
        shard = ttnn.matmul(a, w, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=cfg)
        out = ttnn.all_gather(shard, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shard.deallocate(True)
        if sync:
            ttnn.synchronize_device(mesh)
        y_new = ttnn.add(y, out, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        if y is not res:
            y.deallocate(True)
        y = y_new
    result = chips(y)
    y.deallocate(True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=20)
    p.add_argument(
        "--rows", default="1536,3072", help="rows of the gathered tensor (the tokenizer's 30 s / 60 s passes)"
    )
    p.add_argument("--k", type=int, default=1024, help="inner dim of the per-chip matmul (the tokenizer's out-proj)")
    p.add_argument("--cols", type=int, default=128, help="columns per chip (gathered: 8x)")
    p.add_argument("--dtypes", default="float32,bfloat16")
    p.add_argument("--chains", default="1,48", help="gathers per run (a tokenizer pass has 48 fp32 gathers)")
    p.add_argument("--payload", type=int, default=8704, help="fabric max packet payload (bytes); 0: default")
    a = p.parse_args()
    ttnn.set_fabric_config(**fabric_ring_config(a.payload))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, TP), l1_small_size=65536, trace_region_size=0)
    lines = []
    try:
        g = torch.Generator().manual_seed(0)
        for rows in (int(r) for r in a.rows.split(",")):
            act = ttnn.from_torch(
                torch.randn(1, 1, rows, a.k, generator=g),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            wts = per_chip(mesh, torch.randn(TP, 1, a.k, a.cols, generator=g) * a.k**-0.5, ttnn.bfloat16)
            res = ttnn.from_torch(
                torch.randn(1, 1, rows, TP * a.cols, generator=g),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for chain in (int(c) for c in a.chains.split(",")):
                for name in a.dtypes.split(","):
                    dtype = getattr(ttnn, name)
                    ref = run(mesh, act, wts, res, dtype, True, chain)
                    ref_chips_agree = bool((ref == ref[0]).all())
                    differ = disagree = 0
                    worst = 0.0
                    for _ in range(a.runs):
                        got = run(mesh, act, wts, res, dtype, False, chain)
                        if not torch.equal(got, ref):
                            differ += 1
                            worst = max(worst, (got - ref).abs().max().item())
                        if not (got == got[0]).all():
                            disagree += 1
                    line = (
                        f"all_gather {name:8s} [1, 1, {rows}, {a.cols}] -> {TP * a.cols} x{chain}: {differ}/{a.runs} "
                        f"unsynced runs differ from the synced one (max |diff| {worst:.4g}), {disagree}/{a.runs} with "
                        f"chips disagreeing; synced chips agree: {ref_chips_agree}"
                    )
                    print(line, flush=True)
                    lines.append(line)
            for t in (act, wts, res):
                t.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)
    print("\n".join(["SUMMARY"] + lines), flush=True)


if __name__ == "__main__":
    main()
