# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone check of the fp32 ``ttnn.all_reduce`` race the device ViT works around (tt/vision/vit.py,
VisionNumerics): on a 1x8 Galaxy row an fp32 all-reduce seemed to hand its output to the next op before every chip's
share had landed (random results run to run at [1, 1, 1024, 1280] and [1, 1, 1280, 1280], exact with a
``ttnn.synchronize_device`` after it); bf16 never showed it.

The pattern is the ViT's row-parallel projection: per-chip partial sums from a matmul (``partial = a_c @ w_c``, fp32
out), ``ttnn.all_reduce`` over the row (ring, 2 links, DRAM), then a residual add that consumes the result at once.
Each configuration (``--chains`` projections back to back per run, nothing read in between, as in a ViT pass) runs
``--runs`` times without a sync between the all-reduce and the add, and once more with one (the reference); the final
residual is read back from every chip. Reported per configuration: runs that differ from the synced reference
(bitwise, any chip), runs whose chips disagree with each other, and the largest difference. A clean all-reduce gives
0 / 0 everywhere.

Run on one Galaxy row (TT_VISIBLE_DEVICES = the row's 8 chips):
    python models/experimental/mimo_v2_6_flash/tests/perf/allreduce_fp32_race.py [--runs 20] [--chains 1,57] \
        [--shapes 1024x1280,1280x1280,4096x1280] [--dtypes float32,bfloat16] [--payload 8704]
"""

import argparse

import torch

import ttnn

TP = 8


def fabric_ring_config(payload):
    """FABRIC_1D_RING; ``payload``: the router's max packet payload in bytes (MiMo serves with 8704; 0: default)."""
    kwargs = {"config": ttnn.FabricConfig.FABRIC_1D_RING}
    if payload:
        router = ttnn.FabricRouterConfig()
        router.max_packet_payload_size_bytes = payload
        kwargs["router_config"] = router
    return kwargs


def per_chip(mesh, t, dtype):
    """[TP, ...] -> chip c holds t[c] (shape [1, ...])."""
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 0), mesh_shape=(1, TP)),
    )


def chips(t):
    return torch.stack([ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(t)])


def run(mesh, a, w, res, dtype, sync, chain):
    """``chain`` projections back to back (as a ViT pass's blocks: nothing read in between), each added to the
    running residual; the final residual read from every chip."""
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    y = res
    for _ in range(chain):
        partial = ttnn.matmul(a, w, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=cfg)
        out = ttnn.all_reduce(
            partial, cluster_axis=1, num_links=2, topology=ttnn.Topology.Ring, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        partial.deallocate(True)
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
    p.add_argument("--shapes", default="1024x1280,1280x1280,4096x1280", help="rows x cols of the all-reduced tensor")
    p.add_argument("--k", type=int, default=256, help="inner dim of the per-chip matmul (the ViT's o-proj: 256)")
    p.add_argument("--dtypes", default="float32,bfloat16")
    p.add_argument("--chains", default="1,57", help="projections per run (a ViT pass has 57 reduces)")
    p.add_argument("--payload", type=int, default=8704, help="fabric max packet payload (bytes); 0: default")
    a = p.parse_args()
    ttnn.set_fabric_config(**fabric_ring_config(a.payload))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, TP), l1_small_size=65536, trace_region_size=0)
    lines = []
    try:
        g = torch.Generator().manual_seed(0)
        for shape in a.shapes.split(","):
            rows, cols = (int(v) for v in shape.split("x"))
            act = per_chip(mesh, torch.randn(TP, 1, rows, a.k, generator=g), ttnn.bfloat16)
            wts = per_chip(mesh, torch.randn(TP, 1, a.k, cols, generator=g) * a.k**-0.5, ttnn.bfloat16)
            res = ttnn.from_torch(
                torch.randn(1, 1, rows, cols, generator=g),
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
                        f"all_reduce {name:8s} [1, 1, {rows}, {cols}] x{chain}: {differ}/{a.runs} unsynced runs differ "
                        f"from the synced one (max |diff| {worst:.4g}), {disagree}/{a.runs} with chips disagreeing; "
                        f"synced chips agree: {ref_chips_agree}"
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
