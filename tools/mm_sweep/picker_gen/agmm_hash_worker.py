#!/usr/bin/env python3
"""Bit-exactness probe for the AGMM op: prints a hash of every device's output replica.

WHY THIS AND NOT THE TEST SUITE: the correctness tests assert PCC >= 0.999, which cannot tell a pure
refactor from one that quietly changed the FP32 accumulation order. Several steps of the wavefront work
(spec appendix B phases 1 and 2) are supposed to be BIT-IDENTICAL, and that claim is only worth making if
it is checked. Phase 3 deliberately is not bit-identical (it re-associates the K-sum), so the gate flips to
PCC there -- which is exactly why the earlier phases need this: it separates "plumbing is wrong" from
"numerics moved on purpose".

Deterministic by construction: fixed seed, one iteration, no program-cache replay.

PCC IS ALSO REPORTED, against a torch reference, because replica agreement alone is not correctness: all
`tp` devices computing the same WRONG answer agrees perfectly. The hash catches "numerics moved", the PCC
catches "numerics are wrong", and a pinned config needs both.

argv: M K N tp topology [config]     topology = ring | line; config = "Pk,Ns,Sm,kb,nsb" or "auto"
"""
import hashlib
import json
import sys

import torch
import ttnn

M, K, N = (int(x) for x in sys.argv[1:4])
TP = int(sys.argv[4])
TOPO = sys.argv[5]
# Pin the parallel config so a measurement taken at one config can be validated at that same config; the
# picker's choice is not always the one being benchmarked (nsb in particular).
CFG = sys.argv[6] if len(sys.argv) > 6 else "auto"


def mesh_geometry(tp):
    n = ttnn.get_num_devices()
    if n >= 32:
        return ((4, 8), 0 if tp == 4 else 1)
    if n >= tp:
        return ((1, tp), 1)
    return None


def main():
    res = {"M": M, "K": K, "N": N, "tp": TP, "topology": TOPO, "cfg": CFG, "outcome": "runtime", "err": ""}
    geom = mesh_geometry(TP)
    if geom is None:
        res["err"] = f"need >= {TP} devices"
        print("HASH_JSON " + json.dumps(res), flush=True)
        return
    (rows, cols), cluster_axis = geom
    fabric = ttnn.FabricConfig.FABRIC_1D_RING if TOPO == "ring" else ttnn.FabricConfig.FABRIC_1D
    topology = ttnn.Topology.Ring if TOPO == "ring" else ttnn.Topology.Linear
    ttnn.set_fabric_config(
        fabric,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    cfg = None
    if CFG != "auto":
        pk, ns, sm, kbt, nsb = (int(x) for x in CFG.split(","))
        cfg = ttnn.RegimeAMatmulConfig(k_slices=pk, n_slices=ns, m_slices=sm, k_block_tiles=kbt, n_subblock_tiles=nsb)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
    try:
        sub = [1, 1]
        sub[cluster_axis] = TP
        mesh = parent.create_submesh(ttnn.MeshShape(tuple(sub)))

        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=torch.bfloat16)
        b = torch.randn(K, N, dtype=torch.bfloat16)

        dims = [None, None]
        dims[cluster_axis] = 1
        in0 = ttnn.from_torch(
            a,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )
        wc = ttnn.create_regime_a_weight_memory_config(list(b.shape), ttnn.bfloat16, mesh)
        in1 = ttnn.from_torch(
            b,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=mesh,
            memory_config=wc,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        grid = mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        ttnn.synchronize_device(mesh)
        sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
        buf = ttnn.from_torch(
            torch.zeros((M, K), dtype=torch.float32),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh,
        )
        ttnn.synchronize_device(mesh)

        out = ttnn.experimental.all_gather_regime_a_matmul_async(
            in0,
            in1,
            persistent_output_buffer=buf,
            multi_device_global_semaphore=sems,
            barrier_semaphore=None,
            num_links=2,
            topology=topology,
            cluster_axis=cluster_axis,
            config=cfg,
        )
        ttnn.synchronize_device(mesh)
        stacked = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(0, 1))
        )
        while stacked.dim() > 2:
            stacked = stacked.squeeze(0)
        # Hash the raw bf16 bytes per device replica: any change in accumulation order shows up here, where
        # PCC would not.
        res["hashes"] = [
            hashlib.sha256(r.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
            for r in torch.chunk(stacked, TP, dim=cluster_axis)
        ]
        # PCC per replica against the full-K reference. The gather makes all of K available on every device,
        # so each replica must be the whole a@b, not a shard of it.
        ref = (a.float() @ b.float()).flatten()
        res["pcc"] = []
        for r in torch.chunk(stacked, TP, dim=cluster_axis):
            v = r.float().flatten()
            res["pcc"].append(
                round(float(torch.corrcoef(torch.stack([v, ref]))[0, 1]), 6) if v.numel() == ref.numel() else -1.0
            )
        res["pcc_min"] = min(res["pcc"])
        res["outcome"] = "ok"
    except Exception as e:  # noqa: BLE001
        res["err"] = str(e)[:300]
    finally:
        for s in parent.get_submeshes():
            ttnn.close_mesh_device(s)
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print("HASH_JSON " + json.dumps(res), flush=True)


main()
