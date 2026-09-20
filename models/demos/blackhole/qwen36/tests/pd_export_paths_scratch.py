# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Export-path check without a model: random bfp8 paged caches on a 1x4 mesh, export the same block lists via
runs / blocks / gather and compare bitwise with a host reference; time each path. No weights needed.

    TT_VISIBLE_DEVICES=0,1,6,7 python models/demos/blackhole/qwen36/tests/pd_export_paths_scratch.py
"""
import os
import time
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import pd_transfer

NB, NKV, BLK, HD, NL = int(os.environ.get("NB", "512")), 1, 64, 256, 16


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576)
    mesh.enable_program_cache()
    torch.manual_seed(0)
    layers, host_ref = [], []
    for _ in range(NL):
        pair = []
        for _ in range(2):
            src = torch.randn(NB, 4, BLK, HD, dtype=torch.bfloat16)  # dim 1 = device
            t = ttnn.from_torch(
                src,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
            )
            # host reference = what the device holds (bfp8-rounded), as [NB, n_dev*nkv, blk, hd]
            back = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).to(torch.bfloat16)
            pair.append((t, back))
        layers.append(
            SimpleNamespace(is_full_attention=True, attention=SimpleNamespace(paged_k=pair[0][0], paged_v=pair[1][0]))
        )
        host_ref.append((pair[0][1], pair[1][1]))
    model = SimpleNamespace(layers=layers, mesh_device=mesh, num_devices=4)
    logger.info("caches ready")
    t0 = time.perf_counter()
    pd_transfer.export_warmup(model, max_bucket=int(os.environ.get("WARM_MAX", "64")))
    logger.info(f"warm-up {time.perf_counter() - t0:.1f} s")
    cases = {
        "contig3": [10, 11, 12],
        "desc3": [12, 11, 10],
        "frag3": [38, 57, 50],
        "top3": [NB - 3, NB - 2, NB - 1],
        "contig16": list(range(100, 116)),
        "frag20": [int(x) for x in torch.randperm(NB)[:20]],
        "contig64": list(range(200, 264)),
        "frag64": [int(x) for x in torch.randperm(NB)[:64]],
        "one": [7],
    }
    bad = 0
    for name, ids in cases.items():
        res = {}
        for mode in ("auto", "blocks"):
            os.environ["QWEN36_PD_EXPORT"] = mode
            for _ in range(2):  # second call = warm timing
                t0 = time.perf_counter()
                out = pd_transfer.export_kv_blocks(model, ids)
                dt = time.perf_counter() - t0
            ok = all(torch.equal(k, hk[ids]) and torch.equal(v, hv[ids]) for (k, v), (hk, hv) in zip(out, host_ref))
            res[mode] = (ok, dt)
            bad += not ok
        logger.info(
            f"{name:9s} n={len(ids):3d} runs={len(pd_transfer.coalesce_runs(ids))} "
            + "  ".join(f"{m}: {'OK ' if ok else 'BAD'} {1e3 * dt:7.1f} ms" for m, (ok, dt) in res.items())
        )
    os.environ.pop("QWEN36_PD_EXPORT", None)
    # import round trip with bucket padding: 3 real blocks -> bucket 4, padding rows land in the pad block only
    model.num_devices = 4
    model._pad_kv_block = NB - 1
    before = [
        (
            ttnn.to_torch(l.attention.paged_k, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).to(torch.bfloat16),
            ttnn.to_torch(l.attention.paged_v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).to(torch.bfloat16),
        )
        for l in layers
    ]
    src, dst = [10, 11, 12], [200, 57, 3]
    kv = pd_transfer.export_kv_blocks(model, src)
    pd_transfer.import_warmup(model, max_bucket=8)
    t0 = time.perf_counter()
    pd_transfer.import_kv_blocks(model, dst, kv)
    dt = time.perf_counter() - t0
    ok_imp = True
    for li, l in enumerate(layers):
        for cache, (bk, bv), ref_pair in ((l.attention.paged_k, kv[li], before[li]),):
            pass
        for j, cache in enumerate((l.attention.paged_k, l.attention.paged_v)):
            after = ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).to(torch.bfloat16)
            ref = before[li][j].clone()
            ref[dst] = kv[li][j]  # destination blocks take the exported rows
            mask = torch.ones(NB, dtype=torch.bool)
            mask[NB - 1] = False  # pad block may hold anything
            ok_imp &= torch.equal(after[mask], ref[mask])
    logger.info(
        f"import 3 blocks -> bucket 4 into {dst}: {'OK' if ok_imp else 'BAD'} ({1e3 * dt:.1f} ms); only dst blocks (+pad) changed"
    )
    bad += not ok_imp
    ttnn.close_mesh_device(mesh)
    logger.info("ALL OK" if bad == 0 else f"{bad} MISMATCHES")


if __name__ == "__main__":
    main()
