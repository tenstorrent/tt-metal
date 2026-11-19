# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""WS3 lever INDEX_QTP: the static scorer's index path on each chip's TP quarter of the query rows vs the full-row path,
at the production geometry (see ws3_scorer_bench.py). Checks, per chip, that the partitioned path's scores and top-k are
BIT-identical to the TP quarter of the full-row path's, and times both (slowest chip's summed device kernel time; mask
partitioned after the bucket slice, or before it).

    TT_METAL_DEVICE_PROFILER=1 ... python ws3_qtp_bench.py [positions] [out.json]
"""
import json
import sys
import time
from collections import defaultdict

import torch

import ttnn

SP, TP, CHUNK, RATE, H, DH, TOPK = 8, 4, 4096, 2, 32, 128, 512
S_L = CHUNK // SP
CAP = 528512
STEP = 32768
PC = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=64, head_group_size=0)


def bucket(entries_before: int) -> int:
    need = entries_before + CHUNK // RATE + CHUNK
    full = (CAP // 64) * 64
    for b in sorted({16384} | set(range(STEP, full, STEP))):
        if b >= need:
            return b
    return full


def main():
    positions = [int(p) for p in (sys.argv[1] if len(sys.argv) > 1 else "65536,524288,1036288").split(",")]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(SP, TP))
    rep = ttnn.ReplicateTensorToMesh(mesh)
    seq = ttnn.ShardTensor2dMesh(mesh, dims=(2, None), mesh_shape=(SP, TP))

    def dev(t, mapper, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=mapper)

    torch.manual_seed(1)
    q = dev(torch.randn(1, H, CHUNK, DH), seq)
    w = dev(torch.randn(1, 1, CHUNK, H) * 0.1, seq)
    keys_full = dev(torch.randn(1, 1, CAP, DH), rep)
    out = []

    def run(name, fn, pos, width, rec=True):
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        ttnn.get_latest_programs_perf_data()
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(mesh)
        wall = time.perf_counter() - t0
        ttnn.ReadDeviceProfiler(mesh)
        per_chip = defaultdict(float)
        for chip, plist in ttnn.get_latest_programs_perf_data().items():
            for pr in plist:
                k = pr.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
                if k is not None:
                    per_chip[chip] += k.duration / 1e6
        row = dict(
            pos=pos,
            width=width,
            op=name,
            dev_ms=round(max(per_chip.values()), 3) if per_chip else None,
            wall_ms=round(wall * 1e3, 2),
        )
        if rec:
            print(json.dumps(row), flush=True)
            out.append(row)
        return r

    def chips(t):
        return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]

    for pos in positions:
        E = pos // RATE
        b = bucket(E)
        m = torch.full((1, 1, CHUNK, CAP), float("-inf"))
        m[..., :E] = 0.0
        rows = torch.arange(CHUNK).view(CHUNK, 1)
        cols = torch.arange(CHUNK // RATE).view(1, -1)
        m[0, 0, :, E : E + CHUNK // RATE] = torch.where(RATE * cols + RATE - 1 <= rows, 0.0, float("-inf"))
        mask_full = dev(m, seq)
        del m
        for rep_i in range(2):
            rec = rep_i == 1

            def full_path():
                keys = ttnn.slice(keys_full, [0, 0, 0, 0], [1, 1, b, DH])
                lg = ttnn.experimental.indexer_score_dsa(
                    q, keys, w, chunk_start_idx=b - SP * S_L, kv_len=b, program_config=PC, seq_shard_axes=[0]
                )
                mk = ttnn.to_layout(ttnn.slice(mask_full, [0, 0, 0, 0], [1, 1, S_L, b]), ttnn.ROW_MAJOR_LAYOUT)
                s = ttnn.add(lg, mk)
                return lg, s, ttnn.experimental.topk_large_indices(s, k=TOPK, valid_length=b)

            def qtp_path(partition_first: bool):
                keys = ttnn.slice(keys_full, [0, 0, 0, 0], [1, 1, b, DH])
                qp = ttnn.mesh_partition(q, dim=2, cluster_axis=1)
                wp = ttnn.mesh_partition(w, dim=2, cluster_axis=1)
                lg = ttnn.experimental.indexer_score_dsa(
                    qp, keys, wp, chunk_start_idx=b - SP * S_L, kv_len=b, program_config=PC, seq_shard_axes=[]
                )
                if partition_first:
                    mk = ttnn.slice(
                        ttnn.mesh_partition(mask_full, dim=2, cluster_axis=1), [0, 0, 0, 0], [1, 1, S_L // TP, b]
                    )
                else:
                    mk = ttnn.mesh_partition(ttnn.slice(mask_full, [0, 0, 0, 0], [1, 1, S_L, b]), dim=2, cluster_axis=1)
                s = ttnn.add(lg, ttnn.to_layout(mk, ttnn.ROW_MAJOR_LAYOUT))
                return lg, s, ttnn.experimental.topk_large_indices(s, k=TOPK, valid_length=b)

            lg_f, s_f, i_f = run("full_path", full_path, pos, b, rec)
            lg_q, s_q, i_q = run("qtp_slice_then_partition", lambda: qtp_path(False), pos, b, rec)
            _, _, i_q2 = run("qtp_partition_then_slice", lambda: qtp_path(True), pos, b, rec)
            if rec:
                # bitwise: each chip's partitioned results vs its TP quarter of the full-row results
                pf = [ttnn.mesh_partition(t, dim=2, cluster_axis=1) for t in (lg_f, s_f, i_f)]
                cmp = {}
                for name, a, bq in (
                    ("scores", pf[0], lg_q),
                    ("masked", pf[1], s_q),
                    ("topk", pf[2], i_q),
                    ("topk_pf", pf[2], i_q2),
                ):
                    ca, cb = chips(a), chips(bq)
                    eq = [
                        torch.equal(
                            x.view(torch.int16) if x.dtype == torch.bfloat16 else x,
                            y.view(torch.int16) if y.dtype == torch.bfloat16 else y,
                        )
                        for x, y in zip(ca, cb)
                    ]
                    cmp[name] = f"{sum(eq)}/{len(eq)} chips bit-identical"
                # how many top-k rows have score ties at the selection boundary (an ordering risk the digests must cover)
                row = dict(pos=pos, width=b, op="bitcheck", **cmp)
                print(json.dumps(row), flush=True)
                out.append(row)
                for t in pf:
                    ttnn.deallocate(t)
            for t in (lg_f, s_f, i_f, lg_q, s_q, i_q, i_q2):
                ttnn.deallocate(t)
        ttnn.deallocate(mask_full)
    with open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ws3_qtp_bench.json", "w") as f:
        json.dump(out, f, indent=1)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
