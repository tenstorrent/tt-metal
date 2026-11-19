# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""WS3: the EAGER fused scorer (``TtCSAIndexer.select_indices_fused``) as the last-chunk decoder's ratio-1 index sources run it
(layer 20 and the Reindex layers 24 / 28 / 32 / 36: 512 query rows a chip, the whole index-key cache, 1,052,800 rows at MAXSEQ 1M,
as the op's key operand). Times the full-cache call against the same call on the cache sliced to a 32,768-column ladder above the
scored length ``kv_len`` (lever FUSED_BOUND), and checks per chip that the scored columns, the masked row and the top-k are
bit-identical. Then the rest of select_indices_fused (the chunk's cut block, top-k at valid_length = E).

    TT_METAL_DEVICE_PROFILER=1 ... python ws3_fused_bench.py [entry counts before the chunk] [out.json]
"""
import json
import sys
from collections import defaultdict

import torch

import ttnn

SP, TP, CHUNK, RATE, H, DH, TOPK = 8, 4, 4096, 1, 32, 128, 512
S_L = CHUNK // SP
CAP = 1052800
LADDER = 32768
PC = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=64, head_group_size=0)


def main():
    starts = [int(p) for p in (sys.argv[1] if len(sys.argv) > 1 else "126976,520192,1044480").split(",")]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(SP, TP))
    rep = ttnn.ReplicateTensorToMesh(mesh)
    seq = ttnn.ShardTensor2dMesh(mesh, dims=(2, None), mesh_shape=(SP, TP))
    torch.manual_seed(3)
    q = ttnn.from_torch(
        torch.randn(1, H, CHUNK, DH), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=seq
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, CHUNK, H) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=seq
    )
    keys_full = ttnn.from_torch(
        torch.randn(1, 1, CAP, DH), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
    )
    rows = torch.arange(CHUNK).view(CHUNK, 1)
    cols = torch.arange(CHUNK // RATE).view(1, -1)
    cut = torch.where(RATE * cols + (RATE - 1) <= rows, 0.0, float("-inf")).view(1, 1, CHUNK, -1)
    cut = ttnn.from_torch(cut, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=seq)
    out = []

    def timed(fn):
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        ttnn.get_latest_programs_perf_data()
        r = fn()
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        per = defaultdict(float)
        for chip, plist in ttnn.get_latest_programs_perf_data().items():
            for pr in plist:
                k = pr.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
                if k is not None:
                    per[chip] += k.duration / 1e6
        return r, round(max(per.values()), 3)

    for e0 in starts:
        n_new = CHUNK // RATE
        E = e0 + n_new
        E_tiles = -(-E // 32) * 32
        kv_len = -(-(E_tiles + SP * S_L) // 64) * 64
        W = min(CAP, -(-kv_len // LADDER) * LADDER)

        def path(keys):
            lg = ttnn.experimental.indexer_score_dsa(
                q, keys, w, chunk_start_idx=E_tiles, kv_len=kv_len, program_config=PC, seq_shard_axes=[0]
            )
            seg = ttnn.add(ttnn.slice(lg, [0, 0, 0, e0], [1, 1, S_L, E]), cut)
            ttnn.experimental.slice_write(seg, lg, start=[0, 0, 0, e0], end=[1, 1, S_L, E], step=[1, 1, 1, 1])
            return lg, ttnn.experimental.topk_large_indices(lg, k=TOPK, valid_length=E)

        for rep_i in range(2):
            (lf, i_f), ms_full = timed(lambda: path(keys_full))
            kb = ttnn.slice(keys_full, [0, 0, 0, 0], [1, 1, W, DH])
            (lb, i_b), ms_b = timed(lambda: path(kb))
            _, ms_slice = timed(lambda: ttnn.slice(keys_full, [0, 0, 0, 0], [1, 1, W, DH]))
            if rep_i == 1:
                fs = [ttnn.to_torch(x)[..., :E].contiguous().view(torch.int16) for x in ttnn.get_device_tensors(lf)]
                bs = [ttnn.to_torch(x)[..., :E].contiguous().view(torch.int16) for x in ttnn.get_device_tensors(lb)]
                fi = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(i_f)]
                bi = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(i_b)]
                row = dict(
                    entries_before=e0,
                    kv_len=kv_len,
                    width=W,
                    full_ms=ms_full,
                    bounded_ms=ms_b,
                    key_slice_ms=ms_slice,
                    scores=f"{sum(torch.equal(a, b) for a, b in zip(fs, bs))}/32 bit-identical",
                    topk=f"{sum(torch.equal(a, b) for a, b in zip(fi, bi))}/32 bit-identical",
                )
                print(json.dumps(row), flush=True)
                out.append(row)
            for t in (lf, i_f, lb, i_b, kb):
                ttnn.deallocate(t)
    with open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ws3_fused_bench.json", "w") as f:
        json.dump(out, f, indent=1)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
