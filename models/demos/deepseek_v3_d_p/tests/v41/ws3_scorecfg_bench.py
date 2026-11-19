# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""WS3: ``indexer_score_dsa`` program configs on the INDEX_QTP shape (128 query rows a chip, 32 heads x 128, keys = the
capacity bucket) -- device time (slowest chip's summed kernel time) and per-chip bit equality of the scores against the
production config (q_chunk 64, k_chunk 64). Kernelize runs q32 / k256 (their bead F10).

    TT_METAL_DEVICE_PROFILER=1 ... python ws3_scorecfg_bench.py [buckets] [out.json]
"""
import json
import sys
from collections import defaultdict

import torch

import ttnn

SP, TP, CHUNK, H, DH = 8, 4, 4096, 32, 128
CAP = 528512
CONFIGS = [(64, 64), (32, 256), (64, 256), (64, 128), (32, 128), (128, 256), (32, 512)]


def main():
    buckets = [int(b) for b in (sys.argv[1] if len(sys.argv) > 1 else "65536,294912,524288").split(",")]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(SP, TP))
    rep = ttnn.ReplicateTensorToMesh(mesh)
    flat = ttnn.ShardTensorToMesh(mesh, dim=2)  # 32 row shards, device rank order = (sp, tp) row-major = QTP's quarters
    torch.manual_seed(2)
    q = ttnn.from_torch(
        torch.randn(1, H, CHUNK, DH), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=flat
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, CHUNK, H) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=flat
    )
    keys_full = ttnn.from_torch(
        torch.randn(1, 1, CAP, DH), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
    )
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

    for b in buckets:
        keys = ttnn.slice(keys_full, [0, 0, 0, 0], [1, 1, b, DH])
        ref = None
        for qc, kc in CONFIGS:
            pc = ttnn.IndexerScoreProgramConfig(q_chunk_size=qc, k_chunk_size=kc, head_group_size=0)
            fn = lambda: ttnn.experimental.indexer_score_dsa(
                q, keys, w, chunk_start_idx=b - CHUNK, kv_len=b, program_config=pc, seq_shard_axes=[]
            )
            try:
                r, _ = timed(fn)  # compile
                ttnn.deallocate(r)
                r, ms = timed(fn)
            except Exception as e:  # noqa: BLE001
                row = dict(bucket=b, q_chunk=qc, k_chunk=kc, error=repr(e)[:200])
                print(json.dumps(row), flush=True)
                out.append(row)
                continue
            chips = [ttnn.to_torch(x).view(torch.int16) for x in ttnn.get_device_tensors(r)]
            if ref is None:
                ref, same = chips, "reference"
            else:
                same = f"{sum(torch.equal(a, c) for a, c in zip(ref, chips))}/{len(chips)} chips bit-identical"
            row = dict(bucket=b, q_chunk=qc, k_chunk=kc, dev_ms=ms, vs_64_64=same)
            print(json.dumps(row), flush=True)
            out.append(row)
            ttnn.deallocate(r)
        ttnn.deallocate(keys)
    with open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ws3_scorecfg.json", "w") as f:
        json.dump(out, f, indent=1)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
