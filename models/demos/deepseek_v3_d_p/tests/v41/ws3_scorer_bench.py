# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""WS3 (11-19 prefill TTFT push, lever 1.6): device time of each op of the traced CSA source's index path
(``TtCSAIndexer.select_indices_static``: key slice, ``indexer_score_dsa``, mask slice, mask untilize, row-major add,
``topk_large_indices``) at the production geometry (SP8 x TP4, chunk 4096 -> 512 query rows a chip, 32 index heads x 128,
index-key cache 528,512 rows at MAXSEQ 1M, rate 2) for chunks at several positions, plus the alternatives a lever would use.
Random data: timing only. Per op: the slowest chip's summed DEVICE KERNEL time (device profiler, read after every op).

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
    TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 python ws3_scorer_bench.py [positions, comma list]
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


def bucket(entries_before: int, step: int = STEP) -> int:
    need = entries_before + CHUNK // RATE + CHUNK
    full = (CAP // 64) * 64
    cands = sorted({16384} | set(range(step, full, step)))
    for b in cands:
        if b >= need:
            return b
    return full


def main():
    positions = [int(p) for p in (sys.argv[1] if len(sys.argv) > 1 else "65536,98304,524288,1036288").split(",")]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(SP, TP))
    rep = ttnn.ReplicateTensorToMesh(mesh)
    seq = ttnn.ShardTensor2dMesh(mesh, dims=(2, None), mesh_shape=(SP, TP))

    def dev(t, mapper, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=mapper)

    torch.manual_seed(0)
    q = dev(torch.randn(1, H, CHUNK, DH), seq)
    w = dev(torch.randn(1, 1, CHUNK, H) * 0.1, seq)
    keys_full = dev(torch.randn(1, 1, CAP, DH), rep)
    out = []

    def drain():
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        ttnn.get_latest_programs_perf_data()

    def timed(name, fn, pos, width):
        drain()
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(mesh)
        wall = time.perf_counter() - t0
        ttnn.ReadDeviceProfiler(mesh)
        data = ttnn.get_latest_programs_perf_data()
        per_chip = defaultdict(float)
        nprog = 0
        for chip, plist in data.items():
            for pr in plist:
                k = pr.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
                if k is not None:
                    per_chip[chip] += k.duration / 1e6
                    nprog += 1
        dev_ms = max(per_chip.values()) if per_chip else None
        row = dict(
            pos=pos,
            width=width,
            op=name,
            dev_ms=round(dev_ms, 3) if dev_ms else None,
            wall_ms=round(wall * 1e3, 2),
            progs=nprog,
        )
        print(json.dumps(row), flush=True)
        out.append(row)
        return r

    for pos in positions:
        E = pos // RATE
        b = bucket(E)
        live = E + CHUNK // RATE
        m = torch.full((1, 1, CHUNK, CAP), float("-inf"))
        m[..., :E] = 0.0
        rows = torch.arange(CHUNK).view(CHUNK, 1)
        cols = torch.arange(CHUNK // RATE).view(1, -1)
        m[0, 0, :, E : E + CHUNK // RATE] = torch.where(RATE * cols + RATE - 1 <= rows, 0.0, float("-inf"))
        mask_full = dev(m, seq)  # the slot's persistent TILE score mask
        del m
        for rep_i in range(2):  # the first pass compiles
            tag = "" if rep_i else "warm:"
            keys = timed(tag + "slice_keys", lambda: ttnn.slice(keys_full, [0, 0, 0, 0], [1, 1, b, DH]), pos, b)
            logits = timed(
                tag + "indexer_score_dsa",
                lambda: ttnn.experimental.indexer_score_dsa(
                    q,
                    keys,
                    w,
                    chunk_start_idx=b - SP * S_L,
                    kv_len=b,
                    program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=64, head_group_size=0),
                    seq_shard_axes=[0],
                ),
                pos,
                b,
            )
            mask = timed(tag + "slice_mask", lambda: ttnn.slice(mask_full, [0, 0, 0, 0], [1, 1, S_L, b]), pos, b)
            mask_rm = timed(tag + "untilize_mask", lambda: ttnn.to_layout(mask, ttnn.ROW_MAJOR_LAYOUT), pos, b)
            summed = timed(tag + "add_rm", lambda: ttnn.add(logits, mask_rm), pos, b)
            idx = timed(
                tag + "topk_large", lambda: ttnn.experimental.topk_large_indices(summed, k=TOPK, valid_length=b), pos, b
            )
            # alternatives: top-k bounded at the live width; the scorer at a finer bucket (step 8192 / 16384)
            idx2 = timed(
                tag + "topk_large@live",
                lambda: ttnn.experimental.topk_large_indices(summed, k=TOPK, valid_length=-(-live // 64) * 64),
                pos,
                live,
            )
            for t in (idx, idx2):
                ttnn.deallocate(t)
            for st in (16384, 8192):
                b2 = bucket(E, st)
                if b2 >= b:
                    continue
                k2 = timed(
                    tag + f"slice_keys@step{st}", lambda: ttnn.slice(keys_full, [0, 0, 0, 0], [1, 1, b2, DH]), pos, b2
                )
                l2 = timed(
                    tag + f"indexer_score_dsa@step{st}",
                    lambda: ttnn.experimental.indexer_score_dsa(
                        q,
                        k2,
                        w,
                        chunk_start_idx=b2 - SP * S_L,
                        kv_len=b2,
                        program_config=ttnn.IndexerScoreProgramConfig(
                            q_chunk_size=64, k_chunk_size=64, head_group_size=0
                        ),
                        seq_shard_axes=[0],
                    ),
                    pos,
                    b2,
                )
                m2 = timed(
                    tag + f"slice_mask@step{st}", lambda: ttnn.slice(mask_full, [0, 0, 0, 0], [1, 1, S_L, b2]), pos, b2
                )
                m2r = timed(tag + f"untilize_mask@step{st}", lambda: ttnn.to_layout(m2, ttnn.ROW_MAJOR_LAYOUT), pos, b2)
                s2 = timed(tag + f"add_rm@step{st}", lambda: ttnn.add(l2, m2r), pos, b2)
                i2 = timed(
                    tag + f"topk_large@step{st}",
                    lambda: ttnn.experimental.topk_large_indices(s2, k=TOPK, valid_length=b2),
                    pos,
                    b2,
                )
                for t in (k2, l2, m2, m2r, s2, i2):
                    ttnn.deallocate(t)
            for t in (keys, logits, mask, mask_rm, summed):
                ttnn.deallocate(t)
        ttnn.deallocate(mask_full)
    with open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ws3_scorer_bench.json", "w") as f:
        json.dump(out, f, indent=1)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
