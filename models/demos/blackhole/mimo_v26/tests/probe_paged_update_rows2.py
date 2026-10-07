# SPDX-License-Identifier: Apache-2.0
"""One chip: do back-to-back paged_update_cache calls (same cache tile, consecutive positions) all land? eager back-to-back (inputs prepared up front), and inside a trace."""
import json

import torch

import ttnn


def check(dev, tag, runner, U=2, blk=8, nb=4):
    cache = torch.randn(U * nb, 1, 64, 192).bfloat16()  # junk everywhere so a lost update is visible
    pt = torch.arange(U * nb, dtype=torch.int32).reshape(U, nb)
    kc = ttnn.from_torch(cache, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ptt = ttnn.from_torch(pt, device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos0 = [10, 100]
    cores = ttnn.num_cores_to_corerangeset(U, ttnn.CoreCoord(8, 8), row_wise=True)
    mem = ttnn.create_sharded_memory_config(
        shape=(32, 192),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    srcs, idxs = [], []
    for t in range(blk):
        val = torch.full((1, U, 32, 192), float(t + 1))
        val[:, 1] += 100
        srcs.append(
            ttnn.to_memory_config(
                ttnn.from_torch(val.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), mem
            )
        )
        idxs.append(
            ttnn.from_torch(
                torch.tensor([p + t for p in pos0], dtype=torch.int32),
                device=dev,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        )
    ttnn.synchronize_device(dev)
    runner(dev, kc, srcs, idxs, ptt)
    out = ttnn.to_torch(kc).float()
    res = {}
    for u in range(U):
        res[f"user{u}"] = [float(out[pt[u][(pos0[u] + t) // 64], 0, (pos0[u] + t) % 64, 0]) for t in range(blk)]
    print("PROBE", tag, json.dumps(res), flush=True)


def eager(dev, kc, srcs, idxs, ptt):
    for s, i in zip(srcs, idxs):
        ttnn.experimental.paged_update_cache(kc, s, update_idxs_tensor=i, page_table=ptt)
    ttnn.synchronize_device(dev)


def traced(dev, kc, srcs, idxs, ptt):
    def body():
        for s, i in zip(srcs, idxs):
            ttnn.experimental.paged_update_cache(kc, s, update_idxs_tensor=i, page_table=ptt)

    body()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    body()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
    try:
        check(dev, "eager_back_to_back", eager)
        check(dev, "traced_back_to_back", traced)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
