# SPDX-License-Identifier: Apache-2.0
"""One chip: the VERIFY write patterns of tt/attention.py on a cache full of junk: B = U*block virtual rows with the page-table row of each real user repeated.
 batched: ONE paged_update_cache call, all rows active; skip: block calls, call t has update idx = -1 except rows of token t, same source tensor; seq: block calls with strided slices of the source."""
import json

import torch

import ttnn


def run(dev, mode, U=2, blk=8, nb=8):
    B = U * blk
    junk = torch.randn(U * nb, 1, 64, 192).bfloat16()
    pt_real = torch.arange(U * nb, dtype=torch.int32).reshape(U, nb)
    pt_virt = pt_real.repeat_interleave(blk, 0)
    kc = ttnn.from_torch(junk, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    pos0 = [74, 120][:U]
    # source: row v carries value (v+1) in every element (row 0 of its 32-row tile; other rows zero)
    src = torch.zeros(1, B, 32, 192)
    for v in range(B):
        src[0, v, 0, :] = float(v + 1)
    cores = ttnn.num_cores_to_corerangeset(B, ttnn.CoreCoord(8, 8), row_wise=True)
    mem = ttnn.create_sharded_memory_config(
        shape=(32, 192),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    srcd = ttnn.from_torch(
        src.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ptv = ttnn.from_torch(pt_virt, device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ptr = ttnn.from_torch(pt_real, device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    posv = torch.tensor([pos0[v // blk] + v % blk for v in range(B)], dtype=torch.int32)
    if mode == "batched":
        s = ttnn.to_memory_config(srcd, mem)
        idx = ttnn.from_torch(posv, device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.paged_update_cache(kc, s, update_idxs_tensor=idx, page_table=ptv)
    elif mode == "skip":
        s = ttnn.to_memory_config(srcd, mem)
        for t in range(blk):
            ut = torch.full((B,), -1, dtype=torch.int32)
            ut[t::blk] = posv[t::blk]
            idx = ttnn.from_torch(ut, device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.paged_update_cache(kc, s, update_idxs_tensor=idx, page_table=ptv)
    elif mode == "seq":
        cores_u = ttnn.num_cores_to_corerangeset(U, ttnn.CoreCoord(8, 8), row_wise=True)
        mem_u = ttnn.create_sharded_memory_config(
            shape=(32, 192),
            core_grid=cores_u,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        for t in range(blk):
            st = ttnn.to_memory_config(ttnn.slice(srcd, [0, t, 0, 0], [1, B, 32, 192], [1, blk, 1, 1]), mem_u)
            idx = ttnn.from_torch(posv[t::blk].contiguous(), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.paged_update_cache(kc, st, update_idxs_tensor=idx, page_table=ptr)
    ttnn.synchronize_device(dev)
    out = ttnn.to_torch(kc).float()
    res = {}
    for u in range(U):
        res[f"user{u}"] = [float(out[pt_real[u][(pos0[u] + t) // 64], 0, (pos0[u] + t) % 64, 0]) for t in range(blk)]
    print("PROBE", mode, json.dumps(res), "expected user0", [u * 0 + (t + 1) for t in range(blk)], flush=True)


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        for mode in ("batched", "skip", "seq"):
            run(dev, mode)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
