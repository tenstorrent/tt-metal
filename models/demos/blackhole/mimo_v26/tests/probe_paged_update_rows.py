# SPDX-License-Identifier: Apache-2.0
"""One chip probe for the verify KV write: (1) ttnn.slice with step along dim 1 of a [1,B,32,192] tile tensor, (2) paged_update_cache called `block` times back-to-back for consecutive positions of
the same user (same cache tile): do all rows land? (3) the same with ONE call over virtual users (page table row repeated)."""
import json

import torch

import ttnn


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        B, blk, U = 16, 8, 2
        x = torch.arange(B, dtype=torch.float32).reshape(1, B, 1, 1).expand(1, B, 32, 192).contiguous()
        xt = ttnn.from_torch(x.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        res = {}
        for t in (0, 3):
            st = ttnn.slice(xt, [0, t, 0, 0], [1, B, 32, 192], [1, blk, 1, 1])
            res[f"slice_t{t}_shape"] = list(st.shape)
            res[f"slice_t{t}_rows"] = ttnn.to_torch(st).float()[0, :, 0, 0].tolist()
        # paged update
        nb = 4
        cache = torch.zeros(U * nb, 1, 64, 192)
        pt = torch.arange(U * nb, dtype=torch.int32).reshape(U, nb)
        kc = ttnn.from_torch(cache.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
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
        for t in range(blk):
            val = torch.full((1, U, 32, 192), float(t + 1))
            val[:, 1] += 100
            src = ttnn.from_torch(val.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            src = ttnn.to_memory_config(src, mem)
            idx = ttnn.from_torch(
                torch.tensor([p + t for p in pos0], dtype=torch.int32),
                device=dev,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.experimental.paged_update_cache(kc, src, update_idxs_tensor=idx, page_table=ptt)
        out = ttnn.to_torch(kc).float()
        for u in range(U):
            rows = []
            for t in range(blk):
                p = pos0[u] + t
                rows.append(float(out[pt[u][p // 64], 0, p % 64, 0]))
            res[f"seq_calls_user{u}_rows"] = rows
        print("PROBE", json.dumps(res))
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
