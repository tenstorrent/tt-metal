"""NAN49: all_to_all_async_generic (heads<->sequence, cluster_axis=1) exactness over consecutive calls with num_links 1 / 2 on a ring-fabric 1x8 slot (the selftest does not cover this op)."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    full, md = open_row()
    try:
        n = int(os.environ.get("N", "512"))
        for nl in (2, 1):
            bad_calls = []
            for it in range(int(os.environ.get("ITERS", "12"))):
                x = torch.randint(-8, 8, (8, 8, n, 576)).bfloat16()  # device c: [1,8 heads,n,576]
                t = ttnn.from_torch(
                    x,
                    device=md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                q = ttnn.experimental.all_to_all_async_generic(
                    t, in_dim=1, out_dim=2, num_links=nl, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1
                )
                back = ttnn.experimental.all_to_all_async_generic(
                    q, in_dim=2, out_dim=1, num_links=nl, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1
                )
                ttnn.synchronize_device(md)
                full_heads = x.reshape(64, n, 576)  # head index = c*8 + h
                ok_mid = True
                for c, td in enumerate(ttnn.get_device_tensors(q)):
                    h = ttnn.to_torch(td).reshape(64, n // 8, 576)
                    ok_mid &= torch.equal(h, full_heads[:, c * (n // 8) : (c + 1) * (n // 8)])
                ok_back = all(
                    torch.equal(ttnn.to_torch(td).reshape(8, n, 576), x[c])
                    for c, td in enumerate(ttnn.get_device_tensors(back))
                )
                if not (ok_mid and ok_back):
                    bad_calls.append((it, ok_mid, ok_back))
            res(event="a2a", num_links=nl, calls=int(os.environ.get("ITERS", "12")), bad=bad_calls)
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
