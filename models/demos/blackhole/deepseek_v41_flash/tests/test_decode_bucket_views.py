# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device facts the decode buckets rest on (VLLM_BUCKETS_NOTES.md): is a [1,1,Ub,W] view of a [1,1,U,W] tile tensor zero-copy, do in-place writes through the view (eager and in a trace) land in the
full tensor, what happens to the padding rows, can the key slab be updated / sliced with a padded update.

    python models/demos/blackhole/deepseek_v41_flash/tests/test_decode_bucket_views.py
"""

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.decode_buckets import view_rows


def main():
    md = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), trace_region_size=200_000_000)
    try:
        rows, cols = 4, 8
        U, Ub, W = 8, 4, 1024
        shard = ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols))
        full_t = (
            torch.arange(rows * U, dtype=torch.float32)
            .reshape(1, 1, rows * U, 1)
            .expand(1, 1, rows * U, W)
            .contiguous()
        )
        t = ttnn.from_torch(full_t, device=md, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard)
        v = view_rows(t, Ub)
        print(
            "addr equal:",
            v.buffer_address() == t.buffer_address(),
            "view shape",
            tuple(v.shape),
            "padded",
            tuple(v.padded_shape),
        )
        x = ttnn.from_torch(
            torch.full((1, 1, rows * Ub, W), -5.0),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=shard,
        )
        ttnn.copy(x, v)
        back = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(2, 1), mesh_shape=(rows, cols)))
        back = back[:, :1]  # [1,1,rows*U,W] (column 0)
        per_row = back.reshape(rows, U, W)[:, :, 0]
        print("eager copy through the view: per mesh row, column-0 value of every user:\n", per_row)
        ok = bool((per_row[:, :Ub] == -5.0).all())
        print(
            "users < Ub written:",
            ok,
            "| users >= Ub left intact:",
            bool((per_row[:, Ub:] == torch.arange(rows * U).reshape(rows, U)[:, Ub:].float()).all()),
        )
        # in a trace: add 1 to the view in place each replay
        one = ttnn.from_torch(
            torch.ones(1, 1, rows * Ub, W), device=md, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard
        )
        ttnn.copy(ttnn.add(v, one), v)  # compile
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        ttnn.copy(ttnn.add(v, one), v)
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        back = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(2, 1), mesh_shape=(rows, cols)))[
            :, :1
        ]
        per_row = back.reshape(rows, U, W)[:, :, 0]
        print(
            "after 1 eager + 3 traced increments (expect users<Ub: -5+1+3 = -1; others unchanged or the tile padding value):\n",
            per_row,
        )
        ttnn.release_trace(md, tid)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()
