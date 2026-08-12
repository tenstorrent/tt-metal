# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Peer-merge the decode RoPE cos/sin gathers: one packed table instead of two.

``_decode_rope_tables`` runs the same three ops twice, once per table, on the
same ``rope_pos_ids`` index tensor -- the shape the graph-fusing skill calls a
peer merge.  Packing ``cos`` and ``sin`` into one ``[max_seq, 2*head_dim]``
table makes it one ``ttnn.embedding`` + one ``transpose``, at the cost of two
width slices to split the halves back apart before sharding.

Measured on device kernel time at the shipped decode shapes (run under
``python -m tracy -r -p -v``).  ``head_dim = 128`` is four tiles wide, so the
split is tile-aligned; ``max_seq`` here is the advertised 131072 context, i.e.
the real table size the layer carries.
"""

from __future__ import annotations

import torch

import ttnn

BATCH = 32
HEAD_DIM = 128
MAX_SEQ = 131072
TILE = 32
REPS = 16


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        torch.manual_seed(0)
        cos = torch.randn(MAX_SEQ, HEAD_DIM).to(torch.bfloat16)
        sin = torch.randn(MAX_SEQ, HEAD_DIM).to(torch.bfloat16)
        packed = torch.cat([cos, sin], dim=-1)
        to_dev = lambda t: ttnn.from_torch(  # noqa: E731
            t,
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        t_cos, t_sin, t_packed = to_dev(cos), to_dev(sin), to_dev(packed)
        positions = torch.randint(0, MAX_SEQ, (1, BATCH), dtype=torch.int32)
        idx = ttnn.from_torch(
            positions,
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(i % grid.x, i // grid.x), ttnn.CoreCoord(i % grid.x, i // grid.x))
                for i in range(BATCH)
            }
        )
        memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, (TILE, HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR),
        )

        def shipped():
            out = []
            for table in (t_cos, t_sin):
                rows = ttnn.unsqueeze_to_4D(ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT))
                per_user = ttnn.transpose(rows, 1, 2)
                ttnn.deallocate(rows)
                out.append(ttnn.interleaved_to_sharded(per_user, memcfg))
                ttnn.deallocate(per_user)
            return out

        def merged():
            rows = ttnn.unsqueeze_to_4D(ttnn.embedding(idx, t_packed, layout=ttnn.TILE_LAYOUT))
            per_user = ttnn.transpose(rows, 1, 2)  # [1, batch, 1, 2*d]
            ttnn.deallocate(rows)
            out = []
            for lo in (0, HEAD_DIM):
                half = ttnn.slice(per_user, [0, 0, 0, lo], [1, BATCH, 1, lo + HEAD_DIM])
                out.append(ttnn.interleaved_to_sharded(half, memcfg))
                ttnn.deallocate(half)
            ttnn.deallocate(per_user)
            return out

        # correctness first: the merged form must produce the same two tensors
        a = shipped()
        b = merged()
        for name, x, y in (("cos", a[0], b[0]), ("sin", a[1], b[1])):
            same = torch.equal(ttnn.to_torch(x), ttnn.to_torch(y))
            print(f"MATCH {name}: {same}", flush=True)
        for t in a + b:
            ttnn.deallocate(t)

        for label, fn in (("shipped_two_gathers", shipped), ("merged_packed_table", merged)):
            print(f"GROUP {REPS} decode_rope {label}", flush=True)
            for _ in range(REPS):
                for t in fn():
                    ttnn.deallocate(t)
            ttnn.synchronize_device(mesh)
        for t in (t_cos, t_sin, t_packed, idx):
            ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
