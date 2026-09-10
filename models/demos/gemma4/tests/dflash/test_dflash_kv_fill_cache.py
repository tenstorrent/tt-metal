# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact + device-time check for GEMMA4_DFLASH_KV_FILL_CACHE
(``attention.py``'s ``_write_seq_slice_fill_cache``) against the default
``_write_seq_slice`` path it's meant to replace.

Covers offsets that are NOT tile-aligned (the real case in production --
generate.py's ``context_len`` advances by an arbitrary accepted-token count
each iteration, not a 32-multiple) as well as offset=0 and an aligned offset,
since the whole point of this path is to stay correct in the unaligned case
via the small in-tile prefix read-back.

Run: pytest models/demos/gemma4/tests/dflash/test_dflash_kv_fill_cache.py -k 1x8 -s
"""

import torch
from loguru import logger

import ttnn

from ...tests.test_factory import parametrize_mesh_with_fabric

H = 2  # num_local_kv_heads (small, arbitrary)
D = 128  # head_dim
MAX_SEQ_LEN = 128

# (offset, length) -- includes non-tile-aligned offsets (the real case), an
# aligned offset, offset=0, and a write that crosses a tile boundary.
CASES = [
    (0, 5),
    (5, 3),  # unaligned offset, small length
    (13, 7),  # unaligned, crosses into next tile (13+7=20 < 32, still same tile actually)
    (29, 6),  # unaligned, write crosses the 32-tile boundary (29->35)
    (32, 16),  # aligned offset
    (61, 4),  # unaligned, near end
]


@parametrize_mesh_with_fabric([(1, 8)])
def test_dflash_kv_fill_cache_bit_exact(mesh_device, device_params, reset_seeds):
    from models.demos.gemma4.tt.dflash.attention import _write_seq_slice, _write_seq_slice_fill_cache

    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(mesh_device, "shape") else None

    all_ok = True
    for offset, length in CASES:
        base = torch.randn(1, H, MAX_SEQ_LEN, D, dtype=torch.bfloat16)
        new_rows_torch = torch.randn(1, H, length, D, dtype=torch.bfloat16)

        buf_old = ttnn.from_torch(
            base.clone(), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
        )
        buf_new = ttnn.from_torch(
            base.clone(), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
        )
        rows_old = ttnn.from_torch(
            new_rows_torch.clone(), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
        )
        rows_new = ttnn.from_torch(
            new_rows_torch.clone(), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
        )

        _write_seq_slice(buf_old, rows_old, offset, length)
        _write_seq_slice_fill_cache(buf_new, rows_new, offset, length)

        out_old = ttnn.to_torch(
            buf_old,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if hasattr(mesh_device, "shape") else None,
        )
        out_new = ttnn.to_torch(
            buf_new,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if hasattr(mesh_device, "shape") else None,
        )
        if hasattr(mesh_device, "shape"):
            # Replicated across devices -- just take the first shard, all identical.
            out_old = out_old[0:1]
            out_new = out_new[0:1]

        max_diff = (out_old - out_new).abs().max().item()
        ok = max_diff == 0.0
        all_ok &= ok
        msg = f"[kv-fill-cache] offset={offset:>4} length={length:>3}  max_abs_diff={max_diff}  {'PASS' if ok else 'FAIL'}"
        logger.info(msg)
        print(msg, flush=True)

        for t in (buf_old, buf_new, rows_old, rows_new):
            t.deallocate(True)

    assert all_ok, "GEMMA4_DFLASH_KV_FILL_CACHE path diverged from the default _write_seq_slice path"
