# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4ServePool (task serve-long): Hy4PoolDecoder + a MASKED prefill hand-off.

The stock ``Hy4PoolDecoder.handoff`` copies the chain prefill cache of EVERY mesh row into the pool: right for the needle test (4 new prompts), wrong for a server that prefills
one new prompt while the other mesh rows hold live decode users. ``handoff_rows`` copies only the listed mesh rows (``slice_copy_masked.cpp``: colid element 2 != 0 => the device
copies nothing); everything else is the stock path (same pools, same copy kernel logic, no change to the decode trace).
"""
import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder

MASKED = "slice_copy_masked.cpp"


class Hy4ServePool(Hy4PoolDecoder):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.colmask = upload(
            self.md, HP.colid_masked_host([]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, shard_dim=1, row_dim=0
        )

    def _set_mask(self, keep_rows):
        skip = [r for r in range(int(self.md.shape[0])) if r not in set(keep_rows)]
        host = ttnn.from_torch(
            HP.colid_masked_host(skip),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, 1), mesh_shape=tuple(self.md.shape)),
        )
        ttnn.copy_host_to_device_tensor(host, self.colmask)

    def handoff_rows(self, nrows, keep_rows):
        """chain prefill caches [1,1,N,656] (+ indexer keys) -> pool, ONLY for the mesh rows in ``keep_rows``; nrows = N, a multiple of 512."""
        assert nrows % HP.CHUNK == 0 and nrows // HP.CHUNK <= self.pages
        npg = nrows // HP.CHUNK
        self._set_mask(keep_rows)
        for ly, kv, kp in zip(self.ch.layers, self.kv_pools, self.key_pools):
            rows = ly.attn.cache.rows
            assert int(rows.shape[2]) >= nrows and rows.dtype == ttnn.fp8_e4m3, (rows.shape, rows.dtype)
            HP.slice_copy(rows, kv, self.colmask, 64, HP.ROWB, npg, kernel=MASKED)
            if kp is not None:
                k = ly.attn.indexer.keys
                kb = ttnn.typecast(k, ttnn.bfloat8_b) if k.dtype != ttnn.bfloat8_b else k
                kv4 = ttnn.reshape(kp, [1, 1, self.pages * 64, 128])
                HP.slice_copy(kb, kv4, self.colmask, 8, HP.KEYTILE, npg, kernel=MASKED)
                if kb is not k:
                    ttnn.deallocate(kb)
