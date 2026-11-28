# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-chunk values on the device, so one captured forward replays for every chunk (as tt_prefill_runtime.py's
per-chunk metadata [slot_id, actual_start, actual_end], updated in place before each replay).

``ChunkMeta`` holds one packed uint32 record per chip ([1, 1, 1, WORDS], a persistent DRAM tensor) that the host
rewrites before each chunk (``write``), and the 1-element tensors the trace-safe op forms read (``unpack``: device
slices of the record, the first ops of the forward, so a replay picks up the record written before it):

    slot, start, end   the chunk's cache slot, its start and the end of its real tokens (replicated)
    zero               0 (the slot of the attention's paste view)
    paste[k], k < sp   per chip: the start whose update_padded_kv_cache offset on this chip is SP row k's offset for
                       the chunk at ``start`` (the attention's bf16 paste of every SP row's chunk keys into the
                       replicated gathered prefix, TtAttention._paste_chunk_meta)

``derived`` holds tensors computed once per chunk from these and shared by the layers (the indexer's fp32 RoPE
tables at the chunk start); unpack frees them.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import SP_AXIS, TP_AXIS

WORDS = 16  # slot, start, end, zero, paste[0..sp), padded to a 64-byte record


def local_offset(prefix: int, coord: int, sp: int, rows: int) -> int:
    """update_padded_kv_cache's per-chip write offset (rows of the chip's shard) for a chunk at ``prefix``
    (writer_update_padded_kv_cache.cpp local_cache_offset, in rows)."""
    slab, boundary, off = prefix // (sp * rows), (prefix // rows) % sp, prefix % rows
    return slab * rows + (rows if coord < boundary else (off if coord == boundary else 0))


def paste_starts(start: int, sp: int, rows: int) -> list[list[int]]:
    """[k][r]: the start whose offset on SP row r is SP row k's offset for the chunk at ``start``: with boundary r
    and in-block offset T % rows, local_offset gives T (slab T // rows)."""
    out = []
    for k in range(sp):
        t = local_offset(start, k, sp, rows)
        out.append([(t // rows) * sp * rows + r * rows + t % rows for r in range(sp)])
    return out


class ChunkMeta:
    def __init__(self, mesh, chunk: int):
        self.mesh, self.chunk = mesh, chunk
        self.sp, self.tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
        self.rows = chunk // self.sp
        assert 4 + self.sp <= WORDS, (self.sp, WORDS)
        self.packed = ttnn.from_torch(
            self._record(0, 0, chunk),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._mapper(),
        )

        def scalar():
            return ttnn.from_torch(
                torch.zeros(1, 1, 1, 1, dtype=torch.int64),
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        self.slot, self.start, self.end, self.zero = scalar(), scalar(), scalar(), scalar()
        self.paste = [scalar() for _ in range(self.sp)]
        self.derived = {}
        self.host = (0, 0, chunk)  # (slot, start, end) of the last write

    def _mapper(self):
        dims = [None, None]
        dims[SP_AXIS], dims[TP_AXIS] = 0, 1
        return ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=dims)

    def scalars(self) -> list:
        return [self.slot, self.start, self.end, self.zero, *self.paste]

    def _record(self, slot: int, start: int, end: int) -> torch.Tensor:
        """[sp, tp, 1, WORDS] int64: chip (r, t)'s record (mesh row r = SP row r)."""
        rec = torch.zeros(self.sp, self.tp, 1, WORDS, dtype=torch.int64)
        rec[..., 0], rec[..., 1], rec[..., 2] = slot, start, end
        ps = paste_starts(start, self.sp, self.rows)
        for k in range(self.sp):
            for r in range(self.sp):
                rec[r, :, 0, 4 + k] = ps[k][r]
        if SP_AXIS != 0:
            rec = rec.transpose(0, 1).contiguous()
        return rec

    def write(self, slot: int, start: int, end: int) -> None:
        """Host -> the packed record (in place, before the forward that reads it)."""
        assert start % ttnn.TILE_SIZE == 0 and start < end <= start + self.chunk, (start, end)
        host = ttnn.from_torch(
            self._record(slot, start, end), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._mapper()
        )
        ttnn.copy_host_to_device_tensor(host, self.packed)
        self.host = (int(slot), int(start), int(end))

    def unpack(self) -> None:
        """The record -> the 1-element tensors, in place (device ops: the first ops of a captured forward)."""
        self.free_derived()
        for i, dst in enumerate(self.scalars()):
            word = ttnn.slice(self.packed, [0, 0, 0, i], [1, 1, 1, i + 1])
            ttnn.copy(word, dst)
            ttnn.deallocate(word)

    def free_derived(self) -> None:
        for t in self.derived.values():
            if t.is_allocated():
                ttnn.deallocate(t)
        self.derived = {}
