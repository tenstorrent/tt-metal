# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 prefill cache layout (bead F6.3) and per-request state.

Every attention call is one ``sparse_sdpa`` over a single row-major KV tensor, so a layer's window KV and
its source's compressed KV share one index space (U1). Each KV source owns one replicated, natural-order
tensor laid out as::

    rows [0, 128)                 window slot: the 128 window rows before the chunk (carry)
    rows [128, 128 + chunk)       the chunk's window KV (scratch, rewritten by every layer that reads it)
    rows [W, W + max_seq / r)     compressed KV row j at W + j,  W = 128 + chunk

Consumers read their source's tensor after writing their own window rows into its scratch region (layers
run in order, and the source's window rows are dead once its attention ran). Ratio-0 layers share one
scratch tensor without compressed rows. Each layer's window carry (128 rows) persists across chunks.
Index-K caches hold row j at j. Compressed rows are complete ratio groups only: chunk tokens [s, s+L) add
rows [s // r, (s + L) // r) (padded positions >= L write nothing; graph.md §4 rule 8).
"""

from dataclasses import dataclass

import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

WINDOW_SLOT = 128  # >= sliding_window - 1 carried rows, tile aligned


@dataclass(frozen=True)
class V41CacheGeometry:
    config: type
    max_seq_len: int
    chunk: int
    sp: int

    def __post_init__(self):
        align = 2 * 32 * self.sp  # ratio-2 groups and whole tiles on every SP rank
        assert self.chunk % align == 0, f"chunk {self.chunk} must be a multiple of 2*32*sp = {align}"
        assert self.max_seq_len % self.chunk == 0, "max_seq_len must be a whole number of chunks"
        assert self.config.SLIDING_WINDOW - 1 <= WINDOW_SLOT

    @property
    def window_rows(self) -> int:
        """Rows of the window region: slot + chunk."""
        return WINDOW_SLOT + self.chunk

    def compressed_rows(self, ratio: int) -> int:
        return self.max_seq_len // ratio

    def kv_rows(self, ratio: int) -> int:
        return self.window_rows + (self.compressed_rows(ratio) if ratio else 0)

    @staticmethod
    def new_compressed_rows(start: int, length: int, ratio: int) -> tuple[int, int]:
        """(first row, count) of complete groups the chunk tokens [start, start+length) complete."""
        assert start % ratio == 0, "chunks start on a group boundary"
        first, end = start // ratio, (start + length) // ratio
        return first, end - first

    def kv_row_of_compressed(self, j: int) -> int:
        return self.window_rows + j

    def kv_row_of_window(self, position: int, start: int) -> int:
        """Physical row of absolute token ``position`` (within 127 before the chunk start, or in the chunk)."""
        assert start - WINDOW_SLOT <= position, "older than the window slot"
        return WINDOW_SLOT + (position - start)


class V41PrefillState:
    """Per-request device state: compressed KV / index-K caches per KV source, ratio-0 scratch, window carries."""

    def __init__(self, mesh_device, config, max_seq_len: int, chunk: int, layers: list[int], dtype=ttnn.bfloat16):
        self.mesh_device = mesh_device
        self.config = config
        self.geometry = V41CacheGeometry(config, max_seq_len, chunk, mesh_device.shape[0])
        self.layers = list(layers)
        self.start = 0
        g, d, idim = self.geometry, config.HEAD_DIM, config.INDEX_HEAD_DIM

        def zeros(rows, width):
            return ttnn.from_torch(
                torch.zeros(1, 1, rows, width),
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        sources = [l for l in config.KV_SOURCE_LAYERS if l in self.layers]
        self.kv = {l: zeros(g.kv_rows(config.compress_ratio(l)), d) for l in sources}
        self.index_k = {l: zeros(g.compressed_rows(config.compress_ratio(l)), idim) for l in sources}
        self.swa_scratch = zeros(g.kv_rows(0), d) if any(config.compress_ratio(l) == 0 for l in self.layers) else None
        self.window_carry = {l: zeros(WINDOW_SLOT, d) for l in self.layers}
        # ratio-2 compressor carry: fp32 (kv, score) of a trailing incomplete group, None when groups are complete
        self.compressor_carry = {l: None for l in sources if config.compress_ratio(l) > 1}
        # chunk-transient sharing: the latest index source's top-k and the candidate source's blocks
        self.selection = {}
        self._ccl = get_tt_ccl(mesh_device) if mesh_device.shape[0] > 1 else None
        self._num_links = 2 if is_blackhole() else 1

    def kv_tensor(self, layer: int):
        """The KV tensor ``layer``'s attention reads: its KV source's, or the ratio-0 scratch."""
        ratio = self.config.compress_ratio(layer)
        return self.kv[self.config.kv_source(layer)] if ratio else self.swa_scratch

    def _gather_sp(self, t):
        """[1, 1, rows/sp, W] per SP rank (contiguous token order) -> [1, 1, rows, W] on every rank."""
        if self._ccl is None:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=2,
            multi_device_global_semaphore=self._ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=0),
            barrier_semaphore=self._ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=0),
            num_links=self._num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=0,
        )

    @staticmethod
    def _write_rows(dst, rows, first: int):
        """Write ``rows`` [1, 1, n, W] (row-major) into ``dst`` rows [first, first + n) in place."""
        n, width = rows.shape[2], rows.shape[3]
        ttnn.experimental.slice_write(rows, dst, [0, 0, first, 0], [1, 1, first + n, width], [1, 1, 1, 1])

    def write_window(self, layer: int, window_kv_local):
        """Place ``layer``'s carry and this chunk's window KV (per SP rank ``[1, 1, chunk/sp, d]``) into the
        window region of the tensor its attention reads. Returns that tensor."""
        dst = self.kv_tensor(layer)
        self._write_rows(dst, self.window_carry[layer], 0)
        self._write_rows(dst, ttnn.to_layout(self._gather_sp(window_kv_local), ttnn.ROW_MAJOR_LAYOUT), WINDOW_SLOT)
        return dst

    def write_compressed(self, source: int, kv_rows_local, index_k_rows_local, length: int):
        """Append the chunk's complete groups: per-SP-rank rows (contiguous tokens) of compressed KV
        ``[1, 1, rows/sp, d]`` and index keys ``[1, 1, rows/sp, index_head_dim]``. Only the first ``count``
        rows of the gathered chunk (complete groups of the valid length) are written."""
        ratio = self.config.compress_ratio(source)
        first, count = self.geometry.new_compressed_rows(self.start, length, ratio)
        if count == 0:
            return
        for cache, local, offset in (
            (self.kv[source], kv_rows_local, self.geometry.kv_row_of_compressed(first)),
            (self.index_k[source], index_k_rows_local, first),
        ):
            rows = ttnn.to_layout(self._gather_sp(local), ttnn.ROW_MAJOR_LAYOUT)
            if rows.shape[2] != count:
                rows = ttnn.slice(rows, [0, 0, 0, 0], [1, 1, count, rows.shape[3]])
            self._write_rows(cache, rows, offset)

    def update_window_carry(self, layer: int, length: int):
        """After ``layer``'s attention: its carry becomes the last WINDOW_SLOT rows ending at the chunk's
        last valid token (rows [length, length + WINDOW_SLOT) of the window region)."""
        src = self.kv_tensor(layer)
        carry = ttnn.slice(src, [0, 0, length, 0], [1, 1, length + WINDOW_SLOT, self.config.HEAD_DIM])
        self._write_rows(self.window_carry[layer], carry, 0)

    def set_compressor_carry(self, source: int, carry, length: int):
        """Keep the fp32 projections of a trailing incomplete ratio group (valid length not a multiple of the
        ratio); ``carry`` is the compressor's (kv, score) ``[1, 1, S/sp, head_dim]`` per SP rank."""
        ratio = self.config.compress_ratio(source)
        if ratio == 1:
            return
        remainder = (self.start + length) % ratio
        if remainder == 0:
            self.compressor_carry[source] = None
            return
        first = length - remainder
        self.compressor_carry[source] = tuple(
            ttnn.slice(self._gather_sp(t), [0, 0, first, 0], [1, 1, length, self.config.HEAD_DIM]) for t in carry
        )

    def advance(self, length: int):
        self.start += length
        self.selection = {}
