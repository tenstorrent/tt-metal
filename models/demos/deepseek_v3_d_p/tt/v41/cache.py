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

KV format (epic KV FORMAT, dev-spec D-A), chosen at construction for the compressed layers' KV tensors (their
window region and carries included, since one ``sparse_sdpa`` reads the whole tensor in one format):

* ``BF16_RM`` (stage 1): 512 BF16 per row, holding the values the caller passes (the reference's QDQ values).
* ``SCALED_FP8`` (stage 2): the caller passes unrounded KV; each row is stored as 512 FP8 e4m3 bytes and 4
  FP32 power-of-two scales, one per 128 dims (RoPE dims included, no BF16 RoPE tail) = 528 bytes.

Ratio-0 (sliding-window-only) layers keep BF16 in both stages: their window KV holds the reference's FP8
block-32 values exactly, and no compressed rows share their tensor.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.layout import SP_AXIS, V41MeshLayout
from models.demos.deepseek_v3_d_p.tt.v41.rope import cos_sin
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    MlaKvCacheFormat,
    MlaKvCacheGeometry,
    reconstruct_scaled_fp8_kv_cache,
)

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


class V41ChunkTables:
    """Device-resident position tables of one cache geometry, uploaded once outside the chunk loop, so a chunk's
    forward uploads and writes nothing from the host (trace capture; dev-spec D-G (b)). Per chunk the forward
    only slices them on device.

    * RoPE cos/sin of every position (``rope``), one replicated ``[1, 1, max_seq / stride, 64]`` table per kind:
      ratio-0 or compressed frequencies, rows at token positions (stride 1) or at ratio-r group-first positions
      (stride r: compressed KV, index keys).
    * Window rows (``window_rows``): the per-chip query rows' 128-row causal windows into the KV tensor's window
      region; they depend on the chunk start only through ``min(start, window - 1)``.
    * The indexer's start-relative constants for chunks starting at a multiple of the ratio: the visibility of the
      chunk's own compressed rows (``visibility_tail``; rows before the chunk are visible to every query) and the
      candidate source's pinned newest block (``pin_tail``).

    Per-chip query layout (window rows, indexer tails): chip (a, b) holds the chunk's contiguous query rows
    ``(a * tp + b) * S/(sp*tp)`` onwards, as after the head->sequence all-to-all."""

    def __init__(self, mesh_device, config, max_seq_len: int, chunk: int, layers: list[int]):
        self.mesh_device, self.config = mesh_device, config
        self.sp, self.tp = tuple(mesh_device.shape)
        self.max_seq_len, self.chunk = max_seq_len, chunk
        ratios = {config.compress_ratio(l) for l in layers}
        kinds = {(r > 0, 1) for r in ratios} | {(True, r) for r in ratios if r > 1}
        self._rope = {kind: self._rope_table(*kind) for kind in sorted(kinds)}
        window = config.SLIDING_WINDOW
        self._window = {w: self._window_table(w) for w in {min(s, window - 1) for s in range(0, max_seq_len, chunk)}}
        # ratio 1 needs no visibility table: the score kernel's causal mask is exactly t <= p there
        self._visibility = {r: self._visibility_table(r) for r in ratios if r > 1}
        self._pin = None
        if config.CANDIDATE_SOURCE_LAYER in layers:
            assert config.compress_ratio(config.CANDIDATE_SOURCE_LAYER) == 1, "pin_tail assumes a ratio-1 source"
            self._pin = self._pin_table(config.CANDIDATE_BLOCK_SIZE)

    def _replicated(self, host, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            host,
            device=self.mesh_device,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _per_query_chip(self, host: torch.Tensor, dtype):
        """[chunk, W] query-order rows -> each chip its contiguous chunk/(sp*tp) rows ((sp, tp) chip order)."""
        rows, width = host.shape
        return ttnn.from_torch(
            host.reshape(self.sp, self.tp, rows // (self.sp * self.tp), width),
            device=self.mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, 1)),
        )

    def _rope_table(self, compressed: bool, stride: int):
        positions = torch.arange(0, self.max_seq_len, stride)
        return tuple(self._replicated(t, ttnn.bfloat16) for t in cos_sin(self.config, compressed, positions))

    def _window_table(self, before: int):
        """Rows ``WINDOW_SLOT + i + o`` (o in (-window, 0]) of query i, -1 where i + o < -before (no token)."""
        window = self.config.SLIDING_WINDOW
        i = torch.arange(self.chunk).view(-1, 1)
        offset = torch.arange(-(window - 1), 1).view(1, -1)
        rows = torch.where(before + i + offset >= 0, WINDOW_SLOT + i + offset, -1).to(torch.int32)
        return self._per_query_chip(rows, ttnn.int32)

    def _visibility_table(self, ratio: int):
        """Additive [chunk, chunk / ratio]: query i sees the chunk's compressed row u iff u < (i + 1) // ratio."""
        i = torch.arange(self.chunk).view(-1, 1)
        u = torch.arange(self.chunk // ratio).view(1, -1)
        return self._per_query_chip(torch.where(u >= (i + 1) // ratio, float("-inf"), 0.0), ttnn.bfloat16)

    def _pin_table(self, block: int):
        """Additive [chunk, chunk / block]: +inf at query i's newest block (the chunk's block i // block)."""
        i = torch.arange(self.chunk).view(-1, 1)
        u = torch.arange(-(-self.chunk // block)).view(1, -1)
        return self._per_query_chip(torch.where(u == i // block, float("inf"), 0.0), ttnn.bfloat16)

    def rope(self, compressed: bool, stride: int, start: int):
        """cos, sin of the chunk at ``start``: rows at positions ``start + stride * j`` (j < chunk / stride), each SP
        rank its contiguous share, replicated over TP (``[1, 1, chunk / (stride * sp), 64]`` bf16 tiled)."""
        assert start % stride == 0 and start + self.chunk <= self.max_seq_len, (start, stride, self.max_seq_len)
        first, rows = start // stride, self.chunk // stride
        out = []
        for t in self._rope[(compressed, stride)]:
            t = ttnn.slice(t, [0, 0, first, 0], [1, 1, first + rows, t.shape[3]])
            out.append(ttnn.mesh_partition(t, dim=2, cluster_axis=SP_AXIS) if self.sp > 1 else t)
        return tuple(out)

    def window_rows(self, start: int):
        """Per-chip [1, 1, chunk/(sp*tp), window] int32 tiled rows of the chunk's causal windows (-1: none)."""
        return self._window[min(start, self.config.SLIDING_WINDOW - 1)]

    def visibility_tail(self, ratio: int, width: int):
        """Per-chip additive [1, 1, chunk/(sp*tp), width] mask of the chunk's first ``width`` compressed rows."""
        t = self._visibility[ratio]
        return ttnn.slice(t, [0, 0, 0, 0], [1, 1, t.shape[2], width])

    def pin_tail(self, width: int):
        """Per-chip additive [1, 1, chunk/(sp*tp), width] +inf at each query's newest block among the chunk's
        first ``width`` blocks (a padding query's newest block may lie past them: no pin, as it is unreachable)."""
        return ttnn.slice(self._pin, [0, 0, 0, 0], [1, 1, self._pin.shape[2], width])


class V41PrefillState:
    """Per-request device state: compressed KV / index-K caches per KV source, ratio-0 scratch, window carries,
    DSpark window rings, and the geometry's position tables (``tables``)."""

    FORMATS = (MlaKvCacheFormat.BF16_RM, MlaKvCacheFormat.SCALED_FP8)

    def __init__(
        self,
        mesh_device,
        config,
        max_seq_len: int,
        chunk: int,
        layers: list[int],
        kv_format: MlaKvCacheFormat = MlaKvCacheFormat.BF16_RM,
    ):
        assert kv_format in self.FORMATS, f"V4.1 KV format must be one of {self.FORMATS}, got {kv_format}"
        self.mesh_device = mesh_device
        self.config = config
        layout = V41MeshLayout.of(mesh_device)
        self.geometry = V41CacheGeometry(config, max_seq_len, chunk, layout.sp)
        layout.check_chunk(chunk)
        self.layers = list(layers)
        self.tables = V41ChunkTables(mesh_device, config, max_seq_len, chunk, self.layers)
        self.start = 0
        self.compressed_kv_format = kv_format
        # one 512-dim row with RoPE inside (V4.1 attends over all of it; no separate BF16 RoPE part)
        self.kv_geometry = MlaKvCacheGeometry(latent_dim=config.HEAD_DIM, rope_dim=0)
        g, idim = self.geometry, config.INDEX_HEAD_DIM

        def zeros(rows, width, fmt=MlaKvCacheFormat.BF16_RM):
            return ttnn.from_torch(
                torch.zeros(1, 1, rows, width),
                device=mesh_device,
                dtype=fmt.storage_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        def kv_zeros(rows, ratio):
            fmt = self.kv_format(ratio)
            return zeros(rows, fmt.storage_width(self.kv_geometry), fmt)

        sources = [l for l in config.KV_SOURCE_LAYERS if l in self.layers]
        self.kv = {l: kv_zeros(g.kv_rows(config.compress_ratio(l)), config.compress_ratio(l)) for l in sources}
        self.index_k = {l: zeros(g.compressed_rows(config.compress_ratio(l)), idim) for l in sources}
        self.swa_scratch = (
            kv_zeros(g.kv_rows(0), 0) if any(config.compress_ratio(l) == 0 for l in self.layers) else None
        )
        self.window_carry = {l: kv_zeros(WINDOW_SLOT, config.compress_ratio(l)) for l in self.layers}
        # ratio-2 compressor carry: fp32 (kv, score) of a trailing incomplete group, None when groups are complete
        self.compressor_carry = {l: None for l in sources if config.compress_ratio(l) > 1}
        # chunk-transient sharing: the latest index source's top-k and the candidate source's blocks
        self.selection = {}
        # DSpark window rings (one per DSpark layer, slot p % window), set by the transformer when DSpark runs
        self.dspark_rings = None
        self._ccl = get_tt_ccl(mesh_device) if layout.sp > 1 else None
        self._num_links = 2 if is_blackhole() else 1
        self._sp_topology = per_axis_topology()[SP_AXIS]  # Ring only if the opened fabric wraps SP

    def kv_format(self, ratio: int) -> MlaKvCacheFormat:
        """Storage format of the KV tensors (and window carries) of layers with compress ``ratio``."""
        return self.compressed_kv_format if ratio else MlaKvCacheFormat.BF16_RM

    def layer_kv_format(self, layer: int) -> MlaKvCacheFormat:
        """Format of ``layer``'s KV tensor: what ``write_window`` / ``write_compressed`` expect (reference QDQ
        values for BF16_RM, unrounded values for SCALED_FP8) and what its ``sparse_sdpa`` reads."""
        return self.kv_format(self.config.compress_ratio(layer))

    def _encode(self, rows, fmt: MlaKvCacheFormat):
        """Logical [1, 1, n, 512] rows (any layout) -> the physical row-major rows of ``fmt``."""
        rows = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
        if fmt == MlaKvCacheFormat.BF16_RM:
            return rows
        latent, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
            rows, round_scale_to_power_of_two=True
        )
        packed = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(latent, scales)
        ttnn.deallocate(latent)
        ttnn.deallocate(scales)
        return packed

    def to_host(self, t) -> torch.Tensor:
        """Logical fp32 rows [rows, width] of a state tensor (first device's replica); SCALED_FP8 rows are decoded."""
        if t.dtype == MlaKvCacheFormat.SCALED_FP8.storage_dtype:
            # FP8 bytes leave the device only through a mesh composer (single-device FP8 to_torch is unsupported)
            composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            physical = ttnn.to_torch(t, mesh_composer=composer)[0, 0]
            return reconstruct_scaled_fp8_kv_cache(physical, self.kv_geometry).float()
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])[0, 0].float()

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
            multi_device_global_semaphore=self._ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
            barrier_semaphore=self._ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=SP_AXIS),
            num_links=self._num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self._sp_topology,
            cluster_axis=SP_AXIS,
        )

    @staticmethod
    def _write_rows(dst, rows, first: int):
        """Write ``rows`` [1, 1, n, W] (row-major) into ``dst`` rows [first, first + n) in place."""
        n, width = rows.shape[2], rows.shape[3]
        ttnn.experimental.slice_write(rows, dst, [0, 0, first, 0], [1, 1, first + n, width], [1, 1, 1, 1])

    def write_window(self, layer: int, window_kv_local):
        """Place ``layer``'s carry and this chunk's window KV (per SP rank ``[1, 1, chunk/sp, d]``, logical
        values of ``layer_kv_format(layer)``) into the window region of the tensor its attention reads. Returns
        that tensor."""
        dst = self.kv_tensor(layer)
        self._write_rows(dst, self.window_carry[layer], 0)
        rows = self._encode(self._gather_sp(window_kv_local), self.layer_kv_format(layer))
        self._write_rows(dst, rows, WINDOW_SLOT)
        return dst

    def write_compressed(self, source: int, kv_rows_local, index_k_rows_local, length: int):
        """Append the chunk's complete groups: per-SP-rank rows (contiguous tokens) of compressed KV
        ``[1, 1, rows/sp, d]`` (logical values of the source's format) and index keys
        ``[1, 1, rows/sp, index_head_dim]``. Only the first ``count`` rows of the gathered chunk (complete
        groups of the valid length) are written."""
        ratio = self.config.compress_ratio(source)
        first, count = self.geometry.new_compressed_rows(self.start, length, ratio)
        if count == 0:
            return
        for cache, local, offset, fmt in (
            (self.kv[source], kv_rows_local, self.geometry.kv_row_of_compressed(first), self.kv_format(ratio)),
            (self.index_k[source], index_k_rows_local, first, MlaKvCacheFormat.BF16_RM),
        ):
            rows = ttnn.to_layout(self._gather_sp(local), ttnn.ROW_MAJOR_LAYOUT)
            if rows.shape[2] != count:
                rows = ttnn.slice(rows, [0, 0, 0, 0], [1, 1, count, rows.shape[3]])
            self._write_rows(cache, self._encode(rows, fmt), offset)

    def update_window_carry(self, layer: int, length: int):
        """After ``layer``'s attention: its carry becomes the last WINDOW_SLOT rows ending at the chunk's
        last valid token (rows [length, length + WINDOW_SLOT) of the window region)."""
        src = self.kv_tensor(layer)
        carry = ttnn.slice(src, [0, 0, length, 0], [1, 1, length + WINDOW_SLOT, src.shape[3]])
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
