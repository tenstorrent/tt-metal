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
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.ccl import fabric_num_links
from models.demos.deepseek_v3_d_p.tt.v41.layout import SP_AXIS, V41MeshLayout
from models.demos.deepseek_v3_d_p.tt.v41.rope import cos_sin
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    MlaKvCacheFormat,
    MlaKvCacheGeometry,
    reconstruct_scaled_fp8_kv_cache,
)

WINDOW_SLOT = 128  # >= sliding_window - 1 carried rows, tile aligned


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


# per-chip bytes of the one zero block a state uploads per tensor; larger caches are zeroed by device copies of it (a
# replicated 0.5 GB host upload is a 4 GB host write that outlasts the safe runner's 5 s dispatch timeout)
ZERO_BLOCK_BYTES = 4 << 20


def replicated_zeros(mesh_device, rows: int, width: int, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    """Replicated DRAM zeros [1, 1, rows, width] of ``dtype`` in ``layout`` (row-major, or tiled with ``rows`` a
    multiple of 32): one host upload of at most ZERO_BLOCK_BYTES per chip (whole 32-row groups), copied on device
    over the rest of the rows."""
    step = max(ZERO_BLOCK_BYTES // (width * 2), 32) // 32 * 32  # rows of the block (<= 2 bytes per element)
    assert layout == ttnn.ROW_MAJOR_LAYOUT or rows % 32 == 0, (layout, rows)

    def upload(n):
        return ttnn.from_torch(
            torch.zeros(1, 1, n, width),
            device=mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    if rows <= step:
        return upload(rows)
    block = upload(step)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, width]), dtype, layout, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    for first in range(0, rows, step):
        n = min(step, rows - first)
        piece = block if n == step else ttnn.slice(block, [0, 0, 0, 0], [1, 1, n, width])
        ttnn.experimental.slice_write(piece, out, [0, 0, first, 0], [1, 1, first + n, width], [1, 1, 1, 1])
    ttnn.deallocate(block)
    return out


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
    * The indexer's start-relative visibility of the chunk's own compressed rows for chunks starting at a multiple
      of the ratio (``visibility_tail``; rows before the chunk are visible to every query), and each chip's first
      chunk query index (``query_first``: the candidate source pins each query's newest row from it).

    Per-chip query layout (window rows, indexer tails): chip (a, b) holds the chunk's contiguous query rows
    ``(a * tp + b) * S/(sp*tp)`` onwards, as after the head->sequence all-to-all.

    **Generic mode** (``generic=True``; ``V41GenericPrefillTrace``): a chunk's forward depends on its start only
    through device inputs, so one capture serves every chunk whose score width falls in one bucket. The RoPE
    slices, the pin page (``start + query_first``), the per-query visible counts of the indexer masks and the
    compressed-row indices of the cache writes are host-filled fixed buffers (``prepare_chunk``); the indexer
    scores at the bucket width (``bucket_width``) and masks the rows at or past each query's visible count
    (``mask``) instead of relying on the score kernel's start-relative causal window. No position tables are
    kept on the device (0.67 GB per state at 1M)."""

    def __init__(self, mesh_device, config, max_seq_len: int, chunk: int, layers: list[int], generic: bool = False):
        self.mesh_device, self.config = mesh_device, config
        self.sp, self.tp = tuple(mesh_device.shape)
        self.max_seq_len, self.chunk = max_seq_len, chunk
        self.generic = generic
        self.ratios = ratios = {config.compress_ratio(l) for l in layers}
        self.kinds = kinds = {(r > 0, 1) for r in ratios} | {(True, r) for r in ratios if r > 1}
        window = config.SLIDING_WINDOW
        self._window = {w: self._window_table(w) for w in {min(s, window - 1) for s in range(0, max_seq_len, chunk)}}
        self._query_first = self._query_first_table()
        self._rope, self._visibility = {}, {}
        if generic:
            self._init_generic(ratios, kinds)
            return
        self._rope = {kind: self._rope_table(*kind) for kind in sorted(kinds)}
        # ratio 1 needs no visibility table: the score kernel's causal mask is exactly t <= p there
        self._visibility = {r: self._visibility_table(r) for r in ratios if r > 1}

    # --- generic mode --------------------------------------------------------------------------------------------
    MIN_BUCKET = 8192  # the first score-width bucket (tokens); doubles up to max_seq_len

    def _init_generic(self, ratios, kinds):
        """The fixed per-chunk input buffers and the constants of the bucketed score widths."""
        rows = self.chunk // (self.sp * self.tp)  # this chip's query rows
        self.buckets = []
        w = self.MIN_BUCKET
        while w < self.max_seq_len:
            self.buckets.append(w)
            w *= 2
        self.buckets.append(_round_up(self.max_seq_len, 32))
        self._rope_buffers = {
            kind: tuple(self._replicated(torch.zeros(1, 1, self.chunk // kind[1], 64), ttnn.bfloat16) for _ in range(2))
            for kind in sorted(kinds)
        }
        # the pin page: start + query_first per chip (segment_max / candidate_scores read p_i = pin_base + pin[0] + i)
        self._pin = self._sharded_u32(torch.zeros(self.sp, self.tp, 1, 8, dtype=torch.int32))
        # per-ratio visible counts of this chip's queries [1, 1, rows, 1] fp32 tiled, and the column iota at the
        # widest score ([1, 1, 1, max/ratio + 32] fp32, sliced per bucket)
        self._visible = {r: self._per_query_chip(torch.zeros(self.chunk, 1), ttnn.float32) for r in ratios if r > 0}
        self._iota = {
            r: self._replicated(torch.arange(self.buckets[-1] // r + 32).float().view(1, 1, 1, -1), ttnn.float32)
            for r in ratios
            if r > 0
        }
        # the compressed rows a chunk writes: per ratio [1, 1, 1, chunk / ratio] uint32 (row-major, replicated)
        self._row_index = {
            r: ttnn.from_torch(
                torch.zeros(1, 1, 1, self.chunk // r, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for r in ratios
            if r > 0
        }
        self._query_rows = rows
        self._chunk_start = None

    def _sharded_u32(self, host):
        return ttnn.from_torch(
            host,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, 1)),
        )

    def bucket(self, start: int, length: int) -> int:
        """The score-width bucket (tokens) of the chunk at ``start``: the smallest bucket at or above its visible
        token count ``start + length``."""
        visible = start + length
        for w in self.buckets:
            if visible <= w:
                return w
        raise AssertionError(f"chunk at {start} + {length} beyond max_seq_len {self.max_seq_len}")

    def bucket_width(self, ratio: int, start: int, length: int) -> int:
        """The indexer's score width (rows of a ratio-``ratio`` source) of the chunk at ``start``: its bucket."""
        return self.bucket(start, length) // ratio

    def prepare_chunk(self, start: int, length: int) -> None:
        """Host-fill the chunk's input buffers: RoPE rows, the pin page, the visible counts, the row indices. Runs
        between a chunk's ``_stage`` and its forward (or replay); no device work."""
        assert self.generic
        assert start % self.chunk == 0 or start == 0, start
        chunk = self.chunk
        for (compressed, stride), buffers in self._rope_buffers.items():
            positions = torch.arange(start, start + chunk, stride)
            for host, buffer in zip(cos_sin(self.config, compressed, positions), buffers):
                ttnn.copy_host_to_device_tensor(self._host_replicated(host, ttnn.bfloat16), buffer)
        first = torch.arange(self.sp * self.tp, dtype=torch.int32) * self._query_rows  # chip query offsets
        pin = torch.zeros(self.sp, self.tp, 1, 8, dtype=torch.int32)
        pin[..., 0] = (first + start).reshape(self.sp, self.tp, 1)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                pin,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, 1)),
            ),
            self._pin,
        )
        positions = torch.arange(chunk, dtype=torch.int64) + start  # every query row of the chunk
        for r, buffer in self._visible.items():
            visible = ((positions + 1) // r).float().view(chunk, 1)
            host = visible.reshape(self.sp, self.tp, self._query_rows, 1)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    host,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, 1)),
                ),
                buffer,
            )
        for r, buffer in self._row_index.items():
            rows = torch.arange(chunk // r, dtype=torch.int32) + start // r
            ttnn.copy_host_to_device_tensor(
                self._host_replicated(rows.view(1, 1, 1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), buffer
            )
        self._chunk_start = start

    def _host_replicated(self, host, dtype, layout=ttnn.TILE_LAYOUT):
        """A replicated host tensor (``copy_host_to_device_tensor`` into a replicated device buffer)."""
        return ttnn.from_torch(
            host, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
        )

    def pin(self, start: int):
        """(pin page, pin_base) for ``segment_max`` / ``candidate_scores``: ``p_i = pin_base + pin[0] + i``."""
        if self.generic:
            assert self._chunk_start == start, (self._chunk_start, start)
            return self._pin, 0
        return self._query_first, start

    def mask(self, ratio: int, width: int):
        """Generic: additive bf16 [1, 1, rows, width] of this chip's queries: -inf at the score columns at or past
        the query's visible count (rows the query may not see), 0 elsewhere."""
        assert self.generic
        iota = ttnn.slice(self._iota[ratio], [0, 0, 0, 0], [1, 1, 1, width])
        beyond = ttnn.ge(iota, self._visible[ratio])
        return ttnn.typecast(ttnn.where(beyond, float("-inf"), 0.0), ttnn.bfloat16)

    def row_index(self, ratio: int):
        """Generic: the chunk's compressed row indices of a ratio-``ratio`` source, [1, 1, 1, chunk / ratio]."""
        return self._row_index[ratio]

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
        """Additive [chunk, chunk / ratio + 32]: query i sees the chunk's compressed row u iff u < (i + 1) // ratio
        (the extra tile, past the chunk's rows, is -inf: the indexer's zero-key pad tile)."""
        i = torch.arange(self.chunk).view(-1, 1)
        u = torch.arange(self.chunk // ratio + 32).view(1, -1)
        return self._per_query_chip(torch.where(u >= (i + 1) // ratio, float("-inf"), 0.0), ttnn.bfloat16)

    def _query_first_table(self):
        """Per-chip uint32 row-major [1, 1, 1, 8]: element 0 is the chip's first chunk query index (sp, tp) order."""
        rows = self.chunk // (self.sp * self.tp)
        first = torch.zeros(self.sp, self.tp, 1, 8, dtype=torch.int32)
        first[..., 0] = (torch.arange(self.sp * self.tp, dtype=torch.int32) * rows).reshape(self.sp, self.tp, 1)
        return ttnn.from_torch(
            first,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, 1)),
        )

    def rope(self, compressed: bool, stride: int, start: int):
        """cos, sin of the chunk at ``start``: rows at positions ``start + stride * j`` (j < chunk / stride), each SP
        rank its contiguous share, replicated over TP (``[1, 1, chunk / (stride * sp), 64]`` bf16 tiled)."""
        assert start % stride == 0 and start + self.chunk <= self.max_seq_len, (start, stride, self.max_seq_len)
        first, rows = start // stride, self.chunk // stride
        out = []
        if self.generic:
            assert self._chunk_start == start, (self._chunk_start, start)
            tables = self._rope_buffers[(compressed, stride)]
        else:
            tables = self._rope[(compressed, stride)]
        for t in tables:
            if not self.generic:
                t = ttnn.slice(t, [0, 0, first, 0], [1, 1, first + rows, t.shape[3]])
            out.append(ttnn.mesh_partition(t, dim=2, cluster_axis=SP_AXIS) if self.sp > 1 else t)
        return tuple(out)

    def window_rows(self, start: int):
        """Per-chip [1, 1, chunk/(sp*tp), window] int32 tiled rows of the chunk's causal windows (-1: none)."""
        return self._window[min(start, self.config.SLIDING_WINDOW - 1)]

    def visibility_tail(self, ratio: int, width: int):
        """Per-chip additive [1, 1, chunk/(sp*tp), width] mask of the chunk's first ``width`` compressed rows."""
        assert not self.generic, "generic mode masks the whole score (mask)"
        t = self._visibility[ratio]
        return ttnn.slice(t, [0, 0, 0, 0], [1, 1, t.shape[2], width])

    def query_first(self):
        """Per-chip [1, 1, 1, 8] uint32 row-major: element 0 is the chunk query index of the chip's first query row."""
        return self._query_first


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
        generic: bool = False,
    ):
        """``generic``: position-generic chunk graphs (``V41ChunkTables`` generic mode; ``V41GenericPrefillTrace``):
        the cache writes scatter by an uploaded row index instead of a start-dependent slice."""
        assert kv_format in self.FORMATS, f"V4.1 KV format must be one of {self.FORMATS}, got {kv_format}"
        self.mesh_device = mesh_device
        self.config = config
        layout = V41MeshLayout.of(mesh_device)
        self.geometry = V41CacheGeometry(config, max_seq_len, chunk, layout.sp)
        layout.check_chunk(chunk)
        self.layers = list(layers)
        self.tables = V41ChunkTables(mesh_device, config, max_seq_len, chunk, self.layers, generic=generic)
        self.generic = generic
        self.start = 0
        self.compressed_kv_format = kv_format
        # one 512-dim row with RoPE inside (V4.1 attends over all of it; no separate BF16 RoPE part)
        self.kv_geometry = MlaKvCacheGeometry(latent_dim=config.HEAD_DIM, rope_dim=0)
        g, idim = self.geometry, config.INDEX_HEAD_DIM

        def zeros(rows, width, fmt=MlaKvCacheFormat.BF16_RM):
            return replicated_zeros(mesh_device, rows, width, fmt.storage_dtype)

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
        # ratio-2 compressor carry: fp32 (kv, score) of a trailing incomplete group, None when groups are complete;
        # held in fixed buffers (``carry_buffers``), so a graph captured on the state (the decode pool's join) reads
        # them at one address whatever the prompt
        self.compressor_carry = {l: None for l in sources if config.compress_ratio(l) > 1}
        self.carry_buffers = {
            l: tuple(
                ttnn.from_torch(
                    torch.zeros(1, 1, 1, config.HEAD_DIM),
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                for _ in range(2)
            )
            for l in self.compressor_carry
        }
        # chunk-transient sharing: the latest index source's top-k and the candidate source's CandidateBlocks
        self.selection = {}
        # DSpark window rings (one per DSpark layer, slot p % window), set by the transformer when DSpark runs
        self.dspark_rings = None
        # decode (tt/v41/decode.py): the Engram hash history of the request this state holds
        self.engram_history = None
        self._ccl = get_tt_ccl(mesh_device) if layout.sp > 1 else None
        self._num_links = fabric_num_links()
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
            if self.generic:
                # the destination rows come from the chunk's uploaded row index (``first + i``), so the graph is the
                # same at every chunk start: row ``i`` of the encoded rows lands at ``dst_base + row_index[i]``
                from models.demos.deepseek_v3_d_p.tt.v41.pool import scatter_rows

                scatter_rows(
                    self._encode(rows, fmt), cache, self.tables.row_index(ratio), dst_base=offset - first, rows=count
                )
            else:
                self._write_rows(cache, self._encode(rows, fmt), offset)

    def write_window_rows(self, layer: int, rows):
        """Decode (``tt/v41/decode.py``): place ``layer``'s carry at rows [0, WINDOW_SLOT) and ``rows`` ``[1, 1, n,
        d]`` (the token's window KV on every row, the same on every chip; logical values of
        ``layer_kv_format(layer)``) at WINDOW_SLOT, so row WINDOW_SLOT is the token's and
        ``update_window_carry(layer, 1)`` is the ring shift. Returns the tensor its attention reads."""
        dst = self.kv_tensor(layer)
        self._write_rows(dst, self.window_carry[layer], 0)
        self._write_rows(dst, self._encode(rows, self.layer_kv_format(layer)), WINDOW_SLOT)
        return dst

    def write_compressed_row(self, source: int, kv_rows, index_k_rows, j: int):
        """Decode: compressed row ``j`` of ``source`` from row 0 of ``kv_rows`` ``[1, 1, n, d]`` (logical values of the
        source's format) and ``index_k_rows`` ``[1, 1, n, index_head_dim]`` (the same on every chip)."""
        ratio = self.config.compress_ratio(source)
        for cache, rows, offset, fmt in (
            (self.kv[source], kv_rows, self.geometry.kv_row_of_compressed(j), self.kv_format(ratio)),
            (self.index_k[source], index_k_rows, j, MlaKvCacheFormat.BF16_RM),
        ):
            rows = self._encode(rows, fmt)
            self._write_rows(cache, ttnn.slice(rows, [0, 0, 0, 0], [1, 1, 1, rows.shape[3]]), offset)

    def copy_compressed_row(self, source: int, src: int, dst: int):
        """Decode: copy compressed row ``src`` of ``source`` (KV and index key, as stored) to row ``dst``."""
        for cache, offset in ((self.kv[source], self.geometry.kv_row_of_compressed(0)), (self.index_k[source], 0)):
            row = ttnn.slice(cache, [0, 0, offset + src, 0], [1, 1, offset + src + 1, cache.shape[3]])
            self._write_rows(cache, row, offset + dst)
            ttnn.deallocate(row)

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
        assert remainder == 1, (remainder, ratio)  # one trailing token: the carry buffers hold one row
        for t, buffer in zip(carry, self.carry_buffers[source]):
            row = ttnn.slice(self._gather_sp(t), [0, 0, first, 0], [1, 1, length, self.config.HEAD_DIM])
            ttnn.copy(row, buffer)
            ttnn.deallocate(row)
        self.compressor_carry[source] = self.carry_buffers[source]

    def advance(self, length: int):
        self.start += length
        self.selection = {}
