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

import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.ccl import fabric_num_links
from models.demos.deepseek_v3_d_p.tt.v41.dspark import seed_rows
from models.demos.deepseek_v3_d_p.tt.v41.layout import SP_AXIS, V41MeshLayout
from models.demos.deepseek_v3_d_p.tt.v41.rope import cos_sin
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    MlaKvCacheFormat,
    MlaKvCacheGeometry,
    reconstruct_scaled_fp8_kv_cache,
)

WINDOW_SLOT = 128  # >= sliding_window - 1 carried rows, tile aligned

# "1": a generic state may run a packed chunk (several whole prompts concatenated in one chunk, each isolated by
# its own RoPE positions, window rows, index-mask bounds, carries and scored row; 
# ``TtV41Transformer.prefill_packed``). Off (the default), nothing in the chunk graph changes.
PACK = "TT_V41_PACK"


def pack_enabled() -> bool:
    return os.environ.get(PACK, "0") == "1"


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def pack_segments(lengths: list[int]) -> tuple[tuple[int, int], ...]:
    """The ``(offset, length)`` of each prompt of a pack laid out contiguously from row 0 (layout L-A)."""
    segments, offset = [], 0
    for n in lengths:
        segments.append((offset, int(n)))
        offset += int(n)
    return tuple(segments)


def check_pack(segments, chunk: int, ratios) -> None:
    """A pack fits the chunk and every segment starts and ends on a ratio-group boundary (the compressor pairs
    ``(2i, 2i + 1)`` must not straddle a prompt, and a segment's compressed rows must be whole groups)."""
    align = max([r for r in ratios if r > 0] or [1])
    assert len(segments) >= 1 and all(n > 0 for _, n in segments), segments
    assert all(o % align == 0 and n % align == 0 for o, n in segments), f"segments must be multiples of {align}"
    end = segments[-1][0] + segments[-1][1]
    assert end <= chunk, f"pack of {end} rows does not fit the {chunk}-row chunk"


# The served layouts (``TT_V41_PACK_LAYOUTS``): a comma list of ``<k>x<length>`` (k whole
# prompts of exactly ``length`` tokens, e.g. ``4x1028,3x1028,2x1028,2x2052``) or ``<l1>+<l2>+...`` (distinct
# lengths in row order). A ``prefill_forward`` whose rows (in order) begin with a listed layout's lengths runs them
# as one packed chunk; its capture is made at warm-up. Unset: no packing in the generator.
PACK_LAYOUTS = "TT_V41_PACK_LAYOUTS"


def parse_pack_layouts(spec: str | None, chunk: int, ratios) -> list[tuple[tuple[int, int], ...]]:
    """The layouts of ``spec`` (``PACK_LAYOUTS``' form) as ``pack_segments`` tuples, each checked against the chunk;
    longest (in prompts) first, so a FIFO prefix matches the largest layout it can."""
    layouts = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "x" in item:
            k, n = item.split("x")
            lengths = [int(n)] * int(k)
        else:
            lengths = [int(x) for x in item.split("+")]
        assert len(lengths) >= 2, f"{PACK_LAYOUTS} {item!r}: a layout packs at least two prompts"
        segments = pack_segments(lengths)
        check_pack(segments, chunk, ratios)
        if segments not in layouts:
            layouts.append(segments)
    layouts.sort(key=lambda s: (-len(s), -(s[-1][0] + s[-1][1])))
    return layouts


def match_pack_layout(lengths: list[int], layouts) -> tuple | None:
    """The first layout of ``layouts`` whose lengths are a prefix of ``lengths`` (prompt lengths in row order)."""
    for segments in layouts:
        k = len(segments)
        if len(lengths) >= k and all(int(lengths[i]) == segments[i][1] for i in range(k)):
            return segments
    return None


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

    def __init__(
        self,
        mesh_device,
        config,
        max_seq_len: int,
        chunk: int,
        layers: list[int],
        generic: bool = False,
        dspark: bool = False,
    ):
        """``dspark`` (generic only): keep the DSpark seeding's RoPE pair as a per-chunk host-filled input
        (``dspark_rope``), so the generic chunk graph seeds the rings (speculative decode)."""
        self.mesh_device, self.config = mesh_device, config
        self.sp, self.tp = tuple(mesh_device.shape)
        self.max_seq_len, self.chunk = max_seq_len, chunk
        self.generic = generic
        self.dspark = dspark and generic
        self.ratios = ratios = {config.compress_ratio(l) for l in layers}
        self.kinds = kinds = {(r > 0, 1) for r in ratios} | {(True, r) for r in ratios if r > 1}
        window = config.SLIDING_WINDOW
        self._window = {w: self._window_table(w) for w in {min(s, window - 1) for s in range(0, max_seq_len, chunk)}}
        self._query_first = self._query_first_table()
        self._rope, self._visibility = {}, {}
        self._segments = None  # the packed layout being run (generic states only, TT_V41_PACK)
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
        # the DSpark seed rows' fp32 RoPE pair (``TtV41DSpark.seed_rope``: the rows [a, b) of the chunk's last
        # ``window`` valid positions, <= window + 32 rows; the chunk's start is the only position input the seeding
        # has), host-filled per chunk as the bf16 tables are; ``dspark_rope`` slices the chunk's rows
        self._dspark_rope = (
            tuple(
                self._replicated(torch.zeros(1, 1, self.config.SLIDING_WINDOW + 32, 64), ttnn.float32) for _ in range(2)
            )
            if self.dspark
            else None
        )
        self._query_rows = rows
        self._chunk_start = None
        self._host_cache: dict[int, list] = {}
        self._chunk_masks = None  # per-chunk row-major masks shared by the index sources (begin_chunk_masks)
        # packed chunks (TT_V41_PACK): per ratio the per-query lower bound of the index mask
        # (the first compressed row of the query's segment, [1, 1, rows, 1] fp32 tiled like ``_visible``), the
        # window-row table per layout (``window_rows``) and the packed host fills per layout; ``_segments`` is the
        # layout of the chunk being run (None: a single prompt, the graph of before)
        self._pack = pack_enabled()
        self._base = {}
        if self._pack:
            self._base = {r: self._per_query_chip(torch.zeros(self.chunk, 1), ttnn.float32) for r in ratios if r > 0}
        self._pack_windows: dict[tuple, ttnn.Tensor] = {}
        self._pack_host: dict[tuple, list] = {}
        self._segments = None

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

    HOST_CACHE = 64  # chunk starts whose ``prepare_chunk_host`` tensors are kept (~2 MB each at chunk 5,120)

    def prepare_chunk_host(self, start: int) -> list[tuple[ttnn.Tensor, ttnn.Tensor]]:
        """The host tensors ``prepare_chunk`` copies into the chunk's input buffers, each with its buffer: RoPE
        rows, the pin page, the visible counts, the row indices. Depends on ``start`` only, so it can run ahead of
        the device (on the prefill's prefetch thread), and the last ``HOST_CACHE`` starts' tensors are kept (every
        prompt's first chunk shares one set; 10 ms to build)."""
        assert self.generic
        assert start % self.chunk == 0 or start == 0, start
        cached = self._host_cache.get(start)
        if cached is not None:
            return cached
        pairs = self._prepare_chunk_host(start)
        if len(self._host_cache) >= self.HOST_CACHE:
            self._host_cache.pop(next(iter(self._host_cache)))
        self._host_cache[start] = pairs
        return pairs

    def _prepare_chunk_host(self, start: int) -> list[tuple[ttnn.Tensor, ttnn.Tensor]]:
        chunk = self.chunk
        pairs = []
        for (compressed, stride), buffers in self._rope_buffers.items():
            positions = torch.arange(start, start + chunk, stride)
            for host, buffer in zip(cos_sin(self.config, compressed, positions), buffers):
                pairs.append((self._host_replicated(host, ttnn.bfloat16), buffer))
        first = torch.arange(self.sp * self.tp, dtype=torch.int32) * self._query_rows  # chip query offsets
        pin = torch.zeros(self.sp, self.tp, 1, 8, dtype=torch.int32)
        pin[..., 0] = (first + start).reshape(self.sp, self.tp, 1)
        pairs.append(
            (
                ttnn.from_torch(
                    pin,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, 1)),
                ),
                self._pin,
            )
        )
        positions = torch.arange(chunk, dtype=torch.int64) + start  # every query row of the chunk
        for r, buffer in self._visible.items():
            visible = ((positions + 1) // r).float().view(chunk, 1)
            host = visible.reshape(self.sp, self.tp, self._query_rows, 1)
            pairs.append(
                (
                    ttnn.from_torch(
                        host,
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, 1)),
                    ),
                    buffer,
                )
            )
        for r, buffer in self._row_index.items():
            rows = torch.arange(chunk // r, dtype=torch.int32) + start // r
            pairs.append((self._host_replicated(rows.view(1, 1, 1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), buffer))
        if self._dspark_rope is not None:
            pairs += self._dspark_rope_host(start, chunk)
        return pairs

    def _dspark_rope_host(self, start: int, length: int) -> list[tuple[ttnn.Tensor, ttnn.Tensor]]:
        """The seed rows' cos / sin of the chunk at ``start`` with ``length`` valid tokens (``TtV41DSpark.seed_rope``'s
        values, fp32, padded with zero rows to the buffer), each with its buffer."""
        _, a, b = seed_rows(length, self.config.SLIDING_WINDOW)
        rows = self._dspark_rope[0].shape[2]
        pairs = []
        for host, buffer in zip(cos_sin(self.config, False, torch.arange(start + a, start + b)), self._dspark_rope):
            padded = torch.zeros(1, 1, rows, host.shape[3])
            padded[:, :, : b - a] = host
            pairs.append((self._host_replicated(padded, ttnn.float32), buffer))
        return pairs

    def dspark_rope(self, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """The chunk's seed rows' RoPE pair (``[1, 1, b - a, 64]`` fp32 replicated, what ``TtV41DSpark.seed`` takes)
        for ``length`` valid tokens, sliced from the host-filled buffers (``prepare_chunk`` filled them for this
        chunk's start and length)."""
        assert self._dspark_rope is not None, "the tables were built without DSpark"
        _, a, b = seed_rows(length, self.config.SLIDING_WINDOW)
        return tuple(ttnn.slice(t, [0, 0, 0, 0], [1, 1, b - a, t.shape[3]]) for t in self._dspark_rope)

    def prepare_chunk(
        self, start: int, length: int, host: list | None = None, pairs: list | None = None, segments=None
    ) -> None:
        """Host-fill the chunk's input buffers: RoPE rows, the pin page, the visible counts, the row indices. Runs
        between a chunk's ``_stage`` and its forward (or replay); no device work beyond the copies (ordered on the
        command queue after the replays issued before them). ``host``: ``prepare_chunk_host(start)`` made ahead.
        ``pairs``: a list the (host, buffer) pairs are appended to instead of copied, for a batched upload
        (``TtV41Transformer._stage`` with ``TT_V41_PREFILL_HOST_BATCH=1``). ``segments``: a packed chunk's layout
        (``prepare_pack_host``; ``start`` is 0): the fills are per segment and the chunk's graph adds the index
        mask's lower bound and takes the layout's window rows."""
        assert self.generic
        if segments is not None:
            assert start == 0, start
            items = self.prepare_pack_host(segments) if host is None else host
        else:
            items = self.prepare_chunk_host(start) if host is None else host
            if self._dspark_rope is not None and length != self.chunk:  # a partial final chunk: its own seed rows
                # replace the full chunk's rope rows, so a batched upload writes each buffer once
                rope = self._dspark_rope
                items = [it for it in items if not any(it[1] is b for b in rope)] + self._dspark_rope_host(start, length)
        if pairs is not None:
            pairs.extend(items)
        else:
            for tensor, buffer in items:
                ttnn.copy_host_to_device_tensor(tensor, buffer)
        self._chunk_start = start
        self._segments = None if segments is None else tuple(segments)

    @property
    def segments(self):
        """The layout of the packed chunk being run (``prepare_chunk(..., segments=...)``), or None."""
        return self._segments

    # --- packed chunks (TT_V41_PACK) -------------------------------------------------------
    @staticmethod
    def _segment_starts(segments, chunk: int) -> torch.Tensor:
        """[chunk] int64: the start row of the segment holding each chunk row (the pad rows after the pack form a
        segment of their own at the pack's end: never read by a valid query)."""
        starts = torch.zeros(chunk, dtype=torch.int64)
        for offset, length in segments:
            starts[offset : offset + length] = offset
        end = segments[-1][0] + segments[-1][1]
        starts[end:] = end
        return starts

    def prepare_pack_host(self, segments) -> list[tuple[ttnn.Tensor, ttnn.Tensor]]:
        """``prepare_chunk_host`` of a packed chunk at start 0 (kept per layout): the RoPE rows at each row's
        position within its segment (the compressed tables at the group-first rows' local positions, whole groups
        since segments are ratio-aligned), the pin page and the visible counts as at start 0 (a query's visible
        count ``(i + 1) // r`` is unchanged: its segment's rows are at ``[offset / r, (i + 1) // r)``), the lower
        bound ``offset / r`` per query (``_base``) and the row indices as at start 0 (the pack's compressed rows
        are contiguous)."""
        assert self.generic and self._pack, "packed chunks need TT_V41_PACK=1 at the state's construction"
        segments = tuple(segments)
        cached = self._pack_host.get(segments)
        if cached is not None:
            return cached
        chunk = self.chunk
        starts = self._segment_starts(segments, chunk)
        local = torch.arange(chunk) - starts
        pairs = []
        for (compressed, stride), buffers in self._rope_buffers.items():
            positions = local[::stride]
            for host, buffer in zip(cos_sin(self.config, compressed, positions), buffers):
                pairs.append((self._host_replicated(host, ttnn.bfloat16), buffer))
        single = {id(buffer): tensor for tensor, buffer in self._prepare_chunk_host(0)}  # pin, visible, row index
        pairs.append((single[id(self._pin)], self._pin))
        for r, buffer in self._visible.items():
            pairs.append((single[id(buffer)], buffer))
        for r, buffer in self._base.items():
            base = (starts // r).float().view(chunk, 1).reshape(self.sp, self.tp, self._query_rows, 1)
            pairs.append(
                (
                    ttnn.from_torch(
                        base,
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, 1)),
                    ),
                    buffer,
                )
            )
        for r, buffer in self._row_index.items():
            pairs.append((single[id(buffer)], buffer))
        self._pack_host[segments] = pairs
        return pairs

    def _pack_window_table(self, segments):
        """``_window_table(0)`` of a layout: rows ``WINDOW_SLOT + i + o`` of query i, -1 where ``i + o`` lies before
        the query's segment (the previous prompt's last rows are not its window)."""
        return self._per_query_chip(self.pack_window_rows_host(segments), ttnn.int32)

    def prepare_layouts(self, layouts) -> None:
        """Build the resident window table and the host fills of every layout now (a served packed state: the
        tables are persistent device tensors, so they exist before any capture)."""
        for segments in layouts:
            segments = tuple(segments)
            if segments not in self._pack_windows:
                self._pack_windows[segments] = self._pack_window_table(segments)
            self.prepare_pack_host(segments)

    def pack_window_rows_host(self, segments) -> torch.Tensor:
        """The host form of the layout's window table ([chunk, window] int32, chunk query order)."""
        window = self.config.SLIDING_WINDOW
        i = torch.arange(self.chunk).view(-1, 1)
        offset = torch.arange(-(window - 1), 1).view(1, -1)
        starts = self._segment_starts(tuple(segments), self.chunk).view(-1, 1)
        return torch.where(i + offset >= starts, WINDOW_SLOT + i + offset, -1).to(torch.int32)

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
        if self._segments is not None:  # packed: the columns before the query's segment are not its rows either
            beyond = ttnn.logical_or(beyond, ttnn.lt(iota, self._base[ratio]))
        return ttnn.typecast(ttnn.where(beyond, float("-inf"), 0.0), ttnn.bfloat16)

    # --- the masks built once per chunk (TT_V41_PREFILL_MASK_ONCE) ------------------------
    def begin_chunk_masks(self) -> None:
        """Generic: from here to ``end_chunk_masks`` the index sources share one row-major mask per (ratio, width)
        (``chunk_mask``), built at the first use in this chunk's graph. The mask depends on the chunk only through
        the visible counts (a device input), so it is the same tensor for every index source of a ratio; the score
        kernel writes row-major and ``ttnn.add`` runs row-major, so the per-layer tilize of the score, the mask
        build and the untilize of the sum go: one add a layer."""
        assert self.generic
        self._chunk_masks = {}

    def end_chunk_masks(self) -> None:
        if self._chunk_masks:
            for t in self._chunk_masks.values():
                ttnn.deallocate(t)
        self._chunk_masks = None

    def chunk_mask(self, ratio: int, width: int):
        """The chunk's shared row-major mask of this ratio and score width (``mask``'s values), or None outside
        ``begin_chunk_masks`` .. ``end_chunk_masks`` (the caller builds its own, tiled, as before)."""
        if self._chunk_masks is None:
            return None
        mask = self._chunk_masks.get((ratio, width))
        if mask is None:
            # bf16 from the comparison on (no fp32 intermediate to typecast), untilized once for the chunk
            iota = ttnn.slice(self._iota[ratio], [0, 0, 0, 0], [1, 1, 1, width])
            beyond = ttnn.ge(iota, self._visible[ratio], dtype=ttnn.bfloat16)
            transient = [beyond]
            if self._segments is not None:  # packed: a two-sided bound, [base, visible) per query
                before = ttnn.lt(iota, self._base[ratio], dtype=ttnn.bfloat16)
                beyond = ttnn.logical_or(beyond, before)
                transient += [before, beyond]
            tiled = ttnn.where(beyond, float("-inf"), 0.0)
            mask = ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT)
            for t in transient + [tiled]:  # not the iota slice: at the widest bucket it is the table itself
                ttnn.deallocate(t)
            self._chunk_masks[(ratio, width)] = mask
        return mask

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
        """Per-chip [1, 1, chunk/(sp*tp), window] int32 tiled rows of the chunk's causal windows (-1: none); a
        packed chunk's are its layout's (``_pack_window_table``, resident per layout)."""
        if self._segments is not None:
            assert start == 0, start
            table = self._pack_windows.get(self._segments)
            if table is None:
                table = self._pack_windows[self._segments] = self._pack_window_table(self._segments)
            return table
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
        tiled_index_k: bool = False,
        dspark: bool = False,
        pack_segments: int = 0,
    ):
        """``generic``: position-generic chunk graphs (``V41ChunkTables`` generic mode; ``V41GenericPrefillTrace``):
        the cache writes scatter by an uploaded row index instead of a start-dependent slice. ``tiled_index_k``
        (generic only; ``TT_V41_PREFILL_MASK_ONCE``): keep a tiled twin of each index-K cache with one zero tile past
        its rows (``index_k_tiled``), written by ``write_compressed`` beside the row-major rows, so an index source's
        score kernel reads a tile-aligned prefix of it (one slice) instead of slicing, tilizing and padding the
        row-major rows at every chunk. The row-major cache stays the one the candidate scorer
        and the decode join read. ``dspark`` (generic only): the tables keep the DSpark seed rows' RoPE pair as a
        host-filled chunk input, so a generic chunk graph seeds the rings (``V41ChunkTables.dspark_rope``).
        ``pack_segments``: allocate the window carries of that many packed segments now (a served packed state:
        every persistent tensor before any capture) instead of at a segment's first
        packed chunk."""
        assert kv_format in self.FORMATS, f"V4.1 KV format must be one of {self.FORMATS}, got {kv_format}"
        self.mesh_device = mesh_device
        self.config = config
        layout = V41MeshLayout.of(mesh_device)
        self.geometry = V41CacheGeometry(config, max_seq_len, chunk, layout.sp)
        layout.check_chunk(chunk)
        self.layers = list(layers)
        self.tables = V41ChunkTables(
            mesh_device, config, max_seq_len, chunk, self.layers, generic=generic, dspark=dspark
        )
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
        self.index_k_tiled = {}
        if generic and tiled_index_k:
            self.index_k_tiled = {
                l: replicated_zeros(
                    mesh_device, g.compressed_rows(config.compress_ratio(l)) + 32, idim, ttnn.bfloat16, ttnn.TILE_LAYOUT
                )
                for l in sources
            }
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
        # packed chunks (TT_V41_PACK): each segment's own window carries, per segment index then layer, allocated
        # at a segment's first packed chunk (``segment_window_carry``); the join of a segment takes them instead of
        # ``window_carry`` (which holds the pack's last segment's, as a single prompt's would)
        self.segment_carry: dict[int, dict[int, ttnn.Tensor]] = {}
        for seg in range(pack_segments):
            for layer in self.layers:
                self.segment_window_carry(seg, layer)
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

                encoded = self._encode(rows, fmt)
                scatter_rows(encoded, cache, self.tables.row_index(ratio), dst_base=offset - first, rows=count)
                if cache is self.index_k[source] and source in self.index_k_tiled:
                    scatter_rows(encoded, self.index_k_tiled[source], self.tables.row_index(ratio), rows=count)
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
        assert source not in self.index_k_tiled, "the tiled index-K twin follows write_compressed only (prefill chunks)"
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
        assert source not in self.index_k_tiled, "the tiled index-K twin follows write_compressed only (prefill chunks)"
        for cache, offset in ((self.kv[source], self.geometry.kv_row_of_compressed(0)), (self.index_k[source], 0)):
            row = ttnn.slice(cache, [0, 0, offset + src, 0], [1, 1, offset + src + 1, cache.shape[3]])
            self._write_rows(cache, row, offset + dst)
            ttnn.deallocate(row)

    def segment_window_carry(self, segment: int, layer: int) -> ttnn.Tensor:
        """The window carry buffer of packed segment ``segment`` for ``layer`` (WINDOW_SLOT rows in the layer's KV
        format), allocated at first use."""
        carries = self.segment_carry.setdefault(segment, {})
        if layer not in carries:
            fmt = self.layer_kv_format(layer)
            carries[layer] = replicated_zeros(
                self.mesh_device, WINDOW_SLOT, fmt.storage_width(self.kv_geometry), fmt.storage_dtype
            )
        return carries[layer]

    def update_window_carry(self, layer: int, length: int):
        """After ``layer``'s attention: its carry becomes the last WINDOW_SLOT rows ending at the chunk's
        last valid token (rows [length, length + WINDOW_SLOT) of the window region). A packed chunk also writes
        each segment's carry (the WINDOW_SLOT rows ending at the segment's last token) into the segment's own
        buffer."""
        src = self.kv_tensor(layer)
        carry = ttnn.slice(src, [0, 0, length, 0], [1, 1, length + WINDOW_SLOT, src.shape[3]])
        self._write_rows(self.window_carry[layer], carry, 0)
        segments = self.tables.segments
        if segments is not None:
            for g, (offset, n) in enumerate(segments):
                end = offset + n
                rows = (
                    carry if end == length else ttnn.slice(src, [0, 0, end, 0], [1, 1, end + WINDOW_SLOT, src.shape[3]])
                )
                self._write_rows(self.segment_window_carry(g, layer), rows, 0)

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
