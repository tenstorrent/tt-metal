# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 decode pool: the caches of the users a mesh row decodes together
(``tt/v41/batch_decode.py``).

The batched decode step runs one 32-row tile per chip, row ``u`` the user in slot ``u`` of that chip's mesh
row
(``users_per_rank`` slots, the rest of the tile padding): the SP ranks hold different users and TP splits the
hidden dim and heads as in prefill, so a user's caches live on the four chips of one mesh row. Per KV source
the
rank's pool is one row-major tensor of ``users_per_rank`` slots, each laid out as::

    rows [ring_base(l), ring_base(l) + 128)   layer l's window ring: 128 slots, one per consumer layer of the
    source
    rows [comp_base, comp_base + cap)         compressed KV row j at comp_base + j, cap = max_seq_len / ratio

so one ``sparse_sdpa`` over the pool reads a user's window and selected rows with per-row indices (row ``u``
of the
index rows offset by ``u * slot_rows``). A window ring is indexed by the global step: the step writes every
user's
window row at ring slot ``step % 128`` and the window always covers the whole ring, so the window index rows
are a
constant per layer and no carry is shifted. Index keys are a ``[users_per_rank, 1, cap, idim]`` tensor per
source
(slot ``u`` selected by ``cache_batch_idx`` in the indexer), ratio-0 layers share a ring-only tensor.

Rows reach the pool through :func:`scatter_rows` (``kernels/scatter_rows.cpp``): the destination row of each
source
row comes from a device tensor, so the captured step writes where the step's inputs say, and a join copies a
prefill state's rows into one rank's slot with the other ranks' indices set to the sentinel. The destination
may
store its logical rows as row-major rows, as tiled bf16 (each row lands in its tile's faces, so a pool the
indexer
scores stays tiled and is never converted again) or as block pages of ``BLOCK_ROWS`` rows (the pages the
candidate
scorer gathers); a user's rows are addressed as logical rows ``u * (cap + 32) + t`` in every form.
"""

import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.cache import WINDOW_SLOT, V41PrefillState, replicated_zeros
from models.demos.deepseek_v3_d_p.tt.v41.ccl import fabric_num_links
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import _cores, _split
from models.demos.deepseek_v3_d_p.tt.v41.layout import SP_AXIS, V41MeshLayout
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, MlaKvCacheGeometry

ROWS = 32  # the decode tile
RING = 128  # window ring slots per layer: the sliding window (SLIDING_WINDOW <= 128, asserted)
SENTINEL = 0xFFFFFFFF
_KERNEL = "models/demos/deepseek_v3_d_p/tt/v41/kernels/scatter_rows.cpp"


BLOCK_ROWS = 8  # logical rows per block page (CANDIDATE_BLOCK_SIZE: the candidate scorer reads a block as one page)
RESTORE_PIECE = 65536  # the restore's piece (tokens; at most this many rows a source per gather)
# TT_V41_RESTORE_CCL=own: the restore's gathers on their own global semaphores, not the
# TT_CCL pool used by the captured steps' gathers
RESTORE_CCL = os.environ.get("TT_V41_RESTORE_CCL", "shared")


def _dst_mode(dst, src_width: int, row_bytes: int) -> int:
    """How ``dst`` stores a logical row of ``src_width`` elements (``row_bytes``): 0 row-major rows, 1 tiled
    bf16,
    2 block pages (``BLOCK_ROWS`` rows a page, compared in bytes: a page tensor may carry another dtype)."""
    if dst.layout == ttnn.TILE_LAYOUT:
        assert dst.dtype == ttnn.bfloat16 and dst.shape[3] == src_width, (dst.dtype, tuple(dst.shape), src_width)
        return 1
    if dst.shape[3] == src_width:
        return 0
    assert dst.shape[3] * dst.element_size() == BLOCK_ROWS * row_bytes, (tuple(dst.shape), dst.dtype, row_bytes)
    return 2


def scatter_rows(
    src,
    dst,
    idx,
    src_base: int = 0,
    dst_base: int = 0,
    stride: int = 0,
    rows: int | None = None,
    src_pages: bool = False,
):
    """Copy row ``src_base + i`` of ``src`` to logical row ``dst_base + i * stride + idx[i]`` of ``dst`` for
    ``i``
    below ``rows`` (default: ``idx``'s length), skipping the rows whose ``idx`` is the sentinel; in place on
    ``dst``.
    ``src`` row-major DRAM-interleaved; ``idx`` ``[1, 1, 1, n]`` uint32 row-major (per chip: a sharded index
    tensor
    scatters differently on each chip). ``dst`` DRAM-interleaved in one of three forms of the same logical
    rows
    (``kernels/scatter_rows.cpp``): row-major rows of ``src``'s width; tiled bf16 of that width (the row lands
    in
    its tile's faces); or row-major block pages of ``BLOCK_ROWS * width`` (row ``r`` at page ``r //
    BLOCK_ROWS``,
    the layout ``candidate_scores`` reads). ``src_pages``: ``src`` holds its rows as block pages of
    ``BLOCK_ROWS``
    rows too (``[1, 1, n / BLOCK_ROWS, BLOCK_ROWS * width]``; ``dst`` then row-major rows: the restore of a
    parked
    prefix). One data-movement op."""
    for t, what in ((src, "src"), (dst, "dst"), (idx, "idx")):
        memory = t.memory_config()
        assert memory.buffer_type == ttnn.BufferType.DRAM and not memory.is_sharded(), f"scatter_rows {what}: DRAM"
        if what != "dst":
            assert t.layout == ttnn.ROW_MAJOR_LAYOUT, f"scatter_rows {what}: row-major, got {t.layout}"
    assert idx.dtype == ttnn.uint32 and idx.shape[2] == 1, tuple(idx.shape)
    if src_pages:
        width, row_bytes = dst.shape[3], dst.shape[3] * dst.element_size()
        assert dst.layout == ttnn.ROW_MAJOR_LAYOUT and src.shape[3] * src.element_size() == BLOCK_ROWS * row_bytes, (
            tuple(src.shape),
            src.dtype,
            tuple(dst.shape),
            dst.dtype,
        )
        src_rows = src.shape[2] * BLOCK_ROWS
    else:
        width, row_bytes = src.shape[3], src.shape[3] * src.element_size()
        src_rows = src.shape[2]
    mode = _dst_mode(dst, width, row_bytes)
    assert mode == 2 or src_pages or dst.dtype == src.dtype, (src.dtype, dst.dtype)  # pages carry bytes
    assert mode != 1 or row_bytes % 64 == 0, row_bytes  # whole tiles across
    n = idx.shape[3] if rows is None else rows
    assert 0 < n <= idx.shape[3] and src_base + n <= src_rows, (n, tuple(idx.shape), src_base, tuple(src.shape))
    coords, cores = _cores(src.device(), min(n, 64))
    args = ttnn.RuntimeArgs()
    for (cx, cy), (first, count) in zip(coords, _split(n, len(coords))):
        args[cx][cy] = [
            src.buffer_address(),
            dst.buffer_address(),
            idx.buffer_address(),
            src_base,
            dst_base,
            stride,
            first,
            count,
            n,
        ]
    accessors = (
        ttnn.TensorAccessorArgs(src).get_compile_time_args()
        + ttnn.TensorAccessorArgs(dst).get_compile_time_args()
        + ttnn.TensorAccessorArgs(idx).get_compile_time_args()
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        compile_time_args=[0, row_bytes, mode, BLOCK_ROWS, 2 if src_pages else 0] + accessors,
        runtime_args=args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    per_core = max(count for _, count in _split(n, len(coords))) + 16  # a core's index window, line-aligned
    buffers = max(2 * ((row_bytes + 63) // 64 * 64), (BLOCK_ROWS * row_bytes + 63) // 64 * 64 if src_pages else 0)
    scratch = 64 + (per_core * 4 + 63) // 64 * 64 + buffers
    page = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)
    cbs = [ttnn.CBDescriptor(total_size=scratch, core_ranges=cores, format_descriptors=[page])]
    return ttnn.generic_op([src, idx, dst], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs))


@dataclass(frozen=True)
class SlotLayout:
    """One KV source's slot: its consumer layers' rings, then the compressed rows."""

    layers: tuple  # consumer layers in order (ring index = position in this tuple)
    cap: int  # compressed rows (0 for the ratio-0 tensor)
    ring_slots: int = RING  # ring slots per consumer layer (RING; the verify rows of speculative decode take more)

    @property
    def comp_base(self) -> int:
        return self.ring_slots * len(self.layers)

    @property
    def rows(self) -> int:
        return self.comp_base + self.cap

    def ring_base(self, layer: int) -> int:
        return self.ring_slots * self.layers.index(layer)


class V41DecodePool:
    def __init__(
        self,
        mesh_device,
        config,
        max_seq_len: int,
        layers: list[int],
        users_per_rank: int,
        kv_format,
        ring_slots: int = RING,
        parking: bool = False,
    ):
        """The pool of ``users_per_rank`` slots per mesh row for the layer schedule ``layers`` (all of a
        model's
        layers in serving) up to ``max_seq_len`` positions per user, compressed rows in ``kv_format``.
        ``ring_slots`` (>= RING): the window ring's slots per layer; with ``position_rings`` set, a position's
        slot is
        ``position % ring_slots`` (the join places the carry there) instead of the global step's (speculative
        decode's verify rows, ``tt/v41/spec_verify.py``: rows of one slot at consecutive positions in one
        step).
        ``parking`` (``TT_V41_DEFER_FINAL``): one replicated copy of the 40 window carries per
        slot, the state a parked request's final chunk restores from (``park_carries`` / ``unpark_carries``),
        allocated here, before any capture; the slot's rings are not a durable copy, since every decode step
        writes
        every slot's ring at ``step % RING``, live or idle."""
        assert 0 < users_per_rank <= ROWS, users_per_rank
        assert config.SLIDING_WINDOW <= RING and WINDOW_SLOT == RING
        assert ring_slots >= RING, ring_slots
        self.ring_slots = ring_slots
        self.position_rings = False
        self.mesh_device, self.config = mesh_device, config
        self.layout = V41MeshLayout.of(mesh_device)
        self.sp, self.tp = self.layout.sp, self.layout.tp
        self.max_seq_len, self.layers, self.users = max_seq_len, list(layers), users_per_rank
        self.kv_format = kv_format
        self.kv_geometry = MlaKvCacheGeometry(latent_dim=config.HEAD_DIM, rope_dim=0)  # as V41PrefillState
        idim = config.INDEX_HEAD_DIM
        sources = [l for l in config.KV_SOURCE_LAYERS if l in self.layers]
        self.slots: dict[int, SlotLayout] = {}  # source (or -1 for ratio 0) -> layout
        for src in sources:
            ratio = config.compress_ratio(src)
            consumers = tuple(l for l in self.layers if config.compress_ratio(l) and config.kv_source(l) == src)
            self.slots[src] = SlotLayout(consumers, max_seq_len // ratio, ring_slots)
        swa = tuple(l for l in self.layers if config.compress_ratio(l) == 0)
        if swa:
            self.slots[-1] = SlotLayout(swa, 0, ring_slots)
        self.kv = {}
        # the index keys, [users, 1, cap + 32, idim] bf16 tiled (slot u = cache_batch_idx u of the dense
        # scorer), kept
        # tiled by every write; and, for the candidate source, the same rows as block pages [1, 1, users *
        # (cap + 32)
        # / BLOCK_ROWS, BLOCK_ROWS * idim] bf16 row-major (the pages the candidate scorer gathers). The
        # logical row of
        # user u's compressed row t is u * (cap + 32) + t in both.
        self.index_k = {}
        self.index_k_blocks = {}
        for src, slot in self.slots.items():
            fmt = kv_format if src >= 0 else MlaKvCacheFormat.BF16_RM
            w = fmt.storage_width(self.kv_geometry) if src >= 0 else config.HEAD_DIM
            self.kv[src] = replicated_zeros(mesh_device, users_per_rank * slot.rows, w, fmt.storage_dtype)
            if src >= 0:
                # cap + 32 rows per slot: the indexer's causal window starts past the slot's rows, the last
                # tile
                # (never written, zero keys) is what it scores there
                assert (slot.cap + 32) % 32 == 0, slot.cap  # a user's rows start on a tile and a block page
                rows = users_per_rank * (slot.cap + 32)
                tiled = replicated_zeros(mesh_device, rows, idim, ttnn.bfloat16, ttnn.TILE_LAYOUT)
                self.index_k[src] = ttnn.reshape(tiled, (users_per_rank, 1, slot.cap + 32, idim))
                if src == config.CANDIDATE_SOURCE_LAYER:
                    self.index_k_blocks[src] = replicated_zeros(
                        mesh_device, rows // BLOCK_ROWS, BLOCK_ROWS * idim, ttnn.bfloat16
                    )
        # fp32 (kv, score) projections of every ratio-2 source's last token per slot row (the compressor
        # carry)
        self.carry = {
            src: tuple(self._replicated(torch.zeros(1, 1, ROWS, config.HEAD_DIM), ttnn.float32) for _ in range(2))
            for src in sources
            if config.compress_ratio(src) > 1
        }
        self._window = {l: self._window_rows(l) for l in self.layers}
        self._offsets = {src: self._user_offsets(src) for src in sources}
        self._init_ccl(mesh_device)
        self.step = 0  # the global step: the ring slot every user's next window row goes to is step % RING
        # the parking buffers: per slot, per layer, the carry's shape and format (as
        # V41PrefillState.window_carry)
        self.parked: dict[int, dict[int, ttnn.Tensor]] = {}
        if parking:
            for slot in range(self.sp * users_per_rank):
                self.parked[slot] = {}
                for l in self.layers:
                    fmt = kv_format if config.compress_ratio(l) else MlaKvCacheFormat.BF16_RM
                    self.parked[slot][l] = replicated_zeros(
                        mesh_device, WINDOW_SLOT, fmt.storage_width(self.kv_geometry), fmt.storage_dtype
                    )
        # the restore's identity index page, allocated here, before any capture: a persistent tensor allocated
        # after a capture may be overwritten by that capture's replay (the second
        # restore of a cohort hung on an index page the first restore had allocated lazily and the final
        # chunk's
        # and the join's replays had since clobbered)
        self.restore_piece = RESTORE_PIECE
        self._iota = None
        if parking:
            host = torch.arange(RESTORE_PIECE, dtype=torch.int32).view(1, 1, 1, RESTORE_PIECE)
            self._iota = self._replicated(host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def _replicated(self, host, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            host,
            device=self.mesh_device,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def per_rank(self, host: torch.Tensor, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=True):
        """``host`` ``[sp, ...]`` -> each mesh row its slice (replicated over TP), as ``[1, ...]`` per chip."""
        assert host.shape[0] == self.sp, (tuple(host.shape), self.sp)
        return ttnn.from_torch(
            host,
            device=self.mesh_device if device else None,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, None)),
        )

    def _init_ccl(self, mesh_device) -> None:
        self.mesh_device, self.sp = mesh_device, tuple(mesh_device.shape)[0]
        self._ccl = get_tt_ccl(mesh_device) if self.sp > 1 else None
        self._num_links = fabric_num_links()
        self._sp_topology = per_axis_topology()[SP_AXIS]
        # TT_V41_RESTORE_CCL=own: the restore's own double-buffered gather and barrier semaphores (TT_CCL's
        # form)
        self._own_semaphores = None
        self._own_idx = 0
        if self._ccl is not None and RESTORE_CCL == "own":
            crs = self._ccl.sub_device_crs
            self._own_semaphores = [
                (
                    [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)],
                    ttnn.create_global_semaphore(mesh_device, crs, 0),
                )
                for _ in range(2)
            ]

    def _semaphores(self):
        """The (gather handles, barrier handle) of the next gather: the pool's own if
        ``TT_V41_RESTORE_CCL=own``,
        else TT_CCL's cycled pair for the SP axis."""
        if self._own_semaphores is not None:
            pair = self._own_semaphores[self._own_idx]
            self._own_idx = (self._own_idx + 1) % 2
            return pair
        return (
            self._ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
            self._ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=SP_AXIS),
        )

    def gather_sp(self, t, dim: int):
        """Concatenate ``t`` over the mesh rows along ``dim`` (rank order) on every chip."""
        if self._ccl is None:
            return t
        ag, barrier = self._semaphores()
        return ttnn.experimental.all_gather_async(
            t,
            dim=dim,
            multi_device_global_semaphore=ag,
            barrier_semaphore=barrier,
            num_links=self._num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self._sp_topology,
            cluster_axis=SP_AXIS,
        )

    # --- addressing ----------------------------------------------------------------------------------------
    def source_of(self, layer: int) -> int:
        ratio = self.config.compress_ratio(layer)
        return self.config.kv_source(layer) if ratio else -1

    def kv_tensor(self, layer: int):
        return self.kv[self.source_of(layer)]

    def slot_rows(self, layer: int) -> int:
        return self.slots[self.source_of(layer)].rows

    def ring_row(self, layer: int) -> int:
        """Row of ring slot 0 of ``layer`` in user 0's slot (user u: + u * slot_rows)."""
        return self.slots[self.source_of(layer)].ring_base(layer)

    def comp_row(self, source: int) -> int:
        return self.slots[source].comp_base

    def _window_rows(self, layer: int):
        """``sparse_sdpa`` index rows ``[1, 1, ROWS, RING]`` uint32 row-major of ``layer``'s window: row u (a
        user)
        its ring's rows; pad rows read user 0's ring."""
        base = self.ring_row(layer)
        rows = torch.arange(RING, dtype=torch.int32).view(1, -1) + base
        users = torch.arange(ROWS, dtype=torch.int32).clamp(max=self.users - 1).view(-1, 1) * self.slot_rows(layer)
        return self._replicated((rows + users)[None, None].contiguous(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def window_rows(self, layer: int):
        return self._window[layer]

    def _user_offsets(self, source: int):
        users = torch.arange(ROWS, dtype=torch.int32).clamp(max=self.users - 1) * self.slots[source].rows
        rows = (users + self.comp_row(source)).view(1, 1, ROWS, 1).expand(1, 1, ROWS, self.config.INDEX_TOPK)
        return self._replicated(rows.contiguous(), ttnn.int32)

    def user_offsets(self, source: int):
        """``[1, 1, ROWS, INDEX_TOPK]`` int32 tiled: row u's offset of its compressed row 0 in the KV pool of
        ``source``, on every column."""
        return self._offsets[source]

    # --- joining
    # ----------------------------------------------------------------------------------------------
    def write_index_keys(
        self,
        source: int,
        keys,
        idx,
        dst_base: int = 0,
        stride: int = 0,
        rows: int | None = None,
        src_base: int = 0,
    ):
        """``scatter_rows`` of row-major index keys (from row ``src_base``) into every representation of
        ``source``'s pool (the tiled slots, and the block pages of the candidate source): logical row
        ``dst_base + i * stride + idx[i]``."""
        scatter_rows(keys, self.index_k[source], idx, src_base=src_base, dst_base=dst_base, stride=stride, rows=rows)
        if source in self.index_k_blocks:
            scatter_rows(
                keys, self.index_k_blocks[source], idx, src_base=src_base, dst_base=dst_base, stride=stride, rows=rows
            )

    @staticmethod
    def check_segment(state: V41PrefillState, segment, ratios) -> tuple[int, int, int]:
        """A packed segment ``(offset, length, g)`` of ``state`` (``TtV41Transformer.prefill_packed``):
        ratio-aligned, inside the pack, its carries allocated."""
        offset, length, g = segment
        assert 0 <= offset and 0 < length and offset + length <= state.start, (segment, state.start)
        for ratio in ratios:
            assert ratio <= 1 or (offset % ratio == 0 and length % ratio == 0), (segment, ratio)
        assert g in state.segment_carry, f"segment {g}: no window carries in the state ({sorted(state.segment_carry)})"
        return offset, length, g

    def join_inputs(self, slot: int, state: V41PrefillState, width: int | None = None, segment=None) -> dict:
        """The host index tensors of ``join``: per rank (the slot's rank gets the rows, the others the
        sentinel)
        the ring slots of the carry's 128 rows (position ``state.start - 128 + k`` at ring slot ``(step + k) %
        RING``, so the next step's token, at ``step % RING``, replaces position ``start - 128``), per ratio
        the
        compressed rows written so far (sentinel past them), the slot's row of the carry buffers. ``width``
        (tokens, default the pool's ``max_seq_len``; at least ``state.start``): the compressed index covers
        ``width // ratio`` rows, so a join of a short prompt uploads a short index.
        ``segment`` ``(offset, length, g)``: one prompt of a packed chunk: the compressed
        index covers its ``length // ratio`` rows (``width`` at least ``length``) and the carry row is the
        sentinel
        (segments are ratio-aligned: no trailing token)."""
        rank, u = divmod(slot, self.users)
        assert 0 <= rank < self.sp, slot
        width = self.max_seq_len if width is None else width
        ratios = sorted({self.config.compress_ratio(src) for src in self.slots if src >= 0})
        if segment is None:
            length = state.start
            carried = any(c is not None for c in state.compressor_carry.values())  # an odd prompt's trailing token
        else:
            _, length, _ = self.check_segment(state, segment, ratios)
            carried = False
        assert length <= width <= self.max_seq_len, (length, width, self.max_seq_len)

        def index(n: int, values: torch.Tensor):
            host = torch.full((self.sp, 1, 1, n), -1, dtype=torch.int32)  # -1: the sentinel as uint32
            host[rank, 0, 0, : values.numel()] = values.to(torch.int32)
            return self.per_rank(host, ttnn.uint32, device=False)

        # the carry's row k is position length - RING + k: at ring slot (step + k) % RING, or, with position
        # rings,
        # at the position's own slot (a segment's positions are its own 0 .. length - 1)
        first = length - RING if self.position_rings else self.step
        inputs = {
            "ring": index(RING, (torch.arange(RING) + first) % self.ring_slots),
            "row": index(1, torch.tensor([u])) if carried else index(1, torch.zeros(0)),
        }
        for ratio in ratios:
            inputs[f"comp{ratio}"] = index(width // ratio, torch.zeros(length // ratio))
        return inputs

    def join_ops(self, slot: int, state: V41PrefillState, idx: dict, segment=None) -> None:
        """The device ops of ``join`` on the index buffers ``idx`` (as ``join_inputs``, on device): the same
        graph
        for every state of the geometry and every step (capturable per state, slot and index width; a
        segment's
        per state, offset, slot and width: its source rows start at ``offset / ratio`` and its window carries
        are
        the segment's own, ``state.segment_carry[g]``; no compressor carry)."""
        assert state.compressed_kv_format == self.kv_format, (state.compressed_kv_format, self.kv_format)
        assert state.geometry.max_seq_len <= self.max_seq_len, (state.geometry.max_seq_len, self.max_seq_len)
        cfg, g = self.config, state.geometry
        u = slot % self.users
        offset, carries = 0, state.window_carry
        if segment is not None:
            ratios = {cfg.compress_ratio(src) for src in self.slots if src >= 0}
            offset, _, seg = self.check_segment(state, segment, ratios)
            carries = state.segment_carry[seg]
        for src in state.kv:
            ratio = cfg.compress_ratio(src)
            comp, cap = idx[f"comp{ratio}"], self.slots[src].cap
            first = offset // ratio  # the segment's first compressed row of the state
            # the index width bounds the rows copied (the sentinel past the valid ones), and the state's rows
            # after
            # the segment's first bound them too
            n = min(cap, g.compressed_rows(ratio) - first, comp.shape[3])
            base = u * self.slots[src].rows + self.comp_row(src)
            scatter_rows(
                state.kv[src],
                self.kv[src],
                comp,
                src_base=g.kv_row_of_compressed(first),
                dst_base=base,
                stride=1,
                rows=n,
            )
            self.write_index_keys(
                src, state.index_k[src], comp, dst_base=u * (cap + 32), stride=1, rows=n, src_base=first
            )
        assert carries[self.layers[0]].shape[2] == RING
        for layer in self.layers:
            base = u * self.slot_rows(layer) + self.ring_row(layer)
            scatter_rows(carries[layer], self.kv_tensor(layer), idx["ring"], dst_base=base)
        if segment is not None:
            return  # a segment is whole ratio groups: no compressor carry to place
        for src, dst_pair in self.carry.items():
            # the state's carry buffers (stale after a prefill of complete groups: then idx["row"] is the
            # sentinel
            # and the slot's first step, which starts a group, writes the carry)
            for t, dst in zip(state.carry_buffers[src], dst_pair):
                assert t.shape[2] == 1, tuple(t.shape)  # a prefill's row
                src_rows = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
                dst_rows = ttnn.to_layout(dst, ttnn.ROW_MAJOR_LAYOUT)
                scatter_rows(src_rows, dst_rows, idx["row"])
                ttnn.copy(ttnn.to_layout(dst_rows, ttnn.TILE_LAYOUT), dst)
                ttnn.deallocate(src_rows)
                ttnn.deallocate(dst_rows)

    # --- parking and restoring (TT_V41_DEFER_FINAL) ----------------------------------------
    def park_carries(self, slot: int, state: V41PrefillState) -> None:
        """Copy ``state``'s window carries into ``slot``'s parking buffers (after the slot's join)."""
        for layer, dst in self.parked[slot].items():
            ttnn.copy(state.window_carry[layer], dst)

    def unpark_carries(self, slot: int, state: V41PrefillState) -> None:
        for layer, src in self.parked[slot].items():
            ttnn.copy(src, state.window_carry[layer])

    def iota(self, n: int):
        """The ``[1, 1, 1, RESTORE_PIECE]`` uint32 row-major ``0 .. RESTORE_PIECE - 1`` page on every chip
        (the
        restore's destination index; ``scatter_rows(..., rows=n)`` reads its first ``n`` entries)."""
        assert self._iota is not None and 0 < n <= self.restore_piece, (n, self.restore_piece)
        return self._iota

    def restore_ops(
        self, slot: int, state: V41PrefillState, length: int, piece: int = RESTORE_PIECE, probe=None
    ) -> dict:
        """The device ops of a restore: ``slot``'s first ``length`` tokens of compressed KV and index keys
        (the
        bulk chunks of a parked request, or the width bucket holding them: the rows past the prefix are the
        slot's stale rows, which nothing reads before the final chunk overwrites its own and attention masks
        the
        rest, as the serial path's state keeps stale rows past ``start``) back into ``state``'s caches on
        every
        chip, in pieces of at most ``piece`` tokens. Per source and piece: the slot's rows sliced on every
        rank
        (each rank its own slot ``u``), gathered over the SP axis (``[1, 1, sp * n, W]``: the owning rank's
        segment is at ``rank * n``), and the owner's segment scattered into the state (KV: row-major into
        row-major; index keys: the tiled rows into ``index_k_tiled`` and, untilized, into the row-major
        ``index_k``). Then the window carries from the parking buffers. The host fields are ``restore_host``'s
        (so the ops can be captured per width and replayed for any prefix in the bucket). ``probe(tag)`` is
        called
        after each gather (a memory probe). Returns the op counts."""
        assert state.compressed_kv_format == self.kv_format, (state.compressed_kv_format, self.kv_format)
        assert state.geometry.max_seq_len <= self.max_seq_len, (state.geometry.max_seq_len, self.max_seq_len)
        assert length % 32 == 0 and 0 < length <= state.geometry.max_seq_len, length
        cfg, g = self.config, state.geometry
        rank, u = divmod(slot, self.users)
        assert 0 < piece <= self.restore_piece, (piece, self.restore_piece)
        counts = {"gathers": 0, "scatters": 0, "layouts": 0, "pieces": 0}
        for src in state.kv:
            ratio = cfg.compress_ratio(src)
            n_rows = length // ratio
            step_rows = piece // ratio
            assert step_rows % 32 == 0 and n_rows % 32 == 0, (step_rows, n_rows)
            kv, cap = self.kv[src], self.slots[src].cap
            w = kv.shape[3]
            idim = cfg.INDEX_HEAD_DIM
            base = u * self.slots[src].rows + self.comp_row(src)
            assert n_rows <= cap, (n_rows, cap)
            for a in range(0, n_rows, step_rows):
                b = min(a + step_rows, n_rows)
                n = b - a
                counts["pieces"] += 1
                # the compressed KV rows: 528-byte FP8 pages, which the gather refuses (a DRAM page must be a
                # multiple of 64 bytes on Blackhole), so the slot's rows are packed eight to a 4,224-byte page
                # (``scatter_rows`` into block pages), the pages gathered, and the owner's rows unpacked into
                # the
                # state (``scatter_rows`` from block pages, ``src_pages``)
                assert n % BLOCK_ROWS == 0 and (BLOCK_ROWS * w * kv.element_size()) % 2 == 0, (n, w)
                pages = ttnn.allocate_tensor_on_device(  # bf16 pages of the FP8 rows' bytes: a dtype the CCL moves
                    ttnn.Shape([1, 1, n // BLOCK_ROWS, BLOCK_ROWS * w * kv.element_size() // 2]),
                    ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT,
                    self.mesh_device,
                    ttnn.DRAM_MEMORY_CONFIG,
                )
                scatter_rows(kv, pages, self.iota(n), src_base=base + a, rows=n)
                gathered = self.gather_sp(pages, dim=2)
                ttnn.deallocate(pages)
                if probe is not None:
                    probe(f"kv[{src}] {a}:{b}")
                scatter_rows(
                    gathered,
                    state.kv[src],
                    self.iota(n),
                    src_base=rank * n,
                    dst_base=g.kv_row_of_compressed(a),
                    rows=n,
                    src_pages=True,
                )
                ttnn.deallocate(gathered)
                counts["gathers"] += 1
                counts["scatters"] += 2
                # the index keys: tiled in the pool, both forms in the state
                tiles = ttnn.slice(self.index_k[src], [u, 0, a, 0], [u + 1, 1, b, idim])
                gathered = self.gather_sp(tiles, dim=2)
                ttnn.deallocate(tiles)
                if probe is not None:
                    probe(f"index_k[{src}] {a}:{b}")
                part = ttnn.slice(gathered, [0, 0, rank * n, 0], [1, 1, (rank + 1) * n, idim])
                ttnn.deallocate(gathered)
                if src in state.index_k_tiled:
                    ttnn.experimental.slice_write(
                        part, state.index_k_tiled[src], [0, 0, a, 0], [1, 1, b, idim], [1, 1, 1, 1]
                    )
                rm = ttnn.to_layout(part, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(part)
                ttnn.experimental.slice_write(rm, state.index_k[src], [0, 0, a, 0], [1, 1, b, idim], [1, 1, 1, 1])
                ttnn.deallocate(rm)
                counts["gathers"] += 1
                counts["layouts"] += 1
        self.unpark_carries(slot, state)
        return counts

    @staticmethod
    def restore_host(state: V41PrefillState, length: int) -> None:
        """The host side of a restore of a ``length``-token prefix (a multiple of the chunk): ``state.start``;
        no
        selection, Engram history (re-seeded from the tokens by the final chunk's host pipeline) or compressor
        carry (a bulk ends on a chunk boundary)."""
        assert length % state.geometry.chunk == 0 and 0 < length <= state.geometry.max_seq_len, length
        state.start = length
        state.selection = {}
        state.engram_history = None
        for src in state.compressor_carry:
            state.compressor_carry[src] = None

    def join(self, slot: int, state: V41PrefillState, segment=None) -> None:
        """Copy ``state`` (a prefill's caches, replicated on every chip; or its packed ``segment``) into pool
        slot
        ``slot`` (rank ``slot // users``, row ``slot % users``), eagerly: ``join_ops`` on ``join_inputs``
        uploaded
        for this join."""
        idx = {k: ttnn.to_device(v, self.mesh_device) for k, v in self.join_inputs(slot, state, None, segment).items()}
        self.join_ops(slot, state, idx, segment)
        for t in idx.values():
            ttnn.deallocate(t)
