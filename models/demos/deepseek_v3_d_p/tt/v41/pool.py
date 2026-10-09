# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 decode pool: the caches of the users a mesh row decodes together (``tt/v41/batch_decode.py``).

The batched decode step runs one 32-row tile per chip, row ``u`` the user in slot ``u`` of that chip's mesh row
(``users_per_rank`` slots, the rest of the tile padding): the SP ranks hold different users and TP splits the
hidden dim and heads as in prefill, so a user's caches live on the four chips of one mesh row. Per KV source the
rank's pool is one row-major tensor of ``users_per_rank`` slots, each laid out as::

    rows [ring_base(l), ring_base(l) + 128)   layer l's window ring: 128 slots, one per consumer layer of the source
    rows [comp_base, comp_base + cap)         compressed KV row j at comp_base + j, cap = max_seq_len / ratio

so one ``sparse_sdpa`` over the pool reads a user's window and selected rows with per-row indices (row ``u`` of the
index rows offset by ``u * slot_rows``). A window ring is indexed by the global step: the step writes every user's
window row at ring slot ``step % 128`` and the window always covers the whole ring, so the window index rows are a
constant per layer and no carry is shifted. Index keys are a ``[users_per_rank, 1, cap, idim]`` tensor per source
(slot ``u`` selected by ``cache_batch_idx`` in the indexer), ratio-0 layers share a ring-only tensor.

Rows reach the pool through :func:`scatter_rows` (``kernels/scatter_rows.cpp``): the destination row of each source
row comes from a device tensor, so the captured step writes where the step's inputs say, and a join copies a
prefill state's rows into one rank's slot with the other ranks' indices set to the sentinel.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.v41.cache import WINDOW_SLOT, V41PrefillState, replicated_zeros
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import _cores, _split
from models.demos.deepseek_v3_d_p.tt.v41.layout import V41MeshLayout
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, MlaKvCacheGeometry

ROWS = 32  # the decode tile
RING = 128  # window ring slots per layer: the sliding window (SLIDING_WINDOW <= 128, asserted)
SENTINEL = 0xFFFFFFFF
_KERNEL = "models/demos/deepseek_v3_d_p/tt/v41/kernels/scatter_rows.cpp"


def scatter_rows(src, dst, idx, src_base: int = 0, dst_base: int = 0, stride: int = 0, rows: int | None = None):
    """Copy row ``src_base + i`` of ``src`` to row ``dst_base + i * stride + idx[i]`` of ``dst`` for ``i`` below
    ``rows`` (default: ``idx``'s length), skipping the rows whose ``idx`` is the sentinel; in place on ``dst``. Both
    row-major DRAM-interleaved with rows of equal bytes; ``idx`` ``[1, 1, 1, n]`` uint32 row-major (per chip: a
    sharded index tensor scatters differently on each chip). One data-movement op."""
    for t, what in ((src, "src"), (dst, "dst"), (idx, "idx")):
        memory = t.memory_config()
        assert t.layout == ttnn.ROW_MAJOR_LAYOUT, f"scatter_rows {what}: row-major, got {t.layout}"
        assert memory.buffer_type == ttnn.BufferType.DRAM and not memory.is_sharded(), f"scatter_rows {what}: DRAM"
    assert idx.dtype == ttnn.uint32 and idx.shape[2] == 1, tuple(idx.shape)
    row_bytes = src.shape[3] * src.element_size()
    assert dst.shape[3] * dst.element_size() == row_bytes, (tuple(src.shape), tuple(dst.shape))
    n = idx.shape[3] if rows is None else rows
    assert 0 < n <= idx.shape[3] and src_base + n <= src.shape[2], (n, tuple(idx.shape), src_base, tuple(src.shape))
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
        compile_time_args=[0, row_bytes] + accessors,
        runtime_args=args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    scratch = 64 + (n * 4 + 63) // 64 * 64 + 2 * ((row_bytes + 63) // 64 * 64)
    page = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)
    cbs = [ttnn.CBDescriptor(total_size=scratch, core_ranges=cores, format_descriptors=[page])]
    return ttnn.generic_op([src, idx, dst], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs))


@dataclass(frozen=True)
class SlotLayout:
    """One KV source's slot: its consumer layers' rings, then the compressed rows."""

    layers: tuple  # consumer layers in order (ring index = position in this tuple)
    cap: int  # compressed rows (0 for the ratio-0 tensor)

    @property
    def comp_base(self) -> int:
        return RING * len(self.layers)

    @property
    def rows(self) -> int:
        return self.comp_base + self.cap

    def ring_base(self, layer: int) -> int:
        return RING * self.layers.index(layer)


class V41DecodePool:
    def __init__(self, mesh_device, config, max_seq_len: int, layers: list[int], users_per_rank: int, kv_format):
        """The pool of ``users_per_rank`` slots per mesh row for the layer schedule ``layers`` (all of a model's
        layers in serving) up to ``max_seq_len`` positions per user, compressed rows in ``kv_format``."""
        assert 0 < users_per_rank <= ROWS, users_per_rank
        assert config.SLIDING_WINDOW <= RING and WINDOW_SLOT == RING
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
            self.slots[src] = SlotLayout(consumers, max_seq_len // ratio)
        swa = tuple(l for l in self.layers if config.compress_ratio(l) == 0)
        if swa:
            self.slots[-1] = SlotLayout(swa, 0)
        self.kv = {}
        self.index_k_flat = {}  # [1, 1, users * cap, idim] bf16 row-major: the rows scatter_rows writes
        self.index_k = {}  # the same buffer as [users, 1, cap, idim]: the indexer's cache slots
        for src, slot in self.slots.items():
            fmt = kv_format if src >= 0 else MlaKvCacheFormat.BF16_RM
            w = fmt.storage_width(self.kv_geometry) if src >= 0 else config.HEAD_DIM
            self.kv[src] = replicated_zeros(mesh_device, users_per_rank * slot.rows, w, fmt.storage_dtype)
            if src >= 0:
                # cap + 32 rows per slot: the indexer's causal window starts past the slot's rows, the last tile
                # (never written, zero keys) is what it scores there
                flat = replicated_zeros(mesh_device, users_per_rank * (slot.cap + 32), idim, ttnn.bfloat16)
                self.index_k_flat[src] = flat
                self.index_k[src] = ttnn.reshape(flat, (users_per_rank, 1, slot.cap + 32, idim))
        # fp32 (kv, score) projections of every ratio-2 source's last token per slot row (the compressor carry)
        self.carry = {
            src: tuple(self._replicated(torch.zeros(1, 1, ROWS, config.HEAD_DIM), ttnn.float32) for _ in range(2))
            for src in sources
            if config.compress_ratio(src) > 1
        }
        self._window = {l: self._window_rows(l) for l in self.layers}
        self._offsets = {src: self._user_offsets(src) for src in sources}
        self.step = 0  # the global step: the ring slot every user's next window row goes to is step % RING

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
        """``sparse_sdpa`` index rows ``[1, 1, ROWS, RING]`` uint32 row-major of ``layer``'s window: row u (a user)
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

    # --- joining ----------------------------------------------------------------------------------------------
    def join(self, slot: int, state: V41PrefillState) -> None:
        """Copy ``state`` (a prefill's caches, replicated on every chip) into pool slot ``slot`` (rank ``slot //
        users``, row ``slot % users``): its compressed rows and index keys, every layer's window carry into the
        layer's ring (the carry's row k, position ``state.start - 128 + k``, at ring slot ``(step + k) % RING`` so the
        next step's token, at ``step % RING``, replaces position ``start - 128``), and the compressor carries."""
        assert state.compressed_kv_format == self.kv_format, (state.compressed_kv_format, self.kv_format)
        assert state.geometry.max_seq_len <= self.max_seq_len, (state.geometry.max_seq_len, self.max_seq_len)
        rank, u = divmod(slot, self.users)
        assert 0 <= rank < self.sp, slot
        cfg, g = self.config, state.geometry

        def index(n: int, values: torch.Tensor):
            host = torch.full((self.sp, 1, 1, n), -1, dtype=torch.int32)  # -1: the sentinel as uint32
            host[rank, 0, 0] = values.to(torch.int32)
            return self.per_rank(host, ttnn.uint32)

        # compressed rows and index keys of complete groups
        for src in state.kv:
            ratio = cfg.compress_ratio(src)
            written = state.start // ratio
            if written == 0:
                continue
            idx = index(written, torch.zeros(written, dtype=torch.int64))
            base = u * self.slots[src].rows + self.comp_row(src)
            scatter_rows(state.kv[src], self.kv[src], idx, src_base=g.kv_row_of_compressed(0), dst_base=base, stride=1)
            scatter_rows(
                state.index_k[src], self.index_k_flat[src], idx, dst_base=u * (self.slots[src].cap + 32), stride=1
            )
            ttnn.deallocate(idx)
        # window carries into the rings (the carry's 128 rows are the 128 positions before state.start)
        assert state.window_carry[self.layers[0]].shape[2] == RING
        ring = index(RING, (torch.arange(RING) + self.step) % RING)
        for layer in self.layers:
            base = u * self.slot_rows(layer) + self.ring_row(layer)
            scatter_rows(state.window_carry[layer], self.kv_tensor(layer), ring, dst_base=base)
        ttnn.deallocate(ring)
        # compressor carries (a trailing incomplete ratio-2 group) into row u of the pool's carry buffers
        row = index(1, torch.tensor([u]))
        for src, carry in state.compressor_carry.items():
            if carry is None:
                continue
            for t, dst in zip(carry, self.carry[src]):
                assert t.shape[2] == 1, tuple(t.shape)  # a prefill's row
                src_rows = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
                dst_rows = ttnn.to_layout(dst, ttnn.ROW_MAJOR_LAYOUT)
                scatter_rows(src_rows, dst_rows, row)
                ttnn.copy(ttnn.to_layout(dst_rows, ttnn.TILE_LAYOUT), dst)
                ttnn.deallocate(src_rows)
                ttnn.deallocate(dst_rows)
        ttnn.deallocate(row)
