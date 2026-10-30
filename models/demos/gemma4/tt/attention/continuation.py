# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Position-independent programs for the eager prefill continuation (a prefix-cache hit, a scheduler chunk past
the first, the second chunk of an in-process multi-chunk prefill).

The continuation prefills ``seq_len`` tokens at absolute position ``chunk_start`` with the prior tokens' K/V in the
paged cache. Its programs used to be keyed on that position and on the user's physical block ids: ``ttnn.slice``
hashes its start / end (the RoPE cos / sin slice at ``chunk_start``, the per-block gathers of the sliding window
tail, the page-table chunk cut), and the global layers' chunked SDPA took a Python-int ``chunk_start_idx`` that
is part of its program key. Every hit at a new position compiled all of them again (~2.4 s on the 12B QB2).

``PrefillContinuation`` carries the per-forward device inputs whose *values* hold the position while their shapes
stay fixed:

* ``start_tensor``: int32 ``[1]`` absolute chunk start, the ``chunk_start_idx_tensor`` of the global layers' chunked
  SDPA (read by the kernel at run time);
* ``positions``: uint32 ``[1, seq_len]`` absolute positions, the RoPE cos / sin gather (``ttnn.embedding`` on the 2-D
  caches, as decode does) instead of the 4-D cache slice;
* per kv-cache group, a page-table cut built on the host from the group's table (the fill table for this chunk's
  blocks, the global layers' table at the served context's width (``continuation_sdpa_table_cols``: the chunked
  SDPA compiles its K extent from that width, so it must not follow the prompt), and the sliding layers' *window
  frame*: the blocks covering ``[chunk_start - window, chunk_start + seq_len)`` with a relative start tensor),
  uploaded as fixed-shape tensors.

The sliding layers then run one chunked paged SDPA with ``sliding_window_size`` over the window frame, replacing the
per-block gather + concat + cut + square SDPA. The frame keeps the K walk bounded (``window + seq_len`` rows) under
Gemma 4's fp32-accumulating SDPA, whose kernel otherwise walks the masked prefix from position 0.

``GEMMA4_CONTINUATION_CHUNKED_SDPA`` (default 1) selects this path; ``0`` restores the gather + square SDPA, the
int-keyed chunked SDPA and the sliced RoPE. The bounded sliding ring (``cache_position_modulo``) and the traced
multi-chunk prefill (device-tensor ``chunk_start_idx``) are untouched either way.
"""

import math
import os

import torch
from loguru import logger

import ttnn

from .operations import PREFILL_CHUNK_SIZE

CONTINUATION_CHUNKED_SDPA_ENV = "GEMMA4_CONTINUATION_CHUNKED_SDPA"

#: The served prefix-cache hit is aligned down to this many tokens (generator.SDPA_CHUNK_ALIGN); the continuation
#: SDPA's chunk sizes are capped at it so every hit offset keys the same program.
CONTINUATION_SDPA_ALIGN = 128

#: The chunked SDPA's flexible (tensor-start) path reads the user's page-table row as one stick of ``cols * 4``
#: bytes and requires that to be a multiple of 32 (sdpa_program_factory.cpp ``compute_chunked_params``: "page table
#: stick size must be a multiple of 32"), so every page-table row built here has a multiple of 8 int32 columns; the
#: padding columns are the null block 0 and lie past what the K walk reads.
PAGE_TABLE_STICK_COLS = 8


def pad_page_table_cols(cols: int) -> int:
    """``cols`` rounded up to the chunked SDPA's stick multiple (8 int32 columns = 32 bytes)."""
    return -(-int(cols) // PAGE_TABLE_STICK_COLS) * PAGE_TABLE_STICK_COLS


def continuation_sdpa_table_cols(max_seq_len, chunk_start, seq_len, block_size) -> int:
    """Columns of the page table a global layer's continuation hands the chunked SDPA, at the stick multiple.

    The op's tensor-start path compiles its K extent from that width (``Sk = Sq + cols * block``,
    sdpa_program_factory.cpp ``max_prefix_tokens_flexible``: one kernel binary per value) and walks K at run time
    only up to the start tensor plus the chunk (reader_interleaved.cpp ``valid_Skt_bound = min(chunk_start / 32 +
    valid_Sqt, valid_Skt)``), so the width is free to be one value per model and padded chunk length instead of
    the prompt's block count: the served context's blocks plus the chunk's own (a prompt within a chunk of the
    context ends past it once padded; the walk reads columns up to ``ceil((chunk_start + seq_len) / block)``).
    The chunk length is a program key already (``Sq``), so this adds no program, and the columns past the walk
    hold the null block 0 and are never read. Never below the walk's own need (a prompt past the configured
    context stays correct, on a program of its own); that need alone without a context (``max_seq_len`` None).
    """
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"continuation SDPA table needs block_size > 0 ({block_size})")
    cols = -(-(int(chunk_start) + int(seq_len)) // block_size)
    if max_seq_len:
        cols = max(cols, -(-int(max_seq_len) // block_size) + -(-int(seq_len) // block_size))
    return pad_page_table_cols(cols)


TILE = 32


def continuation_chunked_sdpa_enabled() -> bool:
    """``GEMMA4_CONTINUATION_CHUNKED_SDPA`` (default on)."""
    return os.environ.get(CONTINUATION_CHUNKED_SDPA_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def continuation_window_plan(chunk_start, seq_len, sliding_window, block_size, align):
    """The sliding layers' window frame for a continuation of ``seq_len`` rows at ``chunk_start``.

    Returns ``(first_block, cols, rel_start)``: the frame is page-table columns ``[first_block, first_block + cols)``
    of the user's row, and the chunk starts ``rel_start`` tokens into it. The frame origin is aligned down to
    ``lcm(block_size, align)`` so ``rel_start`` keeps the chunk's own alignment (``align`` = the SDPA's q / k chunk
    size, which the chunk start must be a multiple of). ``cols`` depends on the window, the alignment and the chunk
    length only, never on the position, so the frame's shape (hence the SDPA program) is the same for every hit;
    it is padded to the op's stick multiple (``PAGE_TABLE_STICK_COLS``) with null-block columns past the covered
    range, which the K walk (bounded by ``rel_start + seq_len``) never reads.
    The frame covers ``[chunk_start - window, chunk_start + seq_len)`` (the window's first key is one past its
    start: a query at ``p`` attends ``(p - window, p]``); a hit inside the first window starts at position 0.
    """
    chunk_start, seq_len, sliding_window, block_size = (
        int(v) for v in (chunk_start, seq_len, sliding_window, block_size)
    )
    align = max(TILE, int(align))
    if block_size <= 0 or chunk_start < 0:
        raise ValueError(
            f"continuation window plan needs block_size > 0 and chunk_start >= 0 ({block_size}, {chunk_start})"
        )
    origin_align = _lcm(block_size, align)
    start = max(0, chunk_start - sliding_window)
    origin = (start // origin_align) * origin_align
    rel_start = chunk_start - origin
    cols = pad_page_table_cols(-(-(sliding_window + origin_align + seq_len) // block_size))
    assert rel_start + seq_len <= cols * block_size, (rel_start, seq_len, cols, block_size)
    return origin // block_size, cols, rel_start


def continuation_window_table(table, first_block, cols, valid_cols=None):
    """Columns ``[first_block, first_block + cols)`` of every row of the host page table ``table`` (``[rows, w]``):
    the first ``valid_cols`` (default all) copied, the rest the null block 0 (also past the table's width), so the
    stick-padding columns can only ever address the null block."""
    t = table if table.dim() > 1 else table.unsqueeze(0)
    rows, width = int(t.shape[0]), int(t.shape[-1])
    cols = int(cols)
    valid = cols if valid_cols is None else max(0, min(int(valid_cols), cols))
    out = torch.zeros((rows, cols), dtype=torch.int32)
    lo, hi = int(first_block), min(int(first_block) + valid, width)
    if hi > lo:
        out[:, : hi - lo] = t[:, lo:hi].to(torch.int32)
    return out


def align_sdpa_chunks(program_config, start, cap=CONTINUATION_SDPA_ALIGN):
    """The chunked SDPA needs ``start`` to be a multiple of its q and k chunk sizes (the kernel divides the start by
    the chunk size). Cap both at ``cap`` so every served hit (aligned to ``cap``) keys one program, and shrink them to
    a 32-multiple divisor of ``start`` when it is not a multiple (rare: a chunk start off the hit alignment).
    Returns ``(program_config, align)`` with ``align`` the lcm of the final chunk sizes (32 with the op defaults)."""
    if program_config is None:
        if int(start) % TILE:
            raise ValueError(f"continuation chunk start {start} is not tile-aligned")
        return None, TILE
    q, k = int(program_config.q_chunk_size), int(program_config.k_chunk_size)
    q, k = min(q, int(cap)), min(k, int(cap))
    start = int(start)

    def _fit(chunk):
        while chunk > TILE and start % chunk:
            chunk -= TILE
        return chunk

    if start > 0:
        q, k = _fit(q), _fit(k)
        if start % q or start % k:
            raise ValueError(f"continuation chunk start {start} has no 32-multiple SDPA chunk size")
    if (q, k) != (int(program_config.q_chunk_size), int(program_config.k_chunk_size)):
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=program_config.compute_with_storage_grid_size,
            q_chunk_size=q,
            k_chunk_size=k,
            exp_approx_mode=program_config.exp_approx_mode,
        )
    return program_config, _lcm(q, k)


def sliding_continuation_sdpa(
    tt_q,
    k_cache,
    v_cache,
    window_table,
    start_tensor,
    user_id,
    *,
    sliding_window,
    program_config,
    compute_kernel_config,
    paged_cache_geometry=None,
):
    """The chunked paged SDPA over the window frame: Q ``[1, nh, S, D]`` at ``rel_start`` (``start_tensor``, int32
    ``[1]``) of the frame ``window_table`` (``[rows, cols]``, row ``user_id``), keys in ``(p - window, p]`` per query.
    Q is walked in ``PREFILL_CHUNK_SIZE`` pieces like the global layers' ``chunked_prefill_sdpa`` (a piece's start is
    the start tensor plus its fixed offset, added on device), each padded to the q-chunk multiple and cut back; the
    cuts are at fixed rows, so nothing here is keyed on the position."""
    nh, seq_len, head_dim = int(tt_q.shape[1]), int(tt_q.shape[-2]), int(tt_q.shape[-1])
    q_chunk = int(program_config.q_chunk_size) if program_config is not None else TILE
    cols = int(window_table.shape[-1])
    user_pt = window_table
    owns_user_pt = False
    if int(window_table.shape[0]) > 1:
        user_pt = ttnn.slice(window_table, [int(user_id), 0], [int(user_id) + 1, cols])
        owns_user_pt = True
    outs = []
    start = 0
    while start < seq_len:
        clen = min(PREFILL_CHUNK_SIZE, seq_len - start)
        q_in = tt_q
        owns_q_in = False
        if clen < seq_len:
            q_in = ttnn.slice(tt_q, [0, 0, start, 0], [1, nh, start + clen, head_dim])
            owns_q_in = True
        pad = (-clen) % q_chunk
        if pad:
            unpadded = q_in
            q_in = ttnn.pad(unpadded, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            if owns_q_in:
                unpadded.deallocate(True)
            owns_q_in = True
        piece_start = start_tensor
        if start:
            offset = ttnn.full(
                shape=start_tensor.shape,
                fill_value=start,
                dtype=start_tensor.dtype,
                layout=start_tensor.layout,
                device=start_tensor.device(),
            )
            piece_start = ttnn.add(start_tensor, offset)
            offset.deallocate(True)
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_in,
            k_cache,
            v_cache,
            user_pt,
            chunk_start_idx_tensor=piece_start,
            scale=1.0,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            paged_cache_geometry=paged_cache_geometry,
            sliding_window_size=int(sliding_window),
        )
        if piece_start is not start_tensor:
            piece_start.deallocate(True)
        if owns_q_in:
            q_in.deallocate(True)
        if pad:
            padded = out
            out = ttnn.slice(padded, [0, 0, 0, 0], [1, nh, clen, head_dim])
            padded.deallocate(True)
        outs.append(out)
        start += clen
    if owns_user_pt:
        user_pt.deallocate(True)
    if len(outs) == 1:
        return outs[0]
    out = ttnn.concat(outs, dim=2)
    for piece in outs:
        piece.deallocate(True)
    return out


class PrefillContinuation:
    """The device inputs of one eager continuation forward (see the module docstring).

    Built by ``prepare_inputs_prefill`` with the host page table it uploads (``legacy_table``, in its device layout,
    with its mesh mapper); ``ttnn_prefill_forward`` attaches the per-layer host tables of the vLLM hybrid groups
    with the model's layout callback (owner-lane stacking under lanes). The per-group cuts are cached by host table
    identity so the layers of one group share two uploads. ``release`` frees everything after the layer loop.
    """

    def __init__(
        self,
        mesh_device,
        mesh_mapper,
        *,
        chunk_start,
        seq_len,
        legacy_table=None,
        legacy_mapper=None,
        max_seq_len=None,
    ):
        self.chunk_start = int(chunk_start)
        self.seq_len = int(seq_len)
        # The served context: the global layers' table is sized from it (``continuation_sdpa_table_cols``) so the
        # chunked SDPA's K extent (a compile-time kernel argument, one binary per distinct value) is the same for
        # every prompt length and position.
        self.max_seq_len = int(max_seq_len) if max_seq_len else None
        self.mesh_device = mesh_device
        self._mapper = mesh_mapper
        self._legacy = None
        if legacy_table is not None:
            t = legacy_table if legacy_table.dim() > 1 else legacy_table.unsqueeze(0)
            self._legacy = (t.to(torch.int32), legacy_mapper)
        self._per_layer = None
        self._layout = None
        self._cache = {}
        self._rope = {}
        self._owned = []
        self.start_tensor = self._upload(torch.tensor([self.chunk_start], dtype=torch.int32), ttnn.int32, mesh_mapper)
        # int64 source for the uint32 tensor (the decode positions do the same: no int32 -> uint32 host conversion).
        positions = torch.arange(self.chunk_start, self.chunk_start + self.seq_len, dtype=torch.int64).reshape(1, -1)
        self.positions = self._upload(positions, ttnn.uint32, mesh_mapper)

    # ── construction ────────────────────────────────────────────────────────────────────────────────────────────

    def _upload(self, host, dtype, mapper):
        t = ttnn.from_torch(
            host, device=self.mesh_device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
        )
        self._owned.append(t)
        return t

    def attach_per_layer_tables(self, host_tables, layout):
        """``host_tables``: the per-layer host page tables (torch, or None / device tensors for layers without one);
        ``layout(table, width, layer_idx) -> (host, mapper)`` gives a table its device layout and mesh mapper."""
        self._per_layer = list(host_tables) if host_tables is not None else None
        self._layout = layout

    def _group(self, layer_idx):
        """``(key, host_table, layout_or_None, mapper_or_None)`` of the table the layer's cache is addressed by, or
        None when no host table is known for it (direct device tables: the caller keeps its device-side cut)."""
        if self._per_layer is not None and 0 <= int(layer_idx) < len(self._per_layer):
            pt = self._per_layer[int(layer_idx)]
            if isinstance(pt, torch.Tensor):
                return (id(pt), pt if pt.dim() > 1 else pt.unsqueeze(0), self._layout, None)
            if pt is not None:
                return None  # a device tensor handed straight through
        if self._legacy is not None:
            table, mapper = self._legacy
            return ("legacy", table, None, mapper)
        return None

    def has_host_table(self, layer_idx) -> bool:
        return self._group(layer_idx) is not None

    def _upload_table(self, host, layout, mapper, layer_idx):
        if layout is not None:
            host, mapper = layout(host, None, int(layer_idx))
        return self._upload(host.to(torch.int32), ttnn.int32, mapper)

    # ── per-layer inputs ────────────────────────────────────────────────────────────────────────────────────────

    def fill_table(self, layer_idx, block_size, first_col, cols):
        """This chunk's blocks on the layer's own block grid: columns ``[first_col, first_col + cols)`` of the
        group's table (the host cut of what ``_own_chunk_slice`` sliced on device, keyed on the column)."""
        group = self._group(layer_idx)
        if group is None:
            return None
        key, table, layout, mapper = group
        valid, cols = int(cols), pad_page_table_cols(cols)  # null columns past the chunk's blocks (never written)
        cache_key = (key, "fill", int(block_size), int(first_col), cols)
        dev = self._cache.get(cache_key)
        if dev is None:
            host = continuation_window_table(table, first_col, cols, valid_cols=valid)
            dev = self._upload_table(host, layout, mapper, layer_idx)
            self._cache[cache_key] = dev
        return dev

    def padded_table(self, layer_idx, block_size):
        """The group's table for the global layers' chunked SDPA, ``continuation_sdpa_table_cols`` wide on the
        layer's ``block_size`` grid (the served context's blocks plus the chunk's): the host table's first columns,
        the rest the null block; None without a host table. The width follows neither the prompt nor the device
        table it was uploaded at (the generator sizes that to the prompt's chunk grid, the vLLM bridge's persistent
        per-layer tables grow to the widest prompt so far): the SDPA kernels are compiled per K extent
        (``Sq + cols * block``), and every such width compiled the global layers again (12B QB2, 2026-10-30:
        1.4-2.1 s on the first hit of each 512-token prompt-length class, after a boot warmup at another width)."""
        group = self._group(layer_idx)
        if group is None:
            return None
        key, table, layout, mapper = group
        cols = continuation_sdpa_table_cols(self.max_seq_len, self.chunk_start, self.seq_len, block_size)
        cache_key = (key, "full", int(cols))
        dev = self._cache.get(cache_key)
        if dev is None:
            dev = self._upload_table(continuation_window_table(table, 0, cols), layout, mapper, layer_idx)
            self._cache[cache_key] = dev
        return dev

    def sliding_inputs(self, layer_idx, block_size, sliding_window, align):
        """``(window_table, rel_start_tensor, rel_start, cols)`` of the sliding layers' window frame on the group's
        block grid, or None without a host table."""
        group = self._group(layer_idx)
        if group is None:
            return None
        key, table, layout, mapper = group
        first_block, cols, rel_start = continuation_window_plan(
            self.chunk_start, self.seq_len, sliding_window, block_size, align
        )
        cache_key = (key, "window", int(block_size), int(sliding_window), int(align))
        entry = self._cache.get(cache_key)
        if entry is None:
            # The K walk reads blocks [first_block, first_block + covered); every later column is the null block.
            covered = -(-(rel_start + self.seq_len) // int(block_size))
            host = continuation_window_table(table, first_block, cols, valid_cols=covered)
            window = self._upload_table(host, layout, mapper, layer_idx)
            start = self._upload(torch.tensor([int(rel_start)], dtype=torch.int32), ttnn.int32, self._mapper)
            entry = (window, start)
            self._cache[cache_key] = entry
        return entry[0], entry[1], int(rel_start), int(cols)

    def rope_mats(self, layer_type, caches_2d):
        """``(cos, sin)`` ``[1, 1, seq_len, head_dim]`` for the chunk's positions, gathered once per layer type from
        the 2-D caches ``(cos_2d, sin_2d)`` (``[max_seq_len, head_dim]``, row-major)."""
        mats = self._rope.get(layer_type)
        if mats is None:
            cos_2d, sin_2d = caches_2d
            cos = ttnn.unsqueeze_to_4D(ttnn.embedding(self.positions, cos_2d, layout=ttnn.TILE_LAYOUT))
            sin = ttnn.unsqueeze_to_4D(ttnn.embedding(self.positions, sin_2d, layout=ttnn.TILE_LAYOUT))
            self._owned.extend((cos, sin))
            mats = (cos, sin)
            self._rope[layer_type] = mats
        return mats

    def for_layer(self, layer_idx):
        return LayerContinuation(self, layer_idx)

    def release(self):
        for t in self._owned:
            try:
                t.deallocate(True)
            except Exception as exc:  # a tensor the device already reclaimed
                logger.debug("continuation tensor release: {}", exc)
        self._owned = []
        self._cache = {}
        self._rope = {}


class LayerContinuation:
    """One layer's view of the forward's ``PrefillContinuation`` (``for_layer``): the shared start tensor plus the
    page-table cuts of the layer's own kv-cache group."""

    __slots__ = ("_parent", "layer_idx")

    def __init__(self, parent, layer_idx):
        self._parent = parent
        self.layer_idx = int(layer_idx)

    @property
    def chunk_start(self):
        return self._parent.chunk_start

    @property
    def seq_len(self):
        return self._parent.seq_len

    @property
    def start_tensor(self):
        return self._parent.start_tensor

    def has_host_table(self):
        return self._parent.has_host_table(self.layer_idx)

    def fill_table(self, block_size, first_col, cols):
        return self._parent.fill_table(self.layer_idx, block_size, first_col, cols)

    def sliding_inputs(self, block_size, sliding_window, align):
        return self._parent.sliding_inputs(self.layer_idx, block_size, sliding_window, align)

    def padded_table(self, block_size):
        return self._parent.padded_table(self.layer_idx, block_size)
