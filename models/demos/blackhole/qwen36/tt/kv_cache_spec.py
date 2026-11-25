# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM KV cache specs of Qwen36ForCausalLM and the place of the GDN state inside a KV cache block.

vLLM's hybrid KV cache manager puts the attention layers into one KV cache group and the GDN layers into MambaSpec
groups of the same size, and overlays the groups on one block pool. The layers at position j of every group share one
pair of K and V buffers (a layer slot, the plugin's ``tensor_idx``), block id b names the same bytes in each of them,
and one group owns a block at a time. A GDN layer therefore keeps the state of a request in the K and V buffers of its
layer slot, at the block its group gave the request: the recurrent state as float32 pages, the convolution history as
bfloat16 pages, both read through ``ttnn.experimental.alias_view`` views of the buffers.

This module has no ttnn import, so host tests can check the specs and the page geometry with torch and vLLM alone.
"""

import ast
import math
import os
import re
from dataclasses import dataclass

import torch

FP32_PAGE_BYTES = 4096
BF16_PAGE_BYTES = 2048
# BFLOAT8_B stores a 32x32 tile as 1024 one-byte mantissas plus one shared exponent per 16 values.
BF8_TILE_BYTES = 1088
# With a bank count that is a multiple of 8, the float32 views address the state in slices of half a value head
# ([Dk / 2, Dv] = 8 pages), and a slice that starts on bank 0 covers whole page rows. With other bank counts (7 on
# P100a and on the Bin6 Blackhole Galaxy) a slice is one page.
HALF_HEAD_PAGES = 8
# gdn_decode_step's packed history: 4 tiles per value head, slot 3 the newest token, slot 0 never read.
HIST_SLOTS = 4
# Blackhole P150 and P300 have 8 DRAM banks. The specs are built before the mesh opens, so they plan for DRAM_BANKS or
# for DRAM_BANKS_ENV; GdnBlockState plans again with the device's own count.
DRAM_BANKS = 8
DRAM_BANKS_ENV = "QWEN36_DRAM_BANKS"
# Attention block sizes the prefill supports: a power of two that divides the 2048-token prefill chunk.
BLOCK_SIZES = (64, 128, 256, 512, 1024, 2048)
# gdn_decode_step handles at most this many batch rows.
MAX_DECODE_ROWS = 32
# gdn_spec_step's ctrl word that holds a head: no state or history write, the result is discarded (int32 of 0xFFFFFFFF).
SPEC_STEP_HOLD = -1


class GdnLayoutError(ValueError):
    """The GDN state of one request does not fit one KV cache block of a layer slot."""


def kv_cache_bf8_enabled():
    """QWEN_SDPA_BF8=1 makes Qwen36Model allocate the paged K and V caches as bfloat8_b."""
    return os.environ.get("QWEN_SDPA_BF8", "0") == "1"


def planned_dram_banks():
    """DRAM banks per device that the KV cache specs plan the GDN block layout for: QWEN36_DRAM_BANKS (7 on a Bin6
    Blackhole Galaxy), else DRAM_BANKS."""
    value = os.environ.get(DRAM_BANKS_ENV, "").strip()
    if not value:
        return DRAM_BANKS
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"{DRAM_BANKS_ENV}={value!r} is not a positive DRAM bank count")
    return int(value)


def _cfg(tc, name, default=None):
    return tc.get(name, default) if isinstance(tc, dict) else getattr(tc, name, default)


@dataclass(frozen=True)
class GdnDims:
    """Per-device GDN head geometry at one TP degree."""

    nk: int  # key heads
    nv: int  # value heads
    dk: int
    dv: int
    kconv: int  # convolution taps

    @classmethod
    def from_text_config(cls, text_config, tp):
        nk = int(_cfg(text_config, "linear_num_key_heads"))
        nv = int(_cfg(text_config, "linear_num_value_heads"))
        if nk % tp or nv % tp:
            raise ValueError(f"GDN heads ({nk} key, {nv} value) do not divide by TP={tp}")
        return cls(
            nk=nk // tp,
            nv=nv // tp,
            dk=int(_cfg(text_config, "linear_key_head_dim")),
            dv=int(_cfg(text_config, "linear_value_head_dim")),
            kconv=int(_cfg(text_config, "linear_conv_kernel_dim")),
        )

    @property
    def conv_dim(self):
        """Channels of one convolution input row, [q | k | v]."""
        return 2 * self.nk * self.dk + self.nv * self.dv

    @property
    def head_pages(self):
        """float32 pages of one value head's [Dk, Dv] recurrent state."""
        return self.dk * self.dv // 1024

    def decode_step_problems(self):
        """Why gdn_decode_step cannot run this geometry (empty when it can)."""
        problems = []
        if self.kconv != HIST_SLOTS:
            problems.append(f"conv kernel {self.kconv} != {HIST_SLOTS}")
        if self.nv % self.nk:
            problems.append(f"{self.nv} value heads per device are not a multiple of {self.nk} key heads")
        if 2 * self.nv > 32:
            problems.append(f"{self.nv} value heads per device > 16 (a and b share one 32-wide tile)")
        if self.dk % 32 or self.dv % 32 or self.dk != self.dv:
            problems.append(f"head dims {self.dk} x {self.dv} are not equal multiples of 32")
        if 2 * (2 * self.dk + self.dv) // 32 > 32:
            problems.append(f"a [q | k | v] head row of {2 * self.dk + self.dv} channels does not fit one history tile")
        return problems


def _cdiv(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class GdnBlockLayout:
    """Where one request's GDN state lies in one K and V block of a layer slot.

    Interleaved page p of a buffer sits on bank p % banks at row p // banks. Block b holds tiles [b * tiles, (b + 1)
    * tiles) of a K (or V) buffer, so it owns tile rows ceil(b * tiles / banks) to floor((b + 1) * tiles / banks) - 1
    of every bank (owned_bytes). When the bank count does not divide ``tiles`` (7 banks and power-of-two blocks), some
    banks also hold the row just before or just after those rows, which the layout leaves unused. A float32 or
    bfloat16 view page is used only when its whole page row lies inside owned_bytes, so a block's state starts at its
    own offset (with bfloat8_b caches it depends on the block), and every run of state pages starts on bank 0, as
    ttnn.narrow needs. Value heads 0 .. heads_k - 1 take the first whole float32 rows of the K block, the other heads
    the first whole float32 rows of the V block. The packed history (4 bfloat16 pages per head) takes the last whole
    bfloat16 rows of the V block, or its first rows when K holds every head. gdn_decode_step reads the V heads through
    its second state pool, so their index values are offset by the page count of the K view.
    """

    banks: int
    block_size: int
    tiles: int  # tiles of one K (or V) block
    kv_tile_bytes: int
    dims: GdnDims
    heads_k: int

    @property
    def heads_v(self):
        return self.dims.nv - self.heads_k

    @property
    def even_spread(self):
        """True when every block has the same number of tiles on every bank."""
        return self.tiles % self.banks == 0

    @property
    def bank_bytes(self):
        """Bytes of one block on each bank, with an even spread."""
        return self.tiles // self.banks * self.kv_tile_bytes

    @property
    def hist_tiles(self):
        """bfloat16 pages of one block in the bfloat16 view of V, with an even spread."""
        return self.tiles * self.kv_tile_bytes // BF16_PAGE_BYTES

    @property
    def hist_rows(self):
        """bfloat16 page rows of the packed history."""
        return _cdiv(HIST_SLOTS * self.dims.nv, self.banks)

    @property
    def hist_first(self):
        """First history page of a block, counted from the block's first bfloat16 page, with an even spread."""
        return self.hist_tiles - self.hist_rows * self.banks if self.heads_v else 0

    @property
    def slice_pages(self):
        """float32 pages of one dim-0 slice of the float32 views: half a value head when the bank count is a multiple
        of HALF_HEAD_PAGES, else one page."""
        return HALF_HEAD_PAGES if self.banks % HALF_HEAD_PAGES == 0 else 1

    @property
    def slice_shape(self):
        """[rows, columns] of one dim-0 slice of the float32 views."""
        return [self.dims.dk // 2, self.dims.dv] if self.slice_pages == HALF_HEAD_PAGES else [32, 32]

    def state_slices(self, heads):
        """Dim-0 slices of the float32 views that ``heads`` value heads take."""
        return heads * self.dims.head_pages // self.slice_pages

    @property
    def fp32_slices_are_blocks(self):
        """True when every block is a whole number of float32 pages per bank (alias_view's default slice check)."""
        return self.even_spread and self.bank_bytes % FP32_PAGE_BYTES == 0

    @property
    def hist_slices_are_blocks(self):
        """True when every block is a whole number of bfloat16 pages per bank (alias_view's default slice check)."""
        return self.even_spread and self.bank_bytes % BF16_PAGE_BYTES == 0

    @property
    def workspace_spreads_evenly(self):
        """True when the pages of one [1, Nv, Dk, Dv] float32 state spread evenly over the banks (alias_view's default
        slice check of a view of a GDN layer's batch-1 workspace)."""
        return self.dims.nv * self.dims.head_pages % self.banks == 0

    def owned_bytes(self, block):
        """[lo, hi): the bytes of every bank that ``block`` owns in a K or V buffer."""
        lo = _cdiv(block * self.tiles, self.banks) * self.kv_tile_bytes
        return lo, (block + 1) * self.tiles // self.banks * self.kv_tile_bytes

    def _bytes_per_bank(self, num_blocks):
        """Bytes of every bank up to the end of the last block's owned_bytes."""
        return num_blocks * self.tiles // self.banks * self.kv_tile_bytes

    def fp32_pages(self, num_blocks):
        """Page count of a float32 view of a K or V buffer of ``num_blocks`` blocks: the whole page rows of every
        bank."""
        return self.banks * (self._bytes_per_bank(num_blocks) // FP32_PAGE_BYTES)

    def fp32_view_shape(self, num_blocks):
        """Shape of the float32 views: one dim-0 slice per slice_pages pages."""
        return [self.fp32_pages(num_blocks) // self.slice_pages, 1, *self.slice_shape]

    def workspace_view_shape(self):
        """Shape of a float32 view of one [1, Nv, Dk, Dv] state in the dim-0 slices of the float32 views."""
        return [self.state_slices(self.dims.nv), 1, *self.slice_shape]

    def workspace_rows_shape(self):
        """[1, 1, rows, columns]: one [1, Nv, Dk, Dv] state as the rows that paged_fill_cache writes into the float32
        views, slice_shape[0] rows per slice; the tiles keep their order."""
        rows, cols = self.slice_shape
        return [1, 1, self.state_slices(self.dims.nv) * rows, cols]

    def hist_view_shape(self, num_blocks):
        """Shape of the bfloat16 history view: one dim-0 slice per page."""
        return [self.banks * (self._bytes_per_bank(num_blocks) // BF16_PAGE_BYTES), 1, 32, 32]

    def first_fp32_page(self, block):
        """First whole float32 page of ``block`` in the float32 view of K (and of V)."""
        return self.banks * _cdiv(self.owned_bytes(block)[0], FP32_PAGE_BYTES)

    def _hist_row(self, block):
        lo, hi = self.owned_bytes(block)
        return hi // BF16_PAGE_BYTES - self.hist_rows if self.heads_v else _cdiv(lo, BF16_PAGE_BYTES)

    def hist_page(self, block, head):
        return self.banks * self._hist_row(block) + HIST_SLOTS * head

    def index_host(self, blocks, num_blocks):
        """gdn_decode_step ``state_page_idx`` for rows that hold ``blocks``: int32 [2 * Nv, len(blocks)]. Row h is the
        first float32 page of value head h, row Nv + h the first history page of head h."""
        b = torch.as_tensor(blocks, dtype=torch.int64).reshape(1, -1)
        n, tb = self.banks, self.kv_tile_bytes
        lo = -(-b * self.tiles // n) * tb
        hi = (b + 1) * self.tiles // n * tb
        first = n * (-(-lo // FP32_PAGE_BYTES))
        h = torch.arange(self.dims.nv, dtype=torch.int64).reshape(-1, 1)
        hp = self.dims.head_pages
        rec = torch.where(
            h < self.heads_k,
            first + hp * h,
            self.fp32_pages(num_blocks) + first + hp * (h - self.heads_k),
        )
        hist_row = hi // BF16_PAGE_BYTES - self.hist_rows if self.heads_v else -(-lo // BF16_PAGE_BYTES)
        hist = n * hist_row + HIST_SLOTS * h
        return torch.cat([rec, hist]).to(torch.int32)

    def verify_tables(self, start_blocks, windows, num_blocks, rows):
        """gdn_spec_step ``ctrl`` page and ``ring_page_idx`` (with ``conv_hist``) of a verify of T = windows.shape[1]
        tokens per row: row r reads its full state from block ``start_blocks[r]`` and writes the full state after its
        query token t into block ``windows[r, t]``. ``rows``: the op's batch; rows past ``len(start_blocks)`` are
        padding rows. Returns int32 (ctrl [1, N], idx [rows, 2 * T * Nv]): ctrl word 1 + rows + r * Nv + h is the first
        float32 page of head h's start state, word 1 + rows + rows * Nv + r * Nv + h its first history page (the
        window parity and the mi words are 0); idx entry (r, t * Nv + h) is the first float32 page of the state after
        token t, entry (r, T * Nv + t * Nv + h) its first history page. The start block can be one of the row's own
        windows: the op reads it before it writes any state. A padding row (window block 0) is held
        (SPEC_STEP_HOLD): it writes nothing, so the null block 0 never takes GDN state, whose float32 bytes attention
        would read as bfloat16 (NaN or Inf at positions it masks) when a verify's padded columns point at block 0."""
        nv = self.dims.nv
        windows = torch.as_tensor(windows, dtype=torch.int64)
        starts = torch.as_tensor(start_blocks, dtype=torch.int64).reshape(-1)
        n, T = int(windows.shape[0]), int(windows.shape[1])
        if starts.numel() != n or n > rows:
            raise ValueError(f"{starts.numel()} start blocks and {n} window rows for a verify of {rows} rows")
        starts = torch.cat([starts, torch.zeros(rows - n, dtype=torch.int64)])
        windows = torch.cat([windows, torch.zeros(rows - n, T, dtype=torch.int64)])
        start_pages = self.index_host(starts, num_blocks).to(torch.int64)  # [2 * Nv, rows]
        ctrl = torch.zeros(1, -(-(1 + rows + 2 * rows * nv) // 16) * 16, dtype=torch.int64)
        held = (windows[:, 0] == 0).reshape(rows, 1)
        ctrl[0, 1 + rows : 1 + rows + rows * nv] = start_pages[:nv].T.masked_fill(held, SPEC_STEP_HOLD).reshape(-1)
        ctrl[0, 1 + rows + rows * nv : 1 + rows + 2 * rows * nv] = (
            start_pages[nv:].T.masked_fill(held, SPEC_STEP_HOLD).reshape(-1)
        )
        dest_pages = self.index_host(windows.reshape(-1), num_blocks).to(torch.int64)  # [2 * Nv, rows * T]
        rec = dest_pages[:nv].T.reshape(rows, T * nv)
        hist = dest_pages[nv:].T.reshape(rows, T * nv)
        return ctrl.to(torch.int32), torch.cat([rec, hist], dim=1).to(torch.int32)

    def state_writes(self, block):
        """(pool, first slice, first source slice, end source slice) of the recurrent state of ``block``, in dim-0
        slices of the float32 views: pool 0 is the float32 view of K, pool 1 the float32 view of V."""
        s0 = self.first_fp32_page(block) // self.slice_pages
        k_end = self.state_slices(self.heads_k)
        parts = [(0, s0, 0, k_end)]
        if self.heads_v:
            parts.append((1, s0, k_end, self.state_slices(self.dims.nv)))
        return parts

    def write_tables(self, block):
        """paged_fill_cache page tables that put one request's state into ``block``: per state_writes part the
        slices of its view, then the history pages."""
        tables = [list(range(s0, s0 + hi - lo)) for _, s0, lo, hi in self.state_writes(block)]
        first = self.hist_page(block, 0)
        return tables + [list(range(first, first + HIST_SLOTS * self.dims.nv))]


def plan_gdn_block_layout(*, block_size, kv_heads, head_dim, kv_bf8, dims, banks=DRAM_BANKS):
    """The GdnBlockLayout of a K / V cache block of ``block_size`` tokens on ``banks`` DRAM banks, or GdnLayoutError
    with the reason."""
    tile_bytes = BF8_TILE_BYTES if kv_bf8 else BF16_PAGE_BYTES
    if block_size % 32 or head_dim % 32:
        raise GdnLayoutError(f"block {block_size} x head_dim {head_dim} is not whole 32 x 32 tiles")
    tiles = kv_heads * block_size * head_dim // 1024
    s = tiles // banks * tile_bytes
    # With an even spread every block holds whole bfloat16 pages per bank, so the history sits at one offset from
    # each block's first bfloat16 page (hist_first), as in the operator tests' KV blocks.
    if tiles % banks == 0 and s % BF16_PAGE_BYTES:
        raise GdnLayoutError(f"block {block_size}: a block holds {s} B per bank, not whole bfloat16 history pages")
    hp = dims.head_pages
    lay = GdnBlockLayout(
        banks=banks, block_size=block_size, tiles=tiles, kv_tile_bytes=tile_bytes, dims=dims, heads_k=dims.nv
    )
    # The page offsets inside owned_bytes repeat after this many blocks.
    period = banks * (FP32_PAGE_BYTES // math.gcd(tiles * tile_bytes, FP32_PAGE_BYTES))
    spans = [lay.owned_bytes(b) for b in range(period)]
    rows_k = max(0, min(hi // FP32_PAGE_BYTES - _cdiv(lo, FP32_PAGE_BYTES) for lo, hi in spans))
    heads_k = min(dims.nv, rows_k * banks // hp)
    # restore copies the V heads into the batch-1 workspace through a ttnn.narrow at page heads_k * Dk * Dv / 1024 of
    # the workspace, which must lie on bank 0.
    while heads_k and heads_k < dims.nv and heads_k * hp % banks:
        heads_k -= 1
    heads_v = dims.nv - heads_k
    hist_rows = lay.hist_rows
    if min(hi // BF16_PAGE_BYTES - _cdiv(lo, BF16_PAGE_BYTES) for lo, hi in spans) < hist_rows:
        raise GdnLayoutError(f"block {block_size}: a V block cannot hold the {HIST_SLOTS * dims.nv}-page history")
    if heads_v:
        rows_v = _cdiv(heads_v * hp, banks)
        for lo, hi in spans:
            end = (_cdiv(lo, FP32_PAGE_BYTES) + rows_v) * FP32_PAGE_BYTES
            if end > (hi // BF16_PAGE_BYTES - hist_rows) * BF16_PAGE_BYTES:
                raise GdnLayoutError(
                    f"block {block_size}: K holds {heads_k} of {dims.nv} GDN value heads and V cannot hold the "
                    f"other {heads_v} and the history"
                )
    return GdnBlockLayout(
        banks=banks,
        block_size=block_size,
        tiles=tiles,
        kv_tile_bytes=tile_bytes,
        dims=dims,
        heads_k=heads_k,
    )


def gdn_state_shapes(dims):
    """MambaSpec shapes and dtypes of one GDN layer on one device, as the block stores them: the packed history
    [Nv, 4, 32, 32] bfloat16 and the recurrent state [Nv, Dk, Dv] float32."""
    return (
        ((dims.nv, HIST_SLOTS, 32, 32), (dims.nv, dims.dk, dims.dv)),
        (torch.bfloat16, torch.float32),
    )


def pack_history_rows(dims):
    """int64 [Nv * 4 * 32]: for each row (head h, slot s, tile row r) of the packed history, the row of the convolution
    carry [K - 1, C] viewed as [(K - 1) * C / 32, 32] that it holds. Slot s >= 1 holds carry row s - 1 (slot 3 the
    newest token); tile rows 2c and 2c + 1 hold channel chunk c of the head's [q_hk | k_hk | v_h] row, hk = h // (Nv /
    Nk). Slot 0 and the rows past the head row are never read and hold row 0."""
    rf = dims.nv // dims.nk
    qn, vn = dims.dk // 32, dims.dv // 32
    kd32 = dims.nk * dims.dk // 32
    row_chunks = dims.conv_dim // 32
    out = torch.zeros(dims.nv, HIST_SLOTS, 32, dtype=torch.int64)
    for h in range(dims.nv):
        hk = h // rf
        chunks = (
            [hk * qn + c for c in range(qn)]
            + [kd32 + hk * qn + c for c in range(qn)]
            + [2 * kd32 + h * vn + c for c in range(vn)]
        )
        for s in range(1, HIST_SLOTS):
            for c, ch in enumerate(chunks):
                out[h, s, 2 * c] = out[h, s, 2 * c + 1] = (s - 1) * row_chunks + ch
    return out.reshape(-1)


def unpack_history_rows(dims):
    """int64 [(K - 1) * C / 32]: for each 32-channel row (carry row j, channel chunk ch) of the convolution carry
    [K - 1, C] viewed as [(K - 1) * C / 32, 32], the row of the packed history viewed as [Nv * 4 * 32, 32] that holds it
    (pack_history_rows read backwards). A q or k chunk of key head hk is read from value head hk * (Nv / Nk), the
    first head that holds it."""
    rf = dims.nv // dims.nk
    qn, vn = dims.dk // 32, dims.dv // 32
    kd32 = dims.nk * dims.dk // 32
    out = torch.zeros(dims.kconv - 1, dims.conv_dim // 32, dtype=torch.int64)
    for ch in range(dims.conv_dim // 32):
        if ch < kd32:
            h, c = (ch // qn) * rf, ch % qn
        elif ch < 2 * kd32:
            h, c = ((ch - kd32) // qn) * rf, qn + (ch - kd32) % qn
        else:
            h, c = (ch - 2 * kd32) // vn, 2 * qn + (ch - 2 * kd32) % vn
        for j in range(dims.kconv - 1):
            out[j, ch] = (h * HIST_SLOTS + j + 1) * 32 + 2 * c
    return out.reshape(-1)


def pack_conv_taps(taps, dims):
    """[Nv, 4, 32, 32] gdn_decode_step taps from 4 per-device tap rows [C] (tap 0 = oldest): tile (h, j) holds tap j
    of the head's [q_hk | k_hk | v_h] channels, chunk c in tile rows 2c and 2c + 1."""
    rows = torch.stack([t.reshape(-1) for t in taps])  # [4, C]
    flat = rows.reshape(-1, 32)  # [4 * C / 32, 32]
    idx = pack_history_rows(dims).reshape(dims.nv, HIST_SLOTS, 32)
    out = torch.zeros(dims.nv, HIST_SLOTS, 32, 32, dtype=rows.dtype)
    row_chunks = dims.conv_dim // 32
    used = 2 * (2 * dims.dk + dims.dv) // 32
    for j in range(HIST_SLOTS):
        # Slot 1's gather rows name history row 0; tap j takes the same channels from tap row j.
        out[:, j, :used] = flat[idx[:, 1, :used] + j * row_chunks]
    return out


def choose_block_size(*, kv_heads, head_dim, kv_bf8, dims, min_block_size=None, banks=DRAM_BANKS):
    """(block size, GdnBlockLayout): the smallest size in BLOCK_SIZES, at least ``min_block_size``, whose K and V block
    holds one GDN state on ``banks`` DRAM banks. GdnLayoutError when none does."""
    sizes = [bs for bs in BLOCK_SIZES if min_block_size is None or bs >= min_block_size]
    if not sizes:
        raise GdnLayoutError(f"block size {min_block_size} is above the largest supported block, {BLOCK_SIZES[-1]}")
    reasons = []
    for bs in sizes:
        try:
            return bs, plan_gdn_block_layout(
                block_size=bs, kv_heads=kv_heads, head_dim=head_dim, kv_bf8=kv_bf8, dims=dims, banks=banks
            )
        except GdnLayoutError as e:
            reasons.append(str(e))
    raise GdnLayoutError(
        f"no KV cache block size in {sizes} holds one GDN state ({dims.nv} value heads per device, {'bfloat8_b' if kv_bf8 else 'bfloat16'} K and V, {banks} DRAM banks): "
        + "; ".join(reasons)
    )


def qwen36_layer_names(text_config):
    """(attention layer names, GDN layer names), named as vLLM's Qwen3.5 model names them."""
    n = int(_cfg(text_config, "num_hidden_layers"))
    types = _cfg(text_config, "layer_types", None) or ["linear_attention"] * 3 + ["full_attention"]
    attn, gdn = [], []
    for i in range(n):
        if types[i % len(types)] == "full_attention":
            attn.append(f"model.layers.{i}.self_attn")
        else:
            gdn.append(f"model.layers.{i}.linear_attn")
    return attn, gdn


def layer_index(name):
    """Decoder layer index of a vLLM layer name (``model.layers.<i>.<module>``)."""
    m = re.search(r"layers\.(\d+)\.", name)
    if m is None:
        raise ValueError(f"layer name {name!r} has no 'layers.<index>.'")
    return int(m.group(1))


# The plugin's MESH_DEVICE presets (vllm_tt_plugin.utils.dp_discovery). One engine on a BH Galaxy opens the (8, 4) grid
# (vllm_tt_plugin.worker._resolve_mesh_grid).
_MESH_PRESETS = {
    "N150": (1, 1),
    "P100": (1, 1),
    "P150": (1, 1),
    "P150x2": (1, 2),
    "N300": (1, 2),
    "P300": (1, 2),
    "N150x4": (1, 4),
    "P150x4": (1, 4),
    "T3K": (1, 8),
    "P150x8": (1, 8),
    "P300x2": (1, 4),
    "BH-Galaxy": (8, 4),
}
# Standard DP views a 32-device mesh as (4, 8) and gives each rank a (devices // 8, 8) submesh, or (1, devices) below 8
# devices (vllm_tt_plugin.utils.dp_submeshes).
_GALAXY_DEVICES = 32
_GALAXY_COLS = 8


def mesh_tp_degree(mesh_device=None, visible_devices=None):
    """Tensor-parallel degree of the mesh the plugin opens for this engine: the columns of MESH_DEVICE (a preset or a
    "(rows, cols)" tuple; rows hold replicas). When TT_VISIBLE_DEVICES names a different device count than MESH_DEVICE,
    as for a standard DP rank, the engine's mesh is (1, count), or (count // 8, 8) on a 32-device Galaxy, as the
    plugin's DP discovery splits it. Defaults read the environment."""
    mesh_device = os.environ.get("MESH_DEVICE") if mesh_device is None else mesh_device
    visible_devices = os.environ.get("TT_VISIBLE_DEVICES") if visible_devices is None else visible_devices
    if not mesh_device:
        raise ValueError(
            "Qwen36ForCausalLM sizes its KV cache blocks per device before the mesh opens and reads the device count "
            "from MESH_DEVICE; set MESH_DEVICE (for example P150x8)"
        )
    grid = _MESH_PRESETS.get(mesh_device)
    if grid is None:
        try:
            parsed = ast.literal_eval(mesh_device)
            grid = tuple(int(d) for d in parsed)
        except (ValueError, SyntaxError, TypeError):
            grid = None
        if grid is None or len(grid) != 2:
            raise ValueError(f"MESH_DEVICE={mesh_device!r} is neither a mesh preset nor a (rows, cols) tuple")
    count = len([d for d in (visible_devices or "").split(",") if d.strip()])
    if count and count != grid[0] * grid[1]:
        if grid[0] * grid[1] == _GALAXY_DEVICES and count % _GALAXY_COLS == 0:
            return _GALAXY_COLS
        return count
    return grid[1]


def qwen36_kv_cache_spec(
    text_config,
    tp,
    *,
    kv_bf8,
    mamba_block_size,
    mamba_cache_mode="none",
    min_block_size=None,
    num_speculative_blocks=0,
    banks=None,
):
    """dict[layer name, KVCacheSpec] for vLLM's hybrid KV cache manager, sizes per device at ``tp`` devices.

    One FullAttentionSpec per attention layer, with the smallest block size whose K and V block holds one GDN state
    (choose_block_size), and one MambaSpec per GDN layer, padded to the attention page so that vLLM keeps the
    attention block size. ``mamba_block_size``: tokens per state block in mamba_cache_mode "none" (max_model_len, so a
    request keeps one state block per group). In mode "align" the state blocks have the attention block size, and the
    MambaSpec asks for prefill checkpoints on block boundaries, where Qwen36ForCausalLM splits a prefill.
    ``num_speculative_blocks``: K of a speculative launch, so that vLLM gives a verify one state block per query token
    and row (1 + K per group). ``banks``: DRAM banks per device, by default planned_dram_banks(). Raises ValueError
    for a TP degree or a geometry gdn_decode_step does not serve."""
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    tp = int(tp)
    dims = GdnDims.from_text_config(text_config, tp)
    problems = dims.decode_step_problems()
    if problems:
        raise ValueError(
            f"Qwen36ForCausalLM decodes the GDN state in KV cache blocks with gdn_decode_step, which cannot run this "
            f"checkpoint at TP={tp} (MESH_DEVICE): " + "; ".join(problems) + ". Use a mesh with more devices"
        )
    if tp < 2:
        raise ValueError(
            "Qwen36ForCausalLM keeps the GDN state in KV cache blocks only on the tensor-parallel path; the "
            "single-device GDN layer has no decode on KV cache blocks (not implemented). Use a mesh of 2 or more "
            "devices"
        )
    dim = int(_cfg(text_config, "hidden_size"))
    n_heads = int(_cfg(text_config, "num_attention_heads"))
    n_kv = int(_cfg(text_config, "num_key_value_heads"))
    head_dim = int(_cfg(text_config, "head_dim", None) or dim // n_heads)
    # TPAttention replicates the KV heads when TP exceeds their count.
    n_local_kv = max(1, n_kv // tp)
    block_size, layout = choose_block_size(
        kv_heads=n_local_kv,
        head_dim=head_dim,
        kv_bf8=kv_bf8,
        dims=dims,
        min_block_size=min_block_size,
        banks=planned_dram_banks() if banks is None else int(banks),
    )
    attn_kw = {"state_content_bytes": 2 * head_dim * BF8_TILE_BYTES // 1024} if kv_bf8 else {}
    attn = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=n_kv,
        num_head_slots=n_local_kv,
        head_size=head_dim,
        dtype=torch.bfloat16,
        **attn_kw,
    )
    assert attn.page_size_bytes == 2 * layout.tiles * layout.kv_tile_bytes
    shapes, dtypes = gdn_state_shapes(dims)
    if mamba_cache_mode == "align":
        mamba_kw = dict(block_size=block_size, num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=block_size)
    else:
        # A multiple of the attention block keeps vLLM's scheduler block size (the LCM of the group block sizes) at
        # the attention block size.
        mamba_kw = dict(block_size=-(-int(mamba_block_size) // block_size) * block_size)
    gdn = MambaSpec(
        shapes=shapes,
        dtypes=dtypes,
        page_size_padded=attn.page_size_bytes,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=int(num_speculative_blocks),
        **mamba_kw,
    )
    attn_names, gdn_names = qwen36_layer_names(text_config)
    specs = {name: attn for name in attn_names}
    specs.update({name: gdn for name in gdn_names})
    return dict(sorted(specs.items(), key=lambda item: layer_index(item[0])))
