# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN state of the MTP spec class in vLLM-owned MambaSpec blocks (QWEN36_MAMBA_SPEC=1, num_speculative_blocks = K).

vLLM overlays its KV cache groups: block id b names the same bytes in the attention layer of a layer slot
(``tensor_idx``) and in each GDN layer of that slot, and one group owns a block id at a time. A GDN layer keeps one full
state (recurrent state and conv history) per block, inside the K and V buffers of its slot:

* recurrent state: Nv heads x 16 fp32 pages (one [Dk, Dv] head = 4 x 4 tiles), addressed in a flat fp32 view of K
  ("pool 1") and of V ("pool 2"); a head's 16 pages are consecutive and lie in one pool;
* conv history: the B=1 prefill carry ([kc - 1, C] bf16, the last kc - 1 conv inputs, oldest first) in row-major order,
  32 values per tile row, in ceil((kc - 1) * C / 1024) consecutive tiles of a flat bf16 view of V (4 at TP=8).

Only whole view pages inside the block's bytes are used (SpecBlockLayout). Layout "two_pool": heads fill K first, the
rest go to V, the conv history follows them in V (block 1024 BF8 at TP=8 on 8 banks). Layout "k_only": every head in K,
the conv history at the start of V's block (block 2048 BF8 at TP=8).

A verify (gdn_spec_step with ring_page_idx) reads the start state of row u from its window column
state_accepted_counts[u] - 1 and writes the state after token t into window column t. The conv window of the decode
stays in the layer's private ping-pong pair (TPGatedDeltaNet._verify_win_pair): no verify reads a block's conv history,
and the window holds the conv rows of every column of a row's latest verify. They are written into a block only where
vLLM can read the block as something other than the same row's next verify (``materialize``): the source of a
copy_kv_cache_blocks pair, and in mode "align" a column-0 block whose state ends exactly on a block boundary (vLLM keeps
that state for prefix-cache hits without a copy when the step had no drafts).
"""

import os
import time
from dataclasses import dataclass, field

import torch

FP32_PAGE = 4096
BF16_PAGE = 2048
BF8_TILE = 1088
HEAD_PAGES = 16  # fp32 pages of one [Dk, Dv] = [128, 128] head
HALF_PAGES = 8  # fp32 pages of one [Dk / 2, Dv] slice: the unit of the rec views
WARMUP_REQ = -1  # the prefill request index of the engine's warmup prompts: they write into vLLM's null block 0
ZERO_BLOCKS_PER_FILL = 8  # attention blocks zeroed by one paged_fill_cache (the zero source is this many blocks long)
ZERO_SOURCE_MAX_BYTES = 8 << 20  # cap of that zero source per device
LAYOUTS = ("two_pool", "k_only")
# S = tiles * 1088 is a multiple of 64, so block b's offset b * S modulo a 4096 B page repeats within 64 blocks: the
# whole-page counts of blocks 0 .. 63 are those of every block
_PERIOD = 65


def _tile_bytes(dtype_name):
    return {"bfloat8_b": BF8_TILE, "bfloat16": BF16_PAGE, "float32": FP32_PAGE}[dtype_name]


@dataclass
class SpecBlockLayout:
    """Page addresses of one GDN layer's state inside a vLLM block. Pure host math.

    ``block_bytes``: bytes of one block of the K buffer (and of the V buffer) on ONE DRAM bank: an interleaved buffer
    puts page i in bank i % n_banks at offset (i // n_banks) * page_size, so a block of tiles_per_block K tiles covers
    bytes [b * S, (b + 1) * S) of every bank, S = tiles_per_block / n_banks * tile_bytes. A flat view page p with page
    size s sits in bank p % n_banks at offset (p // n_banks) * s; the pages whose bytes lie inside a block are whole view
    rows. In each pool the heads start at the block's first whole fp32 page rounded up to a multiple of HALF_PAGES (the
    prefill write and restore move a head as two [Dk / 2, Dv] view blocks)."""

    block_bytes: int
    n_banks: int
    num_blocks: int
    nv: int
    conv_pages: int
    layout: str = "two_pool"
    heads_in_k: int = field(init=False)
    pool1_pages: int = field(init=False)
    conv_pages_total: int = field(init=False)

    def __post_init__(self):
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout {self.layout!r} not in {LAYOUTS}")
        S, nb = int(self.block_bytes), int(self.n_banks)
        rows_per_bank = (self.num_blocks * S) // FP32_PAGE
        # pool 1 = the fp32 view of K in HALF_PAGES units: the op splits the two-pool page space at its page count
        self.pool1_pages = (rows_per_bank * nb) // HALF_PAGES * HALF_PAGES
        self.conv_pages_total = (self.num_blocks * S) // BF16_PAGE * nb
        k_heads = min(self._pool_heads(b) for b in range(min(self.num_blocks, _PERIOD)))
        if self.layout == "k_only":
            if k_heads < self.nv:
                raise ValueError(
                    f"k_only: a K block holds {k_heads} whole heads of 16 fp32 pages per block, the layer has {self.nv}"
                )
            self.heads_in_k = self.nv
        else:
            self.heads_in_k = min(self.nv, k_heads)
            if self.nv - self.heads_in_k > k_heads:
                raise ValueError(
                    f"two_pool: K and V blocks hold {2 * k_heads} whole heads of 16 fp32 pages, the layer has {self.nv}"
                )
        for b in range(min(self.num_blocks, _PERIOD)):
            self.conv_page(b)  # raises when the history does not fit

    def _fp32_rows(self, b):
        """[lo, hi): the per-bank fp32 rows whose 4096 bytes lie inside block b."""
        S = int(self.block_bytes)
        return -(-b * S // FP32_PAGE), ((b + 1) * S) // FP32_PAGE

    def phase(self, b):
        """Byte offset, inside block b's range on a bank, of the first whole fp32 page: 0 for every block when a block
        holds a whole number of fp32 pages per bank; else it depends on the block (BF8 1024 on 8 banks: 0 for even
        blocks, 2048 for odd ones). A byte copy between blocks of different phase misplaces the state pages."""
        return -(-int(b) * int(self.block_bytes) // FP32_PAGE) * FP32_PAGE - int(b) * int(self.block_bytes)

    def _first_head_page(self, b):
        lo, _ = self._fp32_rows(b)
        return -(-lo * int(self.n_banks) // HALF_PAGES) * HALF_PAGES

    def _pool_heads(self, b):
        """Whole heads one pool of block b holds from _first_head_page on."""
        _, hi = self._fp32_rows(b)
        return max(0, hi * int(self.n_banks) - self._first_head_page(b)) // HEAD_PAGES

    def _bf16_rows(self, b):
        S = int(self.block_bytes)
        return -(-b * S // BF16_PAGE), ((b + 1) * S) // BF16_PAGE

    def head_pages(self, blocks):
        """int64 [..., Nv]: the first page of head h of each block in the two-pool page space (pages >= pool1_pages
        are V's, offset by pool1_pages). ``blocks`` int tensor of any shape."""
        b = torch.as_tensor(blocks, dtype=torch.int64)
        S, nb = int(self.block_bytes), int(self.n_banks)
        lo = (-(-b * S // FP32_PAGE)) * nb  # first whole fp32 page of the block (K and V have the same geometry)
        lo = -(-lo // HALF_PAGES) * HALF_PAGES
        h = torch.arange(self.nv, dtype=torch.int64)
        nk = self.heads_in_k
        in_k = (h < nk).to(torch.int64)
        page = lo.unsqueeze(-1) + HEAD_PAGES * torch.where(h < nk, h, h - nk)
        return page + (1 - in_k) * self.pool1_pages

    def conv_page(self, b):
        """First page of block b's conv history (conv_pages pages) in the two-pool bf16 page space: pages below
        conv_pages_total are V's flat bf16 view, the rest K's (offset by conv_pages_total). V after its heads first,
        else K after its heads (on 7 banks the head alignment can leave V no room)."""
        nb = int(self.n_banks)
        b16_lo, b16_hi = self._bf16_rows(b)
        head0 = self._first_head_page(b)
        for heads, offset in ((self.nv - self.heads_in_k, 0), (self.heads_in_k, self.conv_pages_total)):
            if heads <= 0:
                return b16_lo * nb + offset
            end = head0 + HEAD_PAGES * heads  # one past the pool's last head page
            after = max(b16_lo, 2 * ((end - 1) // nb + 1))  # bf16 rows after the last fp32 row a head touches
            if (b16_hi - after) * nb >= self.conv_pages:
                return after * nb + offset
            before = 2 * (head0 // nb)  # bf16 rows before the first fp32 row a head touches
            if (before - b16_lo) * nb >= self.conv_pages:
                return b16_lo * nb + offset
        raise ValueError(
            f"block {b}: no room for a conv history of {self.conv_pages} bf16 pages after the heads in V or in K "
            f"(bf16 pages [{b16_lo * nb}, {b16_hi * nb}))"
        )

    def check_block(self, b):
        """Raise unless block b's heads and conv pages are inside the block and disjoint (host tests and bind time)."""
        S, nb = int(self.block_bytes), int(self.n_banks)
        spans = []  # (pool, bank, byte lo, byte hi)
        for p in self.head_pages(torch.tensor([b]))[0].tolist():
            pool = 1 if p < self.pool1_pages else 2
            q = p if pool == 1 else p - self.pool1_pages
            for i in range(HEAD_PAGES):
                r, k = divmod(q + i, nb)
                spans.append((pool, k, r * FP32_PAGE, (r + 1) * FP32_PAGE))
        c0 = self.conv_page(b)
        pool, c0 = (2, c0) if c0 < self.conv_pages_total else (1, c0 - self.conv_pages_total)
        for i in range(self.conv_pages):
            r, k = divmod(c0 + i, nb)
            spans.append((pool, k, r * BF16_PAGE, (r + 1) * BF16_PAGE))
        for pool, k, lo, hi in spans:
            if lo < b * S or hi > (b + 1) * S:
                raise ValueError(f"block {b}: pool {pool} bank {k} bytes [{lo}, {hi}) outside [{b * S}, {(b + 1) * S})")
        spans.sort()
        for (p0, k0, lo0, hi0), (p1, k1, lo1, hi1) in zip(spans, spans[1:]):
            if (p0, k0) == (p1, k1) and lo1 < hi0:
                raise ValueError(f"block {b}: pool {p0} bank {k0} spans [{lo0}, {hi0}) and [{lo1}, {hi1}) overlap")


def conv_carry_pages(kc, C):
    """bf16 tiles of one conv history: (kc - 1) * C values, 1024 per tile."""
    return -(-(int(kc) - 1) * int(C) // 1024)


def kv_block_bytes_per_bank(shape, dtype_name, n_banks):
    """S of SpecBlockLayout for a [num_blocks, heads, block, head_dim] TILE cache."""
    _, heads, block, hd = (int(x) for x in shape)
    tiles = heads * block * hd // 1024
    if tiles % n_banks:
        raise ValueError(f"a block of {tiles} tiles does not split over {n_banks} DRAM banks (block {block})")
    return tiles // n_banks * _tile_bytes(dtype_name)


def gdn_groups_by_slot(per_layer_specs, is_mamba_entry):
    """{GDN layer index: group rank}. vLLM's uniform-page grouping splits the layers of one spec type into groups of
    equal size in layer order, and KV cache tensor j is shared by layer j of every group, so a GDN layer's group is its
    rank among the GDN layers of its layer slot. Checked against the block windows at the first step
    (SpecGdnBlockState.check_groups)."""
    seen = {}
    out = {}
    for i, e in enumerate(per_layer_specs):
        if is_mamba_entry(e):
            slot = e[2]
            out[i] = seen.get(slot, 0)
            seen[slot] = out[i] + 1
    return out


class StepWindows:
    """Host view of one step's GDN windows in ENGINE rows: per group g an int64 [B, 1 + K] block table (0 = null) and
    the int64 [B] start column (state_accepted_counts - 1)."""

    def __init__(self, blocks, start_col):
        self.blocks = blocks  # list over groups of [B, T]
        self.start_col = start_col  # [B]

    def ring_page_idx(self, layout, g):
        """int32 [B, T * Nv]: entry (u, t * Nv + h) = first page of head h in window column t of row u."""
        pages = layout.head_pages(self.blocks[g])  # [B, T, Nv]
        return pages.reshape(pages.shape[0], -1).to(torch.int32)

    def init_pages(self, layout, g):
        """int64 [B, Nv]: first page of head h in row u's start column."""
        b = self.blocks[g]
        start = torch.gather(b, 1, self.start_col.reshape(-1, 1)).reshape(-1)
        return layout.head_pages(start)


def ctrl_page_paged(mi, init_pages, parity, words, hold=()):
    """The gdn_spec_step ctrl page in ring_page_idx mode: word 0 = parity, words 1 .. B = mi (conv carry rows), words
    1 + B + u * Nv + h = first page of head h's start state (HOLD sentinel 0xFFFFFFFF for every head of a held row).
    Returns int32 [1, words]."""
    B, nv = init_pages.shape
    out = torch.zeros(words, dtype=torch.int64)
    out[0] = int(parity)
    m = torch.as_tensor(mi, dtype=torch.int64).reshape(-1).clone()
    init = init_pages.clone()
    for u in hold:
        m[u] = 0
        init[u] = 0xFFFFFFFF
    out[1 : 1 + B] = m
    out[1 + B : 1 + B + B * nv] = init.reshape(-1)
    return as_u32_words(out).reshape(1, -1)


def as_u32_words(x):
    """int64 values in [0, 2**32) as the int32 words with the same bits (the host staging dtype of uint32 pages)."""
    x = torch.as_tensor(x, dtype=torch.int64).clone()
    x[x >= 2**31] -= 2**32
    return x.to(torch.int32)


class SpecBlockBook:
    """Host bookkeeping of the GDN state windows (no device work): per group the ring_page_idx rows and the start pages
    of the staged verify, the window map {block: (engine row, column, group)} of each row's latest verify, the conv
    writes a verify leaves due (mode "align" boundaries) and what each vLLM block copy must do."""

    def __init__(self, layout, G, B, T, cache_mode="none", mamba_block_size=0):
        if cache_mode not in ("none", "align"):
            raise ValueError(f"mamba_cache_mode {cache_mode!r} not supported")
        if cache_mode == "align" and int(mamba_block_size) <= 0:
            raise ValueError("mamba_cache_mode align needs the MambaSpec block size")
        self.layout, self.G, self.B, self.T = layout, int(G), int(B), int(T)
        self.cache_mode, self.mamba_block_size = cache_mode, int(mamba_block_size)
        nv = layout.nv
        self.idx_host = [torch.zeros(self.B, self.T * nv, dtype=torch.int32) for _ in range(self.G)]
        self.init = [torch.zeros(self.B, nv, dtype=torch.int64) for _ in range(self.G)]
        self.window_map = {}  # block id -> (engine row, column, group) of the row's latest verify
        self.row_blocks = {}  # engine row -> the blocks it maps in window_map
        self.step_blocks = None  # the staged verify: [G] int64 [B, T] (engine rows)
        self.step_start = None  # int64 [B] start columns
        self.pending = []  # (engine row, column) whose conv rows are written once the staged verify ran
        self.prefill_blocks = None  # per group int64 [n]: the prefill step's running block of request u
        self.checkpoints = {}  # request u -> [(n, [G] blocks)]: the prefill step's checkpoint states (vLLM's blocks)
        self._pages = [torch.zeros(self.B, self.T, nv, dtype=torch.int64) for _ in range(self.G)]  # head pages
        self._rows = torch.arange(self.B)
        self._map_live, self._map_dirty = None, True  # live rows the window map was built for; a copy changed it
        # chained state moves of the staged verify: {engine row: ([G] start blocks [T], [G] windows [T, T])}, indexed by
        # the pending verify's accepted count - 1; the rows' window maps follow once that count is known (resolve_moves)
        self.moves = {}
        self.pending_blocks = []  # (row, column, group, block) conv writes that name their block (chained rows)

    def set_step(self, blocks, start_col, live_rows, boundary=()):
        """Stage one verify. ``blocks``: [G] int64 [B, T] in ENGINE rows (0 = the null block on rows no request holds);
        ``start_col``: int64 [B] (state_accepted_counts - 1); ``live_rows``: the engine rows the verify computes (the
        others are held). ``boundary``: (row, column) pairs whose conv rows take_pending returns after the verify.
        Returns the groups whose ring_page_idx rows changed (the device copy to make). Runs every decode step, so the
        head pages of a group and the window map are rebuilt only when its blocks, the live rows or a copy changed
        them."""
        if len(blocks) != self.G:
            raise ValueError(f"{len(blocks)} window groups, the model binds {self.G}")
        # chained rows of the previous verify whose count never reached the host: the device window rows are unknown
        stale = bool(self.moves)
        self._map_dirty = self._map_dirty or stale
        changed = []
        for g in range(self.G):
            if stale or self.step_blocks is None or not torch.equal(blocks[g], self.step_blocks[g]):
                self._pages[g] = self.layout.head_pages(blocks[g])  # [B, T, Nv]
                self.idx_host[g] = self._pages[g].reshape(self.B, -1).to(torch.int32)
                changed.append(g)
        self.step_blocks = [b.clone() for b in blocks] if changed else self.step_blocks
        self.step_start = torch.as_tensor(start_col, dtype=torch.int64).reshape(-1).clone()
        self.init = [p[self._rows, self.step_start] for p in self._pages]
        self.pending = [(int(u), int(t)) for u, t in boundary]
        self.pending_blocks = []
        self.moves = {}
        live = tuple(int(u) for u in live_rows)
        if changed or self._map_dirty or live != self._map_live:
            for u in live:
                self.forget_row(u)
                mapped = []
                for g in range(self.G):
                    for t, b in enumerate(blocks[g][u].tolist()):
                        if b > 0:
                            self.window_map[b] = (u, t, g)
                            mapped.append(b)
                self.row_blocks[u] = mapped
            self._map_live, self._map_dirty = live, False
        return changed

    def ctrl_rows(self, mi, parity, words, hold=(), start_col=None, pages=None):
        """int32 [G, words]: the ctrl page of each group (same parity and mi, the group's start pages; see
        ctrl_page_paged). ``start_col`` (int [B]): start pages at these columns of the staged windows instead of the
        staged start columns; ``pages`` (with ``start_col``): a start_pages_snapshot to read them from."""
        if start_col is not None and (self.step_blocks is not None or pages is not None):
            cols = torch.as_tensor(start_col, dtype=torch.int64).reshape(-1)
            init = torch.stack([p[self._rows, cols] for p in (self._start_pages() if pages is None else pages)])
        else:
            init = torch.stack(self.init)  # [G, B, Nv]
        B, nv = self.B, self.layout.nv
        m = torch.as_tensor(mi, dtype=torch.int64).reshape(-1).clone()
        hold = list(hold)
        if hold:
            init = init.clone()
            init[:, hold] = 0xFFFFFFFF
            m[hold] = 0
        out = torch.zeros(self.G, words, dtype=torch.int64)
        out[:, 0] = int(parity)
        out[:, 1 : 1 + B] = m
        out[:, 1 + B : 1 + B + B * nv] = init.reshape(self.G, -1)
        return as_u32_words(out)

    def set_start(self, cols, start_pos=None):
        """Move staged rows' start columns: {engine row: column} (a chained row the host stages after all). A row with
        chained state moves starts from its start block of count column + 1 and writes that count's window; with
        ``start_pos`` (int64 [B], the true start positions) its boundary conv write follows (mode align). Returns the
        groups whose ring_page_idx rows changed."""
        changed = set()
        for u, c in cols.items():
            u, c = int(u), int(c)
            self.step_start[u] = c
            mv = self.moves.pop(u, None)
            if mv is not None:
                self._adopt_window(u, c + 1, mv[1])
                changed.update(range(self.G))
                if start_pos is not None and self._at_boundary(int(start_pos[u])):
                    self.pending.append((u, 0))
            for g in range(self.G):
                self.init[g][u] = self._pages[g][u, c] if mv is None else self.layout.head_pages(mv[0][g][c])
        return sorted(changed)

    def _at_boundary(self, pos):
        return self.cache_mode == "align" and (pos + 1) % self.mamba_block_size == 0

    def set_moves(self, starts, windows, rows):
        """Chained state moves of the staged verify (after set_step): ``starts`` [G] int64 [B, T] start blocks and
        ``windows`` [G] int64 [B, T, T] state windows, entry [u, c - 1] for accepted count c of the pending verify;
        ``rows``: the chained engine rows (the others keep the staged window). The boundary conv writes set_step
        derived for these rows from provisional positions are dropped (chain_boundaries / set_start set them)."""
        self.moves = {int(u): ([s[int(u)].clone() for s in starts], [w[int(u)].clone() for w in windows]) for u in rows}
        self.pending = [(u, t) for u, t in self.pending if u not in self.moves]

    def chain_boundaries(self, start_lo):
        """Mode align, rows with chained state moves: ``start_lo`` (int64 [B]) = each row's start position for accepted
        count 1. The one count whose first query token is the last position of a block gets its conv write into that
        count's window column 0 (the boundary's block); at another count that block holds the row's state after
        token 0 (counts below) or is outside the row's window (counts above)."""
        for u, (_, win) in self.moves.items():
            for c in range(1, self.T + 1):
                if self._at_boundary(int(start_lo[u]) + c - 1):
                    self.pending_blocks.extend((u, 0, g, int(win[g][c - 1, 0])) for g in range(self.G))

    def resolve_moves(self, counts):
        """The pending verify of the staged chained rows accepted ``counts[u]`` (engine rows): their window maps and
        staged pages become those of that count."""
        moves, self.moves = self.moves, {}
        for u, (_, win) in moves.items():
            self._adopt_window(u, int(counts[u]), win)

    def _adopt_window(self, u, c, win):
        self.forget_row(u)
        mapped = []
        for g in range(self.G):
            w = win[g][c - 1]
            self.step_blocks[g][u] = w
            self._pages[g][u] = self.layout.head_pages(w)
            self.idx_host[g][u] = self._pages[g][u].reshape(-1).to(torch.int32)
            for t, b in enumerate(w.tolist()):
                if b > 0:
                    self.window_map[b] = (u, t, g)
                    mapped.append(b)
        self.row_blocks[u] = mapped

    def pages_cat(self):
        """float32 [B, T * G * Nv]: the chain body's start-page table, gathered at column mi': column t * G * Nv + g * Nv
        + h = the first page of head h of the block a row starts from for start column t, its window column t, or, on
        a row with chained state moves, its start block for accepted count t + 1."""
        return torch.stack(self._start_pages(), dim=2).reshape(self.B, -1).to(torch.float32)

    def start_pages_snapshot(self):
        """A copy of the start-page tables (_start_pages) that later stagings do not change."""
        return [p.clone() for p in self._start_pages()]

    def _start_pages(self):
        """[G] int64 [B, T, Nv]: head pages of the start block per start column (column c - 1 for accepted count c)."""
        if not self.moves:
            return self._pages
        pages = [p.clone() for p in self._pages]
        for u, (st, _) in self.moves.items():
            for g in range(self.G):
                pages[g][u] = self.layout.head_pages(st[g])
        return pages

    def windows_cat(self):
        """float32 [B, T * G * T * Nv]: the chain body's window-page table (QWEN36_MAMBA_CHAIN_MOVES), gathered at
        accepted count c = mi' + 1: column (c - 1) * G * T * Nv + g * T * Nv + t * Nv + h = the first page of head h of
        the block that takes the state after query token t, the staged window for every count on a row without chained
        state moves."""
        nv = self.layout.nv
        out = torch.stack(self._pages, dim=1).unsqueeze(1).expand(self.B, self.T, self.G, self.T, nv).clone()
        for u, (_, win) in self.moves.items():
            out[u] = torch.stack([self.layout.head_pages(win[g]) for g in range(self.G)], dim=1)
        return out.reshape(self.B, -1).to(torch.float32)

    def step_start_blocks(self, u):
        """[G] block ids of engine row u's start column in the staged step, or None."""
        if self.step_blocks is None:
            return None
        c = int(self.step_start[u])
        out = [int(b[u, c]) for b in self.step_blocks]
        return out if all(x > 0 for x in out) else None

    def chain_copy_column(self, u, g, src):
        """The window column of engine row u's latest verify (group g) whose state block ``src`` holds, or None when src is
        not a window block of that verify (its block then holds its whole state, conv history included). A row whose
        chained state moves are still unresolved (moves) looks src up in every count's window and needs one column."""
        mv = self.moves.get(int(u))
        if mv is None:
            ent = self.window_map.get(int(src))
            return ent[1] if ent is not None and ent[0] == int(u) and ent[2] == int(g) else None
        cols = {t for w in mv[1][g].tolist() for t, b in enumerate(w) if b == int(src)}
        if len(cols) > 1:
            raise RuntimeError(f"row {u} group {g}: block {src} sits at window columns {sorted(cols)} by count")
        return cols.pop() if cols else None

    def take_pending(self):
        """[(row, column, group, block)] conv writes the staged verify left due; clears them."""
        pending, self.pending = self.pending, []
        out, self.pending_blocks = list(self.pending_blocks), []
        for u, t in pending:
            for b in list(self.row_blocks.get(u, ())):
                ent = self.window_map.get(b)
                if ent is not None and ent[0] == u and ent[1] == t:
                    out.append((u, t, ent[2], b))
        return out

    def copy_plan(self, copies):
        """The device work of vLLM's (src, dst) block copies, in list order: ("copy", src, dst) = the whole block of every
        K and V buffer, when both blocks hold their pages at the same offsets (SpecBlockLayout.phase); else
        ("relayout", src, dst) = every head and the conv history read at src's pages and written at dst's (every copy
        TT receives is a state copy: vLLM copies attention blocks only for partial prefix hits, which the spec class
        refuses with such a layout); ("conv", row, column, group, dst) = that window column's conv rows into dst, for a
        source that is a window column of a row's latest verify (its block holds a stale history). dst then names the
        same state."""
        ops = []
        for src, dst in copies:
            src, dst = int(src), int(dst)
            if src == dst:
                continue
            if src <= 0 or dst <= 0:
                raise ValueError(f"block copy {src} -> {dst} names the null block 0")
            same = self.layout.phase(src) == self.layout.phase(dst)
            ops.append(("copy" if same else "relayout", src, dst))
            ent = self.window_map.get(src)
            self.forget_block(dst)
            if ent is not None:
                ops.append(("conv", ent[0], ent[1], ent[2], dst))
                self._map_dirty = True
                self.window_map[dst] = ent
                self.row_blocks.setdefault(ent[0], []).append(dst)
        return ops

    def forget_row(self, u):
        self._map_dirty = True
        for b in self.row_blocks.pop(int(u), ()):
            if self.window_map.get(b, (None,))[0] == int(u):
                del self.window_map[b]

    def forget_block(self, b):
        ent = self.window_map.pop(int(b), None)
        if ent is not None:
            self._map_dirty = True
        if ent is not None and int(b) in self.row_blocks.get(ent[0], ()):
            self.row_blocks[ent[0]].remove(int(b))

    def set_prefill_blocks(self, per_group):
        """The prefill step's state blocks: [G] int tensors, entry u = request u's running block (window column 0)."""
        if len(per_group) != self.G:
            raise ValueError(f"{len(per_group)} groups of prefill blocks, the model binds {self.G}")
        self.prefill_blocks = [torch.as_tensor(t).reshape(-1).to(torch.int64) for t in per_group]

    def prefill_block(self, g, u):
        if u == WARMUP_REQ:
            return 0
        if self.prefill_blocks is None:
            raise ValueError("QWEN36_MAMBA_SPEC prefill without state_block_ids_per_layer")
        b = int(self.prefill_blocks[g][u])
        if b <= 0:
            raise ValueError(f"prefilled request {u}: group {g} state block is the null block {b}")
        return b

    def set_prefill_checkpoints(self, per_group, positions):
        """vLLM's prefill checkpoints of this step (MambaSpec.num_prefill_checkpoint_blocks): request u must leave the
        state after exactly n tokens in block per_group[g][u, j] of every group g, for each entry j with
        n = positions[u, j] > 0. ``positions`` None (or no entry): none."""
        self.checkpoints = {}
        if positions is None:
            return
        pos = torch.as_tensor(positions).to(torch.int64)
        pos = pos.reshape(pos.shape[0], -1)
        if not bool((pos > 0).any()):
            return
        if per_group is None or len(per_group) != self.G:
            raise ValueError(f"state_checkpoint_positions needs [{self.G}] groups of checkpoint blocks")
        ids = [torch.as_tensor(t).reshape(pos.shape[0], -1) for t in per_group]
        for u, j in (pos > 0).nonzero().tolist():
            blocks = [int(t[u, j]) for t in ids]
            if min(blocks) <= 0:
                raise ValueError(f"request {u}: prefill checkpoint at {int(pos[u, j])} names the null block: {blocks}")
            self.checkpoints.setdefault(u, []).append((int(pos[u, j]), blocks))

    def prefill_checkpoints(self, u):
        """Request u's (n, [G] blocks) of this step, in position order."""
        return sorted(self.checkpoints.get(int(u), ()))


class CheckpointPoints(list):
    """The segment ends of a prefill that leaves vLLM's prefill checkpoints inside its chunk: after the segment that
    ends at n, the bound scratch holds the state after n tokens, and ``save(n)`` writes it into vLLM's block for n."""

    def __init__(self, entries, write):
        super().__init__(n for n, _ in entries)
        self._blocks = dict(entries)
        self._write = write

    def save(self, n):
        self._write(self._blocks[int(n)])


def group_windows(windows_per_layer, layer_groups, G):
    """[G] int64 tensors from the plugin's per-layer list (one tensor per GDN layer, None for attention layers). Raises
    unless every layer of a group got the same window: the layer-slot grouping (gdn_groups_by_slot) must match vLLM's
    KV cache groups."""
    by_group = {}
    for i, g in layer_groups.items():
        w = windows_per_layer[i]
        if w is None:
            raise ValueError(f"GDN layer {i} has no state block window")
        prev = by_group.setdefault(g, w)
        if prev is not w and not torch.equal(torch.as_tensor(prev), torch.as_tensor(w)):
            raise RuntimeError(
                f"GDN layer {i}: its state block window differs from the other layers of inferred group {g}; the "
                "layer-slot grouping (gdn_groups_by_slot) does not match vLLM's KV cache groups"
            )
    return [torch.as_tensor(by_group[g]).to(torch.int64) for g in range(G)]


class SpecGdnBlockState:
    """Binds every GDN layer of ``model`` (the MTP spec class, T = K + 1 verify rows per user) to page views of its layer
    slot's KV buffers: dn._spec_paged (rec views of K and V, conv view of V, the group's ring_page_idx tensor). The host
    state is a SpecBlockBook (``book``); this class runs its device work."""

    def __init__(
        self,
        model,
        per_layer_specs,
        kv_caches,
        T,
        layout="two_pool",
        cache_mode="none",
        mamba_block_size=0,
        partial_prefix_hits=False,
    ):
        import ttnn
        from models.demos.blackhole.qwen36.tt.gdn_block_state import is_mamba_entry

        mesh = model.mesh_device
        self.model, self.mesh, self.T = model, mesh, int(T)
        self.bmax = int(model.args.max_batch_size)
        attn = [(i, e) for i, e in enumerate(per_layer_specs) if not is_mamba_entry(e)]
        if [i for i, _ in attn] != list(model._attention_layer_indices):
            raise ValueError("attention entries of per_layer_specs must be the model's attention layers")
        kv_by_slot = {e[2]: kv_caches[n] for n, (_, e) in enumerate(attn)}
        self.layer_groups = gdn_groups_by_slot(per_layer_specs, is_mamba_entry)
        self.G = max(self.layer_groups.values()) + 1 if self.layer_groups else 0
        n_banks = int(mesh.dram_grid_size().x)
        self.layers = []  # (layer index, TPGatedDeltaNet, group)
        self._slot_layers = {}  # layer slot -> its first GDN layer
        lay = None
        for i, g in sorted(self.layer_groups.items()):
            dn = model.layers[i].attention
            if model.layers[i].is_full_attention:
                raise ValueError(f"layer {i} has a MambaSpec but is a full-attention layer")
            if not getattr(dn, "_spec_fused", False):
                raise ValueError(
                    "QWEN36_MAMBA_SPEC (spec class) needs the gdn_spec_step verify (QWEN36_VERIFY_SPEC_STEP=1)"
                )
            k, v = kv_by_slot[per_layer_specs[i][2]]
            if k.dtype != v.dtype or tuple(k.shape) != tuple(v.shape):
                raise ValueError(f"layer {i}: K {k.dtype} {tuple(k.shape)} and V {v.dtype} {tuple(v.shape)} differ")
            S = kv_block_bytes_per_bank(k.shape, _dtype_name(k.dtype), n_banks)
            this = SpecBlockLayout(S, n_banks, int(k.shape[0]), dn.Nv, conv_carry_pages(dn.K, dn.qkv_dim_tp), layout)
            if lay is None:
                lay = this
            elif (this.block_bytes, this.nv, this.conv_pages, this.num_blocks) != (
                lay.block_bytes,
                lay.nv,
                lay.conv_pages,
                lay.num_blocks,
            ):
                raise ValueError("every GDN layer must have the same state geometry")
            half = [lay.pool1_pages // HALF_PAGES, 1, dn.Dk // 2, dn.Dv]
            dn._spec_paged = _LayerViews(
                rec1=ttnn.experimental.alias_view(k, half, ttnn.float32, check_slices=False),
                rec2=ttnn.experimental.alias_view(v, half, ttnn.float32, check_slices=False),
                conv=ttnn.experimental.alias_view(
                    v, [lay.conv_pages_total, 1, 32, 32], ttnn.bfloat16, check_slices=False
                ),
                conv_k=ttnn.experimental.alias_view(
                    k, [lay.conv_pages_total, 1, 32, 32], ttnn.bfloat16, check_slices=False
                ),
                group=g,
                two_pool=lay.heads_in_k < dn.Nv,
            )
            self.layers.append((i, dn, g))
            self._slot_layers.setdefault(per_layer_specs[i][2], dn)  # its views cover the slot's K and V
        self.layout = lay
        if partial_prefix_hits and lay.block_bytes % FP32_PAGE:
            raise ValueError(
                f"QWEN36_MAMBA_SPEC: a block of {lay.block_bytes} B per bank is not a whole number of fp32 pages, so "
                "state copies move page by page, and partial prefix hits (prefix_match_unit below the block size) "
                "make vLLM copy attention blocks too; use a block of whole fp32 pages (BF8 2048 on 8 banks) or "
                "block-sized prefix hits"
            )
        self.book = SpecBlockBook(lay, self.G, self.bmax, self.T, cache_mode, mamba_block_size)
        self.kv_block_size = int(kv_caches[0][0].shape[2])
        self._zeros = None  # [1, H, n * kv_block_size, D] zero source of _zero_now
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self._idx_dev = [
            ttnn.from_torch(h, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
            for h in self.book.idx_host
        ]
        for _, dn, g in self.layers:
            dn._spec_paged.idx = self._idx_dev[g]
        # the chain body's start-page table: fp32 TILE [B, T * G * Nv], column t * G * Nv + g * Nv + h = first page of
        # head h of group g in window column t (exact in fp32: below 2^24 pages)
        self.pages_dev = ttnn.from_torch(
            self.book.pages_cat(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )
        # QWEN36_MAMBA_CHAIN_MOVES: the chain body's window-page table (SpecBlockBook.windows_cat), selected by count
        from models.demos.blackhole.qwen36.tt.tp_common import mamba_chain_moves_enabled, spec_chain_kernel_enabled

        self.windows_dev = None
        if mamba_chain_moves_enabled():
            self.windows_dev = ttnn.from_torch(
                self.book.windows_cat(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
            )
        # QWEN36_SPEC_CHAIN_KERNEL: pages_dev's words as uint32 RM [B, T * G * Nv], the chain kernel's start-page table
        self.pages_rm_dev = None
        if spec_chain_kernel_enabled():
            self.pages_rm_dev = ttnn.from_torch(
                as_u32_words(self.book.pages_cat().to(torch.int64)),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
            )
        self._up = {}  # host copy of the last upload of pages_dev / windows_dev
        self._tables_moved = False  # the last upload held a row's chained state moves
        self.moves_counts = None  # (wait=False) -> engine-row accepted counts the staged moves follow, None until read
        self.chain_copies = []  # capture_chain_copies: (row, group, src, dst, window column or None, counts)

    cache_mode = property(lambda self: self.book.cache_mode)
    mamba_block_size = property(lambda self: self.book.mamba_block_size)

    # ---------------------------------------------------------------- steps
    def group_windows(self, windows_per_layer):
        return group_windows(windows_per_layer, self.layer_groups, self.G)

    def set_step(self, blocks, start_col, live_rows, boundary=()):
        self.resolve_moves()
        changed = self.book.set_step(blocks, start_col, live_rows, boundary)
        self._upload_idx(changed)
        if changed or self._tables_moved:
            self._upload_tables()

    def _upload_idx(self, groups):
        import ttnn

        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for g in groups:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    self.book.idx_host[g], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
                ),
                self._idx_dev[g],
            )

    def _upload_tables(self):
        """pages_dev (and windows_dev) from the book, each only when it differs from its last upload."""
        import ttnn

        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        tables = [("pages", self.book.pages_cat(), self.pages_dev)]
        if self.windows_dev is not None:
            tables.append(("windows", self.book.windows_cat(), self.windows_dev))
        for name, host, dev in tables:
            if name in self._up and torch.equal(self._up[name], host):
                continue
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep), dev
            )
            if name == "pages" and self.pages_rm_dev is not None:
                words = as_u32_words(host.to(torch.int64))
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(words, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep),
                    self.pages_rm_dev,
                )
            self._up[name] = host
        self._tables_moved = bool(self.book.moves)

    def set_moves(self, starts, windows, rows):
        """A chained step's state moves (SpecBlockBook.set_moves), engine rows; the chain body's tables follow."""
        if self.windows_dev is None:
            raise RuntimeError("chained state moves without QWEN36_MAMBA_CHAIN_MOVES=1 (no window-page table)")
        self.book.set_moves(starts, windows, rows)
        self._upload_tables()

    def chain_boundaries(self, start_lo):
        self.book.chain_boundaries(start_lo)

    def set_start(self, cols, start_pos=None):
        self._upload_idx(self.book.set_start(cols, start_pos))

    def resolve_moves(self, wait=False):
        """Adopt the window of each chained row's actual count once the host has the chained verify's counts (the
        runner read it, or ``wait``: read it now); until then the rows keep the optimistic window map."""
        if not self.book.moves or self.moves_counts is None:
            return
        counts = self.moves_counts(wait)
        if counts is not None:
            self.book.resolve_moves(counts)
            self.moves_counts = None

    def capture_chain_copies(self, copies, rows):
        """A chained step's boundary copies (state_chain_copies_per_layer), before set_step maps this step's windows:
        ``copies`` [G] int64 [B, T, 2] in engine rows, entry [u, c - 1] = (src, dst) when the pending verify accepts c
        tokens of row u; ``rows`` = the chained rows. Records each distinct pair with the counts that select it and the
        pending verify's window column whose conv rows src stands for (src's block holds only its recurrent state)."""
        try:
            self.resolve_moves()
            found = self._chain_copy_entries(copies, rows)
        except RuntimeError:  # a source the older chained verify's count decides: read that count, then look again
            self.resolve_moves(wait=True)
            found = self._chain_copy_entries(copies, rows)
        self.chain_copies = found

    def _chain_copy_entries(self, copies, rows):
        out = []
        for g, cp in enumerate(copies):
            for u in rows:
                by_pair = {}
                for c in range(1, self.T + 1):
                    src, dst = int(cp[u, c - 1, 0]), int(cp[u, c - 1, 1])
                    if src == dst:
                        continue
                    if src <= 0 or dst <= 0:
                        raise ValueError(f"row {u} group {g}: chained copy {src} -> {dst} names the null block 0")
                    by_pair.setdefault((src, dst), []).append(c)
                for (src, dst), cs in by_pair.items():
                    out.append((int(u), g, src, dst, self.book.chain_copy_column(u, g, src), tuple(cs)))
        return out

    def run_chain_copies(self, counts=None, result=None):
        """Run the copies capture_chain_copies recorded, before the step reads or writes any state, and forget them.
        ``counts`` (engine row -> the pending verify's accepted count, known on the host): those copies run. ``result``
        (the pending fused step's int32 ROW_MAJOR result buffer, row RES_COUNTS = counts by engine row): the counts stay
        on the device, so every copy writes into dst where its row's count selects it and into the null block 0 where
        not (padding rows write their states there too)."""
        todo, self.chain_copies = self.chain_copies, []
        if counts is not None:
            todo = [e for e in todo if int(counts.get(e[0], 0)) in e[5]]
        if not todo:
            return
        for e in todo:
            self.book.forget_block(e[3])
        with self._timed("chained block copies", len(todo)), _PageTables(self) as pts:
            if result is None:
                from models.demos.blackhole.qwen36.tt.gdn_block_state import copy_kv_blocks

                for u, g, src, dst, t, _ in todo:
                    if self.layout.phase(src) == self.layout.phase(dst):
                        copy_kv_blocks(self.model, [(src, dst)])
                    else:
                        self._relayout(src, dst, pts)
                    if t is not None:
                        self.materialize(u, t, g, dst, pts)
            else:
                self._chain_copies_on_device(todo, result, pts)

    def _chain_copies_on_device(self, todo, result, pts):
        import ttnn
        from models.demos.blackhole.qwen36.tt.spec_chain import RES_COUNTS

        f32, mc, lay = ttnn.float32, ttnn.DRAM_MEMORY_CONFIG, self.layout
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        W = int(result.shape[1])
        row = ttnn.slice(result, [RES_COUNTS, 0], [RES_COUNTS + 1, W], memory_config=mc)
        cnt = ttnn.typecast(ttnn.to_layout(row, ttnn.TILE_LAYOUT), f32)  # [1, W]: integers, exact in fp32
        ttnn.deallocate(row)
        owned = [cnt]

        def selected(m, dst0, null0, n):
            """int32 ROW_MAJOR [1, n] page table: pages dst0 .. dst0 + n - 1 where m is 1, null0 .. where m is 0."""
            host = lambda x: ttnn.from_torch(x, dtype=f32, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=rep)
            base = host((float(null0) + torch.arange(n, dtype=torch.float32)).reshape(1, n))
            step = host(torch.full((1, n), float(dst0 - null0)))
            sel = ttnn.add(ttnn.multiply(step, m), base)
            out = ttnn.to_layout(ttnn.typecast(sel, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT)
            owned.extend((base, step, sel, out))
            return out

        heads = lambda b: lay.head_pages(torch.tensor([b]))[0]
        for u, g, src, dst, t, cs in todo:
            cu = ttnn.slice(cnt, [0, u], [1, u + 1])  # [1, 1]
            m = None
            for c in cs:
                e = ttnn.typecast(ttnn.to_layout(ttnn.eq(cu, float(c)), ttnn.TILE_LAYOUT), f32)
                m = e if m is None else ttnn.add(m, e)
            owned.extend((cu, m))
            ps, pd, p0 = heads(src), heads(dst), heads(0)
            tabs = {}  # pool -> its page table: a block names the same pages in every slot's views
            for dn in self._slot_layers.values():  # every slot's views: the bytes of src belong to group g in each slot
                for k, (view, h0, h1, off) in enumerate(self._pools(dn)):
                    if h1 <= h0:
                        continue
                    n = 2 * (h1 - h0)
                    s0 = (int(ps[h0]) - off) // HALF_PAGES
                    if k not in tabs:
                        tabs[k] = selected(m, (int(pd[h0]) - off) // HALF_PAGES, (int(p0[h0]) - off) // HALF_PAGES, n)
                    tab = tabs[k]
                    part = ttnn.slice(view, (s0, 0, 0, 0), (s0 + n, 1, dn.Dk // 2, dn.Dv), memory_config=mc)
                    x = ttnn.reshape(part, (1, 1, n * dn.Dk // 2, dn.Dv))
                    ttnn.experimental.paged_fill_cache(view, x, tab, batch_idx=0)
                    ttnn.deallocate(part)
            cd, c0 = lay.conv_page(dst), lay.conv_page(0)
            if (cd < lay.conv_pages_total) != (c0 < lay.conv_pages_total):
                raise RuntimeError(f"block {dst} and the null block keep their conv history in different views")
            total = lay.conv_pages_total
            q = lambda page: page if page < total else page - total
            ctab = selected(m, q(cd), q(c0), lay.conv_pages)
            if t is None:  # src's block holds its conv history: copy it
                for dn in self._slot_layers.values():
                    vs, qs = dn._spec_paged.conv_at(lay.conv_page(src), total)
                    vd, _ = dn._spec_paged.conv_at(cd, total)
                    part = ttnn.slice(vs, (qs, 0, 0, 0), (qs + lay.conv_pages, 1, 32, 32), memory_config=mc)
                    x = ttnn.reshape(part, (1, 1, lay.conv_pages * 32, 32))
                    ttnn.experimental.paged_fill_cache(vd, x, ctab, batch_idx=0)
                    ttnn.deallocate(part)
            else:
                self.materialize(u, t, g, dst, pts, table=ctab)
        for x in owned:
            ttnn.deallocate(x)

    def ctrl_rows(self, mi, parity, words, hold=(), start_col=None, pages=None):
        return self.book.ctrl_rows(mi, parity, words, hold, start_col, pages)

    def step_start_blocks(self, u):
        return self.book.step_start_blocks(u)

    def after_verify(self):
        """Once the staged verify is enqueued and the window parity flipped (verify_win_cur is the window it writes):
        enqueue the conv writes it left due behind it."""
        due = self.book.take_pending()
        with self._timed("boundary conv writes", len(due)), _PageTables(self) as pts:
            for u, t, g, b in due:
                self.materialize(u, t, g, b, pts)

    def _timed(self, kind, n):
        """QWEN36_MAMBA_SPEC_TIMING=1: synchronize around a block operation and log its milliseconds (the synchronizes
        cost time of their own; off by default)."""
        state = self

        class _Timer:
            def __enter__(self):
                self.on = n > 0 and os.environ.get("QWEN36_MAMBA_SPEC_TIMING", "0") == "1"
                if self.on:
                    import ttnn

                    ttnn.synchronize_device(state.mesh)
                    self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                if self.on and exc[0] is None:
                    from loguru import logger

                    import ttnn

                    ttnn.synchronize_device(state.mesh)
                    logger.info(f"[mamba_spec] {kind}: {n} in {(time.perf_counter() - self.t0) * 1e3:.2f} ms")

        return _Timer()

    def forget_row(self, u):
        self.book.forget_row(u)

    # ---------------------------------------------------------------- prefill
    def set_prefill_blocks(self, state_block_ids_per_layer):
        ids = self.group_windows(state_block_ids_per_layer)
        self.book.set_prefill_blocks([t.reshape(t.shape[0], -1)[:, 0] for t in ids])

    def set_prefill_checkpoints(self, ids_per_layer, positions):
        """The plugin's state_checkpoint_ids_per_layer / state_checkpoint_positions (see
        SpecBlockBook.set_prefill_checkpoints)."""
        has = positions is not None and bool((torch.as_tensor(positions) > 0).any())
        if has and ids_per_layer is None:
            raise ValueError("state_checkpoint_positions without state_checkpoint_ids_per_layer")
        self.book.set_prefill_checkpoints(self.group_windows(ids_per_layer) if has else None, positions)

    def prefill_checkpoints(self, u):
        return self.book.prefill_checkpoints(u)

    @property
    def prefill_blocks(self):
        return self.book.prefill_blocks

    @prefill_blocks.setter
    def prefill_blocks(self, per_group):
        self.book.set_prefill_blocks(per_group)

    def prefill_block(self, g, u):
        return self.book.prefill_block(g, u)

    def restore(self, u):
        """Load request u's state block into the bound B=1 prefill scratch of every GDN layer (rec_state [1, Nv, Dk, Dv]
        fp32, conv_carry [1, K - 1, C] bf16). Same contract as GdnPrefixStateCache.restore."""
        import ttnn

        for _, dn, g in self.layers:
            b = self.book.prefill_block(g, u)
            rec = self._read_rec(dn, b)
            ttnn.copy(rec, dn.rec_state)
            ttnn.deallocate(rec)
            carry = self._read_carry(dn, b)
            ttnn.copy(carry, dn.conv_carry)
            ttnn.deallocate(carry)

    def write_prefill(self, u):
        """Write the bound B=1 prefill scratch (rec_state, conv_carry) of every GDN layer into request u's state block."""
        self._write_scratch([self.book.prefill_block(g, u) for g in range(self.G)], "prefill block writes")

    def write_checkpoint(self, blocks):
        """Write the bound B=1 prefill scratch into ``blocks`` ([G] block ids, one per group): a prefill checkpoint,
        the state after the tokens the prefill has run so far."""
        self._write_scratch(blocks, "checkpoint block writes")

    def _write_scratch(self, blocks, kind):
        import ttnn

        with self._timed(kind, len(self.layers)), _PageTables(self) as pts:
            for _, dn, g in self.layers:
                b = int(blocks[g])
                self.book.forget_block(b)
                self._write_rec(dn, b, dn.rec_state, pts)
                rm = ttnn.to_layout(dn.conv_carry, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                self._write_carry(dn, b, rm, pts)
                ttnn.deallocate(rm)

    def warmup(self):
        """Run every eager program of the block paths once on vLLM's null block 0 (prefill write and restore on the
        bound B=1 scratch, the conv write from the window, the whole-block copy and zeroing), so none compiles after
        the decode traces are captured."""
        import ttnn
        from models.demos.blackhole.qwen36.tt.gdn_block_state import zero_kv_blocks

        model = self.model
        prev = model._bind_gdn_prefill_scratch()
        try:
            self.write_prefill(WARMUP_REQ)
            self.restore(WARMUP_REQ)
        finally:
            model._unbind_gdn_prefill_scratch(prev)
        with _PageTables(self) as pts:
            for g in range(self.G):
                self.materialize(0, 0, g, 0, pts)
        mesh = model.mesh_device
        pt = ttnn.from_torch(
            torch.tensor([[0]], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for kv in model._paged_kv_caches:  # copy_kv_blocks' programs, block 0 onto itself
            for t in kv:
                blk = ttnn.slice(
                    t, (0, 0, 0, 0), [1] + [int(x) for x in list(t.shape)[1:]], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.experimental.paged_fill_cache(t, blk, pt, batch_idx=0)
                ttnn.deallocate(blk)
        ttnn.deallocate(pt)
        zero_kv_blocks(model, [0])
        self._zero_now([0])

    # ---------------------------------------------------------------- block hooks
    def copy_blocks(self, copies):
        from models.demos.blackhole.qwen36.tt.gdn_block_state import copy_kv_blocks

        self.resolve_moves()
        ops = self.book.copy_plan(copies)
        n_conv = sum(1 for op in ops if op[0] == "conv")
        with self._timed(f"block copies ({n_conv} with conv writes)", len(ops) - n_conv), _PageTables(self) as pts:
            for op in ops:
                if op[0] == "copy":
                    copy_kv_blocks(self.model, [(op[1], op[2])])
                elif op[0] == "relayout":
                    self._relayout(op[1], op[2], pts)
                else:
                    self.materialize(*op[1:], pts)

    def _relayout(self, src, dst, pts):
        """Copy block src's state into block dst page by page with the layout of each (the blocks hold their whole
        fp32 pages at different offsets): in every layer slot, the head pages of each pool and the conv history."""
        import ttnn

        lay, mc = self.layout, ttnn.DRAM_MEMORY_CONFIG
        ps, pd = lay.head_pages(torch.tensor([src]))[0], lay.head_pages(torch.tensor([dst]))[0]
        for dn in self._slot_layers.values():
            for view, h0, h1, off in self._pools(dn):
                if h1 <= h0:
                    continue
                n = 2 * (h1 - h0)  # [Dk / 2, Dv] slices
                s0 = (int(ps[h0]) - off) // HALF_PAGES
                part = ttnn.slice(view, (s0, 0, 0, 0), (s0 + n, 1, dn.Dk // 2, dn.Dv), memory_config=mc)
                x = ttnn.reshape(part, (1, 1, n * dn.Dk // 2, dn.Dv))
                ttnn.experimental.paged_fill_cache(view, x, pts((int(pd[h0]) - off) // HALF_PAGES, n), batch_idx=0)
                ttnn.deallocate(part)
            vs, qs = dn._spec_paged.conv_at(lay.conv_page(src), lay.conv_pages_total)
            vd, qd = dn._spec_paged.conv_at(lay.conv_page(dst), lay.conv_pages_total)
            part = ttnn.slice(vs, (qs, 0, 0, 0), (qs + lay.conv_pages, 1, 32, 32), memory_config=mc)
            x = ttnn.reshape(part, (1, 1, lay.conv_pages * 32, 32))
            ttnn.experimental.paged_fill_cache(vd, x, pts(qd, lay.conv_pages), batch_idx=0)
            ttnn.deallocate(part)

    def zero_blocks(self, block_ids):
        """vLLM's new_block_ids_to_zero, zeroed now (vLLM applies its block copies after the zeroing, and a copy can
        write one of these blocks). vLLM lists only blocks new to an attention group (it records new block ids for
        AttentionSpec managers), so a GDN state block is never zeroed: the prefill or a copy writes it whole, or it
        holds a valid state on a prefix hit."""
        ids = [int(b) for b in block_ids]
        for b in ids:
            self.book.forget_block(b)
        if ids:
            with self._timed("zeroed attention blocks", len(ids)):
                self._zero_now(ids)

    def _zero_now(self, ids):
        """Zero blocks ``ids`` in every K and V buffer, ZERO_BLOCKS_PER_FILL blocks per paged_fill_cache (the page
        table repeats the last block to fill the source's length)."""
        import ttnn

        kv = self.model._paged_kv_caches
        k0 = kv[0][0]
        if self._zeros is None:
            shape = [int(x) for x in k0.shape]
            tile_bytes = {"bfloat8_b": 1088, "bfloat16": 2048, "float32": 4096}[_dtype_name(k0.dtype)]
            block_bytes = shape[1] * shape[2] * shape[3] // 1024 * tile_bytes
            n = max(1, min(ZERO_BLOCKS_PER_FILL, ZERO_SOURCE_MAX_BYTES // block_bytes))
            self._zeros = ttnn.from_torch(
                torch.zeros([1, shape[1], n * shape[2], shape[3]], dtype=torch.bfloat16),
                dtype=k0.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        n = int(self._zeros.shape[2]) // self.kv_block_size
        for i in range(0, len(ids), n):
            chunk = list(ids[i : i + n])
            chunk += [chunk[-1]] * (n - len(chunk))
            pt = ttnn.from_torch(
                torch.tensor([chunk], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            for k, v in kv:
                ttnn.experimental.paged_fill_cache(k, self._zeros, pt, batch_idx=0)
                ttnn.experimental.paged_fill_cache(v, self._zeros, pt, batch_idx=0)
            ttnn.deallocate(pt)

    def materialize(self, u, t, g, dst, pts, table=None):
        """Write the conv history after token t of engine row u's latest verify (rows t + 1 .. t + kc - 1 of the window
        the next verify reads, verify_win_cur) into block dst of every GDN layer of group g (``table``: a device page
        table to write through instead of dst's conv pages, in dst's view)."""
        import ttnn

        mc = ttnn.DRAM_MEMORY_CONFIG
        for _, dn, gg in self.layers:
            if gg != g:
                continue
            kc, C = dn.K, dn.qkv_dim_tp
            L = kc - 1 + self.T
            row = ttnn.slice(dn.verify_win_cur(), (u, 0, 0), (u + 1, L, C), memory_config=mc)
            rm = ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
            ttnn.deallocate(row)
            carry = ttnn.slice(rm, (0, t + 1, 0), (1, t + kc, C), memory_config=mc)
            ttnn.deallocate(rm)
            self._write_carry(dn, dst, carry, pts, table)
            ttnn.deallocate(carry)

    # ---------------------------------------------------------------- page I/O (eager)
    def _pools(self, dn):
        nk = self.layout.heads_in_k
        out = [(dn._spec_paged.rec1, 0, min(nk, dn.Nv), 0)]
        if nk < dn.Nv:
            out.append((dn._spec_paged.rec2, nk, dn.Nv, self.layout.pool1_pages))
        return out

    def _page_table(self, first, n):
        import ttnn

        return ttnn.from_torch(
            (int(first) + torch.arange(n, dtype=torch.int32)).reshape(1, n),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _write_rec(self, dn, b, rec, pts):
        """rec [1, Nv, Dk, Dv] fp32 TILE -> block b's head pages (paged_fill_cache over [Dk / 2, Dv] view blocks: a
        head's 16 pages are its two half-head view blocks, in the tile order of [Dk, Dv])."""
        import ttnn

        pages = self.layout.head_pages(torch.tensor([b]))[0]
        for view, h0, h1, off in self._pools(dn):
            if h1 <= h0:
                continue
            n = h1 - h0
            # every head in one pool (k_only): a whole-tensor slice returns rec's own buffer, which must survive
            whole = h0 == 0 and h1 == int(rec.shape[1])
            part = (
                rec
                if whole
                else ttnn.slice(rec, (0, h0, 0, 0), (1, h1, dn.Dk, dn.Dv), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            )
            src = ttnn.reshape(part, (1, 1, n * dn.Dk, dn.Dv))
            first = int(pages[h0]) - off
            if first % HALF_PAGES:
                raise ValueError(f"block {b}: head {h0} starts at page {first}, not a multiple of {HALF_PAGES}")
            ttnn.experimental.paged_fill_cache(view, src, pts(first // HALF_PAGES, 2 * n), batch_idx=0)
            if not whole:
                ttnn.deallocate(part)

    def _read_rec(self, dn, b):
        """Block b's recurrent state as a new [1, Nv, Dk, Dv] fp32 TILE tensor."""
        import ttnn

        pages = self.layout.head_pages(torch.tensor([b]))[0]
        parts = []
        for view, h0, h1, off in self._pools(dn):
            if h1 <= h0:
                continue
            n = h1 - h0
            s0 = (int(pages[h0]) - off) // HALF_PAGES
            sl = ttnn.slice(
                view, (s0, 0, 0, 0), (s0 + 2 * n, 1, dn.Dk // 2, dn.Dv), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            parts.append(ttnn.reshape(sl, (1, n, dn.Dk, dn.Dv)))
        if len(parts) == 1:
            return parts[0]
        out = ttnn.concat(parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for p in parts:
            ttnn.deallocate(p)
        return out

    def _write_carry(self, dn, b, carry_rm, pts, table=None):
        """carry_rm [1, kc - 1, C] bf16 ROW_MAJOR -> block b's conv history: its row-major values, 32 per tile row,
        zero padded to whole tiles."""
        import ttnn

        n_rows = (dn.K - 1) * dn.qkv_dim_tp // 32
        pages = self.layout.conv_pages
        mc = ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.reshape(carry_rm, (n_rows, 32))
        z = ttnn.pad(x, [(0, pages * 32 - n_rows), (0, 0)], 0.0, memory_config=mc) if pages * 32 > n_rows else x
        w = ttnn.to_layout(ttnn.reshape(z, (1, 1, pages * 32, 32)), ttnn.TILE_LAYOUT, memory_config=mc)
        if z is not x:
            ttnn.deallocate(z)
        view, q0 = dn._spec_paged.conv_at(self.layout.conv_page(b), self.layout.conv_pages_total)
        ttnn.experimental.paged_fill_cache(view, w, pts(q0, pages) if table is None else table, batch_idx=0)
        ttnn.deallocate(w)

    def _read_carry(self, dn, b):
        """Block b's conv history as a new [1, kc - 1, C] bf16 TILE tensor (conv_carry's shape)."""
        import ttnn

        kc, C = dn.K, dn.qkv_dim_tp
        n_rows = (kc - 1) * C // 32
        pages = self.layout.conv_pages
        mc = ttnn.DRAM_MEMORY_CONFIG
        view, q0 = dn._spec_paged.conv_at(self.layout.conv_page(b), self.layout.conv_pages_total)
        t = ttnn.slice(view, (q0, 0, 0, 0), (q0 + pages, 1, 32, 32), memory_config=mc)
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        ttnn.deallocate(t)
        rows = ttnn.slice(ttnn.reshape(rm, (pages * 32, 32)), (0, 0), (n_rows, 32), memory_config=mc)
        ttnn.deallocate(rm)
        out = ttnn.to_layout(ttnn.reshape(rows, (1, kc - 1, C)), ttnn.TILE_LAYOUT, memory_config=mc)
        ttnn.deallocate(rows)
        return out


class _PageTables:
    """The page tables of one block operation, each built once and shared by every GDN layer (a block id names the
    same pages in each layer slot's views); freed on exit."""

    def __init__(self, state):
        self._state, self._tabs = state, {}

    def __call__(self, first, n):
        key = (int(first), int(n))
        if key not in self._tabs:
            self._tabs[key] = self._state._page_table(*key)
        return self._tabs[key]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        import ttnn

        for t in self._tabs.values():
            ttnn.deallocate(t)
        self._tabs.clear()


@dataclass
class _LayerViews:
    rec1: object
    rec2: object
    conv: object  # bf16 flat view of V
    conv_k: object  # bf16 flat view of K
    group: int
    two_pool: bool
    idx: object = None

    def conv_at(self, page, total):
        """(bf16 view, page in it) of a two-pool conv page (SpecBlockLayout.conv_page)."""
        return (self.conv, page) if page < total else (self.conv_k, page - total)


def _dtype_name(dtype):
    s = str(dtype).lower()
    for n in ("bfloat8_b", "bfloat16", "float32"):
        if n in s:
            return n
    raise ValueError(f"unsupported KV cache dtype {dtype}")
