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

Only whole view pages inside the block's bytes are used (SpecBlockLayout). The heads fill K's block first, the rest go
to V's block, and the conv history follows them in V (BF8 block 1024 at TP=8 on 8 banks: 4 heads in K, 2 in V). When
K's block holds every head, V's block holds only the conv history, at its start (BF8 block 2048 at TP=8, BF8 block 4096
at TP=4).

A verify (gdn_spec_step with ring_page_idx) reads the start state of row u from its window column
mamba_state.accepted_counts[u] - 1 and writes the state after token t into window column t. The conv window of the
decode stays in the layer's private ping-pong pair (TPGatedDeltaNet._verify_win_pair): no verify reads a block's conv
history, and the window holds the conv rows of every column of a row's latest verify. They are written into a block only
where vLLM can read the block as something other than the same row's next verify (``materialize``): the source of a
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
    heads_in_k: int = field(init=False)
    pool1_pages: int = field(init=False)
    conv_pages_total: int = field(init=False)

    def __post_init__(self):
        S, nb = int(self.block_bytes), int(self.n_banks)
        rows_per_bank = (self.num_blocks * S) // FP32_PAGE
        # pool 1 = the fp32 view of K in HALF_PAGES units: the op splits the two-pool page space at its page count
        self.pool1_pages = (rows_per_bank * nb) // HALF_PAGES * HALF_PAGES
        self.conv_pages_total = (self.num_blocks * S) // BF16_PAGE * nb
        k_heads = min(self._pool_heads(b) for b in range(min(self.num_blocks, _PERIOD)))
        self.heads_in_k = min(self.nv, k_heads)
        if self.nv - self.heads_in_k > k_heads:
            raise ValueError(
                f"K and V blocks hold {2 * k_heads} whole heads of 16 fp32 pages per block, the layer has {self.nv}"
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
    rank among the GDN layers of its layer slot. Checked against the layer names of every step's mamba_state groups
    (qwen36_vllm._spec_mamba_groups)."""
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
    the int64 [B] start column (mamba_state.accepted_counts - 1)."""

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
        # head pages of each moved row's start blocks [G, T, Nv] and windows [G, T, T, Nv], computed once in set_moves
        self._move_pages = {}
        self._moves_dropped = False  # forget_row dropped a row's unresolved moves: its device window rows are unknown
        self.pending_blocks = []  # (row, column, group, block) conv writes that name their block (chained rows)

    def set_step(self, blocks, start_col, live_rows, boundary=()):
        """Stage one verify. ``blocks``: [G] int64 [B, T] in ENGINE rows (0 = the null block on rows no request holds);
        ``start_col``: int64 [B] (mamba_state.accepted_counts - 1); ``live_rows``: the engine rows the verify computes
        (the others are held). ``boundary``: (row, column) pairs whose conv rows take_pending returns after the verify.
        Returns the groups whose ring_page_idx rows changed (the device copy to make). Runs every decode step, so the
        head pages of a group and the window map are rebuilt only when its blocks, the live rows or a copy changed
        them."""
        if len(blocks) != self.G:
            raise ValueError(f"{len(blocks)} window groups, the model binds {self.G}")
        # chained rows of the previous verify whose count never reached the host: the device window rows are unknown
        stale = bool(self.moves) or self._moves_dropped
        self._moves_dropped = False
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
        self.moves, self._move_pages = {}, {}
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
            mp = self._move_pages.pop(u, None)
            if mv is not None:
                self._adopt_window(u, c + 1, mv[1], mp)
                changed.update(range(self.G))
                if start_pos is not None and self._at_boundary(int(start_pos[u])):
                    self.pending.append((u, 0))
            for g in range(self.G):
                if mv is None:
                    self.init[g][u] = self._pages[g][u, c]
                else:
                    self.init[g][u] = self.layout.head_pages(mv[0][g][c]) if mp is None else mp[0][g][c]
        return sorted(changed)

    def _at_boundary(self, pos):
        return self.cache_mode == "align" and (pos + 1) % self.mamba_block_size == 0

    def set_moves(self, starts, windows, rows):
        """Chained state moves of the staged verify (after set_step): ``starts`` [G] int64 [B, T] start blocks and
        ``windows`` [G] int64 [B, T, T] state windows, entry [u, c - 1] for accepted count c of the pending verify;
        ``rows``: the chained engine rows (the others keep the staged window). The boundary conv writes set_step
        derived for these rows from provisional positions are dropped (chain_boundaries / set_start set them)."""
        rows = [int(u) for u in rows]
        self.moves = {u: ([s[u].clone() for s in starts], [w[u].clone() for w in windows]) for u in rows}
        self._move_pages = {}
        if rows:  # two vectorized head_pages calls instead of one per row, group and count
            sel = torch.tensor(rows)
            sp = self.layout.head_pages(torch.stack([torch.as_tensor(s)[sel] for s in starts]))  # [G, U, T, Nv]
            wp = self.layout.head_pages(torch.stack([torch.as_tensor(w)[sel] for w in windows]))  # [G, U, T, T, Nv]
            self._move_pages = {u: (sp[:, i], wp[:, i]) for i, u in enumerate(rows)}
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
        pages, self._move_pages = self._move_pages, {}
        for u, (_, win) in moves.items():
            self._adopt_window(u, int(counts[u]), win, pages.get(u))

    def _adopt_window(self, u, c, win, pages=None):
        """Row u takes the window of accepted count c (``pages``: its precomputed (start, window) head pages)."""
        self.forget_row(u)
        mapped = []
        for g in range(self.G):
            w = win[g][c - 1]
            self.step_blocks[g][u] = w
            self._pages[g][u] = self.layout.head_pages(w) if pages is None else pages[1][g][c - 1]
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
            mp = self._move_pages.get(u)
            for g in range(self.G):
                pages[g][u] = self.layout.head_pages(st[g]) if mp is None else mp[0][g]
        return pages

    def windows_cat(self):
        """float32 [B, T * G * T * Nv]: the chain body's window-page table (chained state moves), gathered at
        accepted count c = mi' + 1: column (c - 1) * G * T * Nv + g * T * Nv + t * Nv + h = the first page of head h of
        the block that takes the state after query token t, the staged window for every count on a row without chained
        state moves."""
        nv = self.layout.nv
        out = torch.stack(self._pages, dim=1).unsqueeze(1).expand(self.B, self.T, self.G, self.T, nv).clone()
        for u, (_, win) in self.moves.items():
            mp = self._move_pages.get(u)
            if mp is None:
                out[u] = torch.stack([self.layout.head_pages(win[g]) for g in range(self.G)], dim=1)
            else:
                out[u] = mp[1].permute(1, 0, 2, 3)  # [G, count, T, Nv] -> [count, G, T, Nv]
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
        """Unmap engine row u's window blocks; its unresolved chained moves go too (the row's request ended or prefills
        again). Resolving them later would map the old request's window to the row again, and a block copy from one of
        those ids, by then another request's state block, would take its conv rows from the old verify window."""
        self._map_dirty = True
        self._move_pages.pop(int(u), None)
        if self.moves.pop(int(u), None) is not None:
            self._moves_dropped = True
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
            raise ValueError("QWEN36_MAMBA_SPEC prefill without mamba_state")
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
            raise ValueError(f"mamba_state.checkpoint_positions needs checkpoint_blocks in each of [{self.G}] groups")
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
    """The positions where a prefill must leave vLLM's prefill checkpoint states inside its chunk. ``save(n)`` writes
    the state after n tokens into vLLM's block for n: from the bound scratch when a segment or replay ends at n, or
    from in-replay snapshot ``snap`` of the replay that passed n (QWEN36_PREFILL_INTRACE_SNAP)."""

    def __init__(self, entries, write, write_snap=None):
        super().__init__(n for n, _ in entries)
        self._blocks = dict(entries)
        self._write = write
        self._write_snap = write_snap

    def save(self, n, snap=None):
        if snap is None:
            self._write(self._blocks[int(n)])
        else:
            self._write_snap(self._blocks[int(n)], snap)


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
            this = SpecBlockLayout(S, n_banks, int(k.shape[0]), dn.Nv, conv_carry_pages(dn.K, dn.qkv_dim_tp))
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
                k=k,
                v=v,
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
        # the chain body's window-page table for chained state moves (SpecBlockBook.windows_cat), selected by count
        from models.demos.blackhole.qwen36.tt.tp_common import spec_chain_kernel_enabled

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
        # QWEN36_SPEC_CHAIN_KERNEL: windows_dev's words as uint32 RM (the kernel's window table)
        self.windows_rm_dev = None
        if spec_chain_kernel_enabled():
            self.windows_rm_dev = ttnn.from_torch(
                as_u32_words(self.book.windows_cat().to(torch.int64)),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
            )
        self._up = {}  # host copy of the last upload of pages_dev / windows_dev
        self._io = None  # GdnBlockIO (bind_block_io): the block writes and the restore as one program each
        self._io_zero = None  # GdnBlockIO for the zeroing
        self._tables_moved = False  # the last upload held a row's chained state moves
        self.moves_counts = None  # (wait=False) -> engine-row accepted counts the staged moves follow, None until read
        self.chain_copies = []  # capture_chain_copies: (row, group, src, dst, window column or None, counts)

    cache_mode = property(lambda self: self.book.cache_mode)
    mamba_block_size = property(lambda self: self.book.mamba_block_size)

    # ---------------------------------------------------------------- steps
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
        """pages_dev and windows_dev from the book, each only when it differs from its last upload."""
        import ttnn

        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        tables = [
            ("pages", self.book.pages_cat(), self.pages_dev),
            ("windows", self.book.windows_cat(), self.windows_dev),
        ]
        for name, host, dev in tables:
            if name in self._up and torch.equal(self._up[name], host):
                continue
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep), dev
            )
            rm = {"pages": self.pages_rm_dev, "windows": self.windows_rm_dev}[name]
            if rm is not None:
                words = as_u32_words(host.to(torch.int64))
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(words, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep), rm
                )
            self._up[name] = host
        self._tables_moved = bool(self.book.moves)

    def set_moves(self, starts, windows, rows):
        """A chained step's state moves (SpecBlockBook.set_moves), engine rows; the chain body's tables follow."""
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
        """A chained step's boundary copies (the copies of the mamba_state groups' chain), before set_step maps this
        step's windows: ``copies`` [G] int64 [B, T, 2] in engine rows, entry [u, c - 1] = (src, dst) when the pending
        verify accepts c tokens of row u; ``rows`` = the chained rows. Records each distinct pair with the counts that
        select it and the pending verify's window column whose conv rows src stands for (src's block holds only its
        recurrent state)."""
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
        if self._window_io() is not None:
            with self._timed("chained block copies (kernel)", len(todo)):
                self._io.chain(todo, result)
            return
        with self._timed("chained block copies", len(todo)), _PageTables(self) as pts:
            if result is None:
                for u, g, src, dst, t, _ in todo:
                    if self.layout.phase(src) == self.layout.phase(dst):
                        self._copy_block(src, dst)
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
        owned = []

        def counts_row():
            row = ttnn.slice(result, [RES_COUNTS, 0], [RES_COUNTS + 1, W], memory_config=mc)
            cnt = ttnn.typecast(ttnn.to_layout(row, ttnn.TILE_LAYOUT), f32)  # [1, W]: integers, exact in fp32
            ttnn.deallocate(row)
            owned.append(cnt)
            return cnt

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
        ct = getattr(self, "_copy_traces", None)
        ri = None if ct is None or not ct.tids else ct.result_index(result)
        for u, g, src, dst, t, cs in todo:
            if ri is not None and ct.chained_select(ri, src, u, t, g, cs, dst):  # the selection traced too
                continue
            # per entry: a trace replay may overwrite buffers allocated after its capture, so nothing computed before
            # one entry's replays is read after them
            cu = ttnn.slice(counts_row(), [0, u], [1, u + 1])  # [1, 1]
            m = None
            for c in cs:
                e = ttnn.typecast(ttnn.to_layout(ttnn.eq(cu, float(c)), ttnn.TILE_LAYOUT), f32)
                m = e if m is None else ttnn.add(m, e)
            owned.extend((cu, m))
            ps, pd, p0 = heads(src), heads(dst), heads(0)
            tabs = {}  # pool -> its page table: a block names the same pages in every slot's views
            for k, (_, h0, h1, off) in enumerate(self._pools(next(iter(self._slot_layers.values())))):
                if h1 > h0:
                    n = 2 * (h1 - h0)
                    tabs[k] = selected(m, (int(pd[h0]) - off) // HALF_PAGES, (int(p0[h0]) - off) // HALF_PAGES, n)
            cd, c0 = lay.conv_page(dst), lay.conv_page(0)
            if (cd < lay.conv_pages_total) != (c0 < lay.conv_pages_total):
                raise RuntimeError(f"block {dst} and the null block keep their conv history in different views")
            total = lay.conv_pages_total
            q = lambda page: page if page < total else page - total
            ctab = selected(m, q(cd), q(c0), lay.conv_pages)
            if ct is not None and ct.chained(src, u, t, g, {**tabs, "conv": ctab}):  # the copy traces
                continue
            for dn in self._slot_layers.values():  # every slot's views: the bytes of src belong to group g in each slot
                for k, (view, h0, h1, off) in enumerate(self._pools(dn)):
                    if h1 <= h0:
                        continue
                    n = 2 * (h1 - h0)
                    s0 = (int(ps[h0]) - off) // HALF_PAGES
                    part = ttnn.slice(view, (s0, 0, 0, 0), (s0 + n, 1, dn.Dk // 2, dn.Dv), memory_config=mc)
                    x = ttnn.reshape(part, (1, 1, n * dn.Dk // 2, dn.Dv))
                    ttnn.experimental.paged_fill_cache(view, x, tabs[k], batch_idx=0)
                    ttnn.deallocate(part)
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
        if due and self._window_io() is not None:
            with self._timed("boundary conv writes (kernel)", len(due)):
                self._io.window(due)
            return
        with self._timed("boundary conv writes", len(due)), _PageTables(self) as pts:
            for u, t, g, b in due:
                self.materialize(u, t, g, b, pts)

    def _window_io(self):
        """self._io when the verify windows fit GdnBlockIO.window (checked once: the window pair exists only after
        prepare_spec_verify), else None."""
        if getattr(self, "_io", None) is None:
            return None
        ok = getattr(self, "_window_ok", None)
        if ok is None:
            from loguru import logger

            why = self._io.window_reason()
            ok = self._window_ok = why is None
            logger.info(f"[mamba_spec] block I/O kernel: boundary conv writes {'on' if ok else f'off ({why})'}")
        return self._io if ok else None

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
    def set_prefill_checkpoints(self, per_group, positions):
        self.book.set_prefill_checkpoints(per_group, positions)

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

        if getattr(self, "_io", None) is not None:
            with self._timed("prefill block restores (kernel)", len(self.layers)):
                self._io.restore([self.book.prefill_block(g, u) for g in range(self.G)])
            return
        ct = getattr(self, "_copy_traces", None)
        if ct is not None:  # the copy traces (_CopyTraces.restore)
            with self._timed("prefill block restores (trace)", len(self.layers)):
                if ct.restore([self.book.prefill_block(g, u) for g in range(self.G)]):
                    return
        with self._timed("prefill block restores", len(self.layers)):
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

    def write_checkpoint_snapshot(self, blocks, k):
        """Write in-replay snapshot k of the last prefill replay (QWEN36_PREFILL_INTRACE_SNAP) into ``blocks``: a
        prefill checkpoint that lies inside that replay. The snapshot holds the same state a segment ending there
        would leave in the scratch (TPGatedDeltaNet.copy_snapshot_to takes the same slices for the prefix cache)."""
        import ttnn

        if getattr(self, "_io", None) is not None:
            with self._timed("checkpoint snapshot writes (kernel)", len(self.layers)):
                self._io.write_snapshot(blocks, k)
            for b in blocks:
                self.book.forget_block(int(b))
            return
        if self._replay_write(blocks, "checkpoint snapshot writes", k):
            return
        dram = ttnn.DRAM_MEMORY_CONFIG
        with self._timed("checkpoint snapshot writes", len(self.layers)), _PageTables(self) as pts:
            for _, dn, g in self.layers:
                b = int(blocks[g])
                self.book.forget_block(b)
                rec = ttnn.slice(dn._snap_rec, [k, 0, 0, 0], [k + 1, dn.Nv, dn.Dk, dn.Dv], memory_config=dram)
                self._write_rec(dn, b, rec, pts)
                ttnn.deallocate(rec)
                n = dn.K - 1
                rows = ttnn.slice(dn._snap_conv_rm, [0, n * k, 0], [1, n * k + n, dn.qkv_dim_tp], memory_config=dram)
                self._write_carry(dn, b, rows, pts)
                ttnn.deallocate(rows)

    def _write_scratch(self, blocks, kind):
        import ttnn

        if getattr(self, "_io", None) is not None:
            with self._timed(f"{kind} (kernel)", len(self.layers)):
                self._io.write(blocks)
            for b in blocks:
                self.book.forget_block(int(b))
            return
        if self._replay_write(blocks, kind):
            return
        with self._timed(kind, len(self.layers)), _PageTables(self) as pts:
            for _, dn, g in self.layers:
                b = int(blocks[g])
                self.book.forget_block(b)
                self._write_rec(dn, b, dn.rec_state, pts)
                rm = ttnn.to_layout(dn.conv_carry, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                self._write_carry(dn, b, rm, pts)
                ttnn.deallocate(rm)

    def _replay_write(self, blocks, kind, snap=None):
        """Write the bound prefill scratch (``snap`` None) or in-replay snapshot ``snap`` into ``blocks`` ([G] ids) by
        its captured write trace (_WriteTrace); False when there is none (before capture_write_trace, or its capture
        failed) or it does not apply (the caller writes eagerly)."""
        traces = (
            getattr(self, "_snap_write_traces", {}) if snap is not None else {None: getattr(self, "_write_trace", None)}
        )
        wt = traces.get(snap)
        if wt is None:
            return False
        with self._timed(f"{kind} (trace)", len(self.layers)):
            if not wt.replay(blocks):
                return False
        for b in blocks:
            self.book.forget_block(b)
        return True

    def _prepare_block_traces(self):
        """Allocate the write and copy traces' control tensors (_WriteTrace, _CopyTraces) in the eager warmup, before
        any trace exists. A buffer allocated after a trace's capture can sit where that trace's replay writes its
        intermediates (seen: verify replays overwrote copy-trace controls allocated at capture time). The traces are
        captured last (capture_write_trace, capture_copy_traces)."""
        from loguru import logger

        from models.demos.blackhole.qwen36.tt.tp_common import GDN_SNAP_SLOTS

        try:
            if getattr(self, "_io", None) is not None:
                pass  # the block I/O kernel writes: no write traces
            elif getattr(self, "_wt_pending", None) is None:
                self._wt_pending = _WriteTrace(self)
            # one per in-replay snapshot slot: a prefill checkpoint inside a replay is written from its slot
            if (
                getattr(self, "_io", None) is None
                and getattr(self, "_swt_pending", None) is None
                and all(getattr(dn, "_snap_rec", None) is not None for _, dn, _ in self.layers)
            ):
                self._swt_pending = {k: _WriteTrace(self, snap=k) for k in range(GDN_SNAP_SLOTS)}
            if getattr(self, "_ct_pending", None) is None:
                self._ct_pending = _CopyTraces(self)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[mamba_spec] block trace buffers not allocated ({e!r}): eager block operations")

    def capture_write_trace(self):
        """Capture the prefill block write (_WriteTrace, its tables allocated in warmup) on the bound scratch, so the
        write replays one trace instead of about 550 eager ops. Call after every other trace is captured
        (MTPSpecEngine.capture_decode). A failed capture leaves the eager write in charge."""
        from loguru import logger

        wt = getattr(self, "_wt_pending", None)
        if getattr(self, "_write_trace", None) is not None or wt is None:
            return
        model = self.model
        prev = model._bind_gdn_prefill_scratch()
        try:
            wt.capture()
            self._write_trace = wt
            logger.info(f"[mamba_spec] prefill block write trace captured ({len(self.layers)} GDN layers)")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[mamba_spec] prefill block write trace capture failed ({e!r}): eager writes")
        finally:
            model._unbind_gdn_prefill_scratch(prev)
        done = {}
        try:
            for k, swt in (getattr(self, "_swt_pending", None) or {}).items():
                swt.capture()
                done[k] = swt
            if done:
                logger.info(f"[mamba_spec] checkpoint snapshot write traces captured ({len(done)} snapshot slots)")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[mamba_spec] checkpoint snapshot write trace capture failed ({e!r}): eager writes")
        self._snap_write_traces = done

    def capture_copy_traces(self):
        """Capture the block copy, page copy, conv write and restore traces (_CopyTraces, its controls allocated in
        warmup), so each of these replays instead of running about 60 to 500 eager ops. Call after every other trace
        is captured (MTPSpecEngine.capture_decode). A failed capture leaves the eager ops in charge."""
        from loguru import logger

        ct = getattr(self, "_ct_pending", None)
        if getattr(self, "_copy_traces", None) is not None or ct is None:
            return
        try:
            ct.capture()
            prev = self.model._bind_gdn_prefill_scratch()
            try:
                ct.capture_restore()
            finally:
                self.model._unbind_gdn_prefill_scratch(prev)
            self._copy_traces = ct
            logger.info(f"[mamba_spec] block copy traces captured ({len(ct.tids)} traces)")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[mamba_spec] block copy trace capture failed ({e!r}): eager copies")

    def bind_block_io(self):
        """Route the prefill block write, the checkpoint writes, the restore and the zeroing through GdnBlockIO (one
        program each) when the bound scratch and the views fit its contract; else they keep the op chains and traces."""
        from loguru import logger

        from models.demos.blackhole.qwen36.tt.gdn_block_io import GdnBlockIO

        try:
            io = GdnBlockIO(self, self.model._paged_kv_caches)
            why, zero_why = io.reason(), io.zero_reason()
        except Exception as e:  # noqa: BLE001
            logger.error(f"[mamba_spec] block I/O kernel not bound ({e!r}): op chains and traces")
            return
        self._io = io if why is None else None
        self._io_zero = io if zero_why is None else None
        logger.info(
            f"[mamba_spec] block I/O kernel: moves {'on' if why is None else f'off ({why})'}, "
            f"zeroing {'on' if zero_why is None else f'off ({zero_why})'}"
        )

    def warmup(self):
        """Run every eager program of the block paths once on vLLM's null block 0 (prefill write and restore on the
        bound B=1 scratch, the conv write from the window, the block I/O kernel's boundary conv writes and chained
        copies, the whole-block copy and zeroing), so none compiles after the decode traces are captured."""
        import ttnn
        from models.demos.blackhole.qwen36.tt.gdn_block_state import zero_kv_blocks

        model = self.model
        prev = model._bind_gdn_prefill_scratch()
        try:
            self.bind_block_io()
            self.write_prefill(WARMUP_REQ)
            self.restore(WARMUP_REQ)
            if self._io is not None and all(getattr(dn, "_snap_rec", None) is not None for _, dn, _ in self.layers):
                self.write_checkpoint_snapshot([0] * self.G, 0)
        finally:
            model._unbind_gdn_prefill_scratch(prev)
        with _PageTables(self) as pts:
            for g in range(self.G):
                self.materialize(0, 0, g, 0, pts)
        io = self._window_io()  # the verify windows exist: prepare_verify_trace ran before this warmup
        if io is not None:  # after_verify and run_chain_copies run these programs instead of materialize
            io.warmup(getattr(self, "chain_results", None) or ())
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
        self._prepare_block_traces()

    # ---------------------------------------------------------------- block hooks
    def copy_blocks(self, copies):
        self.resolve_moves()
        ops = self.book.copy_plan(copies)
        n_conv = sum(1 for op in ops if op[0] == "conv")
        with self._timed(f"block copies ({n_conv} with conv writes)", len(ops) - n_conv), _PageTables(self) as pts:
            for op in ops:
                if op[0] == "copy":
                    self._copy_block(op[1], op[2])
                elif op[0] == "relayout":
                    self._relayout(op[1], op[2], pts)
                else:
                    self.materialize(*op[1:], pts)

    def _copy_block(self, src, dst):
        """Whole-block copy src -> dst in every layer slot's K and V (copy_kv_blocks; replayed once the copy traces are
        captured)."""
        from models.demos.blackhole.qwen36.tt.gdn_block_state import copy_kv_blocks

        ct = getattr(self, "_copy_traces", None)
        if ct is None or not ct.block(src, dst):
            copy_kv_blocks(self.model, [(src, dst)])

    def _relayout(self, src, dst, pts):
        """Copy block src's state into block dst page by page with the layout of each (the blocks hold their whole
        fp32 pages at different offsets): in every layer slot, the head pages of each pool and the conv history."""
        import ttnn

        ct = getattr(self, "_copy_traces", None)
        if ct is not None and ct.copy(src, dst):  # the copy traces
            return

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

        if getattr(self, "_io_zero", None) is not None:
            self._io_zero.zero(ids)
            return
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

        ct = getattr(self, "_copy_traces", None)
        if ct is not None and ct.conv(u, t, g, dst, table):  # the copy traces
            return
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

    def _write_rec(self, dn, b, rec, pts, tables=None):
        """rec [1, Nv, Dk, Dv] fp32 TILE -> block b's head pages (paged_fill_cache over [Dk / 2, Dv] view blocks: a
        head's 16 pages are its two half-head view blocks, in the tile order of [Dk, Dv]). ``tables`` (pool -> device
        page table): write through those instead of block b's pages."""
        import ttnn

        pages = None if tables is not None else self.layout.head_pages(torch.tensor([b]))[0]
        for k, (view, h0, h1, off) in enumerate(self._pools(dn)):
            if h1 <= h0:
                continue
            n = h1 - h0
            # every head in K (heads_in_k == Nv): a whole-tensor slice returns rec's own buffer, which must survive
            whole = h0 == 0 and h1 == int(rec.shape[1])
            part = (
                rec
                if whole
                else ttnn.slice(rec, (0, h0, 0, 0), (1, h1, dn.Dk, dn.Dv), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            )
            src = ttnn.reshape(part, (1, 1, n * dn.Dk, dn.Dv))
            if tables is None:
                first = int(pages[h0]) - off
                if first % HALF_PAGES:
                    raise ValueError(f"block {b}: head {h0} starts at page {first}, not a multiple of {HALF_PAGES}")
                tab = pts(first // HALF_PAGES, 2 * n)
            else:
                tab = tables[k]
            ttnn.experimental.paged_fill_cache(view, src, tab, batch_idx=0)
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


class _WriteTrace:
    """The captured write of the bound prefill scratch (``snap`` None, SpecGdnBlockState.write_prefill) or of in-replay
    snapshot slot ``snap`` (write_checkpoint_snapshot) into one block per GDN group, through persistent page tables:
    per group one int32 table per recurrent pool and one for the conv history, restaged per write. Valid while the
    source keeps the addresses the capture baked and the target blocks keep their conv history in the captured
    views."""

    def __init__(self, state, snap=None):
        import ttnn

        self.state, self.tid, self.snap = state, None, snap
        rep = ttnn.ReplicateTensorToMesh(state.mesh)
        self.tables = [self._host(g, 0) for g in range(state.G)]
        self.dev = [
            {
                k: ttnn.from_torch(
                    t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh, mesh_mapper=rep
                )
                for k, t in tabs.items()
            }
            for tabs in self.tables
        ]
        self.views = [self._conv_view(0)] * state.G
        self.fp = None

    def _host(self, g, b):
        """{pool index: int32 [1, n] view-block pages, "conv": int32 [1, conv_pages] conv pages} of block b."""
        st, lay = self.state, self.state.layout
        dn = next(d for _, d, gg in st.layers if gg == g)
        pages = lay.head_pages(torch.tensor([b]))[0]
        out = {}
        for k, (_, h0, h1, off) in enumerate(st._pools(dn)):
            if h1 > h0:
                first = (int(pages[h0]) - off) // HALF_PAGES
                out[k] = (first + torch.arange(2 * (h1 - h0), dtype=torch.int32)).reshape(1, -1)
        page = lay.conv_page(b)
        q0 = page if page < lay.conv_pages_total else page - lay.conv_pages_total
        out["conv"] = (q0 + torch.arange(lay.conv_pages, dtype=torch.int32)).reshape(1, -1)
        return out

    def _conv_view(self, b):
        return self.state.layout.conv_page(b) < self.state.layout.conv_pages_total

    def fingerprint(self):
        if self.snap is not None:
            return [
                (dn._snap_rec.buffer_address(), dn._snap_conv_rm.buffer_address()) for _, dn, _ in self.state.layers
            ]
        return [(dn.rec_state.buffer_address(), dn.conv_carry.buffer_address()) for _, dn, _ in self.state.layers]

    def body(self):
        """The write of every GDN layer through the device tables (the capture and the warm run)."""
        import ttnn

        st, mc, k = self.state, ttnn.DRAM_MEMORY_CONFIG, self.snap
        for _, dn, g in st.layers:
            if k is None:
                st._write_rec(dn, None, dn.rec_state, None, tables=self.dev[g])
                rm = ttnn.to_layout(dn.conv_carry, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
            else:
                rec = ttnn.slice(dn._snap_rec, [k, 0, 0, 0], [k + 1, dn.Nv, dn.Dk, dn.Dv], memory_config=mc)
                st._write_rec(dn, None, rec, None, tables=self.dev[g])
                ttnn.deallocate(rec)
                n = dn.K - 1
                rm = ttnn.slice(dn._snap_conv_rm, [0, n * k, 0], [1, n * k + n, dn.qkv_dim_tp], memory_config=mc)
            st._write_carry(dn, 0, rm, None, self.dev[g]["conv"])
            ttnn.deallocate(rm)

    def capture(self):
        import ttnn

        mesh = self.state.mesh
        self.body()  # compiles every program and allocates its outputs before the capture
        ttnn.synchronize_device(mesh)
        self.tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            self.body()
        finally:
            ttnn.end_trace_capture(mesh, self.tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        self.fp = self.fingerprint()

    def replay(self, blocks):
        """Write the bound scratch into ``blocks`` ([G] block ids); False (nothing done) when the trace does not apply."""
        import ttnn

        if self.tid is None or any(self._conv_view(b) != v for b, v in zip(blocks, self.views)):
            return False
        if self.fingerprint() != self.fp:
            return False
        rep = ttnn.ReplicateTensorToMesh(self.state.mesh)
        for g, b in enumerate(blocks):
            for k, t in self._host(g, int(b)).items():
                if not torch.equal(t, self.tables[g][k]):
                    h = ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)
                    ttnn.copy_host_to_device_tensor(h, self.dev[g][k])
                    self.tables[g][k] = t
        ttnn.execute_trace(self.state.mesh, self.tid, cq_id=0, blocking=False)
        return True


class _CopyTraces:
    """The block operations that read at a position the host picks, as replays of traces captured once: ``block`` = a
    whole-block copy of every layer slot's K and V (copy_kv_blocks), ``copy`` = the page-by-page state copy
    (SpecGdnBlockState._relayout: head pages of each pool and the conv history), ``conv`` = the conv rows after token t
    of engine row u's verify window into a block of group g (materialize; one trace per group, window parity and t).
    Reads use device-indexed slices (ttnn.slice with start tensors), writes use page tables; the host restages the
    small control tensors that changed and replays. Valid while the window buffers keep the addresses the capture
    baked and the blocks keep their conv history in the captured view (else the caller runs the eager ops)."""

    def __init__(self, state):
        import ttnn

        self.state = st = state
        lay, mesh = st.layout, st.mesh
        self._rep = ttnn.ReplicateTensorToMesh(mesh)
        self.host, self.dev = {}, {}
        dn0 = next(iter(st._slot_layers.values()))
        self.pools = [(k, h0, h1, off) for k, (_, h0, h1, off) in enumerate(st._pools(dn0)) if h1 > h0]
        self.conv_v = lay.conv_page(0) < lay.conv_pages_total  # the conv view the traces read and write
        cp = lay.conv_pages
        alias = lambda t, shape, dtype: ttnn.experimental.alias_view(t, shape, dtype, check_slices=False)
        # slice sources: views of the slot's K / V (a view of a view is refused) whose dim 0 is a multiple of the rows a
        # call takes (the device-indexed slice's output is dim 0 / parts); pool 0 lies in K, pool 1 in V
        self.sviews, self.parts = {}, {}
        dns = {id(dn): dn for _, dn, _ in st.layers}
        dns.update({id(dn): dn for dn in st._slot_layers.values()})
        for k, h0, h1, _ in self.pools:
            n = 2 * (h1 - h0)
            V = lay.pool1_pages // HALF_PAGES
            self.parts[k] = V // n
            self.sviews[k] = {
                i: alias((dn._spec_paged.k, dn._spec_paged.v)[k], [V // n * n, 1, dn.Dk // 2, dn.Dv], ttnn.float32)
                for i, dn in dns.items()
            }
            self._ctl(f"src{k}", torch.zeros(4, dtype=torch.int32))
            self._ctl(f"end{k}", torch.tensor([n, 1, dn0.Dk // 2, dn0.Dv], dtype=torch.int32))
            self._ctl(f"dst{k}", torch.zeros(1, n, dtype=torch.int32))
        P = lay.conv_pages_total
        self.cparts = P // cp
        cbase = lambda dn: dn._spec_paged.v if self.conv_v else dn._spec_paged.k  # conv lies in V, else in K
        self.sconv = {i: alias(cbase(dn), [P // cp * cp, 1, 32, 32], ttnn.bfloat16) for i, dn in dns.items()}
        self._ctl("csrc", torch.zeros(4, dtype=torch.int32))
        self._ctl("cend", torch.tensor([cp, 1, 32, 32], dtype=torch.int32))
        self._ctl("cdst", torch.zeros(1, cp, dtype=torch.int32))
        # whole-block copy: every K and V buffer of every layer slot
        self.kv = [t for kv in st.model._paged_kv_caches for t in kv]
        nb = int(self.kv[0].shape[0])
        self._ctl("bsrc", torch.zeros(4, dtype=torch.int32))
        self._ctl("bend", torch.tensor([1] + [int(x) for x in list(self.kv[0].shape)[1:]], dtype=torch.int32))
        self._ctl("bdst", torch.zeros(1, 1, dtype=torch.int32))
        self.bparts = nb
        # conv rows from the window: row u of the [B, kc - 1 + T, C] pair (device-indexed), rows t + 1 .. t + kc - 1
        dnw = st.layers[0][1]
        Bw, L, C = (int(x) for x in list(dnw._verify_win_pair[0].shape))
        Lp = -(-L // 32) * 32  # the indexed slice takes whole tile rows: each window row is one tile row high
        self.win_shape, self.kc = (Bw, L, C), dnw.K
        self.wviews = {
            (id(dn), par): alias(dn._verify_win_pair[par], [Bw, Lp, C], dn._verify_win_pair[par].dtype)
            for _, dn, _ in st.layers
            for par in (0, 1)
        }
        self._ctl("u", torch.zeros(3, dtype=torch.int32))
        self._ctl("uend", torch.tensor([1, Lp, C], dtype=torch.int32))
        self._ctl("mdst", torch.zeros(1, cp, dtype=torch.int32))
        # chained copies: the destination pages selected on the device from a fused step's counts (one trace per result
        # buffer), written straight into this object's copy and conv tables
        self.results = list(getattr(st, "chain_results", None) or [])
        if self.results:
            self.W = int(self.results[0].shape[1])
            if any(int(r.shape[1]) != self.W for r in self.results):
                self.results = []
        if self.results:
            self._ctl("onehot", torch.zeros(1, self.W), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            self._ctl("smask", torch.zeros(1, st.T), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            self._ctl(
                "iota1",
                torch.arange(1, st.T + 1, dtype=torch.float32).reshape(1, -1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
            )
            for k, h0, h1, _ in self.pools:
                for name in (f"base{k}", f"step{k}"):
                    self._ctl(name, torch.zeros(1, 2 * (h1 - h0)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            for name in ("basec", "stepc"):
                self._ctl(name, torch.zeros(1, cp), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        # restore: block pages of each group into the bound prefill scratch
        for g in range(st.G):
            for k, _, _, _ in self.pools:
                self._ctl(f"r{g}src{k}", torch.zeros(4, dtype=torch.int32))
            self._ctl(f"r{g}csrc", torch.zeros(4, dtype=torch.int32))
        self.tids = {}
        self.fp = None
        self.restore_fp = None

    def _ctl(self, name, t, dtype=None, layout=None):
        import ttnn

        dtype = ttnn.int32 if dtype is None else dtype
        layout = ttnn.ROW_MAJOR_LAYOUT if layout is None else layout
        self.host[name] = (t.clone(), dtype, layout)
        self.dev[name] = ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=self.state.mesh,
            mesh_mapper=self._rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _stage(self, name, t):
        import ttnn

        prev, dtype, layout = self.host[name]
        if prev is not None and torch.equal(prev, t):
            return
        h = ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=self._rep)
        ttnn.copy_host_to_device_tensor(h, self.dev[name])
        self.host[name] = (t.clone(), dtype, layout)

    def _set_dev(self, name, src):
        """Write a device page table computed on the device into control ``name`` (its host mirror is then unknown)."""
        import ttnn

        ttnn.copy(src, self.dev[name])
        self.host[name] = (None,) + self.host[name][1:]

    def _conv_view(self, dn):
        return dn._spec_paged.conv if self.conv_v else dn._spec_paged.conv_k

    def fingerprint(self):
        return [tuple(w.buffer_address() for w in dn._verify_win_pair) for _, dn, _ in self.state.layers]

    # ------------------------------------------------------------------ bodies
    def _block_body(self):
        import ttnn

        for t in self.kv:
            x = ttnn.slice(t, self.dev["bsrc"], self.dev["bend"], slice_dim=0, num_devices=self.bparts)
            ttnn.experimental.paged_fill_cache(t, x, self.dev["bdst"], batch_idx=0)
            ttnn.deallocate(x)

    def _copy_body(self):
        import ttnn

        st, cp = self.state, self.state.layout.conv_pages
        for dn in st._slot_layers.values():
            for k, h0, h1, _ in self.pools:
                n = 2 * (h1 - h0)
                x = ttnn.slice(
                    self.sviews[k][id(dn)],
                    self.dev[f"src{k}"],
                    self.dev[f"end{k}"],
                    slice_dim=0,
                    num_devices=self.parts[k],
                )
                y = ttnn.reshape(x, (1, 1, n * dn.Dk // 2, dn.Dv))
                ttnn.experimental.paged_fill_cache(st._pools(dn)[k][0], y, self.dev[f"dst{k}"], batch_idx=0)
                ttnn.deallocate(x)
            x = ttnn.slice(self.sconv[id(dn)], self.dev["csrc"], self.dev["cend"], slice_dim=0, num_devices=self.cparts)
            y = ttnn.reshape(x, (1, 1, cp * 32, 32))
            ttnn.experimental.paged_fill_cache(self._conv_view(dn), y, self.dev["cdst"], batch_idx=0)
            ttnn.deallocate(x)

    def _conv_body(self, g, par, t):
        """materialize's ops for every layer of group g, window half ``par``, token column ``t``; row u by index."""
        import ttnn

        st, mc = self.state, ttnn.DRAM_MEMORY_CONFIG
        _, L, C = self.win_shape
        for _, dn, gg in st.layers:
            if gg != g:
                continue
            row = ttnn.slice(
                self.wviews[(id(dn), par)], self.dev["u"], self.dev["uend"], slice_dim=0, num_devices=self.win_shape[0]
            )
            rm = ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
            ttnn.deallocate(row)
            carry = ttnn.slice(rm, (0, t + 1, 0), (1, t + self.kc, C), memory_config=mc)
            ttnn.deallocate(rm)
            st._write_carry(dn, 0, carry, None, self.dev["mdst"])  # block 0: the captured conv view
            ttnn.deallocate(carry)

    def capture(self):
        import ttnn

        mesh = self.state.mesh
        bodies = {"block": self._block_body, "copy": self._copy_body}
        for i, r in enumerate(self.results):
            bodies[("select", i)] = lambda r=r: self._select_body(r)
        for g in range(self.state.G):
            for par in (0, 1):
                for t in range(self.state.T):
                    bodies[("conv", g, par, t)] = lambda g=g, par=par, t=t: self._conv_body(g, par, t)
        for key, body in bodies.items():
            body()  # compiles every program and allocates its outputs before the capture (block 0 onto itself)
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                body()
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            self.tids[key] = tid
        ttnn.synchronize_device(mesh)
        for name in [f"dst{k}" for k, _, _, _ in self.pools] + ["cdst", "mdst"]:  # the select bodies wrote them
            self.host[name] = (None,) + self.host[name][1:]
        self.fp = self.fingerprint()

    def _select_body(self, r):
        """Destination page tables of one chained copy from result buffer ``r`` (row RES_COUNTS = accepted counts by
        engine row): m = 1 when the count of the row in "onehot" is one of "smask", then each table = base + m * step,
        written into the copy and conv tables. Integers below 2^24: exact in fp32."""
        import ttnn
        from models.demos.blackhole.qwen36.tt.spec_chain import RES_COUNTS

        f32, mc, TL = ttnn.float32, ttnn.DRAM_MEMORY_CONFIG, ttnn.TILE_LAYOUT
        row = ttnn.slice(r, [RES_COUNTS, 0], [RES_COUNTS + 1, self.W], memory_config=mc)
        cnt = ttnn.typecast(ttnn.to_layout(row, TL), f32)
        ttnn.deallocate(row)
        cu = ttnn.sum(ttnn.multiply(cnt, self.dev["onehot"]), dim=1, keepdim=True)  # [1, 1]: the row's count
        ttnn.deallocate(cnt)
        hit = ttnn.typecast(ttnn.to_layout(ttnn.eq(self.dev["iota1"], cu), TL), f32)  # [1, T]: 1 at count - 1
        m = ttnn.sum(ttnn.multiply(hit, self.dev["smask"]), dim=1, keepdim=True)  # [1, 1]
        targets = [(f"dst{k}", f"base{k}", f"step{k}") for k, _, _, _ in self.pools]
        targets += [("cdst", "basec", "stepc"), ("mdst", "basec", "stepc")]
        for name, base, step in targets:
            sel = ttnn.add(ttnn.multiply(self.dev[step], m), self.dev[base])
            out = ttnn.to_layout(ttnn.typecast(sel, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy(out, self.dev[name])
            ttnn.deallocate(sel)
            ttnn.deallocate(out)
        for x in (cu, hit, m):
            ttnn.deallocate(x)

    def _restore_body(self):
        """SpecGdnBlockState.restore's ops for every GDN layer, each group's block by index, into the bound scratch."""
        import ttnn

        st, mc = self.state, ttnn.DRAM_MEMORY_CONFIG
        cp = st.layout.conv_pages
        for _, dn, g in st.layers:
            parts = []
            for k, h0, h1, _ in self.pools:
                x = ttnn.slice(
                    self.sviews[k][id(dn)],
                    self.dev[f"r{g}src{k}"],
                    self.dev[f"end{k}"],
                    slice_dim=0,
                    num_devices=self.parts[k],
                )
                parts.append(ttnn.reshape(x, (1, h1 - h0, dn.Dk, dn.Dv)))
            rec = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=1, memory_config=mc)
            ttnn.copy(rec, dn.rec_state)
            for x in parts:
                ttnn.deallocate(x)
            if rec is not parts[0]:
                ttnn.deallocate(rec)
            t = ttnn.slice(
                self.sconv[id(dn)], self.dev[f"r{g}csrc"], self.dev["cend"], slice_dim=0, num_devices=self.cparts
            )
            rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
            ttnn.deallocate(t)
            n_rows = (dn.K - 1) * dn.qkv_dim_tp // 32
            rows = ttnn.slice(ttnn.reshape(rm, (cp * 32, 32)), (0, 0), (n_rows, 32), memory_config=mc)
            ttnn.deallocate(rm)
            carry = ttnn.to_layout(ttnn.reshape(rows, (1, dn.K - 1, dn.qkv_dim_tp)), ttnn.TILE_LAYOUT, memory_config=mc)
            ttnn.deallocate(rows)
            ttnn.copy(carry, dn.conv_carry)
            ttnn.deallocate(carry)

    def scratch_fingerprint(self):
        return [(dn.rec_state.buffer_address(), dn.conv_carry.buffer_address()) for _, dn, _ in self.state.layers]

    def capture_restore(self):
        """The restore trace, on the bound prefill scratch (the caller binds it)."""
        import ttnn

        mesh = self.state.mesh
        self._restore_body()
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            self._restore_body()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        self.tids["restore"] = tid
        self.restore_fp = self.scratch_fingerprint()

    def restore(self, blocks):
        """Load ``blocks`` ([G] ids) into the bound prefill scratch (SpecGdnBlockState.restore); False when the trace
        does not apply."""
        lay = self.state.layout
        if "restore" not in self.tids or self.scratch_fingerprint() != self.restore_fp:
            return False
        qs = [self._conv_q(b) for b in blocks]
        if any(q is None for q in qs):
            return False
        for g, b in enumerate(blocks):
            ps = lay.head_pages(torch.tensor([int(b)]))[0]
            for k, h0, _, off in self.pools:
                self._stage(
                    f"r{g}src{k}", torch.tensor([(int(ps[h0]) - off) // HALF_PAGES, 0, 0, 0], dtype=torch.int32)
                )
            self._stage(f"r{g}csrc", torch.tensor([qs[g], 0, 0, 0], dtype=torch.int32))
        self._replay("restore")
        return True

    def _replay(self, key):
        import ttnn

        ttnn.execute_trace(self.state.mesh, self.tids[key], cq_id=0, blocking=False)

    # ------------------------------------------------------------------ calls
    def _conv_q(self, b):
        """The first page of block b's conv history in the captured view, or None when it lies in the other view."""
        lay = self.state.layout
        page = lay.conv_page(b)
        if (page < lay.conv_pages_total) != self.conv_v:
            return None
        return page if page < lay.conv_pages_total else page - lay.conv_pages_total

    def block(self, src, dst):
        """Whole-block copy src -> dst in every K and V buffer (copy_kv_blocks)."""
        self._stage("bsrc", torch.tensor([int(src), 0, 0, 0], dtype=torch.int32))
        self._stage("bdst", torch.tensor([[int(dst)]], dtype=torch.int32))
        self._replay("block")
        return True

    def copy(self, src, dst, tables=None):
        """Page-by-page state copy src -> dst (_relayout). ``tables`` ({pool: device page table, "conv": ...}): write
        through those instead of dst's pages (the chained copies' device-selected destination)."""
        lay = self.state.layout
        qs = self._conv_q(src)
        qd = None if tables is not None else self._conv_q(dst)
        if qs is None or (tables is None and qd is None):
            return False
        ps = lay.head_pages(torch.tensor([int(src)]))[0]
        pd = None if tables is not None else lay.head_pages(torch.tensor([int(dst)]))[0]
        for k, h0, h1, off in self.pools:
            n = 2 * (h1 - h0)
            self._stage(f"src{k}", torch.tensor([(int(ps[h0]) - off) // HALF_PAGES, 0, 0, 0], dtype=torch.int32))
            if tables is None:
                first = (int(pd[h0]) - off) // HALF_PAGES
                self._stage(f"dst{k}", (first + torch.arange(n, dtype=torch.int32)).reshape(1, n))
            else:
                self._set_dev(f"dst{k}", tables[k])
        self._stage("csrc", torch.tensor([qs, 0, 0, 0], dtype=torch.int32))
        if tables is None:
            self._stage("cdst", (qd + torch.arange(lay.conv_pages, dtype=torch.int32)).reshape(1, -1))
        else:
            self._set_dev("cdst", tables["conv"])
        self._replay("copy")
        return True

    def result_index(self, result):
        """Index of a fused step's result buffer among the captured select traces, or None."""
        addr = result.buffer_address()
        return next(
            (i for i, r in enumerate(self.results) if r.buffer_address() == addr and ("select", i) in self.tids), None
        )

    def chained_select(self, ri, src, u, t, g, counts, dst):
        """A chained copy whose destination the device selects (dst where row u's count of result buffer ``ri`` is in
        ``counts``, else the null block 0): the select, copy and conv replays, every control staged before them."""
        lay = self.state.layout
        qs, qd, q0 = self._conv_q(src), self._conv_q(dst), self._conv_q(0)
        par = {dn._spec_win_par for _, dn, _ in self.state.layers}
        if None in (qs, qd, q0) or len(par) != 1 or (t is not None and self.fingerprint() != self.fp):
            return False
        onehot = torch.zeros(1, self.W)
        onehot[0, int(u)] = 1.0
        smask = torch.zeros(1, self.state.T)
        for c in counts:
            smask[0, int(c) - 1] = 1.0
        self._stage("onehot", onehot)
        self._stage("smask", smask)
        ps, pd, p0 = (lay.head_pages(torch.tensor([int(b)]))[0] for b in (src, dst, 0))
        for k, h0, h1, off in self.pools:
            n = 2 * (h1 - h0)
            d0, n0 = (int(pd[h0]) - off) // HALF_PAGES, (int(p0[h0]) - off) // HALF_PAGES
            self._stage(f"base{k}", (n0 + torch.arange(n, dtype=torch.float32)).reshape(1, n))
            self._stage(f"step{k}", torch.full((1, n), float(d0 - n0)))
            self._stage(f"src{k}", torch.tensor([(int(ps[h0]) - off) // HALF_PAGES, 0, 0, 0], dtype=torch.int32))
        cp = lay.conv_pages
        self._stage("basec", (q0 + torch.arange(cp, dtype=torch.float32)).reshape(1, cp))
        self._stage("stepc", torch.full((1, cp), float(qd - q0)))
        self._stage("csrc", torch.tensor([qs, 0, 0, 0], dtype=torch.int32))
        if t is not None:
            self._stage("u", torch.tensor([int(u), 0, 0], dtype=torch.int32))
        self._replay(("select", ri))
        for name in [f"dst{k}" for k, _, _, _ in self.pools] + ["cdst", "mdst"]:
            self.host[name] = (None,) + self.host[name][1:]
        self._replay("copy")
        if t is not None:
            self._replay(("conv", int(g), par.pop(), int(t)))
        return True

    def chained(self, src, u, t, g, tables):
        """A chained copy with its destination selected on the device: src's state (and, when ``t`` is not None, the
        conv rows of window column t of row u) through ``tables`` ({pool: page table, "conv": page table}). Every
        control is written before either replay, so no replay can overwrite a table that is still to be read."""
        qs = self._conv_q(src)
        par = {dn._spec_win_par for _, dn, _ in self.state.layers}
        if qs is None or len(par) != 1 or (t is not None and self.fingerprint() != self.fp):
            return False
        ps = self.state.layout.head_pages(torch.tensor([int(src)]))[0]
        for k, h0, _, off in self.pools:
            self._stage(f"src{k}", torch.tensor([(int(ps[h0]) - off) // HALF_PAGES, 0, 0, 0], dtype=torch.int32))
            self._set_dev(f"dst{k}", tables[k])
        self._stage("csrc", torch.tensor([qs, 0, 0, 0], dtype=torch.int32))
        self._set_dev("cdst", tables["conv"])
        if t is not None:
            self._stage("u", torch.tensor([int(u), 0, 0], dtype=torch.int32))
            self._set_dev("mdst", tables["conv"])
        self._replay("copy")
        if t is not None:
            self._replay(("conv", int(g), par.pop(), int(t)))
        return True

    def conv(self, u, t, g, dst, table=None):
        """The conv rows after token t of engine row u's latest verify into block dst of group g (materialize);
        ``table``: a device page table to write through instead of dst's conv pages."""
        if self.fingerprint() != self.fp:
            return False
        lay = self.state.layout
        qd = None if table is not None else self._conv_q(dst)
        if table is None and qd is None:
            return False
        par = {dn._spec_win_par for _, dn, _ in self.state.layers}
        if len(par) != 1:
            return False
        self._stage("u", torch.tensor([int(u), 0, 0], dtype=torch.int32))
        if table is None:
            self._stage("mdst", (qd + torch.arange(lay.conv_pages, dtype=torch.int32)).reshape(1, -1))
        else:
            self._set_dev("mdst", table)
        self._replay(("conv", int(g), par.pop(), int(t)))
        return True


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
    k: object = None  # the slot's K and V buffers the views alias (_CopyTraces aliases them again)
    v: object = None

    def conv_at(self, page, total):
        """(bf16 view, page in it) of a two-pool conv page (SpecBlockLayout.conv_page)."""
        return (self.conv, page) if page < total else (self.conv_k, page - total)


def _dtype_name(dtype):
    s = str(dtype).lower()
    for n in ("bfloat8_b", "bfloat16", "float32"):
        if n in s:
            return n
    raise ValueError(f"unsupported KV cache dtype {dtype}")
