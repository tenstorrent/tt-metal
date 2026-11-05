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

from dataclasses import dataclass, field

import torch

FP32_PAGE = 4096
BF16_PAGE = 2048
BF8_TILE = 1088
HEAD_PAGES = 16  # fp32 pages of one [Dk, Dv] = [128, 128] head
HALF_PAGES = 8  # fp32 pages of one [Dk / 2, Dv] slice: the unit of the rec views
WARMUP_REQ = -1  # the prefill request index of the engine's warmup prompts: they write into vLLM's null block 0
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

    def set_step(self, blocks, start_col, live_rows, boundary=()):
        """Stage one verify. ``blocks``: [G] int64 [B, T] in ENGINE rows (0 = the null block on rows no request holds);
        ``start_col``: int64 [B] (state_accepted_counts - 1); ``live_rows``: the engine rows the verify computes (the
        others are held). ``boundary``: (row, column) pairs whose conv rows take_pending returns after the verify.
        Returns the groups whose ring_page_idx rows changed (the device copy to make)."""
        if len(blocks) != self.G:
            raise ValueError(f"{len(blocks)} window groups, the model binds {self.G}")
        sw = StepWindows(blocks, start_col)
        changed = []
        for g in range(self.G):
            host = sw.ring_page_idx(self.layout, g)
            if not torch.equal(host, self.idx_host[g]):
                self.idx_host[g] = host
                changed.append(g)
            self.init[g] = sw.init_pages(self.layout, g)
        self.step_blocks = [b.clone() for b in blocks]
        self.step_start = torch.as_tensor(start_col, dtype=torch.int64).clone()
        self.pending = [(int(u), int(t)) for u, t in boundary]
        for u in live_rows:
            self.forget_row(u)
            mapped = []
            for g in range(self.G):
                for t, b in enumerate(blocks[g][u].tolist()):
                    if b > 0:
                        self.window_map[int(b)] = (int(u), t, g)
                        mapped.append(int(b))
            self.row_blocks[int(u)] = mapped
        return changed

    def ctrl_rows(self, mi, parity, words, hold=(), start_col=None):
        """int32 [G, words]: the ctrl page of each group (same parity and mi, the group's start pages). ``start_col``
        (int [B]): start pages at these columns of the staged windows instead of the staged start columns."""
        init = self.init
        if start_col is not None and self.step_blocks is not None:
            sw = StepWindows(self.step_blocks, torch.as_tensor(start_col, dtype=torch.int64).reshape(-1))
            init = [sw.init_pages(self.layout, g) for g in range(self.G)]
        return torch.cat([ctrl_page_paged(mi, init[g], parity, words, hold) for g in range(self.G)], dim=0)

    def set_start(self, cols):
        """Move staged rows' start columns: {engine row: column} (a chained row the host stages after all)."""
        for u, c in cols.items():
            self.step_start[u] = int(c)
            for g in range(self.G):
                self.init[g][u] = self.layout.head_pages(self.step_blocks[g][u, int(c)].reshape(1))[0]

    def step_start_blocks(self, u):
        """[G] block ids of engine row u's start column in the staged step, or None."""
        if self.step_blocks is None:
            return None
        c = int(self.step_start[u])
        out = [int(b[u, c]) for b in self.step_blocks]
        return out if all(x > 0 for x in out) else None

    def take_pending(self):
        """[(row, column, group, block)] conv writes the staged verify left due; clears them."""
        pending, self.pending = self.pending, []
        out = []
        for u, t in pending:
            for b in list(self.row_blocks.get(u, ())):
                ent = self.window_map.get(b)
                if ent is not None and ent[0] == u and ent[1] == t:
                    out.append((u, t, ent[2], b))
        return out

    def copy_plan(self, copies):
        """The device work of vLLM's (src, dst) block copies, in list order: ("copy", src, dst) = the whole block of every
        K and V buffer; ("conv", row, column, group, dst) = that window column's conv rows into dst, for a source that is
        a window column of a row's latest verify (its block holds a stale history). dst then names the same state."""
        ops = []
        for src, dst in copies:
            src, dst = int(src), int(dst)
            if src == dst:
                continue
            if src <= 0 or dst <= 0:
                raise ValueError(f"block copy {src} -> {dst} names the null block 0")
            ops.append(("copy", src, dst))
            ent = self.window_map.get(src)
            self.forget_block(dst)
            if ent is not None:
                ops.append(("conv", ent[0], ent[1], ent[2], dst))
                self.window_map[dst] = ent
                self.row_blocks.setdefault(ent[0], []).append(dst)
        return ops

    def forget_row(self, u):
        for b in self.row_blocks.pop(int(u), ()):
            if self.window_map.get(b, (None,))[0] == int(u):
                del self.window_map[b]

    def forget_block(self, b):
        ent = self.window_map.pop(int(b), None)
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

    def __init__(self, model, per_layer_specs, kv_caches, T, layout="two_pool", cache_mode="none", mamba_block_size=0):
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
        self.layout = lay
        self.book = SpecBlockBook(lay, self.G, self.bmax, self.T, cache_mode, mamba_block_size)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self._idx_dev = [
            ttnn.from_torch(h, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
            for h in self.book.idx_host
        ]
        for _, dn, g in self.layers:
            dn._spec_paged.idx = self._idx_dev[g]

    cache_mode = property(lambda self: self.book.cache_mode)
    mamba_block_size = property(lambda self: self.book.mamba_block_size)

    # ---------------------------------------------------------------- steps
    def group_windows(self, windows_per_layer):
        return group_windows(windows_per_layer, self.layer_groups, self.G)

    def set_step(self, blocks, start_col, live_rows, boundary=()):
        import ttnn

        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for g in self.book.set_step(blocks, start_col, live_rows, boundary):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    self.book.idx_host[g], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
                ),
                self._idx_dev[g],
            )

    def ctrl_rows(self, mi, parity, words, hold=(), start_col=None):
        return self.book.ctrl_rows(mi, parity, words, hold, start_col)

    def step_start_blocks(self, u):
        return self.book.step_start_blocks(u)

    def after_verify(self):
        """Once the staged verify is enqueued and the window parity flipped (verify_win_cur is the window it writes):
        enqueue the conv writes it left due behind it."""
        for u, t, g, b in self.book.take_pending():
            self.materialize(u, t, g, b)

    def forget_row(self, u):
        self.book.forget_row(u)

    # ---------------------------------------------------------------- prefill
    def set_prefill_blocks(self, state_block_ids_per_layer):
        ids = self.group_windows(state_block_ids_per_layer)
        self.book.set_prefill_blocks([t.reshape(t.shape[0], -1)[:, 0] for t in ids])

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
        import ttnn

        for _, dn, g in self.layers:
            b = self.book.prefill_block(g, u)
            self.book.forget_block(b)
            self._write_rec(dn, b, dn.rec_state)
            rm = ttnn.to_layout(dn.conv_carry, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self._write_carry(dn, b, rm)
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
        for g in range(self.G):
            self.materialize(0, 0, g, 0)
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
                blk = ttnn.slice(t, (0, 0, 0, 0), [1] + [int(x) for x in list(t.shape)[1:]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.experimental.paged_fill_cache(t, blk, pt, batch_idx=0)
                ttnn.deallocate(blk)
        ttnn.deallocate(pt)
        zero_kv_blocks(model, [0])

    # ---------------------------------------------------------------- block hooks
    def copy_blocks(self, copies):
        from models.demos.blackhole.qwen36.tt.gdn_block_state import copy_kv_blocks

        for op in self.book.copy_plan(copies):
            if op[0] == "copy":
                copy_kv_blocks(self.model, [(op[1], op[2])])
            else:
                self.materialize(*op[1:])

    def zero_blocks(self, block_ids):
        from models.demos.blackhole.qwen36.tt.gdn_block_state import zero_kv_blocks

        for b in block_ids:
            self.book.forget_block(int(b))
        zero_kv_blocks(self.model, [int(b) for b in block_ids])

    def materialize(self, u, t, g, dst):
        """Write the conv history after token t of engine row u's latest verify (rows t + 1 .. t + kc - 1 of the window
        the next verify reads, verify_win_cur) into block dst of every GDN layer of group g."""
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
            self._write_carry(dn, dst, carry)
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

    def _write_rec(self, dn, b, rec):
        """rec [1, Nv, Dk, Dv] fp32 TILE -> block b's head pages (paged_fill_cache over [Dk / 2, Dv] view blocks: a
        head's 16 pages are its two half-head view blocks, in the tile order of [Dk, Dv])."""
        import ttnn

        pages = self.layout.head_pages(torch.tensor([b]))[0]
        for view, h0, h1, off in self._pools(dn):
            if h1 <= h0:
                continue
            n = h1 - h0
            part = ttnn.slice(rec, (0, h0, 0, 0), (1, h1, dn.Dk, dn.Dv), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            src = ttnn.reshape(part, (1, 1, n * dn.Dk, dn.Dv))
            first = int(pages[h0]) - off
            if first % HALF_PAGES:
                raise ValueError(f"block {b}: head {h0} starts at page {first}, not a multiple of {HALF_PAGES}")
            pt = self._page_table(first // HALF_PAGES, 2 * n)
            ttnn.experimental.paged_fill_cache(view, src, pt, batch_idx=0)
            ttnn.deallocate(pt)
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

    def _write_carry(self, dn, b, carry_rm):
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
        pt = self._page_table(q0, pages)
        ttnn.experimental.paged_fill_cache(view, w, pt, batch_idx=0)
        ttnn.deallocate(pt)
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
