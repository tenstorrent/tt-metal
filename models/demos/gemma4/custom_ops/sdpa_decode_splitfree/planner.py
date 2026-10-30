# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The pure-python planner of ``sdpa_decode_splitfree``: the fixed KV chunking, the static (row, unit) -> core
mapping, the scratch layout, the combine order and the L1 budget of both programs. No ttnn import, so the host
suite checks it on the Wormhole (8x8) and Blackhole (13x10) grids.

Vocabulary (tokens are KV positions of one row's paged cache):

* ``chunk`` (``C``, a multiple of 32): the K/V slab one flash step takes through L1 -- the largest rung of
  ``CHUNK_LADDER`` at or below ``GEMMA4_SDPA_SPLITFREE_CHUNK`` (default 256) whose circular buffers fit the
  per-core L1 of the arch (``grid.sdpa_decode_cb_budget_bytes``). It depends on the head dim, the cache format
  and the arch only, never on the batch or the grid, so every decode graph of one build chunks alike.
* ``unit`` (``U = nsub * C``): the work unit. On a layer without a window it is derived from the grid
  (``derive_unit``): the smallest multiple of the chunk whose unit count at ``max_context`` is at most the grid's
  cores, at most ``DEFAULT_UNIT`` (4096) -- so a lone row at the full context puts one unit on (nearly) every core:
  2048 tokens on the 13x10 Blackhole grid (128 units of a 256K context on 130 cores), 2304 on a harvested 12x10,
  4096 on the Wormhole 8x8 (64 units); ``GEMMA4_SDPA_SPLITFREE_UNIT`` pins it. The grid is a build constant of a
  chip, so every decode graph of one build units alike; a row's bits follow the unit (and the chunk), never the
  core mapping. One chunk on a windowed (sliding) layer, whose live range is at most ``window`` tokens. A unit's chunks
  are walked in ascending order on ONE core with the stock online-softmax recurrence; its partial (running max
  ``m``, denominator ``l``, unnormalised numerator ``O``) is written to the scratch. Unit ``u`` of row ``b`` covers
  tokens ``[u*U, (u+1)*U)``; the chunk holding ``cur_pos`` is masked, chunks past it (and before the window
  start) are skipped.
* pair space: ``p = (u mod M) * (B*G) + b*G + g`` for the padded row ``b``, KV group ``g`` and unit ``u``, with
  ``M = min(n_cores, max_units)``; the core of a pair is ``p mod n_cores``. Unit-major, so a lone row's units
  spread round-robin over the grid and the rows of one unit sit on consecutive cores; a core walks its pairs in
  ascending ``p`` and, per pair, ascending ``u`` (``u = u' + k*M``) -- the one order its three kernels share.
  The mapping never reads ``cur_pos``: liveness (which units of a row exist) is decided on device, per row.
* slot: ``(b*G + g) * n_slots + (u mod n_slots)`` of the scratch, ``n_slots`` the most units one row can have
  live at once (all of them without a window; ``ceil((window-1)/U) + 1`` with one). A slot is ``2 + DHt`` fp32
  tiles: ``m``, ``l``, then ``O``'s ``DHt`` tiles.
* combine: pair ``(b, g)`` on core ``(b*G + g) mod n_cores`` merges the live units in ascending ``u`` with the
  stock correction step (``m' = max(m, m_u)``, ``l' = l*e^(m-m') + l_u*e^(m_u-m')``, ``O'`` likewise), then
  normalises and writes the group's head rows in bf16.
"""

import dataclasses
import os

from models.demos.gemma4.custom_ops.grid import grid_cores, sdpa_decode_cb_budget_bytes, sdpa_tile_bytes

UNIT_ENV = "GEMMA4_SDPA_SPLITFREE_UNIT"
CHUNK_ENV = "GEMMA4_SDPA_SPLITFREE_CHUNK"
READS_ENV = "GEMMA4_SDPA_SPLITFREE_READS"  # K/V tile reads a core keeps in flight (default 16)
QK_FP32_ENV = "GEMMA4_SDPA_SPLITFREE_QK_FP32"  # "0": the scores CB in bf16 (the stock's im_df); default fp32
DEFAULT_UNIT = 4096
DEFAULT_CHUNK = 256
DEFAULT_READS = 16
CHUNK_LADDER = (512, 256, 128, 64, 32)
TILE = 32
F32_TILE_BYTES = 4096
BF16_TILE_BYTES = 2048
MAX_ROWS = 32  # the Q tile rows: padded batch rows (B <= 32) and local Q heads (PNHt == 1)
NONE_POS = 0xFFFFFFFF  # cur_pos -1 as the kernels read it: the row is padding


def _env_int(env, name, default, multiple=1):
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return default
    value = int(raw)
    if value < multiple or value % multiple:
        raise ValueError(f"{name} must be a positive multiple of {multiple}, got {value}")
    return value


def ceil_div(a, b):
    return -(-int(a) // int(b))


def derive_unit(chunk, max_context, n_cores, cap=DEFAULT_UNIT):
    """The unit of a layer without a window on a grid of ``n_cores``: the smallest multiple of ``chunk`` whose
    unit count at ``max_context`` is at most ``n_cores`` (a lone row at the full context lands one unit per core,
    the shortest per-core walk the one-level combine allows), never above ``cap`` (``DEFAULT_UNIT``: a wider
    grid or a shorter context does not shrink the unit below what 4096 tokens per core cost) nor below the
    chunk. Depends on the grid, the chunk and the context only -- all build constants of a chip -- never on the
    batch or on ``cur_pos``."""
    chunk, max_context, n_cores = int(chunk), int(max_context), max(1, int(n_cores))
    unit = ceil_div(ceil_div(max_context, n_cores), chunk) * chunk
    return max(chunk, min(int(cap), unit))


def partial_cb_bytes(dht, sk, kv_tile_bytes, window, pos_stick, pt_stick, qk_tile_bytes=BF16_TILE_BYTES):
    """Phase-1 (partials) circular buffers of one core: q; K and V double-buffered over one chunk in the cache
    format; the causal mask (and the window mask); the zero tile; the scores (one chunk, ``qk_tile_bytes`` per
    tile: fp32 by default, see ``QK_FP32_ENV``) and the reduce scaler in the scores' format (the row reduces
    move the scores through both source registers: a bf16 scaler would truncate fp32 scores to bf16 in the
    max and unpack them as bf16 in the sum); the fp32 PV output and accumulator; five fp32 statistics (cur /
    prev max, cur / prev sum, exp max diff); the fp32 unit output (O, m, l); the cur_pos stick for the compute
    and the page-table row."""
    return (
        dht * BF16_TILE_BYTES
        + 4 * sk * dht * kv_tile_bytes
        + (2 if window else 1) * sk * BF16_TILE_BYTES
        + BF16_TILE_BYTES
        + (sk + 1) * qk_tile_bytes
        + 2 * dht * F32_TILE_BYTES
        + 5 * F32_TILE_BYTES
        + (dht + 2) * F32_TILE_BYTES
        + pos_stick
        + pt_stick
    )


def combine_cb_bytes(dht, pos_stick):
    """Phase-2 (combine) circular buffers of one core: the unit partial double-buffered (m, l, O), seven fp32
    statistics, two fp32 accumulators, the bf16 output, the cur_pos stick."""
    return (
        2 * (2 + dht) * F32_TILE_BYTES
        + 7 * F32_TILE_BYTES
        + 2 * dht * F32_TILE_BYTES
        + dht * BF16_TILE_BYTES
        + pos_stick
    )


@dataclasses.dataclass(frozen=True)
class SplitfreePlan:
    grid: tuple
    n_cores: int
    batch: int  # padded rows B
    groups: int  # local KV groups G (the cache's heads this call attends)
    heads_per_group: int
    head_dim: int
    max_context: int  # the page table's reach: max_blocks * block_size
    window: int  # 0 = attend everything
    chunk: int
    unit: int
    max_units: int
    n_slots: int
    modulus: int
    kv_tile_bytes: int
    qk_tile_bytes: int  # the scores CB: 4096 (fp32) or 2048 (bf16)
    reads_in_flight: int
    partial_cb_bytes: int
    combine_cb_bytes: int
    budget_bytes: int

    # ── geometry ──────────────────────────────────────────────────────────────────────────────────────────────
    @property
    def dht(self):
        return self.head_dim // TILE

    @property
    def sk(self):
        return self.chunk // TILE

    @property
    def nsub(self):
        return self.unit // self.chunk

    @property
    def pair_space(self):
        return self.modulus * self.batch * self.groups

    @property
    def tiles_per_slot(self):
        return 2 + self.dht

    @property
    def scratch_tiles(self):
        return self.batch * self.groups * self.n_slots * self.tiles_per_slot

    @property
    def scratch_bytes(self):
        return self.scratch_tiles * F32_TILE_BYTES

    @property
    def scratch_tiles_bound(self):
        """The slots any context needs on this grid: ``derive_unit`` keeps ``max_units <= n_cores`` on a layer
        without a window (a narrower page table than the model's maximum shrinks the unit, never the bound), so a
        scratch sized for ``n_cores`` slots serves every call; a windowed layer's slot count is fixed by the
        window."""
        slots = self.n_slots if self.window else max(self.n_slots, self.n_cores)
        return self.batch * self.groups * slots * self.tiles_per_slot

    @property
    def fits(self):
        return max(self.partial_cb_bytes, self.combine_cb_bytes) <= self.budget_bytes

    # ── the kernels' arithmetic, mirrored for the host tests ──────────────────────────────────────────────────
    def row_range(self, cur_pos):
        """``(first_chunk, tail_chunk, window_start)`` of a row at ``cur_pos`` (None for a padding row)."""
        if cur_pos is None or int(cur_pos) < 0 or int(cur_pos) == NONE_POS:
            return None
        cur_pos = int(cur_pos)
        tail = cur_pos // self.chunk
        window_start = 0
        if self.window:
            window_start = cur_pos + 1 - self.window if cur_pos + 1 > self.window else 0
        return window_start // self.chunk, tail, window_start

    def live_units(self, cur_pos):
        r = self.row_range(cur_pos)
        if r is None:
            return range(0)
        return range(r[0] // self.nsub, r[1] // self.nsub + 1)

    def unit_chunks(self, u, cur_pos):
        r = self.row_range(cur_pos)
        if r is None:
            return range(0)
        return range(max(u * self.nsub, r[0]), min((u + 1) * self.nsub - 1, r[1]) + 1)

    def core_of(self, b, g, u):
        p = (u % self.modulus) * (self.batch * self.groups) + b * self.groups + g
        return p % self.n_cores

    def pairs_of_core(self, core):
        """The ``(u', b, g)`` pairs of a core in walking order (``u'`` the unit class ``u mod M``)."""
        bg_n = self.batch * self.groups
        out = []
        for p in range(core, self.pair_space, self.n_cores):
            up, bg = divmod(p, bg_n)
            out.append((up, bg // self.groups, bg % self.groups))
        return out

    def units_of_core(self, core, cur_pos):
        """The ``(b, g, u)`` a core processes for the padded rows' ``cur_pos`` list, in the kernels' order."""
        out = []
        for up, b, g in self.pairs_of_core(core):
            lu = self.live_units(cur_pos[b])
            if not lu:
                continue
            u = up
            if u < lu.start:
                u += ceil_div(lu.start - u, self.modulus) * self.modulus
            while u < lu.stop:
                out.append((b, g, u))
                u += self.modulus
        return out

    def slot(self, b, g, u):
        return (b * self.groups + g) * self.n_slots + (u % self.n_slots)

    def combine_core(self, b, g):
        return (b * self.groups + g) % self.n_cores

    def combine_order(self, b, g, cur_pos):
        """The unit partials the combine of ``(b, g)`` merges, ascending: ``[(u, slot), ...]``."""
        return [(u, self.slot(b, g, u)) for u in self.live_units(cur_pos)]

    def describe(self):
        return (
            f"B {self.batch} x G {self.groups} (x {self.heads_per_group} heads), head_dim {self.head_dim}, context "
            f"{self.max_context}, window {self.window or 'none'}: chunk {self.chunk} x {self.nsub} = unit {self.unit}, "
            f"{self.max_units} units, {self.n_slots} slots, modulus {self.modulus}, {self.pair_space} pairs on "
            f"{self.n_cores} cores ({self.grid[0]}x{self.grid[1]}); scratch {self.scratch_tiles} tiles = "
            f"{self.scratch_bytes:,} B; CBs partial {self.partial_cb_bytes:,} / combine {self.combine_cb_bytes:,} of "
            f"{self.budget_bytes:,} B" + ("" if self.fits else " (over the L1 budget)")
        )


def plan_splitfree(
    grid,
    batch,
    groups,
    heads_per_group,
    head_dim,
    max_context,
    *,
    window=None,
    kv_dtype=None,
    arch=None,
    env=None,
    pos_stick=128,
    pt_stick=0,
    budget_bytes=None,
):
    """The plan of one call: ``grid`` the ``(x, y)`` compute grid, ``batch`` the padded rows, ``groups`` the local
    KV heads the call attends, ``heads_per_group`` the Q heads of each, ``max_context`` the page table's reach
    in tokens (the model's ``max_seq_len``: ``kv_cache_hybrid`` and the vLLM plugin size every page table to it,
    so it is a build constant like the grid and the unit it decides), ``window`` the sliding window (None / 0:
    attend everything), ``kv_dtype`` the cache format."""
    env = os.environ if env is None else env
    gx, gy = (int(v) for v in grid)
    n_cores = grid_cores((gx, gy))
    batch, groups, heads_per_group, head_dim, max_context = (
        int(v) for v in (batch, groups, heads_per_group, head_dim, max_context)
    )
    window = int(window or 0)
    if min(n_cores, batch, groups, heads_per_group, head_dim, max_context) < 1:
        raise ValueError("plan_splitfree needs positive grid, batch, groups, heads, head_dim and context")
    if head_dim % TILE:
        raise ValueError(f"head_dim must be a multiple of {TILE}, got {head_dim}")
    if batch > MAX_ROWS or groups * heads_per_group > MAX_ROWS:
        raise ValueError(
            f"at most {MAX_ROWS} padded rows and {MAX_ROWS} local Q heads (one Q tile), got {batch} rows, "
            f"{groups} x {heads_per_group} heads"
        )
    if batch * groups > n_cores:
        raise ValueError(f"the combine wants one core per (row, group): {batch} x {groups} > {n_cores} cores")
    kv_tile = sdpa_tile_bytes(kv_dtype)
    budget = sdpa_decode_cb_budget_bytes(arch) if budget_bytes is None else int(budget_bytes)
    chunk_cap = _env_int(env, CHUNK_ENV, DEFAULT_CHUNK, TILE)
    unit_cfg = _env_int(env, UNIT_ENV, 0, TILE)  # 0: derive the unit from the grid (derive_unit)
    reads = _env_int(env, READS_ENV, DEFAULT_READS)
    qk_tile = BF16_TILE_BYTES if str(env.get(QK_FP32_ENV, "") or "").strip() == "0" else F32_TILE_BYTES
    dht = head_dim // TILE
    chosen = None
    for c in CHUNK_LADDER:
        if c > chunk_cap:
            continue
        pcb = partial_cb_bytes(dht, c // TILE, kv_tile, window, pos_stick, pt_stick, qk_tile)
        chosen = (c, pcb)
        if pcb <= budget:
            break
    if chosen is None:
        raise ValueError(f"{CHUNK_ENV}={chunk_cap} is below the smallest chunk {CHUNK_LADDER[-1]}")
    chunk, pcb = chosen
    if window:
        unit = chunk
    elif unit_cfg:
        unit = max(chunk, (unit_cfg // chunk) * chunk)  # pinned by the env
    else:
        unit = derive_unit(chunk, max_context, n_cores)
    max_units = ceil_div(max_context, unit)
    n_slots = max_units if not window else min(max_units, ceil_div(window - 1, unit) + 1)
    modulus = min(n_cores, max_units)
    return SplitfreePlan(
        grid=(gx, gy),
        n_cores=n_cores,
        batch=batch,
        groups=groups,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        max_context=max_context,
        window=window,
        chunk=chunk,
        unit=unit,
        max_units=max_units,
        n_slots=n_slots,
        modulus=modulus,
        kv_tile_bytes=kv_tile,
        qk_tile_bytes=qk_tile,
        reads_in_flight=reads,
        partial_cb_bytes=pcb,
        combine_cb_bytes=combine_cb_bytes(dht, pos_stick),
        budget_bytes=budget,
    )


def scratch_tiles_for(grid, batch, layer_specs, max_context, *, kv_dtype=None, arch=None, env=None):
    """The flat scratch (in fp32 tiles) one model needs on ``grid`` for ``batch`` padded rows at ``max_context``:
    the largest slot pool over its layer types (``scratch_tiles_bound``: one slot per core on a layer without a
    window, so a call on a narrower page table -- a smaller unit, as many units -- still fits), ``layer_specs``
    an iterable of ``(groups, heads_per_group, head_dim, window)``. Planned with the arch's L1 budget and a
    page-table row of 32-token blocks (the widest row a call can bring), so the chunk choice is the call's or a
    smaller one, never a larger."""
    tiles = 0
    for groups, heads_per_group, head_dim, window in layer_specs:
        plan = plan_splitfree(
            grid,
            batch,
            groups,
            heads_per_group,
            head_dim,
            max_context,
            window=window,
            kv_dtype=kv_dtype,
            arch=arch,
            env=env,
            pt_stick=ceil_div(max_context, TILE) * 4,
        )
        tiles = max(tiles, plan.scratch_tiles_bound)
    return tiles
