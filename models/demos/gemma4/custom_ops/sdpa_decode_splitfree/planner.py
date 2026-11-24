# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The pure-python planner of ``sdpa_decode_splitfree``: the fixed KV chunking, the per-row unit, the static
(row, unit) -> core mapping, the scratch layout, the combine order and the L1 budget of both programs. No device
access (``grid`` only names ttnn types inside functions), so the host suite checks it on the Wormhole (8x8) and
Blackhole (13x10) grids.

Vocabulary (tokens are KV positions of one row's paged cache):

* ``chunk`` (``C``, a multiple of 32): the K/V slab one flash step takes through L1 -- the largest rung of
  ``CHUNK_LADDER`` at or below ``GEMMA4_SDPA_SPLITFREE_CHUNK`` (default 256) whose circular buffers fit the
  per-core L1 of the arch (``grid.sdpa_decode_cb_budget_bytes``). It depends on the head dim, the cache format
  and the arch only, never on the batch or the grid, so every decode graph of one build chunks alike.
* ``unit`` (``nsub`` chunks): the work unit. On a layer without a window it follows the row's own position: a row
  of ``n_chunks = cur_pos // C + 1`` chunks splits into at most ``K`` units of
  ``nsub = max(NSUB_MIN, ceil(n_chunks / K))`` chunks (``nsub_of``), computed on device from ``cur_pos``;
  ``K = GEMMA4_SDPA_SPLITFREE_SPLIT`` (default 32) and ``NSUB_MIN = GEMMA4_SDPA_SPLITFREE_MIN_CHUNKS`` (default 1)
  are build constants. A lone 4K row runs 17 one-chunk units, a 256K row 32 units of 32 chunks: the critical
  path is one core walking ``nsub`` chunks, never a fixed 4096 tokens. The unit set is a function of
  ``(cur_pos, K, NSUB_MIN, C)`` only -- never of the batch, the grid or the core -- so a row's bits change with its
  position at the ``nsub`` transitions (``n_chunks = K*m + 1``) and nowhere else. ``GEMMA4_SDPA_SPLITFREE_UNIT=
  <tokens>`` (or ``SPLIT=0``, ``LEGACY_UNIT`` tokens) keeps the earlier fixed unit for A/B and the bit gate
  (``split == 0``). One chunk on a windowed (sliding) layer, whose live range is at most ``window`` tokens. A
  unit's chunks are walked in ascending order on ONE core with the stock online-softmax recurrence; its partial
  (running max ``m``, denominator ``l``, unnormalised numerator ``O``) is written to the scratch. Unit ``u`` of a
  row covers chunks ``[u*nsub, (u+1)*nsub)``; the chunk holding ``cur_pos`` is masked, chunks past it (and before
  the window start) are skipped.
* pair space: ``(u' = u mod M, bg = b*G + g)`` for the padded row ``b``, KV group ``g`` and unit ``u``, with
  ``M = min(n_cores, max_units)``; the core of a pair is ``(u' * stride + bg) mod n_cores`` with ``stride`` the
  smallest integer >= B*G coprime with ``n_cores``: a row's M unit classes land on M distinct cores whatever the
  padded batch (a lone row spreads over ``min(M, n_cores)`` cores inside a B=32 graph as inside a B=1 graph; the
  earlier ``p = u'*BG + bg, core = p mod n`` put a lone row of a B=32 graph on ``n / gcd(32, n)`` cores: 15 of
  120, 65 of 130, 2 of 64), and the B*G rows of one unit class fill a window of consecutive cores, the windows
  tiling the grid. A core walks its pairs in ascending ``bg`` (``u' = (core - bg) * stride^-1 mod n_cores`` when
  below M) and, per pair, ascending ``u`` (``u = u' + k*M``) -- the one order its three kernels share. The
  mapping never reads ``cur_pos``: liveness (which units of a row exist) is decided on device, per row.
* slot: ``(b*G + g) * n_slots + (u mod n_slots)`` of the scratch, ``n_slots`` the most units one row can have
  live at once (``min(K, chunks of the page table)`` with the position's unit, all of them with a fixed unit;
  ``ceil((window-1)/C) + 1`` with a window). A slot is ``2 + DHt`` fp32 tiles: ``m``, ``l``, then ``O``'s
  ``DHt`` tiles.
* combine: jobs ``(bg, blk)`` -- the row-group ``bg = b*G + g`` and the block ``blk < NCB = DHt / CT`` of ``CT``
  O tiles -- on core ``(bg*NCB + blk) mod n_cores``, ``CT`` the smallest divisor of DHt whose ``B*G*NCB`` jobs
  fit the grid (``choose_combine_tiles``: one O tile per job, 16 jobs per row, up to 8 rows on 13x10). A job
  replays its row's statistics sequence (identical ops on identical data: identical bits on every job), merges its
  CT O tiles over the live units in ascending ``u``, then normalises and writes the group's head rows of its tiles
  in bf16. The merge (``GEMMA4_SDPA_SPLITFREE_COMBINE``): ``stats`` (default) takes ``M = max_u m_u`` first (exact,
  order-free), then ``w_u = e^((m_u - M) scale)``, ``L = sum w_u l_u`` and ``O = sum w_u O_u`` accumulated in the
  fp32 dest (one column-broadcast multiply-accumulate per unit and tile); ``pairwise`` is the stock correction
  step (``m' = max(m, m_u)``, ``l' = l*e^(m-m') + l_u*e^(m_u-m')``, ``O'`` likewise), the earlier op's bits with
  its unit. Every op of either merge is on a statistics tile or tile-local, so neither ``CT`` nor the job -> core
  split changes a tile's arithmetic: a lone row's combine runs on 16 cores at once.
"""

import dataclasses
import math
import os

from models.demos.gemma4.custom_ops.grid import grid_cores, sdpa_decode_cb_budget_bytes, sdpa_tile_bytes

SPLIT_ENV = "GEMMA4_SDPA_SPLITFREE_SPLIT"  # K: the most units a row splits into (default 32; 0: the fixed unit)
MIN_CHUNKS_ENV = "GEMMA4_SDPA_SPLITFREE_MIN_CHUNKS"  # NSUB_MIN: the fewest chunks of a unit (default 1)
UNIT_ENV = "GEMMA4_SDPA_SPLITFREE_UNIT"  # <tokens>: the legacy fixed unit (split 0), for A/B and the bit gate
CHUNK_ENV = "GEMMA4_SDPA_SPLITFREE_CHUNK"
READS_ENV = "GEMMA4_SDPA_SPLITFREE_READS"  # K/V tile reads a core keeps in flight (default 16)
QK_FP32_ENV = "GEMMA4_SDPA_SPLITFREE_QK_FP32"  # "0": the scores CB in bf16 (the stock's im_df); default fp32
COMBINE_TILES_ENV = "GEMMA4_SDPA_SPLITFREE_COMBINE_TILES"  # test knob: CT, the O tiles of a combine job
COMBINE_ENV = "GEMMA4_SDPA_SPLITFREE_COMBINE"  # "stats" (default): M first, dest accumulation; "pairwise"
COMBINE_MODES = ("stats", "pairwise")
COMBINE_UNIT_BATCH = 8  # stats combine: the units whose O tiles one read batch brings (double-buffered)
DST_TILES = 4  # fp32 dest accumulation, half sync: four tiles per dest window
DEFAULT_SPLIT = 32
DEFAULT_MIN_CHUNKS = 1
LEGACY_UNIT = 4096  # the fixed unit of SPLIT=0 (the op before the position-derived unit)
DEFAULT_CHUNK = 256
DEFAULT_READS = 16
CHUNK_LADDER = (512, 256, 128, 64, 32)
TILE = 32
F32_TILE_BYTES = 4096
BF16_TILE_BYTES = 2048
MAX_ROWS = 32  # the Q tile rows: padded batch rows (B <= 32) and local Q heads (PNHt == 1)
NONE_POS = 0xFFFFFFFF  # cur_pos -1 as the kernels read it: the row is padding


def _env_int(env, name, default, multiple=1, allow_zero=False):
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return default
    value = int(raw)
    if allow_zero and value == 0:
        return 0
    if value < multiple or value % multiple:
        raise ValueError(f"{name} must be a positive multiple of {multiple}, got {value}")
    return value


def ceil_div(a, b):
    return -(-int(a) // int(b))


def unit_stride(bg_n, n_cores):
    """``(stride, stride^-1 mod n_cores)`` of the pair -> core mapping: the smallest integer >= ``bg_n``
    (rows x groups) coprime with ``n_cores``, so ``u' -> u' * stride mod n_cores`` is a bijection (a row's unit
    classes on distinct cores) and the ``bg_n`` pairs of one unit class fill consecutive cores."""
    n_cores, bg_n = int(n_cores), max(1, int(bg_n))
    if n_cores == 1:
        return 1, 0
    stride = bg_n
    while math.gcd(stride, n_cores) != 1:
        stride += 1
    return stride, pow(stride, -1, n_cores)


def nsub_of_chunks(n_chunks, split, nsub_min):
    """Chunks per unit of a row of ``n_chunks`` chunks under the position-derived unit: ``max(nsub_min,
    ceil(n_chunks / split))``, so the row splits into at most ``split`` units (the kernels: ``row_of``)."""
    return max(int(nsub_min), ceil_div(n_chunks, split))


def choose_combine_tiles(bg_n, dht, n_cores):
    """``CT``, the O tiles of one combine job: the smallest divisor of ``dht`` for which the ``bg_n * dht / CT``
    jobs (rows x groups x blocks) fit ``n_cores`` -- one job per core, the most cores per row. A scheduling choice
    only (it never changes a tile's arithmetic), so it may follow the batch: 1 for B <= 8, 2 for B = 16, 4 for
    B = 32 at head_dim 512 on 13x10. ``dht`` itself when even one block per row-group does not fit."""
    dht, bg_n, n_cores = int(dht), max(1, int(bg_n)), max(1, int(n_cores))
    for ct in range(1, dht + 1):
        if dht % ct == 0 and bg_n * (dht // ct) <= n_cores:
            return ct
    return dht


def combine_granularity(ct):
    """The O tiles one dest window of the combine takes (the combine program's ``DHT_GRANULARITY`` and the stats
    merge's dest block): the largest of 4 / 2 / 1 that divides ``ct``."""
    return next(g for g in (DST_TILES, 2, 1) if g <= ct and ct % g == 0)


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


def combine_cb_bytes(ct, pos_stick, stats_slots=None):
    """Phase-2 (combine) circular buffers of one core for jobs of ``ct`` O tiles. Pairwise (``stats_slots``
    None): the unit partial double-buffered (m, l, the job's O tiles), seven fp32 statistics, two fp32
    accumulators, the bf16 output, the cur_pos stick. Stats-first (``stats_slots`` the most live units of a row):
    the row's m and l in one batch, its w, the O tiles double-buffered in batches of ``COMBINE_UNIT_BATCH`` units
    of one dest window, M and L, the fp32 accumulator, the bf16 output, the cur_pos stick."""
    if stats_slots is None:
        return (
            2 * (2 + ct) * F32_TILE_BYTES
            + 7 * F32_TILE_BYTES
            + 2 * ct * F32_TILE_BYTES
            + ct * BF16_TILE_BYTES
            + pos_stick
        )
    return (
        3 * int(stats_slots) * F32_TILE_BYTES
        + 2 * COMBINE_UNIT_BATCH * combine_granularity(ct) * F32_TILE_BYTES
        + 2 * F32_TILE_BYTES
        + ct * F32_TILE_BYTES
        + ct * BF16_TILE_BYTES
        + pos_stick
    )


@dataclasses.dataclass(frozen=True)
class SplitfreePlan:
    split: int  # K: a row splits into at most K units of nsub_of(cur_pos) chunks (0: the fixed unit of ``nsub``)
    nsub_min: int  # NSUB_MIN: the fewest chunks of a position-derived unit
    grid: tuple
    n_cores: int
    batch: int  # padded rows B
    groups: int  # local KV groups G (the cache's heads this call attends)
    heads_per_group: int
    head_dim: int
    max_context: int  # the page table's reach: max_blocks * block_size
    window: int  # 0 = attend everything
    chunk: int
    nsub: int  # chunks of the fixed unit (split 0: the legacy unit; one on a windowed layer); 0 with split > 0
    max_units: int
    n_slots: int
    modulus: int
    stride: int  # the unit-class stride over the cores (>= batch * groups, coprime with n_cores)
    stride_inv: int  # stride^-1 mod n_cores (0 on a one-core grid)
    kv_tile_bytes: int
    qk_tile_bytes: int  # the scores CB: 4096 (fp32) or 2048 (bf16)
    reads_in_flight: int
    combine_tiles: int  # CT: the O tiles of one combine job (a divisor of DHt)
    stats_combine: bool  # the stats-first merge (M first, dest accumulation); False: the pairwise correction step
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
    def unit(self):
        """The fixed unit in tokens (split 0); 0 when the row's position decides it (``unit_of``)."""
        return self.nsub * self.chunk

    @property
    def pair_space(self):
        return self.modulus * self.batch * self.groups

    @property
    def ncb(self):
        """NCB: the O blocks (combine jobs) of one (row, group)."""
        return self.dht // self.combine_tiles

    @property
    def combine_jobs(self):
        return self.batch * self.groups * self.ncb

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
        """The slots any page-table reach needs: ``K`` per (row, group) with the position's unit (a row never has
        more than ``K`` units, whatever its length); with a fixed unit a narrower page table than the model's
        maximum has fewer units (``n_slots`` is the bound). A windowed layer's slot count is fixed by the
        window."""
        slots = self.split if self.split else self.n_slots
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

    def nsub_of(self, cur_pos):
        """Chunks per unit of a row at ``cur_pos`` (None for a padding row): the fixed ``nsub``, or
        ``max(nsub_min, ceil(n_chunks / split))`` from the row's own chunk count (the kernels' ``row_of``)."""
        r = self.row_range(cur_pos)
        if r is None:
            return None
        return nsub_of_chunks(r[1] + 1, self.split, self.nsub_min) if self.split else self.nsub

    def unit_of(self, cur_pos):
        """Tokens per unit of a row at ``cur_pos`` (None for a padding row)."""
        nsub = self.nsub_of(cur_pos)
        return None if nsub is None else nsub * self.chunk

    def live_units(self, cur_pos):
        r = self.row_range(cur_pos)
        if r is None:
            return range(0)
        nsub = self.nsub_of(cur_pos)
        return range(r[0] // nsub, r[1] // nsub + 1)

    def unit_chunks(self, u, cur_pos):
        r = self.row_range(cur_pos)
        if r is None:
            return range(0)
        nsub = self.nsub_of(cur_pos)
        return range(max(u * nsub, r[0]), min((u + 1) * nsub - 1, r[1]) + 1)

    def core_of(self, b, g, u):
        return ((u % self.modulus) * self.stride + b * self.groups + g) % self.n_cores

    def pairs_of_core(self, core):
        """The ``(u', b, g)`` pairs of a core in walking order: ascending ``bg``, the unit class
        ``u' = (core - bg) * stride^-1 mod n_cores`` when it is below the modulus (``M <= n_cores``: at most one
        per bg)."""
        out = []
        for bg in range(self.batch * self.groups):
            up = ((core - bg) % self.n_cores) * self.stride_inv % self.n_cores
            if up < self.modulus:
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

    def combine_core(self, b, g, blk=0):
        """The core of combine job ``(b, g, blk)``: job ``(b*G + g) * NCB + blk`` mod ``n_cores``."""
        return ((b * self.groups + g) * self.ncb + blk) % self.n_cores

    def combine_jobs_of_core(self, core, cur_pos):
        """The ``(b, g, blk)`` combine jobs of a core for the padded rows' ``cur_pos`` list, in the kernels' order
        (jobs ``core, core + n_cores, ...``; padding rows skipped)."""
        out = []
        for j in range(core, self.combine_jobs, self.n_cores):
            bg, blk = divmod(j, self.ncb)
            b, g = divmod(bg, self.groups)
            if self.row_range(cur_pos[b]) is not None:
                out.append((b, g, blk))
        return out

    def combine_tiles_of(self, blk):
        """The O tiles (columns of the head dim, in tiles) combine block ``blk`` merges and writes."""
        return range(blk * self.combine_tiles, (blk + 1) * self.combine_tiles)

    def combine_order(self, b, g, cur_pos):
        """The unit partials the combine of ``(b, g)`` merges, ascending: ``[(u, slot), ...]``."""
        return [(u, self.slot(b, g, u)) for u in self.live_units(cur_pos)]

    def describe(self):
        if self.split:
            unit = f"chunk {self.chunk}, unit max({self.nsub_min}, ceil(chunks / {self.split})) chunks"
        else:
            unit = f"chunk {self.chunk} x {self.nsub} = unit {self.unit}"
        return (
            f"B {self.batch} x G {self.groups} (x {self.heads_per_group} heads), head_dim {self.head_dim}, context "
            f"{self.max_context}, window {self.window or 'none'}: {unit}, "
            f"{self.max_units} units, {self.n_slots} slots, modulus {self.modulus}, {self.pair_space} pairs on "
            f"{self.n_cores} cores ({self.grid[0]}x{self.grid[1]}); {'stats' if self.stats_combine else 'pairwise'} "
            f"combine, {self.combine_jobs} jobs of {self.combine_tiles} O tiles; scratch {self.scratch_tiles} tiles = "
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
    so it is a build constant like the grid), ``window`` the sliding window (None / 0: attend everything),
    ``kv_dtype`` the cache format. The unit: ``SPLIT_ENV`` (default ``DEFAULT_SPLIT``) units at most per row of
    ``nsub_of(cur_pos)`` chunks, or the fixed ``UNIT_ENV`` tokens (``SPLIT=0``: ``LEGACY_UNIT``); one chunk with a
    window."""
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
    unit_raw = str(env.get(UNIT_ENV, "") or "").strip().lower()
    if unit_raw == "grid":
        raise ValueError(f"{UNIT_ENV}=grid is retired: the unit follows the row's position ({SPLIT_ENV})")
    split_raw = str(env.get(SPLIT_ENV, "") or "").strip()
    if unit_raw and split_raw not in ("", "0"):
        raise ValueError(f"{UNIT_ENV}={unit_raw} pins the fixed unit, {SPLIT_ENV}={split_raw} the position's")
    split = 0 if unit_raw else _env_int(env, SPLIT_ENV, DEFAULT_SPLIT, allow_zero=True)
    nsub_min = _env_int(env, MIN_CHUNKS_ENV, DEFAULT_MIN_CHUNKS)
    unit_cfg = _env_int(env, UNIT_ENV, LEGACY_UNIT, TILE)
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
    max_chunks = ceil_div(max_context, chunk)
    if window:
        split, nsub = 0, 1  # one chunk per unit: the live range is at most the window
    elif split:
        nsub = 0  # from the row's position, on device
    else:
        nsub = max(1, unit_cfg // chunk)  # the legacy 4096, or the env's pin
    max_units = min(split, max_chunks) if split else ceil_div(max_chunks, nsub)
    n_slots = max_units if not window else min(max_units, ceil_div(window - 1, chunk) + 1)
    modulus = min(n_cores, max_units)
    stride, stride_inv = unit_stride(batch * groups, n_cores)
    ct = _env_int(env, COMBINE_TILES_ENV, 0)
    if ct and dht % ct:
        raise ValueError(f"{COMBINE_TILES_ENV}={ct} must divide the head dim in tiles {dht}")
    ct = ct or choose_combine_tiles(batch * groups, dht, n_cores)
    combine = str(env.get(COMBINE_ENV, "") or "").strip().lower() or COMBINE_MODES[0]
    if combine not in COMBINE_MODES:
        raise ValueError(f"{COMBINE_ENV} must be one of {COMBINE_MODES}, got {combine}")
    stats_combine = combine == "stats"
    return SplitfreePlan(
        split=split,
        nsub_min=nsub_min,
        stride=stride,
        stride_inv=stride_inv,
        grid=(gx, gy),
        n_cores=n_cores,
        batch=batch,
        groups=groups,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        max_context=max_context,
        window=window,
        chunk=chunk,
        nsub=nsub,
        max_units=max_units,
        n_slots=n_slots,
        modulus=modulus,
        kv_tile_bytes=kv_tile,
        qk_tile_bytes=qk_tile,
        reads_in_flight=reads,
        combine_tiles=ct,
        stats_combine=stats_combine,
        partial_cb_bytes=pcb,
        combine_cb_bytes=combine_cb_bytes(ct, pos_stick, n_slots if stats_combine else None),
        budget_bytes=budget,
    )


def scratch_tiles_for(grid, batch, layer_specs, max_context, *, kv_dtype=None, arch=None, env=None):
    """The flat scratch (in fp32 tiles) one model needs on ``grid`` for ``batch`` padded rows at ``max_context``:
    the largest slot pool over its layer types (``scratch_tiles_bound``: ``K`` slots per (row, group) on a layer
    without a window, whatever the page table's reach), ``layer_specs``
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
