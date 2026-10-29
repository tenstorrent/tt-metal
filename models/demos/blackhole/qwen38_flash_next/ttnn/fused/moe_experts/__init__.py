# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``moe_experts``: the routed experts of one MoE layer at TP=8 (64 local experts per die), rows <= 2, as one
generic-op program on 24 cores (3 per DRAM bank), bit-identical to ``ttnn.experimental.moe_compute``'s local-output
path, on today's cached BF4 layout (no re-cache).  See x35_moe_kernel/DESIGN.md (Design A) and x37_moe_experts/REPORT.md.

Per call: every core reads the die's owner row and the drain core's top-10 indices and builds the same list of active
experts (rows <= 2 share one chunk per expert; no tilize cores, no mapping scan, no e_t page); the 12 gate/up cores
(8 pair units, 4 half units: x natural K order, one DEST, SiLU * up on the pack thread, Float16_b: moe_compute's compute.cpp
call for call) read the 80 hidden tiles' rows 0-1 from the gathered hidden's L1 shard and send their in2 tiles to the
16 down cores (24 down units: K in the cached W2's ring order, one DEST, pack_untilize_dest) which write the rows of
the tokens that selected the expert to the local combine buffer page ``k_slot * rows + row``.  The weights stream
continuously over the active experts (``slots``-block ring, trid pipelined reads of 11,520 B).  Replaces the untilize,
the two routing moves, the fill (unowned pages are not written), the tilize cores, the metadata gather and the 24-hop
a2a ring of ``moe_compute``.

This file holds the layout arithmetic the program and its tests share (a Python mirror of the cached layout of
``ttnn.experimental.moe_compute_utils`` / ``moe_ring_common.h`` for the 2560 / 640 expert on the 8-bank ring), the
worker placement and the program builder.
"""

from __future__ import annotations

import atexit
import os

import ttnn

from .. import program as fp

NAME = "moe_experts"
FLAG = "QWEN38_MOE_EXPERTS"
CALLS = {"total": 0}  # moe_experts() launches (program builds, eager or captured), for the A/B logs


def enabled() -> bool:
    """QWEN38_MOE_EXPERTS (default on, "0" off): the model's decode MoE (rows <= 2, fused post form) runs this program
    (x37 A/B at TP=8 B=1: -1.22 ms per token, tokens unchanged)."""
    return os.environ.get(FLAG, "1") == "1"


def _report_calls() -> None:
    if CALLS["total"]:
        print(f"MOE_EXPERTS_CALLS total={CALLS['total']}", flush=True)


atexit.register(_report_calls)
HIDDEN = 2560
INTERMEDIATE = 640
HT = HIDDEN // fp.TILE  # 80 K tiles of the gate/up matmul
NT = INTERMEDIATE // fp.TILE  # 20 K tiles of the down matmul = gate/up output columns
RING = 8  # the ring = the DRAM banks of the die
TOP_K = 10
EXPERTS_PER_DEVICE = 64
BF4_TILE_BYTES = fp.TILE_BYTES[ttnn.bfloat4_b]
TILES_PER_TXN = 20  # one Blackhole NoC packet of BF4 tiles (11,520 B)
BLOCK_TILES = 2 * TILES_PER_TXN  # a block: two transactions, 4 tiles wide x 10 K rows (or 2 wide x 20 half rows)
BLOCK_BYTES = BLOCK_TILES * BF4_TILE_BYTES
BANK_BLOCKS_PER_EXPERT = 10  # W0/W1 bank piece: 10 blocks = 400 tiles per (expert, bank)
W2_BLOCKS_PER_EXPERT = 5  # per ring core: iteration 0 (2 blocks), iteration 1 (2 blocks), the half iteration (1 block)
W2_TILES_PER_CORE_EXPERT = W2_BLOCKS_PER_EXPERT * BLOCK_TILES  # 200
W01_TILES_PER_BANK_EXPERT = BANK_BLOCKS_PER_EXPERT * BLOCK_TILES  # 400


def _is_big(core: int, n_big: int, n_cores: int) -> bool:
    return n_big > 0 and (core * n_big) % n_cores < n_big


def shard_tiles(n_tiles: int, core: int, n_cores: int = RING) -> int:
    """moe_ring::shard_tiles: the gate/up columns (and down-matmul K tiles of the ring order) of ring core ``core``."""
    return n_tiles // n_cores + (1 if _is_big(core, n_tiles % n_cores, n_cores) else 0)


COLUMNS = [shard_tiles(NT, c) for c in range(RING)]  # [3, 2, 3, 2, 3, 2, 3, 2]
COLUMN_START = [sum(COLUMNS[:c]) for c in range(RING + 1)]  # [0, 3, 5, 8, 10, 13, 15, 18, 20]


def w2_k_order(core: int) -> list[int]:
    """The natural gate/up column index (0..19) of the k-th K row of ring core ``core``'s down matmul: the cached W2
    stores its K rows in the ring order, core, core - 1, ... (``prepare_w2_tensor_for_moe_compute``), each source's
    columns in ascending order."""
    order = []
    for step in range(RING):
        src = (core - step) % RING
        order += list(range(COLUMN_START[src], COLUMN_START[src + 1]))
    return order


def stream_block_ranges() -> dict[str, tuple[int, int]]:
    """The gate/up units of one expert and their block ranges ``[first, last)`` in the expert's stream (ring cores'
    slices back to back: a pair unit = 8 blocks (2 columns, K 80 as 8 x 10 rows), a half unit = 4 blocks (1 column,
    K 80 as 4 x 20 stored rows)); bank b holds stream blocks 10 b .. 10 b + 9."""
    ranges, block = {}, 0
    for c in range(RING):
        ranges[f"P{c}"] = (block, block + 8)
        block += 8
        if COLUMNS[c] == 3:
            ranges[f"H{c}"] = (block, block + 4)
            block += 4
    assert block == RING * BANK_BLOCKS_PER_EXPERT
    return ranges


# ---------------------------------------------------------------- placement


def bank_workers(mesh) -> list[ttnn.CoreCoord]:
    """The logical worker core of each DRAM bank (the NOC-0 reader cores of the optimal bank assignment) of the
    mesh's reference die: the one whose workers sit furthest east (largest column sum, ties: the first), as the
    streamed ops of this repository place their readers (a reader west of its die's own bank worker reads slower)."""
    best = None
    rows, cols = int(mesh.shape[0]), int(mesh.shape[1])
    for coord in (ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)):
        assignment = mesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, coord)
        workers = [assignment[b] for b in sorted(assignment)]
        east = sum(w.x for w in workers)
        if best is None or east > best[0]:
            best = (east, workers)
    return list(best[1])


def row_placement(mesh, per_bank: int, exclude=()) -> dict[int, list[ttnn.CoreCoord]]:
    """``per_bank`` logical cores per DRAM bank on the row of the bank's worker: nearest first, a row shared by two
    banks split at the right bank's worker column, a short bank topped up from the rest of its row, then the row
    above, then below (the placement the gpt-oss streamed ops measured fastest: readers off the bank's row lose NoC
    arbitration to the row's own).  Returns ``{bank: [CoreCoord] * per_bank}``; the bank's own worker is first."""
    workers = bank_workers(mesh)
    grid = mesh.compute_with_storage_grid_size()
    used = {tuple(c) for c in exclude}
    out = {b: [] for b in range(len(workers))}
    cand = {}
    for b, w in enumerate(workers):
        mates = sorted(c.x for c in workers if c.y == w.y)
        k, n = mates.index(w.x), len(mates)
        lo = w.x if k else 0
        hi = mates[k + 1] if k + 1 < n else grid.x
        cand[b] = [(w.y, range(lo, hi)), (w.y, range(grid.x)), (w.y - 1, range(grid.x)), (w.y + 1, range(grid.x))]
    for phase in range(4):
        for b, w in enumerate(workers):
            y, xs = cand[b][phase]
            if not 0 <= y < grid.y:
                continue
            for x in sorted(xs, key=lambda x: (abs(x - w.x), x)):
                if len(out[b]) < per_bank and (x, y) not in used:
                    used.add((x, y))
                    out[b].append(ttnn.CoreCoord(x, y))
    for b in out:
        if len(out[b]) != per_bank:
            raise RuntimeError(f"row placement: {len(out[b])} cores for bank {b}, wanted {per_bank}")
    return out


def reader_vcs(placed: list[tuple[ttnn.CoreCoord, int, int]]) -> list[int]:
    """The NoC virtual channel of each ``(core, bank, reader index)``: ``(bank + reader) & 3``, bumped while an
    earlier core of the same row has it."""
    vcs = []
    for i, (core, bank, r) in enumerate(placed):
        vc = (bank + r) & 3
        for j in range(i):
            if placed[j][0].y == core.y and vcs[j] == vc:
                vc = (vc + 1) & 3
        vcs.append(vc)
    return vcs


# ---------------------------------------------------------------- the program

BF16, U16, U32 = ttnn.bfloat16, ttnn.uint16, ttnn.uint32
CB_X, CB_OUT, CB_W, CB_GOUT, CB_TOK, CB_IN2, CB_MD, CB_SCRATCH, CB_MDW = 0, 1, 3, 4, 5, 6, 7, 8, 9
SEM_IN2, SEM_ACK = (0, 1), (2, 3)  # per-parity counters of the in2 arrivals / acknowledgements
MD_PAGE_BYTES = 384
KERNELS = {name: fp.kernel_source(NAME, f"{name}_moe_experts.cpp") for name in ("reader", "compute", "writer")}
KERNELS["reader_fused"] = fp.kernel_source(NAME, "reader_moe_experts_fused.cpp")  # QWEN38_MOE_FUSE=1 only
UNITS_PER_BANK = 3


def shard_banks(weights) -> list[int]:
    """The DRAM bank id of each shard (= ring core) of a cached expert tensor: the order of its memory config's shard grid."""
    return [int(r.start.x) for r in weights.memory_config().shard_spec.grid.ranges()]


class CoreRole:
    """One of the 24 cores: its gate/up unit (``g_kind`` 0 none, 1 pair, 2 half; its weight blocks as (bank id, block in
    the bank's piece of the expert); the first natural column of its in2 tiles) and its down units ``w_units``: a list
    of up to 2 of ``(w_kind, first output column tile, blocks)`` (kind 1: 4 wide, 2 blocks; kind 2: 2 wide, 1 block) of
    ring core ``ring`` (the K order of the down matmul is that ring core's)."""

    def __init__(self, core, ring, g_kind, g_blocks, g_col, w_units):
        self.core, self.ring = core, ring
        self.g_kind, self.g_blocks, self.g_col = g_kind, g_blocks, g_col
        self.w_units = w_units

    @property
    def w_blocks(self):
        return [b for _, _, blocks in self.w_units for b in blocks]


RECT_SHAPES = ((6, 4), (4, 6), (8, 3), (3, 8))


def rectangle_cores(mesh, exclude=(), shapes=RECT_SHAPES):
    """The 24 cores as one NoC rectangle (so that the in2 exchange is one multicast): the first rectangle of ``shapes``
    free of the block's reserved cores (``placement.reserved_cores``: the storage cores, the bank reader cores of the
    dense matmuls, the routing drain core) and of ``exclude``; when no shape is free of all of them, the first free of
    the storage cores and the drain core alone (the bank reader cores only host the dense linears' L1 outputs; a program
    CB that clashes with a live L1 buffer is refused by the runtime, not corrupted)."""
    from .. import placement

    grid = mesh.compute_with_storage_grid_size()
    hard = frozenset(placement.STORAGE_CORES) | {(int(placement.drain_core(mesh).x), int(placement.drain_core(mesh).y))}
    for taken in (
        placement.reserved_cores(mesh) | frozenset(tuple(c) for c in exclude),
        hard | frozenset(tuple(c) for c in exclude),
    ):
        for width, height in shapes:
            try:
                rect = placement.free_rectangle_in(grid.x, grid.y, taken, width, height)
            except RuntimeError:
                continue
            return placement.core_range(rect), [ttnn.CoreCoord(x, y) for x, y in placement.rectangle_cores(rect)]
    raise RuntimeError(f"no free rectangle of 24 cores in the {grid.x}x{grid.y} compute grid")


def assign_banks(mesh, cores, banks: list[int]) -> dict[int, list[ttnn.CoreCoord]]:
    """Ring core (shard) ``s`` -> its 3 cores: the nearest (Manhattan distance to the bank's reader worker) free cores,
    globally greedy by distance; a bank's cores sorted nearest first."""
    workers = bank_workers(mesh)
    pairs = sorted(
        (abs(c.x - workers[banks[s]].x) + abs(c.y - workers[banks[s]].y), s, c.y, c.x)
        for s in range(RING)
        for c in cores
    )
    taken, out = set(), {s: [] for s in range(RING)}
    for dist, s, y, x in pairs:
        if (x, y) in taken or len(out[s]) >= UNITS_PER_BANK:
            continue
        taken.add((x, y))
        out[s].append(ttnn.CoreCoord(x, y))
    assert all(len(v) == UNITS_PER_BANK for v in out.values())
    return out


def core_set_of(cores) -> ttnn.CoreRangeSet:
    """The fewest CoreRanges for ``cores``: runs of consecutive x on a row, equal runs of consecutive rows stacked (one
    range = one kernel group = one binary multicast of the dispatcher)."""
    by_row = {}
    for c in cores:
        by_row.setdefault(c.y, []).append(c.x)
    runs = []
    for y in sorted(by_row):
        xs = sorted(by_row[y])
        start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            runs.append([start, prev, y, y])
            if x is not None:
                start = prev = x
    merged = []
    for run in runs:
        for m in merged:
            if m[0] == run[0] and m[1] == run[1] and m[3] + 1 == run[2]:
                m[3] = run[3]
                break
        else:
            merged.append(run)
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, x1, y0, y1 in merged]
    )


_PLANS: dict = {}


def model_exclude(mesh) -> frozenset:
    """Cores whose L1 holds a live tensor or a persistent buffer of the decode step while the experts run: the block's
    reserved cores (the hidden's 5 storage cores, every die's bank reader cores of the dense linears, the routing drain
    core) and the transport cores of the row gather / reduce programs (their persistent landing buffers, row 0)."""
    from .. import gr_fold, placement

    cores = set(placement.reserved_cores(mesh))
    for pair in gr_fold.TRANSPORT.values():
        cores.update((int(c.x), int(c.y)) for c in pair)
    return frozenset(cores)


def plan(mesh, w0_w1, exclude=(), where="bank"):
    """Memoized ``_plan`` (one placement per mesh / bank order / exclusion set; computed at the first, eager call)."""
    key = (id(mesh), tuple(shard_banks(w0_w1)), frozenset(tuple(c) for c in exclude), where)
    if key not in _PLANS:
        roles, cores = _plan(mesh, w0_w1, exclude, where)
        assert not {(c.x, c.y) for c in cores} & {tuple(c) for c in exclude}, "moe_experts cores overlap excluded cores"
        _PLANS[key] = (roles, cores)
    return _PLANS[key]


def _plan(mesh, w0_w1, exclude=(), where="bank"):
    """The 24 cores of a NoC rectangle (3 per bank, near the bank's worker) and their units, per ring core ``s`` (shard
    ``s`` = bank ``banks[s]``): j0 = the pair unit P_s alone (4.4 us per expert, the DRAM-bound period); an even ring
    core's j1 = the half gate/up unit H_s + the small down unit g2, its j2 = the two big down units g0, g1; an odd
    ring core's j1 = g0 + g2, its j2 = g1 (down unit times 1.5 / 0.8 us).  12 cores have a gate/up unit (8 pair + 4
    half), the 16 j1 / j2 cores the 24 down units, so an in2 tile goes to 16 cores, not 24.  Returns
    ``(roles, rectangle)``."""
    banks = shard_banks(w0_w1)
    if where == "rect":
        _, cores = rectangle_cores(mesh, exclude)
        placement = assign_banks(mesh, cores, banks)
    else:  # 3 cores on the row of each bank's worker, off the cores whose L1 this program reads (the hidden's shard, the routing)
        from .. import placement as block_placement

        drain = block_placement.drain_core(mesh)
        hard = [*block_placement.STORAGE_CORES, (int(drain.x), int(drain.y))]
        by_bank = row_placement(mesh, UNITS_PER_BANK, [*hard, *exclude])
        placement = {s: by_bank[banks[s]] for s in range(RING)}
    ranges = stream_block_ranges()

    def g_blocks(name):
        first, last = ranges[name]
        return [(banks[b // BANK_BLOCKS_PER_EXPERT], b % BANK_BLOCKS_PER_EXPERT) for b in range(first, last)]

    roles = []
    for s in range(RING):
        cs = placement[s]
        banks_s = banks[s]
        g0 = (1, 10 * s, [(banks_s, 0), (banks_s, 1)])
        g1 = (1, 10 * s + 4, [(banks_s, 2), (banks_s, 3)])
        g2 = (2, 10 * s + 8, [(banks_s, 4)])
        roles.append(CoreRole(cs[0], s, 1, g_blocks(f"P{s}"), COLUMN_START[s], []))
        if (
            COLUMNS[s] == 3
        ):  # the half gate/up unit (2.2 us) + the small down unit; the other core takes the two big ones
            roles.append(CoreRole(cs[1], s, 2, g_blocks(f"H{s}"), COLUMN_START[s] + 2, [g2]))
            roles.append(CoreRole(cs[2], s, 0, [], 0, [g0, g1]))
        else:
            roles.append(CoreRole(cs[1], s, 0, [], 0, [g0, g2]))
            roles.append(CoreRole(cs[2], s, 0, [], 0, [g1]))
    return roles, [r.core for r in roles]


def _check_inputs(hidden, indices, owner, w0_w1, w2, out) -> int:
    rows = fp.rows_of(hidden)
    if rows > 2:
        raise ValueError(f"moe_experts serves 1 or 2 rows, got {rows}")
    if not fp.is_tile_width(hidden, HIDDEN) or hidden.dtype != BF16 or not fp.in_l1(hidden):
        raise ValueError("moe_experts hidden must be a TILE bf16 [1, 1, rows, 2560] tensor in L1")
    if indices.dtype != U16 or tuple(indices.shape)[-2:] != (rows, TOP_K) or not fp.in_l1(indices):
        raise ValueError(
            f"moe_experts indices must be RM uint16 [.., {rows}, {TOP_K}] in L1, got {list(indices.shape)}"
        )
    if tuple(int(v) for v in owner.shape) != (1, 512) or owner.dtype != U16:
        raise ValueError("moe_experts owner must be a uint16 [1, 512] row")
    if (
        tuple(int(v) for v in out.shape) != (TOP_K, rows, HIDDEN)
        or out.dtype != BF16
        or out.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(f"moe_experts out must be RM bf16 [{TOP_K}, {rows}, {HIDDEN}], got {list(out.shape)}")
    if w0_w1.dtype != ttnn.bfloat4_b or w2.dtype != ttnn.bfloat4_b:
        raise ValueError("moe_experts weights must be BFLOAT4_B")
    return rows


def moe_experts_program(hidden, indices, owner, w0_w1, w2, out, **kwargs) -> "ttnn.ProgramDescriptor":
    kernels, cbs, sems, _cores = program_parts(hidden, indices, owner, w0_w1, w2, out, **kwargs)
    return fp.program_descriptor(kernels, cbs=cbs, semaphores=sems)


def program_parts(
    hidden,
    indices,
    owner,
    w0_w1,
    w2,
    out,
    *,
    rows,
    dump=None,
    stage=3,
    slots=8,
    exclude=(),
    where="bank",
    x_fragments=1,
    sem_base=0,
    wait_sem=None,
):
    """``(kernels, cbs, semaphores, cores)`` of the program; ``sem_base``: the first of its 4 semaphore ids,
    ``wait_sem``: the id of a semaphore (on the same cores, 0 at launch) the reader waits on before it reads the
    index rows (the fused form: another group of the program raises it when the indices are written)."""
    mesh = out.device()
    roles, role_cores = plan(mesh, w0_w1, exclude, where)
    banks = shard_banks(w0_w1)
    cores = core_set_of(role_cores)
    placed = [(r.core, banks[r.ring], j) for r, j in zip(roles, [0, 1, 2] * RING)]
    vcs = reader_vcs(placed)
    coords = [mesh.worker_core_from_logical_core(r.core) for r in roles]
    g_cores = [c for c, r in zip(coords, roles) if r.g_kind]
    named = [
        ("rows", rows),
        ("cb_x", CB_X),
        ("cb_out", CB_OUT),
        ("cb_w", CB_W),
        ("cb_gout", CB_GOUT),
        ("cb_tok", CB_TOK),
        ("cb_in2", CB_IN2),
        ("cb_md", CB_MD),
        ("cb_mdw", CB_MDW),
        ("cb_scratch", CB_SCRATCH),
        ("slots", slots),
        ("owner_bytes", 2 * 512),
        ("stage", stage),
        ("dump", 0 if dump is None else 1),
        ("sem_in2_0", sem_base),
        ("sem_in2_1", sem_base + 1),
        ("sem_ack_0", sem_base + 2),
        ("sem_ack_1", sem_base + 3),
        *(() if wait_sem is None else (("wait_sem", wait_sem),)),
        ("n_g", sum(1 for r in roles if r.g_kind)),
        ("n_w", sum(1 for r in roles if r.w_units)),
        ("x_fragments", x_fragments),
        ("activation_limit_bits", 0),
    ]
    pad_g, pad_w = [(0, 0)] * (8), [(0, 0)] * 4
    w_cores = [c for c, r in zip(coords, roles) if r.w_units]
    reader_args, compute_args, writer_args = [], [], []
    dump_t = out if dump is None else dump
    for r, vc in zip(roles, vcs):
        g = (r.g_blocks + pad_g)[:8]
        w = (r.w_blocks + pad_w)[:4]
        kinds = ([u[0] for u in r.w_units] + [0, 0])[:2]
        cols = ([u[1] for u in r.w_units] + [0, 0])[:2]
        reader_args.append(
            (
                r.core,
                [
                    w0_w1.buffer_address(),
                    w2.buffer_address(),
                    hidden.buffer_address(),
                    indices.buffer_address(),
                    owner.buffer_address(),
                    vc,
                    r.g_kind,
                    len(r.g_blocks),
                    len(r.w_blocks),
                    *[v for pair in g for v in pair],
                    *[v for pair in w for v in pair],
                ],
            )
        )
        compute_args.append((r.core, [r.g_kind, len(r.w_units), *kinds, *w2_k_order(r.ring)]))
        writer_args.append(
            (
                r.core,
                [
                    out.buffer_address(),
                    dump_t.buffer_address(),
                    vc,
                    r.g_kind,
                    len(r.w_units),
                    *kinds,
                    r.g_col,
                    *cols,
                    *[v for c in w_cores for v in (c.x, c.y)],
                    *[v for c in g_cores for v in (c.x, c.y)],
                ],
            )
        )
    cbs = [
        fp.cb_descriptor(CB_X, BF16, fp.TILE_BYTES[BF16], HT, cores),
        fp.cb_descriptor(CB_OUT, BF16, fp.TILE_BYTES[BF16], 8, cores),
        fp.cb_descriptor(CB_W, ttnn.bfloat4_b, BF4_TILE_BYTES, slots * BLOCK_TILES, cores),
        fp.cb_descriptor(CB_GOUT, BF16, fp.TILE_BYTES[BF16], 4, cores),
        fp.cb_descriptor(CB_TOK, U32, 16, 2, cores),
        fp.cb_descriptor(CB_IN2, BF16, fp.TILE_BYTES[BF16], 2 * NT, cores),
        fp.cb_descriptor(CB_MD, U32, MD_PAGE_BYTES, 1, cores),
        fp.cb_descriptor(CB_MDW, U32, MD_PAGE_BYTES, 1, cores),
        fp.cb_descriptor(CB_SCRATCH, U32, 2 * 512 + 128, 1, cores),
    ]
    sems = [fp.semaphore_descriptor(sem_base + i, cores, 0) for i in range(4)]
    if wait_sem is not None:
        sems.append(fp.semaphore_descriptor(wait_sem, cores, 0))
    reader = fp.reader_kernel(
        KERNELS["reader"] if wait_sem is None else KERNELS["reader_fused"],
        cores,
        [*fp.accessor_args(hidden), *fp.accessor_args(indices), *fp.accessor_args(owner)],
        reader_args,
        named=named,
    )
    writer = fp.writer_kernel(
        KERNELS["writer"], cores, [*fp.accessor_args(out), *fp.accessor_args(dump_t)], writer_args, named=named
    )
    compute = fp.compute_kernel(
        KERNELS["compute"],
        cores,
        [],
        compute_args,
        named=named,
        fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest=False,
        approx=True,
        dst_full_sync=False,
        bfp8_pack_precise=False,
    )
    return [reader, writer, compute], cbs, sems, role_cores


def moe_experts(
    hidden, indices, owner, w0_w1, w2, out, *, dump=None, stage=3, slots=8, exclude=(), where="bank", x_fragments=1
):
    """The routed experts of the die: writes the owned ``(slot, row)`` pages of ``out`` (the ``[10, rows, 2560]`` local
    combine buffer; the other pages are untouched).  ``hidden``: TILE bf16 ``[1, 1, rows, 2560]`` in L1 (the gathered
    hidden's 5-core shard); ``indices``: the drain core's L1 shard of the top-10 expert ids; ``owner``: this die's
    ``[1, 512]`` uint16 owner row (1 = owned); ``w0_w1`` / ``w2``: the cached BF4 tensors."""
    rows = _check_inputs(hidden, indices, owner, w0_w1, w2, out)
    CALLS["total"] += 1
    io = [hidden, indices, owner, w0_w1, w2] + ([] if dump is None else [dump]) + [out]
    fp.run_program(
        io,
        moe_experts_program(
            hidden,
            indices,
            owner,
            w0_w1,
            w2,
            out,
            rows=rows,
            dump=dump,
            stage=stage,
            slots=slots,
            exclude=exclude,
            where=where,
            x_fragments=x_fragments,
        ),
    )
    return out
