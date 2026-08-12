# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off of the rms_norm cross-core statistics COMBINE.

Not part of the op. Reconstructs ONLY the collective: every core of a reduction
rectangle starts with `rows_t` bf16 stat tiles resident in L1 (a pinned HEIGHT shard
— no reader, no DRAM, no statistics pipeline, no apply pass) and the group must land
`rsqrt(sum_of_partials + eps)` in every member's output shard.

Variants (see kernels/combine_dataflow.cpp for the protocols):
  baseline        the op's current two-level gather -> root finalize -> mcast-back.
  baseline_nohs   the same, with the return multicast's PRE_HANDSHAKE bit cleared —
                  isolates what the op pays for the mcast_pipe readiness ack.
  allgather       CANDIDATE: row leaders multicast their branch sums to the whole
                  rectangle; every core sums the S2 branch tiles and finalizes.
  flat_allgather  every member multicasts its OWN partial; every core sums G tiles.

  sum_mcast       the salvage: the baseline's gather tree and its ONE broadcast, but
                  the root broadcasts the raw SUM and every core finalizes locally.
  no_collective_ablation
                  PAYLOAD ABLATION (wrong answer by design): no collective at all,
                  every core finalizes its own partial. The program-launch + finalize
                  floor.

Held fixed across variants (the precision contract is NOT a lever): bf16 stat pages,
MathFidelity::HiFi2, fp32_dest_acc_en=False, identical tile counts, identical
per-core placement, identical finalize chain. Measured max relative error is identical
across variants on every geometry, so nothing here trades precision for speed.

=============================================================================
MEASURED — blackhole_p150b (11x10 worker grid, 1.35 GHz), DEVICE KERNEL DURATION [ns],
one fresh dispatch per cell; repeated only where a call sat near the noise band (~3-5%).
=============================================================================
geometry            blks  baseline  nohs   allgather  flat_ag   sum_mcast  ablation
focus_11x10  (110c)   1      3990   3470      21874   162753/       4211      1139
                                                      173868       /4190
focus_11x10_b4        4     15294      -     260870        -       13544  (1.12x)
                          /15207/                                 /13706/
                           15322/                                  13679/
                           15478                                   13681
focus_11x10_b16      16     60133      -          -        -       51318  (1.17x)  10277
wshard_8x1     (8c)   1      2204      -       2329        -        2235      1090
wshard_9x1     (9c)   1      2254      -       2423        -        2306
wshard_7x4    (28c)   1      2918   3304       5807     49566       2880      1136
wshard_8x4    (32c)   1      2970      -       4273        -        3021
wshard_7x4_b4         4      9803      -      79704        -        9881
bshard_8x1_r16        2     30658      -      31992        -       30767     16006
  (8 groups x 8c, R=16)
bshard_8x1_r16_b1     1     15626      -      16203     24996      15669
col_1x8  (1-wide)     1      2037      -       2193      3789        2076
small_3x3      (9c)   1      2570      -       3267        -        2559
small_4x2_r4_b2       2      9144      -      13278        -        9147

READING OF THE RESULT — why the all-gather loses.
Cost of each EXTRA broadcast, from the (senders, fan-out) sweep above:
    ny=1  ->  1 broadcast              flat vs baseline
    ny=3,  9 dests   +697 ns / 2 extra  = ~350 ns per broadcast
    ny=4, 28 dests  +2889 ns / 3 extra  = ~960 ns per broadcast
    ny=10,110 dests +17884 ns / 9 extra = ~1990 ns per broadcast
    110 senders, 110 dests: +169672 ns / 109 extra = ~1557 ns per broadcast
i.e. ~13-18 ns PER DESTINATION PER BROADCAST, and it ADDS UP across senders:
multicasts into the same rectangle do NOT overlap (the mcast path reservation is a
shared resource), so an S-sender all-gather costs S x one broadcast no matter how
concurrently it is issued. The baseline already uses the minimum broadcast count (1),
so there is no room on that axis — this is the same ranking examples/tensix_all_reduce
measured independently (`mcast_all_gather` loses to `reduce_root_mcast`).
The per-sender write BARRIER cannot explain it: 110 senders barriering concurrently
would cost one broadcast latency, not 110 (173868 ns is 110x a single broadcast).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_ZERO = 4  # pinned: page 0 = identity tile for the combine, page 1 = arrival flags
CB_STAT_PARTIAL = 7  # pinned input shard
CB_STAT_GATHER = 8
CB_STAT_SUM = 9
CB_RSTD_SEND = 10
CB_STAT_GATHER2 = 15
CB_OUT = 16  # pinned output shard
CB_BRANCH_SUM = 18
CB_AG = 19

SEM_GATHER = 0
SEM_MCAST_READY = 1
SEM_MCAST_CONSUMED = 2
SEM_GATHER2 = 3
SEM_AG_FREE = 4

MCAST_CT_BASE = 7  # combine_dataflow.cpp: McastArgs<7, 15>
MCAST_RT_BASE = 15

MODE = {
    "baseline": 0,
    "baseline_nohs": 0,
    "allgather": 1,
    "flat_allgather": 2,
    "no_collective_ablation": 3,
    "sum_mcast": 4,
}
# The ablation is deliberately NOT in the default variant list: it computes the wrong
# answer on purpose (it says how much of the wall is the collective at all).
VARIANTS = ("baseline", "baseline_nohs", "allgather", "flat_allgather", "sum_mcast")
ABLATIONS = ("no_collective_ablation",)


@dataclass(frozen=True)
class Geom:
    """One reduction-group geometry.

    nx, ny      group rectangle (cores along the grid x / y axis). The op's
                `_tree_for_box` picks (s1, s2) = (nx, ny) for a fully populated
                multi-row box and (G, 1) — the flat gather — otherwise; this bench
                mirrors that: ny == 1 IS the flat tree.
    rows_t      stat tiles per core per block (the op's block_row_tiles).
    num_blocks  blocks in the op's per-core block loop.
    num_groups  concurrent groups packed into the worker grid (NoC contention).
    """

    nx: int
    ny: int
    rows_t: int = 1
    num_blocks: int = 1
    num_groups: int = 1

    @property
    def g(self):
        return self.nx * self.ny

    @property
    def tiles_per_core(self):
        return self.rows_t * self.num_blocks


# The geometries the op actually lands on (see tests/.../perf_zone_harness.py CASES),
# plus small boxes where a dropped contribution is a loud correctness failure.
GEOMS = {
    # (1,1,32,7168) INTERLEAVED — THE FOCUS SHAPE: G=110 in one 11x10 box, C=2-3, R=1.
    "focus_11x10": Geom(11, 10, 1, 1, 1),
    # Same box, more blocks: amortizes launch overhead, exercises slot reuse, and
    # stands in for the PREFILL geometries (8192 rows => many blocks per core).
    "focus_11x10_b4": Geom(11, 10, 1, 4, 1),
    "focus_11x10_b16": Geom(11, 10, 1, 16, 1),
    # WIDTH-sharded perf geometries: wshard1024 (8,1) and wshard7168 (7,4).
    "wshard_8x1": Geom(8, 1, 1, 1, 1),
    "wshard_7x4": Geom(7, 4, 1, 1, 1),
    "wshard_9x1": Geom(9, 1, 1, 1, 1),
    "wshard_8x4": Geom(8, 4, 1, 1, 1),
    "wshard_7x4_b4": Geom(7, 4, 1, 4, 1),
    # BLOCK-sharded bshard1024 = [1024,128] on (8,8): 8 concurrent 8-core groups,
    # R=16 tile-rows, 2 blocks.
    "bshard_8x1_r16": Geom(8, 1, 16, 2, 8),
    "bshard_8x1_r16_b1": Geom(8, 1, 16, 1, 8),
    # HEIGHT-shard / degenerate-group representative: a 1-core group has no
    # collective at all; a 1x8 column is the op's nx == 1 flat case.
    "col_1x8": Geom(1, 8, 1, 1, 1),
    # Small boxes for the strict correctness gate.
    "small_2x2": Geom(2, 2, 1, 1, 1),
    "small_4x2": Geom(4, 2, 1, 1, 1),
    "small_4x2_r4_b2": Geom(4, 2, 4, 2, 1),
    "small_3x3": Geom(3, 3, 1, 1, 1),
}


def _f32_bits(x):
    import struct

    return int(struct.unpack("<I", struct.pack("<f", float(x)))[0])


@dataclass
class Layout:
    geom: Geom
    cores: list  # shard order: [(x, y), ...]
    core_ranges: ttnn.CoreRangeSet
    groups: list  # [{"box": (x0,y0,x1,y1), "cores": [...]}]


def build_layout(device, geom: Geom) -> Layout:
    grid = device.compute_with_storage_grid_size()
    across = grid.x // geom.nx
    down = grid.y // geom.ny
    if across == 0 or down == 0 or geom.num_groups > across * down:
        raise ValueError(
            f"cannot place {geom.num_groups} groups of {geom.ny}x{geom.nx} on a {grid.y}x{grid.x} worker grid"
        )
    ranges, groups = [], []
    for gi in range(geom.num_groups):
        gx = (gi % across) * geom.nx
        gy = (gi // across) * geom.ny
        box = (gx, gy, gx + geom.nx - 1, gy + geom.ny - 1)
        cores = [(gx + x, gy + y) for y in range(geom.ny) for x in range(geom.nx)]
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(box[0], box[1]), ttnn.CoreCoord(box[2], box[3])))
        groups.append({"box": box, "cores": cores})
    core_ranges = ttnn.CoreRangeSet(ranges)
    # Shard order == CoreRangeSet order, row-major inside each range. Derive the core
    # list from the range set itself so the torch<->core mapping cannot drift.
    cores = []
    for rng in core_ranges.ranges():
        for y in range(rng.start.y, rng.end.y + 1):
            for x in range(rng.start.x, rng.end.x + 1):
                cores.append((x, y))
    return Layout(geom, cores, core_ranges, groups)


def make_tensors(device, geom: Geom, layout: Layout, torch_partials):
    """(input, zeros, output) L1 HEIGHT-sharded tensors.

    `torch_partials` is [num_cores, tiles_per_core] — the per-core, per-tile stat
    value, broadcast over the whole 32x32 tile.
    """
    import torch

    n = len(layout.cores)
    t = geom.tiles_per_core
    in_torch = torch.zeros((n * 32 * t, 32), dtype=torch.float32)
    for c in range(n):
        for j in range(t):
            in_torch[(c * t + j) * 32 : (c * t + j + 1) * 32, :] = float(torch_partials[c][j])

    def sharded(rows_per_core):
        return ttnn.create_sharded_memory_config(
            shape=(rows_per_core, 32),
            core_grid=layout.core_ranges,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    tt_in = ttnn.from_torch(
        in_torch.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded(32 * t),
    )
    # Page 0 = the combine's identity operand, page 1 = the arrival flag words. HOST
    # zeroed, which is what makes the monotone flag protocol race-free (a fast sender
    # may write a flag before the receiving core's kernel starts).
    tt_zero = ttnn.from_torch(
        torch.zeros((n * 32 * 2, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded(64),
    )
    tt_out = ttnn.from_torch(
        torch.zeros((n * 32 * t, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded(32 * t),
    )
    return tt_in, tt_zero, tt_out


def _cb(index, core_ranges, num_pages, page_size, data_format):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _pinned_cb(index, tensor, core_ranges):
    desc = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
    desc.core_ranges = core_ranges
    return desc


def create_program_descriptor(device, geom, layout, tt_in, tt_zero, tt_out, variant, epsilon=1e-6):
    if variant not in MODE:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    mode = MODE[variant]
    if mode == 2 and geom.num_blocks > 1:
        # The flat all-gather makes EVERY member a sender, so the slot-reuse ack would
        # need G-1 unicast atomics per core (or a mcast whose sender sits inside its
        # own rect, which the atomic-increment API excludes). Single-block only.
        raise ValueError("flat_allgather is single-block only (see combine_dataflow.cpp)")
    # Mirrors the op's `_tree_for_box`: the two-level tree only for a fully populated
    # box that is more than one core wide AND more than one core tall; everything else
    # is the flat root-gather. `nx == 1` is deliberately NOT a tree there (level 1 would
    # be a self-write, costing an extra hop for the same number of tile adds).
    s1, s2 = (geom.nx, geom.ny) if (geom.nx > 1 and geom.ny > 1) else (geom.g, 1)
    g = geom.g
    rows_t, nb = geom.rows_t, geom.num_blocks
    ag_span = g if mode == 2 else (1 if mode == 4 else s2)
    tile_bytes = tt_in.buffer_aligned_page_size()
    fmt = tt_in.dtype

    # One mcast family per group, all adopting the same semaphore ids so the CT block
    # is uniform (exactly the op's wiring). num_active = G-1 is explicit.
    mcast_cfg = ttnn.McastConfig(noc=ttnn.NOC.NOC_1, sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED])
    mcasts = []
    for grp in layout.groups:
        x0, y0, x1, y1 = grp["box"]
        rect = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])
        root = ttnn.CoreCoord(x1, y1)  # the op's convention: members[-1] == the box corner
        mcasts.append(ttnn.Mcast2D(device, rect, root, mcast_cfg, g - 1))
    pre_handshake = None if variant != "baseline_nohs" else False
    mcast_ct = list(mcasts[0].compile_time_args(pre_handshake))

    df_ct = [mode, s1, s2, g, SEM_GATHER, SEM_GATHER2, SEM_AG_FREE]
    assert len(df_ct) == MCAST_CT_BASE
    df_ct += mcast_ct
    cp_ct = [mode, s1, s2, g, _f32_bits(epsilon)]

    df_rt = ttnn.RuntimeArgs()
    cp_rt = ttnn.RuntimeArgs()
    for gi, grp in enumerate(layout.groups):
        x0, y0, x1, y1 = grp["box"]

        def virt(x, y):
            c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            return int(c.x), int(c.y)

        vx0, vy0 = virt(x0, y0)
        vx1, vy1 = virt(x1, y1)
        root_vx, root_vy = virt(x1, y1)
        for idx, (cx, cy) in enumerate(grp["cores"]):
            row = idx // geom.nx
            col = idx % geom.nx
            is_root = 1 if (cx, cy) == (x1, y1) else 0
            if s2 > 1:
                # Two-level tree: leader = the box's right-hand column, root = the
                # bottom-right corner (the op's convention), level-1 slot = my column.
                leader_x, leader_y = virt(x1, cy)
                is_leader = 1 if cx == x1 else 0
                slot = col
            else:
                # FLAT tree: leader == root and every member's level-1 slot is its index
                # in the whole group — exactly what the op collapses to.
                leader_x, leader_y = root_vx, root_vy
                is_leader = is_root
                slot = idx
            if mode == 2:
                ag_slot = idx
            elif is_leader:
                ag_slot = row if s2 > 1 else 0
            else:
                ag_slot = 0
            core_rt = [
                rows_t,
                nb,
                slot,
                is_leader,
                is_root,
                row,
                ag_slot,
                leader_x,
                leader_y,
                root_vx,
                root_vy,
                vx0,
                vy0,
                vx1,
                vy1,
            ] + list(mcasts[gi].runtime_args(ttnn.CoreCoord(cx, cy)))
            assert len(core_rt) == MCAST_RT_BASE + 4, f"runtime-arg layout drifted: {len(core_rt)}"
            df_rt[cx][cy] = core_rt
            cp_rt[cx][cy] = [rows_t, nb, is_leader, is_root]

    crs = layout.core_ranges
    cbs = [
        _pinned_cb(CB_STAT_PARTIAL, tt_in, crs),
        _pinned_cb(CB_ZERO, tt_zero, crs),
        _pinned_cb(CB_OUT, tt_out, crs),
        _cb(CB_STAT_GATHER, crs, rows_t * s1, tile_bytes, fmt),
        _cb(CB_STAT_SUM, crs, rows_t, tile_bytes, fmt),
    ]
    if mode in (0, 4) and s2 > 1:
        cbs.append(_cb(CB_STAT_GATHER2, crs, rows_t * s2, tile_bytes, fmt))
    if mode == 0:
        cbs.append(_cb(CB_RSTD_SEND, crs, rows_t, tile_bytes, fmt))
    if mode != 0:
        cbs.append(_cb(CB_AG, crs, rows_t * ag_span, tile_bytes, fmt))
    if (mode in (0, 4) and s2 > 1) or mode == 1:
        cbs.append(_cb(CB_BRANCH_SUM, crs, rows_t, tile_bytes, fmt))
    semaphores = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0)
        for i in (SEM_GATHER, SEM_MCAST_READY, SEM_MCAST_CONSUMED, SEM_GATHER2, SEM_AG_FREE)
    ]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "combine_dataflow.cpp"),
            core_ranges=crs,
            compile_time_args=df_ct,
            runtime_args=df_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "combine_compute.cpp"),
            core_ranges=crs,
            compile_time_args=cp_ct,
            runtime_args=cp_rt,
            # The user's precision contract, identical in every variant.
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def partials_for(geom, layout, kind="exact"):
    """Per-core, per-tile stat values.

    exact    every core contributes 2^(tile index % 3). Powers of two with equal
             addends, so EVERY partial sum is exact in bf16 whatever the reduction
             order — the reference is then order-independent and a dropped/duplicated
             contribution is the only thing that can move the result.
    distinct per-core-distinct values (small G only): catches a slot-mapping bug that
             equal addends would hide.
    """
    n = len(layout.cores)
    t = geom.tiles_per_core
    if kind == "exact":
        return [[float(2 ** (j % 3)) for j in range(t)] for _ in range(n)]
    return [[float(2 ** (j % 3)) * (1.0 + 0.25 * ((c % 4) + 1)) for j in range(t)] for c in range(n)]


def reference(geom, layout, partials, epsilon=1e-6):
    """[num_cores, tiles_per_core] expected rstd — the group sum's rsqrt on every member."""
    per_group = {}
    for gi, grp in enumerate(layout.groups):
        idx = [layout.cores.index(c) for c in grp["cores"]]
        per_group[gi] = [sum(partials[i][j] for i in idx) for j in range(geom.tiles_per_core)]
    out = [None] * len(layout.cores)
    for gi, grp in enumerate(layout.groups):
        for c in grp["cores"]:
            out[layout.cores.index(c)] = [(s + epsilon) ** -0.5 for s in per_group[gi]]
    return out


def run(device, geom_name, variant, partial_kind="exact", epsilon=1e-6):
    """One dispatch. Returns (device_kernel_ns, max_rel_err, out_col0, expected)."""
    import torch

    geom = GEOMS[geom_name]
    layout = build_layout(device, geom)
    partials = partials_for(geom, layout, partial_kind)
    tt_in, tt_zero, tt_out = make_tensors(device, geom, layout, partials)
    desc = create_program_descriptor(device, geom, layout, tt_in, tt_zero, tt_out, variant, epsilon)

    ttnn.ReadDeviceProfiler(device)  # flush the tensor-upload traffic
    out = ttnn.generic_op([tt_in, tt_zero, tt_out], desc)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    ns = None
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if entry is not None:
                d = float(entry.duration)
                ns = d if ns is None else max(ns, d)

    got = ttnn.to_torch(out).to(torch.float32)
    if MODE[variant] == 3:
        # The no-collective ablation finalizes each core's OWN partial.
        exp = [[(p + epsilon) ** -0.5 for p in row] for row in partials]
    else:
        exp = reference(geom, layout, partials, epsilon)
    t = geom.tiles_per_core
    max_rel = 0.0
    col0 = []
    for c in range(len(layout.cores)):
        row = []
        for j in range(t):
            # rstd is consumed as an OperandKind::Col broadcast, i.e. column 0 alone;
            # the column-valid SFPU leaves columns 16..31 unfinalized by design.
            v = float(got[(c * t + j) * 32, 0])
            row.append(v)
            e = exp[c][j]
            max_rel = max(max_rel, abs(v - e) / abs(e))
        col0.append(row)
    return ns, max_rel, col0, exp
