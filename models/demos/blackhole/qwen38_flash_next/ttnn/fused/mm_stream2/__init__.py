# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``mm_stream2`` (QWEN38_MM_STREAM2=1, default off): a decode DRAM-sharded linear (one live row) on many compute cores, each
reading ONE contiguous byte range of a re-laid weight (x60).  This module: the placement and weight re-layout helpers shared
by the read micro-benchmark, the builder and the matmul program."""

from __future__ import annotations

import os

import ttnn

from .. import program as fp

NAME = "mm_stream2"
FLAG = "QWEN38_MM_STREAM2"
MAX_PACKET_BYTES = 16384
_WORKERS: dict = {}


def enabled() -> bool:
    return os.environ.get(FLAG, "0") == "1"


def bank_workers(mesh) -> list:
    """Logical worker core per DRAM bank (the optimal NOC0 reader), of the reference die (the one whose workers sit furthest east:
    the readers west of a bank's own worker read slower and the slowest chip sets the step; as gpt_oss fused_decode/stream.py).
    """
    key = id(mesh)
    cached = _WORKERS.get(key)
    if cached is None or cached[0] is not mesh:
        best = None
        rows, cols = mesh.shape[0], mesh.shape[1]
        for coord in (ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)):
            a = mesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, coord)
            workers = [a[b] for b in sorted(a)]
            east = sum(w.x for w in workers)
            if best is None or east > best[0]:
                best = (east, workers)
        cached = _WORKERS[key] = (mesh, best[1])
    return list(cached[1])


def row_placement(mesh, per_bank, exclude=(), per_bank_list=None):
    """``per_bank`` reader cores per DRAM bank on the row of the bank's optimal NOC0 worker (nearest first; a row shared by two
    banks is split at the right bank's worker column; then the rows above / below): readers off the bank's row lose NOC
    arbitration to the row's own (gpt_oss fused_decode/stream.py row_placement: 490 GB/s vs 464).  ``per_bank_list``: a count per
    bank instead.  Returns ``{bank: [CoreCoord]}`` and never uses a core of ``exclude``."""
    opt = bank_workers(mesh)
    grid = mesh.compute_with_storage_grid_size()
    used = {tuple(c) for c in exclude}
    want = per_bank_list or [per_bank] * len(opt)
    out, cand = {}, {}
    for b, w in enumerate(opt):
        mates = sorted(c.x for c in opt if c.y == w.y)
        k, n = mates.index(w.x), len(mates)
        lo = w.x if k else 0
        hi = mates[k + 1] if k + 1 < n else grid.x
        cand[b] = [(w.y, range(lo, hi)), (w.y, range(grid.x)), (w.y - 1, range(grid.x)), (w.y + 1, range(grid.x))]
        out[b] = []
    for phase in range(4):
        for b, w in enumerate(opt):
            y, xs = cand[b][phase]
            if not 0 <= y < grid.y:
                continue
            for x in sorted(xs, key=lambda x: (abs(x - w.x), x)):
                if len(out[b]) < want[b] and (x, y) not in used:
                    used.add((x, y))
                    out[b].append(ttnn.CoreCoord(x, y))
    for b in out:
        if len(out[b]) != want[b]:
            raise RuntimeError(f"row placement: {len(out[b])} cores for bank {b} (wanted {want[b]})")
    return out


def assign_vcs(placed):
    """NOC virtual channel per reader of ``placed`` = [(core, bank, r)] (distinct from earlier cores on the same row)."""
    vcs = []
    for i, (core, bank, r) in enumerate(placed):
        vc = (bank + r) & 0x3
        for j in range(i):
            if placed[j][0].y == core.y and vcs[j] == vc:
                vc = (vc + 1) & 0x3
        vcs.append(vc)
    return vcs


def merge_cores(cores):
    """The cores as a CoreRangeSet with adjacent cores merged into rectangles: a CoreRangeSet built from single-core ranges stays
    one range per core, and the dispatcher pays ~0.4 us per range for every CB / kernel / semaphore of the program (x60).
    """
    cores = list(cores)
    out = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
    for c in cores[1:]:
        out = out.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
    return out


def page_tiles(block_tiles: int, tile_bytes: int) -> int:
    """The most tiles of a NOC packet (<= 16 KB) that divide a block."""
    return max(d for d in range(1, block_tiles + 1) if block_tiles % d == 0 and d * tile_bytes <= MAX_PACKET_BYTES)


def weight_memory_config(banks: int, tiles_per_bank: int):
    """DRAM width-sharded: bank b's shard = ``tiles_per_bank`` tiles stacked ([tiles * 32, 32] shard), tile order = the layout's."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            (tiles_per_bank * fp.TILE, fp.TILE),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


# ------------------------------------------------------------------------------------------------ the weight re-layout

import math

import torch

from .. import mm_stream as m1
from .. import placement

TILE = fp.TILE
KERNELS = {
    "hub": fp.kernel_source(NAME, "hub_mm_stream2.cpp"),
    "reader": fp.kernel_source(NAME, "reader_mm_stream2.cpp"),
    "shard": fp.kernel_source("mm_stream", "shard_mm_stream.cpp"),
    "writer": fp.kernel_source("mm_stream", "writer_mm_stream.cpp"),
    "compute": fp.kernel_source("mm_stream", "compute_mm_stream.cpp"),
}
CB_X, CB_W, CB_OUT, CB_STAGE = m1.CB_X, m1.CB_W, m1.CB_OUT, m1.CB_STAGE
SEM_X, SEM_HUB = m1.SEM_X, m1.SEM_HUB
FID = m1.FID
BANKS = 8


def column_groups(n: int, nb: int, cols: int, banks: int = BANKS):
    """``(bank, first column in the bank, ncols, first global tile column)`` of every core (m1.column_groups)."""
    return m1.column_groups(n, nb, banks, cols)


def relayout(w: torch.Tensor, nb: int, cols: int, banks: int = BANKS) -> torch.Tensor:
    """A weight ``[K, N]`` (N <= banks * nb * 32, zero padded) -> ``[T * 32, banks * 32]`` with T = (K / 32) * nb: bank b's shard column
    holds, core group after core group (m1.column_groups), the group's tiles K row by K row (``ncols`` adjacent tile columns per row):
    one core's whole weight is ONE contiguous byte range of its bank's shard (offset = first column in the bank * K tiles * tile bytes).
    """
    k, n = w.shape
    kt = k // TILE
    wp = torch.zeros(k, banks * nb * TILE, dtype=w.dtype)
    wp[:, :n] = w
    tiles = wp.reshape(kt, TILE, banks * nb, TILE).permute(2, 0, 1, 3)  # [col, k, 32, 32]
    out = torch.zeros(kt * nb, banks, TILE, TILE, dtype=w.dtype)
    for b, c0, ncols, _first in column_groups(n, nb, cols, banks):
        seg = tiles[b * nb + c0 : b * nb + c0 + ncols].permute(1, 0, 2, 3).reshape(kt * ncols, TILE, TILE)  # [k, j]
        out[kt * c0 : kt * (c0 + ncols), b] = seg
    return out.permute(0, 2, 1, 3).reshape(kt * nb * TILE, banks * TILE)


def unlayout(r: torch.Tensor, k: int, n: int, nb: int, cols: int, banks: int = BANKS) -> torch.Tensor:
    """The inverse of ``relayout`` (the real columns; the padding is dropped)."""
    kt = k // TILE
    t = r.reshape(kt * nb, TILE, banks, TILE).permute(0, 2, 1, 3)  # [tile, bank, 32, 32]
    wp = torch.zeros(banks * nb, kt, TILE, TILE, dtype=r.dtype)
    for b, c0, ncols, _first in column_groups(n, nb, cols, banks):
        seg = t[kt * c0 : kt * (c0 + ncols), b].reshape(kt, ncols, TILE, TILE).permute(1, 0, 2, 3)
        wp[b * nb + c0 : b * nb + c0 + ncols] = seg
    return wp.permute(1, 2, 0, 3).reshape(k, banks * nb * TILE)[:, :n]


def cache_name(base: str, k: int, n: int, nb: int, cols: int) -> str:
    """A new cache file tag for the re-laid weight (never the stock tensorbin's name)."""
    return f"{base}_mm2_k{k}_n{n}_nb{nb}_c{cols}"


def to_device(mesh, r: torch.Tensor, dtype, nb: int, kt: int, cache_file_name=None):
    """The re-laid host tensor ``[T*32, 8*32]`` -> one DRAM width-sharded tensor (tile column per bank)."""
    return ttnn.as_tensor(
        r.reshape(1, 1, *r.shape),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=weight_memory_config(BANKS, kt * nb),
        cache_file_name=cache_file_name,
    )


def from_stock(mesh, weight, k: int, n: int, nb: int, cols: int, cache_file_name=None):
    """Re-lay a stock DRAM-sharded weight (its decoded values; BFP8 -> bf16 -> BFP8 is exact for decoded values) into the x60 layout."""
    w = ttnn.to_torch(weight).reshape(-1, weight.shape[-1])[:k, :n].to(torch.bfloat16)
    return to_device(mesh, relayout(w, nb, cols), weight.dtype, nb, k // TILE, cache_file_name)


# ----------------------------------------------------------------------------------------------------------- the program

_PLANS: dict = {}


def _plan(mesh, groups, act_cores, exclude, place="rect"):
    """Per group: its core (row placement next to the group's bank, never on excluded / activation / output cores), the hub core, the
    VCs, the compute ranges (the cores' merged range set, each with its core count)."""
    place = os.environ.get("X60_PLACE", place)
    key = (id(mesh), tuple(groups), tuple(sorted(map(tuple, exclude))), tuple(act_cores), place)
    if key in _PLANS:
        return _PLANS[key]
    taken = set(map(tuple, exclude)) | set(map(tuple, act_cores)) | set(placement.reserved_cores(mesh))
    per_bank = [sum(1 for g in groups if g[0] == b) for b in range(BANKS)]
    placed, assign = [], {}
    seen = [0] * BANKS
    if (
        place == "two"
    ):  # study: one block of columns next to each side's bank workers (two rectangles: few ranges, near the banks)
        grid0 = mesh.compute_with_storage_grid_size()
        workers = bank_workers(mesh)
        xs = sorted({w.x for w in workers})
        a_ = {}
        for side, wx in enumerate(xs):
            banks_s = [b for b, w in enumerate(workers) if w.x == wx]
            idx = [i for i, g in enumerate(groups) if g[0] in banks_s]
            width = int(os.environ.get("X60_TWOW", "3"))
            cells = [
                (x, y)
                for x in range(wx + 1, min(grid0.x, wx + 1 + width))
                for y in range(grid0.y)
                if (x, y) not in taken
            ]
            cells = cells[: max(len(idx), 0)] if os.environ.get("X60_TWOFILL", "prefix") == "prefix" else cells
            asg, _ = m1._assign(
                cells,
                [groups[i] for i in idx],
                list(range(BANKS)),
                m1.bank_workers_dict(mesh)
                if hasattr(m1, "bank_workers_dict")
                else {b: (workers[b].x, workers[b].y) for b in range(BANKS)},
            )
            for j, i in enumerate(idx):
                a_[i] = asg[j]
        for i, g in enumerate(groups):
            core = ttnn.CoreCoord(*a_[i])
            placed.append((core, g[0], seen[g[0]]))
            assign[i] = core
            seen[g[0]] += 1
    elif (
        place == "rect"
    ):  # study: ONE rectangle (one range per CB / kernel / semaphore), each group on the nearest free core
        grid0 = mesh.compute_with_storage_grid_size()
        rect = None
        for w, h in sorted(
            (
                (w, h)
                for w in range(1, grid0.x + 1)
                for h in range(1, grid0.y + 1)
                if len(groups) <= w * h < len(groups) + 4
            ),
            key=lambda t: (t[0] * t[1], -t[0]),
        ):
            try:
                rect = placement.free_rectangle_in(grid0.x, grid0.y, taken, w, h)
                break
            except RuntimeError:
                continue
        rect_cores_ = placement.rectangle_cores(rect)
        a_, _ = m1._assign(rect_cores_, groups, list(range(BANKS)), m1.bank_workers(mesh))
        for i, g in enumerate(groups):
            core = ttnn.CoreCoord(*a_[i])
            placed.append((core, g[0], seen[g[0]]))
            assign[i] = core
            seen[g[0]] += 1
    else:
        placed_by_bank = row_placement(mesh, 0, exclude=taken, per_bank_list=per_bank)
        for i, g in enumerate(groups):
            b = g[0]
            core = placed_by_bank[b][seen[b]]
            placed.append((core, b, seen[b]))
            assign[i] = core
            seen[b] += 1
    vcs = assign_vcs(placed)
    cores = {(c.x, c.y) for c, _, _ in placed}
    grid = mesh.compute_with_storage_grid_size()
    hub = None
    # the hub next to the activation shard cores, so that hub + shards are ONE rectangle (the staging CB is then one core range)
    ac = sorted(map(tuple, act_cores), key=lambda c: (c[1], c[0]))
    if ac and len({c[1] for c in ac}) == 1:
        cand = (ac[-1][0] + 1, ac[0][1])
        if cand not in taken and cand not in cores and cand[0] < grid.x:
            hub = ttnn.CoreCoord(*cand)
    for y in range(grid.y - 1, -1, -1) if hub is None else ():
        for x in range(grid.x - 1, -1, -1):
            if (x, y) not in taken and (x, y) not in cores:
                hub = ttnn.CoreCoord(x, y)
                break
        if hub:
            break
    merged = merge_cores([c for c, _, _ in placed])
    ranges = []
    for r in merged.ranges():
        n = (r.end.x - r.start.x + 1) * (r.end.y - r.start.y + 1)
        ranges.append((r, n))
    assert sum(n for _, n in ranges) == len(placed), "compute ranges must hold exactly the compute cores"
    _PLANS[key] = (assign, vcs, hub, merged, ranges)
    return _PLANS[key]


def mm_stream2_program(
    act, weight, out, *, k: int, n: int, nb: int, blk: int, cols: int, slots: int = 6, exclude=(), place="rect"
):
    """Program for ``act @ weight`` (weight in the x60 layout of ``relayout(.., nb, cols)``): row 0 of the output tiles."""
    mesh = out.device()
    kt = k // TILE
    wdtype = weight.dtype
    tile_bytes = fp.TILE_BYTES[wdtype]
    if kt % blk or cols > 4:
        raise ValueError("blk must divide the K tiles and cols <= 4")
    shard_cores = [
        c
        for r in weight.memory_config().shard_spec.grid.ranges()
        for c in placement.rectangle_cores((r.start.x, r.start.y, r.end.x, r.end.y))
    ]
    if [tuple(c) for c in shard_cores] != [(b, 0) for b in range(BANKS)]:
        raise ValueError(f"mm_stream2 needs the weight sharded one tile column per DRAM bank 0..7, got {shard_cores}")
    groups = column_groups(n, nb, cols)
    count = len(groups)
    act_cores = [
        c
        for r in act.memory_config().shard_spec.grid.ranges()
        for c in placement.rectangle_cores((r.start.x, r.start.y, r.end.x, r.end.y))
    ]
    shards = len(act_cores)
    shard_tiles = kt // shards
    out_cores = set()
    out_l1 = out.memory_config().is_sharded()
    if out_l1:
        out_cores = {
            c
            for r in out.memory_config().shard_spec.grid.ranges()
            for c in placement.rectangle_cores((r.start.x, r.start.y, r.end.x, r.end.y))
        }
    live = set(map(tuple, exclude)) | out_cores | set(map(tuple, act_cores))
    assign, vcs, hub_cc, merged, ranges = _plan(mesh, groups, act_cores, live, place)
    cores_used = {(c.x, c.y) for c in assign.values()} | {(hub_cc.x, hub_cc.y)}
    if cores_used & live:
        raise RuntimeError(f"mm_stream2 cores overlap live cores: {sorted(cores_used & live)}")
    active_set = merged
    hub_set = ttnn.CoreRangeSet([ttnn.CoreRange(hub_cc, hub_cc)])
    act_ccs = [ttnn.CoreCoord(*c) for c in act_cores]
    shard_set = merge_cores(act_ccs)
    x_cores = merged.merge(hub_set).merge(
        shard_set
    )  # the compact activation: at ONE address on the compute, hub and shard cores (no staging CB)
    stage_cores = hub_set.merge(shard_set)
    _abl = os.environ.get("X60_ABL", "")
    if "stagehub" in _abl:
        stage_cores = hub_set
    elif "stageshard" in _abl:
        stage_cores = shard_set
    sem_cores = merged.merge(hub_set).merge(shard_set)
    hub_xy = mesh.worker_core_from_logical_core(hub_cc)
    shard_out_tiles = 1
    out_shard_cores = []
    if out_l1:
        spec = out.memory_config().shard_spec
        if spec.shape[0] != fp.TILE or str(spec.orientation) != "ShardOrientation.ROW_MAJOR":
            raise ValueError("mm_stream2 writes L1 width shards of one row tile, row-major")
        out_shard_cores = [
            c for r in spec.grid.ranges() for c in placement.rectangle_cores((r.start.x, r.start.y, r.end.x, r.end.y))
        ]
        shard_out_tiles = spec.shape[1] // fp.TILE
    named = [
        ("out_l1", int(out_l1)),
        ("shard_out_tiles", shard_out_tiles),
        ("cb_x", CB_X),
        ("cb_w", CB_W),
        ("cb_out", CB_OUT),
        ("cb_stage", CB_X),
        ("sem_x", SEM_X),
        ("sem_hub", SEM_HUB),
        ("kt", kt),
        ("blk", blk),
        ("slots", slots),
        ("shards", shards),
        ("shard_tiles", shard_tiles),
        ("w_tile_bytes", tile_bytes),
    ]
    shard_xy = []
    for c in out_shard_cores:
        p_ = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        shard_xy += [p_.x, p_.y]
    tiny = ttnn.TileDescriptor(1, fp.TILE)

    def tiny_cb(index, pages, cores_):
        return ttnn.CBDescriptor(
            total_size=pages * 64,
            core_ranges=cores_,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=64, tile=tiny)
            ],
        )

    def cset(idx):
        return merge_cores([assign[i] for i in idx])

    ncols_set = sorted({g[2] for g in groups})
    by_nc = {nc: [i for i in range(count) if groups[i][2] == nc] for nc in ncols_set}
    cbs = [tiny_cb(CB_X, kt, x_cores), tiny_cb(CB_OUT, cols, active_set)]
    kernels = []
    for nc, idx in by_nc.items():
        cs = cset(idx)
        cbs.append(fp.cb_descriptor(CB_W, wdtype, tile_bytes, slots * blk * nc, cs))
        block_tiles = blk * nc
        rnamed = named + [
            ("cb_w", CB_W),
            ("blocks", kt // blk),
            ("block_tiles", block_tiles),
            ("page_tiles", page_tiles(block_tiles, tile_bytes)),
        ]
        rt = [(assign[i], [weight.buffer_address(), groups[i][0], groups[i][1] * kt * tile_bytes, vcs[i]]) for i in idx]
        kernels.append(fp.reader_kernel(KERNELS["reader"], cs, [], rt, named=rnamed))
    ranges_args = [len(ranges), count]
    for r, nn in ranges:
        a = mesh.worker_core_from_logical_core(r.start)
        b = mesh.worker_core_from_logical_core(r.end)
        ranges_args += [a.x, a.y, b.x, b.y, nn]
    kernels.append(fp.reader_kernel(KERNELS["hub"], hub_set, [], [(hub_cc, ranges_args)], named=named))
    kernels.append(
        fp.reader_kernel(
            KERNELS["shard"],
            shard_set,
            [],
            [(c, [act.buffer_address(), hub_xy.x, hub_xy.y, i]) for i, c in enumerate(act_ccs)],
            named=named,
        )
    )
    writer_args = [
        (assign[i], [out.buffer_address(), groups[i][3], groups[i][2], hub_xy.x, hub_xy.y, *shard_xy])
        for i in range(count)
    ]
    kernels.append(fp.writer_kernel(KERNELS["writer"], active_set, fp.accessor_args(out), writer_args, named=named))
    kernels.append(
        fp.compute_kernel(
            KERNELS["compute"],
            active_set,
            [],
            [(assign[i], [groups[i][2]]) for i in range(count)],
            named=named,
            fidelity=FID[wdtype],
            fp32_dest=True,
            approx=False,
            dst_full_sync=False,
        )
    )
    abl = os.environ.get("X60_ABL", "")
    if abl:  # dispatch-cost ablation (study only): empty kernels, optionally without some of the kernel groups
        empty = fp.kernel_source(NAME, "empty_mm_stream2.cpp")
        for kd in kernels:
            if "empty" in abl:
                kd.kernel_source = empty
        nr = len(by_nc)
        tags = ["reader"] * nr + ["hub", "shard", "writer", "compute"]
        kernels = [kd for kd, t in zip(kernels, tags) if f"no{t}" not in abl]
        keep = [t for t in abl.split(",") if t.startswith("cbs=")]
        if keep:  # keep only the CBs named by the letters (w, x, o)
            letters = keep[0][4:]
            cbs = [
                c
                for c in cbs
                if {CB_W: "w", CB_X: "x", CB_OUT: "o"}.get(c.format_descriptors[0].buffer_index) in letters
            ]
        if "nosem" in abl:
            return fp.program_descriptor(kernels, cbs=cbs, semaphores=[])
    return fp.program_descriptor(
        kernels,
        cbs=cbs,
        semaphores=[fp.semaphore_descriptor(SEM_X, sem_cores, 0), fp.semaphore_descriptor(SEM_HUB, sem_cores, 0)],
    )


def mm_stream2(act, weight, *, k: int, n: int, nb: int, blk: int, cols: int, slots: int = 6, exclude=()):
    """``act @ weight`` (row 0) as a DRAM TILE tensor ``[1, 1, 32, ceil(n / 32) * 32]`` bf16 (the other rows are not written)."""
    out = fp.allocate((1, 1, fp.TILE, math.ceil(n / fp.TILE) * fp.TILE), ttnn.bfloat16, ttnn.TILE_LAYOUT, act.device())
    fp.run_program(
        [act, weight, out],
        mm_stream2_program(act, weight, out, k=k, n=n, nb=nb, blk=blk, cols=cols, slots=slots, exclude=exclude),
    )
    return out


# ----------------------------------------------------------------------------------------- the model's entry (QWEN38_MM_STREAM2=1)

import atexit

# (K, N) -> (columns per core, K rows per block, placement): the decode linears this program serves (tuned by the x60 sweep)
SHAPES = {
    (2560, 2112): (4, 10, "rect"),
    (2560, 2304): (3, 10, "rect"),
    (2560, 512): (1, 10, "rect"),
    (2560, 352): (1, 16, "rect"),
}
CALLS: dict = {}
STOCK_CALLS: dict = {}
_OUT: dict = {}  # (mesh, K, N, dtype) -> (shape, memory config, topology) of the stock op's result
_RELAID: dict = {}  # id(weight) -> (weight, re-laid mesh tensor)


def _report_calls() -> None:
    if CALLS or STOCK_CALLS:
        print(
            f"MM_STREAM2_CALLS total={sum(CALLS.values())} {dict(sorted(CALLS.items()))} stock_first={dict(sorted(STOCK_CALLS.items()))}",
            flush=True,
        )


atexit.register(_report_calls)


def admits(act, weight, memory_config) -> bool:
    if not enabled() or memory_config != ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG:
        return False
    if len(act.shape) != 4 or int(act.shape[-2]) != 1 or not act.memory_config().is_sharded():
        return False
    if weight.dtype not in (ttnn.bfloat8_b, ttnn.bfloat16) or len(weight.shape) != 4:
        return False
    return (int(weight.shape[-2]), int(weight.shape[-1])) in SHAPES and int(act.shape[-1]) == int(weight.shape[-2])


def relaid_weight(mesh, weight, k, n, cols):
    """The weight re-laid for ``mm_stream2`` (built once per weight tensor, at its first, eager call; per device: the shards differ)."""
    hit = _RELAID.get(id(weight))
    if hit is not None and hit[0] is weight:
        return hit[1]
    nb = weight.memory_config().shard_spec.shape[1] // TILE
    kt = k // TILE
    parts = []
    for t in ttnn.get_device_tensors(weight):
        w = ttnn.to_torch(t).reshape(-1, t.shape[-1])[:k, :n].to(torch.bfloat16)
        parts.append(relayout(w, nb, cols))
    stacked = torch.stack(parts).unsqueeze(0) if len(parts) > 1 else parts[0].reshape(1, 1, *parts[0].shape)
    if len(parts) > 1:
        mapper = ttnn.ShardTensor2dMesh(mesh, tuple(mesh.shape), dims=(None, 1))
        new = ttnn.from_torch(
            stacked,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight.dtype,
            mesh_mapper=mapper,
            memory_config=weight_memory_config(BANKS, kt * nb),
        )
    else:
        new = ttnn.from_torch(
            stacked,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight.dtype,
            memory_config=weight_memory_config(BANKS, kt * nb),
        )
    _RELAID[id(weight)] = (weight, new)
    return new


def linear(act, weight, *, memory_config, program_config, compute_kernel_config, **rest):
    """``ttnn.linear`` (same arguments); with QWEN38_MM_STREAM2=1 and an admitted call the re-laid streamed program instead (row 0 of the
    result is defined, the stock op's shape / L1 width-sharded memory config / topology; learned from the shape's first, stock call).
    """
    stock = lambda: ttnn.linear(  # noqa: E731
        act,
        weight,
        memory_config=memory_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        **rest,
    )
    if not admits(act, weight, memory_config):
        return stock()
    mesh = act.device()
    k, n = int(weight.shape[-2]), int(weight.shape[-1])
    key = (id(mesh), k, n, weight.dtype)
    if key not in _OUT:
        relaid_weight(mesh, weight, k, n, SHAPES[(k, n)][0])
        result = stock()
        _OUT[key] = (tuple(int(v) for v in result.shape), result.memory_config(), result.tensor_topology())
        STOCK_CALLS[(k, n)] = STOCK_CALLS.get((k, n), 0) + 1
        return result
    hit = _RELAID.get(id(weight))
    if (
        hit is None or hit[0] is not weight
    ):  # first call of THIS weight: eager warmup builds its re-laid copy and runs the stock op
        cols, blk, place = SHAPES[(k, n)]
        relaid_weight(mesh, weight, k, n, cols)
        return stock()
    shape, out_cfg, topology = _OUT[key]
    from .. import moe_experts

    cols, blk, place = SHAPES[(k, n)]
    nb = weight.memory_config().shard_spec.shape[1] // TILE
    out = fp.allocate(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, out_cfg)
    exclude = moe_experts.model_exclude(mesh)
    fp.run_program(
        [act, hit[1], out],
        mm_stream2_program(act, hit[1], out, k=k, n=n, nb=nb, blk=blk, cols=cols, exclude=exclude, place=place),
    )
    out.update_tensor_topology(topology)
    CALLS[(k, n)] = CALLS.get((k, n), 0) + 1
    return out
