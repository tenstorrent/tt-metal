# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``mm_stream`` (QWEN38_MM_STREAM=1, default off): a decode DRAM-sharded linear (one live row, K x N, today's cached weight
layout) on many more compute cores than the stock program's one or two per bank.

The stock ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` computes on the reader cores of the banks (8 or 16
cores): per output tile the same K-ordered fp32-DEST matmuls (~25 ns per HiFi2 tile, 49 ns HiFi4), plus the spill / reload
of its fp32 partials at every K block.  Here each core owns ``cols`` adjacent output tile columns of one bank's width
shard (the bank shard is row-major tiles, so the tiles of one K row of a core are one contiguous read), streams them in K
order, accumulates ALL K in one fp32 DEST (no spill: the stock's spill is a lossless fp32 round trip) and packs bf16 as the
stock does.  The activation (the gathered hidden, L1 width shards) holds one live row: each shard core compacts row 0 of its
tiles (a 1 x 32 tile = 64 B: face rows 0 of both faces), writes it to ONE hub core, which multicasts the 80 compact tiles
(5 KB instead of 160 KB) to the compute rectangle; the matmul reads them as 1 x 32 tiles and writes 1 x 32 output tiles,
whose two 32-byte halves the writer puts at bytes 0 and 512 of page = global output tile of a DRAM TILE tensor (row 0 only).
"""

from __future__ import annotations

import math
import os

import ttnn

from .. import placement
from .. import program as fp

NAME = "mm_stream"
FLAG = "QWEN38_MM_STREAM"
KERNELS = {
    n: fp.kernel_source(NAME, f"{n}_mm_stream.cpp") for n in ("hub", "shard", "reader", "writer", "compute", "idle")
}
CB_X, CB_W, CB_OUT, CB_STAGE = 0, 1, 2, 3
SEM_X, SEM_HUB = 0, 1
FID = {
    ttnn.bfloat8_b: ttnn.MathFidelity.HiFi2,
    ttnn.bfloat16: ttnn.MathFidelity.HiFi4,
    ttnn.bfloat4_b: ttnn.MathFidelity.LoFi,
}
RECT_SHAPES = (
    (6, 6),
    (6, 5),
    (8, 4),
    (4, 8),
    (7, 5),
    (5, 7),
    (8, 5),
    (6, 4),
    (4, 6),
    (4, 4),
    (8, 2),
    (5, 4),
    (3, 8),
    (9, 4),
    (8, 8),
    (10, 5),
    (9, 8),
    (8, 9),
    (10, 8),
    (9, 9),
    (10, 9),
    (10, 10),
)
_WORKERS: dict = {}


def enabled() -> bool:
    return os.environ.get(FLAG, "0") == "1"


def column_groups(n: int, nb: int, banks: int, cols: int):
    """``(bank, first column in the bank, ncols, first global tile)`` of every core: the real tile columns of each bank's
    shard (the padding tiles of the last banks are not computed) in groups of ``cols``."""
    total = math.ceil(n / fp.TILE)
    groups = []
    for b in range(banks):
        real = max(0, min(nb, total - b * nb))
        for c0 in range(0, real, cols):
            groups.append((b, c0, min(cols, real - c0), b * nb + c0))
    return groups


def shard_banks(weights) -> list[int]:
    """The DRAM bank id of each shard, in shard order: the cores of the shard grid, range by range (a cached ring-order tensor
    has one CoreRange per bank, a freshly sharded one a single CoreRange (0, 0)-(banks - 1, 0): both must give every bank).
    """
    if os.environ.get("X52_OLDBANKS"):
        return [int(r.start.x) for r in weights.memory_config().shard_spec.grid.ranges()]
    return [
        int(c[0])
        for r in weights.memory_config().shard_spec.grid.ranges()
        for c in placement.rectangle_cores((r.start.x, r.start.y, r.end.x, r.end.y))
    ]


def bank_workers(mesh) -> dict:
    """DRAM bank -> (x, y) logical worker of the reference die."""
    key = id(mesh)
    if key not in _WORKERS:
        a = mesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, ttnn.MeshCoordinate(0, 0))
        _WORKERS[key] = {b: (a[b].x, a[b].y) for b in a}
    return _WORKERS[key]


_PLANS: dict = {}


def _assign(rect_cores, groups, banks, workers):
    """Greedy group -> core assignment (groups in bank order, each the nearest free core to its bank's reader worker) and its
    total Manhattan distance."""
    free, assign, cost = list(rect_cores), {}, 0
    for i in sorted(range(len(groups)), key=lambda i: groups[i][0]):
        w = workers[banks[groups[i][0]]]
        best = min(free, key=lambda c: abs(c[0] - w[0]) + abs(c[1] - w[1]))
        cost += abs(best[0] - w[0]) + abs(best[1] - w[1])
        free.remove(best)
        assign[i] = best
    return assign, cost


def _place(mesh, groups, banks, act_cores, taken_extra):
    """The compute rectangle (the first free one of RECT_SHAPES holding the groups, scanned from the grid's far corner: x1
    measured it faster than the one nearest the banks), the hub core and the group -> core assignment."""
    count = len(groups)
    taken = set(taken_extra) | set(act_cores) | set(placement.reserved_cores(mesh))
    grid = mesh.compute_with_storage_grid_size()
    rect = None
    for w, h in RECT_SHAPES:
        if w * h < count:
            continue
        try:
            rect = placement.free_rectangle_in(grid.x, grid.y, taken, w, h)
            break
        except RuntimeError:
            continue
    if rect is None:
        raise RuntimeError(f"no free rectangle for {count} cores")
    rect_cores = placement.rectangle_cores(rect)
    hub_rect = placement.free_rectangle_in(grid.x, grid.y, taken | set(rect_cores), 1, 1)
    assign, _ = _assign(rect_cores, groups, banks, bank_workers(mesh))
    return rect, rect_cores, ttnn.CoreCoord(hub_rect[0], hub_rect[1]), assign


def mm_stream_program(act, weight, out, *, k: int, n: int, blk: int, cols: int = 2, slots: int = 6, exclude=()):
    mesh = out.device()
    banks = shard_banks(weight)
    nb = weight.memory_config().shard_spec.shape[1] // fp.TILE
    kt = k // fp.TILE
    if kt % blk or cols > 4:
        raise ValueError("blk must divide the K tiles and cols <= 4")
    wdtype = weight.dtype
    tile_bytes = fp.TILE_BYTES[wdtype]
    if (len(banks) != 8 or len(set(banks)) != 8) and not os.environ.get("X52_OLDBANKS"):
        raise ValueError(f"mm_stream needs the 8 DRAM banks' shards, got {banks}")
    groups = column_groups(n, nb, len(banks), cols)
    count = len(groups)
    act_cores = [
        c
        for r in act.memory_config().shard_spec.grid.ranges()
        for c in placement.rectangle_cores((r.start.x, r.start.y, r.end.x, r.end.y))
    ]
    shards = len(act_cores)
    shard_tiles = kt // shards
    out_cores = set()
    if out.memory_config().is_sharded():
        out_cores = {
            c
            for r in out.memory_config().shard_spec.grid.ranges()
            for c in placement.rectangle_cores((r.start.x, r.start.y, r.end.x, r.end.y))
        }
    key = (id(mesh), k, n, cols, tuple(sorted(map(tuple, exclude))), tuple(act_cores), tuple(sorted(out_cores)))
    if key not in _PLANS:
        _PLANS[key] = _place(
            mesh,
            groups,
            banks,
            act_cores,
            set(map(tuple, exclude)) | (set() if os.environ.get("X52_NOOUTEX") else out_cores),
        )
    rect, rect_cores, hub_cc, assign = _PLANS[key]
    live = set(map(tuple, exclude)) | (set() if os.environ.get("X52_NOOUTEX") else out_cores) | set(act_cores)
    cores_used = {tuple(c) for c in rect_cores} | {(hub_cc.x, hub_cc.y)}
    if cores_used & live:  # asserted at every build: the program never sits on a live L1 tensor's cores
        raise RuntimeError(f"mm_stream cores overlap live cores: {sorted(cores_used & live)}")
    active_set = fp.core_set([fp.CoreWork(ttnn.CoreCoord(*assign[i]), 0, 0) for i in range(count)])
    rect_range = placement.core_range(rect)
    hub_set = ttnn.CoreRangeSet([ttnn.CoreRange(hub_cc, hub_cc)])
    act_ccs = [ttnn.CoreCoord(*c) for c in act_cores]
    shard_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in act_ccs])
    x_cores = ttnn.CoreRangeSet([rect_range, ttnn.CoreRange(hub_cc, hub_cc)])
    stage_cores = ttnn.CoreRangeSet([ttnn.CoreRange(hub_cc, hub_cc), *[ttnn.CoreRange(c, c) for c in act_ccs]])
    sem_cores = ttnn.CoreRangeSet(
        [rect_range, ttnn.CoreRange(hub_cc, hub_cc), *[ttnn.CoreRange(c, c) for c in act_ccs]]
    )
    xy0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(rect[0], rect[1]))
    xy1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(rect[2], rect[3]))
    hub_xy = mesh.worker_core_from_logical_core(hub_cc)
    out_l1 = out.memory_config().is_sharded()
    out_shard_cores, shard_out_tiles = [], 1
    if out_l1:
        spec = out.memory_config().shard_spec
        if spec.shape[0] != fp.TILE or str(spec.orientation) != "ShardOrientation.ROW_MAJOR":
            raise ValueError("mm_stream writes L1 width shards of one row tile, row-major")
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
        ("cb_stage", CB_STAGE),
        ("sem_x", SEM_X),
        ("sem_hub", SEM_HUB),
        ("kt", kt),
        ("blk", blk),
        ("slots", slots),
        ("bank_tiles", nb),
        ("w_tile_bytes", tile_bytes),
        ("shards", shards),
        ("shard_tiles", shard_tiles),
    ]
    shard_xy = []
    for (
        c
    ) in (
        out_shard_cores
    ):  # the output shard cores' NoC coordinates (x, y per shard): per-core runtime args after the first three
        p_ = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        shard_xy += [p_.x, p_.y]
    reader_args, compute_args, writer_args = [], [], []
    for i, (b, c0, ncols, first) in enumerate(groups):
        core = ttnn.CoreCoord(*assign[i])
        reader_args.append((core, [weight.buffer_address(), banks[b], c0, ncols, (b + i) & 3]))
        compute_args.append((core, [ncols]))
        writer_args.append((core, [out.buffer_address(), first, ncols, hub_xy.x, hub_xy.y, *shard_xy]))
    tiny = ttnn.TileDescriptor(1, fp.TILE)

    def tiny_cb(index, pages, cores_):  # 1 x 32 bf16 tiles: the two 16-value faces of one row = 64 B
        return ttnn.CBDescriptor(
            total_size=pages * 64,
            core_ranges=cores_,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=64, tile=tiny)
            ],
        )

    cbs = [
        tiny_cb(CB_X, kt, x_cores),
        *[
            # the weight CB ring must hold exactly `slots` blocks of THIS core's ncols (the reader wraps its slots at
            # slots x blk x ncols pages, the CB pointer at the CB's size): one CB per distinct ncols (x52 root cause of the
            # wrong columns of the last, narrower core of a bank: a CB sized for the widest core)
            fp.cb_descriptor(
                CB_W,
                wdtype,
                tile_bytes,
                slots * blk * nc,
                fp.core_set(
                    [fp.CoreWork(ttnn.CoreCoord(*assign[i]), 0, 0) for i in range(count) if groups[i][2] == nc]
                ),
            )
            for nc in sorted({g[2] for g in groups})
        ],
        tiny_cb(CB_OUT, cols, active_set),
        fp.cb_descriptor(
            CB_STAGE, ttnn.bfloat16, 64, kt, stage_cores
        ),  # the compact activation: at one address on the hub and the shard cores
    ]
    hub = fp.reader_kernel(
        KERNELS["hub"], hub_set, [], [(hub_cc, [xy0.x, xy0.y, xy1.x, xy1.y, len(rect_cores)])], named=named
    )
    shard = fp.reader_kernel(
        KERNELS["shard"],
        shard_set,
        [],
        [(c, [act.buffer_address(), hub_xy.x, hub_xy.y, i]) for i, c in enumerate(act_ccs)],
        named=named,
    )
    reader = fp.reader_kernel(KERNELS["reader"], active_set, [], reader_args, named=named)
    writer = fp.writer_kernel(
        KERNELS["writer"],
        active_set,
        fp.accessor_args(out),
        writer_args,
        named=named,
        defines=([("X52_SKIP_WRITES", "1")] if os.environ.get("X52_SKIP_WRITES") else [])
        + ([("X52_DPRINT", "1")] if os.environ.get("X52_DPRINT") else []),
    )
    compute = fp.compute_kernel(
        KERNELS["compute"],
        active_set,
        [],
        compute_args,
        named=named,
        fidelity=FID[wdtype],
        fp32_dest=True,
        approx=False,
        dst_full_sync=False,
    )
    kernels = [hub, shard, reader, writer, compute]
    idle_cores = [c for c in rect_cores if tuple(c) not in {tuple(assign[i]) for i in range(count)}]
    if (
        idle_cores
    ):  # they receive the hub's semaphore multicast: consume and clear it (else the next program finds it set)
        kernels.append(
            fp.writer_kernel(
                KERNELS["idle"],
                fp.core_set([fp.CoreWork(ttnn.CoreCoord(*c), 0, 0) for c in idle_cores]),
                [],
                [(ttnn.CoreCoord(*c), [hub_xy.x, hub_xy.y]) for c in idle_cores],
                named=named,
            )
        )
    return fp.program_descriptor(
        kernels,
        cbs=cbs,
        semaphores=[fp.semaphore_descriptor(SEM_X, sem_cores, 0), fp.semaphore_descriptor(SEM_HUB, sem_cores, 0)],
    )


def mm_stream(act, weight, *, k: int, n: int, blk: int, cols: int = 2, slots: int = 6, exclude=()):
    """``act @ weight`` for the live row (row 0) of one row tile; returns the DRAM TILE tensor ``[1, 1, 32, ceil(n / 32) * 32]``
    bf16 whose row 0 is the product (the other rows are not written)."""
    mesh = act.device()
    out = fp.allocate((1, 1, fp.TILE, math.ceil(n / fp.TILE) * fp.TILE), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh)
    fp.run_program(
        [act, weight, out],
        mm_stream_program(act, weight, out, k=k, n=n, blk=blk, cols=cols, slots=slots, exclude=exclude),
    )
    return out


# ---------------------------------------------------------------- the model's entry: a drop-in for the decode ttnn.linear

import atexit

# (K, N) -> (columns per core, K block): the shapes whose decode linear this program serves (x52 prototype: bitwise vs the stock)
SHAPES = {(2560, 2112): (2, 10), (2560, 2304): (2, 10), (2560, 512): (1, 10), (2560, 352): (2, 10)}
CALLS: dict = {}  # (K, N) -> launches of the streamed program (program builds, eager or captured), for the A/B logs
STOCK_CALLS: dict = (
    {}
)  # (K, N) -> admitted shapes that ran the stock op (the first call of a shape learns the output layout)
_OUT: dict = {}  # (mesh, K, N, weight dtype) -> (output shape, output memory config, topology) of the stock op's result


def _report_calls() -> None:
    if CALLS or STOCK_CALLS:
        print(
            f"MM_STREAM_CALLS total={sum(CALLS.values())} {dict(sorted(CALLS.items()))} stock_first={dict(sorted(STOCK_CALLS.items()))}",
            flush=True,
        )


atexit.register(_report_calls)


def admits(act, weight, memory_config) -> bool:
    """QWEN38_MM_STREAM=1, one live row (decode rows == 1: the compact activation carries row 0 only), a shape of ``SHAPES``,
    the activation an L1 width shard (the gathered hidden), BF8 / BF16 weights in today's DRAM-sharded layout."""
    if not enabled() or memory_config != ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG:
        return False
    if len(act.shape) != 4 or int(act.shape[-2]) != 1 or not act.memory_config().is_sharded():
        return False
    if weight.dtype not in (ttnn.bfloat8_b, ttnn.bfloat16) or len(weight.shape) != 4:
        return False
    return (int(weight.shape[-2]), int(weight.shape[-1])) in SHAPES and int(act.shape[-1]) == int(weight.shape[-2])


def linear(act, weight, *, memory_config, program_config, compute_kernel_config, **rest):
    """``ttnn.linear`` (same arguments) for the decode linears; with QWEN38_MM_STREAM=1 and an admitted call the streamed
    program instead.  The result has EXACTLY the stock op's shape, layout and L1 width-sharded memory config (learned from
    the shape's first, stock call -- eager warmup, before any capture) and mesh topology; only ROW 0 of its tile is
    written (decode rows == 1: every consumer reads row 0; the other rows of the padded tile are not defined)."""
    stock = lambda: ttnn.linear(  # noqa: E731
        act,
        weight,
        memory_config=memory_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        **rest,
    )
    from .. import mm_stream2  # x60: QWEN38_MM_STREAM2=1 re-laid weights (inert otherwise)

    if mm_stream2.enabled():
        return mm_stream2.linear(
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
        result = stock()
        _OUT[key] = (tuple(int(v) for v in result.shape), result.memory_config(), result.tensor_topology())
        STOCK_CALLS[(k, n)] = STOCK_CALLS.get((k, n), 0) + 1
        return result
    shape, out_cfg, topology = _OUT[key]
    from .. import moe_experts

    cols, blk = SHAPES[(k, n)]
    out = fp.allocate(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, out_cfg)
    exclude = frozenset() if os.environ.get("X52_NOEXCL") else moe_experts.model_exclude(mesh)
    fp.run_program(
        [act, weight, out], mm_stream_program(act, weight, out, k=k, n=n, blk=blk, cols=cols, exclude=exclude)
    )
    out.update_tensor_topology(topology)
    CALLS[(k, n)] = CALLS.get((k, n), 0) + 1
    return out
