# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lane split of the GDN step (``QWEN38_GDN_LANE_SPLIT=1``): the V-split conv fan-out for lane rows 8, 16 and 32.

The conv cores of ``conv_split_v`` (one core per 32-row x 32-channel conv tile, all lanes at once) send each q/k/v tile to
the head cores.  At rows 1 / 2 / 4 a head core is one (lane, head, s) item, which needs ``rows * 6 * VB`` head cores.  Here
a head core serves ``G`` lanes of ONE (head, s) block, one after the other: the q/k/v tiles of the block hold every lane's
row, so they are sent once per lane GROUP (not once per lane), and the lane mask tile selects the lane's row of the delta.
Lane group ``j`` = lanes ``[j*G, min((j+1)*G, rows))`` for every (head, s) (the VB peers of a head process the same lane at
the same time: the o exchange of ``writer_vb.cpp`` needs it); head core (head, s, j) is pool core ``(head*VB + s)*NG + j``,
NG = ceil(rows / G), head cores = 6 * VB * NG.  The pool is every grid core outside the 40-core conv rectangle (x 6-9,
y 0-9) in column-major order (x 0-5, then x >= 10), so 70 cores on the 11 x 10 grid.  Option A of plan_nov05/g1_gdnlane:
the conv cores take no items.

Kernels (``kernels_ls/``): ``conv_reader_ls.cpp`` (the conv reader with row-limited reads: dead rows zero-filled, the newest
ring slot written for the live rows only), ``reader_ls.cpp`` and ``compute_ls.cpp`` (the V-split head kernels with an item
loop); ``conv_writer.cpp``, ``conv_compute.cpp`` and ``writer_vb.cpp`` are the stage-2 files as they are (the writer
already loops over items).  The lane mask table here has 32 tiles (``conv_split_v`` has 4).  With the flag off nothing
here is imported (``gdn_step.run``); at other rows ``admits`` is False and the default programs run.  Bit-identical to the
default program on the live rows (tests/test_fused_gdn_step_ls_device.py).

Env: ``QWEN38_GDN_LANE_SPLIT`` (0), ``QWEN38_GDN_LS_VB`` (1; 2 and 4 are untested, parametrized in the device test),
``QWEN38_GDN_LS_G`` (auto = the smallest G whose head cores fit the pool), ``QWEN38_GDN_LS_HOIST`` (0; 1 = the q/k l2
norms run once per core), ``QWEN38_GDN_LS_CONVHEAD`` (0; option B: 1 = the conv cores take head items after their conv
phase, the head pool grows from 70 to 110 cores; see ``run_convhead``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import ttnn

from .. import gr_read
from .. import program as fp
from .conv_split import CONV_ROWS, CONV_TILES, SEM_QK, SEM_V
from .conv_split_v import CONV_COLUMNS, CONV_X0, SEM_X, conv_core, scaler_tile_words

ENV = "QWEN38_GDN_LANE_SPLIT"
VB_ENV, G_ENV, HOIST_ENV = "QWEN38_GDN_LS_VB", "QWEN38_GDN_LS_G", "QWEN38_GDN_LS_HOIST"
CONVHEAD_ENV = "QWEN38_GDN_LS_CONVHEAD"
ROWS_SUPPORTED = (8, 16, 32)
VB_SUPPORTED = (1, 2, 4)
KERNEL_DIR = "kernels_ls"
CB_QKV, CB_OBF, CB_OMINE, CB_SQO = 11, 10, 29, 30
CB_CONVOUT, CB_CONVSUM = 28, 31  # option B: the conv result and sum CBs of a conv core that also takes head items
CONV_ONLY_CBS = (0, 1, 2, 11)
HEAD_COLUMNS = 6
LANE_TILES = 32  # rows of a tile: the mask table has one tile per lane


def enabled() -> bool:
    return os.environ.get(ENV, "1") == "1"


def hoist_enabled() -> bool:
    return os.environ.get(HOIST_ENV, "1") == "1"


def convhead_enabled() -> bool:
    return os.environ.get(CONVHEAD_ENV, "1") == "1"


def kernel(file: str) -> str:
    """``ttnn/fused/gdn_step/kernels_ls/<file>`` relative to the repository root (a directory the kernel census
    ``*/kernels/*.cpp`` does not scan); the file must exist."""

    relative = f"{fp.KERNEL_ROOT}/gdn_step/{KERNEL_DIR}/{file}"
    if not (fp.REPO_ROOT / relative).is_file():
        raise FileNotFoundError(f"lane-split kernel source {relative} is absent under {fp.REPO_ROOT}")
    return relative


@dataclass(frozen=True)
class Plan:
    rows: int
    vb: int
    g: int  # lanes per head core (the largest group)
    ng: int  # lane groups
    hoist: bool
    host_cores: int = 0  # conv cores that also run a head block (option B); 0 = option A

    @property
    def vt(self) -> int:
        return 4 // self.vb

    @property
    def st(self) -> int:
        return 4 * self.vt

    @property
    def head_cores(self) -> int:
        return HEAD_COLUMNS * self.vb * self.ng

    def group(self, j: int) -> range:
        return range(j * self.g, min((j + 1) * self.g, self.rows))

    @property
    def variant(self) -> str:
        """The program meta variant: one name per setting (rows is a meta field of its own)."""

        base = f"step_lane_split_vb{self.vb}_g{self.g}_h{int(self.hoist)}"
        return base + (f"_ch{self.host_cores}" if self.host_cores else "")


def head_pool(grid_x: int, grid_y: int) -> list[tuple[int, int]]:
    """Every grid core outside the conv rectangle, column-major."""

    return [
        (x, y)
        for x in range(grid_x)
        for y in range(grid_y)
        if not (CONV_X0 <= x < CONV_X0 + CONV_COLUMNS and y < CONV_ROWS)
    ]


def extended_pool(grid_x: int, grid_y: int) -> list[tuple[int, int]]:
    """Option B pool: the plain pool, then the conv cores, the highest conv tile first (the v tiles have the fewest
    destinations, so their cores finish the conv phase first; the 26 cores of rows 16 / 32 are then x 8-9 and x 7 y 4-9).
    """

    return [*head_pool(grid_x, grid_y), *(conv_core(t) for t in reversed(range(CONV_TILES)))]


def plan_for(
    rows: int, grid_x: int, grid_y: int, vb: int | None = None, g: int | str | None = None, hoist=None, convhead=None
):
    """The plan for ``rows`` on a grid_x x grid_y grid, or None when it does not fit (rows not supported, a grid without
    the conv rectangle, or no G with 6 * VB * ceil(rows / G) <= pool cores).  ``vb``, ``g`` (an int or "auto") and
    ``hoist`` default to the environment."""

    vb = int(os.environ.get(VB_ENV, "1")) if vb is None else int(vb)
    g = os.environ.get(G_ENV, "auto") if g is None else g
    hoist = hoist_enabled() if hoist is None else bool(hoist)
    convhead = convhead_enabled() if convhead is None else bool(convhead)
    if rows not in ROWS_SUPPORTED or vb not in VB_SUPPORTED:
        return None
    if grid_y < CONV_ROWS or grid_x < CONV_X0 + CONV_COLUMNS:
        return None
    plain = len(head_pool(grid_x, grid_y))
    pool = plain + (CONV_TILES if convhead else 0)  # option B: the conv cores take head items too
    if str(g) == "auto":
        for candidate in range(1, rows + 1):
            ng = -(-rows // candidate)
            if HEAD_COLUMNS * vb * ng <= pool:
                return Plan(rows, vb, candidate, ng, hoist, max(0, HEAD_COLUMNS * vb * ng - plain))
        return None
    g = int(g)
    if not 1 <= g <= rows:
        return None
    ng = -(-rows // g)
    return (
        Plan(rows, vb, g, ng, hoist, max(0, HEAD_COLUMNS * vb * ng - plain)) if HEAD_COLUMNS * vb * ng <= pool else None
    )


def admits(rows: int, mesh) -> bool:
    if not enabled():
        return False
    grid = mesh.compute_with_storage_grid_size()
    return plan_for(rows, grid.x, grid.y) is not None


def head_core(plan: Plan, head: int, s: int, j: int, pool) -> tuple[int, int]:
    return pool[(head * plan.vb + s) * plan.ng + j]


def destinations(tile: int, plan: Plan, *, k_tile0: int, v_tile0: int, qk_repeat: int):
    """[((head, s, j), slot, sem)] the head cores that receive conv tile ``tile``: a q/k tile goes to every (head of the
    key head, s, group) core (slot c / 4 + c), a v tile (h, cc) to block cc // VT of head h in every group (slot 8 + cc % VT).
    """

    vt = plan.vt
    if tile < v_tile0:
        g, slot = (tile // 4, tile % 4) if tile < k_tile0 else ((tile - k_tile0) // 4, 4 + (tile - k_tile0) % 4)
        return [
            ((g * qk_repeat + r, s, j), slot, SEM_QK)
            for r in range(qk_repeat)
            for s in range(plan.vb)
            for j in range(plan.ng)
        ]
    head, cc = (tile - v_tile0) // 4, (tile - v_tile0) % 4
    return [((head, cc // vt, j), 8 + cc % vt, SEM_V) for j in range(plan.ng)]


def mask_tile_count() -> int:
    return LANE_TILES


def _range(x0, y0, x1, y1) -> ttnn.CoreRange:
    return ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))


def range_spans(cores) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) rectangles covering exactly ``cores``: per-column runs of consecutive y, adjacent columns with the
    same run stacked into one rectangle."""

    columns: dict[int, list[list[int]]] = {}
    for x, y in sorted(set(cores)):
        runs = columns.setdefault(x, [])
        if runs and runs[-1][1] + 1 == y:
            runs[-1][1] = y
        else:
            runs.append([y, y])
    spans: list[list[int]] = []
    for x in sorted(columns):
        for y0, y1 in columns[x]:
            for span in spans:
                if span[2] == x - 1 and span[1] == y0 and span[3] == y1:
                    span[2] = x
                    break
            else:
                spans.append([x, y0, x, y1])
    return [(a, b, c, d) for a, b, c, d in spans]


def _constants(mesh):
    """(scalers, masks) DRAM tensors replicated on the mesh, once per mesh: ``scalers`` [2, 32, 32] fp32 (tile 0 = 1.0, tile
    1 = 1/128), ``masks`` [32, 32, 32] bf16 (tile ``lane`` has ones in row ``lane``)."""

    cached = getattr(mesh, "_qwen38_gdn_ls_constants", None)
    if cached is not None:
        return cached
    import torch

    scalers = torch.tensor([w[0] for w in scaler_tile_words()], dtype=torch.int32).view(torch.float32)
    scalers = scalers.view(2, 1, 1).expand(2, fp.TILE, fp.TILE).contiguous()
    masks = torch.zeros(LANE_TILES, fp.TILE, fp.TILE, dtype=torch.float32)
    for lane in range(LANE_TILES):
        masks[lane, lane, :] = 1.0
    made = []
    for host, dtype in ((scalers, ttnn.float32), (masks, ttnn.bfloat16)):
        made.append(
            ttnn.from_torch(
                host,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        )
    mesh._qwen38_gdn_ls_constants = tuple(made)
    return mesh._qwen38_gdn_ls_constants


def run_lane_split(m, projected, older, newest, taps, constants, norm, recurrent, out):
    """The lane-split program; ``m`` is the ``gdn_step`` module."""

    rows = fp.rows_of(projected)
    mesh = projected.device()
    grid = mesh.compute_with_storage_grid_size()
    plan = plan_for(rows, grid.x, grid.y)
    if plan is None:
        raise ValueError(f"gdn_step lane split does not fit rows={rows} on the {grid.x}x{grid.y} grid")
    if plan.host_cores:  # option B (QWEN38_GDN_LS_CONVHEAD=1 and the head blocks do not fit the plain pool)
        return run_convhead(m, plan, projected, older, newest, taps, constants, norm, recurrent, out)
    vb, vt, st, ng = plan.vb, plan.vt, plan.st, plan.ng
    HEADS, QK_REPEAT = m.HEADS, m.QK_REPEAT
    BF16, FP32 = m.BF16, m.FP32
    pool = head_pool(grid.x, grid.y)
    head_cores = [pool[i] for i in range(plan.head_cores)]
    assert len(set(head_cores)) == plan.head_cores
    spans = range_spans(head_cores)
    head_set = ttnn.CoreRangeSet([_range(*s) for s in spans])
    conv_range = _range(CONV_X0, 0, CONV_X0 + CONV_COLUMNS - 1, CONV_ROWS - 1)
    conv_set = ttnn.CoreRangeSet([conv_range])
    union = ttnn.CoreRangeSet([*(_range(*s) for s in spans), conv_range])
    noc = gr_read.noc_map(mesh)

    def core_of(head, s, j):
        return head_core(plan, head, s, j, pool)

    blocks = [(head, s, j) for head in range(HEADS) for s in range(vb) for j in range(ng)]
    assert len(blocks) == plan.head_cores
    max_items = max(len(plan.group(j)) for j in range(ng))

    reader_cta = [rows, HEADS, m.K_TILE0, m.V_TILE0, m.Z_TILE0, m.A_TILE, m.B_TILE, vt]
    for tensor in (projected, constants, norm, recurrent):
        reader_cta.extend(fp.accessor_args(tensor))
    scalers, masks = _constants(mesh)
    for tensor in (scalers, masks):
        reader_cta.extend(fp.accessor_args(tensor))
    reader_addrs = [
        projected.buffer_address(),
        constants.buffer_address(),
        norm.buffer_address(),
        recurrent.buffer_address(),
    ]
    writer_cta = [rows, HEADS, vt, vb, *fp.accessor_args(recurrent), *fp.accessor_args(out)]
    writer_addrs = [recurrent.buffer_address(), out.buffer_address()]
    defines = [("INP_FLOAT32", "1")]

    # state: two items deep when a core has several (the next lane's read overlaps this lane's compute); decayed state
    # (two copies): this block's tiles only
    resized = {7: st * (2 if max_items > 1 else 1), 20: st, 21: st}
    cbs = [
        fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], resized.get(index, pages), head_set)
        for index, dtype, pages in m.CBS
        if index not in CONV_ONLY_CBS
    ]
    cbs.append(
        ttnn.CBDescriptor(
            total_size=st * fp.TILE_BYTES[FP32],
            core_ranges=head_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=m.CB_SNEW, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
                ttnn.CBFormatDescriptor(buffer_index=m.CB_OUTS, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
            ],
        )
    )
    cbs.append(fp.cb_descriptor(CB_OMINE, BF16, fp.TILE_BYTES[BF16], vt, head_set))
    cbs.append(fp.cb_descriptor(CB_SQO, FP32, fp.TILE_BYTES[FP32], 4, head_set))
    cbs.append(fp.cb_descriptor(CB_QKV, BF16, fp.TILE_BYTES[BF16], 12, union))
    for index, pages in ((0, 1), (1, 3), (2, 4), (10, 1)):
        cbs.append(fp.cb_descriptor(index, BF16, fp.TILE_BYTES[BF16], pages, conv_set))

    def core(xy):
        return ttnn.CoreCoord(*xy)

    def peers(head, j):
        return [v for s in range(vb) for v in noc[core_of(head, s, j)]]

    reader = fp.reader_kernel(
        kernel("reader_ls.cpp"),
        head_set,
        reader_cta,
        [
            (
                core(core_of(h, s, j)),
                [
                    *reader_addrs,
                    len(plan.group(j)),
                    plan.group(j)[0],
                    h,
                    s,
                    scalers.buffer_address(),
                    masks.buffer_address(),
                ],
            )
            for h, s, j in blocks
        ],
        defines=defines,
    )
    writer = fp.writer_kernel(
        fp.kernel_source(m.NAME, "writer_vb.cpp"),
        head_set,
        writer_cta,
        [
            (
                core(core_of(h, s, j)),
                [
                    *writer_addrs,
                    len(plan.group(j)),
                    *(v for lane in plan.group(j) for v in (lane, h, s, *peers(h, j))),
                ],
            )
            for h, s, j in blocks
        ],
        defines=defines,
    )
    compute = fp.compute_kernel(
        kernel("compute_ls.cpp"),
        head_set,
        [vt],
        [(core(core_of(h, s, j)), [len(plan.group(j)), s]) for h, s, j in blocks],
        defines=[*defines, *(("LS_HOIST_L2", "1"),) * plan.hoist],
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=True,
        unpack_to_dest_fp32=m.FP32_COPY_CBS,
    )

    conv_cta = [rows]
    for tensor in (projected, *older, *taps, newest):
        conv_cta.extend(fp.accessor_args(tensor))
    conv_addrs = [t.buffer_address() for t in (projected, *older, *taps, newest)]
    tiles = range(CONV_TILES)
    conv_xy = [conv_core(t) for t in tiles]
    conv_reader = fp.reader_kernel(
        kernel("conv_reader_ls.cpp"),
        conv_set,
        conv_cta,
        [(core(c), [*conv_addrs, t]) for t, c in zip(tiles, conv_xy)],
        defines=defines,
    )
    dest_args = {}
    for t in tiles:
        flat = []
        dests = destinations(t, plan, k_tile0=m.K_TILE0, v_tile0=m.V_TILE0, qk_repeat=QK_REPEAT)
        for (head, s, j), slot, sem in dests:
            x, y = noc[core_of(head, s, j)]
            flat.extend([x, y, slot, sem])
        dest_args[t] = [len(dests), *flat]
    conv_writer = fp.writer_kernel(
        fp.kernel_source(m.NAME, "conv_writer.cpp"),
        conv_set,
        [],
        [(core(c), dest_args[t]) for t, c in zip(tiles, conv_xy)],
        defines=defines,
    )
    conv_compute = fp.compute_kernel(
        fp.kernel_source(m.NAME, "conv_compute.cpp"),
        conv_set,
        [],
        [],
        defines=defines,
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=True,
        unpack_to_dest_fp32=m.FP32_COPY_CBS,
    )

    meta = fp.program_meta(
        m.NAME,
        plan.variant,
        rows,
        reads=(projected, *older, *taps, norm, recurrent),
        writes=(recurrent, newest, out),
        dram_bytes=rows * HEADS * 2 * 4 * 4 * fp.TILE_BYTES[FP32],
        flops=rows
        * (2 * 4 * m.QKV_WIDTH + 6 * 2 * m.QK_WIDTH + 8 * HEADS * m.HEAD_DIM * m.HEAD_DIM + 6 * m.VALUE_WIDTH),
        cores=plan.head_cores + CONV_TILES,
    )
    descriptor = fp.program_descriptor(
        [reader, writer, compute, conv_reader, conv_writer, conv_compute],
        cbs,
        semaphores=[fp.semaphore_descriptor(i, union) for i in (SEM_QK, SEM_V, SEM_X)],
    )
    fp.run_program(
        [projected, *older, *taps, constants, scalers, masks, norm, recurrent, newest, out], descriptor, meta=meta
    )
    return out


def run_convhead(m, plan, projected, older, newest, taps, constants, norm, recurrent, out):
    """Option B of plan_nov05/g1_gdnlane: the lane split program with the conv cores taking head blocks.

    Head blocks fill ``extended_pool``: the 70 plain pool cores (running reader_ls / writer_vb / compute_ls as option A),
    then ``plan.host_cores`` conv cores (highest conv tile first) running the combined kernels ``conv_head_*_ls.cpp``: the
    conv phase of the conv kernels, then the head kernels, in ONE kernel per RISC.  The other conv cores run the conv
    kernels of option A.  The combined kernels keep the conv result in CB 28 (sum in CB 31) because CB 10 and 11 belong to
    the head phase of the core (CB 11 is its receive area).  A conv core finishes the conv phase (tile and semaphore
    increments delivered, both barriers) before its head items start; the conv phase waits for no other core (DRAM reads, a
    local compute, NoC writes into preallocated CB 11 slots, atomic increments), so every head wait ends: no cycle."""

    rows = fp.rows_of(projected)
    mesh = projected.device()
    grid = mesh.compute_with_storage_grid_size()
    vb, vt, st, ng = plan.vb, plan.vt, plan.st, plan.ng
    HEADS, QK_REPEAT = m.HEADS, m.QK_REPEAT
    BF16, FP32 = m.BF16, m.FP32
    plain_n = len(head_pool(grid.x, grid.y))
    pool = extended_pool(grid.x, grid.y)
    head_cores = [pool[i] for i in range(plan.head_cores)]
    assert len(set(head_cores)) == plan.head_cores and plan.host_cores == plan.head_cores - plain_n > 0
    plain_cores, host_cores = head_cores[:plain_n], head_cores[plain_n:]
    host_xy = set(host_cores)
    conv_xy = {t: conv_core(t) for t in range(CONV_TILES)}
    conv_only = [c for c in conv_xy.values() if c not in host_xy]
    tile_of = {c: t for t, c in conv_xy.items()}
    plain_spans, host_spans, only_spans = range_spans(plain_cores), range_spans(host_cores), range_spans(conv_only)
    plain_set = ttnn.CoreRangeSet([_range(*s) for s in plain_spans])
    host_set = ttnn.CoreRangeSet([_range(*s) for s in host_spans])
    head_set = ttnn.CoreRangeSet([_range(*s) for s in (*plain_spans, *host_spans)])
    conv_range = _range(CONV_X0, 0, CONV_X0 + CONV_COLUMNS - 1, CONV_ROWS - 1)
    union = ttnn.CoreRangeSet([*(_range(*s) for s in plain_spans), conv_range])
    noc = gr_read.noc_map(mesh)

    def core_of(head, s, j):
        return head_core(plan, head, s, j, pool)

    blocks = [(head, s, j) for head in range(HEADS) for s in range(vb) for j in range(ng)]
    assert len(blocks) == plan.head_cores
    max_items = max(len(plan.group(j)) for j in range(ng))

    reader_cta = [rows, HEADS, m.K_TILE0, m.V_TILE0, m.Z_TILE0, m.A_TILE, m.B_TILE, vt]
    for tensor in (projected, constants, norm, recurrent):
        reader_cta.extend(fp.accessor_args(tensor))
    scalers, masks = _constants(mesh)
    for tensor in (scalers, masks):
        reader_cta.extend(fp.accessor_args(tensor))
    reader_addrs = [
        projected.buffer_address(),
        constants.buffer_address(),
        norm.buffer_address(),
        recurrent.buffer_address(),
    ]
    writer_cta = [rows, HEADS, vt, vb, *fp.accessor_args(recurrent), *fp.accessor_args(out)]
    writer_addrs = [recurrent.buffer_address(), out.buffer_address()]
    defines = [("INP_FLOAT32", "1")]
    conv_cta = [rows]
    for tensor in (projected, *older, *taps, newest):
        conv_cta.extend(fp.accessor_args(tensor))
    conv_addrs = [t.buffer_address() for t in (projected, *older, *taps, newest)]
    dest_args = {}
    for t in range(CONV_TILES):
        flat = []
        dests = destinations(t, plan, k_tile0=m.K_TILE0, v_tile0=m.V_TILE0, qk_repeat=QK_REPEAT)
        for (head, s, j), slot, sem in dests:
            x, y = noc[core_of(head, s, j)]
            flat.extend([x, y, slot, sem])
        dest_args[t] = [len(dests), *flat]

    resized = {7: st * (2 if max_items > 1 else 1), 20: st, 21: st}
    cbs = [
        fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], resized.get(index, pages), head_set)
        for index, dtype, pages in m.CBS
        if index not in CONV_ONLY_CBS
    ]
    cbs.append(
        ttnn.CBDescriptor(
            total_size=st * fp.TILE_BYTES[FP32],
            core_ranges=head_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=m.CB_SNEW, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
                ttnn.CBFormatDescriptor(buffer_index=m.CB_OUTS, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
            ],
        )
    )
    cbs.append(fp.cb_descriptor(CB_OMINE, BF16, fp.TILE_BYTES[BF16], vt, head_set))
    cbs.append(fp.cb_descriptor(CB_SQO, FP32, fp.TILE_BYTES[FP32], 4, head_set))
    cbs.append(fp.cb_descriptor(CB_QKV, BF16, fp.TILE_BYTES[BF16], 12, union))
    conv_only_set = ttnn.CoreRangeSet([_range(*s) for s in only_spans]) if only_spans else None
    for index, pages in ((0, 1), (1, 3), (2, 4), (CB_CONVOUT, 1), (CB_CONVSUM, 1)):  # host cores: conv inputs, out, sum
        cbs.append(fp.cb_descriptor(index, BF16, fp.TILE_BYTES[BF16], pages, host_set))
    if conv_only_set is not None:  # the other conv cores: as option A
        for index, pages in ((0, 1), (1, 3), (2, 4), (10, 1)):
            cbs.append(fp.cb_descriptor(index, BF16, fp.TILE_BYTES[BF16], pages, conv_only_set))

    def core(xy):
        return ttnn.CoreCoord(*xy)

    def peers(head, j):
        return [v for s in range(vb) for v in noc[core_of(head, s, j)]]

    def head_reader_args(h, s, j):
        return [
            *reader_addrs,
            len(plan.group(j)),
            plan.group(j)[0],
            h,
            s,
            scalers.buffer_address(),
            masks.buffer_address(),
        ]

    def head_writer_args(h, s, j):
        return [*writer_addrs, len(plan.group(j)), *(v for lane in plan.group(j) for v in (lane, h, s, *peers(h, j)))]

    def head_compute_args(h, s, j):
        return [len(plan.group(j)), s]

    plain_blocks = [b for b in blocks if core_of(*b) not in host_xy]
    host_blocks = [b for b in blocks if core_of(*b) in host_xy]
    assert len(plain_blocks) == plain_n and len(host_blocks) == plan.host_cores
    hoist_defines = [*defines, *(("LS_HOIST_L2", "1"),) * plan.hoist]
    compute_config = dict(fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True, unpack_to_dest_fp32=m.FP32_COPY_CBS)

    reader = fp.reader_kernel(
        kernel("reader_ls.cpp"),
        plain_set,
        reader_cta,
        [(core(core_of(*b)), head_reader_args(*b)) for b in plain_blocks],
        defines=defines,
    )
    writer = fp.writer_kernel(
        fp.kernel_source(m.NAME, "writer_vb.cpp"),
        plain_set,
        writer_cta,
        [(core(core_of(*b)), head_writer_args(*b)) for b in plain_blocks],
        defines=defines,
    )
    compute = fp.compute_kernel(
        kernel("compute_ls.cpp"),
        plain_set,
        [vt],
        [(core(core_of(*b)), head_compute_args(*b)) for b in plain_blocks],
        defines=hoist_defines,
        **compute_config,
    )
    host_reader = fp.reader_kernel(
        kernel("conv_head_reader_ls.cpp"),
        host_set,
        [*reader_cta, *conv_cta[1:]],
        [(core(core_of(*b)), [*head_reader_args(*b), *conv_addrs, tile_of[core_of(*b)]]) for b in host_blocks],
        defines=defines,
    )
    host_writer = fp.writer_kernel(
        kernel("conv_head_writer_ls.cpp"),
        host_set,
        writer_cta,
        [(core(core_of(*b)), [*head_writer_args(*b), *dest_args[tile_of[core_of(*b)]]]) for b in host_blocks],
        defines=defines,
    )
    host_compute = fp.compute_kernel(
        kernel("conv_head_compute_ls.cpp"),
        host_set,
        [vt],
        [(core(core_of(*b)), head_compute_args(*b)) for b in host_blocks],
        defines=hoist_defines,
        **compute_config,
    )
    kernels = [reader, writer, compute, host_reader, host_writer, host_compute]
    if conv_only_set is not None:
        kernels += [
            fp.reader_kernel(
                kernel("conv_reader_ls.cpp"),
                conv_only_set,
                conv_cta,
                [(core(c), [*conv_addrs, tile_of[c]]) for c in conv_only],
                defines=defines,
            ),
            fp.writer_kernel(
                fp.kernel_source(m.NAME, "conv_writer.cpp"),
                conv_only_set,
                [],
                [(core(c), dest_args[tile_of[c]]) for c in conv_only],
                defines=defines,
            ),
            fp.compute_kernel(
                fp.kernel_source(m.NAME, "conv_compute.cpp"),
                conv_only_set,
                [],
                [],
                defines=defines,
                **compute_config,
            ),
        ]

    meta = fp.program_meta(
        m.NAME,
        plan.variant,
        rows,
        reads=(projected, *older, *taps, norm, recurrent),
        writes=(recurrent, newest, out),
        dram_bytes=rows * HEADS * 2 * 4 * 4 * fp.TILE_BYTES[FP32],
        flops=rows
        * (2 * 4 * m.QKV_WIDTH + 6 * 2 * m.QK_WIDTH + 8 * HEADS * m.HEAD_DIM * m.HEAD_DIM + 6 * m.VALUE_WIDTH),
        cores=plan.head_cores + CONV_TILES - plan.host_cores,
    )
    descriptor = fp.program_descriptor(
        kernels, cbs, semaphores=[fp.semaphore_descriptor(i, union) for i in (SEM_QK, SEM_V, SEM_X)]
    )
    fp.run_program(
        [projected, *older, *taps, constants, scalers, masks, norm, recurrent, newest, out], descriptor, meta=meta
    )
    return out
