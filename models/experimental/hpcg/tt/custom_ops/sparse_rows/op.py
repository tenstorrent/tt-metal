# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Sparse-row op: one persistent launch that streams 27-slot (or 1-slot) matrix rows and gathers x from L1.

Each core owns a share of the rows of every phase block (SparseRowPlan) and loads its x window pieces into L1
once. Per row segment, the reader and writer stream the fp32 matrix values and 32-bit column offsets from DRAM,
and four processors gather x[col] from the window (reader, writer, and the compute core's pack and unpack
threads). The math thread sums the 27 products on the SFPU in HPCG's column order.

Modes
  gs    Gauss-Seidel: each phase updates one colour in place, x = (b - s + d x) / d (HPCG's ComputeSYMGS_ref
        update). Cores push updated rows into the windows of the neighbours that hold them and wait for their
        neighbours between phases. Phases 8 + b are SpMV passes over block b on the current x, written to out2
        (RESID: the pre-smoother's residual SpMV; restf: block 0 gets r - A z, the restriction's input).
  spmv  out = A x for every phase block (1-slot plans: restriction and prolongation gathers; addb: out = b + A x).
"""

import ttnn

from models.experimental.hpcg.tt.layout import TILE, allocate, u32_rows

KERNEL_ROOT = "models/experimental/hpcg/tt/custom_ops/sparse_rows/kernels"

# circular buffers must end below 1,572,864 B of L1 and start 111,616 B in (measured: 1,500,480 B of CBs ended at
# 1,612,096 B), so at most 1,461,248 B of CBs per core
L1_BUDGET = 1461248
F_WSPLIT, F_ZEROX, F_RESTF, F_ADDB = 1, 2, 4, 8  # launch option flags, as in kernels/gather.h
CB = dict(
    CB_V=0,
    CB_G=1,
    CB_B=2,
    CB_G2=5,
    CB_RDY=6,
    CB_G3=8,
    CB_OFF3=9,
    CB_MB=10,
    CB_G4=11,
    CB_OFF4=12,
    CB_V2=13,
    CB_OUT=16,
    CB_WIN=24,
    CB_OFF=25,
    CB_META=26,
    CB_META2=27,
    CB_OFF2=28,
)
NGATH = 4  # gathering processors per core: reader, writer, pack thread, unpack thread
NGRP = 3  # slot groups per segment: a work unit is (segment, group); its reads are issued one unit ahead
GDEP = (8, 6, 4, 3, 2)  # gathered pages in flight per gatherer, largest that fits L1 first
VDEP = 27  # matrix-value pages in flight (reader)
# NoC per core: reader on NoC 0 / writer on NoC 1, swapped inside this core rectangle (logical x0, x1, y0, y1).
# DRAM read responses congest some links; the swap measured fastest for every launch with an SpMV pass.
NOC_FLIP_RECT = (7, 12, 0, 9)


_NOC_XY = {}


def noc_coords(device):
    """NoC (x, y) of every worker core, indexed by logical core id x + y * grid_x (looked up once per device)."""
    if id(device) not in _NOC_XY:
        g = device.compute_with_storage_grid_size()
        _NOC_XY[id(device)] = [
            (c.x, c.y)
            for c in (device.worker_core_from_logical_core(ttnn.CoreCoord(i % g.x, i // g.x)) for i in range(g.x * g.y))
        ]
    return _NOC_XY[id(device)]


_COORDS = {}


def logical_coords(device):
    """ttnn.CoreCoord of every worker core, indexed by logical core id x + y * grid_x (made once per device)."""
    if id(device) not in _COORDS:
        g = device.compute_with_storage_grid_size()
        _COORDS[id(device)] = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(g.x * g.y)]
    return _COORDS[id(device)]


def debug_buffer(device, zeros=True):
    """Per-core cycle counters written by KDEBUG builds: row 2 * core = reader, 2 * core + 1 = writer, 16 words.
    Every launch takes one as an argument (written only by kernel_debug builds). zeros=False: allocated, not filled
    (for runs without kernel_debug, which never write or read it)."""
    g = device.compute_with_storage_grid_size()
    if not zeros:
        return allocate(device, (2 * g.x * g.y, 16), ttnn.ROW_MAJOR_LAYOUT)
    return u32_rows(device, [[0] * 16] * (2 * g.x * g.y))


_RANGES = {}


def core_ranges(nc, gx):
    """The first nc logical cores, row by row (made once per core count: descriptors copy what they are given)."""
    if (nc, gx) not in _RANGES:
        full, part = divmod(nc, gx)
        rs = []
        if full:
            rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
        if part:
            rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(part - 1, full)))
        _RANGES[(nc, gx)] = ttnn.CoreRangeSet(rs)
    return _RANGES[(nc, gx)]


def _flip_ranges(nc, gx, rect):
    """The first nc logical cores split by whether they lie in rect (x0, x1, y0, y1; () for none) -> {flipped: [(x0,
    x1, y0, y1) rectangles]}, each row's runs merged with the rows above when they span the same x range."""
    out = {}
    for y in range(-(-nc // gx)):
        n = min(gx, nc - y * gx)  # cores in this row: x = 0 .. n - 1
        f0 = rect and rect[2] <= y <= rect[3]
        runs = [(False, 0, n - 1)] if not f0 else []
        if f0:
            a, b = rect[0], min(rect[1], n - 1)
            runs = [(False, 0, min(a, n) - 1), (True, a, b), (False, b + 1, n - 1)]
        for f, x0, x1 in runs:
            if x1 < x0:
                continue
            rs = out.setdefault(bool(f), [])
            last = rs[-1] if rs else None
            if last and last[0] == x0 and last[1] == x1 and last[3] == y - 1:
                rs[-1] = (x0, x1, last[2], y)
            else:
                rs.append((x0, x1, y, y))
    return out


_GROUPS = {}


def _kernel_groups(nc, gx, rect):
    """(core ranges, reader config, writer config) of each NoC assignment of a launch on the first nc cores: reader
    on NoC 0 and writer on NoC 1, swapped inside rect (_flip_ranges). Made once per (nc, gx, rect)."""
    key = (nc, gx, tuple(rect))
    if key not in _GROUPS:
        flipped = _flip_ranges(nc, gx, rect)
        groups = []
        for f, rects in flipped.items():
            cr = core_ranges(nc, gx)
            if len(flipped) > 1:
                cr = ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, x1, y0, y1 in rects]
                )
            rcfg = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1 if f else ttnn.NOC.NOC_0
            )
            wcfg = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0 if f else ttnn.NOC.NOC_1
            )
            groups.append((cr, rcfg, wcfg))
        _GROUPS[key] = groups
    return _GROUPS[key]


_CBS = {}


def _cb(nc, gx, idx, nbytes, page, face):
    """A circular buffer of fp32 pages on the first nc cores (face: pages of one 16 x 16 tile face), made once per
    shape: launches share most of their buffers' shapes."""
    key = (nc, gx, idx, nbytes, page, face)
    if key not in _CBS:
        tile = {"tile": ttnn._ttnn.program_descriptor.TileDescriptor(16, 16)} if face else {}
        fd = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=page, **tile)
        _CBS[key] = ttnn.CBDescriptor(total_size=nbytes, core_ranges=core_ranges(nc, gx), format_descriptors=[fd])
    return _CBS[key]


_SEMS = {}


def _semaphores(nc, gx, cores):
    """The 16 semaphores of a launch on the first nc cores (all start at 0), made once per core count."""
    if (nc, gx) not in _SEMS:
        _SEMS[(nc, gx)] = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in range(16)]
    return _SEMS[(nc, gx)]


def slot_pages(n_tiles):
    """Pages per slot in the 27-slot value / offset tensors: odd, so the 27 slot pieces of one segment sit in
    different DRAM banks (pages interleave over 8 banks; a multiple-of-8 stride put all 27 reads on one bank)."""
    return n_tiles + (1 if n_tiles % 2 == 0 else 0)


_CFG = []


def _compute_config():
    """The compute kernels' configuration (the same for every launch; made once)."""
    if not _CFG:
        _CFG.append(_make_compute_config())
    return _CFG[0]


def _make_compute_config():
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False
    )
    m = [ttnn.UnpackToDestMode.Default] * 64
    for i in (CB["CB_V"], CB["CB_G"], CB["CB_B"], CB["CB_G2"], CB["CB_G3"], CB["CB_G4"], CB["CB_V2"]):
        m[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(m)
    return cfg


class SparseRowOp:
    """Builds the program once; each call launches it (trace-capturable).

    values, offsets: fp32 matrix values and uint32 window byte offsets, slot by slot in 4 KB pages (27-slot plans:
    slot_pages pages per slot); the kernels read raw pages, so values may be TILE or ROW_MAJOR rows of 1024.
    x: the input vector (gs: updated in place). out: spmv output. b: gs right-hand side, or addb's addend. out2: RESID output. zerox: x = 0 on entry (the windows are cleared on chip, x is not read).
    tpw: words per L1 page (1024, or 256 = one face on coarse grids). wsplit: the writer loads half the window.
    debug: a debug_buffer(device) for the cycle counters (one is allocated if None).
    """

    def __init__(
        self,
        device,
        plan,
        values,
        offsets,
        mode,
        phases,
        x,
        out,
        b=None,
        out2=None,
        zerox=False,
        tpw=TILE,
        addb=False,
        restf=False,
        wsplit=False,
        noc_flip=None,
        kernel_debug=False,
        debug=None,
        table_phases=None,
    ):
        if mode not in ("gs", "spmv"):
            raise ValueError(f"mode must be 'gs' or 'spmv', got {mode!r}")
        if tpw not in (256, TILE):
            raise ValueError(f"tpw must be 256 or {TILE}, got {tpw}")
        for t in (x, out):
            if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT:
                raise ValueError("x and out must be fp32 TILE tensors")
        if values.dtype != ttnn.float32 or not (
            values.layout == ttnn.TILE_LAYOUT or (values.layout == ttnn.ROW_MAJOR_LAYOUT and values.shape[-1] == TILE)
        ):
            raise ValueError("values must be fp32, TILE or ROW_MAJOR rows of 1024 (4 KB pages)")
        g = device.compute_with_storage_grid_size()
        noc_xy = noc_coords(device)
        # the grid-dependent sizes (metadata row bytes, pages per slot, window bytes) as runtime arguments: the kernel
        # binaries then do not depend on the level or the grid size, so a process loads each binary once (about 2-3 ms
        # per binary)
        # table_phases: phases of a table to share (another launch on the plan whose phases start with these): the
        # kernels then run only the first len(phases) of its phases
        meta_t, nseg, nseg_gs, nseg_r, _ = plan.meta_tensor(device, table_phases or phases, noc_xy, tpw)
        nph = 0
        if table_phases is not None:
            if list(table_phases[: len(phases)]) != list(phases):
                raise ValueError("phases must be a prefix of table_phases")
            nph = len(phases)
            if all(p < 8 for p in phases) and all(p >= 8 for p in table_phases[nph:]):
                # the table's Gauss-Seidel phases, then only SpMV passes: the table's Gauss-Seidel counts are this
                # launch's (all its segments are Gauss-Seidel ones), with no RESTF segments
                nseg, nseg_r = nseg_gs, [0] * plan.nc
            else:
                nseg, nseg_gs, nseg_r = plan.segment_counts(phases, tpw)
        meta_b = meta_t.shape[1] * 4  # one DRAM page per core: its row, a multiple of 64 bytes
        self.keep = [meta_t]
        ntlv = slot_pages(plan.pout // TILE) if plan.nslot == 27 else plan.pout // TILE
        offw = plan.nslot == 27  # the writer reads its own slots' values and the other gatherers' offsets
        defs = dict(
            CB,
            NG=NGATH,
            NSLOT=plan.nslot,
            NGRP=min(NGRP, plan.nslot),
            TPW=tpw,
        )
        if mode == "gs":
            defs["GS"] = 1
        if offw:
            defs["OFFW"] = 1
            defs["FASTMAC"] = 1  # SFPU multiply-add on the occupied faces only, one pass per slot
        # launch options: one runtime flags word (kernels/gather.h), not defines, so launches that differ only in
        # options share a kernel binary
        flags = (
            (F_WSPLIT if wsplit else 0)
            | (F_ZEROX if zerox else 0)
            | (F_RESTF if restf and out2 is not None else 0)
            | (F_ADDB if addb else 0)
        )
        if kernel_debug:
            defs["KDEBUG"] = 1
        b = b if b is not None else x
        o2 = out2 if out2 is not None else out
        dbg = debug if debug is not None else debug_buffer(device)
        self.keep.append(dbg)
        cores = core_ranges(plan.nc, g.x)
        # data-movement kernels: one common argument list (a core finds its row from its logical position, so no
        # per-core lists to build); the compute kernel keeps per-core segment counts
        ax, av, aoff, ameta, ab, adbg = (t.buffer_address() for t in (x, values, offsets, meta_t, b, dbg))
        aout, ao2 = out.buffer_address(), o2.buffer_address()
        reader_common = [ax, av, aoff, ameta, ab, g.x, adbg, meta_b, ntlv, plan.win_b, flags, nph]
        writer_common = [aout, ameta, g.x, aoff, 0, 0, adbg, av, ao2, ax, meta_b, ntlv, flags, nph]

        def accessors(*ts):
            return [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]

        # META's accessor takes the values' compile-time arguments (interleaved DRAM, 4 KB pages): an accessor's page
        # size is a compile-time argument, only a default the kernels override with META's row size at run time, so
        # META's own would make one binary per row width
        reader_ct = accessors(x, values, offsets, values, b, dbg)
        writer_ct = accessors(out, values, offsets, dbg, values, o2, x)

        # circular buffers; L1 pages hold tpw words (256: one 16x16 face, so unpack / pack move one face)
        pg = tpw * 4

        def cb(idx, nbytes, page=4096, tile=False):
            return _cb(plan.nc, g.x, idx, nbytes, page, tile and tpw == 256)

        own = [j % NGATH for j in range(plan.nslot)]
        c2 = own.count(2)
        c3 = own.count(3)
        ring = max(2 * c2, 1)  # pack-thread offset pages in flight (2 segments)
        defs["RING"] = ring
        ngrp = defs["NGRP"]
        gsz = -(-plan.nslot // ngrp)
        # offset pages per unit of the reader (gatherer 0) and writer (gatherer 1): their most slots in one group
        mg = [
            max(
                1,
                max(
                    sum(1 for j in range(lo, min(lo + gsz, plan.nslot)) if own[j] == o)
                    for lo in range(0, plan.nslot, gsz)
                ),
            )
            for o in (0, 1)
        ]
        if offw:  # reader's value pages: a multiple of its pages per segment keeps every group's pages contiguous
            nvr = sum(1 for j in range(plan.nslot) if own[j] != 1)
            nvw = plan.nslot - nvr
            vpages = nvr * max(1, VDEP // nvr)
        else:
            vpages = min(VDEP, 2 * plan.nslot)
        win_b = plan.win_b

        def make_cbs(gd):
            cbs = [
                cb(CB["CB_V"], vpages * pg, pg, True),
                cb(CB["CB_G"], gd * pg, pg, True),
                cb(CB["CB_G2"], gd * pg, pg, True),
                cb(CB["CB_RDY"], 64, 32),
                cb(CB["CB_OUT"], 2 * pg, pg, True),
                cb(CB["CB_WIN"], win_b, win_b),
                cb(CB["CB_OFF"], 2 * mg[0] * pg, pg),
                cb(CB["CB_META"], meta_b, meta_b),
                cb(CB["CB_META2"], meta_b, meta_b),
                cb(CB["CB_OFF2"], 2 * mg[1] * pg, pg),
                cb(CB["CB_G3"], gd * pg, pg, True),
                cb(CB["CB_OFF3"], ring * pg, pg),
                cb(CB["CB_MB"], 128, 128),
                cb(CB["CB_G4"], 4 * pg, pg, True),
                # CB_OFF4: a multiple of the unpack gatherer's offset pages per segment, so no group's pages
                # straddle the end of the CB
                cb(CB["CB_OFF4"], 2 * max(1, c3) * pg, pg),
            ]
            if mode == "gs" or addb:
                cbs.append(cb(CB["CB_B"], 2 * pg, pg, True))
            if offw:
                cbs.append(cb(CB["CB_V2"], 2 * nvw * pg, pg, True))
            return cbs

        for gd in GDEP:  # large grids (x windows near 1 MB): fewer gathered pages in flight so the CBs fit L1
            cbs = make_cbs(gd)
            if sum(c.total_size for c in cbs) <= L1_BUDGET:
                break
        total = sum(c.total_size for c in cbs)
        if total > L1_BUDGET:
            raise ValueError(
                f"L1: {total} B of circular buffers (window {win_b} B) exceed {L1_BUDGET} B; grid too large"
            )
        self.l1_total, self.win_b = total, win_b

        if noc_flip is None:
            noc_flip = NOC_FLIP_RECT if (mode == "spmv" or out2 is not None) else ()
        define_list = [(k, str(v)) for k, v in defs.items()]
        kernels = []
        for cr, rcfg, wcfg in _kernel_groups(plan.nc, g.x, noc_flip):
            kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=f"{KERNEL_ROOT}/reader.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cr,
                    compile_time_args=reader_ct,
                    defines=define_list,
                    common_runtime_args=reader_common,
                    config=rcfg,
                ),
                ttnn.KernelDescriptor(
                    kernel_source=f"{KERNEL_ROOT}/writer.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cr,
                    compile_time_args=writer_ct,
                    defines=define_list,
                    common_runtime_args=writer_common,
                    config=wcfg,
                ),
            ]
        # compute kernel: per-core arguments, as (core, list) pairs on cached CoreCoords (half the cost of filling a
        # ttnn.RuntimeArgs core by core; as common arguments, one list or one per group of equal cores, they made
        # coarse launches 4 to 15 us slower). Launches with the same rows and options (CG start, CG SpMV) share it
        ckey = ("compute", id(device), tuple(phases), tpw, flags, nph, tuple(define_list))
        if ckey not in plan._tables:
            plan._tables[ckey] = ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_ROOT}/compute.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=[],
                defines=define_list,
                runtime_args=list(zip(logical_coords(device), [[*v, flags, nph] for v in zip(nseg, nseg_gs, nseg_r)])),
                config=_compute_config(),
            )
        kernels.append(plan._tables[ckey])
        sems = _semaphores(plan.nc, g.x, cores)
        self.prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        # generic_op's last tensor is the launch's output; gs updates x in place
        self.io = (
            [x, values, offsets, meta_t, out]
            if mode != "gs"
            else [values, offsets, meta_t, b] + ([out2] if out2 is not None else []) + [x]
        )

    def __call__(self):
        return ttnn.generic_op(self.io, self.prog)
