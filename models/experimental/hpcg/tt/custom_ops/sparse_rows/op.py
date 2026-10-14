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

from models.experimental.hpcg.tt.layout import TILE, u32_rows

KERNEL_ROOT = "models/experimental/hpcg/tt/custom_ops/sparse_rows/kernels"

# circular buffers must end below 1,572,864 B of L1 and start 111,616 B in (measured: 1,500,480 B of CBs ended at
# 1,612,096 B), so at most 1,461,248 B of CBs per core
L1_BUDGET = 1461248
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


def noc_coords(device):
    """NoC (x, y) of every worker core, indexed by logical core id x + y * grid_x."""
    g = device.compute_with_storage_grid_size()
    return [
        (c.x, c.y)
        for c in (device.worker_core_from_logical_core(ttnn.CoreCoord(i % g.x, i // g.x)) for i in range(g.x * g.y))
    ]


def debug_buffer(device):
    """Per-core cycle counters written by KDEBUG builds: row 2 * core = reader, 2 * core + 1 = writer, 16 words.
    Every launch takes one as an argument (written only by kernel_debug builds)."""
    g = device.compute_with_storage_grid_size()
    return u32_rows(device, [[0] * 16] * (2 * g.x * g.y))


def core_ranges(nc, gx):
    """The first nc logical cores, row by row."""
    full, part = divmod(nc, gx)
    rs = []
    if full:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
    if part:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(part - 1, full)))
    return ttnn.CoreRangeSet(rs)


def slot_pages(n_tiles):
    """Pages per slot in the 27-slot value / offset tensors: odd, so the 27 slot pieces of one segment sit in
    different DRAM banks (pages interleave over 8 banks; a multiple-of-8 stride put all 27 reads on one bank)."""
    return n_tiles + (1 if n_tiles % 2 == 0 else 0)


def _compute_config():
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
        meta, nseg, nseg_gs, nseg_r = plan.meta(phases, noc_xy, tpw)
        meta_t = u32_rows(device, meta)
        self.keep = [meta_t]
        meta_b = meta.shape[1] * 4
        offw = plan.nslot == 27  # the writer reads its own slots' values and the other gatherers' offsets
        defs = dict(
            CB,
            NG=NGATH,
            META_B=meta_b,
            NSLOT=plan.nslot,
            WINB=plan.win_b,
            NGRP=min(NGRP, plan.nslot),
            TPW=tpw,
            NTLV=slot_pages(plan.pout // TILE) if plan.nslot == 27 else plan.pout // TILE,
        )
        if mode == "gs":
            defs["GS"] = 1
        if offw:
            defs["OFFW"] = 1
            defs["FASTMAC"] = 1  # SFPU multiply-add on the occupied faces only, one pass per slot
        if addb:
            defs["ADDB"] = 1
        if out2 is not None:
            defs["RESID"] = 1
            if restf:
                defs["RESTF"] = 1
        if zerox:
            defs["ZEROX"] = 1
        if wsplit:
            defs["WSPLIT"] = 1
        if kernel_debug:
            defs["KDEBUG"] = 1
        b = b if b is not None else x
        o2 = out2 if out2 is not None else out
        dbg = debug if debug is not None else debug_buffer(device)
        self.keep.append(dbg)
        cores = core_ranges(plan.nc, g.x)
        reader_args, writer_args, compute_args = {}, {}, {}
        for i in range(plan.nc):
            c = (i % g.x, i // g.x)
            reader_args[c] = [
                x.buffer_address(),
                values.buffer_address(),
                offsets.buffer_address(),
                meta_t.buffer_address(),
                b.buffer_address(),
                i,
                dbg.buffer_address(),
            ]
            writer_args[c] = [
                out.buffer_address(),
                meta_t.buffer_address(),
                i,
                offsets.buffer_address(),
                noc_xy[i][0],
                noc_xy[i][1],
                dbg.buffer_address(),
                values.buffer_address(),
                o2.buffer_address(),
                x.buffer_address(),
            ]
            compute_args[c] = [nseg[i], nseg_gs[i], nseg_r[i]]

        def accessors(*ts):
            return [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]

        reader_ct = accessors(x, values, offsets, meta_t, b, dbg)
        writer_ct = accessors(out, meta_t, offsets, dbg, values, o2, x)

        # circular buffers; L1 pages hold tpw words (256: one 16x16 face, so unpack / pack move one face)
        pg = tpw * 4
        tile_desc = ttnn._ttnn.program_descriptor.TileDescriptor(16, 16) if tpw == 256 else None

        def cb(idx, nbytes, page=4096, tile=False):
            fd = ttnn.CBFormatDescriptor(
                buffer_index=idx,
                data_format=ttnn.float32,
                page_size=page,
                **({"tile": tile_desc} if tile and tile_desc else {}),
            )
            return ttnn.CBDescriptor(total_size=nbytes, core_ranges=cores, format_descriptors=[fd])

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
        flipped = {}
        for i in range(plan.nc):
            cx, cy = i % g.x, i // g.x
            f = bool(noc_flip) and noc_flip[0] <= cx <= noc_flip[1] and noc_flip[2] <= cy <= noc_flip[3]
            flipped.setdefault(f, []).append(i)
        define_list = [(k, str(v)) for k, v in defs.items()]
        kernels = []
        for f, idx in flipped.items():
            cr = (
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(i % g.x, i // g.x), ttnn.CoreCoord(i % g.x, i // g.x)) for i in idx]
                )
                if len(flipped) > 1
                else cores
            )
            rcfg = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1 if f else ttnn.NOC.NOC_0
            )
            wcfg = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0 if f else ttnn.NOC.NOC_1
            )
            ra, wa = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i in idx:
                c = (i % g.x, i // g.x)
                ra[c[0]][c[1]] = reader_args[c]
                wa[c[0]][c[1]] = writer_args[c]
            kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=f"{KERNEL_ROOT}/reader.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cr,
                    compile_time_args=reader_ct,
                    defines=define_list,
                    runtime_args=ra,
                    config=rcfg,
                ),
                ttnn.KernelDescriptor(
                    kernel_source=f"{KERNEL_ROOT}/writer.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cr,
                    compile_time_args=writer_ct,
                    defines=define_list,
                    runtime_args=wa,
                    config=wcfg,
                ),
            ]
        ca = ttnn.RuntimeArgs()
        for (cx, cy), v in compute_args.items():
            ca[cx][cy] = v
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_ROOT}/compute.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=[],
                defines=define_list,
                runtime_args=ca,
                config=_compute_config(),
            )
        )
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in range(16)]
        self.prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        # generic_op's last tensor is the launch's output; gs updates x in place
        self.io = (
            [x, values, offsets, meta_t, out]
            if mode != "gs"
            else [values, offsets, meta_t, b] + ([out2] if out2 is not None else []) + [x]
        )

    def __call__(self):
        return ttnn.generic_op(self.io, self.prog)
