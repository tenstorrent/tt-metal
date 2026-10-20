# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Matrix generation op: the device writes the tables the sparse-row launches read, one launch for a list of jobs.

StencilJob: one level's 27-slot matrix values and window offsets (and optionally b and zero vectors), the tables a
Gauss-Seidel launch reads (sparse_rows): for row q and slot j, the value V[j, q] and the byte offset of its column in
the core's L1 window of x. TransferJob: the 1-slot weights and window offsets of a grid transfer launch (restriction
or prolongation). On the host they took building the tables (reference.problem.Level, Level.link), deriving the
offsets from the column tables (SparseRowPlan) and uploading them: about 1.5 GB at 192^3. Here each core computes its
own rows from the storage-order formulas and the plan's window pieces (SparseRowPlan(..., stencil=L) and
SparseRowPlan(..., transfer=..., host_tables=False)), so only a few KB of per-core tables go to the device. Run once at
setup; the result is bit-identical to the host tables (tests/test_hpcg.py).
"""

import numpy as np
import ttnn

from models.experimental.hpcg.host import native
from models.experimental.hpcg.tt.layout import TILE, u32_rows
from models.experimental.hpcg.tt.custom_ops.sparse_rows.op import core_ranges

KERNEL = "models/experimental/hpcg/tt/custom_ops/stencil_matrix/kernels/generate.cpp"
CB = dict(CB_S0=0, CB_S1=1, CB_M0=2, CB_M1=3, CB_Z=4, CB_X0=5, CB_X1=6)
XCAP = 2048  # value exceptions per staging set (generate.cpp)
SET_PAGES = 55  # staging set per RISC: 27 offset pages, 27 value pages, one b page (generate.cpp's SET)
JOB_ARGS = 10  # common arguments per job (generate.cpp)


def _rows(device, n):
    """A new fp32 ROW_MAJOR [n, 1024] device tensor (not filled)."""
    return ttnn.empty(ttnn.Shape([n, TILE]), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _segment(head_words, tables, plan):
    """A job's META segment for every core of its plan: head_words header words, then the tables (the same for every
    core), then the window piece lo and l1 tables (R words each, R = the most pieces of a core; a core reads its own
    count, [4]); [30 .. 34) and [34], [35] get their word offsets."""
    tab = np.concatenate(tables, dtype=np.int32)
    plo, _, pl1, npc = plan.piece_arrays
    nc, w0, R = plan.nc, head_words + len(tab), int(npc.max())  # w0: word offset of the piece lo table
    seg = np.zeros((nc, w0 + 2 * R), np.int32)  # the uploaded word type: no conversion
    seg[:, 4] = npc
    seg[:, 6 : 6 + plan.own[0].size] = plan.own.reshape(nc, -1)
    seg[:, 34:36] = [w0, w0 + R]
    seg[:, head_words:w0] = tab
    seg[:, w0 : w0 + R] = plo[:, :R]  # zero past a core's own count
    seg[:, w0 + R :] = pl1[:, :R]
    return seg


class StencilJob:
    """values, offsets: new fp32 ROW_MAJOR [27 * pages per slot, 1024] device tensors (offsets are int32 bytes under
    an fp32 label, as tt.layout.rows_to_device uploads them). b: an fp32 TILE vector of the level to fill with the
    right-hand side, or None. zeros: up to two fp32 TILE vectors of the level to set to zero (the initial guess x
    and CG's direction p), written alongside so they need no separate fill."""

    kind = 0

    def __init__(self, device, L, plan, b=None, zeros=()):
        if len(zeros) > 2:
            raise ValueError("at most two zero vectors")
        if plan.off is not None or plan.nslot != 27:
            raise ValueError("needs a 27-slot plan made with SparseRowPlan(..., stencil=L)")
        self.plan, self.ntlv = plan, L.stride // TILE
        self.values, self.offsets = (_rows(device, 27 * self.ntlv) for _ in range(2))
        self.b, self.zeros = b, list(zeros)
        tables = [L.line_band, L.band_y0, L.band_ny, L.band_start]
        self.seg = _segment(36, tables, plan)
        self.seg[:, :4] = [L.nx, L.nb, L.nl, self.ntlv]
        self.seg[:, 22:30] = [L.starts[c] + L.count(c) for c in range(8)]
        self.seg[:, 30:34] = np.cumsum([36] + [len(t) for t in tables[:-1]])  # word offsets of the four tables

    def tensors(self):
        return [self.values, self.offsets] + self.zeros + ([self.b] if self.b is not None else [])


class TransferJob:
    """weights, offsets: new fp32 ROW_MAJOR [out rows / 1024, 1024] device tensors, the 1-slot values and window
    offsets of a grid transfer launch planned with SparseRowPlan(..., transfer=(kind, F, C), host_tables=False)."""

    def __init__(self, device, plan):
        if plan.transfer is None or plan.off is not None:
            raise ValueError("needs a transfer plan made with host_tables=False")
        name, F, C = plan.transfer
        self.kind = {"restrict": 1, "prolong": 2}[name]
        self.plan, self.ntlv = plan, plan.pout // TILE
        self.values, self.offsets = (_rows(device, self.ntlv) for _ in range(2))
        O = C if self.kind == 1 else F  # the level whose rows are the out rows
        NB = len(plan.out_blocks)
        pad = np.zeros(NB, np.int64)
        native.lib().hpcg_transfer_pad(self.kind, *F.geo(), *C.geo(), NB, pad)
        ftab = [F.line_band, F.band_y0, F.band_ny, F.band_start]
        ctab = [C.line_band, C.band_y0, C.band_ny, C.band_start]
        self.seg = _segment(51, ftab + ctab, plan)
        self.seg[:, :4] = [F.nx, F.nb, F.nl, self.ntlv]
        ob = range(8) if self.kind == 1 else [0]
        self.seg[:, 22 : 22 + NB] = [O.starts[c] + O.count(c) for c in ob]
        offs = np.cumsum([51] + [len(t) for t in ftab + ctab[:-1]])  # word offsets of the eight tables
        self.seg[:, 30:34] = offs[:4]
        self.seg[:, 36:39] = [C.nx, C.nb, C.nl]
        self.seg[:, 39:43] = offs[4:]
        self.seg[:, 43 : 43 + NB] = np.maximum(pad, 0)
        self.b, self.zeros = None, []

    def tensors(self):
        return [self.values, self.offsets]


class StencilMatrixOp:
    """One launch that runs every job (StencilJob / TransferJob) in order, on as many cores as the largest job's plan.
    Every job's tables are written once the launch completes."""

    def __init__(self, device, jobs):
        g = device.compute_with_storage_grid_size()
        nc = max(j.plan.nc for j in jobs)
        # each core's META row: the jobs' segments one after another (16-word aligned), in whole 4 KB pages: the
        # tensor's page size is then the same for every launch, and so is the kernel binary (a tensor accessor's page
        # size is a compile-time argument)
        at, w = [], 0
        for j in jobs:
            at.append(w)
            w += -(-j.seg.shape[1] // 16) * 16
        w = -(-w // TILE) * TILE
        meta = np.zeros((nc, w), np.int32)
        for j, a in zip(jobs, at):
            meta[: j.seg.shape[0], a : a + j.seg.shape[1]] = j.seg
        self.meta = u32_rows(device, meta.reshape(-1, TILE))
        meta_pages = w // TILE
        cores = core_ranges(nc, g.x)
        defs = [(k, str(v)) for k, v in CB.items()]  # nothing grid-dependent: one binary for every job list and size
        data = jobs[0].values
        ct = ttnn.TensorAccessorArgs(data).get_compile_time_args()
        for j in jobs:  # one accessor serves every table and vector: interleaved DRAM, the same arguments
            for t in j.tensors():
                assert ttnn.TensorAccessorArgs(t).get_compile_time_args() == ct, "tensors need one accessor layout"
        ct = list(ct) + list(ttnn.TensorAccessorArgs(self.meta).get_compile_time_args())
        args = [self.meta.buffer_address(), meta_pages, len(jobs)]
        for j, a in zip(jobs, at):
            bt = j.b if j.b is not None else j.values  # unused slots point at the values (never written)
            zt = j.zeros + [j.values] * (2 - len(j.zeros))
            args += [j.kind, j.plan.nc, a, j.values.buffer_address(), j.offsets.buffer_address()]
            args += [bt.buffer_address(), int(j.b is not None), len(j.zeros)]
            args += [zt[0].buffer_address(), zt[1].buffer_address()]

        def cb(idx, nbytes, page):
            fd = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=page)
            return ttnn.CBDescriptor(total_size=nbytes, core_ranges=cores, format_descriptors=[fd])

        cbs = [cb(CB[f"CB_S{h}"], 2 * SET_PAGES * 4096, 4096) for h in (0, 1)]
        cbs += [cb(CB[f"CB_M{h}"], meta_pages * 4096, 4096) for h in (0, 1)]
        cbs.append(cb(CB["CB_Z"], 4096, 4096))
        cbs += [cb(CB[f"CB_X{h}"], 2 * XCAP * 4, 2 * XCAP * 4) for h in (0, 1)]
        kernels = []
        for half, cfg in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor())):
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=ct,
                    defines=defs,
                    common_runtime_args=[g.x, half] + args,  # a core's row: its logical position
                    config=cfg,
                )
            )
        self.prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        self.io = [self.meta] + [t for j in jobs for t in j.tensors()]

    def __call__(self):
        return ttnn.generic_op(self.io, self.prog)
