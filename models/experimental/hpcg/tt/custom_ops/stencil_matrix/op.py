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

import os

import numpy as np
import ttnn

from models.experimental.hpcg.host import native
from models.experimental.hpcg.tt import constants as C
from models.experimental.hpcg.tt.layout import TILE, allocate, u32_rows
from models.experimental.hpcg.tt.custom_ops.sparse_rows.op import core_ranges

KERNEL = "models/experimental/hpcg/tt/custom_ops/stencil_matrix/kernels/generate.cpp"
KERNEL_TRISC = "models/experimental/hpcg/tt/custom_ops/stencil_matrix/kernels/generate_trisc.cpp"
# shared with xhpcg/tt/generate.cpp and the kernels (tt/shared_constants.h, where each is described)
CB = C.CBS["STENCIL_MATRIX"]
DM_SLOTS, MB_BYTES, CHUNKS_RISC1_DEFAULT = C.DM_SLOTS, C.MB_BYTES, C.CHUNKS_RISC1_DEFAULT
XCAP, SET_PAGES, JOB_ARGS, MIN_META_PAGES = C.XCAP, C.SET_PAGES, C.JOB_ARGS, C.MIN_META_PAGES
CHUNKS_RISC1 = {}  # per core row y: overrides of CHUNKS_RISC1_DEFAULT


def _chunk_shares(rows):
    """Per core row, how many of every 8 chunks the writer RISC (NoC 1) takes, packed a nibble per row in 2 words."""
    shares = [CHUNKS_RISC1.get(y, CHUNKS_RISC1_DEFAULT) for y in range(rows)]
    return [sum(v << (4 * i) for i, v in enumerate(shares[k : k + 8])) for k in (0, 8)]


def _launch_id():
    """A random nonzero 32-bit nonce for each generation launch: the TRISCs tell this launch's messages from words any
    earlier launch, of this process or another, left in L1 (kernels/fill.h)."""
    return int.from_bytes(os.urandom(4), "little") or 1


def _rows(device, n):
    """A new fp32 ROW_MAJOR [n, 1024] device tensor (not filled)."""
    return allocate(device, (n, TILE), ttnn.ROW_MAJOR_LAYOUT)


def _tables(L):
    """Level L's storage-order tables (host/native_setup.cpp's Geo): line_band, band_y0, band_ny, band_start."""
    return [L.line_band, L.band_y0, L.band_ny, L.band_start]


class StencilJob:
    """values, offsets: new fp32 ROW_MAJOR [27 * pages per slot, 1024] device tensors (offsets are int32 bytes under
    an fp32 label, as tt.layout.rows_to_device uploads them). b: an fp32 TILE vector of the level to fill with the
    right-hand side, or None. zeros: up to two fp32 TILE vectors of the level to set to zero (the initial guess x
    and CG's direction p), written alongside so they need no separate fill."""

    kind = 0

    def __init__(self, device, L, plan, b=None, zeros=(), tables=None):
        """tables: (values, offsets) allocated beforehand (StencilJob.allocate), or None."""
        if len(zeros) > 2:
            raise ValueError("at most two zero vectors")
        if plan.off is not None or plan.nslot != 27:
            raise ValueError("needs a 27-slot plan made with SparseRowPlan(..., stencil=L)")
        self.plan, self.ntlv = plan, L.stride // TILE
        self.values, self.offsets = tables if tables is not None else StencilJob.allocate(device, L)
        self.b, self.zeros = b, list(zeros)
        self.levels, self.hw = [L], 36  # levels whose tables the job reads; header words of its META segment
        self.head = {0: [L.nx, L.nb, L.nl, self.ntlv], 22: [L.starts[c] + L.count(c) for c in range(8)]}

    @staticmethod
    def allocate(device, L):
        """The values and offsets tensors of level L (they depend on its size only, not on the plan)."""
        return tuple(_rows(device, 27 * (L.stride // TILE)) for _ in range(2))

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
        ob = range(8) if self.kind == 1 else [0]
        self.levels, self.hw = [F, C], 51
        self.head = {
            0: [F.nx, F.nb, F.nl, self.ntlv],
            22: [O.starts[c] + O.count(c) for c in ob],
            36: [C.nx, C.nb, C.nl],
            43: np.maximum(pad, 0),
        }
        self.b, self.zeros = None, []

    def tensors(self):
        return [self.values, self.offsets]


class StencilMatrixOp:
    """One launch that runs every job (StencilJob / TransferJob) in order, on as many cores as the largest job's plan.
    Every job's tables are written once the launch completes."""

    def __init__(self, device, jobs, ncores=None, job_slots=None):
        """jobs: StencilJob / TransferJob list; empty: a launch that only loads the kernel binaries (warm_up).
        ncores: cores of the launch (default: the most any job's plan uses). job_slots: argument slots for this many
        jobs (default len(jobs)); with the same cores, slots and table pages two launches share one built program."""
        g = device.compute_with_storage_grid_size()
        nc = ncores if ncores is not None else max((j.plan.nc for j in jobs), default=1)
        # META: a shared region read by every core (each level's storage-order tables, once), then one row per core
        # (the jobs' segments one after another, 16-word aligned: header, then the core's window piece lo and l1
        # tables). Both are whole 4 KB pages: the tensor's page size is then the same for every launch, and so is the
        # kernel binary (a tensor accessor's page size is a compile-time argument). A core reads the shared pages and
        # its row into one L1 buffer, so the word offsets in the headers count from the start of the shared region.
        toff, shared, sw = {}, [], 0
        for j in jobs:
            for L in j.levels:
                if id(L) not in toff:
                    toff[id(L)] = sw + np.cumsum([0] + [len(t) for t in _tables(L)[:-1]])
                    shared += _tables(L)
                    sw += sum(len(t) for t in _tables(L))
        S = -(-sw // TILE) * TILE if jobs else TILE  # a tensor holds at least one page
        at, w = [], 0
        for j in jobs:
            at.append(S + w)  # the segment's first word in L1
            w += -(-(j.hw + 2 * j.plan.piece_arrays[0].shape[1]) // 16) * 16
        w = -(-w // TILE) * TILE
        meta = np.zeros(S + nc * w, np.int32)  # the uploaded word type: no conversion
        if shared:
            meta[:sw] = np.concatenate(shared, dtype=np.int32)
        rows = meta[S:].reshape(nc, w)
        for j, a in zip(jobs, at):
            plo, _, pl1, npc = j.plan.piece_arrays
            k, R = j.plan.nc, plo.shape[1]
            seg = rows[:k, a - S : a - S + j.hw + 2 * R]
            for o, v in j.head.items():
                seg[:, o : o + len(v)] = v
            seg[:, 4] = npc
            seg[:, 6 : 6 + j.plan.own[0].size] = j.plan.own.reshape(k, -1)
            seg[:, 30:34] = toff[id(j.levels[0])]
            seg[:, 34:36] = [a + j.hw, a + j.hw + R]
            if len(j.levels) > 1:
                seg[:, 39:43] = toff[id(j.levels[1])]
            seg[:, j.hw : j.hw + R] = plo  # zero past a core's own count
            seg[:, j.hw + R :] = pl1
        self.meta = u32_rows(device, meta.reshape(-1, TILE))
        shared_pages, meta_pages = S // TILE, w // TILE
        cores = core_ranges(nc, g.x)
        # nothing grid-dependent: one binary for every job list and size
        defs = [(k, str(v)) for k, v in CB.items()] + [("DMSLOTS", str(DM_SLOTS))]
        # one accessor serves every table and vector (interleaved DRAM, the same arguments for both layouts used)
        ct = ttnn.TensorAccessorArgs(jobs[0].values if jobs else self.meta).get_compile_time_args()
        vec = next((t for j in jobs for t in j.tensors()[2:]), None)  # a TILE vector (b or a zero vector), if any
        assert vec is None or ttnn.TensorAccessorArgs(vec).get_compile_time_args() == ct, "one accessor layout"
        ct = list(ct) + list(ttnn.TensorAccessorArgs(self.meta).get_compile_time_args())
        args = [self.meta.buffer_address(), shared_pages if jobs else 0, meta_pages, len(jobs)]
        for j, a in zip(jobs, at):
            bt = j.b if j.b is not None else j.values  # unused slots point at the values (never written)
            zt = j.zeros + [j.values] * (2 - len(j.zeros))
            args += [j.kind, j.plan.nc, a, j.values.buffer_address(), j.offsets.buffer_address()]
            args += [bt.buffer_address(), int(j.b is not None), len(j.zeros)]
            args += [zt[0].buffer_address(), zt[1].buffer_address()]
        args += [0] * (JOB_ARGS * ((job_slots or len(jobs)) - len(jobs)))  # unused slots (the kernel reads len(jobs))

        def cb(idx, nbytes, page):
            fd = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=page)
            return ttnn.CBDescriptor(total_size=nbytes, core_ranges=cores, format_descriptors=[fd])

        cbs = [cb(CB[f"CB_S{h}"], 2 * SET_PAGES * 4096, 4096) for h in (0, 1)]
        # at least MIN_META_PAGES, so launches with small tables (the fine level's, the warm-up) share a program
        meta_cb = max(shared_pages + meta_pages, MIN_META_PAGES) * 4096
        cbs += [cb(CB[f"CB_M{h}"], meta_cb, 4096) for h in (0, 1)]
        cbs.append(cb(CB["CB_Z"], 4096, 4096))
        cbs += [cb(CB[f"CB_X{h}"], 2 * XCAP * 4, 2 * XCAP * 4) for h in (0, 1)]
        cbs.append(cb(CB["CB_W"], MB_BYTES, MB_BYTES))

        def program(launch):
            # data-movement kernels on both RISCs (a core's row: its logical position), and the TRISCs' share of the
            # offset stores (kernels/generate_trisc.cpp)
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=ct,
                    defines=defs,
                    common_runtime_args=[g.x, half] + args[:4] + [launch] + _chunk_shares(g.y) + args[4:],
                    config=cfg,
                )
                for half, cfg in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor()))
            ]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL_TRISC,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[],
                    defines=defs,
                    common_runtime_args=[launch],
                    config=ttnn.ComputeConfigDescriptor(),
                )
            )
            return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

        self._program, self._called = program, False
        self.prog = program(_launch_id())
        self.io = [self.meta] + ([t for j in jobs for t in j.tensors()] or [self.meta])  # generic_op takes >= 2

    def __call__(self):
        if self._called:  # every launch needs its own id (kernels/fill.h)
            self.prog = self._program(_launch_id())
        self._called = True
        return ttnn.generic_op(self.io, self.prog)


def warm_up(device, ncores):
    """A launch with no jobs on ncores cores, shaped like a one-job launch on them: the process loads the kernel
    binaries and builds that program (about 2 ms on the P150's host the first time) while the caller plans the fine
    level, so the fine level's launch (one job, the same cores) reuses the program and starts at once. TT-NN releases
    the Python GIL while it builds and enqueues a program, so this can run in a thread."""
    StencilMatrixOp(device, [], ncores=ncores, job_slots=1)()
