# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Stencil matrix op: the device writes one level's 27-slot matrix values and window offsets (and optionally b).

These are the tables a Gauss-Seidel launch reads (sparse_rows): for row q and slot j, the value V[j, q] and the
byte offset of its column in the core's L1 window of x. On the host they took building two [27, P] tables
(reference.problem.Level), deriving the offsets from the column table (SparseRowPlan) and uploading both: about
1.5 GB at 192^3. Here each core computes its own rows from the storage-order formula and the plan's window pieces
(SparseRowPlan(..., stencil=L)), so only a few KB of per-core tables go to the device. One launch, run once at setup;
the result is bit-identical to the host tables (tests/test_hpcg.py).
"""

import numpy as np
import ttnn

from models.experimental.hpcg.tt.layout import TILE, u32_rows
from models.experimental.hpcg.tt.custom_ops.sparse_rows.op import core_ranges

KERNEL = "models/experimental/hpcg/tt/custom_ops/stencil_matrix/kernels/generate.cpp"
CB = dict(CB_S0=0, CB_S1=1, CB_M0=2, CB_M1=3, CB_Z=4)
SET_PAGES = 55  # staging set per RISC: 27 offset pages, 27 value pages, one b page (generate.cpp's SET)


class StencilMatrixOp:
    """values, offsets: new fp32 ROW_MAJOR [27 * pages per slot, 1024] device tensors (offsets are int32 bytes under
    an fp32 label, as tt.layout.rows_to_device uploads them). b: an fp32 TILE vector of the level to fill with the
    right-hand side, or None. zeros: up to two fp32 TILE vectors of the level to set to zero (the initial guess x
    and CG's direction p), written alongside so they need no separate fill."""

    def __init__(self, device, L, plan, b=None, zeros=()):
        if len(zeros) > 2:
            raise ValueError("at most two zero vectors")
        if plan.off is not None or plan.nslot != 27:
            raise ValueError("needs a 27-slot plan made with SparseRowPlan(..., stencil=L)")
        g = device.compute_with_storage_grid_size()
        ntlv = L.stride // TILE
        shape = ttnn.Shape([27 * ntlv, TILE])
        self.values, self.offsets = (
            ttnn.empty(shape, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device) for _ in range(2)
        )
        tables = [L.line_band, L.band_y0, L.band_ny, L.band_start]
        tab = np.concatenate([np.asarray(t, np.int64) for t in tables])  # the same for every core
        plo, _, pl1, npc = plan.piece_arrays
        nc, w0 = plan.nc, 36 + len(tab)  # w0: word offset of the piece lo table (then the piece l1 table)
        # rows of whole 4 KB pages: the tensor's page size is then the same at every grid size, and so is the kernel
        # binary (a tensor accessor's page size is a compile-time argument)
        w = -(-(w0 + 2 * int(npc.max())) // TILE) * TILE
        meta = np.zeros((nc, w), np.int64)
        meta[:, :6] = [L.nx, L.nb, L.nl, ntlv, 0, 0]
        meta[:, 4] = npc
        meta[:, 6:22] = plan.own.reshape(nc, -1)
        meta[:, 22:30] = [L.starts[c] + L.count(c) for c in range(8)]
        meta[:, 30:34] = np.cumsum([36] + [len(t) for t in tables[:-1]])  # word offsets of the four tables
        meta[:, 34] = w0
        meta[:, 35] = w0 + npc
        meta[:, 36:w0] = tab
        for k in range(nc):
            n = npc[k]
            meta[k, w0 : w0 + n] = plo[k, :n]
            meta[k, w0 + n : w0 + 2 * n] = pl1[k, :n]
        self.meta = u32_rows(device, meta.reshape(-1, TILE))
        meta_pages = w // TILE
        cores = core_ranges(plan.nc, g.x)
        bt = b if b is not None else self.values
        defs = [(k, str(v)) for k, v in CB.items()]  # nothing grid-dependent: one binary for every level and size
        zt = list(zeros) + [bt] * (2 - len(zeros))  # unused slots point at b (never written)
        ct = [
            a
            for t in (self.values, self.offsets, self.meta, bt, *zt)
            for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()
        ]

        def cb(idx, nbytes, page):
            fd = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=page)
            return ttnn.CBDescriptor(total_size=nbytes, core_ranges=cores, format_descriptors=[fd])

        cbs = [cb(CB[f"CB_S{h}"], 2 * SET_PAGES * 4096, 4096) for h in (0, 1)]
        cbs += [cb(CB[f"CB_M{h}"], meta_pages * 4096, 4096) for h in (0, 1)]
        cbs.append(cb(CB["CB_Z"], 4096, 4096))
        kernels = []
        head = [t.buffer_address() for t in (self.values, self.offsets, self.meta, bt)]
        tail = [meta_pages, int(b is not None), len(zeros), zt[0].buffer_address(), zt[1].buffer_address()]
        for half, cfg in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor())):
            ra = ttnn.RuntimeArgs()
            for k in range(plan.nc):
                ra[k % g.x][k // g.x] = head + [k, half] + tail
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=ct,
                    defines=defs,
                    runtime_args=ra,
                    config=cfg,
                )
            )
        self.prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        self.io = [self.meta, self.values, self.offsets] + list(zeros) + ([b] if b is not None else [])

    def __call__(self):
        return ttnn.generic_op(self.io, self.prog)
