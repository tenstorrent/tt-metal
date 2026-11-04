# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""2D exact ternary prefill linear: Y = X' W'^T in Bonsai's own ternary weights, as linear.py, laid out for large K.

Each core row is a row group (a slice of the token row tiles); its 13 cores split the output tile columns in units of 8
(scales.py's broadcast tiles). The row's first core multicasts the group's x' tiles per kb to the row only
(kernels/t2_reader.cpp), so each core receives 1/10 of x' (linear.py multicasts all of it to every core); each core
streams its columns' ternary tiles per kb from the decode records (kernels/t2_writer.cpp) and keeps fp32 accumulators in
L1 (packer L1 accumulation), so it can hold any number of columns (kernels/t2_compute.cpp)."""

import math
import os

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import KBLOCK, TILE, pack_records_inline
from models.demos.blackhole.bonsai2.tt.tprefill.scales import pack_scale_bcast, scales_from_records

KERNELS = "models/demos/blackhole/bonsai2/tt/tprefill/kernels/"
UNIT = 8  # output tile columns per broadcast-scale tile
ACC_BUDGET = 460 * 1024  # L1 for the fp32 accumulators of a column block


def _upload(device, a, dtype=ttnn.uint32):
    return ttnn.from_torch(
        torch.from_numpy(a.view(np.int32 if dtype == ttnn.uint32 else np.int16)),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class TernaryPrefillLinear2:
    def __init__(self, device, rec, ppr, K, N, sbc, glu=False):
        """rec: the decode records on the device; sbc: pack_scale_bcast's tiles on the device."""
        self.device, self.rec, self.ppr, self.K, self.N, self.sbc, self.glu = device, rec, ppr, K, N, sbc, glu

    @classmethod
    def from_host(cls, device, t_kn, s_nb):
        rec, ppr = pack_records_inline(t_kn, s_nb)
        return cls(device, _upload(device, rec), ppr, *t_kn.shape, _upload(device, pack_scale_bcast(s_nb), ttnn.uint16))

    @classmethod
    def from_records(cls, device, rec_dev, rec_host, ppr, K, N, glu=False):
        """Share decode records already on the device (rec_host: the same records on the host, for the scales)."""
        sbc = pack_scale_bcast(scales_from_records(rec_host, ppr, K, N))
        return cls(device, rec_dev, ppr, K, N, _upload(device, sbc, ttnn.uint16), glu)

    @classmethod
    def from_decode(cls, lin, rec_host):
        """Share a TernaryDecodeLinear3's records (inline group16 scales; GLU pairs too); rec_host: the same records on
        the host."""
        assert lin.inline_scales and lin.group16
        glu = getattr(lin, "glu", False)
        return cls.from_records(lin.w.device(), lin.w, rec_host, lin.ppr, lin.K, lin.N, glu=glu)

    def __call__(self, x):
        """x: [..., T, K] bf16 TILE (T a multiple of 32) -> [..., T, N (or N / 2 with glu)] bf16 TILE."""
        device, K, N, glu = self.device, self.K, self.N, self.glu
        lead, T = list(x.shape)[:-2], list(x.shape)[-2]
        assert list(x.shape)[-1] == K and T % TILE == 0, (x.shape, K)
        Kt, Nt, KB, Mt = K // TILE, N // TILE, K // KBLOCK, T // TILE
        assert Nt % UNIT == 0, Nt
        Nt_out = Nt // 2 if glu else Nt
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(lead + [T, Nt_out * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device
        )
        grid = device.compute_with_storage_grid_size()
        GX, GY = grid.x, grid.y
        mb, me = divmod(Mt, GY)
        m_of = [mb + (1 if y < me else 0) for y in range(GY)]
        m_first = np.cumsum([0] + m_of[:-1]).tolist()
        ub, ue = divmod(Nt // UNIT, GX)
        u_of = [ub + (1 if xx < ue else 0) for xx in range(GX)]
        n_first = (UNIT * np.cumsum([0] + u_of[:-1])).tolist()
        n_of = [UNIT * u for u in u_of]
        m_max = max(m_of)
        assert m_max <= 7, f"T={T}: {m_max} row tiles per row group (DEST holds 7 products and the scale tile)"
        NBLK = max(UNIT, min(max(n_of), ACC_BUDGET // (m_max * 4096)) // UNIT * UNIT)
        nblk = -(-max(n_of) // NBLK)  # every core of a row takes every chunk (rows share the column split)
        vx = [device.worker_core_from_logical_core(ttnn.CoreCoord(xx, 0)).x for xx in range(GX)]
        runs, start = [], 0
        for i in range(1, GX + 1):
            if i == GX or vx[i] != vx[i - 1] + 1:
                runs.append((start, i - 1))
                start = i
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for y in range(GY):
            vy = device.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y
            s0 = device.worker_core_from_logical_core(ttnn.CoreCoord(0, y))
            rects = []
            for a, b in runs:
                rects += [vx[a], vy, vx[b], vy, (b - a + 1) - (1 if a == 0 else 0)]
            for xx in range(GX):
                rd[xx][y] = [1 if xx == 0 else 0, s0.x, s0.y, m_first[y], m_of[y], nblk, GX, len(runs)] + rects
                wr[xx][y] = [n_first[xx], n_of[xx], m_first[y], m_of[y]]
                cp[xx][y] = [n_of[xx], m_of[y], nblk, m_max * NBLK]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])

        def cb(index, fmt, page, pages, cores=crs):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        # x' chunks (4 m tiles each): one ring for every core, a multiple of every row group's chunk (a chunk never
        # wraps it) holding at least two chunks (separate descriptors per row group gave wrong results)
        pages = 1
        for mm in set(m_of):
            pages = math.lcm(pages, 4 * mm)
        while pages < 8 * m_max:
            pages *= 2
        x_cbs = [cb(0, ttnn.bfloat16, 2048, pages)]
        dm = [("GLU_PAIRS", "1")] if glu else []
        dbg = [(d, "1") for d in os.environ.get("BONSAI_T2_DEFINES", "").split(",") if d]
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "t2_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=ttnn.TensorAccessorArgs(x).get_compile_time_args(),
                    runtime_args=rd,
                    common_runtime_args=[x.buffer_address(), Kt],
                    config=ttnn.WriterConfigDescriptor(),  # BRISC, NOC 0
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "t2_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=[NBLK]
                    + ttnn.TensorAccessorArgs(self.rec).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(self.sbc).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    defines=dm,
                    runtime_args=wr,
                    common_runtime_args=[
                        self.rec.buffer_address(),
                        self.ppr,
                        self.sbc.buffer_address(),
                        out.buffer_address(),
                        Nt_out,
                        KB,
                    ],
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "t2_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[NBLK],
                    defines=dm + dbg,
                    runtime_args=cp,
                    common_runtime_args=[KB],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, dst_full_sync_en=True
                    ),
                ),
            ],
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0) for i in range(2)],
            cbs=x_cbs
            + [
                cb(1, ttnn.bfloat8_b, 256, 2 * NBLK * 4),
                cb(3, ttnn.float32, 2048, 2 * (NBLK // UNIT)),
                cb(24, ttnn.float32, 4096, m_max * NBLK),
                cb(16, ttnn.bfloat16, 2048, 4),
            ],
        )
        return ttnn.generic_op([x, self.rec, self.sbc, out], program)
