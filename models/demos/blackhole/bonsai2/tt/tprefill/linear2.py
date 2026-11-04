# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""2D exact ternary prefill linear: Y = X' W'^T in Bonsai's own ternary weights, as linear.py, laid out for large K.

Each core row is a row group (a slice of the token row tiles); its 13 cores split the output tile columns evenly (GLU: in
(gate, up) pairs). The row's first core multicasts the group's x' tiles per kb to the row only
(kernels/t2_reader.cpp), so each core receives 1/10 of x' (linear.py multicasts all of it to every core); each core
streams its columns' ternary tiles per kb from the decode records (kernels/t2_writer.cpp) and keeps fp32 accumulators in
L1 (packer L1 accumulation), so it can hold any number of columns (kernels/t2_compute.cpp). J = 4 kbs share one DEST
pass per column (scales.ratio_chain), so the packer and the DEST handshake run once per 4 kbs."""

import os

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import KBLOCK, TILE, pack_records_inline
from models.demos.blackhole.bonsai2.tt.tprefill.scales import pack_ratio_group, ratio_chain, scales_from_records

KERNELS = "models/demos/blackhole/bonsai2/tt/tprefill/kernels/"
L1_BUDGET = 1100 * 1024  # L1 for the x' slots, the fp32 accumulators and the weight / ratio slots of a column block


def _chain(KB):
    """J: kbs per DEST pass (and per x' chunk), BONSAI_T2_KPC (default 4), halved until it divides KB."""
    j = int(os.environ.get("BONSAI_T2_KPC", "4"))
    while KB % j:
        j //= 2
    return j


def _upload(device, a, dtype=ttnn.uint32):
    return ttnn.from_torch(
        torch.from_numpy(a.view(np.int32 if dtype == ttnn.uint32 else np.int16)),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class TernaryPrefillLinear2:
    def __init__(self, device, rec, ppr, K, N, s_nb, glu=False):
        """rec: the decode records on the device; s_nb [N, K / 128] float16: their scales (for the ratio tiles)."""
        self.device, self.rec, self.ppr, self.K, self.N, self.glu = device, rec, ppr, K, N, glu
        self.J = _chain(K // KBLOCK)
        self.sbc = _upload(device, pack_ratio_group(ratio_chain(s_nb, self.J), self.J), ttnn.uint16)

    @classmethod
    def from_host(cls, device, t_kn, s_nb):
        rec, ppr = pack_records_inline(t_kn, s_nb)
        return cls(device, _upload(device, rec), ppr, *t_kn.shape, s_nb)

    @classmethod
    def from_records(cls, device, rec_dev, rec_host, ppr, K, N, glu=False):
        """Share decode records already on the device (rec_host: the same records on the host, for the scales)."""
        return cls(device, rec_dev, ppr, K, N, scales_from_records(rec_host, ppr, K, N), glu)

    @classmethod
    def from_decode(cls, lin, rec_host):
        """Share a TernaryDecodeLinear3's records (inline group16 scales; GLU pairs too); rec_host: the same records on
        the host."""
        assert lin.inline_scales and lin.group16
        glu = getattr(lin, "glu", False)
        return cls.from_records(lin.w.device(), lin.w, rec_host, lin.ppr, lin.K, lin.N, glu=glu)

    def __call__(self, x, split=None, residual=None):
        """x: [..., T, K] bf16 TILE (T a multiple of 32) -> [..., T, N (or N / 2 with glu)] bf16 TILE.
        split = (na, ha, hb) (tile columns): two outputs instead, columns < na and the rest, each head-major
        [1, heads, T, h * 32] when its head size h (ha / hb) > 0, else token-major [..., T, columns].
        residual: a tensor of the output's shape added to it (out = residual + x W'^T, one bf16 rounding)."""
        device, K, N, glu = self.device, self.K, self.N, self.glu
        lead, T = list(x.shape)[:-2], list(x.shape)[-2]
        assert list(x.shape)[-1] == K and T % TILE == 0, (x.shape, K)
        Kt, Nt, KB, Mt = K // TILE, N // TILE, K // KBLOCK, T // TILE
        Nt_out = Nt // 2 if glu else Nt
        kpc = self.J  # kbs per DEST pass and per x' multicast chunk
        na, ha, hb = split if split is not None else (Nt_out, 0, 0)
        assert not (glu and split), "GLU outputs are not split"
        assert residual is None or not (glu or split), "a residual goes with a plain output"

        def part(cols, h):
            shape = [1, cols // h, T, h * TILE] if h else lead + [T, cols * TILE]
            return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

        out = part(na, ha)
        out_b = part(Nt_out - na, hb) if split is not None else out
        grid = device.compute_with_storage_grid_size()
        GX, GY = grid.x, grid.y
        mb, me = divmod(Mt, GY)
        m_of = [mb + (1 if y < me else 0) for y in range(GY)]
        m_first = np.cumsum([0] + m_of[:-1]).tolist()
        # columns split as evenly as possible (GLU: in (gate, up) pairs); a core's columns may start mid-unit
        step = 2 if glu else 1
        ub, ue = divmod(Nt // step, GX)
        n_of = [step * (ub + (1 if xx < ue else 0)) for xx in range(GX)]
        n_first = np.cumsum([0] + n_of[:-1]).tolist()
        m_max = max(m_of)
        assert m_max <= 7, f"T={T}: {m_max} row tiles per row group (DEST holds 7 products and the scale tile)"
        xbytes = 2 * m_max * 4 * kpc * 2048  # two x' slots (t2_reader.cpp)
        per_col = m_max * 4096 + 2 * kpc * 4 * 256 + 2 * 2048  # accumulators, weight and ratio slots
        NBLK = max(2, min(max(n_of), (L1_BUDGET - xbytes) // per_col) // step * step)
        nblk = -(-max(n_of) // NBLK)  # every core of a row takes every chunk (rows share the column split)
        vx = [device.worker_core_from_logical_core(ttnn.CoreCoord(xx, 0)).x for xx in range(GX)]
        runs, start = [], 0
        for i in range(1, GX + 1):
            if i == GX or vx[i] != vx[i - 1] + 1:
                runs.append((start, i - 1))
                start = i
        rows = sum(1 for mm in m_of if mm)  # active core rows (m_of is front-loaded)
        vys = [device.worker_core_from_logical_core(ttnn.CoreCoord(0, yy)).y for yy in range(GY)]
        assert all(vys[i + 1] == vys[i] + 1 for i in range(GY - 1)), vys  # a column multicast is one rectangle
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for y in range(GY):
            vy = device.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y
            s0 = device.worker_core_from_logical_core(ttnn.CoreCoord(0, y))
            rects = []
            for a, b in runs:
                rects += [vx[a], vy, vx[b], vy, (b - a + 1) - (1 if a == 0 else 0)]
            for xx in range(GX):
                rd[xx][y] = [1 if xx == 0 else 0, s0.x, s0.y, m_first[y], m_of[y], nblk, GX, len(runs)] + rects
                wr[xx][y] = [n_first[xx], n_of[xx], m_first[y], m_of[y], 1 if y == 0 else 0, rows - 1] + [
                    vx[xx],
                    vys[0],
                    vx[xx],
                    vys[min(1, rows - 1)],
                    vys[rows - 1],
                ]
                cp[xx][y] = [n_of[xx], m_of[y], nblk, m_max * NBLK]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])

        def cb(index, fmt, page, pages, cores=crs):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        # x' chunks: fixed slots of m_max rows x 4 kpc tiles (every row group's slots line up in the one CB descriptor;
        # separate descriptors per row group gave wrong results), two per ring
        pages = 2 * m_max * 4 * kpc
        x_cbs = [cb(0, ttnn.bfloat16, 2048, pages)]
        dm = [("GLU_PAIRS", "1")] if glu else []
        dbg = [(d, "1") for d in os.environ.get("BONSAI_T2_DEFINES", "").split(",") if d]
        rdef = [("RESIDUAL", "1")] if residual is not None else []
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "t2_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=[kpc] + ttnn.TensorAccessorArgs(x).get_compile_time_args(),
                    runtime_args=rd,
                    defines=dbg,
                    common_runtime_args=[x.buffer_address(), Kt, m_max],
                    config=ttnn.WriterConfigDescriptor(),  # BRISC, NOC 0
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "t2_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=[NBLK, kpc]
                    + ttnn.TensorAccessorArgs(self.rec).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(self.sbc).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(out).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(out_b).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(residual if residual is not None else out).get_compile_time_args(),
                    defines=dm + dbg + rdef,
                    runtime_args=wr,
                    common_runtime_args=[
                        self.rec.buffer_address(),
                        self.ppr,
                        self.sbc.buffer_address(),
                        out.buffer_address(),
                        Nt_out,
                        KB,
                        out_b.buffer_address(),
                        na,
                        ha,
                        hb,
                        Mt,
                        (residual if residual is not None else out).buffer_address(),
                    ],
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "t2_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[NBLK, kpc],
                    defines=dm + dbg + rdef,
                    runtime_args=cp,
                    common_runtime_args=[KB],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("BONSAI_T2_FIDELITY", "LoFi")),
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=True,
                    ),
                ),
            ],
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0) for i in range(4)],
            cbs=x_cbs
            + [
                cb(1, ttnn.bfloat8_b, 256, 2 * kpc * NBLK * 4),
                cb(3, ttnn.float32, 2048, 2 * NBLK),
                cb(24, ttnn.float32, 4096, m_max * NBLK),
                cb(16, ttnn.bfloat16, 2048, 4),
                cb(4, ttnn.bfloat16, 2048, 4),
            ],
        )
        if split is None:
            # generic_op returns the last tensor of the list: the output goes last
            return ttnn.generic_op(
                [x, self.rec, self.sbc] + ([residual] if residual is not None else []) + [out], program
            )
        ttnn.generic_op([x, self.rec, self.sbc, out, out_b], program)
        return out, out_b
