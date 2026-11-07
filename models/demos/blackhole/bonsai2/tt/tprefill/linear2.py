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
        # row groups (each multicasts its x' along its line of cores) and column groups (each multicasts its weights
        # along its line): BONSAI_T2_TRANSPOSE=1 (default) puts the row groups on the 13 core columns and the column
        # groups on the 10 core rows, else (default) the other way round (fewer, larger matmul calls per row group)
        transpose = os.environ.get("BONSAI_T2_TRANSPOSE", "0") == "1"
        NR, NC = (GX, GY) if transpose else (GY, GX)
        # row tiles per row group as even as possible; a group with more than 7 (DEST holds 7 products and the scale
        # tile) takes them in rounds of near-equal blocks, one program per round (T = 4096: 13 = 7 + 6)
        mb, me = divmod(Mt, NR)
        m_tot = [mb + (1 if i < me else 0) for i in range(NR)]
        g_first = np.cumsum([0] + m_tot[:-1]).tolist()
        n_rounds = -(-max(m_tot) // 7)
        blocks = [[mm // n_rounds + (1 if k < mm % n_rounds else 0) for k in range(n_rounds)] for mm in m_tot]
        rounds = [
            ([g_first[i] + sum(blocks[i][:k]) for i in range(NR)], [blocks[i][k] for i in range(NR)])
            for k in range(n_rounds)
        ]
        # columns split as evenly as possible (GLU: in (gate, up) pairs); a core's columns may start mid-unit
        step = 2 if glu else 1
        ub, ue = divmod(Nt // step, NC)
        n_of = [step * (ub + (1 if i < ue else 0)) for i in range(NC)]
        n_first = np.cumsum([0] + n_of[:-1]).tolist()
        m_max = max(max(mo) for _, mo in rounds)
        assert m_max <= 7, m_max
        xdt = x.dtype
        xtb = 1088 if xdt == ttnn.bfloat8_b else 2048  # x' tile bytes (bf16 or bfp8)
        xbytes = 2 * m_max * 4 * kpc * xtb  # two x' slots (t2_reader.cpp)
        per_col = m_max * 4096 + 2 * kpc * 4 * 256 + 2 * 2048  # accumulators, weight and ratio slots
        NBLK = max(2, min(max(n_of), (L1_BUDGET - xbytes) // per_col) // step * step)
        nblk = -(-max(n_of) // NBLK)  # every core of a row group takes every chunk (they share its x')

        def core(rg, cg):  # logical (x, y)
            return (rg, cg) if transpose else (cg, rg)

        def virt(rg, cg):
            c = device.worker_core_from_logical_core(ttnn.CoreCoord(*core(rg, cg)))
            return c.x, c.y

        def rects(pts, sender=None):
            """Multicast rectangles over a line of cores (virtual coordinates in line order), split where the
            coordinates jump: [x0, y0, x1, y1, destinations] each, the sender (in a rectangle, not a destination)
            excluded from the counts."""
            out, i = [], 0
            while i < len(pts):
                j = i
                while j + 1 < len(pts) and abs(pts[j + 1][0] - pts[j][0]) + abs(pts[j + 1][1] - pts[j][1]) == 1:
                    j += 1
                nd = (j - i + 1) - (1 if sender is not None and i <= sender <= j else 0)
                out += [*pts[i], *pts[j], nd]
                i = j + 1
            return out

        def run(m_first, m_of):
            rows = sum(1 for mm in m_of if mm)  # active row groups (m_of is front-loaded)
            rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for rg in range(NR):
                xline = [virt(rg, cg) for cg in range(NC)]  # x' multicast from cg = 0
                xr = rects(xline, sender=0)
                for cg in range(NC):
                    wline = [virt(r, cg) for r in range(1, rows)]  # weight multicast from rg = 0 to the active rows
                    wrc = rects(wline)
                    cx, cy = core(rg, cg)
                    rd[cx][cy] = [1 if cg == 0 else 0, *xline[0], m_first[rg], m_of[rg], nblk, NC, len(xr) // 5] + xr
                    wr[cx][cy] = (
                        [n_first[cg], n_of[cg], m_first[rg], m_of[rg], 1 if rg == 0 else 0, rows - 1]
                        + [
                            *virt(0, cg),
                            len(wrc) // 5,
                        ]
                        + wrc
                    )
                    cp[cx][cy] = [n_of[cg], m_of[rg], nblk, m_max * NBLK]
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
            x_cbs = [cb(0, xdt, xtb, pages)]
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
                return ttnn.generic_op(
                    [x, self.rec, self.sbc] + ([residual] if residual is not None else []) + [out], program
                )
            ttnn.generic_op([x, self.rec, self.sbc, out, out_b], program)
            return out, out_b

        for m_first, m_of in rounds:
            res = run(m_first, m_of)  # every round writes its rows of the same outputs
        return res
