# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""2D folded-scale ternary prefill linear with a shared fold: Y = X' W'^T, W'[n, k] = t[k, n] s5[n, k / 128] (s5: the
block scale rounded to 5 significant bits, exact as a LoFi srcA operand).

Layout and x' movement as linear2.py (row groups on core rows, x' multicast along each row from its first core, rounds
of <= 7 token row tiles). Per column block and group of J kbs, the column's weights are nb x J units (column, kb) of 4
ternary tiles. Each active core of a core column folds a contiguous 1/R share of them (whole unit rows: CP columns' units
of one kb, kernels/t6_writer.cpp) on the FPU
(W^T = diag(s5) @ t^T, kernels/t6_compute.cpp) into Bfp8_b tiles (exact: a face row is one output column, so its 16
values share one scale) and multicasts them to the same slot of every core of the column (kernels/t6_writer.cpp), so
each core gets the whole group of folded tiles and runs plain matmuls on them (no SFPU pass). The fold of group g + 1
runs before the matmuls of group g, so the exchange has a group's matmuls to land."""

import os

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import KBLOCK, TILE, pack_records_inline
from models.demos.blackhole.bonsai2.tt.tprefill.scales import scales_from_records

KERNELS = "models/demos/blackhole/bonsai2/tt/tprefill/kernels/"
WTB = 1088  # Bfp8_b folded tile bytes


def _chain(KB):
    """J: kbs per group (per x' chunk and per folded-tile slot), BONSAI_T6_KPC (default 4), halved until it divides KB."""
    j = int(os.environ.get("BONSAI_T6_KPC", "4"))
    while KB % j:
        j //= 2
    return j


def round_sig(s_nb: np.ndarray, bits: int = 5) -> np.ndarray:
    """Scales rounded to `bits` significant bits (round to nearest)."""
    m, e = np.frexp(s_nb.astype(np.float64))
    return np.round(m * 2**bits) / 2**bits * 2.0**e


class TernaryPrefillLinear6:
    def __init__(self, device, rec, ppr, K, N, s_nb, glu=False):
        """rec: the decode records on the device; s_nb [N, K / 128] float16: their scales."""
        self.device, self.rec, self.ppr, self.K, self.N, self.glu = device, rec, ppr, K, N, glu
        self.J = _chain(K // KBLOCK)
        Nt, KB = N // TILE, K // KBLOCK
        # scale rows: page nt * KB + kb = the 32 bf16 scales of output tile column nt at kb (64 B)
        rows = round_sig(s_nb).astype(np.float32).reshape(Nt, TILE, KB).transpose(0, 2, 1).reshape(Nt * KB, TILE)
        self.srow = ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(rows)).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @classmethod
    def from_host(cls, device, t_kn, s_nb):
        rec, ppr = pack_records_inline(t_kn, s_nb)
        dev = ttnn.from_torch(
            torch.from_numpy(rec.view(np.int32)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return cls(device, dev, ppr, *t_kn.shape, s_nb)

    @classmethod
    def from_records(cls, device, rec_dev, rec_host, ppr, K, N, glu=False):
        return cls(device, rec_dev, ppr, K, N, scales_from_records(rec_host, ppr, K, N), glu)

    @classmethod
    def from_decode(cls, lin, rec_host):
        assert lin.inline_scales and lin.group16
        glu = getattr(lin, "glu", False)
        return cls.from_records(lin.w.device(), lin.w, rec_host, lin.ppr, lin.K, lin.N, glu=glu)

    def __call__(self, x, split=None, residual=None):
        """As TernaryPrefillLinear2.__call__."""
        device, K, N, glu = self.device, self.K, self.N, self.glu
        lead, T = list(x.shape)[:-2], list(x.shape)[-2]
        assert list(x.shape)[-1] == K and T % TILE == 0, (x.shape, K)
        Kt, Nt, KB, Mt = K // TILE, N // TILE, K // KBLOCK, T // TILE
        Nt_out = Nt // 2 if glu else Nt
        kpc = self.J
        KT = 4 * kpc
        na, ha, hb = split if split is not None else (Nt_out, 0, 0)
        assert not (glu and split), "GLU outputs are not split"
        assert residual is None or not (glu or split), "a residual goes with a plain output"

        def part(cols, h):
            shape = [1, cols // h, T, h * TILE] if h else lead + [T, cols * TILE]
            return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

        out = part(na, ha)
        out_b = part(Nt_out - na, hb) if split is not None else out
        grid = device.compute_with_storage_grid_size()
        NR, NC = grid.y, grid.x
        # full-sync DEST (default): 8 fp32 tiles; else half sync, 4 tiles per half (the packer overlaps the math).
        # A matmul pass takes one token row tile and CP output columns (unpacks one x' tile per CP products; at
        # 4096 tokens 1 x 8 beats 7 x 1 by ~15%)
        half = os.environ.get("BONSAI_T6_HALF", "0") == "1"
        CP = int(os.environ.get("BONSAI_T6_CP", "4" if half else "8"))  # output columns per matmul pass
        PASS = int(os.environ.get("BONSAI_T6_RP", "0")) or ((4 if half else 8) // CP - (0 if half or CP > 1 else 1))
        RMAX = int(os.environ.get("BONSAI_T6_ROWS", "7"))  # row tiles per row group and round
        mb, me = divmod(Mt, NR)
        m_tot = [mb + (1 if i < me else 0) for i in range(NR)]
        g_first = np.cumsum([0] + m_tot[:-1]).tolist()
        n_rounds = -(-max(m_tot) // RMAX)
        blocks = [[mm // n_rounds + (1 if k < mm % n_rounds else 0) for k in range(n_rounds)] for mm in m_tot]
        rounds = [
            ([g_first[i] + sum(blocks[i][:k]) for i in range(NR)], [blocks[i][k] for i in range(NR)])
            for k in range(n_rounds)
        ]
        step = 2 if glu else 1
        ub, ue = divmod(Nt // step, NC)
        n_of = [step * (ub + (1 if i < ue else 0)) for i in range(NC)]
        n_first = np.cumsum([0] + n_of[:-1]).tolist()
        m_max = max(max(mo) for _, mo in rounds)
        xdt = x.dtype
        xtb = 1088 if xdt == ttnn.bfloat8_b else 2048
        FD = int(os.environ.get("BONSAI_T6_FD", "2"))  # groups of fold inputs in their rings
        SD = int(os.environ.get("BONSAI_T6_SD", "1"))  # groups of folded share tiles in their ring
        XR = int(os.environ.get("BONSAI_T6_XR", "2"))  # x' slots (groups) in its ring
        WR = int(os.environ.get("BONSAI_T6_WR", "2"))  # folded-tile slots (groups) in their ring
        xbytes = XR * m_max * KT * xtb
        per_col = m_max * 4096 + WR * KT * WTB  # fp32 accumulators, the folded tiles' ring
        rows_min = min(sum(1 for mm in mo if mm) for _, mo in rounds)

        def units(nblk_, rows_):  # fold units per core and group, at most (t6_writer.cpp)
            return -(-(-(-nblk_ // CP)) * kpc // rows_) * CP

        def l1(nblk_):  # every CB's bytes (run())
            um_ = units(nblk_, rows_min)
            return xbytes + nblk_ * per_col + um_ * (FD * (4 * 256 + 64 + 2048) + SD * 4 * WTB) + 2 * 4 * 2048

        # CB bytes: L1 above the firmware / kernel reservation (~161 KB in the model)
        cap = int(os.environ.get("BONSAI_T6_L1", "1400000"))
        NBLK = min(max(n_of), 512) // step * step
        while NBLK > step and l1(NBLK) > cap:
            NBLK -= step
        assert l1(NBLK) <= cap, (l1(NBLK), cap)
        nblk = -(-max(n_of) // NBLK)

        def virt(rg, cg):
            c = device.worker_core_from_logical_core(ttnn.CoreCoord(cg, rg))
            return c.x, c.y

        def rects(pts, sender=None):
            """Multicast rectangles over a line of cores (as linear2.py): [x0, y0, x1, y1, destinations] each."""
            res, i = [], 0
            while i < len(pts):
                j = i
                while j + 1 < len(pts) and abs(pts[j + 1][0] - pts[j][0]) + abs(pts[j + 1][1] - pts[j][1]) == 1:
                    j += 1
                nd = (j - i + 1) - (1 if sender is not None and i <= sender <= j else 0)
                if nd:
                    res += [*pts[i], *pts[j], nd]
                i = j + 1
            return res

        def run(m_first, m_of):
            rows = sum(1 for mm in m_of if mm)  # active row groups (front-loaded)
            um = units(NBLK, rows)
            rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for rg in range(NR):
                xline = [virt(rg, cg) for cg in range(NC)]
                xr = rects(xline, sender=0)
                for cg in range(NC):
                    wline = [virt(r, cg) for r in range(rows)]
                    wrc = rects(wline, sender=rg) if rg < rows else []
                    rd[cg][rg] = [1 if cg == 0 else 0, *xline[0], m_first[rg], m_of[rg], nblk, NC, len(xr) // 5] + xr
                    wr[cg][rg] = (
                        [n_first[cg], n_of[cg], m_first[rg], m_of[rg], rg, rows, *virt(rg, cg)]
                        + [v for p in wline for v in p]
                        + [len(wrc) // 5]
                        + wrc
                    )
                    cp[cg][rg] = [n_of[cg], m_of[rg], nblk, m_max * NBLK, rg, rows]
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NC - 1, NR - 1))])

            def cb(index, fmt, page, pages):
                return ttnn.CBDescriptor(
                    total_size=page * pages,
                    core_ranges=crs,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
                )

            dm = [("GLU_PAIRS", "1")] if glu else []
            dbg = [(d, "1") for d in os.environ.get("BONSAI_T6_DEFINES", "").split(",") if d]
            rdef = [("RESIDUAL", "1")] if residual is not None else []
            ct = [NBLK, kpc, um, PASS, CP]
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
                        kernel_source=KERNELS + "t6_writer.cpp",
                        core_ranges=crs,
                        compile_time_args=ct
                        + ttnn.TensorAccessorArgs(self.rec).get_compile_time_args()
                        + ttnn.TensorAccessorArgs(self.srow).get_compile_time_args()
                        + ttnn.TensorAccessorArgs(out).get_compile_time_args()
                        + ttnn.TensorAccessorArgs(out_b).get_compile_time_args()
                        + ttnn.TensorAccessorArgs(residual if residual is not None else out).get_compile_time_args(),
                        defines=dm + dbg + rdef,
                        runtime_args=wr,
                        common_runtime_args=[
                            self.rec.buffer_address(),
                            self.ppr,
                            self.srow.buffer_address(),
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
                        config=ttnn.ReaderConfigDescriptor(),  # NCRISC, NOC 1
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=KERNELS + "t6_compute.cpp",
                        core_ranges=crs,
                        compile_time_args=ct,
                        defines=dm + dbg + rdef,
                        runtime_args=cp,
                        common_runtime_args=[KB],
                        config=ttnn.ComputeConfigDescriptor(
                            math_fidelity=ttnn.MathFidelity.LoFi,
                            fp32_dest_acc_en=True,
                            dst_full_sync_en=not half,
                        ),
                    ),
                ],
                semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0) for i in range(6)],
                cbs=[
                    cb(0, xdt, xtb, XR * m_max * KT),  # x' chunks (t2_reader.cpp)
                    cb(1, ttnn.bfloat8_b, 256, FD * um * 4),  # this core's share of ternary tiles
                    cb(3, ttnn.bfloat16, 64, FD * um),  # their scale rows
                    cb(6, ttnn.bfloat16, 2048, FD * um),  # their diag(s5) tiles
                    cb(7, ttnn.bfloat8_b, WTB, SD * um * 4),  # this core's folded tiles, to multicast
                    cb(5, ttnn.bfloat8_b, WTB, WR * NBLK * KT),  # every folded tile of a group, two groups
                    cb(24, ttnn.float32, 4096, m_max * NBLK),
                    cb(16, ttnn.bfloat16, 2048, 4),
                    cb(4, ttnn.bfloat16, 2048, 4),
                ],
            )
            if split is None:
                return ttnn.generic_op(
                    [x, self.rec, self.srow] + ([residual] if residual is not None else []) + [out], program
                )
            ttnn.generic_op([x, self.rec, self.srow, out, out_b], program)
            return out, out_b

        for m_first, m_of in rounds:
            res = run(m_first, m_of)
        return res
