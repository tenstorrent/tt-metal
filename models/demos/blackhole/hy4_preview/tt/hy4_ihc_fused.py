# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview iHC (hc_mult 4, NO comb / Sinkhorn): fused JIT kernels, own copy (no edit of deepseek_v41_flash).

pre-kernel  (`Hy4IHCFused.pre`): ONE generic_op on a 4x8 worker rectangle: stream x [T,1,4,D] fp32 -> RMS-scaled mixes projection (hc_fn [8, 4D], hc_scale
folded into the weights) -> pre/post weights (sigmoid, +eps, x2) -> weighted collapse of the 4 streams -> RMS norm * w -> h bf16 [1,1,T,D] (+ optional bf16
row-major [T,1,1,D]) and the pre|post tile pp [1,1,32,32] fp32 (row t = token, columns 0..3 pre, 4..7 post).  T = 1..16 token rows.
Cores exchange partial sums through L1 writes (reader kernel; two all-to-all exchanges: mixes, sum of squares of the collapsed rows).
Flag-gated by the callers (HY4_IHC_FUSED=1 in the layer code, default OFF).
"""
import math
import os
import struct

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ihc_kernels")
NC, GX, GY = 32, 4, 8
f32, bf16 = ttnn.float32, ttnn.bfloat16


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _kernel(name, core_set, ct, rt, cfg, common=None):
    return ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/{name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=common or [],
        config=cfg,
    )


_CST = {}  # (mesh id, T, dim, eps): persistent L1 constants tensor shared by ALL Hy4IHCFused instances
_DBG = {}  # mesh id -> debug/timestamp tensor (HY4_IHC_DBG=1 only)


class Hy4IHCFused:
    def __init__(self, device, fn8, base8, scale2, dim=6144, hc_eps=1e-6, norm_eps=1e-5):
        assert fn8.shape == (8, 4 * dim) and dim % (32 * NC) == 0
        self.device, self.dim, self.hc_eps, self.norm_eps = device, dim, hc_eps, norm_eps
        self.NT = dim // 32
        self.GPC = self.NT // NC
        rep = ttnn.ReplicateTensorToMesh(device)

        def up(t, dtype=f32):
            return ttnn.from_torch(
                t.contiguous(),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        self._up = up
        GPC = self.GPC
        # packed fn tiles Wp_j[c, 8i + m] = fn[m, i*D + 32j + c], bf16 (hc_scale is applied after the RMS scaling: diagonal tile SD)
        fnb = fn8.float().reshape(8, 4, self.NT, 32)  # [m, i, tile, c]
        wp = fnb.permute(2, 3, 1, 0).reshape(self.NT, 32, 32)  # [tile, c, (i, m)]
        self.wdt = ttnn.bfloat8_b if os.environ.get("HY4_IHC_W8", "0") == "1" else ttnn.bfloat16
        self.wt = up(wp.reshape(1, 1, self.NT * 32, 32), self.wdt)
        # per core k: page 2k = base (rows replicated, columns 0..7), page 2k + 1 = hc_scale per column (pre x4, post x4), rows replicated
        b = torch.zeros(NC, 2, 32, 32)
        b[:, 0, :, :8] = base8.float()
        sc = torch.cat([scale2[0].float().repeat(4), scale2[1].float().repeat(4)])
        b[:, 1, :, :8] = sc  # rows replicated: hc_scale per column
        self.base = up(b.reshape(NC * 2, 1, 32, 32))
        self._w4 = {}

    def norm_weight(self, w, T):
        """fp32 norm weight [D] (or [1,1,1,D]) * sqrt(D), rows replicated: [1,1,T,D] fp32 TILE (cached per (id, T))."""
        key = (id(w), T)
        ent = self._w4.get(key)
        if ent is None:
            wt = w.float().reshape(1, 1, 1, self.dim) * math.sqrt(self.dim)
            ent = (w, self._up(wt.repeat(1, 1, T, 1)))
            self._w4[key] = ent
        return ent[1]

    def _consts(self, mesh, T):
        """persistent L1 tensor (one shard of NCST fp32 tiles per core of the 4x8 rectangle, identical content): the constant tiles of the compute kernel."""
        key = (id(mesh), T, self.dim, self.hc_eps)
        c = _CST.get(key)
        if c is None:
            G = (T + 7) // 8
            n = 10 + G
            t = torch.zeros(n, 32, 32)
            t[0, :, 8] = 1.0 / (4 * self.dim)  # ONES8
            t[1, 8, :] = 1.0  # SELSS
            RR = 2 if T <= 2 else (4 if T <= 4 else (8 if T <= 8 else 16))
            for r in range(RR):
                for m in range(32 // RR):
                    t[2, r, r + RR * m] = 1.0  # FL: sums the row blocks of the recv tiles (slots of RR rows)
            for r in range(16):
                t[3, r, r] = t[3, r + 16, r] = 1.0  # FR
            t[4, :16, 0:4] = 1.0  # SCALE: pre x1, post x2
            t[4, :16, 4:8] = 2.0
            t[5, :16, 0:4] = self.hc_eps  # OFF
            for r in range(32):
                t[6, r, 8 * (r & 3) : 8 * (r & 3) + 8] = 1.0  # MASK
            for i in range(4):
                for m in range(8):
                    t[7, 8 * i + m, m] = 1.0  # FOLD
            for g in range(G):
                for tl in range(8):
                    if g * 8 + tl < T:
                        t[8 + g, g * 8 + tl, 4 * tl : 4 * tl + 4] = 1.0  # SELALL(g)
            t[8 + G] = 1.0  # ONES
            for r in range(32):
                t[9 + G, r, r] = 1.0  # IDENT
            host = t.reshape(1, 1, n * 32, 32).repeat(1, 1, NC, 1).reshape(NC * n * 32, 32)
            grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [n * 32, 32], ttnn.ShardOrientation.ROW_MAJOR),
            )
            c = ttnn.from_torch(
                host,
                device=mesh,
                dtype=f32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mc,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            _CST[key] = c
        return c

    def pre(self, x, w4, emit_rm=False, y=None, pp_prev=None, x_out=None):
        """x: packed streams [1,1,4T,D] fp32 TILE (row 4t + i = stream i of token t), w4 from norm_weight -> (h bf16 [1,1,T,D] TILE, h_tok bf16 [T,1,1,D] ROW_MAJOR | None, pp fp32 [1,1,32,32]).
        With y ([1,1,T,D] bf16 TILE, the previous sub-block output) and pp_prev (the pp of the previous pre call of the SAME streams): the identity-comb
        expand x_new = x + post_prev * y is done first, inside this kernel, and x_new is written to x_out (default: a new tensor; pass x for in place);
        the call then returns (h, h_tok, pp, x_new)."""
        _, _, R4, D = (int(v) for v in x.shape)
        T = R4 // 4
        EXPAND = int(y is not None)
        assert (y is None) == (pp_prev is None)
        assert R4 == 4 * T and 1 <= T <= 16 and D == self.dim and x.dtype == f32
        G = (T + 7) // 8
        GPC, NT = self.GPC, self.NT
        mesh = x.device()
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, T, D]), bf16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        rm = None
        if emit_rm:
            rm = ttnn.allocate_tensor_on_device(
                ttnn.Shape([T, 1, 1, D]), bf16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
        rm_t = rm if emit_rm else out
        pp = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 32, 32]), f32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        cores = [(cx, cy) for cy in range(GY) for cx in range(GX)]  # core k = cy * GX + cx
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
        p0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        p1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(GX - 1, GY - 1))
        assert p1.x - p0.x == GX - 1 and p1.y - p0.y == GY - 1
        rt = ttnn.RuntimeArgs()
        for k, (cx, cy) in enumerate(cores):
            rt[cx][cy] = [k]
        NCST = 10 + G
        RR = 2 if T <= 2 else (4 if T <= 4 else (8 if T <= 8 else 16))
        if not EXPAND:
            x_new = x
        elif x_out is not None:
            x_new = x_out
        else:
            x_new = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, R4, D]), f32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
        y_t = y if EXPAND else pp  # dummies keep the accessor structure
        q_t = pp_prev if EXPAND else pp
        cst_t = self._consts(mesh, T)
        (
            X,
            W,
            BASE,
            RECV,
            PART,
            PP,
            A,
            P2,
            SLOT,
            TOT,
            W4,
            CST,
            ONES,
            SQ,
            Z,
            Q,
            E,
            RAW,
            MIX,
            HS,
            SQH,
            RS,
            TMP,
            OUT,
            RMB,
            DBGW,
            SD,
            ZM,
            U,
            XR,
            YC,
            MC,
            INC,
        ) = range(33)
        cbs = (
            [
                _cb(core_set, X, G * GPC, 4096, f32),
                _cb(core_set, W, GPC, 1088 if self.wdt == ttnn.bfloat8_b else 2048, self.wdt),
                _cb(core_set, BASE, 1, 4096, f32),
                _cb(core_set, RECV, RR // 2, 4096, f32),
                _cb(core_set, PART, 1, 4096, f32),
                _cb(core_set, PP, 1, 4096, f32),
                _cb(core_set, A, G, 4096, f32),
                _cb(core_set, P2, 1, 4096, f32),
                _cb(core_set, SLOT, 2, 4096, f32),
                _cb(core_set, TOT, 1, 4096, f32),
                _cb(core_set, W4, GPC, 4096, f32),
                _cb(core_set, SQ, G * GPC, 4096, f32),
                _cb(core_set, Z, G, 4096, f32),
                _cb(core_set, Q, G, 4096, f32),
                _cb(core_set, E, 1, 4096, f32),
                _cb(core_set, RAW, 1, 4096, f32),
                _cb(core_set, MIX, 1, 4096, f32),
                _cb(core_set, HS, GPC, 2048, bf16),
                _cb(core_set, SQH, GPC, 4096, f32),
                _cb(core_set, RS, 1, 4096, f32),
                _cb(core_set, TMP, GPC, 4096, f32),
                _cb(core_set, OUT, GPC, 2048, bf16),
                _cb(core_set, RMB, max(1, (GPC * T * 64 + 4095) // 4096), 4096, bf16),
                _cb(core_set, DBGW, 1, 4096, f32),
                _cb(core_set, SD, 1, 4096, f32),
                _cb(core_set, ZM, G, 4096, f32),
                _cb(core_set, U, 1, 4096, f32),
            ]
            + (
                [
                    _cb(core_set, XR, G * GPC, 4096, f32),
                    _cb(core_set, YC, GPC, 2048, bf16),
                    _cb(core_set, MC, G, 4096, f32),
                ]
                if EXPAND
                else []
            )
            + []
        )
        DBG = int(os.environ.get("HY4_IHC_DBG", "0"))
        DUAL = int(os.environ.get("HY4_IHC_DUAL", "1"))
        dbg = _DBG.get(id(mesh))
        if dbg is None:
            dbg = _DBG[id(mesh)] = ttnn.allocate_tensor_on_device(
                ttnn.Shape([NC, 1, 32, 32]), f32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
        self.dbg = dbg
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CST, cst_t))
        eps_n, eps_2, inv_n = _bits(self.norm_eps), _bits(D * self.norm_eps), _bits(1.0 / (4 * D))
        reader = _kernel(
            "ihc_pre_reader.cpp",
            core_set,
            [X, W, BASE, RECV, PART, PP, A, P2, SLOT, TOT, W4, T, NT, G, GPC, NC, p0.x, p0.y, GX, SD, CST]
            + _acc(x)
            + _acc(self.wt)
            + _acc(self.base)
            + _acc(w4)
            + _acc(pp)
            + _acc(dbg)
            + [int(os.environ.get("HY4_IHC_ABL", "0")), DBG, EXPAND, XR, YC, MC]
            + _acc(y_t)
            + _acc(q_t)
            + [DUAL, 1088 if self.wdt == ttnn.bfloat8_b else 2048],
            rt,
            ttnn.ReaderConfigDescriptor(),
            common=[
                x.buffer_address(),
                self.wt.buffer_address(),
                self.base.buffer_address(),
                w4.buffer_address(),
                pp.buffer_address(),
                dbg.buffer_address(),
                y_t.buffer_address(),
                q_t.buffer_address(),
            ],
        )
        writer = _kernel(
            "ihc_pre_writer.cpp",
            core_set,
            [OUT, RMB, int(emit_rm), T, D * 2, CST, ONES, G, GPC, _bits(self.hc_eps), inv_n]
            + _acc(out)
            + _acc(rm_t)
            + _acc(dbg)
            + [DBG, DBGW, EXPAND, X, NT]
            + _acc(x_new)
            + [PART, RECV, SLOT, NC, p0.x, p0.y, GX, DUAL, TOT],
            rt,
            ttnn.WriterConfigDescriptor(),
            common=[out.buffer_address(), rm_t.buffer_address(), dbg.buffer_address(), x_new.buffer_address()],
        )
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("HY4_IHC_FID", "HiFi4")),
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,
        )
        if EXPAND:  # the exact fp32 SFPU add of the expand needs the stream tiles unpacked straight into dest
            modes = [ttnn.UnpackToDestMode.Default] * 64
            modes[XR] = ttnn.UnpackToDestMode.UnpackToDestFp32
            cfg.unpack_to_dest_mode = modes
        compute = _kernel(
            "ihc_pre_compute.cpp",
            core_set,
            [
                X,
                W,
                BASE,
                RECV,
                PART,
                PP,
                A,
                P2,
                TOT,
                W4,
                CST,
                ONES,
                SQ,
                Z,
                Q,
                E,
                RAW,
                MIX,
                HS,
                SQH,
                RS,
                TMP,
                OUT,
                G,
                GPC,
                eps_n,
                eps_2,
                SD,
                ZM,
                U,
                DBGW,
                DBG,
                int(os.environ.get("HY4_IHC_CABL", "0")),
                EXPAND,
                XR,
                YC,
                MC,
                INC,
                T,
            ],
            ttnn.RuntimeArgs(),
            cfg,
        )
        if os.environ.get("HY4_IHC_NOPCB") == "1":
            cbs = cbs[:2]  # launch-overhead probe (kernels return immediately)
        prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
        prog.custom_program_hash = _hash(
            0x5D1,
            T,
            D,
            eps_n,
            eps_2,
            _bits(self.hc_eps),
            emit_rm,
            int(os.environ.get("HY4_IHC_ABL", "0")),
            EXPAND,
            DUAL,
            tuple(_acc(y_t)),
            tuple(_acc(q_t)),
            tuple(_acc(x_new)),
            DBG,
            int(os.environ.get("HY4_IHC_CABL", "0")),
            os.environ.get("HY4_IHC_FID", "HiFi4"),
            str(self.wdt),
            tuple(_acc(x)),
            tuple(_acc(self.wt)),
            tuple(_acc(self.base)),
            tuple(_acc(w4)),
            tuple(_acc(pp)),
            tuple(_acc(out)),
            tuple(_acc(rm_t)),
        )
        ttnn.generic_op([x, y_t, q_t, self.wt, self.base, w4, cst_t, pp, dbg, x_new, out, rm_t], prog)
        return (out, rm, pp, x_new) if EXPAND else (out, rm, pp)
