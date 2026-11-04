# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a `moe_post` / mHC post-mix: X'_j = post_j * (y [+ y2]) + sum_i C[i][j] * X_i as ONE generic_op program.

Replaces, at the end of the FFN sublayer (B rows <= 32): the shared-expert add, mix_streams (stacked path) or the
stream-major post (typecast, multiply, permute, 4 x (2 slices, multiply, add)), and the to_flat reshape + DRAM copy of
the layer output (from_stream_major's slices + concat above 4 rows); after the attention sublayer the same program
replaces mix_streams / the stream-major post and writes the residual in the layout the fused FFN pre reads.

Program: 64 cores (8 x 8), core k owns column tiles {2k, 2k+1}: reads y (+ y2) (bf16), the 8 residual tiles of its
columns (flat page order, or the stacked residual re-laid out row by row) and the T rows of mhc_pre's compact
coefficient tile (row t = post_0..3 | C[i][j] at 4 + 4i + j), expands the 20 coefficients into column-0 tiles, and
computes the 4 output streams with exact fp32 SFPU products / sums (column broadcasts), the mix_streams fp32 arithmetic.
Output: flat page order (the layer's [1,1,T,16384] contract, or the stream-major [1,4,T,4096]) or stacked [T,1,4,4096].
"""

from __future__ import annotations

import os

import torch

import ttnn

KERNEL_DIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "mhc_post")
FP32, BF16 = ttnn.float32, ttnn.bfloat16
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
GRID = 8
X_FLAT, X_STACKED = 0, 1
OUT_FLAT, OUT_STACKED = 0, 1
CB = dict(Y=0, Y2=1, X=2, COEF=3, CROW=4, YS=5, OUT=6, STK=7)
# PACKED decode mode (kernels/mhc_post/*_p.cpp): a core's 4 streams x 2 column tiles x t token rows in one tile
PACKED_ROWS_MAX = 4


def reference_step(y, x, post, comb, y2=None):
    """y [T, 4096], x [T, 4, 4096], post [T, 4], comb [T, 4, 4] (C[i][j]) -> X' [T, 4, 4096] (float64)."""
    ys = y.double() + (y2.double() if y2 is not None else 0.0)
    return post.double().unsqueeze(-1) * ys.unsqueeze(1) + torch.einsum("tij,tid->tjd", comb.double(), x.double())


class MhcPost:
    """Stateless fused post-mix program of one device (no weights): call with the sublayer output(s), the residual and
    mhc_pre's coefficient tile."""

    def __init__(self, mesh_device, *, cores: list | None = None, cb_base: int = 0):
        """cores (megakernel): 64 logical CoreCoords, core k (column tiles {2k, 2k+1}) = cores[k] (default: the 8 x 8
        grid at the origin, k = 8 x + y); cb_base shifts the CB indices (kernels: POST_CB0)."""
        self.device = mesh_device
        self.cb_base = cb_base
        if cores is None:
            cores = [ttnn.CoreCoord(k // GRID, k % GRID) for k in range(GRID * GRID)]
        from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges

        assert len(cores) == GRID * GRID, len(cores)
        self.core_list = list(cores)
        self.cores = merged_ranges([(c.x, c.y) for c in cores])
        self.defines = [("POST_CB0", str(cb_base))] if cb_base else []
        self.last_packed = False
        self.calls = 0

    def consumers_stacked(self) -> dict:
        """For a STACKED output read by mhc_pre's 64 K cores (K core j = stream j // 16, column tiles 8 (j % 16) ..):
        {k: the K cores reading core k's column tiles {2k, 2k+1}}."""
        return {k: [s * 16 + (2 * k) // 8 for s in range(4)] for k in range(GRID * GRID)}

    @staticmethod
    def _validate(y, y2, x, coef, t, x_mode, out_mode, out_shape):
        """The page arithmetic of the kernels: tile layout everywhere; y / y2 = [.., T, 4096] bf16 or fp32 (CB page =
        the full tile); x = the flat ([1,1,T,16384] / [1,4,T,4096]) or stacked ([T,1,4,4096]) fp32 residual; coef =
        mhc_pre's [1,1,32,32] fp32 tile; the output shape matches out_mode."""
        flat = ((1, 1, t, 16384), (1, 4, t, 4096))

        def bad(what):
            raise ValueError(f"mhc_post: {what} (t={t}, x_mode={x_mode}, out_mode={out_mode})")

        for name, v in (("y", y), ("y2", y2)):
            if v is None:
                continue
            shp = tuple(v.shape)
            if (
                v.layout != ttnn.TILE_LAYOUT
                or v.dtype not in (BF16, FP32)
                or shp[-2:] != (t, 4096)
                or any(d != 1 for d in shp[:-2])
            ):
                bad(f"{name} must be a [1,1,T,4096] bf16/fp32 tile tensor, got {shp} {v.dtype} {v.layout}")
        xs = tuple(x.shape)
        if x.layout != ttnn.TILE_LAYOUT or x.dtype != FP32:
            bad(f"residual must be fp32 tile, got {x.dtype} {x.layout}")
        if (
            (x_mode == X_STACKED and xs != (t, 1, 4, 4096))
            or (x_mode == X_FLAT and xs not in flat)
            or x_mode not in (X_FLAT, X_STACKED)
        ):
            bad(f"residual shape {xs}")
        if coef.layout != ttnn.TILE_LAYOUT or coef.dtype != FP32 or tuple(coef.shape) != (1, 1, 32, 32):
            bad(f"coefficient tile must be [1,1,32,32] fp32, got {tuple(coef.shape)} {coef.dtype}")
        if (
            (out_mode == OUT_STACKED and out_shape != (t, 1, 4, 4096))
            or (out_mode == OUT_FLAT and out_shape not in flat)
            or out_mode not in (OUT_FLAT, OUT_STACKED)
        ):
            bad(f"output shape {out_shape}")

    def __call__(self, y, x, coef, *, t: int, x_mode: int, out_mode: int, y2=None, out_shape=None,
                 out_memory_config=L1):  # fmt: skip
        out = self.output(t, out_mode, out_shape, out_memory_config)
        roles, cbs, io = self.parts(y, x, coef, out, t=t, x_mode=x_mode, out_mode=out_mode, y2=y2)
        suffix = "_p" if self.last_packed else ""
        kernels = [
            ttnn.KernelDescriptor(kernel_source=os.path.join(KERNEL_DIR, f"{name}{suffix}.cpp"),
                                  source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                                  compile_time_args=ct, runtime_args=_rt_from(rt), defines=defs, config=cfg)
            for name, (ct, rt, defs, cfg) in roles.items()
        ]  # fmt: skip
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        self.calls += 1
        return out

    def output(self, t: int, out_mode: int, out_shape=None, out_memory_config=L1):
        if out_shape is None:
            out_shape = (t, 1, 4, 4096) if out_mode == OUT_STACKED else (1, 1, t, 16384)
        return ttnn.empty(out_shape, FP32, ttnn.TILE_LAYOUT, self.device, out_memory_config)

    def parts(self, y, x, coef, out, *, t: int, x_mode: int, out_mode: int, y2=None, wait: tuple | None = None,
              signal: tuple | None = None, packed: bool | None = None, fused_v2: bool = False):  # fmt: skip
        """(roles {reader / writer / compute: (compile-time args, {core: runtime args}, defines, config)}, cbs, io).
        Megakernel hand-offs: wait = (sem id, producer count) before y is read; signal = (sem id, {k: [consumer NoC
        (x, y)]}) after core k's last write. packed (default: t <= 4): the PACKED decode kernels (`*_p.cpp`: the
        core's 8 t rows in ONE tile, same arguments, bitwise the same result) and their CB sizes; a megakernel that
        includes the unpacked kernel files passes packed=False. `last_packed` records the choice."""
        if not 1 <= t <= 32 or (x_mode == X_STACKED or out_mode == OUT_STACKED) and t > 4:
            raise ValueError(f"mhc_post: t={t}, x_mode={x_mode}, out_mode={out_mode}")
        self._validate(y, y2, x, coef, t, x_mode, out_mode, tuple(out.shape))
        packed = t <= PACKED_ROWS_MAX if packed is None else packed
        if packed and t > PACKED_ROWS_MAX:
            raise ValueError(f"mhc_post: packed mode needs t <= {PACKED_ROWS_MAX}, got {t}")
        self.last_packed = packed
        # Stage 14c GLM53_MHC_MODE containing "pv2" (per-tile path only): the fused SFPI mix (kernels/mhc_post/
        # mhc_post_v2_sfpu.hpp) over one compact coefficient tile, valid token bands only
        # fused_v2: the caller's compute runs with dst_full_sync_en — the v2 post-mix holds 8 fp32 DST tiles (slots
        # 0..7; half sync has 4): the MkLayer / MkSparseLayer owner composites pass it; standalone MhcPost / MkMoE never
        v2 = (not packed) and fused_v2 and "pv2" in os.environ.get("GLM53_MHC_MODE", "").split(",")
        base_defs = list(self.defines) + ([("MPOST_V2", "1")] if v2 else [])
        # "cp" (with pv2): the coefficient tensor holds mhc_pre's Sinkhorn page in plane layout (mhc_pre MHC_COEF_PLANE)
        if v2 and "cp" in os.environ.get("GLM53_MHC_MODE", "").split(","):
            base_defs.append(("MPOST_COEF_PLANE", "1"))
        # Stage 14b P1-3 option 2 (GLM53_MPOST_P_FMA=1; opt-in until its accuracy row: B=1 numerics change): the PACKED
        # path (t <= 4) of a pv2 composite takes pv2's per-element arithmetic (each stream term ONE SFPMAD,
        # compute_p.cpp), so a narrow width's rows are bitwise the native widths' (P1-3: w1 vs w8 differed by pv2's
        # single rounding against packed's mul-then-add)
        if (
            packed
            and fused_v2
            and "pv2" in os.environ.get("GLM53_MHC_MODE", "").split(",")
            and os.environ.get("GLM53_MPOST_P_FMA", "0") in ("1", "2")
        ):  # 1: the LLK column broadcast per term; 2: the broadcast fused into the SFPU pass (bitwise 1, cheaper)
            base_defs.append(("MPOST_P_FMA", os.environ["GLM53_MPOST_P_FMA"]))
        add = y2 is not None
        y2t = y2 if add else y
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())
        rtd = lambda fn: {(int(c.x), int(c.y)): [int(v) for v in fn(k)] for k, c in enumerate(self.core_list)}
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        rdefs = list(base_defs)
        if wait is not None:
            rdefs += [("MK_POST_WAIT_SEM", str(wait[0])), ("MK_POST_WAIT_COUNT", str(wait[1]))]
        reader = ([x_mode, int(add)] + acc(y) + acc(y2t) + acc(x) + acc(coef),
                  rtd(lambda k: [y.buffer_address(), y2t.buffer_address(), x.buffer_address(), coef.buffer_address(),
                                 2 * k, t]),
                  rdefs, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default))  # fmt: skip
        wdefs, sig = list(base_defs), (lambda k: [])
        if os.environ.get("GLM53_MHC_POST_ROWWRITE") == "1":  # Stage 14 glue trim: flat output, decode rows only
            wdefs.append(("MK_POST_ROWWRITE", "1"))
        if signal is not None:
            wdefs.append(("MK_POST_SIGNAL_SEM", str(signal[0])))
            sig = lambda k: [len(signal[1][k])] + [v for xy in signal[1][k] for v in xy]
        writer = ([out_mode] + acc(out), rtd(lambda k: [out.buffer_address(), 2 * k, t] + sig(k)), wdefs,
                  dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default))  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in (CB["X"], CB["COEF"], CB["YS"]):
            modes[self.cb_base + cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        for cb, tt in ((CB["Y"], y), (CB["Y2"], y2t)):
            if tt.dtype == FP32:  # an fp32 summand through SrcA would be truncated to tf32
                modes[self.cb_base + cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        compute = ([int(add)], rtd(lambda k: [t] if v2 else []), list(base_defs), cc)
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(
            total_size=page * n, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=self.cb_base + i, data_format=dt, page_size=page)],
        )  # fmt: skip
        ypage = 2048 if y.dtype == BF16 else 4096
        y2page = 2048 if y2t.dtype == BF16 else 4096
        if packed:  # reader_p.cpp: packed operand + raw source pages behind it; writer_p.cpp: 2 assembly pages
            cbs = [cb(CB["Y"], y.dtype, ypage, 3), cb(CB["Y2"], y2t.dtype, y2page, 3), cb(CB["X"], FP32, 4096, 12),
                   cb(CB["COEF"], FP32, 4096, 5), cb(CB["CROW"], FP32, 4096, 1), cb(CB["YS"], FP32, 4096, 1),
                   cb(CB["OUT"], FP32, 4096, 1), cb(CB["STK"], FP32, 4096, 2)]  # fmt: skip
        else:
            cbs = [cb(CB["Y"], y.dtype, ypage, 2), cb(CB["Y2"], y2t.dtype, y2page, 2), cb(CB["X"], FP32, 4096, 8),
                   cb(CB["COEF"], FP32, 4096, 1 if v2 else 20), cb(CB["CROW"], FP32, 4096, 1), cb(CB["YS"], FP32, 4096, 1),
                   cb(CB["OUT"], FP32, 4096, 4), cb(CB["STK"], FP32, 4096, 1)]  # fmt: skip
        return dict(reader=reader, writer=writer, compute=compute), cbs, [y, y2t, x, coef, out]


def _rt_from(rtd: dict):
    rt = ttnn.RuntimeArgs()
    for (x, y), args in rtd.items():
        rt[x][y] = args
    return rt
