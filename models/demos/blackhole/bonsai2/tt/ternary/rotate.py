# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode rotation x' = blockdiag(H_1024) D x for the ternary linears (kernels/rot_*.cpp): one 1024-block per core,
H_1024 = H_32 (x) H_32 so each block is two 32x32 matmuls. In: x [.., 32, K] bf16 TILE (row 0 = the token). Out: C
row-major copies of x' [C, K] bf16 in DRAM, one per bank, as the ternary decode linear reads them."""

import os

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.core_ranges import core_xy, row_major_crs
from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import hadamard

KERNELS = "models/demos/blackhole/bonsai2/tt/ternary/kernels/"
# x' as a [K / 1024, 1024] L1 tensor (2-KB pages over the cores' L1, written once) instead of C DRAM copies [C, K]:
# the ternary decode linears read it without queueing behind their own weight streams (BONSAI_XPAGED=0: copies)
XPAGED = os.environ.get("BONSAI_XPAGED", "1") == "1"


def _xr_alloc(device, K, C):
    if XPAGED:
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([K // 1024, 1024]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([C, K]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )


def _u_tm(u, K, common):
    """ROT_U_TM for a token-major u ([1, 1, heads, D], the decode SDPA's output) instead of a [.., K] row."""
    if u is None or list(u.shape)[-1] == K:
        return []
    assert common and list(u.shape)[-2] * list(u.shape)[-1] == K and 1024 % u.shape[-1] == 0, (u.shape, K)
    return [("ROT_U_TM", str(u.shape[-1]))]


class DecodeRotation:
    def __init__(self, device, signs: np.ndarray, copies: int = 8, chunk_src=None):
        """chunk_src[c]: the input tile (32 values) that feeds rotated position chunk c (identity when None)."""
        K = signs.shape[0]
        assert K % 1024 == 0, K
        self.K, self.B, self.copies = K, K // 1024, copies
        self.chunk_src = list(range(K // 32)) if chunk_src is None else [int(c) for c in chunk_src]
        h32 = hadamard(32) * np.sqrt(32)  # +-1
        h = np.stack([h32, h32 / 32.0]).astype(np.float32)  # H32, H32 / 32: product = H_1024 factors, exact in bf16
        self.h = ttnn.from_torch(torch.from_numpy(h), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # signs D as +-1 tiles: block b's [32, 32] (element (jh, jl) = D[1024 b + 32 jh + jl]), tiled like the block
        d = torch.from_numpy(np.where(signs < 0, -1.0, 1.0).astype(np.float32)).reshape(self.B * 32, 32)
        self.m = ttnn.from_torch(d.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.device = device

    def __call__(self, x: ttnn.Tensor, debug_copy=False, u=None, sig=False) -> ttnn.Tensor:
        """u: rotate silu(x) * u instead of x (the MLP's gate/up product, one op instead of a mul and a rotation);
        sig: sigmoid(x) * u (the attention's gated output)."""
        device, K, B, C = self.device, self.K, self.B, self.copies
        out = _xr_alloc(device, K, C)
        grid = device.compute_with_storage_grid_size()
        coords = [core_xy(b, grid.x) for b in range(B)]
        crs = row_major_crs(len(coords), grid.x)

        def cb(index, fmt, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for b, c in enumerate(coords):
            rd[c.x][c.y] = (
                [x.buffer_address(), self.m.buffer_address(), self.h.buffer_address(), b]
                + self.chunk_src[32 * b : 32 * b + 32]
                + ([u.buffer_address()] if u is not None else [])
            )
            wr[c.x][c.y] = [out.buffer_address(), b]
        common = self.chunk_src == list(range(K // 32))
        rd_c, wr_c, dm_def = [], [], []
        if common:
            # One set of args for all cores (kernels/core_span.hpp): block b = core index.
            rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            rd_c = [x.buffer_address(), self.m.buffer_address(), self.h.buffer_address(), grid.x] + (
                [u.buffer_address()] if u is not None else [0]
            )
            wr_c = [out.buffer_address(), grid.x]
            dm_def = [("ROT_COMMON", "1")]
        rd_ct = []
        for t in (x, self.m, self.h) + ((u,) if u is not None else ()):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "rot_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    runtime_args=rd,
                    common_runtime_args=rd_c,
                    defines=([("ROT_GLU", "1")] if u is not None else []) + dm_def + _u_tm(u, K, common),
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "rot_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=[C, 2048 if XPAGED else K * 2]
                    + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    runtime_args=wr,
                    common_runtime_args=wr_c,
                    defines=dm_def + ([("X_PAGED", "1")] if XPAGED else []),
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "rot_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[],
                    runtime_args=[],
                    defines=([("ROT_DEBUG_COPY", "1")] if debug_copy else [])
                    + ([("ROT_GLU", "1")] if u is not None else [])
                    + ([("ROT_SIGMUL", "1")] if sig else []),
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                    ),
                ),
            ],
            semaphores=[],
            cbs=[
                cb(0, ttnn.bfloat16, 2048, 1),
                cb(1, ttnn.bfloat16, 2048, 1),
                cb(28, ttnn.bfloat16, 2048, 1),
                cb(3, ttnn.bfloat16, 4096, 1),
                cb(2, ttnn.bfloat16, 2048, 2),
                cb(24, ttnn.bfloat16, 2048, 1),
                cb(25, ttnn.bfloat16, 2048, 1),
                cb(16, ttnn.bfloat16, 2048, 1),
            ]
            + (
                [cb(4, ttnn.bfloat16, 2048, 1), cb(26, ttnn.bfloat16, 2048, 1), cb(27, ttnn.bfloat16, 2048, 1)]
                if u is not None
                else []
            ),
        )
        return ttnn.generic_op([x, self.m, self.h] + ([u] if u is not None else []) + [out], program)

    # CB indices of the fused form (off the ternary decode linear's c_0, c_1, c_2, c_3, c_6, c_16)
    FUSED_CBS = {"ROT_CB_X": 8, "ROT_CB_M": 9, "ROT_CB_H": 10, "ROT_CB_SCR": 11, "ROT_CB_U": 12, "ROT_CB_Y": 17}

    def fused_parts(self, x, u, sig, first: int, grid_x: int, hub_xy, xl1_addr: int, spare: int = None):
        """The rotation inside a ternary decode linear's program (decode_linear3.TernaryDecodeLinear3.rotated): on cores
        first .. first + B - 1 (row-major, grid_x wide); each writes its block into the hub's (the linear's core 0,
        NOC hub_xy) x' buffer and bumps its counter (rot_writer.cpp ROT_TO_HUB). -> (kernels, cbs, input tensors)."""
        from models.demos.blackhole.bonsai2.tt.core_ranges import row_major_range_crs

        B, K = self.B, self.K
        crs = row_major_range_crs(first, first + B, grid_x)
        coords = [core_xy(c, grid_x) for c in range(first, first + B)]
        cbm = self.FUSED_CBS

        def cb(index, fmt, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        defs = [(k, str(v)) for k, v in cbm.items()] + [("ROT_FIRST", str(first))]
        common = self.chunk_src == list(range(K // 32))
        rd, rd_c = ttnn.RuntimeArgs(), []
        if common:
            rd_c = [x.buffer_address(), self.m.buffer_address(), self.h.buffer_address(), grid_x] + (
                [u.buffer_address()] if u is not None else [0]
            )
            defs_rd = defs + [("ROT_COMMON", "1")]
        else:
            for b, c in enumerate(coords):
                rd[c.x][c.y] = (
                    [x.buffer_address(), self.m.buffer_address(), self.h.buffer_address(), b]
                    + self.chunk_src[32 * b : 32 * b + 32]
                    + ([u.buffer_address()] if u is not None else [])
                )
            defs_rd = defs
        rd_ct = []
        for t in (x, self.m, self.h) + ((u,) if u is not None else ()):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        glu = [("ROT_GLU", "1")] if u is not None else []
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KERNELS + "rot_reader.cpp",
                core_ranges=crs,
                compile_time_args=rd_ct,
                runtime_args=rd,
                common_runtime_args=rd_c,
                defines=glu + defs_rd + _u_tm(u, K, common),
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNELS + "rot_writer.cpp",
                core_ranges=crs,
                compile_time_args=[],
                common_runtime_args=[hub_xy[0], hub_xy[1], xl1_addr, xl1_addr + 2 * K + 16, grid_x],
                defines=defs + [("ROT_TO_HUB", "1")],
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNELS + "rot_compute.cpp",
                core_ranges=crs,
                compile_time_args=[],
                defines=glu + ([("ROT_SIGMUL", "1")] if sig else []) + defs,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        bf = ttnn.bfloat16
        cbs = [
            cb(cbm["ROT_CB_X"], bf, 2048, 1),
            cb(cbm["ROT_CB_M"], bf, 2048, 1),
            cb(28, bf, 2048, 1),
            cb(cbm["ROT_CB_SCR"], bf, 4096, 1),
            cb(cbm["ROT_CB_H"], bf, 2048, 2),
            cb(24, bf, 2048, 1),
            cb(25, bf, 2048, 1),
            cb(cbm["ROT_CB_Y"], bf, 2048, 1),
        ] + ([cb(cbm["ROT_CB_U"], bf, 2048, 1), cb(26, bf, 2048, 1), cb(27, bf, 2048, 1)] if u is not None else [])
        return kernels, cbs, [x, self.m, self.h] + ([u] if u is not None else []), []


class NormRotation:
    """RMSNorm + rotation in one op (kernels/nrot_*.cpp) for a decode activation of width K: returns the normalised
    activation (bf16 TILE in L1, row 0 = the token; the a/b projection and the shared attention code read it) and the
    C rotated copies x' the ternary linears read. Replaces the sharded norm's 3 ops and the rotation op.
    On by default (BONSAI_NORM_ROT=0 turns it off): correct (norm 0.18% from torch fp32). Once the rotation and norm
    ops stopped paying per-core launches (core_ranges.py), traced decode went 29.91 -> 29.47 ms with it; before, it
    measured no faster (35.89 / 35.91 vs 35.79 / 35.81)."""

    def __init__(self, device, signs: np.ndarray, norm_w, eps, copies: int = 8):
        K = signs.shape[0]
        assert K % 1024 == 0, K
        self.K, self.B, self.copies, self.eps = K, K // 1024, copies, eps
        w = torch.as_tensor(norm_w, dtype=torch.float32).reshape(-1)[:K]
        d = torch.from_numpy(np.where(signs < 0, -1.0, 1.0).astype(np.float32))
        h32 = torch.from_numpy((hadamard(32) * np.sqrt(32)).astype(np.float32))
        blocks = lambda v: [v[1024 * b : 1024 * b + 1024].reshape(32, 32) for b in range(self.B)]
        # OM: the raw pages (32 rows of 32 values: 16 of face row 0, 16 of face row 1) are unpacked as tiles, so mark
        # raw element i (i % 32 < 16 = a token value) at the tile position that holds raw element i.
        om = torch.zeros(32, 32)
        for r in range(32):
            for c in range(32):
                i = ((r // 16) * 2 + c // 16) * 256 + (r % 16) * 16 + (c % 16)
                om[r, c] = 1.0 if i % 32 < 16 else 0.0
        tiles = [h32, h32 / 32.0, om] + blocks(w) + blocks(w * d) + [torch.ones(32, 32)]
        self.consts = ttnn.from_torch(torch.cat(tiles, 0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.device = device

    def __call__(self, x):
        device, K, B, C = self.device, self.K, self.B, self.copies
        out = _xr_alloc(device, K, C)
        xn = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(x.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        # the normalised blocks tiled ([B * 32, 32]: block b's element (jh, jl) = xn[1024 b + 32 jh + jl]), for the
        # GDN step's a / b dot products (prerot.put_xnt; gdn_step.py ab_dot)
        xnt = ttnn.allocate_tensor_on_device(
            ttnn.Shape([B * 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        grid = device.compute_with_storage_grid_size()
        crs = row_major_crs(B, grid.x)

        def cb(index, fmt, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        bf, f32 = ttnn.bfloat16, ttnn.float32
        noc_xy = []  # the B cores' NOC coordinates: each sends its sum-of-squares partial to the others
        for c in range(B):
            v = device.worker_core_from_logical_core(core_xy(c, grid.x))
            noc_xy += [v.x, v.y]
        rd_ct = (
            ttnn.TensorAccessorArgs(x).get_compile_time_args()
            + ttnn.TensorAccessorArgs(self.consts).get_compile_time_args()
        )
        wr_ct = (
            [C, 2048 if XPAGED else K * 2]
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + ttnn.TensorAccessorArgs(xn).get_compile_time_args()
            + ttnn.TensorAccessorArgs(xnt).get_compile_time_args()
        )
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "nrot_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    common_runtime_args=[x.buffer_address(), self.consts.buffer_address(), grid.x, B] + noc_xy,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "nrot_writer.cpp",
                    core_ranges=crs,
                    defines=[("X_PAGED", "1")] if XPAGED else [],
                    compile_time_args=wr_ct,
                    common_runtime_args=[out.buffer_address(), xn.buffer_address(), grid.x, xnt.buffer_address()],
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "nrot_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[],
                    common_runtime_args=[grid.x, B, _f32_bits(1.0 / K), _f32_bits(self.eps)],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=True,
                        math_approx_mode=False,
                        dst_full_sync_en=True,  # 8 fp32 DEST tiles: the squares go 8 at a time
                    ),
                ),
            ],
            semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0)],
            cbs=[
                cb(0, bf, 2048, 1),
                cb(2, f32, 4096, 6),
                cb(3, bf, 2048, 2),
                cb(18, f32, 4096, 1),
                cb(19, f32, 4096, B),
                cb(24, bf, 2048, 1),
                cb(25, f32, 4096, 2),
                cb(26, f32, 4096, 1),
                cb(27, f32, 4096, 1),
                cb(28, f32, 4096, 1),
                cb(29, f32, 4096, 1),
                cb(16, bf, 2048, 1),
                cb(17, bf, 2048, 1),
                cb(20, bf, 2048, 1),
            ],
        )
        ttnn.generic_op([x, self.consts, out, xn, xnt], program)
        from models.demos.blackhole.bonsai2.tt.ternary import prerot

        prerot.put_xnt(xn, xnt)
        return xn, out

    def fused_parts(self, x, u, sig, first: int, grid_x: int, hub_xy, xl1_addr: int, spare: int = None):
        """Norm + rotation inside a ternary decode linear's program (decode_linear3.TernaryDecodeLinear3.rotated) on
        the spare cores after the linear's (spare of them: P = min(B, spare) cores, ceil(B / P) blocks each): only x'
        (no xn), each block into the hub's (core 0's) x' buffer (nrot_writer.cpp NR_TO_HUB). u: an xnt tensor
        ([B * 32, 32] bf16) to fill with the normalised blocks tiled (the GDN step's a / b), or None. sig: unused.
        -> (kernels, cbs, input tensors, semaphores)."""
        from models.demos.blackhole.bonsai2.tt.core_ranges import row_major_range_crs

        device, K, B = self.device, self.K, self.B
        P = min(B, spare if spare is not None else B)
        bpc = -(-B // P)
        P = -(-B // bpc)
        crs = row_major_range_crs(first, first + P, grid_x)
        cbm = {"NR_CB_X": 8, "NR_CB_C": 9, "NR_CB_SC": 10, "NR_CB_Y": 11}

        def cb(index, fmt, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        bf, f32 = ttnn.bfloat16, ttnn.float32
        noc_xy = []
        for c in range(first, first + P):
            v = device.worker_core_from_logical_core(core_xy(c, grid_x))
            noc_xy += [v.x, v.y]
        xnt = u
        defs = [(k, str(v)) for k, v in cbm.items()] + [("NR_FIRST", str(first)), ("NR_BPC", str(bpc))]
        defs_x = defs + ([("NR_XNT", "1")] if xnt is not None else [])
        rd_ct = (
            ttnn.TensorAccessorArgs(x).get_compile_time_args()
            + ttnn.TensorAccessorArgs(self.consts).get_compile_time_args()
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KERNELS + "nrot_reader.cpp",
                core_ranges=crs,
                compile_time_args=rd_ct,
                common_runtime_args=[x.buffer_address(), self.consts.buffer_address(), grid_x, B] + noc_xy,
                defines=defs,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNELS + "nrot_writer.cpp",
                core_ranges=crs,
                compile_time_args=ttnn.TensorAccessorArgs(xnt).get_compile_time_args() if xnt is not None else [],
                common_runtime_args=[hub_xy[0], hub_xy[1], xl1_addr, xl1_addr + 2 * K + 16, grid_x, B]
                + ([xnt.buffer_address()] if xnt is not None else [0]),
                defines=defs_x + [("NR_TO_HUB", "1")],
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNELS + "nrot_compute.cpp",
                core_ranges=crs,
                compile_time_args=[],
                common_runtime_args=[grid_x, B, _f32_bits(1.0 / K), _f32_bits(self.eps)],
                defines=defs_x + [("NR_ONLY_X", "1")],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                    dst_full_sync_en=True,
                ),
            ),
        ]
        cbs = [
            cb(cbm["NR_CB_X"], bf, 2048, bpc),
            cb(cbm["NR_CB_C"], f32, 4096, 4 + 2 * bpc),
            cb(cbm["NR_CB_SC"], bf, 2048, 2 * bpc),
            cb(18, f32, 4096, 1),
            cb(19, f32, 4096, P),
            cb(24, bf, 2048, bpc),
            cb(25, f32, 4096, 2 * bpc),
            cb(26, f32, 4096, 1),
            cb(27, f32, 4096, 1),
            cb(28, f32, 4096, 1),
            cb(29, f32, 4096, 1),
            cb(cbm["NR_CB_Y"], bf, 2048, 1),
        ] + ([cb(20, bf, 2048, 1)] if xnt is not None else [])
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0)]
        return kernels, cbs, [x, self.consts] + ([xnt] if xnt is not None else []), sems


def _f32_bits(v):
    import struct

    return struct.unpack("<I", struct.pack("<f", float(v)))[0]
