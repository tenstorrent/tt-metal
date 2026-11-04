# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode rotation x' = blockdiag(H_1024) D x for the ternary linears (kernels/rot_*.cpp): one 1024-block per core,
H_1024 = H_32 (x) H_32 so each block is two 32x32 matmuls. In: x [.., 32, K] bf16 TILE (row 0 = the token). Out: C
row-major copies of x' [C, K] bf16 in DRAM, one per bank, as the ternary decode linear reads them."""

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import hadamard

KERNELS = "models/demos/blackhole/bonsai2/tt/ternary/kernels/"


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
        # sign masks: bf16 sign bit (0x8000) where D = -1, block b as 512 uint32 (two bf16 each, low half first)
        bits = (signs < 0).astype(np.uint32).reshape(self.B, 512, 2)
        masks = (bits[..., 0] << 15) | (bits[..., 1] << 31)
        self.m = ttnn.from_torch(
            torch.from_numpy(masks.astype(np.uint32).view(np.int32)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.device = device

    def __call__(self, x: ttnn.Tensor, debug_copy=False, u=None) -> ttnn.Tensor:
        """u: rotate silu(x) * u instead of x (the MLP's gate/up product, one op instead of a mul and a rotation)."""
        device, K, B, C = self.device, self.K, self.B, self.copies
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([C, K]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        grid = device.compute_with_storage_grid_size()
        coords = [ttnn.CoreCoord(b % grid.x, b // grid.x) for b in range(B)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])

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
                    defines=([("ROT_GLU", "1")] if u is not None else []) + dm_def,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "rot_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=[C, K * 2] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    runtime_args=wr,
                    common_runtime_args=wr_c,
                    defines=dm_def,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "rot_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[],
                    runtime_args=[],
                    defines=([("ROT_DEBUG_COPY", "1")] if debug_copy else [])
                    + ([("ROT_GLU", "1")] if u is not None else []),
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                    ),
                ),
            ],
            semaphores=[],
            cbs=[
                cb(0, ttnn.bfloat16, 2048, 1),
                cb(1, ttnn.uint32, 2048, 1),
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


class NormRotation:
    """RMSNorm + rotation in one op (kernels/nrot_*.cpp) for a decode activation of width K: returns the normalised
    activation (bf16 TILE in L1, row 0 = the token; the a/b projection and the shared attention code read it) and the
    C rotated copies x' the ternary linears read. Replaces the sharded norm's 3 ops and the rotation op.
    Not enabled (BONSAI_NORM_ROT=1 turns it on): correct (norm 0.18% from torch fp32), but no faster on the P150. The
    op takes 19.8 us against 13.4 (rotation) + ~8 (norm), and traced decode measured 35.89 / 35.91 ms with it vs
    35.79 / 35.81 without (alternating runs)."""

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
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([C, K]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        xn = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(x.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        grid = device.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(b % grid.x, b // grid.x), ttnn.CoreCoord(b % grid.x, b // grid.x))
                for b in range(B)
            ]
        )

        def cb(index, fmt, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        bf, f32 = ttnn.bfloat16, ttnn.float32
        rd_ct = (
            ttnn.TensorAccessorArgs(x).get_compile_time_args()
            + ttnn.TensorAccessorArgs(self.consts).get_compile_time_args()
        )
        wr_ct = (
            [C, K * 2]
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + ttnn.TensorAccessorArgs(xn).get_compile_time_args()
        )
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "nrot_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    common_runtime_args=[x.buffer_address(), self.consts.buffer_address(), grid.x, B],
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "nrot_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=wr_ct,
                    common_runtime_args=[out.buffer_address(), xn.buffer_address(), grid.x],
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
            semaphores=[],
            cbs=[
                cb(0, bf, 2048, 1),
                cb(2, f32, 4096, 6),
                cb(3, bf, 2048, 2 * B),
                cb(24, bf, 2048, 1),
                cb(25, f32, 4096, 2 * B),
                cb(26, f32, 4096, 1),
                cb(27, f32, 4096, 1),
                cb(28, f32, 4096, 1),
                cb(29, f32, 4096, 1),
                cb(16, bf, 2048, 1),
                cb(17, bf, 2048, 1),
            ],
        )
        ttnn.generic_op([x, self.consts, out, xn], program)
        return xn, out


def _f32_bits(v):
    import struct

    return struct.unpack("<I", struct.pack("<f", float(v)))[0]
