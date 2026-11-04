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
    def __init__(self, device, signs: np.ndarray, copies: int = 8):
        K = signs.shape[0]
        assert K % 1024 == 0, K
        self.K, self.B, self.copies = K, K // 1024, copies
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

    def __call__(self, x: ttnn.Tensor, debug_copy=False) -> ttnn.Tensor:
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
            rd[c.x][c.y] = [x.buffer_address(), self.m.buffer_address(), self.h.buffer_address(), b]
            wr[c.x][c.y] = [out.buffer_address(), b]
        rd_ct = []
        for t in (x, self.m, self.h):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "rot_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    runtime_args=rd,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "rot_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=[C, K * 2] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    runtime_args=wr,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "rot_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[],
                    runtime_args=[],
                    defines=[("ROT_DEBUG_COPY", "1")] if debug_copy else [],
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
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
            ],
        )
        return ttnn.generic_op([x, self.m, self.h, out], program)
