# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill rotation x' = blockdiag(H_1024) D x over the last dimension (what the ternary weights expect, as
ternary/rotate.DecodeRotation for one token): D a sign per input, H the normalized 1024-point Sylvester Hadamard
(entries +-1/32, exact in bf16). Per 1024-wide block one matmul at HiFi3 (call_ttnn; the kernel factors the block's transform, kernels/hrot_compute.cpp): H (SrcA) needs 1 bit and HiFi3 covers all
of the bf16 activation's mantissa (SrcB) with SrcA's high bits; fp32 accumulation, bf16 out."""

import os

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.core_ranges import row_major_crs
from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import hadamard

BLOCK = 1024


KERNELS = "models/demos/blackhole/bonsai2/tt/tprefill/kernels/"


class PrefillRotation:
    def __init__(self, device, signs: np.ndarray):
        K = signs.size
        assert K % BLOCK == 0, K
        self.K, self.nb = K, K // BLOCK
        self.signs = ttnn.from_torch(
            torch.from_numpy(np.asarray(signs, dtype=np.float32)).reshape(1, 1, 1, K).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.h = ttnn.from_torch(
            torch.from_numpy(hadamard(BLOCK).astype(np.float32)).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True, packer_l1_acc=False
        )

        # weight tiles of the kernel version: per input tile k, +-diag(d_k) H32 / 32 (tiles (k, +), (k, -))
        h32 = torch.from_numpy(hadamard(32).astype(np.float32)) / np.sqrt(32.0)  # +-1/32
        d = torch.from_numpy(np.asarray(signs, dtype=np.float32)).reshape(-1, 32, 1)  # [Kt, 32 (i), 1]
        m = d * h32  # [Kt, 32, 32]: row i scaled by d_i
        tiles = torch.stack([m, -m], 1).reshape(-1, 32)  # [Kt * 2 * 32, 32]
        self.htiles = ttnn.from_torch(tiles.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.device = device

    def __call__(self, x, gate=None):
        """x [..., T, K] bf16 TILE -> x' (same shape), one op (kernels/hrot_*.cpp).
        gate: rotate x * sigmoid(gate) instead, with x the head-major attention output [1, H, T, D] (H D = K) and gate
        token-major [..., T, K] (the shared attention code's concatenate_heads, sigmoid and multiply in one pass)."""
        device, K, nb = self.device, self.K, self.nb
        shape = list(x.shape) if gate is None else list(gate.shape)
        T = shape[-2]
        if gate is not None:
            assert list(x.shape)[1] * list(x.shape)[3] == K and list(x.shape)[2] == T, (x.shape, gate.shape)
        Mt, Kt = T // 32, K // 32
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        grid = device.compute_with_storage_grid_size()
        ncores = grid.x * grid.y
        per_b = max(1, ncores // nb)  # cores per 1024-block
        coords, args = [], []
        for b in range(nb):
            base, extra = divmod(Mt, per_b)
            start = 0
            for i in range(per_b):
                n = base + (1 if i < extra else 0)
                args.append([b, start, n])
                start += n
        coords = [ttnn.CoreCoord(c % grid.x, c // grid.x) for c in range(len(args))]
        crs = row_major_crs(len(coords), grid.x)
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for cc, a in zip(coords, args):
            rd[cc.x][cc.y] = a
            wr[cc.x][cc.y] = a
            cp[cc.x][cc.y] = a

        def cb(index, pages):
            return ttnn.CBDescriptor(
                total_size=2048 * pages,
                core_ranges=crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=2048)
                ],
            )

        gdef = [("GATE", "1")] if gate is not None else []
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "hrot_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=ttnn.TensorAccessorArgs(x).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(self.htiles).get_compile_time_args()
                    + (ttnn.TensorAccessorArgs(gate).get_compile_time_args() if gate is not None else []),
                    runtime_args=rd,
                    defines=gdef,
                    common_runtime_args=[x.buffer_address(), self.htiles.buffer_address(), Kt]
                    + ([gate.buffer_address(), Mt, list(x.shape)[3] // 32] if gate is not None else []),
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "hrot_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    runtime_args=wr,
                    common_runtime_args=[out.buffer_address(), Kt],
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "hrot_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[],
                    runtime_args=cp,
                    defines=([("HROT_DENSE", "1")] if os.environ.get("BONSAI_HROT_DENSE") == "1" else []) + gdef,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True, dst_full_sync_en=True
                    ),
                ),
            ],
            semaphores=[],
            cbs=[cb(0, 64), cb(1, 64), cb(16, 4)] + ([cb(2, 32), cb(3, 64)] if gate is not None else []),
        )
        if gate is not None:
            return ttnn.generic_op([x, self.htiles, gate, out], program)  # the output goes last
        return ttnn.generic_op([x, self.htiles, out], program)

    def call_ttnn(self, x):
        """The same with ttnn matmuls per 1024-block (the first version; slower)."""
        in_shape = list(x.shape)
        xd = ttnn.multiply(x, self.signs)  # rank 4 after the broadcast
        shape = list(xd.shape)
        parts = []
        for b in range(self.nb):
            start = [0] * (len(shape) - 1) + [b * BLOCK]
            end = shape[:-1] + [(b + 1) * BLOCK]
            xb = ttnn.slice(xd, start, end) if self.nb > 1 else xd
            parts.append(ttnn.matmul(xb, self.h, compute_kernel_config=self.cfg))
            if self.nb > 1:
                ttnn.deallocate(xb)
        ttnn.deallocate(xd)
        if self.nb == 1:
            y = parts[0]
        else:
            y = ttnn.concat(parts, dim=-1)
            for p in parts:
                ttnn.deallocate(p)
        return ttnn.reshape(y, in_shape) if len(in_shape) != len(shape) else y
