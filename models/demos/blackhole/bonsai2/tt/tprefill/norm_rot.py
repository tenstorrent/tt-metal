# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill (residual add +) RMSNorm + rotation in one op (kernels/anr_*.cpp): h = a (+ b), n = h / rms(h) * w and
x' = blockdiag(H_1024) D n, the input the ternary prefill linears take. Replaces the residual add, the norm and the
rotation (three DRAM round trips of the activation) with one pass. Two cores per 32-token row tile (each half of the
1024-wide blocks; they exchange their sums of squares), one when the row tiles outnumber half the cores. Device time at
2048 tokens: ~225 us with n, ~260 us with the add, ~157 us x' only: about 270-320 GB/s of activation traffic.
BONSAI_NROT_P=1 forces one core per row."""

import os
import struct

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.core_ranges import row_major_crs
from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import hadamard

KERNELS = "models/demos/blackhole/bonsai2/tt/tprefill/kernels/"


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def vector_tiles(v: np.ndarray) -> np.ndarray:
    """v [K] -> [K / 256, 32, 32] float32 (logical tiles): per column tile t its 4 SFPU lane vectors (v = 2 fc + par:
    lane (s, i) = v[32 t + 16 fc + 2 i + par]) at SFPU rows 4 (t % 8) + v of tile t / 8 (SFPU row q = face q / 8, face
    rows 4 (k / 2) .. + 3, columns of parity k % 2, k = q % 8; scales.py)."""
    K = v.size
    Kt = K // 32
    out = np.zeros((Kt // 8, 32, 32), dtype=np.float32)
    for t in range(Kt):
        z = t % 8
        for vv in range(4):
            fc, par = vv >> 1, vv & 1
            q = 4 * z + vv
            f, k = q // 8, q % 8
            lanes = v[32 * t + 16 * fc + 2 * np.arange(8) + par]
            for s in range(4):
                out[t // 8, 16 * (f // 2) + 4 * (k >> 1) + s, 16 * (f % 2) + 2 * np.arange(8) + (k & 1)] = lanes
    return out


class PrefillNormRot:
    def __init__(self, device, signs: np.ndarray, norm_w, eps):
        K = signs.size
        assert K % 1024 == 0, K
        w = np.asarray(torch.as_tensor(norm_w, dtype=torch.float32).reshape(-1)[:K])
        d = np.where(np.asarray(signs) < 0, -1.0, 1.0).astype(np.float32)
        h32 = hadamard(32).astype(np.float32) / np.sqrt(32.0)  # +-1/32
        tiles = [np.ones((1, 32, 32), np.float32), h32[None], -h32[None], vector_tiles(w * d), vector_tiles(w)]
        consts = np.concatenate(tiles, 0).reshape(-1, 32)
        self.nconst = consts.shape[0] // 32
        self.consts = ttnn.from_torch(
            torch.from_numpy(consts), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.K, self.eps, self.device = K, eps, device

    def __call__(self, a, b=None, want_n=False):
        """a (+ b) [..., T, K] bf16 TILE -> (h or None, n or None, x'), each [..., T, K] bf16 TILE."""
        device, K = self.device, self.K
        shape = list(a.shape)
        T = shape[-2]
        assert shape[-1] == K and T % 32 == 0, shape
        Mt, Kt = T // 32, K // 32
        grid = device.compute_with_storage_grid_size()
        assert Mt <= grid.x * grid.y, Mt

        def alloc():
            return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

        xr = alloc()
        n = alloc() if want_n else None
        h = alloc() if b is not None else None
        nb = Kt // 32
        P = 2 if 2 * Mt <= grid.x * grid.y and nb >= 2 else 1  # cores per row tile (block halves, sums exchanged)
        P = min(P, int(os.environ.get("BONSAI_NROT_P", "2")))
        coords = [ttnn.CoreCoord(c % grid.x, c // grid.x) for c in range(P * Mt)]
        crs = row_major_crs(len(coords), grid.x)
        rd, rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        half = -(-nb // 2)
        kmax = 32 * (half if P == 2 else nb)
        for i, c in enumerate(coords):
            m, j = divmod(i, P)
            b0, b1 = (0, nb) if P == 1 else ((0, half) if j == 0 else (half, nb))
            rt[c.x][c.y] = [m, b0, b1, P - 1]
            p = device.worker_core_from_logical_core(coords[i ^ 1] if P == 2 else c)
            rd[c.x][c.y] = [m, b0, b1, P - 1, p.x, p.y]
        defines = ([("HAS_ADD", "1")] if b is not None else []) + ([("WRITE_N", "1")] if want_n else [])
        bb, nn, hh = (b if b is not None else a), (n if n is not None else xr), (h if h is not None else xr)

        def cb(index, fmt, pages):
            page = 4096 if fmt == ttnn.float32 else 2048
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        bf, f32 = ttnn.bfloat16, ttnn.float32
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "anr_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=ttnn.TensorAccessorArgs(a).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(bb).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(self.consts).get_compile_time_args(),
                    defines=defines,
                    runtime_args=rd,
                    common_runtime_args=[
                        a.buffer_address(),
                        bb.buffer_address(),
                        self.consts.buffer_address(),
                        Kt,
                        Mt,
                        self.nconst,
                    ],
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "anr_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=ttnn.TensorAccessorArgs(xr).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(nn).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(hh).get_compile_time_args(),
                    defines=defines,
                    runtime_args=rt,
                    common_runtime_args=[xr.buffer_address(), nn.buffer_address(), hh.buffer_address(), Kt, Mt],
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "anr_compute.cpp",
                    core_ranges=crs,
                    defines=defines,
                    runtime_args=rt,
                    common_runtime_args=[Kt, Mt, _f32_bits(1.0 / K), _f32_bits(self.eps)],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("BONSAI_NROT_FIDELITY", "HiFi4")),
                        fp32_dest_acc_en=True,
                        math_approx_mode=False,
                        dst_full_sync_en=True,
                    ),
                ),
            ],
            semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0)],
            cbs=[
                cb(0, bf, 16),
                cb(1, bf, 16),
                cb(2, f32, self.nconst),
                cb(3, bf, kmax),
                cb(4, f32, 8),
                cb(5, f32, 1),
                cb(6, f32, 1),
                cb(7, bf, 32),
                cb(16, bf, 8),
                cb(17, bf, 8),
                cb(18, bf, 8),
                cb(8, f32, 1),
            ],
        )
        ttnn.generic_op([a, bb, self.consts, xr, nn, hh], program)
        return h, n, xr
