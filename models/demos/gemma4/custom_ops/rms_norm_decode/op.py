# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``RmsNormDecode``: a multi-core RMSNorm on an interleaved one-row-tile activation, as one program.

``x`` is ``[1, 1, R, W]`` (bf16, TILE, DRAM or L1 interleaved, ``R <= 32`` rows on one row tile, ``W`` a multiple
of 32), ``gamma`` an optional ``[1, 1, 1, W]`` bf16 TILE row (the ``RMSNorm.tt_weight_tile`` form), ``eps`` and an
optional scalar: ``out = x * rsqrt(mean(x^2) + eps) * scalar * gamma`` in ``x``'s memory config.

``N`` cores (a divisor of the ``W/32`` column tiles, default the largest one up to 16) each own a contiguous block
of column tiles: they square their tiles (FPU, fp32 dest), reduce the row sums (FPU row reduce with a unit scaler)
into one fp32 partial tile, exchange the partials over the NoC (every core writes its partial into slot ``i`` of
every core's receive buffer and bumps its semaphore), sum the ``N`` partials, take ``rsqrt(sum / W + eps) *
scalar`` (SFPU), and scale their own tiles by it (column broadcast) and by gamma (row broadcast), one bf16 pack.
This replaces the decode ``to_memory_config (interleaved -> width sharded), rms_norm (sharded program), sharded_to
_interleaved`` triple of ``tt/rms_norm.py`` (3 programs, 2 of them copies) with one program that reads and writes
the interleaved residual directly; for the router it also folds the ``scale`` and ``hidden**-0.5`` multiplies.
"""

import os
import struct
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class RmsNormDecode:
    CB_X = 0
    CB_G = 1
    CB_SCALER = 2
    CB_SQ = 3  # fp32 x^2 tiles
    CB_PART = 4  # fp32 partial row-sum tile (column 0)
    CB_RECV = 5  # fp32 partials of all N cores (slot = sender)
    CB_INV = 6  # fp32 rsqrt column tile
    CB_OUT = 16
    SEM = 0

    @staticmethod
    def golden(x: torch.Tensor, gamma, eps: float, scalar: float = 1.0) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * scalar
        return out * gamma.float() if gamma is not None else out

    @staticmethod
    def num_cores(cols_t: int) -> int:
        cap = max(1, int(os.environ.get("GEMMA4_NORM_FUSED_CORES", "16")))
        for n in range(min(cap, cols_t), 0, -1):
            if cols_t % n == 0:
                return n
        return 1

    @classmethod
    def op(cls, x: ttnn.Tensor, gamma, eps: float, scalar: float = 1.0) -> ttnn.Tensor:
        assert (
            x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16 and not x.is_sharded()
        ), "x: bf16 TILE interleaved"
        shape = [int(v) for v in x.shape]
        assert (
            len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and shape[2] <= TILE
        ), f"x must be [1,1,<=32,W], got {shape}"
        W = shape[3]
        assert W % TILE == 0 and x.padded_shape[-2] == TILE, f"width {W} must be a multiple of 32 on one row tile"
        has_gamma = gamma is not None
        if has_gamma:
            assert gamma.layout == ttnn.TILE_LAYOUT and tuple(int(v) for v in gamma.shape) == (1, 1, 1, W), gamma.shape
        cols_t = W // TILE
        N = cls.num_cores(cols_t)
        per = cols_t // N
        device = x.device()
        out = ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, device, x.memory_config())
        grid = device.compute_with_storage_grid_size()
        assert N <= grid.x * grid.y
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(N)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        noc = [device.worker_core_from_logical_core(c) for c in cores]
        tile_bytes = TILE * TILE * 2
        f32_tile = TILE * TILE * 4

        def cb(idx, pages, page=tile_bytes, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
            )

        cbs = [
            cb(cls.CB_X, per),
            cb(cls.CB_G, per if has_gamma else 1),
            cb(cls.CB_SCALER, 1),
            cb(cls.CB_SQ, per, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_PART, 1, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_RECV, N, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_INV, 1, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_OUT, per),
        ]
        gamma_t = gamma if has_gamma else x
        reader_ct = [cls.CB_X, cls.CB_G, cls.CB_SCALER, cls.CB_RECV, per, N, int(has_gamma), cls.SEM]
        reader_ct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(gamma_t).get_compile_time_args()
        writer_ct = [cls.CB_PART, cls.CB_RECV, cls.CB_OUT, per, N, cls.SEM] + ttnn.TensorAccessorArgs(
            out
        ).get_compile_time_args()
        compute_ct = [
            cls.CB_X, cls.CB_G, cls.CB_SCALER, cls.CB_SQ, cls.CB_PART, cls.CB_RECV, cls.CB_INV, cls.CB_OUT,
            per, N, int(has_gamma), _f32_bits(eps), _f32_bits(1.0 / W), _f32_bits(scalar),
        ]  # fmt: skip
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        coords = []
        for c in noc:
            coords += [c.x, c.y]
        for i, core in enumerate(cores):
            reader_rt[core.x][core.y] = [x.buffer_address(), gamma_t.buffer_address(), i * per]
            writer_rt[core.x][core.y] = [out.buffer_address(), i * per, i] + coords
            compute_rt[core.x][core.y] = [0]
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_rms_norm_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_rms_norm_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_rms_norm_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
        ]
        sems = [ttnn.SemaphoreDescriptor(cls.SEM, ttnn.CoreType.WORKER, core_set, 0)]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        io = [x, gamma, out] if has_gamma else [x, out]
        return ttnn.generic_op(io, program)
