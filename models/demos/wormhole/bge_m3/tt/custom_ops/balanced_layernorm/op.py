# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LayerNorm with a residual add (the model's interleaved LN), model-local.

Same result as stock ttnn.layer_norm (HiFi2, fp32 DEST), bitwise: the compute kernel
follows the stock layernorm.cpp sequence. It is faster per tile row: * rstd, * gamma
and + beta stay in DST, so each output tile is packed once instead of three fp32
pack/unpack round trips. B8/S512 on 12x10 (Galaxy): 38.8 us against 55.5 us.

Tile rows are split evenly over the grid (the first R mod N cores take one more).
The default grid is the smallest one that reaches the minimum rows per core; fewer
active cores are faster at the same rows per core (B8 8x8 38.2 us, 12x10 46.2 us).
Inputs: x and residual bf8/bf16 TILE interleaved [B, 1, S, W]; gamma and beta
row-major bf16 [1, 1, W/32, 32], the LayerNorm1D weight layout (norm.py).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import ttnn

KERNEL_ROOT = "models/demos/wormhole/bge_m3/tt/custom_ops/balanced_layernorm/kernels"

CB_A, CB_B, CB_SCALER, CB_EPS, CB_GAMMA, CB_BETA = 0, 1, 2, 3, 4, 5
CB_X, CB_XMM, CB_EX, CB_EX2, CB_EX2PE, CB_OUT = 6, 7, 9, 10, 11, 16
BLOCK = 4  # tiles per compute block with fp32 DEST (the stock value)


def _tile_bytes(dtype) -> int:
    return {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.float32: 4096}[dtype]


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _bf16_bits_hi(x: float) -> int:
    return _f32_bits(x) & 0xFFFF0000


def default_grid(rows: int, grid_x: int, grid_y: int) -> tuple[int, int]:
    """Smallest x*y (x <= grid_x, y <= grid_y) that keeps ceil(rows / (grid_x * grid_y)) rows per core."""
    rows_max = -(-rows // (grid_x * grid_y))
    need = -(-rows // rows_max)
    best = (grid_x, grid_y)
    for y in range(1, grid_y + 1):
        for x in range(1, grid_x + 1):
            if x * y >= need and x * y < best[0] * best[1]:
                best = (x, y)
    return best


@dataclass(frozen=True)
class BalancedLayerNormPlan:
    rows: int  # tile rows R
    num_cores: int

    def core_rows(self, c: int) -> tuple[int, int]:
        """(row_start, num_rows) of core c."""
        base, extra = divmod(self.rows, self.num_cores)
        return c * base + min(c, extra), base + (1 if c < extra else 0)


def bge_balanced_layernorm(
    x: ttnn.Tensor,
    residual: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
    *,
    eps: float,
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
    output_dtype=ttnn.bfloat8_b,
    grid: tuple[int, int] | None = None,
) -> ttnn.Tensor:
    shape = tuple(x.shape)
    if tuple(residual.shape) != shape or shape[-1] % (32 * BLOCK) or (shape[0] * shape[1] * shape[2]) % 32:
        raise ValueError(f"x {shape} and residual {tuple(residual.shape)} must match and be tile aligned")
    if x.layout != ttnn.TILE_LAYOUT or residual.layout != ttnn.TILE_LAYOUT:
        raise ValueError("x and residual must be TILE_LAYOUT")
    for w in (gamma, beta):
        if (
            w.layout != ttnn.ROW_MAJOR_LAYOUT
            or w.dtype != ttnn.bfloat16
            or tuple(w.shape) != (1, 1, shape[-1] // 32, 32)
        ):
            raise ValueError(f"gamma/beta must be row-major bf16 [1, 1, {shape[-1] // 32}, 32], got {tuple(w.shape)}")
    device = x.device()
    if grid is None:
        g = device.compute_with_storage_grid_size()
        grid = default_grid(shape[0] * shape[1] * shape[2] // 32, int(g.x), int(g.y))
    width = shape[-1]
    wt = width // 32
    plan = BalancedLayerNormPlan(rows=shape[0] * shape[1] * shape[2] // 32, num_cores=grid[0] * grid[1])

    output = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), output_dtype, ttnn.TILE_LAYOUT, device, memory_config)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))})

    def cb(idx, tiles, dtype):
        size = _tile_bytes(dtype)
        return ttnn.CBDescriptor(
            total_size=tiles * size,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)],
        )

    f32 = ttnn.float32
    cbs = [
        cb(CB_A, 2 * BLOCK, x.dtype),
        cb(CB_B, 2 * BLOCK, residual.dtype),
        cb(CB_SCALER, 1, ttnn.bfloat16),
        cb(CB_EPS, 1, ttnn.bfloat16),
        cb(CB_GAMMA, wt, ttnn.bfloat16),
        cb(CB_BETA, wt, ttnn.bfloat16),
        cb(CB_X, wt, f32),
        cb(CB_XMM, wt, f32),
        cb(CB_EX, 1, f32),
        cb(CB_EX2, 1, f32),
        cb(CB_EX2PE, 1, f32),
        cb(CB_OUT, 2 * BLOCK, output_dtype),
    ]

    reader_rt, writer_rt, compute_rt = [], [], []
    for c in range(plan.num_cores):
        coord = (c % grid[0], c // grid[0])
        row_start, num_rows = plan.core_rows(c)
        reader_rt.append(
            (
                coord,
                [
                    x.buffer_address(),
                    residual.buffer_address(),
                    gamma.buffer_address(),
                    beta.buffer_address(),
                    row_start,
                    num_rows,
                ],
            )
        )
        writer_rt.append((coord, [output.buffer_address(), row_start, num_rows]))
        compute_rt.append((coord, [num_rows]))

    def accessor(t):
        return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[wt, BLOCK, _bf16_bits_hi(eps)]
        + accessor(x)
        + accessor(residual)
        + accessor(gamma)
        + accessor(beta),
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[wt, BLOCK] + accessor(output),
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[wt, BLOCK, _f32_bits(1.0 / width)],
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        ),
    )
    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([x, residual, gamma, beta, output], descriptor)
    return output
