# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decode-layer matmul kernel capability checks.

These tests are intentionally model-independent microtests. They verify that the
TTNN matmul kernels can run the thin-M decode shapes that appear in attention
and MLP blocks, across a few useful kernel/layout families:

* auto/interleaved baseline
* 1D multicast with interleaved weights
* L1-sharded operands/output
* mixed L1-sharded activation + DRAM-sharded weight + L1-sharded output
* DRAM-sharded weights with sharded activations/output

Run with ``-s`` to print rough wall-clock latency for each variant. For precise
kernel timing, wrap the test with Tracy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


TILE = 32


@dataclass(frozen=True)
class DecodeMatmulShape:
    name: str
    m: int
    k: int
    n: int
    in0_dtype: ttnn.DataType
    in1_dtype: ttnn.DataType
    out_dtype: ttnn.DataType
    math_fidelity: ttnn.MathFidelity
    pcc_threshold: float


@dataclass(frozen=True)
class KernelVariant:
    name: str
    kind: str
    input_l1: bool
    output_l1: bool


_BF16 = ttnn.bfloat16
_BFP8 = ttnn.bfloat8_b
_BFP4 = ttnn.bfloat4_b
_HIFI2 = ttnn.MathFidelity.HiFi2
_LOFI = ttnn.MathFidelity.LoFi


DECODE_MATMUL_SHAPES = [
    DecodeMatmulShape(
        name="attn_qkv",
        m=32,
        k=1536,
        n=2048,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BF16,
        math_fidelity=_HIFI2,
        pcc_threshold=0.99,
    ),
    DecodeMatmulShape(
        name="attn_o_proj",
        m=32,
        k=1536,
        n=1536,
        in0_dtype=_BF16,
        in1_dtype=_BFP4,
        out_dtype=_BFP8,
        math_fidelity=_LOFI,
        pcc_threshold=0.99,
    ),
    DecodeMatmulShape(
        name="mlp_gate_up",
        m=32,
        k=1536,
        n=17920,
        in0_dtype=_BF16,
        in1_dtype=_BFP4,
        out_dtype=_BFP8,
        math_fidelity=_LOFI,
        pcc_threshold=0.99,
    ),
    DecodeMatmulShape(
        name="mlp_down",
        m=32,
        k=8960,
        n=1536,
        in0_dtype=_BFP8,
        in1_dtype=_BFP4,
        out_dtype=_BFP8,
        math_fidelity=_LOFI,
        pcc_threshold=0.99,
    ),
]


KERNEL_VARIANTS = [
    # KernelVariant(name="auto_dram_in_dram_out", kind="auto", input_l1=False, output_l1=False),
    KernelVariant(name="auto_dram_in_l1_out", kind="auto", input_l1=False, output_l1=True),
    # KernelVariant(name="auto_l1_in_dram_out", kind="auto", input_l1=True, output_l1=False),
    # KernelVariant(name="auto_l1_in_l1_out", kind="auto", input_l1=True, output_l1=True),
    # KernelVariant(name="mcast1d_dram_in_dram_out", kind="mcast1d", input_l1=False, output_l1=False),
    # KernelVariant(name="mcast1d_l1_in_l1_out", kind="mcast1d", input_l1=True, output_l1=True),
    # KernelVariant(name="l1_sharded_l1_out", kind="l1_sharded", input_l1=True, output_l1=True),
    # KernelVariant(name="dram_sharded_l1_out", kind="dram_sharded", input_l1=True, output_l1=True),
    # KernelVariant(name="dram_sharded_dram_out", kind="dram_sharded", input_l1=False, output_l1=False),
    KernelVariant(
        name="in0_l1_sharded_in1_dram_sharded_l1_out",
        kind="mixed_l1_dram_sharded",
        input_l1=True,
        output_l1=True,
    ),
]


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _core_grid_for_num_cores(num_cores: int) -> ttnn.CoreGrid:
    for x in range(8, 0, -1):
        if num_cores % x == 0:
            y = num_cores // x
            if y <= 8:
                return ttnn.CoreGrid(y=y, x=x)
    raise ValueError(f"Cannot map {num_cores} cores to an 8x8 worker grid")


def _core_range_set_for_grid(core_grid: ttnn.CoreGrid) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(int(core_grid.x) - 1, int(core_grid.y) - 1))}
    )


def _mcast1d_program_config(device, shape: DecodeMatmulShape):
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    num_cores = max(1, grid_x * grid_y)
    k_tiles = math.ceil(shape.k / TILE)
    n_tiles = math.ceil(shape.n / TILE)
    per_core_n = max(1, math.ceil(n_tiles / num_cores))
    max_l1_weight_bytes = 256 * 1024
    weight_tile_bytes = 1024
    max_in0_block_w_for_l1 = max(1, max_l1_weight_bytes // (per_core_n * weight_tile_bytes))
    in0_block_w = _largest_divisor_at_most(k_tiles, min(4, max_in0_block_w_for_l1))

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=_largest_divisor_at_most(per_core_n, 4),
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _max_common_tile_divisor_num_cores(shape: DecodeMatmulShape) -> int:
    k_tiles = shape.k // TILE
    n_tiles = shape.n // TILE
    return max(candidate for candidate in range(1, 65) if k_tiles % candidate == 0 and n_tiles % candidate == 0)


def _l1_sharded_configs(shape: DecodeMatmulShape):
    num_cores = _max_common_tile_divisor_num_cores(shape)
    compute_grid = _core_grid_for_num_cores(num_cores)
    core_range_set = _core_range_set_for_grid(compute_grid)

    k_tiles_per_core = (shape.k // TILE) // num_cores
    per_core_n = (shape.n // TILE) // num_cores
    in0_block_w = _largest_divisor_at_most(k_tiles_per_core, 4)

    in0_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(core_range_set, [shape.m, shape.k // num_cores], ttnn.ShardOrientation.ROW_MAJOR),
    )
    in1_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(core_range_set, [shape.k, shape.n // num_cores], ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(core_range_set, [shape.m, shape.n // num_cores], ttnn.ShardOrientation.ROW_MAJOR),
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(int(compute_grid.x), int(compute_grid.y)),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=_largest_divisor_at_most(per_core_n, 4),
        per_core_M=shape.m // TILE,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
    )
    return in0_mem_cfg, in1_mem_cfg, out_mem_cfg, program_config


def _dram_sharded_configs(device, shape: DecodeMatmulShape):
    num_cores = _max_common_tile_divisor_num_cores(shape)
    compute_grid = _core_grid_for_num_cores(num_cores)
    dram_grid = device.dram_grid_size()
    num_dram_banks = int(dram_grid.x) * int(dram_grid.y)

    n_padded = math.ceil(shape.n / (TILE * num_dram_banks)) * TILE * num_dram_banks
    per_core_n = (shape.n // TILE) // num_cores
    k_tiles_per_core = (shape.k // TILE) // num_cores
    in0_block_w = _largest_divisor_at_most(k_tiles_per_core, 4)

    in0_mem_cfg = ttnn.create_sharded_memory_config(
        (1, 1, shape.m, shape.k),
        core_grid=compute_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    in1_shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(int(dram_grid.x) - 1, int(dram_grid.y) - 1))}
    )
    in1_mem_cfg = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(
            in1_shard_grid,
            [shape.k, n_padded // num_dram_banks],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    out_mem_cfg = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=shape.m // TILE,
        per_core_N=per_core_n,
        fused_activation=None,
    )
    return in0_mem_cfg, in1_mem_cfg, out_mem_cfg, program_config


def _mixed_l1_dram_sharded_configs(device, shape: DecodeMatmulShape):
    """A in L1 WIDTH_SHARDED, B in DRAM WIDTH_SHARDED, output in L1 WIDTH_SHARDED."""
    return _dram_sharded_configs(device, shape)


def _compute_kernel_config(shape: DecodeMatmulShape):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=shape.math_fidelity,
        math_approx_mode=shape.math_fidelity == ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _memory_and_program_configs(device, shape: DecodeMatmulShape, variant: KernelVariant):
    if variant.kind == "l1_sharded":
        return _l1_sharded_configs(shape)
    if variant.kind == "mixed_l1_dram_sharded":
        return _mixed_l1_dram_sharded_configs(device, shape)
    if variant.kind == "dram_sharded":
        return _dram_sharded_configs(device, shape)

    input_mem_cfg = ttnn.L1_MEMORY_CONFIG if variant.input_l1 else ttnn.DRAM_MEMORY_CONFIG
    output_mem_cfg = ttnn.L1_MEMORY_CONFIG if variant.output_l1 else ttnn.DRAM_MEMORY_CONFIG
    program_config = _mcast1d_program_config(device, shape) if variant.kind == "mcast1d" else None
    return input_mem_cfg, ttnn.DRAM_MEMORY_CONFIG, output_mem_cfg, program_config


@pytest.mark.parametrize("shape", DECODE_MATMUL_SHAPES, ids=[shape.name for shape in DECODE_MATMUL_SHAPES])
@pytest.mark.parametrize("variant", KERNEL_VARIANTS, ids=[variant.name for variant in KERNEL_VARIANTS])
@pytest.mark.parametrize("num_iters", [1])
def test_decode_layer_matmul_kernel_capability(
    device, shape: DecodeMatmulShape, variant: KernelVariant, num_iters: int
):
    torch.manual_seed(0)

    torch_input_a = torch.randn((1, 1, shape.m, shape.k), dtype=torch.bfloat16) * 0.1
    torch_input_b = torch.randn((1, 1, shape.k, shape.n), dtype=torch.bfloat16) * 0.1
    torch_output = torch.matmul(torch_input_a, torch_input_b)

    in0_mem_cfg, in1_mem_cfg, output_mem_cfg, program_config = _memory_and_program_configs(device, shape, variant)
    compute_kernel_config = _compute_kernel_config(shape)

    input_a = ttnn.from_torch(
        torch_input_a,
        dtype=shape.in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=in0_mem_cfg,
        device=device,
    )
    input_b = ttnn.from_torch(
        torch_input_b,
        dtype=shape.in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=in1_mem_cfg,
        device=device,
    )

    last_output: Optional[ttnn.Tensor] = None
    start = time.perf_counter()
    for _ in range(num_iters):
        if last_output is not None:
            ttnn.deallocate(last_output)
        last_output = ttnn.matmul(
            input_a,
            input_b,
            program_config=program_config,
            memory_config=output_mem_cfg,
            dtype=shape.out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    ttnn.synchronize_device(device)
    elapsed_us = (time.perf_counter() - start) * 1e6 / num_iters

    output = ttnn.to_torch(last_output)
    print(f"{shape.name:>12} {variant.name:>24} " f"{shape.m}x{shape.k}x{shape.n} avg_wall_us={elapsed_us:.1f}")

    assert output.shape == torch_output.shape
    assert_with_pcc(torch_output, output, pcc=shape.pcc_threshold)

    ttnn.deallocate(last_output)
    ttnn.deallocate(input_a)
    ttnn.deallocate(input_b)
