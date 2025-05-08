# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.utility_functions import is_wormhole_b0, torch_random, is_wormhole_b0, is_grayskull, is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
def test_ttnn_experimental_tensor_exp(device, height, width):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((1, 1, height, width), -1, 1, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.exp)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.exp(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("m_size", [32])
@pytest.mark.parametrize("k_size", [32])
@pytest.mark.parametrize("n_size", [32])
def test_ttnn_matmul(device, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch_random((1, 1, m_size, k_size), -1, 1, dtype=torch.bfloat16)
    torch_input_tensor_b = torch_random((1, 1, k_size, n_size), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, device=device, layout=ttnn.TILE_LAYOUT)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("input_a_is_sharded", [True, False])
@pytest.mark.parametrize("output_is_sharded", [True, False])
@pytest.mark.parametrize("m_size, num_cores", [[5632, 22]])
@pytest.mark.parametrize("k_size, n_size", [[64, 64], [64, 256]])
@pytest.mark.parametrize("input_a_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("input_b_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_ttnn_linear(
    device, input_a_is_sharded, output_is_sharded, m_size, k_size, n_size, num_cores, input_a_dtype, input_b_dtype
):
    grid_size = (6, 4)
    compute_grid_size = device.compute_with_storage_grid_size()

    input_shape_a = [1, 1, m_size, k_size]
    input_shape_b = [1, 1, k_size, n_size]
    bias_shape = [1, 1, 1, n_size]

    interleaved_memory_config = ttnn.DRAM_MEMORY_CONFIG
    sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    output_memory_config = sharded_memory_config if output_is_sharded else interleaved_memory_config

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=k_size // 32,
        out_subblock_h=8 // (n_size // 32),
        out_subblock_w=n_size // 32,
        per_core_M=m_size // 32 // num_cores,
        per_core_N=n_size // 32,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )

    with ttnn.tracer.trace():
        torch_input_tensor_a = torch.randn(input_shape_a).bfloat16().float()
        torch_input_tensor_b = torch.randn(input_shape_b).bfloat16().float()
        torch_bias = torch.randn(bias_shape).bfloat16().float()
        torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b + torch_bias

        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=interleaved_memory_config,
            dtype=input_a_dtype,
        )
        input_tensor_b = ttnn.from_torch(
            torch_input_tensor_b,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=interleaved_memory_config,
            dtype=input_b_dtype,
        )

        bias = ttnn.from_torch(
            torch_bias,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=interleaved_memory_config,
            dtype=input_b_dtype,
        )
        if input_a_is_sharded:
            input_tensor_a = ttnn.interleaved_to_sharded(
                input_tensor_a,
                grid_size,
                [m_size // num_cores, k_size],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )

        output_tensor = ttnn.linear(
            input_tensor_a,
            input_tensor_b,
            bias=bias,
            program_config=program_config,
            memory_config=output_memory_config,
            dtype=input_a_dtype,
        )
        if output_is_sharded:
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, interleaved_memory_config)

        output_tensor = ttnn.to_torch(output_tensor)
        ttnn.tracer.visualize(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9996)


@pytest.mark.skipif(is_grayskull(), reason="parallelization not supported for GS")
@pytest.mark.parametrize("m_size", [32])
@pytest.mark.parametrize("k_size", [8192])
@pytest.mark.parametrize("n_size", [1024])
def test_ttnn_matmul_dram_sharded(device, m_size, k_size, n_size):
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=1, x=8)

    torch_input_tensor_in0 = torch.randn((1, 1, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_in1 = torch.randn((1, 1, k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_in0 @ torch_input_tensor_in1

    # in0 ttnn tensor
    input_tensor_in0 = ttnn.from_torch(
        torch_input_tensor_in0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )
    # in0 shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = (32, 1024)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    input_tensor_in0 = ttnn.to_memory_config(input_tensor_in0, sharded_mem_config)

    # in1 shard config
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_shape = (8192, 128) if is_blackhole() else (8192, 96)
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    input_tensor_in1 = ttnn.from_torch(
        torch_input_tensor_in1,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=in1_mem_config,
    )

    # output shard config
    out_shard_shape = (32, 128)
    out_shard_spec = ttnn.ShardSpec(shard_grid, out_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    out_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=32,
        per_core_M=1,
        per_core_N=4,
        fused_activation=None,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    output_tensor = ttnn.matmul(
        input_tensor_in0,
        input_tensor_in1,
        program_config=program_config,
        memory_config=out_sharded_mem_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )

    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9999)


@pytest.mark.parametrize("H, num_cores", [[64, 64]])
@pytest.mark.parametrize("num_slices", [2])
def test_sharded_partial_op(device, H, num_cores, num_slices):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)
    in0_shape = [1, 1, H, 64]
    W = in0_shape[-1]

    interleaved_mem_config = ttnn.L1_MEMORY_CONFIG
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.ones(in0_shape).bfloat16().float()
    out_initial = torch.randn(in0_shape).bfloat16().float()

    in0_t = ttnn.from_torch(
        in0,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=interleaved_mem_config,
    )
    out_tt_tensor = ttnn.from_torch(
        out_initial,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=interleaved_mem_config,
    )

    height_shard_spec = [H // 2, W]

    for slice_index in range(num_slices):
        in0_t_slice = ttnn.interleaved_to_sharded_partial(
            in0_t,
            grid_size,
            height_shard_spec,
            num_slices,
            slice_index,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        assert in0_t_slice.is_sharded()

        ttnn.sharded_to_interleaved_partial(
            in0_t_slice,
            out_tt_tensor,
            num_slices,
            slice_index,
            memory_config=interleaved_mem_config,
        )

    pt_out = in0

    tt_out = ttnn.to_torch(out_tt_tensor)

    assert_with_pcc(pt_out, tt_out)
