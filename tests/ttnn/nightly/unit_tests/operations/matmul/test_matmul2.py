# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
from loguru import logger
from models.common.utility_functions import is_wormhole_b0
from models.common.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, roundup32
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)


@pytest.mark.parametrize("batch", [16, 96])
@pytest.mark.parametrize("in0_sharded", [False], ids=["in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("M", [32, 128])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("num_loops", [2])
def test_matmul_1d_in0_batched(
    device,
    batch,
    in0_sharded,
    out_sharded,
    M,
    N,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
    num_loops,
):
    grid_size = (12, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")
    num_cores = grid_size[0] * grid_size[1]
    K = 128
    in0_shape = [batch, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    for _ in range(num_loops):
        in0 = torch.randn(in0_shape).bfloat16().float()
        in1 = torch.randn(in1_shape).bfloat16().float()
        bias = torch.randn(bias_shape).bfloat16().float()

        in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
        bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

        output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

        if in0_sharded:
            in0_t = ttnn.interleaved_to_sharded(
                in0_t,
                grid_size,
                [M, K // num_cores],
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )

        per_core_M = M // 32

        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=per_core_M,
            per_core_N=1,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        output_t = ttnn.linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=output_mem_config,
            dtype=activations_dtype,
        )

        in0_t.deallocate()
        in1_t.deallocate()
        bias_t.deallocate()
        if out_sharded:
            output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
        pt_out = in0 @ in1 + bias

        tt_out = tt2torch_tensor(output_t)
        output_t.deallocate()
        passing, output = comp_pcc(pt_out, tt_out)
        logger.info(output)
        assert passing


@pytest.mark.parametrize("packer_l1_acc", [True, False], ids=["pack_l1", "no_pack_l1"])
@pytest.mark.parametrize("fp32_acc_mode", [True, False], ids=["fp32", "no_fp32"])
@pytest.mark.parametrize("batch", [16, 96])
@pytest.mark.parametrize("in0_sharded", [False], ids=["in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("M", [32, 128])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("num_loops", [2])
def test_linear_fp32_acc_l1(
    device,
    packer_l1_acc,
    fp32_acc_mode,
    batch,
    in0_sharded,
    out_sharded,
    M,
    N,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
    num_loops,
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")
    num_cores = grid_size[0] * grid_size[1]
    K = 128
    in0_shape = [batch, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    for _ in range(num_loops):
        in0 = torch.randn(in0_shape).bfloat16().float()
        in1 = torch.randn(in1_shape).bfloat16().float()
        bias = torch.randn(bias_shape).bfloat16().float()

        in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
        bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

        output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

        if in0_sharded:
            in0_t = ttnn.interleaved_to_sharded(
                in0_t,
                grid_size,
                [M, K // num_cores],
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )

        per_core_M = M // 32

        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=per_core_M,
            per_core_N=1,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=fp32_acc_mode,
            packer_l1_acc=packer_l1_acc,
        )

        output_t = ttnn.linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=output_mem_config,
            dtype=activations_dtype,
            compute_kernel_config=compute_kernel_config,
        )
        if out_sharded:
            output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
        pt_out = in0 @ in1 + bias

        tt_out = tt2torch_tensor(output_t)

        passing, output = comp_pcc(pt_out, tt_out)
        logger.info(output)
        assert passing


@pytest.mark.parametrize("packer_l1_acc", [True, False], ids=["pack_l1", "no_pack_l1"])
@pytest.mark.parametrize("fp32_acc_mode", [True, False], ids=["fp32", "no_fp32"])
@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("in1_sharded", [True, False], ids=["in1_sharded", "in1_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("B, H, M, K, N, out_subblock_h, out_subblock_w", [[2, 16, 384, 64, 128, 1, 4]])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("num_loops", [2])
def test_matmul_no_mcast_fp32_acc_l1(
    device,
    packer_l1_acc,
    fp32_acc_mode,
    in0_sharded,
    in1_sharded,
    out_sharded,
    B,
    H,
    M,
    K,
    N,
    out_subblock_h,
    out_subblock_w,
    activations_dtype,
    function_level_defaults,
    num_loops,
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    num_cores = grid_size[0] * grid_size[1]
    in0_shape = [B, H, M, K]
    in1_shape = [B, H, K, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    for _ in range(num_loops):
        in0 = torch.randn(in0_shape).bfloat16().float()
        in1 = torch.randn(in1_shape).bfloat16().float()

        in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

        output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

        if in0_sharded:
            in0_t = ttnn.interleaved_to_sharded(
                in0_t,
                grid_size,
                [B * H * M // num_cores, K],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.COL_MAJOR,
            )
        if in1_sharded:
            in1_t = ttnn.interleaved_to_sharded(
                in1_t,
                grid_size,
                [B * H * K // num_cores, N],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.COL_MAJOR,
            )

        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=K // 32,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=B * H * M // num_cores // 32,
            per_core_N=N // 32,
        )

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=fp32_acc_mode,
            packer_l1_acc=packer_l1_acc,
        )

        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=output_mem_config,
            dtype=activations_dtype,
            compute_kernel_config=compute_kernel_config,
        )
        if out_sharded:
            output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

        pt_out = in0 @ in1

        tt_out = tt2torch_tensor(output_t)

        passing, output = comp_pcc(pt_out, tt_out)
        logger.info(output)
        assert passing


@pytest.mark.parametrize("packer_l1_acc", [True, False], ids=["pack_l1", "no_pack_l1"])
@pytest.mark.parametrize(
    "fp32_acc_mode",
    [
        True,
    ],
    ids=["fp32"],
)
@pytest.mark.parametrize("batch", [16, 32])
@pytest.mark.parametrize("in0_sharded", [False], ids=["in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("M", [32, 128])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("activations_dtype", [ttnn.float32])
@pytest.mark.parametrize("weights_dtype", [ttnn.float32])
@pytest.mark.parametrize("num_loops", [2])
def test_matmul_1d_fp32_input_output(
    device,
    packer_l1_acc,
    fp32_acc_mode,
    batch,
    in0_sharded,
    out_sharded,
    M,
    N,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
    num_loops,
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")
    num_cores = grid_size[0] * grid_size[1]
    K = 128
    in0_shape = [batch, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    for _ in range(num_loops):
        in0 = torch.rand(in0_shape).float()
        in1 = torch.rand(in1_shape).float()
        bias = torch.rand(bias_shape).float()

        in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
        bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

        output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

        if in0_sharded:
            in0_t = ttnn.interleaved_to_sharded(
                in0_t,
                grid_size,
                [M, K // num_cores],
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )

        per_core_M = M // 32

        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=per_core_M,
            per_core_N=1,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=fp32_acc_mode,
            packer_l1_acc=packer_l1_acc,
        )

        output_t = ttnn.linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=output_mem_config,
            dtype=activations_dtype,
            compute_kernel_config=compute_kernel_config,
        )
        if out_sharded:
            output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
        pt_out = in0 @ in1 + bias

        tt_out = tt2torch_tensor(output_t)

        passing, output = comp_pcc(pt_out, tt_out)
        logger.info(output)
        assert passing


@pytest.mark.parametrize("packer_l1_acc", [True, False], ids=["pack_l1", "no_pack_l1"])
@pytest.mark.parametrize(
    "fp32_acc_mode",
    [
        True,
    ],
    ids=["fp32"],
)
@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("in1_sharded", [True, False], ids=["in1_sharded", "in1_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("B, H, M, K, N, out_subblock_h, out_subblock_w", [[2, 16, 384, 64, 128, 1, 4]])
@pytest.mark.parametrize("activations_dtype", [ttnn.float32])
@pytest.mark.parametrize("num_loops", [2])
def test_matmul_no_mcast_fp32_input_output(
    device,
    packer_l1_acc,
    fp32_acc_mode,
    in0_sharded,
    in1_sharded,
    out_sharded,
    B,
    H,
    M,
    K,
    N,
    out_subblock_h,
    out_subblock_w,
    activations_dtype,
    function_level_defaults,
    num_loops,
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    num_cores = grid_size[0] * grid_size[1]
    in0_shape = [B, H, M, K]
    in1_shape = [B, H, K, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    for _ in range(num_loops):
        in0 = torch.rand(in0_shape).float() * 1000.0
        in1 = torch.rand(in1_shape).float() * 1000.0

        in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

        output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

        if in0_sharded:
            in0_t = ttnn.interleaved_to_sharded(
                in0_t,
                grid_size,
                [B * H * M // num_cores, K],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.COL_MAJOR,
            )
        if in1_sharded:
            in1_t = ttnn.interleaved_to_sharded(
                in1_t,
                grid_size,
                [B * H * K // num_cores, N],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.COL_MAJOR,
            )

        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=K // 32,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=B * H * M // num_cores // 32,
            per_core_N=N // 32,
        )

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=fp32_acc_mode,
            packer_l1_acc=packer_l1_acc,
        )

        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=output_mem_config,
            dtype=activations_dtype,
            compute_kernel_config=compute_kernel_config,
        )
        if out_sharded:
            output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

        pt_out = in0 @ in1

        tt_out = tt2torch_tensor(output_t)

        passing, output = comp_pcc(pt_out, tt_out)
        logger.info(output)
        assert passing


@pytest.mark.parametrize("packer_l1_acc", [True, False], ids=["pack_l1", "no_pack_l1"])
@pytest.mark.parametrize(
    "fp32_acc_mode",
    [
        True,
        False,
    ],
    ids=["fp32", "no_fp32"],
)
@pytest.mark.parametrize("in0_sharded", [False], ids=["in0_unsharded"])
@pytest.mark.parametrize("in1_sharded", [True], ids=["in1_sharded"])
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize("B, H, M, K, N, out_subblock_h, out_subblock_w", [[2, 16, 384, 128, 64, 2, 2]])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("num_loops", [2])
def test_matmul_no_untilize_output_param(
    device,
    packer_l1_acc,
    fp32_acc_mode,
    in0_sharded,
    in1_sharded,
    out_sharded,
    B,
    H,
    M,
    K,
    N,
    out_subblock_h,
    out_subblock_w,
    activations_dtype,
    output_dtype,
    function_level_defaults,
    num_loops,
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    num_cores = grid_size[0] * grid_size[1]
    in0_shape = [B, H, M, K]
    in1_shape = [B, H, K, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    for _ in range(num_loops):
        in0 = torch.rand(in0_shape).float() * 1000.0
        in1 = torch.rand(in1_shape).float() * 1000.0

        in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

        output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

        if in0_sharded:
            in0_t = ttnn.interleaved_to_sharded(
                in0_t,
                grid_size,
                [B * H * M // num_cores, K],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.COL_MAJOR,
            )
        if in1_sharded:
            in1_t = ttnn.interleaved_to_sharded(
                in1_t,
                grid_size,
                [B * H * K // num_cores, N],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.COL_MAJOR,
            )

        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=K // 32,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=B * H * M // num_cores // 32,
            per_core_N=N // 32,
        )

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=fp32_acc_mode,
            packer_l1_acc=packer_l1_acc,
        )

        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=output_mem_config,
            dtype=output_dtype,
            compute_kernel_config=compute_kernel_config,
        )
        if out_sharded:
            output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

        pt_out = in0 @ in1

        tt_out = tt2torch_tensor(output_t)

        passing, output = comp_pcc(pt_out, tt_out)
        logger.info(output)
        assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M", [1600])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("K", [256, 512])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_matmul_2d(
    device,
    in0_sharded,
    out_sharded,
    M,
    N,
    K,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    grid_size = (8, 5)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[1], K // grid_size[0]],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=10,
        per_core_N=4,
        transpose_mcast=False,
        fused_activation=None,
    )
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_interleaved"])
@pytest.mark.parametrize("in1_sharded", [True, False], ids=["in1_sharded", "in1_interleaved"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_interleaved"])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_matmul_2d_in0_height_sharded_in1_width_sharded(
    device,
    in0_sharded,
    in1_sharded,
    out_sharded,
    activations_dtype,
    weights_dtype,
    output_dtype,
    function_level_defaults,
):
    M = 6 * 32
    N = 12 * 32
    K = 2 * 32

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    grid_size = (6, 6)
    compute_grid_size = device.compute_with_storage_grid_size()

    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_block_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    # Generate the tensor
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            ttnn.CoreCoord(1, grid_size[0]),
            [M // grid_size[0], K],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    if in1_sharded:
        in1_t = ttnn.interleaved_to_sharded(
            in1_t,
            ttnn.CoreCoord(grid_size[1], 1),
            [K, N // grid_size[1]],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // 32,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=M // (32 * grid_size[0]),
        per_core_N=N // (32 * grid_size[1]),
        transpose_mcast=False,
        fused_activation=None,
    )
    output_mem_config = sharded_block_mem_config if out_sharded else interleaved_mem_config
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=output_dtype,
    )

    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M", [1600])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_matmul_2d_transposed(
    device,
    in0_sharded,
    out_sharded,
    M,
    N,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
):
    K = 256
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    grid_size = (10, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[0], K // grid_size[1]],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=5,
        per_core_N=4,
        transpose_mcast=True,
        fused_activation=None,
    )
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


def test_resharded_binary_to_matmul(device, function_level_defaults):
    grid_size_binary = device.compute_with_storage_grid_size()
    num_cores_binary = 98
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores_binary > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores_binary} cores to run this test but core grid is {compute_grid_size}")
    grid_size_matmul = (10, 8)
    if grid_size_matmul[0] > compute_grid_size.x or grid_size_matmul[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size_matmul} grid size to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, 6272, 512]
    in1_shape = in0_shape
    weight_shape = [1, 1, 512, 256]
    bias_shape = [1, 1, 1, 256]
    H = in0_shape[-2]
    W = in0_shape[-1]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    height_sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    block_sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    weight = torch.randn(weight_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config)
    weight_t = torch2tt_tensor(weight, device, tt_memory_config=interleaved_mem_config)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config)[0]

    in0_t = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size_binary,
        [H // num_cores_binary, W],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    in1_t = ttnn.interleaved_to_sharded(
        in1_t,
        grid_size_binary,
        [H // num_cores_binary, W],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    output_binary_t = ttnn.add(in0_t, in1_t, memory_config=interleaved_mem_config)
    output_binary_t = ttnn.interleaved_to_sharded(
        output_binary_t,
        grid_size_matmul,
        [math.ceil((H // 32) / grid_size_matmul[0]) * 32, W // grid_size_matmul[1]],
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size_matmul,
        in0_block_w=2,
        out_subblock_h=5,
        out_subblock_w=1,
        per_core_M=20,
        per_core_N=1,
        transpose_mcast=True,
        fused_activation=None,
    )
    output_matmul_t = ttnn.linear(
        output_binary_t,
        weight_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=block_sharded_mem_config,
    )
    output_matmul_t = ttnn.sharded_to_interleaved(output_matmul_t, interleaved_mem_config)

    tt_out = tt2torch_tensor(output_matmul_t)

    pt_out = (in0 + in1) @ weight

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M", [32, 64])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("K", [2048, 4096])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_matmul_1d_in0(
    device,
    in0_sharded,
    out_sharded,
    M,
    K,
    N,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")
    num_cores = grid_size[0] * grid_size[1]
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, K // num_cores],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M // 32,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out, 0.98)
    logger.info(output)
    assert passing


# Have at least one example of 1d matmul with in1 mcasted that runs on WH
def test_sharded_matmul_1d_in1_wormhole(device, function_level_defaults):
    M = 4096
    K = 64
    N = 256
    grid_size = (8, 4)
    num_cores = grid_size[0] * grid_size[1]
    dtype = ttnn.bfloat16

    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)[0]

    output_mem_config = sharded_mem_config

    in0_t = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M // num_cores, K],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M // 32 // num_cores,
        per_core_N=N // 32,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=dtype,
    )
    output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("in1_sharded", [True, False], ids=["in1_sharded", "in1_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize(
    "B, H, M, K, N, out_subblock_h, out_subblock_w",
    [[12, 16, 384, 64, 384, 1, 6], [12, 16, 384, 384, 64, 4, 2]],
)
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat8_b])
def test_sharded_matmul_no_mcast(
    device,
    in0_sharded,
    in1_sharded,
    out_sharded,
    B,
    H,
    M,
    K,
    N,
    out_subblock_h,
    out_subblock_w,
    activations_dtype,
    function_level_defaults,
):
    grid_size = (12, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    num_cores = grid_size[0] * grid_size[1]
    in0_shape = [B, H, M, K]
    in1_shape = [B, H, K, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [B * H * M // num_cores, K],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )
    if in1_sharded:
        in1_t = ttnn.interleaved_to_sharded(
            in1_t,
            grid_size,
            [B * H * K // num_cores, N],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // 32,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=B * H * M // num_cores // 32,
        per_core_N=N // 32,
    )

    output_t = ttnn.matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = in0 @ in1

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing
