# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.utility_functions import is_wormhole_b0, is_grayskull
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, roundup32
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
@pytest.mark.parametrize("enable_async, num_loops", ((True, 2), (False, 1)))
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
    enable_async,
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
    device.enable_async(enable_async)
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
    device.enable_async(False)


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize("packer_l1_acc", [True, False], ids=["pack_l1", "no_pack_l1"])
@pytest.mark.parametrize("fp32_acc_mode", [True, False], ids=["fp32", "no_fp32"])
@pytest.mark.parametrize("batch", [16, 96])
@pytest.mark.parametrize("in0_sharded", [False], ids=["in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("M", [32, 128])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("enable_async, num_loops", ((True, 2), (False, 1)))
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
    enable_async,
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
    device.enable_async(enable_async)
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

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
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
    device.enable_async(False)


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize("packer_l1_acc", [True, False], ids=["pack_l1", "no_pack_l1"])
@pytest.mark.parametrize("fp32_acc_mode", [True, False], ids=["fp32", "no_fp32"])
@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("in1_sharded", [True, False], ids=["in1_sharded", "in1_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("B, H, M, K, N, out_subblock_h, out_subblock_w", [[2, 16, 384, 64, 128, 1, 4]])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("enable_async, num_loops", ((True, 2), (False, 1)))
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
    enable_async,
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
    device.enable_async(enable_async)
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

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
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
    device.enable_async(False)


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
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
@pytest.mark.parametrize("enable_async, num_loops", ((True, 2), (False, 1)))
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
    enable_async,
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
    device.enable_async(enable_async)
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

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
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
    device.enable_async(False)


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
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
@pytest.mark.parametrize("enable_async, num_loops", ((True, 2), (False, 1)))
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
    enable_async,
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
    device.enable_async(enable_async)
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

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
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
    device.enable_async(False)


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
@pytest.mark.parametrize("enable_async, num_loops", ((True, 2), (False, 1)))
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
    enable_async,
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if is_grayskull() and (fp32_acc_mode or packer_l1_acc):
        pytest.skip(f"Need Grayskull doesn't support fp32_acc_mode or packer_l1_acc")
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
    device.enable_async(enable_async)
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

        if is_grayskull():
            compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
            )
        else:
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
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
    device.enable_async(False)
