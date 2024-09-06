# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, roundup32
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from models.utility_functions import skip_for_blackhole


def find_max_subblock(out_block_h, out_block_w):
    max_product = 0
    best_h = 1
    best_w = 1

    for h in range(1, out_block_h + 1):
        if out_block_h % h == 0:
            for w in range(1, out_block_w + 1):
                if out_block_w % w == 0 and h * w <= 8:
                    if h * w > max_product:
                        max_product = h * w
                        best_h = h
                        best_w = w
    if out_block_w > best_w:
        best_h = 1
    return best_h, best_w, max_product


def pad_to_dram_banks(num, lcm=32 * 12):
    remainder = num % lcm
    if remainder == 0:
        return num
    padding_needed = lcm - remainder
    padded_number = num + padding_needed
    return padded_number


def run_test_matmul_in1_dram_sharded(
    device,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    M,
    K,
    N,
    fidelity,
    has_bias,
    activation,
    grid_size,
    in0_dtype,
    in1_dtype,
    out_dtype,
    function_level_defaults,
    use_program_cache,
):
    if is_grayskull() and (N == 4096 or K == 32768):
        pytest.skip("Skipping too large tensor test on Grayskull")

    if is_grayskull():
        N_padded = N
        num_banks = 8
    else:
        N_padded = pad_to_dram_banks(N)
        num_banks = 12

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    in1_shard_shape = [K, N_padded // num_banks]
    bias_shape = [1, 1, N]
    bias_shard_shape = [32, N_padded // num_banks]
    num_cores = grid_size[0] * grid_size[1]

    in0_block_h = M // 32
    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    logger.debug("N_padded " + str(N_padded))
    logger.debug("in0 block h w " + str(in0_block_h * 32) + " " + str(in0_block_w * 32))
    logger.debug("in1 block h w " + str(in0_block_w * 32) + " " + str(out_block_w * 32))
    logger.debug("out block h w " + str(out_block_h * 32) + " " + str(out_block_w * 32))
    logger.debug("out subblock h w " + str(out_subblock_h * 32) + " " + str(out_subblock_w * 32))

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)

    logger.debug("in1_shard_shape " + str(in1_shard_shape))
    logger.debug("in1_shard_grid " + str(in1_shard_grid))

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=in0_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=in1_mem_config, tt_dtype=in1_dtype)

    if has_bias:
        bias = torch.randn(bias_shape).bfloat16().float()
        bias_padded = bias.unsqueeze(2)
        bias_padded = torch.nn.functional.pad(bias_padded, (0, 0, 0, 32 - bias_padded.size(2)), "constant", 0)
        bias_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
        bias_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), bias_shard_grid)})
        bias_shard_spec = ttnn.ShardSpec(bias_shard_grid, bias_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        bias_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, bias_shard_spec
        )
        bias_t = torch2tt_tensor(bias_padded, device, tt_memory_config=bias_mem_config, tt_dtype=ttnn.bfloat16)

    in0_t = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M, int(in0_block_w * 32)],
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w // 4,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fused_activation=None,
    )

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    if has_bias:
        output_t = ttnn.linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=sharded_mem_config,
            dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=sharded_mem_config,
            dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = in0 @ in1
    if has_bias:
        pt_out += bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@skip_for_blackhole("Segfault on BH, see #12349")
@pytest.mark.parametrize(
    "fidelity",
    [
        ttnn.MathFidelity.HiFi2,
        ttnn.MathFidelity.LoFi,
    ],
    ids=["HiFi2", "LoFi"],
)
@pytest.mark.parametrize(
    "has_bias",
    [
        False,
        True,
    ],
    ids=["no_bias", "bias"],
)
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, grid_size",
    # "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, grid_size, in0_dtype, in1_dtype, out_dtype",
    [
        (False, True, True, 32, 8192, 1280, None, (8, 1)),
        (False, True, True, 32, 8192, 4096, None, (8, 2)),
        (False, True, True, 32, 8192, 1024, None, (8, 1)),
        (False, True, True, 32, 32768, 1024, None, (8, 2)),
        # (False, True, True, 32, 4096, 6144, None, (8, 2), ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
        # (False, True, True, 32, 4096, 14336, None, (8, 2), ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b),
        # (False, True, True, 32, 14336, 4096, None, (8, 2), ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b),
        # (False, True, True, 32, 4096, 14336, None, (8, 2), ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b),
    ],
)
def test_matmul_in1_dram_sharded_with_program_cache(
    device,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    M,
    K,
    N,
    fidelity,
    has_bias,
    activation,
    grid_size,
    in0_dtype,
    in1_dtype,
    out_dtype,
    function_level_defaults,
    use_program_cache,
):
    for _ in range(2):
        run_test_matmul_in1_dram_sharded(
            device,
            in0_sharded,
            out_sharded,
            in1_in_dram,
            M,
            K,
            N,
            fidelity,
            has_bias,
            activation,
            grid_size,
            in0_dtype,
            in1_dtype,
            out_dtype,
            function_level_defaults,
            use_program_cache,
        )
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        )
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, in0_dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)
    assert device.num_program_cache_entries() == 3


def run_test_matmul_in1_dram_sharded_mm_chain(
    device,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    M,
    K,
    N,
    fidelity,
    has_bias,
    activation,
    grid_size,
    in0_dtype,
    in1_dtype,
    out_dtype,
    function_level_defaults,
    use_program_cache,
):
    if is_grayskull() and (N == 4096 or K == 32768):
        pytest.skip("Skipping too large tensor test on Grayskull")

    if is_grayskull():
        N_padded = N
        num_banks = 8
    else:
        N_padded = pad_to_dram_banks(N)
        num_banks = 12

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    in1_shard_shape = [K, N_padded // num_banks]
    num_cores = grid_size[0] * grid_size[1]

    in0_block_h = M // 32
    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    logger.debug("N_padded " + str(N_padded))
    logger.debug("in0 block h w " + str(in0_block_h * 32) + " " + str(in0_block_w * 32))
    logger.debug("in1 block h w " + str(in0_block_w * 32) + " " + str(out_block_w * 32))
    logger.debug("out block h w " + str(out_block_h * 32) + " " + str(out_block_w * 32))
    logger.debug("out subblock h w " + str(out_subblock_h * 32) + " " + str(out_subblock_w * 32))

    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_shard_grid = (grid_size[0] - 1, grid_size[1] - 1)
    in0_shard_shape = [M, int(in0_block_w * 32)]
    in0_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in0_shard_grid)})
    in0_shard_spec = ttnn.ShardSpec(in0_shard_grid, in0_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=in0_dtype)

    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=in1_mem_config, tt_dtype=in1_dtype)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w // 4,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fused_activation=None,
    )

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # 1st mm
    output_t = ttnn.matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        memory_config=sharded_mem_config,
        dtype=out_dtype,
        compute_kernel_config=compute_kernel_config,
    )

    for _ in range(200):
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=sharded_mem_config,
            dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )

    output_t = output_t.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    pt_out = in0 @ in1

    tt_out = tt2torch_tensor(output_t)

    print(tt_out)
    print(pt_out)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert True


@skip_for_blackhole("Segfaulting on BH, see #12349")
@pytest.mark.parametrize(
    "fidelity",
    [
        ttnn.MathFidelity.HiFi2,
    ],
    ids=[
        "HiFi2",
    ],
)
@pytest.mark.parametrize(
    "has_bias",
    [
        False,
    ],
    ids=["no_bias"],
)
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
    ],
)
def test_matmul_in1_dram_sharded_with_mm_chain(
    device,
    fidelity,
    has_bias,
    in0_dtype,
    in1_dtype,
    out_dtype,
    function_level_defaults,
    use_program_cache,
):
    M = 32
    K = 4096
    N = 4096
    grid_size = (8, 2)
    run_test_matmul_in1_dram_sharded_mm_chain(
        device,
        True,
        True,
        True,
        M,
        K,
        N,
        fidelity,
        has_bias,
        None,
        grid_size,
        in0_dtype,
        in1_dtype,
        out_dtype,
        function_level_defaults,
        use_program_cache,
    )


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("packer_l1_acc", [True, False], ids=["pack_l1", "no_pack_l1"])
@pytest.mark.parametrize(
    "fp32_acc_mode",
    [
        True,
    ],
    ids=["fp32"],
)
@pytest.mark.parametrize(
    "fidelity",
    [
        ttnn.MathFidelity.LoFi,
    ],
    ids=["LoFi"],
)
@pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize(
    "M, K, N, activation, in0_sharded, fuse_batch",
    [
        (1024, 1024, 1024, None, True, True),
        (1024, 8192, 4096, None, False, False),
    ],
)
def test_matmul_2d_in1_dram_sharded(
    device,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    M,
    K,
    N,
    activation,
    in0_sharded,
    fuse_batch,
    function_level_defaults,
):
    if is_grayskull():
        N_padded = N
        num_banks = 8
    else:
        N_padded = pad_to_dram_banks(N)
        num_banks = 12

    if fuse_batch:
        in0_shape = [1, 1, M, K]
    else:
        in0_shape = [1, 2, M, K]
    in1_shape = [1, 1, K, N]
    in1_shard_shape = [K, N_padded // num_banks]
    bias_shape = [1, 1, N]
    bias_shard_shape = [32, N_padded // num_banks]
    grid_size = (8, 4)

    in0_block_h = M // grid_size[1] // 32
    in0_block_w = K // grid_size[0] // 32
    out_block_h = M // grid_size[1] // 32
    out_block_w = N // grid_size[0] // 32

    # full block too large to fit in L1
    if in0_block_h * in0_block_w >= 48 or in0_block_w * out_block_w >= 48:
        in0_block_w = in0_block_w // 2

    if out_block_w < 4:
        out_subblock_w = out_block_w
        out_subblock_h = out_block_h // out_subblock_w
    else:
        out_subblock_w = 4
        out_subblock_h = 1

    logger.debug("in0 block w h " + str(in0_block_w * 32) + " " + str(in0_block_h * 32))
    logger.debug("in1 block w h " + str(out_block_w * 32) + " " + str(in0_block_w * 32))
    logger.debug("out block w h " + str(out_block_w * 32) + " " + str(out_block_h * 32))
    logger.debug("out subblock w h " + str(out_subblock_w * 32) + " " + str(out_subblock_h * 32))

    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    interleaved_mem_config_L1 = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    interleaved_mem_config_DRAM = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttnn.bfloat16)
    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[1], K // grid_size[0]],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    in1 = torch.randn(in1_shape).bfloat16().float()
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=in1_mem_config, tt_dtype=ttnn.bfloat16)

    if has_bias:
        bias = torch.ones(bias_shape).bfloat16().float()
        bias_padded = bias.unsqueeze(2)
        bias_padded = torch.nn.functional.pad(bias_padded, (0, 0, 0, 32 - bias_padded.size(2)), "constant", 0)
        bias_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
        bias_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), bias_shard_grid)})
        bias_shard_spec = ttnn.ShardSpec(bias_shard_grid, bias_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        bias_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, bias_shard_spec
        )
        bias_t = torch2tt_tensor(bias_padded, device, tt_memory_config=bias_mem_config, tt_dtype=ttnn.bfloat16)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w // 4,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=False,
        fused_activation=activation,
        fuse_batch=fuse_batch,
    )

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
            fp32_dest_acc_en=fp32_acc_mode,
            packer_l1_acc=packer_l1_acc,
        )
    if has_bias:
        output_t = ttnn.linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=sharded_mem_config if in0_sharded else interleaved_mem_config_DRAM,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=sharded_mem_config if in0_sharded else interleaved_mem_config_DRAM,
            compute_kernel_config=compute_kernel_config,
        )

    if in0_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config_DRAM)
    tt_out = tt2torch_tensor(output_t)

    pt_out = in0 @ in1
    if has_bias:
        pt_out = pt_out + bias

    if activation != None:
        pt_out = torch.nn.functional.gelu(pt_out)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing
