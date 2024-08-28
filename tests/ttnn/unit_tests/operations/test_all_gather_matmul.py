# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from ttnn import experimental as ttl
import ttnn
from ttnn import ShardTensorToMesh
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.test_all_gather import is_unsupported_case


def run_all_gather_matmul_on_t3000_impl(
    t3k_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    # Matmul params
    matmul_output_dim,
    matmul_config,
    matmul_weights_dtype,
    max_in0_block_w,
    # Memory configs
    mem_config_ag,
    mem_config_mm,
    mem_config_weights=None,
    num_iters=1,
):
    # Set the default config
    if mem_config_weights is None:
        mem_config_weights = mem_config_ag

    # Skip unsupported cases
    (is_known_failure, message) = is_unsupported_case(
        ag_output_shape, dim, mem_config_ag, num_devices, num_links, ag_input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    devices = t3k_mesh_device.get_devices()

    logger.info(f"All Gather output shape: {ag_output_shape}")
    logger.info(f"dim: {dim}")

    ##### Create input tensor for the all gather #####
    _, _, _, hidden_dim = ag_output_shape
    input_tensor = torch.randn(ag_output_shape).float()
    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(ttl.tensor.Tensor(t, ag_input_dtype).to(layout).to(devices[i], mem_config_ag))
    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

    ##### Create the weight matrix for the matmul #####
    weights_tensor = torch.randn([1, 1, hidden_dim, matmul_output_dim * num_devices]).float()
    weight_tt = ttnn.as_tensor(
        weights_tensor,
        dtype=matmul_weights_dtype,
        layout=layout,
        device=t3k_mesh_device,
        memory_config=mem_config_weights,
        mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=dim),
    )

    ##### Configs for ttnn.matmul #####
    if matmul_config == "matmul_1d":
        core_grid = (8, 4)
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=core_grid,
            in0_block_w=min(max_in0_block_w, hidden_dim // 32 // core_grid[0]),  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max(1, ag_output_shape[2] // 32 // core_grid[1]),  # M / TILE_HEIGHT / Grid_Size
            per_core_N=max(1, matmul_output_dim // 32 // core_grid[0]),  # N / TILE_WIDTH / Grid_Size
            mcast_in0=True,
            fused_activation=None,  # ttnn.UnaryOpType.SILU,
            fuse_batch=True,
        )
    elif matmul_config == "matmul_2d":
        core_grid = (8, 4)
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=core_grid,
            in0_block_w=min(max_in0_block_w, hidden_dim // 32 // core_grid[0]),  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max(1, ag_output_shape[2] // 32 // core_grid[1]),  # M / TILE_HEIGHT / Grid_Size
            per_core_N=max(1, matmul_output_dim // 32 // core_grid[0]),  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,  # ttnn.UnaryOpType.SILU,
            fuse_batch=False,
        )
    else:
        raise ValueError(f"Unsupported matmul_config: {matmul_config}")

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    ##### Perform the torch ops #####
    matmul_output = torch.chunk(torch.matmul(input_tensor, weights_tensor), num_devices, dim)

    ##### Perform the TT ops #####
    for i in range(num_iters):
        # # all_gather
        # tt_all_gather_out_tensor = ttnn.all_gather(input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config_ag)

        # # matmul
        # tt_matmul_output = ttnn.matmul(
        #     tt_all_gather_out_tensor,
        #     weight_tt,
        #     memory_config=mem_config_mm,
        #     program_config=program_config,
        #     compute_kernel_config=compute_kernel_config,
        # )

        # Test ttnn all_gather_matmul
        tt_all_gather_out_tensor, tt_matmul_output, tt_datacopy_out_tensor = ttl.all_gather_matmul(
            input_tensor_mesh,
            weight_tt,
            dim,
            (0, 4),
            num_links=num_links,
            memory_config_ag=mem_config_ag,
            memory_config_mm=mem_config_mm,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

        logger.info(f"Done iteration {i}")

        # Synchronize the devices
        for d in devices:
            ttnn.synchronize_device(d)

    ##### Compare the outputs #####
    print("Checking outputs for All Gather Matmul (All Gather)")
    for i, t in enumerate(ttnn.get_device_tensors(tt_all_gather_out_tensor)):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        if ag_input_dtype == ttl.tensor.DataType.BFLOAT16:
            eq, output = comp_equal(tt_output_tensor, input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, input_tensor)
        logger.info(f"Output {i}: {output}")
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        assert eq, f"{i} FAILED: {output}"

    # print("Checking outputs for All Gather Matmul (Datacopy)")
    # for i, t in enumerate(ttnn.get_device_tensors(tt_datacopy_out_tensor)):
    #     tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    #     if ag_input_dtype == ttl.tensor.DataType.BFLOAT16:
    #         eq, output = comp_equal(tt_output_tensor, input_tensor)
    #     else:
    #         eq, output = comp_pcc(tt_output_tensor, input_tensor)
    #     logger.info(f"Output {i}: {output}")
    #     if not eq:
    #         logger.error(f"output mismatch for tensor {i}")
    #     assert eq, f"{i} FAILED: {output}"

    print("Checking outputs for Matmul")
    for i, t in enumerate(ttnn.get_device_tensors(tt_matmul_output)):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        eq, output = comp_pcc(tt_output_tensor, matmul_output[i])
        logger.info(f"Output {i}: {output}")
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        assert eq, f"{i} FAILED: {output}"


# @skip_for_wormhole_b0()  # Used to disable test
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "matmul_config",
    [
        "matmul_2d",
    ],
)
@pytest.mark.parametrize(
    "num_devices, num_links, ag_output_shape, dim, layout, matmul_output_dim, max_in0_block_w, matmul_weights_dtype",
    [
        (
            8,
            1,
            [1, 1, 32, 16 * 32],
            3,
            ttl.tensor.Layout.TILE,
            1024,
            2,
            ttl.tensor.DataType.BFLOAT16,
        ),
        (
            8,
            1,
            [1, 1, 128, 128 * 32],
            3,
            ttl.tensor.Layout.TILE,
            1024,
            16,
            ttl.tensor.DataType.BFLOAT16,
        ),
        (
            8,
            1,
            [1, 1, 32, 1024 * 16],
            3,
            ttl.tensor.Layout.TILE,
            1024,
            16,  # NOTE: 64 for some reason gives lower perf
            ttl.tensor.DataType.BFLOAT16,
        ),
        (
            8,
            1,
            [1, 1, 1024, 1024 * 32],
            3,
            ttl.tensor.Layout.TILE,
            1024,
            16,
            ttl.tensor.DataType.BFLOAT16,
        ),
        (  # AllGather + Fused QKV Matmul llama 2k prefill
            8,
            1,
            [1, 1, 2048, 8192],
            3,
            ttl.tensor.Layout.TILE,
            1280,
            8,
            ttl.tensor.DataType.BFLOAT16,
        ),
        (  # AllGather + FF1 Matmul llama 1k prefill
            8,
            1,
            [1, 1, 1024, 8192],
            3,
            ttl.tensor.Layout.TILE,
            4096,
            4,
            ttl.tensor.DataType.BFLOAT16,
        ),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
    ],
)
@pytest.mark.parametrize(
    "mem_config_ag, mem_config_mm",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_async",
    [
        True,
        # False,
    ],
)
def test_all_gather_matmul_on_t3000_post_commit(
    t3k_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    matmul_output_dim,
    matmul_config,
    matmul_weights_dtype,
    max_in0_block_w,
    mem_config_ag,
    mem_config_mm,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_matmul_on_t3000_impl(
        t3k_device_mesh,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        matmul_output_dim,
        matmul_config,
        matmul_weights_dtype,
        max_in0_block_w,
        mem_config_ag,
        mem_config_mm,
    )


# @skip_for_wormhole_b0() # Used to disable test
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "matmul_config",
    [
        "matmul_1d",
    ],
)
@pytest.mark.parametrize(
    "num_devices, num_links, ag_output_shape, dim, layout, matmul_output_dim, max_in0_block_w, matmul_weights_dtype",
    [
        (
            8,
            1,
            [1, 1, 32, 16 * 32],
            3,
            ttl.tensor.Layout.TILE,
            1024,
            2,
            ttl.tensor.DataType.BFLOAT16,
        ),
        (  # Llama decode FF1
            8,
            1,
            [1, 1, 32, 1024 * 8],
            3,
            ttl.tensor.Layout.TILE,
            4096,
            2,  # TODO: update
            ttl.tensor.DataType.BFLOAT4_B,
        ),
        (  # Llama decode Fused QKV
            8,
            1,
            [1, 1, 32, 1024 * 8],
            3,
            ttl.tensor.Layout.TILE,
            1280,
            2,  # TODO: update
            ttl.tensor.DataType.BFLOAT4_B,
        ),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
    ],
)
@pytest.mark.parametrize(
    "mem_config_ag, mem_config_mm",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_async",
    [
        True,
        False,
    ],
)
def test_all_gather_matmul_1d_on_t3000_post_commit(
    t3k_device_mesh,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    matmul_output_dim,
    matmul_config,
    matmul_weights_dtype,
    max_in0_block_w,
    mem_config_ag,
    mem_config_mm,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_matmul_on_t3000_impl(
        t3k_device_mesh,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        matmul_output_dim,
        matmul_config,
        matmul_weights_dtype,
        max_in0_block_w,
        mem_config_ag,
        mem_config_mm,
    )


# @skip_for_wormhole_b0() # Used to disable test
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "matmul_config",
    [
        "matmul_1d",
    ],
)
@pytest.mark.parametrize(
    "num_devices, num_links, ag_output_shape, dim, layout, matmul_output_dim, max_in0_block_w, matmul_weights_dtype",
    [
        (  # Llama decode Selfout
            8,
            1,
            [1, 1, 32, 1024 * 8],
            3,
            ttl.tensor.Layout.TILE,
            1024,
            8,
            ttnn.bfloat8_b,
        ),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
    ],
)
@pytest.mark.parametrize(
    "mem_config_ag, mem_config_mm, mem_config_weights",
    [
        (
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(7, 0),
                            ),
                        }
                    ),
                    [
                        32,  # shard_height
                        8192 // 8,  # shard_width_hidden_dim_across_8_cores
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_async",
    [
        True,
        False,
    ],
)
def test_all_gather_matmul_1d_llama_selfout_on_t3000_post_commit(
    t3k_device_mesh,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    matmul_output_dim,
    matmul_config,
    matmul_weights_dtype,
    max_in0_block_w,
    mem_config_ag,
    mem_config_mm,
    mem_config_weights,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_matmul_on_t3000_impl(
        t3k_mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        matmul_output_dim,
        matmul_config,
        matmul_weights_dtype,
        max_in0_block_w,
        mem_config_ag,
        mem_config_mm,
        mem_config_weights,
    )
