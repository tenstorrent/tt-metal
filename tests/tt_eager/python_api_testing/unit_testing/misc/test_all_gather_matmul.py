# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from ttnn import experimental as ttl
import ttnn
from ttnn import ShardTensorToMesh
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
import itertools


def is_unsupported_case(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout):
    if layout == ttl.tensor.Layout.ROW_MAJOR and input_dtype == ttl.tensor.DataType.BFLOAT8_B:
        return True, "Invalid combination"

    if num_devices < 2:
        return True, "Requires multiple devices to run"
    elif num_devices == 2 and num_links <= 2:
        return True, "Not enough links to run"

    if input_shape[dim] % num_devices != 0 or (dim == 3 and input_shape[dim] // num_devices % 32 != 0):
        return True, "Unsupported test case"

    ## Check that we can readback results
    fast_dispatch_page_size_limit = 55 * 1024
    elem_size = 2 if input_dtype == ttl.tensor.DataType.BFLOAT16 else 1
    if layout == ttl.tensor.Layout.ROW_MAJOR and (input_shape[dim] * elem_size) > fast_dispatch_page_size_limit:
        # Fast dispatch currently can't breakup readback of large pages into multiple smaller pages and is
        # limited to ~55K pages.
        return True, "Fast dispatch can't support reading back this page size in one shot"

    # Check that we can fit in L1 (if L1 config)
    tensor_size_bytes = elem_size
    for i in input_shape:
        tensor_size_bytes *= i
    num_l1_banks = 64
    if mem_config.buffer_type == ttl.tensor.BufferType.L1 and tensor_size_bytes > num_l1_banks * 50 * 1024:
        return True, "L1 buffer can't support large tensor sizes"

    # Check that each chip has a non-zero amount of data available
    min_sized_chunks_on_dim = input_shape[dim]
    if dim == 3:
        min_sized_chunks_on_dim //= 32
    if dim == 2:
        if layout == ttl.tensor.Layout.TILE:
            min_sized_chunks_on_dim //= 32
    if min_sized_chunks_on_dim < num_devices:
        return (
            True,
            f"Input shape {input_shape} incompatible with {num_devices} on dim {dim} because some chips will have no tensor",
        )

    if (
        input_shape == [8, 8, 256, 384]
        and dim == 1
        and layout == ttl.tensor.Layout.TILE
        and input_dtype == ttl.tensor.DataType.BFLOAT8_B
    ):
        return True, "Known failure"

    return False, ""


def run_all_gather_on_t3000_impl(
    t3k_device_mesh,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    matmul_output_dim,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
    enable_async=False,
):
    # Use Async mode based on test input config
    for device in t3k_device_mesh.get_devices():
        device.enable_async(enable_async)

    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")

    (is_known_failure, message) = is_unsupported_case(
        input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    devices = t3k_device_mesh.get_devices()

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    # Create input tensor for the all gather
    _, _, _, hidden_dim = input_shape
    input_tensor = torch.rand(input_shape).bfloat16()
    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(ttl.tensor.Tensor(t, input_dtype).to(layout).to(devices[i], mem_config))
    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

    # Create the weight matrix for the matmul
    weights_tensor = torch.rand([1, 1, hidden_dim, matmul_output_dim * num_devices]).bfloat16()
    weight_tt = ttnn.as_tensor(
        weights_tensor,
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        memory_config=mem_config,
        mesh_mapper=ShardTensorToMesh(t3k_device_mesh, dim=3),
    )

    # torch matmul output
    matmul_output = torch.chunk(torch.matmul(input_tensor, weights_tensor), num_devices, 3)

    # Configs for ttnn.matmul
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=16,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=32,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
        fused_activation=None,
        fuse_batch=True,
        mcast_in0=True,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Perform the ops
    for i in range(num_iters):
        # all_gather
        # tt_out_tensor = ttnn.all_gather(input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config)

        # matmul
        # tt_matmul_output = ttnn.matmul(
        #     tt_out_tensor,
        #     weight_tt,
        #     memory_config=mem_config,
        #     program_config=program_config,
        #     compute_kernel_config=compute_kernel_config,
        # )

        # Test ttnn all_gather_matmul
        tt_all_gather_out_tensor, _, tt_datacopy_out_tensor = ttnn.all_gather_matmul(
            input_tensor_mesh,
            weight_tt,
            dim,
            (0, 1),
            num_links=num_links,
            memory_config=mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

        logger.info(f"Done iteration {i}")

    # print("Checking outputs for All Gather")
    # for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
    #     tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    #     if input_dtype == ttl.tensor.DataType.BFLOAT16:
    #         eq, output = comp_equal(tt_output_tensor, input_tensor)
    #     else:
    #         eq, output = comp_pcc(tt_output_tensor, input_tensor)
    #     logger.info(f"Output {i}: {output}")
    #     if not eq:
    #         logger.error(f"output mismatch for tensor {i}")
    #     assert eq, f"{i} FAILED: {output}"

    # print("Checking outputs for Matmul")
    # for i, t in enumerate(ttnn.get_device_tensors(tt_matmul_output)):
    #     tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    #     eq, output = comp_pcc(tt_output_tensor, matmul_output[i])
    #     logger.info(f"Output {i}: {output}")
    #     if not eq:
    #         logger.error(f"output mismatch for tensor {i}")
    #     assert eq, f"{i} FAILED: {output}"

    print("Checking outputs for All Gather Matmul (All Gather)")
    for i, t in enumerate(ttnn.get_device_tensors(tt_all_gather_out_tensor)):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        if input_dtype == ttl.tensor.DataType.BFLOAT16:
            eq, output = comp_equal(tt_output_tensor, input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, input_tensor)
        logger.info(f"Output {i}: {output}")
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        assert eq, f"{i} FAILED: {output}"

    print("Checking outputs for All Gather Matmul (Datacopy)")
    for i, t in enumerate(ttnn.get_device_tensors(tt_datacopy_out_tensor)):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        if input_dtype == ttl.tensor.DataType.BFLOAT16:
            eq, output = comp_equal(tt_output_tensor, input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, input_tensor)
        logger.info(f"Output {i}: {output}")
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
    assert eq, f"{i} FAILED: {output}"


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout, matmul_output_dim",
    [
        # (8, 1, [1, 1, 32, 512], 3, ttl.tensor.Layout.TILE, 1024),  # https://github.com/tenstorrent/tt-metal/issues/9686
        (
            8,
            1,
            [1, 1, 32, 16 * 32],
            3,
            ttl.tensor.Layout.TILE,
            1024,
        ),
        (
            8,
            1,
            [1, 1, 128, 128 * 32],
            3,
            ttl.tensor.Layout.TILE,
            1024,
        ),
        (
            8,
            1,
            [1, 1, 1024, 1024 * 32],
            3,
            ttl.tensor.Layout.TILE,
            1024,
        ),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
        # ttl.tensor.DataType.BFLOAT8_B,          # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.DRAM),
        # ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1),  # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
def test_all_gather_on_t3000_post_commit(
    t3k_device_mesh,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    matmul_output_dim,
    mem_config,
    use_program_cache,
    function_level_defaults,
):
    run_all_gather_on_t3000_impl(
        t3k_device_mesh,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        matmul_output_dim,
        mem_config,
        use_program_cache,
        function_level_defaults,
    )
