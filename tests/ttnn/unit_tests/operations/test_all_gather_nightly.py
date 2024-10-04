# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
from tests.ttnn.unit_tests.operations.test_all_gather import (
    is_unsupported_case,
    run_all_gather_on_t3000_impl,
)
from ttnn import ShardTensorToMesh


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (4, 1, [4, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
        (8, 1, [8, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
        (8, 1, [8, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
        (8, 1, [8, 8, 256, 384], 1, ttnn.ROW_MAJOR_LAYOUT),
        # (4, 2, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT),
        (8, 1, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT),
        (4, 1, [8, 5, 13, 384], 3, ttnn.ROW_MAJOR_LAYOUT),
        (8, 1, [8, 5, 13, 512], 3, ttnn.ROW_MAJOR_LAYOUT),
        (4, 1, [8, 5, 32, 384], 3, ttnn.TILE_LAYOUT),
        (8, 1, [8, 5, 32, 512], 3, ttnn.TILE_LAYOUT),
        (4, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_line_all_gather_on_t3000_nightly(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    run_all_gather_on_t3000_impl(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Linear,
        enable_async=enable_async,
        num_iters=num_iters,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (
            4,
            2,
            [8, 8, 256, 384],
            1,
            ttnn.TILE_LAYOUT,
        ),  # test cases with num_links = 2 is currently not supported by new mesh fixture
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_line_all_gather_on_t3000_nightly_two_link(
    pcie_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    run_all_gather_on_t3000_impl(
        pcie_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        enable_async=enable_async,
    )


def run_line_all_gather_instances(
    t3k_mesh_device,
    num_devices,
    num_instances,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    if t3k_mesh_device.get_num_devices() != 8:
        pytest.skip("Not T3000!")

    t3k_mesh_device.enable_async(enable_async)

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    (is_known_failure, message) = is_unsupported_case(
        input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    t3k_device = []

    for device in t3k_mesh_device.get_devices():
        t3k_device.append(device)

    t3000_device_rows = [
        [t3k_device[4], t3k_device[0], t3k_device[3], t3k_device[7]],
        [t3k_device[5], t3k_device[1], t3k_device[2], t3k_device[6]],
    ]
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    input_tensor = torch.rand(input_shape).bfloat16()

    ttnn_tensor = ttnn.from_torch(input_tensor, mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=dim))
    input_tensor_mesh = ttnn.to_device(ttnn_tensor, t3k_mesh_device)

    result_mesh_tensors = []
    for loop in range(num_iters):
        for i, devices in enumerate(t3000_device_rows):
            tt_out_tensor = ttnn.all_gather(
                input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config, topology=ttnn.Topology.Linear
            )
            result_mesh_tensors.append(tt_out_tensor)

    for loop in range(num_iters):
        ## Wait for completion
        for i, devices in enumerate(t3000_device_rows):
            for d in devices:
                ttnn.synchronize_device(d)

        for tt_out_tensor in result_mesh_tensors:
            for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
                tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
                if input_dtype == ttnn.bfloat16:
                    eq, output = comp_equal(tt_output_tensor, input_tensor)
                else:
                    eq, output = comp_pcc(tt_output_tensor, input_tensor)
                if not eq:
                    logger.error(f"output mismatch for tensor {i}")
                assert eq, f"{i} FAILED: {output}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_instances, num_links, input_shape, dim, layout",
    [
        (4, 1, 1, [4, 1, 33, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
        # (4, 1, 2, [8, 8, 256, 384], 1, ttnn.TILE_LAYOUT),
        (4, 1, 1, [8, 5, 13, 384], 3, ttnn.ROW_MAJOR_LAYOUT),
        (4, 1, 1, [8, 5, 32, 384], 3, ttnn.TILE_LAYOUT),
        (4, 1, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
        (4, 2, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_line_all_gather_on_t3000_nightly_instances(
    t3k_mesh_device,
    num_devices,
    num_instances,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    run_line_all_gather_instances(
        t3k_mesh_device,
        num_devices,
        num_instances,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        enable_async,
        num_iters,
    )
