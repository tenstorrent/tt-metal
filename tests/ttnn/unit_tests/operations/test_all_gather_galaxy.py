# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import tt_lib as ttl
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
import itertools

from tests.ttnn.unit_tests.operations.test_all_gather import is_unsupported_case

from ttnn import ShardTensorToMesh, ShardTensor2dMesh


def run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
    device_mesh,
    num_devices_per_line,
    input_shape_per_all_gather,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_all_gather_instances=1,
    num_iters=1,
    cluster_axis=0,
):
    if len(device_mesh.get_devices()) != 32:
        pytest.skip("Not TG!")
    for device in device_mesh.get_devices():
        device.enable_async(enable_async)

    full_mesh_input_shape = input_shape_per_all_gather
    tensor_height_per_all_gather = input_shape_per_all_gather[-2]
    full_mesh_input_shape[-2] *= num_all_gather_instances

    full_tensor = torch.zeros(full_mesh_input_shape, dtype=torch.bfloat16)

    for i in range(num_all_gather_instances):
        full_tensor[0, 0, i * tensor_height_per_all_gather : (i + 1) * tensor_height_per_all_gather, :] = torch.rand(
            input_shape_per_all_gather
        ).bfloat16()

    shard_dims = (-1, -2) if cluster_axis == 0 else (-2, -1)
    ttnn_tensor = ttnn.from_torch(
        full_tensor,
        dtype=input_dtype,
        device=device_mesh,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ShardTensor2dMesh(
            device_mesh, shard_grid=(num_all_gather_instances, num_devices_per_line), shard_dimensions=shard_dims
        ),
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device_mesh)

    for _ in range(num_iters):
        ttnn_tensor = ttnn.line_all_gather(
            ttnn_tensor, dim=dim, cluster_axis=cluster_axis, device_mesh=device_mesh, num_links=num_links
        )

    tt_output_tensor = ttnn.to_torch(mesh_composer=ttnn.from_device(ttnn_tensor))
    if input_dtype == ttl.tensor.DataType.BFLOAT16:
        eq, output = comp_equal(tt_output_tensor, full_tensor)
    else:
        eq, output = comp_pcc(tt_output_tensor, full_tensor)
    if not eq:
        logger.error(f"output mismatch for tensor")
    assert eq, f"FAILED: {output}"


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (8, 1, [1, 1, 32, 1280], 1, ttl.tensor.Layout.TILE),
        (8, 1, [1, 1, 32, 2048], 1, ttl.tensor.Layout.TILE),
        (8, 1, [1, 1, 32, 2304], 1, ttl.tensor.Layout.TILE),
        (8, 1, [1, 1, 32, 4096], 1, ttl.tensor.Layout.TILE),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,  # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttl.tensor.MemoryConfig(
            buffer_type=ttl.tensor.BufferType.DRAM
        ),  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1),
    ],
)
@pytest.mark.parametrize("replication_factor", [1])  # , 2, 3, 8])
@pytest.mark.parametrize("enable_async", [True, False])
@pytest.mark.parametrize("device_mesh", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_line_all_gather_on_TG_rows_post_commit(
    device_mesh,
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
    replication_factor,
    num_iters=1,
):
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        device_mesh,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        num_all_gather_instances=replication_factor,
        cluster_axis=0,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (4, 1, [1, 1, 32, 16384], 3, ttl.tensor.Layout.TILE),
        # (4, 1, [1, 1, 32, 2304], 1, ttl.tensor.Layout.TILE),
        # (4, 1, [1, 1, 32, 4096], 1, ttl.tensor.Layout.TILE),
        # (4, 1, [1, 1, 32, 6656], 1, ttl.tensor.Layout.TILE),
        # (4, 3, [1, 1, 32, 16384], 3, ttl.tensor.Layout.TILE),
        # (4, 3, [1, 1, 32, 2304], 1, ttl.tensor.Layout.TILE),
        # (4, 3, [1, 1, 32, 4096], 1, ttl.tensor.Layout.TILE),
        # (4, 3, [1, 1, 32, 6656], 1, ttl.tensor.Layout.TILE),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
        # ttl.tensor.DataType.BFLOAT8_B, # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttl.tensor.MemoryConfig(
            buffer_type=ttl.tensor.BufferType.DRAM
        ),  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("replication_factor", [1])  # , 2, 3, 8])
@pytest.mark.parametrize("device_mesh", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_line_all_gather_on_TG_cols_post_commit(
    device_mesh,
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
    replication_factor,
    num_iters=1,
):
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        device_mesh,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        num_all_gather_instances=replication_factor,
        cluster_axis=1,
    )
