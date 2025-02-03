# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)

from tests.ttnn.unit_tests.operations.ccl.test_all_gather_TG_post_commit import (
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows,
)
from tests.ttnn.unit_tests.operations.ccl.test_reduce_scatter_TG_nightly import (
    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows,
)
from tests.ttnn.unit_tests.operations.ccl.test_all_reduce_async import (
    run_all_reduce_with_mesh_tensor_along_row,
)


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 1),
    ],
    # [(4, 3), (4, 2)], Multi-links fails https://github.com/tenstorrent/tt-metal/issues/16699
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,  # hang??
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("shard_grid_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "tensor_mem_layout, output_shape, dim, input_shard_shape,shard_grid,layout",
    (
        (  # AllGather after SDPA (~160 us)
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (1, 32, 32, 128),
            1,
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            ttnn.TILE_LAYOUT,
        ),
        (  # AllGather after Binary Mult+Silu (~160 us)
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 1, 32, 3840),
            3,
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 4))}),
            ttnn.TILE_LAYOUT,
        ),
    ),
)
@pytest.mark.parametrize("replication_factor", [8])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_line_all_gather_sharded_on_TG_rows_llama(
    mesh_device,
    num_devices,
    output_shape,
    input_shard_shape,
    shard_grid,
    shard_grid_orientation,
    tensor_mem_layout,
    dim,
    num_links,
    input_dtype,
    layout,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=10,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        input_shard_shape,
        shard_grid_orientation,
    )

    logger.warning("sharding not used due to issue #16699")

    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        output_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,  # tensor_mem_layout,
        dim,
        num_links,
        input_dtype,
        layout,
        ttnn.BufferType.L1,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        # input_shard_spec=input_shard_spec,
        num_all_gather_instances=replication_factor,
        cluster_axis=1,
        use_all_gather_async=True,
        enable_persistent_fabric=True,
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 2),
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout, per_chip_input_shape, dim, input_shard_shape,shard_grid,layout",
    (
        (  # ReduceScatter After FF1/3  (~100 us)
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 1, 32, 3840),
            3,
            (32, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))}),
            ttnn.TILE_LAYOUT,
        ),
    ),
)
@pytest.mark.parametrize("shard_grid_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("replication_factor", [8])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_line_reduce_scatter_sharded_on_TG_rows_llama(
    mesh_device,
    num_devices,
    per_chip_input_shape,
    tensor_mem_layout,
    input_shard_shape,
    shard_grid,
    shard_grid_orientation,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=10,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        input_shard_shape,
        shard_grid_orientation,
    )

    logger.warning("sharding not used due to issue #16699")

    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_input_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,  # tensor_mem_layout,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        # input_shard_spec=input_shard_spec,
        num_iters=num_iters,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=1,
        use_reduce_scatter_async=True,
        enable_persistent_fabric=True,
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, layout",
    [
        (4, 2, [1, 1, 32, 1280], ttnn.TILE_LAYOUT),  # AllReduce after QKV (~110 us)
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("replication_factor", [8])  # 1, 8])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_line_all_reduce_on_TG_rows_llama(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=10,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")

    logger.warning("sharding not used due to issue #16699")

    run_all_reduce_with_mesh_tensor_along_row(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        num_all_reduce_instances=replication_factor,
        cluster_axis=1,
        enable_persistent_fabric=True,
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, layout",
    [
        (8, 1, [1, 1, 32, 2048], ttnn.TILE_LAYOUT),  # AllReduce after DO and AllReduce after FF2 (~240 us)
        # multi-links fail https://github.com/tenstorrent/tt-metal/issues/16699
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_line_all_reduce_on_TG_cols_llama(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=10,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")

    logger.warning("sharding not used due to issue #16699")

    run_all_reduce_with_mesh_tensor_along_row(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        num_all_reduce_instances=replication_factor,
        cluster_axis=0,
        enable_persistent_fabric=True,
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
    )
