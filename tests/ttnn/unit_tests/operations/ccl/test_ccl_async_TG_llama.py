# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.utility_functions import skip_for_grayskull

from tests.ttnn.unit_tests.operations.ccl.test_all_gather_TG_post_commit import (
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows,
)
from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import (
    run_all_reduce_impl,
    RING_CRS,
    NORM_CRS,
    LM_HEAD_CRS,
    QKV_CRS,
    FF1_CRS,
)
from models.demos.llama3_subdevices.tt.model_config import (
    PREFETCHER_NOC1_GRID,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler


NUM_ITERATIONS = 75


def get_core_range_set(output_core_grid):
    if isinstance(output_core_grid, ttnn.CoreGrid):
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(output_core_grid.x - 1, output_core_grid.y - 1)),
            ]
        )
    else:
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in output_core_grid
            ]
        )
    return output_core_range_set


CORE_RANGE_SET_1x1 = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
    }
)


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices",
    [
        4,
    ],
)
@pytest.mark.parametrize(
    "num_iters, warmup_iters",
    [
        (NUM_ITERATIONS, 10),
    ],
)
@pytest.mark.parametrize("shard_grid_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "tensor_mem_layout, output_shape, num_links, dim, input_shard_shape,input_shard_grid,output_shard_shape, output_shard_grid, layout, input_dtype",
    (
        (  # AllGather after SDPA
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (1, 32, 32, 128),
            3,
            1,
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (32, 128),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 1)),
                }
            ),
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
        ),
        (  # AllGather after Binary Mult+Silu
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 1, 32, 3840),
            3,
            3,
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 4))}),
            (32, 160),
            get_core_range_set(PREFETCHER_NOC1_GRID),
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
        ),
        (  # AllGather for layernorm
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 1, 32, 128),
            1,
            3,
            (32, 32),
            CORE_RANGE_SET_1x1,
            (32, 128),
            CORE_RANGE_SET_1x1,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
        ),
        (  # AllGather for sampling values
            ttnn.TensorMemoryLayout.INTERLEAVED,
            (1, 1, 32, 256),
            1,
            3,
            None,
            None,
            None,
            None,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
        ),
        (  # AllGather for sampling indices
            ttnn.TensorMemoryLayout.INTERLEAVED,
            (1, 1, 32, 256),
            1,
            3,
            None,
            None,
            None,
            None,
            ttnn.TILE_LAYOUT,
            ttnn.uint16,
        ),
    ),
    ids=[
        "sdpa",
        "binary_mult",
        "layernorm",
        "sampling_values",
        "sampling_indices",
    ],
)
@pytest.mark.parametrize("replication_factor", [8])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 17068032, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_all_gather_tg_llama(
    mesh_device,
    num_devices,
    output_shape,
    input_shard_shape,
    input_shard_grid,
    output_shard_shape,
    output_shard_grid,
    shard_grid_orientation,
    tensor_mem_layout,
    dim,
    num_links,
    input_dtype,
    layout,
    use_program_cache,
    function_level_defaults,
    replication_factor,
    num_iters,
    warmup_iters,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")
    if input_shard_grid is not None and input_shard_shape is not None:
        input_shard_spec = ttnn.ShardSpec(
            input_shard_grid,
            input_shard_shape,
            shard_grid_orientation,
        )
    else:
        input_shard_spec = None

    if output_shard_grid is not None and output_shard_shape is not None:
        output_shard_spec = ttnn.ShardSpec(
            output_shard_grid,
            output_shard_shape,
            shard_grid_orientation,
        )
    else:
        output_shard_spec = None

    profiler = BenchmarkProfiler()

    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        output_shape,
        tensor_mem_layout,
        dim,
        num_links,
        input_dtype,
        layout,
        ttnn.BufferType.L1,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        input_shard_spec=input_shard_spec,
        output_shard_spec=output_shard_spec,
        num_all_gather_instances=replication_factor,
        cluster_axis=1,
        profiler=profiler,
        trace_mode=True,
        use_all_gather_async=True,
        use_persistent_output=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "output_shape, cluster_axis, num_links, input_num_cores, input_core_range_set, output_num_cores, output_core_range_set, input_dtype, output_dtype",
    [
        ([1, 1, 32, 2048], 0, 4, 24, RING_CRS, 16, NORM_CRS, ttnn.bfloat8_b, None),  # FF2/DO all reduce
        ([1, 1, 32, 1280], 1, 3, 24, RING_CRS, 10, QKV_CRS, ttnn.bfloat8_b, ttnn.bfloat16),  # QKV all reduce
        ([1, 1, 32, 3584], 1, 3, 24, RING_CRS, 28, FF1_CRS, ttnn.bfloat8_b, None),  # FF1 all reduce
        ([1, 1, 32, 16 * 1024], 1, 3, 32, LM_HEAD_CRS, 32, LM_HEAD_CRS, ttnn.bfloat8_b, None),  # LM head all reduce
    ],
    ids=[
        "ff2",
        "qkv",
        "ff1",
        "lm_head",
    ],
)
@pytest.mark.parametrize(
    "num_iters, warmup_iters",
    [
        (NUM_ITERATIONS, 10),
    ],
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_all_reduce_tg_llama(
    mesh_device,
    output_shape,
    cluster_axis,
    input_dtype,
    output_dtype,
    num_links,
    input_num_cores,
    input_core_range_set,
    output_num_cores,
    output_core_range_set,
    num_iters,
    warmup_iters,
    trace_mode,
    use_program_cache,
    function_level_defaults,
    ensure_devices_tg,
):
    profiler = BenchmarkProfiler()

    run_all_reduce_impl(
        mesh_device,
        output_shape,
        cluster_axis,
        input_dtype,
        num_links,
        input_num_cores,
        input_core_range_set,
        output_num_cores,
        output_core_range_set,
        output_dtype=output_dtype,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        trace_mode=trace_mode,
        validate_all=False,
        profiler=profiler,
    )
