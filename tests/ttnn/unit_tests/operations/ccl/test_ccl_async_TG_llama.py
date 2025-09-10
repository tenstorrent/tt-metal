# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
import os

from conftest import is_6u
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.tests_common.skip_reasons import LEGACY_CCL_SKIP
from models.utility_functions import skip_for_grayskull
from ttnn import ShardTensor2dMesh, ConcatMesh2dToTensor
from models.perf.benchmarking_utils import BenchmarkProfiler
from tracy import signpost


from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import (
    run_all_reduce_impl,
    RING_CRS,
    NORM_CRS,
    LM_HEAD_CRS,
    QKV_CRS,
    FF1_CRS,
)

from tests.ttnn.unit_tests.operations.ccl.test_llama_all_gather_matmul import (
    run_llama_all_gather_matmul_impl,
    BINARY_MULT_CRS,
)
from models.demos.llama3_70b_galaxy.tt.model_config import (
    PREFETCHER_NOC1_GRID,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler


NUM_ITERATIONS = 75


NUM_BUFFERS = 16


def report_mismatches(golden, actual, max_printable=None):
    printed = 0
    for w in range(golden.shape[0]):
        for z in range(golden.shape[1]):
            for y in range(0, golden.shape[2], 32):
                for x in range(0, golden.shape[3], 32):
                    print_it = (max_printable is None or printed < max_printable) and golden[w, z, y, x] != actual[
                        w, z, y, x
                    ]
                    if print_it:
                        printed += 1
                        print(
                            f"output mismatch for tensor at [{w}, {z}, {y}, {x}]: expected {golden[w, z, y, x]} != actual {actual[w, z, y, x]}"
                        )


def print_tile_corners_of_tensor(t):
    for w in range(t.shape[0]):
        for z in range(t.shape[1]):
            str = ""
            for x in range(0, t.shape[3], 32):
                str += f"{x:<5} "[:5]
            print(f"     {str}")
            for y in range(0, t.shape[2], 32):
                str_vals = f"y={y:<3} "[:5]
                for x in range(0, t.shape[3], 32):
                    yy = 0
                    xx = 0
                    val = int(t[w, z, y + yy, x + xx].item())
                    str_vals += f"{val:<5} "[:5]
                print(f"{str_vals}")


def run_with_trace(
    mesh_device,
    all_gather_topology,
    input_tensor,
    dim,
    persistent_output_tensor,
    num_links,
    cluster_axis,
    output_mem_config,
    ccl_semaphore_handles,
    worker_sub_device_id,
    n_worker=None,
    n_buffer=None,
    num_iter=20,
    warmup_iters=0,
    use_all_gather_async=False,
    profiler=BenchmarkProfiler(),
):
    # Compile Run
    logger.info("Compiling model")
    if use_all_gather_async:
        tt_out_tensor = ttnn.experimental.all_gather_async(
            input_tensor,
            dim,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=all_gather_topology,
            multi_device_global_semaphore=ccl_semaphore_handles[0],
            persistent_output_tensor=persistent_output_tensor,
            num_links=num_links,
            memory_config=output_mem_config,
            subdevice_id=worker_sub_device_id,
        )
    else:
        # Legacy ccl call removed until new implementation is done - see https://github.com/tenstorrent/tt-metal/issues/26649
        pytest.skip(LEGACY_CCL_SKIP)
        # tt_out_tensor = ttnn.all_gather(
        #     input_tensor,
        #     dim=dim,
        #     cluster_axis=cluster_axis,
        #     mesh_device=mesh_device,
        #     num_links=num_links,
        #     memory_config=output_mem_config,
        #     topology=all_gather_topology,
        # )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")

    def capture_trace(n_iters):
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(n_iters):
            if use_all_gather_async:
                tt_out_tensor = ttnn.experimental.all_gather_async(
                    input_tensor,
                    dim,
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    topology=all_gather_topology,
                    multi_device_global_semaphore=ccl_semaphore_handles[i % NUM_BUFFERS],
                    persistent_output_tensor=persistent_output_tensor,
                    num_links=num_links,
                    memory_config=output_mem_config,
                    subdevice_id=worker_sub_device_id,
                )
            else:
                # Legacy ccl call removed until new implementation is done - see https://github.com/tenstorrent/tt-metal/issues/26649
                pytest.skip(LEGACY_CCL_SKIP)
                # tt_out_tensor = ttnn.all_gather(
                #     input_tensor,
                #     dim=dim,
                #     cluster_axis=cluster_axis,
                #     mesh_device=mesh_device,
                #     num_links=num_links,
                #     memory_config=output_mem_config,
                #     topology=all_gather_topology,
                # )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        return trace_id

    if warmup_iters > 0:
        trace_id_warmup = capture_trace(warmup_iters)
    trace_id = capture_trace(num_iter)

    # Run the op
    logger.info("Starting Trace perf test...")
    profiler.start("all-gather-async-trace-warmup")
    if warmup_iters > 0:
        ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
        ttnn.release_trace(mesh_device, trace_id_warmup)
        ttnn.synchronize_device(mesh_device)
    profiler.end("all-gather-async-trace-warmup")

    profiler.start("all-gather-async-trace")
    signpost("start")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    profiler.end("all-gather-async-trace")
    time_taken = profiler.get_duration("all-gather-async-trace") - profiler.get_duration(
        "all-gather-async-trace-warmup"
    )
    effective_iter = num_iter - warmup_iters
    logger.info(f"Time taken e2e: {time_taken} s")
    logger.info(f"Time per iter e2e: {time_taken / effective_iter} s")
    logger.info(f"Time per iter e2e: {time_taken / effective_iter * 1e6} us")

    return tt_out_tensor


def run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
    mesh_device,
    num_devices_per_line,
    per_chip_output_shape,
    tensor_memory_layout,
    dim,
    num_links,
    input_dtype,
    layout,
    buffer_type: ttnn.BufferType,
    function_level_defaults,
    input_shard_spec: ttnn.ShardSpec = None,
    output_shard_spec: ttnn.ShardSpec = None,
    num_all_gather_instances: int = 1,
    num_iters: int = 1,
    warmup_iters: int = 0,
    cluster_axis: int = 0,
    tile=(32, 32),
    trace_mode=False,
    debug=False,
    profiler=BenchmarkProfiler(),
    # New all-gather-async and persistent fabric params
    use_all_gather_async=False,
    use_persistent_output=False,
    linear=True,
):
    if use_persistent_output and not use_all_gather_async:
        pytest.skip("Persistent output tensor requires all-gather-async")

    input_shape_per_chip = list(per_chip_output_shape)
    input_shape_per_chip[dim] //= num_devices_per_line
    tensor_height_per_all_gather = per_chip_output_shape[-2]

    full_mesh_input_shape = list(per_chip_output_shape)
    ## The `all_gather_instances_concat_dim` is the dimension we will split the cluster spanning tensor along in order to split it
    ## off into per-all-gather tensors
    all_gather_instances_concat_dim = 1 if dim == 0 else 0
    full_mesh_input_shape[all_gather_instances_concat_dim] *= num_all_gather_instances
    logger.info(
        f"per_chip_output_shape: {full_mesh_input_shape}, dim: {dim}, all_gather_instances_concat_dim: {all_gather_instances_concat_dim}, num_devices_per_line: {num_devices_per_line}"
    )

    all_gather_instances_goldens = []
    full_input_tensor_unfractured = torch.rand(full_mesh_input_shape, dtype=torch.bfloat16)

    input_mem_config = ttnn.MemoryConfig(tensor_memory_layout, buffer_type=buffer_type, shard_spec=input_shard_spec)
    shard_dims = (dim, all_gather_instances_concat_dim) if cluster_axis == 0 else (all_gather_instances_concat_dim, dim)
    concat_dims = shard_dims

    mesh_shape = (
        (num_devices_per_line, num_all_gather_instances)
        if cluster_axis == 0
        else (num_all_gather_instances, num_devices_per_line)
    )

    if input_shard_spec is not None and output_shard_spec is None:
        output_shard_shape = list(input_shard_spec.shape)
        if dim == len(per_chip_output_shape) - 1:
            output_shard_shape[1] *= num_devices_per_line
        else:
            output_shard_shape[0] *= num_devices_per_line
        output_shard_spec = ttnn.ShardSpec(
            input_shard_spec.grid,
            output_shard_shape,
            input_shard_spec.orientation,
        )
    output_mem_config = ttnn.MemoryConfig(tensor_memory_layout, buffer_type=buffer_type, shard_spec=output_shard_spec)
    ttnn_tensor = ttnn.from_torch(
        full_input_tensor_unfractured,
        tile=ttnn.Tile(tile),
        dtype=input_dtype,
        device=mesh_device,
        layout=layout,
        memory_config=input_mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dims),
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    ttnn_tensor = ttnn.to_memory_config(ttnn_tensor, input_mem_config)
    if linear:
        all_gather_topology = ttnn.Topology.Linear
        wrap_mesh = False
    else:
        all_gather_topology = ttnn.Topology.Ring
        wrap_mesh = False

    ttnn_persistent_output_tensor = None
    if use_persistent_output:
        ttnn_persistent_output_tensor = ttnn.from_torch(
            torch.zeros(per_chip_output_shape),
            tile=ttnn.Tile(tile),
            dtype=input_dtype,
            device=mesh_device,
            layout=layout,
            memory_config=output_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    sub_device_stall_group = []
    if use_all_gather_async:
        compute_grid_size = mesh_device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice(
            [
                ccl_sub_device_crs,
            ]
        )
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_stall_group = [worker_sub_device_id]
        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        mesh_device.set_sub_device_stall_group(sub_device_stall_group)
        # create global semaphore handles
        ccl_semaphore_handles = [
            [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]
            for _ in range(NUM_BUFFERS)
        ]
    try:
        # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor)
        if trace_mode:
            ttnn_tensor_out = run_with_trace(
                input_tensor=ttnn_tensor,
                dim=dim,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                persistent_output_tensor=ttnn_persistent_output_tensor,
                num_links=num_links,
                output_mem_config=output_mem_config,
                ccl_semaphore_handles=ccl_semaphore_handles,
                worker_sub_device_id=worker_sub_device_id,
                all_gather_topology=all_gather_topology,
                num_iter=num_iters,
                warmup_iters=warmup_iters,
                use_all_gather_async=use_all_gather_async,
                profiler=profiler,
            )

        else:
            signpost("start")
            for i in range(num_iters):
                if use_all_gather_async:
                    logger.info("Running all-gather async")
                    ttnn_tensor_out = ttnn.experimental.all_gather_async(
                        ttnn_tensor,
                        dim,
                        cluster_axis=cluster_axis,
                        mesh_device=mesh_device,
                        topology=all_gather_topology,
                        multi_device_global_semaphore=ccl_semaphore_handles[i % NUM_BUFFERS],
                        persistent_output_tensor=ttnn_persistent_output_tensor,
                        num_links=num_links,
                        memory_config=output_mem_config,
                        subdevice_id=worker_sub_device_id,
                    )
                else:
                    # Legacy ccl call removed until new implementation is done - see https://github.com/tenstorrent/tt-metal/issues/26649
                    pytest.skip(LEGACY_CCL_SKIP)
                    # ttnn_tensor_out = ttnn.all_gather(
                    #     ttnn_tensor,
                    #     dim=dim,
                    #     cluster_axis=cluster_axis,
                    #     mesh_device=mesh_device,
                    #     num_links=num_links,
                    #     memory_config=output_mem_config,
                    #     topology=all_gather_topology,
                    # )
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            signpost("stop")
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise e
    finally:
        mesh_device.reset_sub_device_stall_group()

    # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor_out)
    tt_output_tensor = ttnn.to_torch(
        ttnn_tensor_out, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=concat_dims)
    )
    output_tensors_list = torch.chunk(tt_output_tensor, num_all_gather_instances, dim=all_gather_instances_concat_dim)
    output_golden = torch.zeros(tt_output_tensor.shape)

    # Check the tensor addresses
    if use_persistent_output:
        persistent_output_tensors = ttnn.get_device_tensors(ttnn_persistent_output_tensor)
        output_tensors = ttnn.get_device_tensors(ttnn_tensor_out)

        for persistent_tensor, output_tensor in zip(persistent_output_tensors, output_tensors):
            assert (
                persistent_tensor.buffer_address() == output_tensor.buffer_address()
            ), "Persistent tensor address mismatch"

    # Repeat the input tensor to represent the fact that the full concatenated input tensor lives across every
    # device in the line
    repeat_factor = [1] * len(output_golden.shape)
    repeat_factor[dim] = num_devices_per_line
    output_golden[:, :, :, :] = full_input_tensor_unfractured.repeat(repeat_factor)

    eq = True
    if input_dtype == ttnn.bfloat16:
        eq, output = comp_equal(tt_output_tensor, output_golden)
        if not eq and debug is True:
            logger.error(f"found mismatches")
            report_mismatches(tt_output_tensor, output_golden, 100)
            print_tile_corners_of_tensor(tt_output_tensor)
    else:
        eq, output = comp_pcc(tt_output_tensor, output_golden)
    if not eq:
        logger.error(f"output mismatch for tensor: {output}")

    assert eq, f"FAILED: {output}"


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


@pytest.mark.skipif(not is_6u(), reason="This test is only for 6U devices")
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
            4,
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
            4,
            3,
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 4))}),
            (32, 160),
            get_core_range_set(PREFETCHER_NOC1_GRID),
            ttnn.TILE_LAYOUT,
            ttnn.bfloat8_b,
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
    "device_params", [{"trace_region_size": 17068032, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True
)
def test_all_gather_6u_llama(
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
        linear=False,
    )


# Enumerate the post-commit cases explicitly
@pytest.mark.skipif(is_6u(), reason="This test is not for 6U devices")
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


@pytest.mark.skipif(is_6u(), reason="This test is not for 6U devices")
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "output_shape, cluster_axis, num_links, input_num_cores, input_core_range_set, output_num_cores, output_core_range_set, input_dtype, output_dtype",
    [
        ([1, 1, 32, 2048], 0, 3, 24, RING_CRS, 16, NORM_CRS, ttnn.bfloat8_b, None),  # FF2/DO all reduce
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


@pytest.mark.skipif(not is_6u(), reason="This test is only for 6U devices")
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "output_shape, cluster_axis, num_links, input_num_cores, input_core_range_set, output_num_cores, output_core_range_set, input_dtype, output_dtype",
    [
        ([1, 1, 32, 2048], 0, 4, 24, RING_CRS, 16, NORM_CRS, ttnn.bfloat8_b, None),  # FF2/DO all reduce
        ([1, 1, 32, 1280], 1, 4, 24, RING_CRS, 10, QKV_CRS, ttnn.bfloat8_b, ttnn.bfloat16),  # QKV all reduce
        ([1, 1, 32, 3584], 1, 4, 24, RING_CRS, 28, FF1_CRS, ttnn.bfloat8_b, None),  # FF1 all reduce
        ([1, 1, 32, 16 * 1024], 1, 4, 32, LM_HEAD_CRS, 32, LM_HEAD_CRS, ttnn.bfloat8_b, None),  # LM head all reduce
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
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
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
def test_all_reduce_6U_llama(
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
    function_level_defaults,
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
        linear=False,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "cluster_axis, num_links, input_num_cores, input_core_range_set, output_num_cores, output_core_range_set",
    [
        (
            1,
            3,
            30,
            BINARY_MULT_CRS,
            24,
            RING_CRS,
        ),
    ],
    ids=[
        "binary_mult",
    ],
)
@pytest.mark.parametrize(
    "num_iters",
    [
        (NUM_ITERATIONS),
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
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, output_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid, in1_is_dram_interleaved",
    [
        (
            1,
            32,
            3584,
            2048,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
            PREFETCHER_NOC1_GRID,
            False,
        ),
    ],
    ids=[
        "ff2",
    ],
)
def test_llama_all_gather_matmul(
    mesh_device,
    cluster_axis,
    in0_dtype,
    in1_dtype,
    output_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    grid,
    in1_is_dram_interleaved,
    num_links,
    input_num_cores,
    input_core_range_set,
    output_num_cores,
    output_core_range_set,
    num_iters,
    trace_mode,
    function_level_defaults,
    ensure_devices_tg,
):
    run_llama_all_gather_matmul_impl(
        mesh_device,
        # shape params shared by AG and MM
        B,
        M,
        K,
        N,
        cluster_axis,
        in0_dtype,
        # MM params for in1
        in1_dtype,
        num_links,
        input_num_cores,
        input_core_range_set,
        output_num_cores,
        output_core_range_set,
        # rest of mm params
        output_dtype,
        fidelity,
        has_bias,
        fp32_acc_mode,
        packer_l1_acc,
        grid,
        in1_is_dram_interleaved,
        # common params
        num_iters=num_iters,
        trace_mode=trace_mode,
        validate_all=False,
    )
