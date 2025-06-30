# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.utility_functions import skip_for_grayskull
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import (
    SUB_DEVICE_CRS,
    QKV_CRS,
    RING_CRS,
    FF1_CRS,
    FF1_CRS_RS_OUT,
    NORM_CRS,
)
from tracy import signpost

PACKET_WORKER_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 2)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(2, 3)),
    ]
)


def gen_tensor(dim, shard_height, shard_width, num_devices_scatter, num_devices_fracture, num_cores, scheme="random"):
    torch.manual_seed(2005)
    factor = 0
    torch_fracture_tensors = []
    for _ in range(num_devices_fracture):
        torch_scatter_tensors = []
        for _ in range(num_devices_scatter):
            torch_input_tensors = []
            for _ in range(num_cores):
                for _ in range(shard_width // 32):
                    if scheme == "random":
                        torch_input_tensors.append(torch.rand(1, 1, shard_height, 32))
                    elif scheme == "sequential":
                        torch_input_tensors.append(torch.ones(1, 1, shard_height, 32) * factor)
                        factor += 1
                    else:
                        raise ValueError(f"Invalid scheme: {scheme}")
            torch_scatter_tensors.append(torch.cat(torch_input_tensors, dim=dim))

        torch_fracture_tensors.append(torch.cat(torch_scatter_tensors, dim=1))

    return torch.cat(torch_fracture_tensors, dim=0)


def run_reduce_scatter_test(
    mesh_device,
    dim,
    shard_height,
    shard_width,
    num_devices_scatter,
    num_devices_fracture,
    num_cores,
    num_iters,
    warmup_iters,
    trace_mode,
    num_links=3,
    scheme="random",
    use_regular_grid=False,
    input_grid=None,
    output_grid=None,
    dtype=ttnn.bfloat8_b,
    profiler=BenchmarkProfiler(),
    topology=ttnn.Topology.Linear,
    use_noc1_only=False,
):
    num_pages_per_packet = 4
    cyclic_buffer_size = 8

    # input, output, interm core range set
    compute_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    subdevice_shard_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(compute_grid[0] - 1, compute_grid[1] - 1),
            ),
        }
    )
    if input_grid is not None:
        input_shard_cores_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(input_grid[0] - 1, input_grid[1] - 1),
                ),
            }
        )
    if output_grid is not None:
        output_shard_cores_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(output_grid[0] - 1, output_grid[1] - 1),
                ),
            }
        )
        tensor_width_in_tiles = num_cores * shard_width
        output_num_cores = output_grid[0] * output_grid[1]

    # input, output, interm memory config
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            input_shard_cores_grid if use_regular_grid else RING_CRS,
            [shard_height, shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    packet_workers_persistent_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            subdevice_shard_cores_grid if use_regular_grid else SUB_DEVICE_CRS,
            [shard_height, num_devices_scatter * num_pages_per_packet * 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            output_shard_cores_grid if use_regular_grid else FF1_CRS_RS_OUT,
            [
                shard_height,
                tensor_width_in_tiles // output_num_cores // num_devices_scatter if use_regular_grid else 32,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    output_tensor_goldens_list = []
    tt_input_tensors_list = []
    tt_intermediate_tensors_list = []
    for iter in range(num_iters):
        input = gen_tensor(
            dim, shard_height, shard_width, num_devices_scatter, num_devices_fracture, num_cores, scheme=scheme
        )

        intermediate_tensor = torch.zeros(
            [
                num_devices_fracture,
                num_devices_scatter,
                shard_height,
                num_devices_scatter
                * num_pages_per_packet
                * 32
                * packet_workers_persistent_mem_config.shard_spec.num_cores(),
            ]
        )

        intermediate_outputs = torch.chunk(input, chunks=num_devices_scatter, dim=1)
        output = torch.zeros(intermediate_outputs[0].shape)

        for i in range(0, len(intermediate_outputs)):
            output += intermediate_outputs[i]

        scattered_output = torch.chunk(output, chunks=num_devices_scatter, dim=dim)
        scattered_output = torch.cat(scattered_output, dim=1)

        output_tensor_goldens_list.append(scattered_output)

        tt_input = ttnn.from_torch(
            input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=sharded_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, dims=(0, 1), mesh_shape=[num_devices_fracture, num_devices_scatter]
            ),
        )
        if iter < cyclic_buffer_size:
            tt_intermediate = ttnn.from_torch(
                intermediate_tensor,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=packet_workers_persistent_mem_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, dims=(0, 1), mesh_shape=[num_devices_fracture, num_devices_scatter]
                ),
            )
            tt_intermediate_tensors_list.append(tt_intermediate)
        tt_input_tensors_list.append(tt_input)

    ccl_sub_device_crs = subdevice_shard_cores_grid if use_regular_grid is not None else SUB_DEVICE_CRS
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
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

    tt_out_tensor_list = []

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []
        for i in range(n_iters):
            buffer_index = 0 if trace_mode else i
            tt_out_tensor = ttnn.experimental.llama_reduce_scatter(
                tt_input_tensors_list[buffer_index],
                tt_intermediate_tensors_list[buffer_index % cyclic_buffer_size],
                dim,
                ccl_semaphore_handles[buffer_index],
                worker_sub_device_id,
                cluster_axis=1,
                mesh_device=mesh_device,
                num_links=num_links,
                memory_config=output_mem_config,
                topology=topology,
                use_noc1_only=use_noc1_only,
            )
            if not trace_mode:
                ttnn.synchronize_device(mesh_device)
            if store_all_results:
                tt_output_list.append(tt_out_tensor)
        if store_all_results:
            return tt_output_list
        else:
            return [tt_out_tensor]

    if trace_mode:
        # compile run:
        logger.info("Compiling model")
        tt_out_tensor_list = run_op(1, store_all_results=False)

        logger.info("Capturing Warmup")
        print("Warmup iteration: ", warmup_iters)
        if warmup_iters > 0:
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            run_op(warmup_iters, store_all_results=False)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)

        logger.info("Capturing Trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_outs = run_op(num_iters, store_all_results=False)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        logger.info("Starting Trace perf test...")
        profiler.start("reduce-scatter-trace-warmup")
        if warmup_iters > 0:
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)
        profiler.end("reduce-scatter-trace-warmup")

        signpost("start")
        profiler.start("reduce-scatter-trace")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)
        profiler.end("reduce-scatter-trace")
        signpost("stop")

        time_taken = profiler.get_duration("reduce-scatter-trace") - profiler.get_duration(
            "reduce-scatter-trace-warmup"
        )
        logger.info(f"Time taken e2e: {time_taken} s")
    else:
        signpost("start")
        tt_out_tensor_list = run_op(num_iters, store_all_results=True)
        signpost("stop")

    mesh_device.reset_sub_device_stall_group()

    passed = True
    first_failed_tensor_index = None
    failed_indices = []
    expected_pcc = 0.999 if dtype == ttnn.bfloat8_b else 0.9999
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_torch_tensor = ttnn.to_torch(
            tt_out_tensor_list[tensor_index],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, mesh_shape=[num_devices_fracture, num_devices_scatter], dims=(0, 1)
            ),
        )
        eq, output_results = comp_pcc(tt_torch_tensor, output_tensor_goldens_list[tensor_index], expected_pcc)
        logger.info(f"Output tensor {tensor_index} has result {output_results}")
        if not eq:
            passed = False
            first_failed_tensor_index = tensor_index
            failed_indices = torch.where(tt_torch_tensor != output_tensor_goldens_list[tensor_index])
            break

    logger.info(f"Device has {mesh_device.num_program_cache_entries()} program cache entries")
    assert (
        mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
    ), f"Device {i} has {mesh_device.num_program_cache_entries()} program cache entries"

    if not passed:
        logger.info(f"Failed indices: {failed_indices}")
        assert eq, f"{first_failed_tensor_index} FAILED: {output_results}"


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 300000,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize("use_noc1_only", [False])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_fabric_reduce_scatter_tg_trace(mesh_device, trace_mode, use_noc1_only):
    # Only run these tests on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Not TG!")

    dim = 3
    shard_height = 32
    shard_width = 160
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = 24
    num_iters = 75
    warmup_iters = 10
    trace_mode = trace_mode

    run_reduce_scatter_test(
        mesh_device,
        dim,
        shard_height,
        shard_width,
        num_devices_scatter,
        num_devices_fracture,
        num_cores,
        num_iters,
        warmup_iters,
        trace_mode,
        use_noc1_only=use_noc1_only,
        num_links=3,
        scheme="random",
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize("use_noc1_only", [False, True])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_fabric_reduce_scatter_tg_no_trace(mesh_device, trace_mode, use_noc1_only):
    # Only run these tests on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Not TG!")

    dim = 3
    shard_height = 32
    shard_width = 160
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = 24
    num_iters = 30
    warmup_iters = 0
    trace_mode = trace_mode

    run_reduce_scatter_test(
        mesh_device,
        dim,
        shard_height,
        shard_width,
        num_devices_scatter,
        num_devices_fracture,
        num_cores,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=3,
        scheme="random",
        topology=ttnn.Topology.Linear,
        use_noc1_only=use_noc1_only,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 100000,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True, False])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),  # TODO: Once fabric can be initialized on a SubMesh, revert to (1, 2)
    ],
    indirect=True,
)
@pytest.mark.parametrize("shard_height", [32])
@pytest.mark.parametrize("shard_width", [64])
@pytest.mark.parametrize("input_grid", [(5, 4)])
@pytest.mark.parametrize("output_grid", [(5, 2)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_fabric_reduce_scatter_regular_grid_2_dev(
    mesh_device, trace_mode, shard_height, shard_width, input_grid, output_grid, dtype
):
    # Only run these tests on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (8, 8):
        pytest.skip("Not TG!")

    dim = 3
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = input_grid[0] * input_grid[1]
    num_iters = 30
    warmup_iters = 0
    run_reduce_scatter_test(
        mesh_device,
        dim,
        shard_height,
        shard_width,
        num_devices_scatter,
        num_devices_fracture,
        num_cores,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=1,
        scheme="random",
        use_regular_grid=True,
        input_grid=input_grid,
        output_grid=output_grid,
        dtype=dtype,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 100000,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),  # TODO: Once fabric can be initialized on a SubMesh, revert to (1, 4)
    ],
    indirect=True,
)
@pytest.mark.parametrize("shard_height", [32])
@pytest.mark.parametrize("shard_width", [64])
@pytest.mark.parametrize("input_grid", [(5, 5)])
@pytest.mark.parametrize("output_grid", [(5, 1)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_fabric_reduce_scatter_regular_grid_4_dev(
    mesh_device, trace_mode, shard_height, shard_width, input_grid, output_grid, dtype
):
    # Only run these tests on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (8, 8):
        pytest.skip("Not TG!")

    dim = 3
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = input_grid[0] * input_grid[1] - 5  # test padding
    num_iters = 30
    warmup_iters = 0
    run_reduce_scatter_test(
        mesh_device,
        dim,
        shard_height,
        shard_width,
        num_devices_scatter,
        num_devices_fracture,
        num_cores,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=3,
        scheme="random",
        use_regular_grid=True,
        input_grid=input_grid,
        output_grid=output_grid,
        dtype=dtype,
    )
