# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.utility_functions import skip_for_grayskull
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tracy import signpost


def gen_tensor(dim, per_device_output_shape, mesh_axes, mesh_shape, cluster_axis, scheme="random"):
    factor = 0

    if cluster_axis is None:
        # Linearized grid case - treat all devices as a single linear sequence
        total_devices = mesh_shape[0] * mesh_shape[1]

        # Create output tensors for each device in the linearized grid
        single_device_output_tensors = []
        for device_idx in range(total_devices):
            if scheme == "random":
                single_device_output_tensors.append(torch.rand(per_device_output_shape))
            elif scheme == "sequential":
                single_device_output_tensors.append(torch.ones(per_device_output_shape) * factor)
                factor += 1
            else:
                raise ValueError(f"Invalid scheme: {scheme}")

        # Create input tensor by concatenating all slices along dim
        torch_input_tensor = torch.cat(single_device_output_tensors, dim=dim)

        # For output tensor, we need to arrange slices according to mesh layout
        # First, arrange them in rows and columns according to mesh_shape
        output_tensors_2d = []
        device_idx = 0
        for row in range(mesh_shape[0]):
            row_tensors = []
            for col in range(mesh_shape[1]):
                row_tensors.append(single_device_output_tensors[device_idx])
                device_idx += 1
            # Concatenate along mesh_axes[1] (column dimension)
            row_tensor = torch.cat(row_tensors, dim=mesh_axes[1])
            output_tensors_2d.append(row_tensor)

        # Concatenate rows along mesh_axes[0] (row dimension)
        torch_output_tensor = torch.cat(output_tensors_2d, dim=mesh_axes[0])

        # Create input tensor with same structure as output for mesh mapping
        input_tensors_2d = []
        for row in range(mesh_shape[0]):
            row_tensors = []
            for col in range(mesh_shape[1]):
                row_tensors.append(torch_input_tensor)
            row_tensor = torch.cat(row_tensors, dim=mesh_axes[1])
            input_tensors_2d.append(row_tensor)
        torch_input_tensor = torch.cat(input_tensors_2d, dim=mesh_axes[0])

        return torch_input_tensor, torch_output_tensor

    # Original logic for when cluster_axis is specified
    non_cluster_axes = [i for i in range(len(mesh_axes)) if i != cluster_axis]

    torch_input_tensor = None
    for axis in non_cluster_axes:
        per_axis_tensor = None
        temp_per_axis_tensors = []
        temp_per_axis_output_tensors = []
        for _ in range(mesh_shape[axis]):
            single_device_output_tensors = []
            for _ in range(mesh_shape[cluster_axis]):
                if scheme == "random":
                    single_device_output_tensors.append(torch.rand(per_device_output_shape))
                elif scheme == "sequential":
                    single_device_output_tensors.append(torch.ones(per_device_output_shape) * factor)
                    factor += 1
                else:
                    raise ValueError(f"Invalid scheme: {scheme}")
            single_device_input_tensor = torch.cat(single_device_output_tensors, dim=dim)
            cluster_axis_tensors = []
            for _ in range(mesh_shape[cluster_axis]):
                cluster_axis_tensors.append(single_device_input_tensor)

            cluster_axis_tensor = torch.cat(cluster_axis_tensors, dim=mesh_axes[cluster_axis])
            cluster_axis_output_tensor = torch.cat(single_device_output_tensors, dim=mesh_axes[cluster_axis])

            temp_per_axis_tensors.append(cluster_axis_tensor)
            temp_per_axis_output_tensors.append(cluster_axis_output_tensor)
        per_axis_tensor = torch.cat(temp_per_axis_tensors, dim=axis)
        per_axis_output_tensor = torch.cat(temp_per_axis_output_tensors, dim=axis)

        if torch_input_tensor is None:
            torch_input_tensor = per_axis_tensor
            torch_output_tensor = per_axis_output_tensor
        else:
            torch_input_tensor = torch.cat([torch_input_tensor, per_axis_tensor], dim=axis)
            torch_output_tensor = torch.cat([torch_output_tensor, per_axis_output_tensor], dim=axis)

    return torch_input_tensor, torch_output_tensor


def run_mesh_partition_test(
    mesh_device,
    per_device_output_shape,
    dim,
    num_iters,
    warmup_iters,
    trace_mode,
    dtype,
    layout,
    cluster_axis,
    mesh_axes,
    mesh_shape,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    scheme="random",
    profiler=BenchmarkProfiler(),
):
    mesh_device.enable_program_cache()
    torch.manual_seed(2005)

    output_tensor_goldens_list = []
    tt_input_tensors_list = []
    for it in range(num_iters):
        input, output = gen_tensor(dim, per_device_output_shape, mesh_axes, mesh_shape, cluster_axis, scheme=scheme)

        output_tensor_goldens_list.append(output)

        tt_input = ttnn.from_torch(
            input,
            device=mesh_device,
            layout=layout,
            dtype=dtype,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=mesh_axes, mesh_shape=mesh_shape),
        )
        if it == 0:
            logger.info(f"input shape {input.shape}")
            logger.info(f"tt_input per-device shape {tt_input.shape}")
        tt_input_tensors_list.append(tt_input)

    tt_out_tensor_list = []

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []
        for i in range(n_iters):
            buffer_index = 0 if trace_mode else i
            tt_out_tensor = ttnn.mesh_partition(
                tt_input_tensors_list[buffer_index],
                dim,
                cluster_axis=cluster_axis,
                memory_config=output_memory_config,
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
        logger.info("Warmup iteration: ", warmup_iters)
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

    passed = True
    first_failed_tensor_index = None
    failed_indices = []
    expected_pcc = 0.999 if dtype == ttnn.bfloat8_b else 1.0
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_torch_tensor = ttnn.to_torch(
            tt_out_tensor_list[tensor_index],
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=mesh_axes),
        )
        if tensor_index == 0:
            logger.info(f"tt_output per-device shape {tt_out_tensor_list[tensor_index].shape}")
            logger.info(f"golden shape {output_tensor_goldens_list[tensor_index].shape}")
            logger.info(f"tt_torch_tensor shape {tt_torch_tensor.shape}")
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
    ), f"Device {mesh_device.id} has {mesh_device.num_program_cache_entries()} program cache entries"

    if not passed:
        logger.info(f"Failed indices: {failed_indices}")
        assert eq, f"{first_failed_tensor_index} FAILED: {output_results}"


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 10000,
            "dispatch_core_type": ttnn.DispatchCoreType.WORKER,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("per_device_output_shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("cluster_axis", [0, 1, None])
@pytest.mark.parametrize("mesh_axes", [[0, 1]])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_mesh_partition(
    mesh_device,
    mesh_shape,
    trace_mode,
    per_device_output_shape,
    dtype,
    layout,
    dim,
    cluster_axis,
    mesh_axes,
    input_memory_config,
    output_memory_config,
):
    num_iters = 2
    warmup_iters = 0

    run_mesh_partition_test(
        mesh_device,
        per_device_output_shape,
        dim,
        num_iters,
        warmup_iters,
        trace_mode,
        dtype,
        layout,
        cluster_axis,
        mesh_axes,
        mesh_shape,
        input_memory_config,
        output_memory_config,
        scheme="random",
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 16384,
            "dispatch_core_type": ttnn.DispatchCoreType.WORKER,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True, False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("per_device_output_shape, dim", [((16, 1, 1, 7168), 0), ((1, 1, 8, 7168), 2)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("cluster_axis", [0, 1, None])
@pytest.mark.parametrize("mesh_axes", [[0, 1]])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG])
def test_mesh_partition_rm(
    mesh_device,
    mesh_shape,
    trace_mode,
    per_device_output_shape,
    dtype,
    layout,
    dim,
    cluster_axis,
    mesh_axes,
    input_memory_config,
    output_memory_config,
):
    num_iters = 2
    warmup_iters = 0

    run_mesh_partition_test(
        mesh_device,
        per_device_output_shape,
        dim,
        num_iters,
        warmup_iters,
        trace_mode,
        dtype,
        layout,
        cluster_axis,
        mesh_axes,
        mesh_shape,
        input_memory_config,
        output_memory_config,
        scheme="random",
    )
