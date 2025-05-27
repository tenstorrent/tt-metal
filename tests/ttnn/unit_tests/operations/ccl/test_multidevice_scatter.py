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
from tracy import signpost


def gen_tensor(dim, per_device_output_shape, mesh_axes, mesh_shape, cluster_axis, scheme="random"):
    torch.manual_seed(2005)
    factor = 0
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

    logger.info(f"torch_input_tensor shape: {torch_input_tensor.shape}")
    logger.info(f"torch_output_tensor shape: {torch_output_tensor.shape}")
    return torch_input_tensor, torch_output_tensor


def run_multidevice_scatter_test(
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

    output_tensor_goldens_list = []
    tt_input_tensors_list = []
    for _ in range(num_iters):
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
        tt_input_tensors_list.append(tt_input)

    tt_out_tensor_list = []

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []
        for i in range(n_iters):
            buffer_index = 0 if trace_mode else i
            tt_out_tensor = ttnn.multidevice_scatter(
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

    mesh_device.reset_sub_device_stall_group()

    passed = True
    first_failed_tensor_index = None
    failed_indices = []
    expected_pcc = 0.999 if dtype == ttnn.bfloat8_b else 1.0
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_torch_tensor = ttnn.to_torch(
            tt_out_tensor_list[tensor_index],
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=mesh_axes),
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
    ), f"Device {mesh_device.id} has {mesh_device.num_program_cache_entries()} program cache entries"

    if not passed:
        logger.info(f"Failed indices: {failed_indices}")
        assert eq, f"{first_failed_tensor_index} FAILED: {output_results}"


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
    "mesh_shape, mesh_device", [pytest.param((1, 2), (1, 2), id="1x2_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("per_device_output_shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("mesh_axes", [[0, 1]])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_multidevice_scatter(
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

    run_multidevice_scatter_test(
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
        scheme="sequential",
    )
