# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
import os

is_RING_6U = os.environ.get("RING_6U", "0") == "1"
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
from models.demos.llama3_subdevices.tt.model_config import set_tg_attention_config
from tracy import signpost

PACKET_WORKER_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 2)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(2, 3)),
    ]
)

LINEAR_TOPOLOGY = True
if LINEAR_TOPOLOGY:
    TOPOLOGY = ttnn.Topology.Linear
    WRAP_MESH = False
else:
    TOPOLOGY = ttnn.Topology.Ring
    WRAP_MESH = True


def gen_tensor(dim, height, width, num_devices_scatter, num_devices_fracture, num_cores, scheme="random"):
    factor = 1
    shard_height = height // num_devices_scatter
    torch_fracture_tensors = []
    for _ in range(num_devices_fracture):
        torch_scatter_tensors = []
        for _ in range(num_devices_scatter):
            torch_input_tensors = []
            factor = 1
            for _ in range(num_devices_scatter):
                if scheme == "random":
                    torch_input_tensors.append(torch.rand(1, 1, shard_height, width))
                elif scheme == "sequential":
                    torch_input_tensors.append(torch.ones(1, 1, shard_height, width))
                    factor += 1
                else:
                    raise ValueError(f"Invalid scheme: {scheme}")
            torch_scatter_tensors.append(torch.cat(torch_input_tensors, dim=dim - 1))

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
):
    num_pages_per_packet = 4
    cyclic_buffer_size = 8

    model_config = {}
    model_config = set_tg_attention_config(model_config, 4096)
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

    head_dim = num_cores * shard_width // (8 + 2 * 1)  # 128
    M = shard_height
    qkv_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            model_config["CREATE_HEAD_OUTPUT_MEMCFG"].shard_spec.grid,
            [M, head_dim],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    output_tensor_q_goldens_list = []
    output_tensor_k_goldens_list = []
    output_tensor_v_goldens_list = []
    tt_input_tensors_list = []
    tt_intermediate_tensors_list = []
    for iter in range(num_iters):
        input = gen_tensor(
            dim,
            shard_height,
            num_cores * shard_width,
            num_devices_scatter,
            num_devices_fracture,
            num_cores,
            scheme=scheme,
        )
        reduced_input_tensor = input.sum(dim=1)
        reduced_input_tensor_reshaped = reduced_input_tensor.reshape(8, 32, 10, 128)
        q_output_tensor_golden = reduced_input_tensor_reshaped[:, :, :8, :]
        k_output_tensor_golden = reduced_input_tensor_reshaped[:, :, 8:9, :]
        v_output_tensor_golden = reduced_input_tensor_reshaped[:, :, 9:10, :]
        output_tensor_q_goldens_list.append(q_output_tensor_golden)
        output_tensor_k_goldens_list.append(k_output_tensor_golden)
        output_tensor_v_goldens_list.append(v_output_tensor_golden)

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

        # intermediate_outputs = torch.chunk(input, chunks=num_devices_scatter, dim=1)
        # output = torch.zeros(intermediate_outputs[0].shape)

        # for i in range(0, len(intermediate_outputs)):
        #     output += intermediate_outputs[i]

        # scattered_output = torch.chunk(output, chunks=num_devices_scatter, dim=dim)
        # scattered_output = torch.cat(scattered_output, dim=1)

        tt_input = ttnn.from_torch(
            input,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
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
                layout=ttnn.ROW_MAJOR_LAYOUT,
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

    tt_out_tensor_q_list = []
    tt_out_tensor_k_list = []
    tt_out_tensor_v_list = []

    def run_op(n_iters, store_all_results=True):
        for i in range(n_iters):
            buffer_index = 0 if trace_mode else i
            tt_out_tensor_q, tt_out_tensor_k, tt_out_tensor_v = ttnn.experimental.llama_rs_create_heads(
                tt_input_tensors_list[buffer_index],
                tt_intermediate_tensors_list[buffer_index % cyclic_buffer_size],
                dim,
                ccl_semaphore_handles[buffer_index],
                worker_sub_device_id,
                cluster_axis=1,
                mesh_device=mesh_device,
                topology=TOPOLOGY,
                num_links=num_links,
                num_heads=8,
                num_kv_heads=1,
                memory_config=output_mem_config,
                qkv_memory_config=qkv_mem_config,
            )
            if not trace_mode:
                ttnn.synchronize_device(mesh_device)
            if store_all_results:
                tt_out_tensor_q_list.append(tt_out_tensor_q)
                tt_out_tensor_k_list.append(tt_out_tensor_k)
                tt_out_tensor_v_list.append(tt_out_tensor_v)
        if store_all_results:
            return tt_out_tensor_q_list, tt_out_tensor_k_list, tt_out_tensor_v_list
        else:
            return [tt_out_tensor_q], [tt_out_tensor_k], [tt_out_tensor_v]

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
        tt_out_tensor_q_list, tt_out_tensor_k_list, tt_out_tensor_v_list = run_op(num_iters, store_all_results=False)
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

    for tensor_index in range(len(tt_out_tensor_list[0])):
        tt_torch_tensor_q = ttnn.to_torch(
            tt_out_tensor_list[0][tensor_index],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, mesh_shape=[num_devices_fracture, num_devices_scatter], dims=(0, 1)
            ),
        )
        tt_torch_tensor_k = ttnn.to_torch(
            tt_out_tensor_list[1][tensor_index],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, mesh_shape=[num_devices_fracture, num_devices_scatter], dims=(0, 1)
            ),
        )
        tt_torch_tensor_v = ttnn.to_torch(
            tt_out_tensor_list[2][tensor_index],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, mesh_shape=[num_devices_fracture, num_devices_scatter], dims=(0, 1)
            ),
        )
        eq, output_results = comp_pcc(tt_torch_tensor_q, output_tensor_q_goldens_list[tensor_index], expected_pcc)
        logger.info(f"Output q tensor {tensor_index} has result {output_results}")
        if not eq:
            passed = False
            first_failed_tensor_index = tensor_index
            failed_indices = torch.where(tt_torch_tensor_q != output_tensor_q_goldens_list[tensor_index])
            break

        eq, output_results = comp_pcc(
            tt_torch_tensor_k[:, :, 0, :].unsqueeze(2), output_tensor_k_goldens_list[tensor_index], expected_pcc
        )
        logger.info(f"Output k tensor {tensor_index} has result {output_results}")
        if not eq:
            passed = False
            first_failed_tensor_index = tensor_index
            failed_indices = torch.where(tt_torch_tensor_k != output_tensor_k_goldens_list[tensor_index])
            break

        eq, output_results = comp_pcc(
            tt_torch_tensor_v[:, :, 0, :].unsqueeze(2), output_tensor_v_goldens_list[tensor_index], expected_pcc
        )
        logger.info(f"Output v tensor {tensor_index} has result {output_results}")
        if not eq:
            passed = False
            first_failed_tensor_index = tensor_index
            failed_indices = torch.where(tt_torch_tensor_v != output_tensor_v_goldens_list[tensor_index])
            break

    logger.info(f"Device has {mesh_device.num_program_cache_entries()} program cache entries")
    assert (
        mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
    ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"

    if not passed:
        logger.info(f"Failed indices: {failed_indices}")
        assert eq, f"{first_failed_tensor_index} FAILED: {output_results}"


@pytest.mark.skipif(not is_RING_6U, reason="This test is only for 6U devices")
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 269312,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_rs_create_heads_6u_trace(mesh_device, trace_mode, dtype, use_program_cache):
    # Only run these tests on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Not TG!")

    dim = 3
    shard_height = 32
    shard_width = 64
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = 20
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
        num_links=4,
        scheme="random",
        dtype=dtype,
    )


@pytest.mark.skipif(is_RING_6U, reason="This test is only for TG devices")
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 241664,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_rs_create_heads_tg_trace(mesh_device, trace_mode, dtype):
    # Only run these tests on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Not TG!")

    dim = 3
    shard_height = 32
    shard_width = 64
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = 20
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
        num_links=3,
        scheme="random",
        dtype=dtype,
    )


@pytest.mark.skipif(is_RING_6U, reason="This test is only for TG devices")
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_rs_create_heads_tg_no_trace(mesh_device, trace_mode, dtype):
    # Only run these tests on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Not TG!")

    dim = 3
    shard_height = 32
    shard_width = 64
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = 20
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
        dtype=dtype,
    )
