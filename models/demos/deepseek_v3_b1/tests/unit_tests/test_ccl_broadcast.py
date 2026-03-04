# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
CCL TP/SP Broadcast Test

Tests the deepseek_minimal_broadcast operation implemented using the generic op infrastructure.
This test validates neighbor-exchange broadcast correctness on a 2D mesh.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.micro_ops.ccl_broadcast.op import DeepseekMinimalBroadcast
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import (
    build_broadcast_test_inputs,
    create_fabric_router_config,
)
from models.perf.benchmarking_utils import BenchmarkProfiler


def _build_chunk_stamped_sender_tensor(output_shape, chunk_size_bytes, iteration_idx):
    """
    Build a sender tensor where each chunk has a distinct value.
    This catches stale-iteration forwarding when host launches multiple iterations.
    """
    total_elems = output_shape[0] * output_shape[1]
    elems_per_chunk = chunk_size_bytes // 2  # bf16 = 2 bytes
    sender = torch.zeros(output_shape, dtype=torch.bfloat16)

    offset = 0
    chunk_idx = 0
    # Keep values small enough to stay well-behaved in bf16.
    base = (iteration_idx % 8) * 16
    while offset < total_elems:
        chunk_elems = min(elems_per_chunk, total_elems - offset)
        sender.view(-1)[offset : offset + chunk_elems] = float(base + chunk_idx)
        offset += chunk_elems
        chunk_idx += 1

    return sender


@pytest.mark.parametrize(
    "mesh_rows, mesh_cols, sender_row, sender_col, output_shape, input_shard_shape, tensor_mem_layout",
    [
        (
            4,
            2,
            1,
            0,
            [1, 7168],
            (1, 7168),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("num_iters, num_warmup_iter", [(30, 15)])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_ccl_broadcast(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    num_iters,
    num_warmup_iter,
):
    if is_slow_dispatch():
        pytest.skip("Skipping trace mode in slow dispatch")

    num_devices = mesh_rows * mesh_cols

    # Validate mesh size
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    # Create submesh
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    bcast_core = ttnn.CoreCoord(0, 0)
    test_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_row=sender_row,
        sender_col=sender_col,
        output_shape=output_shape,
        input_shard_shape=input_shard_shape,
        tensor_mem_layout=tensor_mem_layout,
        layout=layout,
        input_dtype=input_dtype,
        bcast_core=bcast_core,
        num_links=1,
    )
    sender_tensor = test_inputs.input_tensor_torch
    input_tensor_mesh = test_inputs.input_tensor_mesh
    output_tensor = test_inputs.output_tensor_mesh
    semaphores = test_inputs.semaphores

    # Compute expected output using golden function
    torch_expected = DeepseekMinimalBroadcast.golden(sender_tensor)

    # Run broadcast operation
    logger.info(f"Running CCL broadcast: sender=({sender_row},{sender_col}), mesh={mesh_rows}x{mesh_cols}")
    sender_coord = ttnn.MeshCoordinate(sender_row, sender_col)
    bcast_config = DeepseekMinimalBroadcast.configure(
        mesh_device=submesh,
        input_tensor_mesh=input_tensor_mesh,
        output_tensor=output_tensor,
        sender_coord=sender_coord,
        semaphores=semaphores,
    )
    assert (
        bcast_config.chunk_size_bytes,
        bcast_config.last_chunk_size_bytes,
        bcast_config.num_chunks,
    ) == (14336, 14336, 1), "Unexpected broadcast chunk tuple for neighbor-exchange test configuration"

    profiler = BenchmarkProfiler()

    # Compile Run
    logger.info("Compiling model")
    ttnn_result = DeepseekMinimalBroadcast.op(
        input_tensor_mesh,
        output_tensor,
        sender_coord,
        semaphores=semaphores,
    )
    ttnn.synchronize_device(submesh)

    # Capture warmup trace
    logger.info("Capturing warmup trace")
    trace_id_warmup = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_warmup_iter):
        ttnn_result = DeepseekMinimalBroadcast.op(
            input_tensor_mesh,
            output_tensor,
            sender_coord,
            semaphores=semaphores,
        )
    ttnn.end_trace_capture(submesh, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh)

    # Capture main trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_iters):
        ttnn_result = DeepseekMinimalBroadcast.op(
            input_tensor_mesh,
            output_tensor,
            sender_coord,
            semaphores=semaphores,
        )
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)

    # Execute warmup trace
    logger.info("Executing warmup trace...")
    profiler.start("deepseek-broadcast-warmup")
    ttnn.execute_trace(submesh, trace_id_warmup, blocking=False)
    ttnn.release_trace(submesh, trace_id_warmup)
    ttnn.synchronize_device(submesh)
    profiler.end("deepseek-broadcast-warmup")

    # Execute main trace with signposts for profiling
    logger.info("Starting Trace perf test...")
    signpost("start")
    profiler.start("deepseek-broadcast-trace")

    ttnn.execute_trace(submesh, trace_id, blocking=False)
    ttnn.release_trace(submesh, trace_id)
    ttnn.synchronize_device(submesh)

    profiler.end("deepseek-broadcast-trace")
    signpost("stop")

    # Verify output - all devices should have the sender's data
    logger.info("Verifying broadcast results...")
    output_tensor_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )

    slice_size = output_shape[0]
    all_passed = True
    for device_idx in range(num_devices):
        start = device_idx * slice_size
        end = start + slice_size
        received = output_tensor_torch[start:end, :]

        assert received.shape == torch_expected.shape, f"Shape mismatch at device {device_idx}"

        if not torch.allclose(received, torch_expected, rtol=1e-3, atol=1e-3):
            logger.error(f"Output mismatch for device {device_idx}")
            all_passed = False
        else:
            logger.info(f"Device {device_idx}: PASSED")

    assert all_passed, "Not all devices received the correct broadcast data"
    logger.info("CCL broadcast neighbor-exchange test passed!")


@pytest.mark.parametrize(
    "mesh_rows, mesh_cols, sender_row, sender_col, output_shape, input_shard_shape, tensor_mem_layout",
    [
        (4, 2, 0, 0, [1, 7168], (1, 7168), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_ccl_broadcast_loop(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    num_iters,
    num_links,
):
    """
    Test CCL broadcast called multiple times without trace.
    Validates PacketHeaderPool::reset(), semaphore_dec, noc_semaphore_wait_min.
    """
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    bcast_core = ttnn.CoreCoord(0, 0)
    test_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_row=sender_row,
        sender_col=sender_col,
        output_shape=output_shape,
        input_shard_shape=input_shard_shape,
        tensor_mem_layout=tensor_mem_layout,
        layout=layout,
        input_dtype=input_dtype,
        bcast_core=bcast_core,
        num_links=num_links,
    )
    sender_tensor = test_inputs.input_tensor_torch
    input_tensor_mesh = test_inputs.input_tensor_mesh
    output_tensor = test_inputs.output_tensor_mesh
    semaphores = test_inputs.semaphores

    sender_coord = ttnn.MeshCoordinate(sender_row, sender_col)
    torch_expected = DeepseekMinimalBroadcast.golden(sender_tensor)

    logger.info(f"Running CCL broadcast loop: {num_iters} internal iterations, sender=({sender_row},{sender_col})")
    ttnn_result = DeepseekMinimalBroadcast.op(
        input_tensor_mesh,
        output_tensor,
        sender_coord,
        semaphores=semaphores,
        num_links=num_links,
        num_iterations=num_iters,
    )
    ttnn.synchronize_device(submesh)

    output_tensor_torch = ttnn.to_torch(ttnn_result, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    slice_size = output_shape[0]
    for device_idx in range(num_devices):
        start = device_idx * slice_size
        end = start + slice_size
        received = output_tensor_torch[start:end, :]
        assert torch.allclose(
            received, torch_expected, rtol=1e-3, atol=1e-3
        ), f"Device {device_idx} data mismatch after {num_iters} iterations"
        logger.info(f"Device {device_idx}: PASSED")

    logger.info(f"CCL broadcast loop test PASSED! ({num_iters} iterations)")


@pytest.mark.parametrize(
    "mesh_rows, mesh_cols, sender_row, sender_col, output_shape, input_shard_shape, tensor_mem_layout",
    [
        (4, 2, 1, 0, [1, 7168], (1, 7168), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("num_host_iters", [8])
@pytest.mark.parametrize("chunk_size_bytes", [1024])
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": create_fabric_router_config(15232)}],
    indirect=True,
)
def test_ccl_broadcast_host_iter_stamped_chunks(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    num_host_iters,
    chunk_size_bytes,
    num_links,
):
    """
    Host-driven iteration correctness test.
    Launches num_host_iters times with num_iterations=1 and iteration-stamped chunk payloads.
    Detects stale-iteration forwarding that output-value-only random tests can miss.
    """
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    sender_coord = ttnn.MeshCoordinate(sender_row, sender_col)
    slice_size = output_shape[0]
    bcast_core = ttnn.CoreCoord(0, 0)

    for host_iter in range(num_host_iters):
        sender_tensor = _build_chunk_stamped_sender_tensor(output_shape, chunk_size_bytes, host_iter)
        iter_inputs = build_broadcast_test_inputs(
            mesh_device=submesh,
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
            sender_row=sender_row,
            sender_col=sender_col,
            output_shape=output_shape,
            input_shard_shape=input_shard_shape,
            tensor_mem_layout=tensor_mem_layout,
            layout=layout,
            input_dtype=input_dtype,
            bcast_core=bcast_core,
            num_links=num_links,
            input_tensor_torch=sender_tensor,
        )
        input_tensor_mesh = iter_inputs.input_tensor_mesh
        output_tensor = iter_inputs.output_tensor_mesh
        semaphores = iter_inputs.semaphores

        ttnn_result = DeepseekMinimalBroadcast.op(
            input_tensor_mesh,
            output_tensor,
            sender_coord,
            semaphores=semaphores,
            chunk_size_bytes=chunk_size_bytes,
            num_links=num_links,
            num_iterations=1,
        )
        ttnn.synchronize_device(submesh)

        output_tensor_torch = ttnn.to_torch(ttnn_result, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
        for device_idx in range(num_devices):
            start = device_idx * slice_size
            end = start + slice_size
            received = output_tensor_torch[start:end, :]
            assert torch.allclose(
                received, sender_tensor, rtol=1e-3, atol=1e-3
            ), f"Host-iter {host_iter}: device {device_idx} received stale/incorrect chunked broadcast data"
