# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Forward op unit test.

Tests the per-device forward operation without sockets — data is pre-loaded
in the input tensor (like broadcast's non-socket path). This allows direct
perf comparison between forward and broadcast ops.

Test setup mirrors MoE dimensions:
  - M=1, K=7168 (bfloat16)
  - Tile layout [1, 32], height-sharded on a single core per device
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.micro_ops.forward.op import DeepseekForward
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import create_fabric_router_config, run_trace_benchmark

MAX_PAYLOAD_SIZE = 15232
FORWARD_CORE = ttnn.CoreCoord(10, 8)


def _build_forward_test_tensors(
    mesh_device,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    forward_core,
    tile=None,
):
    """
    Build input and output tensor meshes for forward test.

    The input tensor is replicated to all devices (every device has the same data).
    The output tensor is zeroed on all devices.
    """
    if tile is None:
        tile = ttnn.Tile((1, 32))

    input_tensor_torch = torch.rand(output_shape, dtype=torch.bfloat16)

    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(forward_core, forward_core)])
    shard_spec = ttnn.ShardSpec(shard_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec)

    input_tensor_mesh = ttnn.from_torch(
        input_tensor_torch,
        dtype=input_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    output_tensor_mesh = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=input_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    return input_tensor_torch, input_tensor_mesh, output_tensor_mesh


@pytest.mark.parametrize(
    "output_shape, input_shard_shape, tensor_mem_layout",
    [
        ([1, 7168], (1, 7168), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("num_iters, num_warmup_iter", [(30, 15)])
@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2)])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(MAX_PAYLOAD_SIZE),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_forward(
    bh_2d_mesh_device,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    num_iters,
    num_warmup_iter,
    mesh_rows,
    mesh_cols,
):
    """
    Trace-benchmarked forward op test (no sockets).

    Data is pre-loaded in the input tensor. The forward kernel reads from the
    CB (backed by the input tensor) and writes to the output tensor — same
    work as the socket path minus the socket read.
    """
    if is_slow_dispatch():
        pytest.skip("Skipping trace mode in slow dispatch")

    num_devices = int(mesh_rows) * int(mesh_cols)

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((int(mesh_rows), int(mesh_cols))))

    input_tensor_torch, input_tensor_mesh, output_tensor = _build_forward_test_tensors(
        mesh_device=submesh,
        output_shape=output_shape,
        input_shard_shape=input_shard_shape,
        tensor_mem_layout=tensor_mem_layout,
        layout=layout,
        input_dtype=input_dtype,
        forward_core=FORWARD_CORE,
    )

    torch_expected = DeepseekForward.golden(input_tensor_torch)

    logger.info(
        "Running forward op: mesh={}x{} shape={} payload={}B",
        mesh_rows,
        mesh_cols,
        output_shape,
        output_shape[0] * output_shape[1] * 2,
    )

    def run_forward():
        return DeepseekForward.op(
            output_tensor,
            input_tensor_mesh=input_tensor_mesh,
            forward_core=FORWARD_CORE,
        )

    ttnn_result = run_trace_benchmark(
        submesh,
        run_forward,
        num_warmup_iter=num_warmup_iter,
        num_iter=num_iters,
        profiler_name="deepseek-forward",
    )

    logger.info("Verifying forward results...")
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
            max_diff = (received - torch_expected).abs().max().item()
            logger.error(f"Device {device_idx}: FAILED (max_diff={max_diff:.6f})")
            all_passed = False
        else:
            logger.info(f"Device {device_idx}: PASSED")

    assert all_passed, "Forward test: not all devices received correct data"
    logger.info("Forward trace benchmark test PASSED!")


@pytest.mark.parametrize(
    "output_shape, input_shard_shape, tensor_mem_layout",
    [
        ([1, 7168], (1, 7168), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2)])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_forward_loop(
    bh_2d_mesh_device,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    num_iters,
    mesh_rows,
    mesh_cols,
):
    """
    Forward op called with num_iterations>1 (kernel-internal loop, no trace).

    Validates that the kernel loop works correctly when data is pre-loaded.
    Comparable to test_ccl_broadcast_loop.
    """
    num_devices = int(mesh_rows) * int(mesh_cols)

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((int(mesh_rows), int(mesh_cols))))

    input_tensor_torch, input_tensor_mesh, output_tensor = _build_forward_test_tensors(
        mesh_device=submesh,
        output_shape=output_shape,
        input_shard_shape=input_shard_shape,
        tensor_mem_layout=tensor_mem_layout,
        layout=layout,
        input_dtype=input_dtype,
        forward_core=FORWARD_CORE,
    )

    torch_expected = DeepseekForward.golden(input_tensor_torch)

    logger.info(f"Running forward loop: {num_iters} internal iterations")
    ttnn_result = DeepseekForward.op(
        output_tensor,
        input_tensor_mesh=input_tensor_mesh,
        forward_core=FORWARD_CORE,
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

    logger.info(f"Forward loop test PASSED! ({num_iters} iterations)")


@pytest.mark.parametrize(
    "output_shape, input_shard_shape, tensor_mem_layout",
    [
        ([1, 7168], (1, 7168), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("num_iters, num_warmup_iter", [(30, 15)])
@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2)])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(MAX_PAYLOAD_SIZE),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_forward_cross_column(
    bh_2d_mesh_device,
    output_shape,
    input_shard_shape,
    tensor_mem_layout,
    layout,
    input_dtype,
    num_iters,
    num_warmup_iter,
    mesh_rows,
    mesh_cols,
):
    """
    Trace-benchmarked cross-column forward test (no sockets).

    Entry-column devices (col 0) read from their input tensor and forward
    data to partner devices in col 1 via fabric. Verifies both columns
    end up with identical data.
    """
    if is_slow_dispatch():
        pytest.skip("Skipping trace mode in slow dispatch")

    if int(mesh_cols) < 2:
        pytest.skip("Cross-column test requires at least 2 columns")
    num_devices = int(mesh_rows) * int(mesh_cols)

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((int(mesh_rows), int(mesh_cols))))

    input_tensor_torch, input_tensor_mesh, output_tensor = _build_forward_test_tensors(
        mesh_device=submesh,
        output_shape=output_shape,
        input_shard_shape=input_shard_shape,
        tensor_mem_layout=tensor_mem_layout,
        layout=layout,
        input_dtype=input_dtype,
        forward_core=FORWARD_CORE,
    )

    compute_grid = submesh.compute_with_storage_grid_size()
    num_cores = compute_grid.x * compute_grid.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
    cross_col_sem = ttnn.create_global_semaphore(submesh, available_cores, 0)

    torch_expected = DeepseekForward.golden(input_tensor_torch)

    logger.info(f"Running cross-column forward: mesh={mesh_rows}x{mesh_cols}")

    def run_forward_cross_column():
        return DeepseekForward.op(
            output_tensor,
            input_tensor_mesh=input_tensor_mesh,
            forward_core=FORWARD_CORE,
            cross_column_semaphore=cross_col_sem,
            enable_cross_column=True,
            entry_column=0,
        )

    ttnn_result = run_trace_benchmark(
        submesh,
        run_forward_cross_column,
        num_warmup_iter=num_warmup_iter,
        num_iter=num_iters,
        profiler_name="deepseek-forward-cross-column",
    )

    logger.info("Verifying cross-column results...")
    output_tensor_torch = ttnn.to_torch(ttnn_result, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    slice_size = output_shape[0]
    all_passed = True
    for device_idx in range(num_devices):
        start = device_idx * slice_size
        end = start + slice_size
        received = output_tensor_torch[start:end, :]
        if torch.allclose(received, torch_expected, rtol=1e-3, atol=1e-3):
            logger.info(f"Device {device_idx}: PASSED")
        else:
            max_diff = (received - torch_expected).abs().max().item()
            logger.error(f"Device {device_idx}: FAILED (max_diff={max_diff:.6f})")
            all_passed = False

    assert all_passed, "Cross-column forward: not all devices received correct data"
    logger.info("Cross-column forward trace benchmark test PASSED!")
