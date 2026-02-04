# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.nightly.t3000.ccl.test_mesh_partition import gen_tensor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "op_name, input_shape, partition_dim, cluster_axis, memory_config, output_shape",
    [
        (
            "kvpe_mesh_partition",
            [1, 32, 64, 576],
            1,
            1,
            ttnn.L1_MEMORY_CONFIG,
            [1, 4, 64, 576],
        ),
    ],
    ids=[
        "kvpe_mesh_partition",
    ],
)
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((8, 4), id="8x4_grid")],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 595968,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_mesh_partition_trace_mode(
    mesh_device,
    batch_size,
    op_name,
    input_shape,
    partition_dim,
    cluster_axis,
    memory_config,
    output_shape,
    warmup_iters,
    num_iters,
):
    """
    Test the permute operations from mla1d.py with trace mode.

    These operations transpose tensor dimensions:
    1. kvpe_mesh_partition (line 1179): [1, 32, 64, 576] → [1, 4, 64, 576], dim=1, cluster_axis=1

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - L1 interleaved memory layout
    """
    torch.manual_seed(0)

    # Create random tensor for input
    torch_input_tensor, torch_output_tensor = gen_tensor(
        dim=partition_dim,
        per_device_output_shape=output_shape,
        mesh_axes=[0, 1],
        mesh_shape=mesh_device.shape,
        cluster_axis=cluster_axis,
    )

    # Create ttnn tensor with L1 interleaved memory config
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[0, 1], mesh_shape=mesh_device.shape),
    )

    # Compile run
    logger.info(f"Compiling permute operation: {op_name}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Partition dim: {partition_dim}")
    logger.info(f"  Cluster axis: {cluster_axis}")
    logger.info(f"  Memory config: {memory_config}")
    logger.info(f"  Output shape: {output_shape}")

    tt_output_tensor = ttnn.mesh_partition(tt_input_tensor, partition_dim, cluster_axis, memory_config=memory_config)
    ttnn.synchronize_device(mesh_device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.mesh_partition(
            tt_input_tensor, partition_dim, cluster_axis, memory_config=memory_config
        )
    ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.mesh_partition(
            tt_input_tensor, partition_dim, cluster_axis, memory_config=memory_config
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler = BenchmarkProfiler()
    profiler.start("warmup")
    ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(mesh_device, trace_id_warmup)
    profiler.end("warmup")
    ttnn.synchronize_device(mesh_device)

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("main")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    profiler.end("main")
    signpost("stop")
    ttnn.synchronize_device(mesh_device)

    # Verify the input and output shapes
    assert tt_input_tensor.shape == input_shape, f"Input shape mismatch: {tt_input_tensor.shape} != {input_shape}"
    assert tt_output_tensor.shape == output_shape, f"Output shape mismatch: {tt_output_tensor.shape} != {output_shape}"

    # Verify correctness
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    # Use mesh composer to concatenate distributed shards back to full tensor
    torch_output_from_tt = ttnn.to_torch(
        tt_output_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=[0, 1]),
    )

    # Compare the full mesh tensor from TTNN with the expected full mesh tensor
    assert (
        torch_output_from_tt.shape == torch_output_tensor.shape
    ), f"Shape mismatch: {torch_output_from_tt.shape} != {torch_output_tensor.shape}"

    # Use PCC for comparison (mesh partition should preserve values exactly)
    assert_with_pcc(torch_output_tensor, torch_output_from_tt, 0.9999)

    logger.info(f"✓ Trace mode {op_name} test passed with correct output")
