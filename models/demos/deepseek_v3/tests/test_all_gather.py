# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.utils.config_dataclass import AllGatherAsyncConfig


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 2789376,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("enable_trace", [True, False])
def test_all_gather_async(mesh_device, enable_trace):
    """
    Args:
        enable_trace: If True, capture and execute trace for all_gather_async operation.
    """
    logger.info(f"Running test_all_gather_async with mesh_device: {mesh_device.shape}, enable_trace: {enable_trace}")

    ccl = CCL(mesh_device)
    stats_shape = [1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE]
    torch_stats = torch.randn(stats_shape, dtype=torch.bfloat16)

    width_sharded_l1_memcfg = ttnn.create_sharded_memory_config(
        shape=stats_shape,
        core_grid=ttnn.CoreGrid(x=1, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    tt_stats = ttnn.from_torch(
        torch_stats,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=width_sharded_l1_memcfg,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(f"tt_stats shape: {tt_stats.shape}")

    all_gather_output_shape = [1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE * mesh_device.shape[1]]
    all_gather_output_memcfg = ttnn.create_sharded_memory_config(
        shape=all_gather_output_shape,
        core_grid=ttnn.CoreGrid(x=mesh_device.shape[1], y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
    )

    # Configure all_gather similar to distributed_rms_norm.py
    all_gather_config = AllGatherAsyncConfig(
        dim=3,
        cluster_axis=1,
        mesh_device=mesh_device,
        memory_config=all_gather_output_memcfg,
        topology=ttnn.Topology.Linear,
    )

    cfg = {"all_gather": all_gather_config.__dict__}

    # Synchronize device before all_gather
    ttnn.synchronize_device(mesh_device)

    # Trace state variables
    trace_id: int | None = None
    trace_input: ttnn.Tensor | None = None
    trace_output: ttnn.Tensor | None = None

    if enable_trace:
        # 1) Warm-up compile run (no trace) to keep compilation out of capture
        logger.info("Running warm-up all_gather_async step (no trace)...")
        tt_stats_gathered = ttnn.experimental.all_gather_async(
            tt_stats, **ccl.populate_all_gather_runtime_args(cfg["all_gather"])
        )

        # 2) Allocate persistent device input
        trace_input = ttnn.from_torch(
            torch_stats,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=width_sharded_l1_memcfg,
            layout=ttnn.TILE_LAYOUT,
        )

        # 3) Capture all_gather graph
        ccl.reset_sem_counters()
        ttnn.synchronize_device(mesh_device)
        logger.info("Begin capturing all_gather_async trace...")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_output = ttnn.experimental.all_gather_async(
            trace_input, **ccl.populate_all_gather_runtime_args(cfg["all_gather"])
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info("All_gather_async trace capture complete.")

        # Synchronize after trace capture
        ttnn.synchronize_device(mesh_device)

        # Verify output shape from trace
        expected_gathered_shape = list(stats_shape)
        expected_gathered_shape[-1] = stats_shape[-1] * mesh_device.shape[1]  # All gather along dim 3
        assert trace_output.shape == tuple(
            expected_gathered_shape
        ), f"Expected shape {expected_gathered_shape}, got {trace_output.shape}"

        # Execute trace again with updated input to verify it works
        logger.info("Executing trace with updated input...")
        # Update persistent input
        host_input = ttnn.from_torch(
            torch_stats,
            device=None,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_input, trace_input)
        ttnn.deallocate(host_input)

        ccl.reset_sem_counters()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        # Use trace output
        tt_gathered_stats = trace_output
    else:
        # Apply all_gather_async with CCL runtime args (this is what we're testing)
        tt_gathered_stats = ttnn.experimental.all_gather_async(
            tt_stats, **ccl.populate_all_gather_runtime_args(cfg["all_gather"])
        )

        # Synchronize device after all_gather
        ttnn.synchronize_device(mesh_device)

    logger.info(f"tt_gathered_stats shape: {tt_gathered_stats.shape}")

    # Verify output shape
    expected_gathered_shape = list(stats_shape)
    expected_gathered_shape[-1] = stats_shape[-1] * mesh_device.shape[1]  # All gather along dim 3
    assert tt_gathered_stats.shape == tuple(
        expected_gathered_shape
    ), f"Expected shape {expected_gathered_shape}, got {tt_gathered_stats.shape}"

    # Cleanup
    ttnn.deallocate(tt_stats)
    if not enable_trace:
        ttnn.deallocate(tt_gathered_stats)
    else:
        # Cleanup trace resources
        if trace_input is not None:
            ttnn.deallocate(trace_input)
        if trace_output is not None:
            ttnn.deallocate(trace_output)

    logger.info("Test passed!")


if __name__ == "__main__":
    pytest.main([__file__])
