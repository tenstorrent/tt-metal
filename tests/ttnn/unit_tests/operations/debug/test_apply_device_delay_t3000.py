# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import time
from loguru import logger
import ttnn
from tracy import signpost


def run_apply_device_delay_test(
    mesh_device,
    mesh_shape,
    delays,
    num_iters,
    warmup_iters,
    trace_mode,
    queue_id=0,
    subdevice_id=None,
):
    """Run the apply_device_delay test optimized for T3000."""
    mesh_device.enable_program_cache()

    logger.info(f"Running T3000 apply_device_delay test")
    logger.info(f"Mesh shape: {mesh_shape}")
    logger.info(f"Delays: {delays}")
    logger.info(f"Iterations: {num_iters}, Warmup: {warmup_iters}, Trace: {trace_mode}")

    def run_delay_op(n_iters):
        """Execute the apply_device_delay operation."""
        for i in range(n_iters):
            ttnn.apply_device_delay(
                mesh_device,
                delays,
            )
            if not trace_mode:
                ttnn.synchronize_device(mesh_device)

    if trace_mode:
        # Compilation run
        logger.info("Compiling apply_device_delay operation")
        run_delay_op(1)

        # Warmup with trace
        if warmup_iters > 0:
            logger.info(f"Capturing warmup trace")
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=queue_id)
            run_delay_op(warmup_iters)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=queue_id)
            ttnn.synchronize_device(mesh_device)

        # Main trace capture
        logger.info(f"Capturing main trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=queue_id)
        run_delay_op(num_iters)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=queue_id)
        ttnn.synchronize_device(mesh_device)

        # Execute traces with timing
        logger.info("Starting trace execution")
        if warmup_iters > 0:
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)

        signpost("start")
        start_time = time.time()
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)
        end_time = time.time()
        signpost("stop")

    else:
        # Direct execution without trace
        logger.info("Starting direct execution")
        signpost("start")
        start_time = time.time()
        run_delay_op(num_iters)
        end_time = time.time()
        signpost("stop")

    execution_time = end_time - start_time
    logger.info(f"Execution time: {execution_time:.4f} seconds")

    # Calculate expected minimum time based on delays
    max_delay_cycles = max(max(row) for row in delays)
    expected_min_time = (max_delay_cycles * num_iters) / 1e9  # Assuming ~1GHz clock
    logger.info(f"Max delay per iteration: {max_delay_cycles} cycles")
    logger.info(f"Expected minimum time: {expected_min_time:.4f} seconds")

    # Cleanup
    mesh_device.reset_sub_device_stall_group()

    # Verify program cache
    cache_entries = mesh_device.num_program_cache_entries()
    logger.info(f"Program cache entries: {cache_entries}")
    expected_cache_entries = 1
    assert (
        cache_entries == expected_cache_entries
    ), f"Expected {expected_cache_entries} cache entries, got {cache_entries}"


@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("base_delay", [10000])
@pytest.mark.parametrize("num_iters, warmup_iters", [(5, 5)])
def test_apply_device_delay_t3000(
    mesh_device,
    mesh_shape,
    trace_mode,
    base_delay,
    num_iters,
    warmup_iters,
    device_params,
):
    """Test apply_device_delay on T3000 with various delay patterns."""

    # Generate delay patterns # random
    import random

    random.seed(42)  # For reproducibility
    delays = []
    for row in range(mesh_shape[0]):
        delay_row = []
        for col in range(mesh_shape[1]):
            delay = base_delay + random.randint(0, 5000)
            delay_row.append(delay)
        delays.append(delay_row)

    run_apply_device_delay_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        delays=delays,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        trace_mode=trace_mode,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 100000,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
def test_apply_device_delay_trace_t3000(
    mesh_device,
    mesh_shape,
    trace_mode,
    device_params,
):
    """Trace test for apply_device_delay on T3000."""

    # Create a complex delay pattern
    delays = [[10000, 12000, 14000, 16000], [11000, 0, 15000, 17000]]

    run_apply_device_delay_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        delays=delays,
        num_iters=20,
        warmup_iters=5,
        trace_mode=trace_mode,
    )
