# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import time
from loguru import logger
import ttnn


def generate_delay_matrix(mesh_shape, base_delay=10000, variation=5000):
    """Generate a delay matrix with varying delays for each device in the mesh."""
    delays = []
    for row in range(mesh_shape[0]):
        delay_row = []
        for col in range(mesh_shape[1]):
            # Create different delays for each device
            delay = base_delay + (row * mesh_shape[1] + col) * variation // (mesh_shape[0] * mesh_shape[1])
            delay_row.append(delay)
        delays.append(delay_row)
    return delays


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
    """Run the apply_device_delay test with the given parameters."""
    mesh_device.enable_program_cache()

    logger.info(f"Running apply_device_delay test with mesh_shape={mesh_shape}")
    logger.info(f"Delays matrix: {delays}")
    logger.info(f"Trace mode: {trace_mode}, iterations: {num_iters}")

    # Setup subdevice if needed
    if subdevice_id is not None:
        compute_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
        subdevice_cores_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(compute_grid[0] - 1, compute_grid[1] - 1),
                ),
            }
        )

        worker_sub_device = ttnn.SubDevice([subdevice_cores_grid])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_stall_group = [worker_sub_device_id]
        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        mesh_device.set_sub_device_stall_group(sub_device_stall_group)
        subdevice_id = worker_sub_device_id

    def run_delay_op(n_iters):
        """Execute the apply_device_delay operation."""
        for i in range(n_iters):
            # Try top-level access first, then fallback to module access
            try:
                ttnn.apply_device_delay(
                    mesh_device,
                    delays,
                    queue_id=queue_id,
                    subdevice_id=subdevice_id,
                )
            except AttributeError:
                ttnn.operations.debug.apply_device_delay(
                    mesh_device,
                    delays,
                    queue_id=queue_id,
                    subdevice_id=subdevice_id,
                )
            if not trace_mode:
                ttnn.synchronize_device(mesh_device)

    # Measure execution time
    start_time = time.time()

    if trace_mode:
        # Compilation run
        logger.info("Compiling apply_device_delay operation")
        run_delay_op(1)

        # Warmup with trace
        if warmup_iters > 0:
            logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=queue_id)
            run_delay_op(warmup_iters)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=queue_id)
            ttnn.synchronize_device(mesh_device)

        # Main trace capture
        logger.info(f"Capturing main trace with {num_iters} iterations")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=queue_id)
        run_delay_op(num_iters)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=queue_id)
        ttnn.synchronize_device(mesh_device)

        # Execute traces
        logger.info("Executing traces")
        if warmup_iters > 0:
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)

        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)
    else:
        # Direct execution without trace
        logger.info(f"Executing {num_iters} iterations without trace")
        run_delay_op(num_iters)

    end_time = time.time()
    execution_time = end_time - start_time

    logger.info(f"Total execution time: {execution_time:.4f} seconds")

    # Calculate expected minimum time based on delays
    max_delay_cycles = max(max(row) for row in delays)
    expected_min_time = (max_delay_cycles * num_iters) / 1e9  # Assuming ~1GHz clock
    logger.info(f"Maximum delay per iteration: {max_delay_cycles} cycles")
    logger.info(f"Expected minimum time: {expected_min_time:.4f} seconds")

    # Cleanup subdevice if used
    if subdevice_id is not None:
        mesh_device.reset_sub_device_stall_group()

    # Verify program cache usage
    cache_entries = mesh_device.num_program_cache_entries()
    logger.info(f"Device has {cache_entries} program cache entries")

    # For delay operations, we expect exactly 1 cache entry per iteration without trace,
    # or 1 cache entry total with trace
    expected_cache_entries = 1 if trace_mode else num_iters
    assert (
        cache_entries == expected_cache_entries
    ), f"Expected {expected_cache_entries} cache entries, got {cache_entries}"


@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL},
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False, True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((1, 1), (1, 1), id="1x1_grid"),
        pytest.param((1, 2), (1, 2), id="1x2_grid"),
        pytest.param((2, 2), (2, 2), id="2x2_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("base_delay", [10000, 15000, 20000])
@pytest.mark.parametrize("num_iters", [1, 3])
@pytest.mark.parametrize("warmup_iters", [0, 1])
def test_apply_device_delay_basic(
    mesh_device,
    mesh_shape,
    trace_mode,
    base_delay,
    num_iters,
    warmup_iters,
    device_params,
):
    """Test basic apply_device_delay functionality with various delay values."""
    # Generate delay matrix with variation
    delays = generate_delay_matrix(mesh_shape, base_delay=base_delay, variation=5000)

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
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL},
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False, True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((2, 4), (2, 4), id="2x4_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("queue_id", [0])
def test_apply_device_delay_large_mesh(
    mesh_device,
    mesh_shape,
    trace_mode,
    queue_id,
    device_params,
):
    """Test apply_device_delay with larger mesh configurations."""
    # Create a more complex delay pattern for large mesh
    delays = []
    for row in range(mesh_shape[0]):
        delay_row = []
        for col in range(mesh_shape[1]):
            # Create a gradient pattern
            delay = 10000 + (row * 2000) + (col * 1000)
            delay_row.append(delay)
        delays.append(delay_row)

    run_apply_device_delay_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        delays=delays,
        num_iters=2,
        warmup_iters=1,
        trace_mode=trace_mode,
        queue_id=queue_id,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL},
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((1, 2), (1, 2), id="1x2_grid"),
    ],
    indirect=["mesh_device"],
)
def test_apply_device_delay_with_subdevice(
    mesh_device,
    mesh_shape,
    trace_mode,
    device_params,
):
    """Test apply_device_delay with subdevice configuration."""
    delays = [[12000, 18000]]  # Different delays for each device

    run_apply_device_delay_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        delays=delays,
        num_iters=2,
        warmup_iters=0,
        trace_mode=trace_mode,
        subdevice_id=True,  # Will be set up in the test function
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL},
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False, True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((1, 1), (1, 1), id="1x1_grid"),
    ],
    indirect=["mesh_device"],
)
def test_apply_device_delay_edge_cases(
    mesh_device,
    mesh_shape,
    trace_mode,
    device_params,
):
    """Test edge cases for apply_device_delay."""
    test_cases = [
        {"delays": [[10000]], "description": "minimum delay"},
        {"delays": [[20000]], "description": "maximum delay"},
        {"delays": [[0]], "description": "zero delay"},
    ]

    for case in test_cases:
        logger.info(f"Testing {case['description']}")
        run_apply_device_delay_test(
            mesh_device=mesh_device,
            mesh_shape=mesh_shape,
            delays=case["delays"],
            num_iters=1,
            warmup_iters=0,
            trace_mode=trace_mode,
        )


@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 100000},
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((2, 2), (2, 2), id="2x2_grid"),
    ],
    indirect=["mesh_device"],
)
def test_apply_device_delay_performance(
    mesh_device,
    mesh_shape,
    trace_mode,
    device_params,
):
    """Performance test with many iterations and trace mode."""
    # Create a pattern with varying delays
    delays = [[10000, 12000], [15000, 18000]]

    run_apply_device_delay_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        delays=delays,
        num_iters=10,
        warmup_iters=2,
        trace_mode=trace_mode,
    )
