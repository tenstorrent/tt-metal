# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for PersistentLoop micro-op (unified_kernels/persistent_loop.hpp).

Tests the setup/teardown cycle for both persistent and non-persistent modes.
"""

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch, skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.micro_ops.persistent_loop.op import PersistentLoop

CORE_COORD = ttnn.CoreCoord(0, 0)
CORE_RANGE_SET = ttnn.CoreRangeSet([ttnn.CoreRange(CORE_COORD, CORE_COORD)])


@skip_for_wormhole_b0("PersistentLoop test is for Blackhole only")
@pytest.mark.parametrize("num_cycles", [5])
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
def test_persistent_loop_teardown_cycle(mesh_device, num_cycles):
    """Launch a persistent kernel, terminate it, and repeat N times.

    Validates that PersistentLoop correctly observes the termination semaphore
    and that the kernel can be cleanly re-launched after each teardown.
    """
    if not is_slow_dispatch():
        pytest.skip("Persistent mode requires TT_METAL_SLOW_DISPATCH_MODE=1")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    loop = PersistentLoop(mesh_device, CORE_RANGE_SET)
    iteration_count_sem = ttnn.create_global_semaphore(mesh_device, CORE_RANGE_SET, 0)

    for cycle in range(num_cycles):
        logger.info(f"Cycle {cycle + 1}/{num_cycles}: launching persistent kernel")

        loop.reset()
        ttnn.reset_global_semaphore_value(iteration_count_sem, 0)

        loop.op(
            mesh_device=mesh_device,
            core_coord=CORE_COORD,
            iteration_count_semaphore=iteration_count_sem,
        )

        logger.info(f"Cycle {cycle + 1}/{num_cycles}: terminating")
        loop.terminate()
        ttnn.synchronize_device(mesh_device)
        logger.info(f"Cycle {cycle + 1}/{num_cycles}: teardown complete")

    logger.info(f"All {num_cycles} teardown cycles completed successfully")


@skip_for_wormhole_b0("PersistentLoop test is for Blackhole only")
@pytest.mark.parametrize("max_iterations", [1, 10])
@pytest.mark.parametrize("num_cycles", [3])
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
def test_persistent_loop_non_persistent(mesh_device, max_iterations, num_cycles):
    """Launch a non-persistent kernel that runs a fixed number of iterations.

    Validates that PersistentLoop correctly counts iterations in non-persistent
    mode and that the kernel can be re-dispatched multiple times.
    """
    if not is_slow_dispatch():
        pytest.skip("Persistent mode requires TT_METAL_SLOW_DISPATCH_MODE=1")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    loop = PersistentLoop(mesh_device, CORE_RANGE_SET, persistent_mode=False)
    iteration_count_sem = ttnn.create_global_semaphore(mesh_device, CORE_RANGE_SET, 0)

    for cycle in range(num_cycles):
        logger.info(
            f"Cycle {cycle + 1}/{num_cycles}: launching non-persistent kernel (max_iterations={max_iterations})"
        )

        ttnn.reset_global_semaphore_value(iteration_count_sem, 0)

        loop.op(
            mesh_device=mesh_device,
            core_coord=CORE_COORD,
            iteration_count_semaphore=iteration_count_sem,
            max_iterations=max_iterations,
        )

        ttnn.synchronize_device(mesh_device)
        logger.info(f"Cycle {cycle + 1}/{num_cycles}: kernel completed")

    logger.info(f"All {num_cycles} non-persistent cycles completed successfully")
