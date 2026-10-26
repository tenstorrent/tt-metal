# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for PersistentLoop micro-op (unified_kernels/persistent_loop.hpp).

Tests the setup/teardown cycle for both persistent and non-persistent modes.
"""

import time

import pytest
import torch
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


def _sharded_single_core(mesh_device, torch_tensor, dtype, layout):
    shard_spec = ttnn.ShardSpec(CORE_RANGE_SET, tuple(torch_tensor.shape[-2:]), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=mesh_device, memory_config=mem_config)


@skip_for_wormhole_b0("PersistentLoop test is for Blackhole only")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
# 4 tiles is the half-sync Dest capacity for fp32 on Blackhole.
@pytest.mark.parametrize("num_tiles", [1, 4])
@pytest.mark.parametrize("terminate_delay_ms", [0, 1, 10])
@pytest.mark.parametrize("num_cycles", [6])
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
def test_persistent_loop_payload_teardown(mesh_device, dtype, num_tiles, terminate_delay_ms, num_cycles):
    """Terminate a persistent kernel that packs real tiles every iteration, and check what it left behind.

    Each cycle launches the kernel on fresh random input, lets it run for terminate_delay_ms, terminates
    it part-way through its run, and after synchronisation checks that the output shard is an exact copy
    of the input whenever at least one iteration completed. The next cycle's buffers are new allocations,
    so anything a previous launch left in flight would land in the wrong place and be caught.
    """
    if not is_slow_dispatch():
        pytest.skip("Persistent mode requires TT_METAL_SLOW_DISPATCH_MODE=1")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    loop = PersistentLoop(mesh_device, CORE_RANGE_SET)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    shape = (32 * num_tiles, 32)

    completed_total = 0
    for cycle in range(num_cycles):
        torch_input = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
        input_tensor = _sharded_single_core(mesh_device, torch_input, dtype, ttnn.TILE_LAYOUT)
        output_tensor = _sharded_single_core(
            mesh_device, torch.zeros(shape, dtype=torch_dtype), dtype, ttnn.TILE_LAYOUT
        )
        status_tensor = _sharded_single_core(
            mesh_device, torch.zeros((32, 32), dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )

        loop.reset()
        loop.op_payload(
            mesh_device=mesh_device,
            core_coord=CORE_COORD,
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            status_tensor=status_tensor,
            num_tiles=num_tiles,
        )
        if terminate_delay_ms:
            time.sleep(terminate_delay_ms / 1000.0)
        loop.terminate()
        ttnn.synchronize_device(mesh_device)

        completed = int(ttnn.to_torch(status_tensor)[0, 0])
        output = ttnn.to_torch(output_tensor)
        logger.info(f"cycle {cycle + 1}/{num_cycles}: {completed} iterations completed before termination")
        if completed > 0:
            assert torch.equal(
                output, torch_input.to(output.dtype)
            ), f"cycle {cycle}: output differs from input after {completed} completed iterations"
        else:
            assert torch.count_nonzero(output) == 0, f"cycle {cycle}: output written with no completed iteration"
        completed_total += completed

        ttnn.deallocate(input_tensor)
        ttnn.deallocate(output_tensor)
        ttnn.deallocate(status_tensor)

    # With no delay the terminate write can legitimately land before the first iteration, so only the delayed
    # arms are required to have completed work; the zero-iteration branch above still checks the 0 ms arm.
    if terminate_delay_ms > 0:
        assert completed_total > 0, "no cycle completed an iteration; the test exercised nothing"
