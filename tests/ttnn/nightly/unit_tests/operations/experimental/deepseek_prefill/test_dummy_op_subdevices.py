# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Experiment scaffold: mirror TtMoe's 2-sub-device split (row 0 = dispatch,
rows 1.. = shared) and run dummy_op on the dispatch sub-device while a host
for-loop runs a "routed-expert-like" matmul on shared_sd and bumps a global
semaphore between iterations to drive the reader kernel's per-iter wait.
"""

import pytest
import torch
import ttnn
from loguru import logger
from tracy import signpost


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "debug_print_global_semaphore",
    [True, False],
    ids=["with_print", "without_print"],
)
@pytest.mark.usefixtures("tracy_profile")
def test_dummy_op_on_dispatch_subdevice(device, debug_print_global_semaphore):
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = grid.x, grid.y
    assert grid_y > 1, f"need grid_y > 1 for a dispatch/shared split, got grid_y={grid_y}"

    # Same topology as TtMoe (tt_moe.py:268-283 with overlap enabled).
    dispatch_sd_rows = 1
    dispatch_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, dispatch_sd_rows - 1))}
    )
    shared_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, dispatch_sd_rows), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
    )
    dispatch_sd = ttnn.SubDevice([dispatch_cores])
    shared_sd = ttnn.SubDevice([shared_cores])

    sd_manager_id = device.create_sub_device_manager([dispatch_sd, shared_sd], 0)
    dispatch_sd_id = ttnn.SubDeviceId(0)
    shared_sd_id = ttnn.SubDeviceId(1)
    try:
        num_iter = 4

        # Monotonic counter of completed iterations: starts at 0, host bumps
        # to 1, 2, ..., num_iter. Reader's iter-i wait is for *sem >= i+1.
        sem = ttnn.create_global_semaphore(
            device, dispatch_cores, initial_value=0, buffer_type=ttnn.BufferType.L1_SMALL
        )

        # dummy_op input: tile count must be divisible by num_cores * 8.
        tiles_per_core = 8 * 8
        num_tiles = grid_x * tiles_per_core
        t = ttnn.from_torch(
            torch.zeros((32, 32 * num_tiles), dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Two DRAM-interleaved tensors for the pointless matmul that simulates
        # the routed-expert workload running concurrent with dummy_op.
        mm_a = ttnn.from_torch(
            torch.randn((2048, 7 * 1024), dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mm_b = ttnn.from_torch(
            torch.randn((7 * 1024, 512), dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Pre-built program config + compute kernel config, mirroring tt_shared_expert
        # which always passes explicit configs to ttnn.matmul. Values mirror what the
        # auto-config would have generated, except `compute_with_storage_grid_size` is
        # deprecated for sub-device use; we pass `allowed_worker_cores=shared_cores`
        # so the matmul stays inside shared_sd.
        # M=64 tiles, N=16 tiles. Shared grid is grid_x x (grid_y - dispatch_sd_rows) = 8 x 7
        # on Wormhole. per_core_M=16 -> num_blocks_y=4 (<=7); per_core_N=2 -> num_blocks_x=8
        # (<=8); both divide M/N evenly so no padding. Keeps the matmul inside shared_cores.
        matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y - dispatch_sd_rows),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=16,
            per_core_N=2,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
            allowed_worker_cores=shared_cores,
        )
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        for iteration_idx in range(3):
            logger.debug("iteration {} start", iteration_idx)
            signpost(header=f"iteration {iteration_idx}")

            # Load sub-device manager AFTER all tensor uploads/sem allocation (which use
            # the full-grid default allocator). This mirrors TtMoe.forward, which only
            # loads its sub-device manager right before the sub-device-aware ops.
            device.load_sub_device_manager(sd_manager_id)

            # Launch dummy_op (non-blocking). The reader kernel will block on the
            # semaphore between its iter loop iterations.
            ttnn.experimental.deepseek_prefill.dummy_op(
                t, num_iter=num_iter, global_semaphore=sem, subdevice_id=dispatch_sd_id
            )
            logger.debug("dummy_op enqueued on dispatch_sd (num_iter={})", num_iter)

            # Stall only on shared_sd so host commands (the readback) don't wait on
            # dispatch_sd, which is busy running dummy_op and would deadlock. Only
            # set when we're actually issuing host-side reads inside the loop.
            if debug_print_global_semaphore:
                device.set_sub_device_stall_group([shared_sd_id])
                logger.debug("set_sub_device_stall_group([shared_sd_id])")

            # Each iter: run a "pointless" matmul confined to shared_sd (so it
            # doesn't fight dummy_op for dispatch_sd cores), then bump the
            # semaphore to unblock dummy_op's reader for iter i.
            for i in range(num_iter):
                ttnn.matmul(
                    mm_a,
                    mm_b,
                    program_config=matmul_program_config,
                    compute_kernel_config=compute_kernel_config,
                    sub_device_id=shared_sd_id,
                    global_semaphore=sem,
                )
                # logger.debug("matmul iter {} enqueued on shared_sd", i)
                # if debug_print_global_semaphore:
                #     logger.debug("iter {}: global_semaphore = {}", i, ttnn.read_global_semaphore_value(sem))

            if debug_print_global_semaphore:
                device.reset_sub_device_stall_group()
                logger.debug("reset_sub_device_stall_group()")

            ttnn.reset_global_semaphore_value(sem, 0)
            ttnn.synchronize_device(device)

            logger.debug("iteration {} end", iteration_idx)
    finally:
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(sd_manager_id)
