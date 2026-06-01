# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Trace capture/replay for the DRAM-core prefetcher.

`queue_dram_core_prefetcher_request` does not go through the command queue: it
serializes a request into socket pages and a host worker thread fans them out to
the DRAM sender cores over NOC. When called with a `cq_id` whose command queue is
mid trace-capture, the request must be *captured* (not sent) and re-sent on every
`execute_trace` of that trace, so a captured matmul that consumes the GCB is
refilled on each replay.

This test captures (queue request + consuming linear) into a trace, replays it
several times, and asserts the matmul output is correct on every replay.
"""

import os
import zlib

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.prefetcher_common import round_up as _round_up, bytes_per_tile as _bytes_per_tile
from tests.ttnn.unit_tests.operations.transformers.test_prefetcher_BH_dram_core_large import _bank_receivers_row_major


pytestmark = [
    run_for_blackhole("DRAM-core prefetcher requires Blackhole"),
    pytest.mark.skipif(
        os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0") != "1",
        reason="TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set",
    ),
]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("replay_count", [1, 3])
def test_dram_core_prefetcher_trace_replay(device, replay_count):
    """Capture a (prefetcher-request + linear) pair into a trace and replay it
    `replay_count` times; each replay must refill the GCB and produce the right
    matmul output."""
    # ---- Topology (qkv_small shape; adapts to harvested DRAM bank counts) ----
    num_dram_banks = device.dram_grid_size().x
    num_receivers_per_bank = 1
    ring_size = num_dram_banks * num_receivers_per_bank
    ring_cols = num_dram_banks
    ring_rows = num_receivers_per_bank
    dtype = ttnn.bfloat16

    receiver_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ring_cols - 1, ring_rows - 1))}
    )

    M = 32
    k_tiles_per_shard = 8
    n_tiles_per_receiver = 1
    K = k_tiles_per_shard * ring_size * ttnn.TILE_SIZE
    N = ring_size * n_tiles_per_receiver * ttnn.TILE_SIZE

    # ---- Weight (B): width-sharded in DRAM across the banks ----
    torch.manual_seed(zlib.crc32(b"trace_replay"))
    pt_weight = torch.randn(1, 1, K, N)
    dram_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
    )
    weight_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [K, N // num_dram_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_weight = ttnn.as_tensor(
        pt_weight, device=device, dtype=dtype, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
    )

    # ---- Activation (A): width-sharded on receiver cores; persistent across replays ----
    pt_act = torch.randn(1, 1, M, K)
    K_per_shard = _round_up(-(-K // ring_size), ttnn.TILE_SIZE)
    act_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, K_per_shard),
        core_grid=receiver_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_act = ttnn.from_torch(
        pt_act, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=act_mem_config
    )

    # ---- Matmul program config (1D-mcast gather_in0) ----
    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N // ring_size // ttnn.TILE_SIZE
    out_subblock_w = min(out_block_w, 8)
    while out_subblock_w > 1 and out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(ring_cols, ring_rows),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=ttnn.CoreRangeSet([]),
        num_global_cb_receivers=num_receivers_per_bank,
        untilize_out=False,
    )

    # ---- DRAM-sender GlobalCircularBuffer ----
    tile_bytes = _bytes_per_tile(dtype)
    in1_block_size_bytes = k_tiles_per_shard * n_tiles_per_receiver * tile_bytes
    gcb_size = ring_size * in1_block_size_bytes
    bank_to_receivers = [
        (b, _bank_receivers_row_major(b, num_receivers_per_bank, ring_cols)) for b in range(num_dram_banks)
    ]
    gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
        device, [program_config], [tt_weight], bank_to_receivers=bank_to_receivers, size=gcb_size
    )

    output_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, N // ring_size),
        core_grid=receiver_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    expected = pt_act.float() @ pt_weight.float()

    def fill_and_matmul():
        ttnn.experimental.queue_dram_core_prefetcher_request(device, [(tt_weight, ring_size)], global_cb=gcb, cq_id=0)
        return ttnn.linear(
            tt_act,
            tt_weight,
            program_config=program_config,
            memory_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            global_cb=gcb,
        )

    ttnn.experimental.start_dram_core_prefetcher(device)
    try:
        # ---- Warmup: compile kernels + balance the GCB before trace capture. The
        # queue here is sent immediately (cq 0 not capturing) and consumed by the matmul. ----
        logger.info("Warmup (compile) run")
        tt_out = fill_and_matmul()
        ttnn.synchronize_device(device)
        warmup_torch = ttnn.to_torch(tt_out)
        passing, msg = comp_pcc(expected, warmup_torch, 0.999)
        assert passing, f"warmup PCC failed: {msg}"

        # ---- Capture: the queue request is captured into the trace (cq 0 mid-capture),
        # NOT sent now. The linear is captured too. ----
        logger.info("Capturing trace")
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        tt_out = fill_and_matmul()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        # ---- Replay: each execute_trace must replay the captured prefetcher request
        # (refilling the GCB) and the matmul. ----
        for i in range(replay_count):
            logger.info(f"Replay {i + 1}/{replay_count}")
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            out_torch = ttnn.to_torch(tt_out)
            passing, msg = comp_pcc(expected, out_torch, 0.999)
            assert passing, f"replay {i + 1} PCC failed: {msg}"

        ttnn.release_trace(device, trace_id)
    finally:
        ttnn.experimental.stop_dram_core_prefetcher(device)
        ttnn.synchronize_device(device)
