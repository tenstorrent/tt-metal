# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end matmul test for the DRAM-core prefetcher path.

Wires:
- ttnn.dram_prefetcher(global_cb=gcb)
- ttnn.linear(global_cb=gcb, gather_in0=True 1D-mcast)
- Verifies via PCC against torch.matmul(in0, weight).
"""

import math
import os
import pytest
import torch
import ttnn
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def _dram_programmable_enabled() -> bool:
    return os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0") == "1"


def _round_up(n, m):
    return ((n + m - 1) // m) * m


@pytest.mark.skipif(
    not _dram_programmable_enabled(), reason="TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set"
)
def test_dram_core_prefetcher_matmul(device):
    arch = getattr(device, "arch", lambda: None)()
    if arch is not None and "BLACKHOLE" not in str(arch).upper():
        pytest.skip("DRAM-core prefetcher matmul requires Blackhole")

    # Async slow dispatch: the prefetcher (DRISC cores) and matmul (worker cores) live on
    # disjoint programmable-core types, so the SD CQ will launch them concurrently and only
    # serialize when their core sets intersect.
    ttnn.device.enable_asynchronous_slow_dispatch(device)

    # ---- Topology: 2 DRAM bank senders → 2 worker receivers each ----
    num_dram_banks = 2
    num_receivers_per_bank = 2
    ring_size = num_dram_banks * num_receivers_per_bank  # 4

    # Receivers laid out so each bank's receivers form their own contiguous row.
    receiver_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ring_size - 1, 0))}
    )

    # ---- Shapes: M x K @ K x N ----
    M = 32
    K = ring_size * ttnn.TILE_SIZE  # 128
    N = ring_size * ttnn.TILE_SIZE  # 128
    assert K % (ring_size * ttnn.TILE_SIZE) == 0, "K must split evenly across ring"
    assert N % (ring_size * ttnn.TILE_SIZE) == 0, "N must split evenly across ring"

    # ---- Weight tensor (B): width-sharded in DRAM across `num_dram_banks` banks ----
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
        pt_weight,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=weight_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # ---- Activation tensor (A): width-sharded on receiver cores ----
    pt_act = torch.randn(1, 1, M, K)
    K_per_shard = _round_up(math.ceil(K / ring_size), ttnn.TILE_SIZE)
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

    # ---- tensor_addrs (host-allocated L1 tensor expected by the op contract; unused on DRAM-core path) ----
    addrs = ttnn.from_torch(
        torch.zeros(1, 1),
        device=device,
        dtype=ttnn.uint32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                [1, 1],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )

    # ---- DramSenderGlobalCircularBuffer: each bank → its disjoint subset of receivers ----
    # GCB size big enough for the whole weight slice per receiver.
    block_size_bytes = (K * (N // num_dram_banks) // num_receivers_per_bank) * 2  # bfloat16
    gcb_size = _round_up(block_size_bytes * 4, 4096)
    bank_to_receivers = []
    for b in range(num_dram_banks):
        # Disjoint per-bank receiver subset.
        start = b * num_receivers_per_bank
        end = start + num_receivers_per_bank - 1
        bank_to_receivers.append(
            (b, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(start, 0), ttnn.CoreCoord(end, 0))}))
        )
    gcb = ttnn.create_global_circular_buffer_with_dram_senders(device, bank_to_receivers, gcb_size)

    # ---- Matmul program config: 1D-mcast gather_in0, num_global_cb_receivers=1 ----
    in0_block_w = K // ring_size // ttnn.TILE_SIZE or 1
    while in0_block_w > 0 and (K // ttnn.TILE_SIZE) % in0_block_w != 0:
        in0_block_w -= 1
    if in0_block_w == 0:
        in0_block_w = 1
    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N // ring_size // ttnn.TILE_SIZE
    out_subblock_w = max(1, out_block_w)
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(num_dram_banks, num_receivers_per_bank),
        in0_block_w=in0_block_w,
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

    # ---- Run: prefetcher (async) → matmul (consumes via gcb) → stop drains ----
    ttnn.start_dram_core_prefetcher(
        device,
        [tt_weight, addrs],
        num_layers=1,
        global_cb=gcb,
    )

    tt_out = ttnn.linear(
        tt_act,
        tt_weight,
        program_config=program_config,
        memory_config=output_mem_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
        global_cb=gcb,
    )
    ttnn.stop_dram_core_prefetcher(device)

    # ---- Verify ----
    out_torch = ttnn.to_torch(tt_out)
    expected = pt_act.float() @ pt_weight.float()
    passing, output_str = comp_pcc(expected, out_torch, 0.99)
    logger.info(f"DRAM-core prefetcher matmul: {output_str}")
    assert passing, f"PCC check failed: {output_str}"
