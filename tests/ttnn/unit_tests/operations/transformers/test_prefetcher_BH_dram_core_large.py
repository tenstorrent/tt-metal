# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Parameterized end-to-end matmul tests for the DRAM-core prefetcher path, modeled after
test_prefetcher_BH (single Blackhole, simplified single-rank).

Each parameterized case:
- Builds a fresh DramSenderGlobalCircularBuffer for one matmul.
- Pushes the weight via ttnn.dram_prefetcher(run_on_dram_cores=True).
- Runs ttnn.linear with the matching dram_sender_global_cb.
- PCC against torch.matmul.

Known prototype limits (each is a follow-up):
- Multi-tensor or multi-layer in a single dram_prefetcher call leaves the DRISC cores in reset
  state (tt-triage: "Core is in reset"); root cause not yet identified. We work around by
  running one (prefetcher, matmul) pair per test case.
- > 2 DRAM banks reproducibly hangs all 8 DRISC kernels in reset on launch; cap is 2 banks.
- > 2 receivers/bank produces wrong values (gather_in0 ring traversal doesn't match the
  (bank, recv_in_bank) layout we use here). Cap is 2 receivers/bank.
- in0_block_w_tiles is hard-coded to 1 in the program factory.
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


# Parameterize over the matmul shape. K and N are in tiles of the per-rank-N-tile-per-receiver
# unit (so the actual K and N are these times ring_size * TILE_SIZE). All cases use the same
# 2-bank * 2-receiver-per-bank topology (the proven-working subset).
@pytest.mark.skipif(
    not _dram_programmable_enabled(), reason="TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set"
)
@pytest.mark.parametrize(
    "name,k_tiles_per_shard,n_tiles_per_receiver",
    [
        # name             K_tiles_per_shard   N_tiles_per_receiver
        ("qkv_small", 4, 1),  # K=128, N=128
        ("qkv_med", 16, 1),  # K=512, N=128
        ("qkv_large", 32, 1),  # K=1024, N=128
        ("ff_wide", 4, 2),  # K=128, N=256
        ("ff_widest", 8, 4),  # K=256, N=512
    ],
)
def test_dram_core_prefetcher_BH_param(device, name, k_tiles_per_shard, n_tiles_per_receiver):
    arch = getattr(device, "arch", lambda: None)()
    if arch is not None and "BLACKHOLE" not in str(arch).upper():
        pytest.skip("DRAM-core prefetcher matmul requires Blackhole")

    ttnn.device.enable_asynchronous_slow_dispatch(device)

    # ---- Topology: 2 DRAM banks * 2 receivers/bank → ring of 4 (proven-working subset) ----
    num_dram_banks = 2
    num_receivers_per_bank = 2
    ring_size = num_dram_banks * num_receivers_per_bank  # 4

    receiver_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ring_size - 1, 0))}
    )

    # ---- Shapes ----
    M = 32
    K = k_tiles_per_shard * ttnn.TILE_SIZE
    N = ring_size * n_tiles_per_receiver * ttnn.TILE_SIZE  # so each receiver gets `n_tiles_per_receiver` N-tiles

    # ---- Weight tensor (B): width-sharded in DRAM across `num_dram_banks` banks ----
    torch.manual_seed(hash(name) % 0x7FFFFFFF)
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
        pt_weight, device=device, dtype=ttnn.bfloat16, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
    )

    # ---- Activation (A): width-sharded on receiver cores; K split evenly across the ring ----
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

    # ---- tensor_addrs (unused by the DRAM-core path; required by the op contract) ----
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

    # ---- DramSenderGlobalCircularBuffer ----
    block_size_bytes = (K * (N // num_dram_banks) // num_receivers_per_bank) * 2  # bfloat16
    gcb_size = _round_up(block_size_bytes * 4, 4096)
    bank_to_receivers = []
    for b in range(num_dram_banks):
        start = b * num_receivers_per_bank
        end = start + num_receivers_per_bank - 1
        bank_to_receivers.append(
            (b, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(start, 0), ttnn.CoreCoord(end, 0))}))
        )
    gcb = ttnn.create_dram_sender_global_circular_buffer(device, bank_to_receivers, gcb_size)
    logger.info(
        f"[{name}] M={M} K={K} N={N} ring={ring_size} "
        f"K_per_shard={K_per_shard} block_size={block_size_bytes} gcb_size={gcb_size}"
    )

    # ---- Matmul program config (1D-mcast gather_in0, in0_block_w=1 to match the DRISC factory) ----
    in0_block_w = 1
    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N // ring_size // ttnn.TILE_SIZE
    out_subblock_w = out_block_w
    while out_subblock_w > 1 and out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1
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

    # ---- Run: prefetcher → matmul ----
    ttnn.dram_prefetcher([tt_weight, addrs], num_layers=1, run_on_dram_cores=True, dram_sender_global_cb=gcb)
    tt_out = ttnn.linear(
        tt_act,
        tt_weight,
        program_config=program_config,
        memory_config=output_mem_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
        dram_sender_global_cb=gcb,
    )

    # ---- Verify ----
    out_torch = ttnn.to_torch(tt_out)
    expected = pt_act.float() @ pt_weight.float()
    passing, output_str = comp_pcc(expected, out_torch, 0.99)
    logger.info(f"[{name}] {output_str}")
    assert passing, f"[{name}] PCC check failed: {output_str}"
