# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Parameterized end-to-end matmul tests for the DRAM-core prefetcher path on single Blackhole.

Topology: 8 DRAM banks * `recv_per_bank` receivers/bank -> ring of 8 * recv_per_bank.
Production Llama-3.1-8B on single BH uses recv_per_bank=8 (ring=64) for the prefetcher
matmuls; smaller rings are exercised here too for the QKV/FF cases that fit at lower
recv_per_bank.

Each parameterized case:
- Builds a fresh DRAM-sender GlobalCircularBuffer via
  ttnn.create_global_circular_buffer_with_dram_senders for one matmul.
- Pushes the weight via ttnn.dram_prefetcher(global_cb=gcb); the op infers the DRAM-core
  program factory from gcb.sender_core_type() == "dram".
- Runs ttnn.linear with the same gcb.
- PCC against torch.matmul.

Receiver layout: each bank's receivers are laid out at row-major-adjacent ring positions
[b*recv_per_bank, (b+1)*recv_per_bank) on a (RING_COLS x RING_ROWS) rectangle. This is the
layout gather_in0 expects (it walks worker_cores_vec row-major).

DRISC L1 budget: 2 ping-pong stage buffers must fit in ~80 KB. The factory automatically
splits each per-K-block DMA into M chunks (M divides num_receivers) when 2 * dma_block > 80 KB
-- the kernel does M subset-pushes per K-block with fifo_wr_ptr rewinds so all receivers
end up with their full contiguous slice at the same fifo offset. M=1 keeps the original
single-push-per-K-block path; M=2 unlocks FF1 (K=4096 N=14336) at ring=64.

Known prototype limits (each is a follow-up):
- in0_block_w_tiles can be overridden via `dram_core_k_block_w_tiles` op param, but values
  >=4 hang on a fifo-wrap edge case (gather_in0 already uses kbw=1 so this is non-blocking).
- Fast dispatch on the DRAM-core path is not implemented yet (slow dispatch only).

Multi-tensor in a single dram_prefetcher call (the "DRISC in reset" issue from earlier
versions) is verified working post-fix in test_prefetcher_BH_multi_tensor.py. Tests below
still run one (prefetcher, matmul) pair per case to keep the parametrize cases focused
on shape correctness.
"""

import math
import os
import pytest
import torch
import ttnn
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


# DRAM bank count is queried at runtime via `device.dram_grid_size().x`:
#   - P150/P300 (unharvested): 8 banks → ring=8*recv_per_bank.
#   - P100 (1 column harvested): 7 banks → ring=7*recv_per_bank.
# The receiver grid is laid out as `num_dram_banks` columns × `recv_per_bank` rows so the
# rectangle is always clean. Test names that suggest a fixed ring size (e.g. "r64") refer
# to the unharvested case; on P100 the same parametrization runs at ring=7*recv_per_bank.


def _dram_programmable_enabled() -> bool:
    return os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0") == "1"


def _round_up(n, m):
    return ((n + m - 1) // m) * m


def _bytes_per_tile(dtype) -> int:
    return {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}[dtype]


def _bank_receivers_row_major(bank_idx: int, recv_per_bank: int, ring_cols: int):
    """Row-major-adjacent receivers: bank b -> ring positions [b*recv_per_bank, (b+1)*recv_per_bank)."""
    cores = []
    for k in range(recv_per_bank):
        ring_pos = bank_idx * recv_per_bank + k
        col = ring_pos % ring_cols
        row = ring_pos // ring_cols
        cores.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
    return ttnn.CoreRangeSet(cores)


# Parameterize over (K, N) and (recv_per_bank, dtype). K and N are in tiles of
# (ring_size, n_tiles_per_recv) units so the shapes work out cleanly across rings:
#   K = k_tiles_per_shard * ring_size * TILE_SIZE
#   N = ring_size * n_tiles_per_receiver * TILE_SIZE
# Constraint: K_tiles_total = k_tiles_per_shard * ring_size must be a multiple of ring_size
# (trivially true by construction) so gather_in0 doesn't pad K.
@pytest.mark.skipif(
    not _dram_programmable_enabled(), reason="TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set"
)
@pytest.mark.parametrize(
    "name,k_tiles_per_shard,n_tiles_per_receiver,recv_per_bank,dtype",
    [
        # --- Legacy ring=8 cases (recv_per_bank=1, bf16) ---
        ("qkv_small", 8, 1, 1, ttnn.bfloat16),  # K=256, N=8*32
        ("qkv_med", 16, 1, 1, ttnn.bfloat16),  # K=512, N=8*32
        ("qkv_large", 32, 1, 1, ttnn.bfloat16),  # K=1024, N=8*32
        ("ff_wide", 8, 2, 1, ttnn.bfloat16),  # K=256, N=8*2*32
        ("ff_widest", 16, 4, 1, ttnn.bfloat16),  # K=512, N=8*4*32
        # --- Llama-3.1-8B production shapes at ring=64 (recv_per_bank=8, bf8_b) ---
        # K=4096, all use k_tiles_per_shard=2 (= 4096/64/32).
        ("o_proj_r64", 2, 2, 8, ttnn.bfloat8_b),  # K=4096, N=4096
        ("qkv_bf8_r64", 2, 6, 8, ttnn.bfloat8_b),  # K=4096, N=12288 (combined Q+K+V at bf8_b)
        ("ff1_r64", 2, 7, 8, ttnn.bfloat8_b),  # K=4096, N=14336 (FF1 gate/up)
    ],
)
def test_dram_core_prefetcher_BH_param(device, name, k_tiles_per_shard, n_tiles_per_receiver, recv_per_bank, dtype):
    arch = getattr(device, "arch", lambda: None)()
    if arch is not None and "BLACKHOLE" not in str(arch).upper():
        pytest.skip("DRAM-core prefetcher matmul requires Blackhole")

    ttnn.device.enable_asynchronous_slow_dispatch(device)

    # ---- Topology (queries DRAM bank count so the test adapts to harvested BHs) ----
    num_dram_banks = device.dram_grid_size().x
    num_receivers_per_bank = recv_per_bank
    ring_size = num_dram_banks * num_receivers_per_bank
    ring_cols = num_dram_banks
    ring_rows = num_receivers_per_bank
    assert ring_size == ring_cols * ring_rows, f"ring_size {ring_size} != ring_cols {ring_cols} * ring_rows {ring_rows}"

    receiver_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ring_cols - 1, ring_rows - 1))}
    )

    # ---- Shapes ----
    M = 32
    K = k_tiles_per_shard * ring_size * ttnn.TILE_SIZE
    N = ring_size * n_tiles_per_receiver * ttnn.TILE_SIZE

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
        pt_weight, device=device, dtype=dtype, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
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

    # ---- DRAM-sender GlobalCircularBuffer ----
    # gcb_size must be a multiple of matmul's receiver fifo_page_size to avoid remote_cb_wait_front
    # wrap-math overshoot. fifo_page_size = in0_block_w * per_core_N * tile_bytes (matmul-computed
    # in0_block_w = K_per_shard_tiles).
    tile_bytes = _bytes_per_tile(dtype)
    in1_block_size_bytes = k_tiles_per_shard * n_tiles_per_receiver * tile_bytes
    # Per-receiver weight footprint = K_tiles_total * n_tiles_per_receiver * tile_bytes
    max_tensor_bytes = (k_tiles_per_shard * ring_size) * n_tiles_per_receiver * tile_bytes
    gcb_size = max(in1_block_size_bytes, (max_tensor_bytes // in1_block_size_bytes) * in1_block_size_bytes)

    bank_to_receivers = [
        (b, _bank_receivers_row_major(b, num_receivers_per_bank, ring_cols)) for b in range(num_dram_banks)
    ]
    gcb = ttnn.create_global_circular_buffer_with_dram_senders(device, bank_to_receivers, gcb_size)
    logger.info(
        f"[{name}] M={M} K={K} N={N} ring={ring_size} recv/bank={num_receivers_per_bank} dtype={dtype} "
        f"K_per_shard={K_per_shard} fifo_page={in1_block_size_bytes} gcb_size={gcb_size}"
    )

    # ---- Matmul program config (1D-mcast gather_in0) ----
    in0_block_w = 1  # DRISC factory's kbw defaults to 1
    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N // ring_size // ttnn.TILE_SIZE
    out_subblock_w = min(out_block_w, 8)
    while out_subblock_w > 1 and out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(ring_cols, ring_rows),
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

    # ---- Run: prefetcher (async) -> matmul (consumes via gcb) -> stop drains ----
    ttnn.start_dram_core_prefetcher(device, [tt_weight, addrs], num_layers=1, global_cb=gcb)
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
    # Lower PCC threshold for bf8_b cases (precision noise is larger than at bf16).
    pcc_threshold = 0.99 if dtype == ttnn.bfloat16 else 0.999
    passing, output_str = comp_pcc(expected, out_torch, pcc_threshold)
    logger.info(f"[{name}] {output_str}")
    assert passing, f"[{name}] PCC check failed: {output_str}"
