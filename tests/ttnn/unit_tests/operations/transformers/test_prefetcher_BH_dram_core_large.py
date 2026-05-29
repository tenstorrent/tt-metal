# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the DRAM-core prefetcher path on single Blackhole.

Two scopes:

1. ``test_dram_core_prefetcher_BH_param`` — parameterized shape coverage with
   PCC vs ``torch.matmul``. Topology: 8 DRAM banks × ``recv_per_bank``
   receivers/bank → ring of 8 × recv_per_bank. Production Llama-3.1-8B on
   single BH uses recv_per_bank=8 (ring=64); smaller rings are exercised for
   QKV/FF cases that fit at lower recv_per_bank.

   Each case builds a fresh DRAM-sender GlobalCircularBuffer, launches the
   DRAM-core prefetcher via ``ttnn.experimental.start_dram_core_prefetcher``,
   runs ``ttnn.linear`` with the same gcb, and PCC-checks against
   ``torch.matmul``. ``stop_dram_core_prefetcher`` drains the pipeline after
   the matmul finishes.

   Receiver layout: each bank's receivers sit at row-major-adjacent ring
   positions ``[b*recv_per_bank, (b+1)*recv_per_bank)`` on a
   ``(RING_COLS × RING_ROWS)`` rectangle — the layout ``gather_in0`` expects.

   DRISC L1 budget: 2 ping-pong stage buffers must fit in ~80 KB. The
   factory automatically splits each per-K-block DMA into M chunks
   (M divides num_receivers) when ``2 * dma_block > 80 KB``; the kernel does
   M subset-pushes per K-block with fifo_wr_ptr rewinds so all receivers end
   up with their full contiguous slice at the same fifo offset. M=1 keeps
   the original single-push-per-K-block path; M=2 unlocks FF1
   (K=4096 N=14336) at ring=64.

2. ``test_dram_core_prefetcher_multi_tensor`` — multi-tensor smoke. The
   DRAM-core kernel's main loop iterates
   ``for layer in num_layers: for t in num_tensors:`` and prior versions
   were documented as leaving the DRISC cores in reset state when
   num_tensors > 1. After the chunked-DMA + receiver-layout fixes this case
   re-verifies the multi-tensor path end-to-end against a discard receiver
   (no matmul; the goal is to confirm both ops complete without
   hang/OOM/PCC failure across multiple ``(num_tensors, num_layers)`` combos).
"""

import math
import os
import zlib
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.prefetcher_common import round_up as _round_up, bytes_per_tile as _bytes_per_tile


# DRAM bank count is queried at runtime via `device.dram_grid_size().x`:
#   - P150/P300 (unharvested): 8 banks → ring=8*recv_per_bank.
#   - P100 (1 column harvested): 7 banks → ring=7*recv_per_bank.
# The receiver grid is laid out as `num_dram_banks` columns × `recv_per_bank` rows so the
# rectangle is always clean. Test names that suggest a fixed ring size (e.g. "r64") refer
# to the unharvested case; on P100 the same parametrization runs at ring=7*recv_per_bank.


pytestmark = [
    run_for_blackhole("DRAM-core prefetcher requires Blackhole"),
    pytest.mark.skipif(
        os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0") != "1",
        reason="TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set",
    ),
]


def _bank_receivers_row_major(bank_idx: int, recv_per_bank: int, ring_cols: int):
    """Row-major-adjacent receivers: bank b -> ring positions [b*recv_per_bank, (b+1)*recv_per_bank)."""
    cores = []
    for k in range(recv_per_bank):
        ring_pos = bank_idx * recv_per_bank + k
        col = ring_pos % ring_cols
        row = ring_pos // ring_cols
        cores.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
    return ttnn.CoreRangeSet(cores)


# ---------------------------------------------------------------------------
# Parameterized shape coverage (PCC vs torch.matmul)
# ---------------------------------------------------------------------------
# Parameterize over (K, N) and (recv_per_bank, dtype). K and N are in tiles of
# (ring_size, n_tiles_per_recv) units so the shapes work out cleanly across rings:
#   K = k_tiles_per_shard * ring_size * TILE_SIZE
#   N = ring_size * n_tiles_per_receiver * TILE_SIZE
# Constraint: K_tiles_total = k_tiles_per_shard * ring_size must be a multiple of ring_size
# (trivially true by construction) so gather_in0 doesn't pad K.
@pytest.mark.parametrize(
    "name,k_tiles_per_shard,n_tiles_per_receiver,recv_per_bank,dtype,gcb_size_misalign_bytes",
    [
        # --- Legacy ring=8 cases (recv_per_bank=1, bf16) ---
        ("qkv_small", 8, 1, 1, ttnn.bfloat16, 0),  # K=256, N=8*32
        ("qkv_med", 16, 1, 1, ttnn.bfloat16, 0),  # K=512, N=8*32
        ("qkv_large", 32, 1, 1, ttnn.bfloat16, 0),  # K=1024, N=8*32
        ("ff_wide", 8, 2, 1, ttnn.bfloat16, 0),  # K=256, N=8*2*32
        ("ff_widest", 16, 4, 1, ttnn.bfloat16, 0),  # K=512, N=8*4*32
        # --- Llama-3.1-8B production shapes at ring=64 (recv_per_bank=8, bf8_b) ---
        # K=4096, all use k_tiles_per_shard=2 (= 4096/64/32).
        ("o_proj_r64", 2, 2, 8, ttnn.bfloat8_b, 0),  # K=4096, N=4096
        ("qkv_bf8_r64", 2, 6, 8, ttnn.bfloat8_b, 0),  # K=4096, N=12288 (combined Q+K+V at bf8_b)
        ("ff1_r64", 2, 7, 8, ttnn.bfloat8_b, 0),  # K=4096, N=14336 (FF1 gate/up)
        # --- Misaligned-gcb_size cases: gcb_size is NOT a multiple of in1_block_size.
        # Exercises the wrap-adjustment path the previous factory always avoided by
        # snapping gcb_size to LCM(in1_block_size). Misalignment is one L1_ALIGNMENT
        # (=16 B on BH) so the underlying CB invariant `page_size % L1_ALIGNMENT == 0`
        # still holds. ---
        ("qkv_small_misaligned", 8, 1, 1, ttnn.bfloat16, 16),
        ("ff_wide_misaligned", 8, 2, 1, ttnn.bfloat16, 16),
    ],
)
def test_dram_core_prefetcher_BH_param(
    device, name, k_tiles_per_shard, n_tiles_per_receiver, recv_per_bank, dtype, gcb_size_misalign_bytes
):
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
    torch.manual_seed(zlib.crc32(name.encode()))
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

    # ---- DRAM-sender GlobalCircularBuffer (via matmul-aware factory) ----
    # in1_block_size is the matmul's receiver fifo_page_size = in0_block_w * per_core_N * tile_bytes
    # (matmul-computed in0_block_w = K_per_shard_tiles).
    tile_bytes = _bytes_per_tile(dtype)
    in1_block_size_bytes = k_tiles_per_shard * n_tiles_per_receiver * tile_bytes
    # Per-receiver weight footprint = num_blocks * in1_block_size (the factory's minimum).
    gcb_size = ring_size * in1_block_size_bytes
    if gcb_size_misalign_bytes != 0:
        gcb_size += gcb_size_misalign_bytes
        assert (
            gcb_size % in1_block_size_bytes != 0
        ), "misalignment test requires gcb_size to NOT be a multiple of in1_block_size"

    bank_to_receivers = [
        (b, _bank_receivers_row_major(b, num_receivers_per_bank, ring_cols)) for b in range(num_dram_banks)
    ]
    gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
        device, [program_config], [tt_weight], bank_to_receivers=bank_to_receivers, size=gcb_size
    )
    logger.info(
        f"[{name}] M={M} K={K} N={N} ring={ring_size} recv/bank={num_receivers_per_bank} dtype={dtype} "
        f"K_per_shard={K_per_shard} fifo_page={in1_block_size_bytes} gcb_size={gcb_size}"
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
    ttnn.experimental.start_dram_core_prefetcher(device)
    ttnn.experimental.queue_dram_core_prefetcher_request(device, [(tt_weight, ring_size)], num_layers=1, global_cb=gcb)
    tt_out = ttnn.linear(
        tt_act,
        tt_weight,
        program_config=program_config,
        memory_config=output_mem_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
        global_cb=gcb,
    )
    ttnn.experimental.stop_dram_core_prefetcher(device)

    # ---- Verify ----
    out_torch = ttnn.to_torch(tt_out)
    expected = pt_act.float() @ pt_weight.float()
    # Lower PCC threshold for bf8_b cases (precision noise is larger than at bf16).
    pcc_threshold = 0.999 if dtype == ttnn.bfloat16 else 0.99
    passing, output_str = comp_pcc(expected, out_torch, pcc_threshold)
    logger.info(f"[{name}] {output_str}")
    assert passing, f"[{name}] PCC check failed: {output_str}"


# ---------------------------------------------------------------------------
# create_global_circular_buffer_for_matmul_1d factory
# ---------------------------------------------------------------------------
# Drive the same shape as the qkv_small case (K=256, N=8*32, bf16) but build the GCB
# via the matmul-aware factory instead of calling create_global_circular_buffer_with_dram_senders
# directly. End-to-end PCC check confirms the factory's bank-to-receivers mapping +
# size validation match what the matmul + prefetcher expect.
@pytest.mark.parametrize(
    "layers_buffered",
    [
        1,  # minimum: exactly one layer worth of pages (num_blocks * in1_block_size)
        2,  # 2 layers buffered: double-buffer between prefetcher and matmul
    ],
)
def test_create_global_circular_buffer_for_matmul_1d(device, layers_buffered):
    num_dram_banks = device.dram_grid_size().x
    recv_per_bank = 1
    ring_size = num_dram_banks * recv_per_bank
    ring_cols = num_dram_banks
    ring_rows = recv_per_bank
    dtype = ttnn.bfloat16
    k_tiles_per_shard = 8  # K = 8 * ring_size * TILE_SIZE
    n_tiles_per_receiver = 1  # N = ring_size * 1 * TILE_SIZE
    tile_bytes = _bytes_per_tile(dtype)

    M = 32
    K = k_tiles_per_shard * ring_size * ttnn.TILE_SIZE
    N = ring_size * n_tiles_per_receiver * ttnn.TILE_SIZE
    receiver_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ring_cols - 1, ring_rows - 1))}
    )

    # ---- Weight + activation (same setup as the qkv_small case) ----
    torch.manual_seed(zlib.crc32(f"factory_{layers_buffered}".encode()))
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

    # ---- Matmul program config (must match the factory's expectations) ----
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
        num_global_cb_receivers=recv_per_bank,
        untilize_out=False,
    )

    # ---- Build GCB via the matmul-aware factory ----
    # gather_in0 matmul uses actual_in0_block_w = weight_K_tiles / ring_size (= k_tiles_per_shard
    # here), not program_config.in0_block_w; the factory matches that. The minimum useful size is
    # num_blocks (= ring_size) * in1_block_size — one full layer's worth of pages.
    weight_K_tiles = K // ttnn.TILE_SIZE
    actual_in0_block_w = weight_K_tiles // ring_size
    in1_block_size = actual_in0_block_w * program_config.per_core_N * tile_bytes
    num_blocks = ring_size
    size = num_blocks * in1_block_size * layers_buffered
    bank_to_receivers = [(b, _bank_receivers_row_major(b, recv_per_bank, ring_cols)) for b in range(num_dram_banks)]
    gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
        device, [program_config], [tt_weight], bank_to_receivers=bank_to_receivers, size=size
    )
    logger.info(
        f"[factory layers_buffered={layers_buffered}] in1_block={in1_block_size} num_blocks={num_blocks} size={size}"
    )

    # ---- Run: prefetcher (async) -> matmul (consumes via gcb) -> stop drains ----
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
    ttnn.experimental.start_dram_core_prefetcher(device)
    ttnn.experimental.queue_dram_core_prefetcher_request(device, [(tt_weight, ring_size)], num_layers=1, global_cb=gcb)
    tt_out = ttnn.linear(
        tt_act,
        tt_weight,
        program_config=program_config,
        memory_config=output_mem_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
        global_cb=gcb,
    )
    ttnn.experimental.stop_dram_core_prefetcher(device)

    out_torch = ttnn.to_torch(tt_out)
    expected = pt_act.float() @ pt_weight.float()
    passing, output_str = comp_pcc(expected, out_torch, 0.999)
    logger.info(f"[factory layers_buffered={layers_buffered}] {output_str}")
    assert passing, f"[factory layers_buffered={layers_buffered}] PCC check failed: {output_str}"


def test_create_global_circular_buffer_for_matmul_1d_rejects_undersized(device):
    """Factory must TT_FATAL when size < num_blocks * in1_block_size (matmul would deadlock)."""
    num_dram_banks = device.dram_grid_size().x
    recv_per_bank = 1
    ring_size = num_dram_banks * recv_per_bank
    dtype = ttnn.bfloat16
    tile_bytes = _bytes_per_tile(dtype)

    K = 8 * ring_size * ttnn.TILE_SIZE
    N = ring_size * ttnn.TILE_SIZE
    pt_weight = torch.zeros(1, 1, K, N)
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

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(num_dram_banks, recv_per_bank),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=ttnn.CoreRangeSet([]),
        num_global_cb_receivers=recv_per_bank,
        untilize_out=False,
    )

    weight_K_tiles = K // ttnn.TILE_SIZE
    actual_in0_block_w = weight_K_tiles // ring_size
    in1_block_size = actual_in0_block_w * program_config.per_core_N * tile_bytes
    num_blocks = ring_size
    min_size = num_blocks * in1_block_size
    bank_to_receivers = [
        (b, _bank_receivers_row_major(b, recv_per_bank, num_dram_banks)) for b in range(num_dram_banks)
    ]
    with pytest.raises(RuntimeError, match="must be at least num_blocks"):
        ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
            device,
            [program_config],
            [tt_weight],
            bank_to_receivers=bank_to_receivers,
            size=min_size - 16,
        )


# ---------------------------------------------------------------------------
# Multi-tensor smoke (discard receiver)
# ---------------------------------------------------------------------------
_MT_NUM_RECV_PER_BANK = 2
_MT_K = 2048
# N must be divisible by `num_dram_banks * _MT_NUM_RECV_PER_BANK * TILE_SIZE` for any
# supported BH bank count. lcm(7, 8) * 2 * 32 = 56 * 64 = 3584, so 3584 works on both
# P100 (7 banks) and P150/P300 (8 banks). The DRAM-core prefetcher auto-derives K-sub
# blocking from the DRISC L1 budget; K=2048 lands on the K-sub-with-M=1 path here.
_MT_N = 3584
_MT_TILE_BYTES = 1088  # bf8_b


def _make_mt_weight(device, seed: int, num_dram_banks: int) -> ttnn.Tensor:
    torch.manual_seed(seed)
    pt_weight = torch.randn(1, 1, _MT_K, _MT_N)
    dram_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
    )
    weight_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [_MT_K, _MT_N // num_dram_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.as_tensor(
        pt_weight, device=device, dtype=ttnn.bfloat8_b, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
    )


@pytest.mark.parametrize("num_tensors,num_layers", [(1, 1), (2, 1), (3, 1), (2, 5), (3, 10)])
def test_dram_core_prefetcher_multi_tensor(device, num_tensors, num_layers):
    # Query the chip's actual DRAM bank count so the test adapts to harvested BHs
    # (P100 has 7, P150/P300 have 8). The receiver grid is laid out as
    # `num_dram_banks` columns × `_MT_NUM_RECV_PER_BANK` rows.
    num_dram_banks = device.dram_grid_size().x
    num_receivers = num_dram_banks * _MT_NUM_RECV_PER_BANK

    # Build `num_tensors` weight tensors, all same shape (so num_blocks matches).
    weights = [_make_mt_weight(device, seed=0xB100 + i, num_dram_banks=num_dram_banks) for i in range(num_tensors)]
    # The op contract requires an addrs tensor as the last input (unused on DRAM-core path).
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

    # The DRAM-core prefetcher auto-derives k_block_w_tiles = ceil(k_tiles / ring_size)
    # and pushes ring_size blocks per (layer, tensor) regardless of k_tiles. GCB and
    # consumer counts must follow ring_size, not k_tiles.
    k_tiles = _MT_K // ttnn.TILE_SIZE
    ring_size = num_receivers
    k_block_w_tiles = (k_tiles + ring_size - 1) // ring_size
    n_tiles_per_recv = (_MT_N // num_dram_banks // _MT_NUM_RECV_PER_BANK) // ttnn.TILE_SIZE
    push_page_size = k_block_w_tiles * n_tiles_per_recv * _MT_TILE_BYTES
    per_recv_bytes_per_tensor = ring_size * push_page_size
    gcb_size = _round_up(per_recv_bytes_per_tensor, push_page_size)

    bank_to_receivers = [
        (b, _bank_receivers_row_major(b, _MT_NUM_RECV_PER_BANK, ring_cols=num_dram_banks))
        for b in range(num_dram_banks)
    ]
    gcb = ttnn.experimental.create_global_circular_buffer_with_dram_senders(device, bank_to_receivers, gcb_size)

    # Per receiver per (layer, tensor): ring_size pushes.
    num_iters_total = num_layers * num_tensors * ring_size
    logger.info(
        f"[multi_tensor] num_tensors={num_tensors} num_layers={num_layers} K={_MT_K} N={_MT_N} "
        f"banks={num_dram_banks} ring={num_receivers} push_page={push_page_size} gcb_size={gcb_size} "
        f"num_iters_total={num_iters_total}"
    )

    # Sender: push all `num_tensors` weights through the prefetcher, num_layers times.
    ttnn.experimental.start_dram_core_prefetcher(device)
    ttnn.experimental.queue_dram_core_prefetcher_request(
        device,
        [(w, ring_size) for w in weights],
        num_layers=num_layers,
        global_cb=gcb,
    )
    # Receiver: discard all pushed data.
    ttnn.experimental.test_dram_prefetcher_consumer(
        device,
        num_iters=num_iters_total,
        page_size_bytes=push_page_size,
        global_cb=gcb,
    )
    ttnn.experimental.stop_dram_core_prefetcher(device)
    ttnn.synchronize_device(device)
    logger.info(f"[multi_tensor] num_tensors={num_tensors} num_layers={num_layers} completed cleanly")
