# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the Tensor prefetcher path on single Blackhole.

Three scopes:

1. ``test_tensor_prefetcher_BH_param`` — parameterized shape coverage with
   PCC vs ``torch.matmul``. Topology: 8 DRAM banks × ``recv_per_bank``
   receivers/bank → ring of 8 × recv_per_bank. Production Llama-3.1-8B on
   single BH uses recv_per_bank=8 (ring=64); smaller rings are exercised for
   QKV/FF cases that fit at lower recv_per_bank.

   Each case builds a fresh DRAM-sender GlobalCircularBuffer, launches the
   Tensor prefetcher via ``ttnn.experimental.start_tensor_prefetcher``,
   runs ``ttnn.linear`` with the same gcb, and PCC-checks against
   ``torch.matmul``. ``stop_tensor_prefetcher`` drains the pipeline after
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

2. ``test_tensor_prefetcher_multi_tensor`` — multi-tensor smoke. The
   DRAM-core kernel's main loop iterates
   ``for layer in num_layers: for t in num_tensors:`` and prior versions
   were documented as leaving the DRISC cores in reset state when
   num_tensors > 1. After the chunked-DMA + receiver-layout fixes this case
   re-verifies the multi-tensor path end-to-end against a discard receiver
   (no matmul; the goal is to confirm both ops complete without
   hang/OOM/PCC failure across multiple ``(num_tensors, num_layers)`` combos).

3. ``test_tensor_prefetcher_trace_replay`` — trace capture/replay. Captures
   a (queue request + consuming linear) pair into a trace, replays it
   ``replay_count`` times, and PCC-checks the matmul on every replay. See the
   comment on the test for the capture-vs-replay semantics it covers.
"""

import math
import time
import zlib
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    round_up as _round_up,
    bytes_per_tile as _bytes_per_tile,
    bank_receivers_strided as _bank_receivers_strided,
    bank_receivers_contiguous as _bank_receivers_contiguous,
    make_recv_contig_weight as _make_recv_contig_weight,
    tensor_prefetcher_session,
    require_tensor_prefetcher,
)


# DRAM bank count is queried at runtime via `device.dram_grid_size().x`:
#   - P150/P300 (unharvested): 8 banks → ring=8*recv_per_bank.
#   - P100 (1 column harvested): 7 banks → ring=7*recv_per_bank.
# The receiver grid is laid out as `num_dram_banks` columns × `recv_per_bank` rows so the
# rectangle is always clean. Test names that suggest a fixed ring size (e.g. "r64") refer
# to the unharvested case; on P100 the same parametrization runs at ring=7*recv_per_bank.


pytestmark = run_for_blackhole("Tensor prefetcher requires Blackhole")


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    require_tensor_prefetcher(device)


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
def test_tensor_prefetcher_BH_param(
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
    output_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, N // ring_size),
        core_grid=receiver_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    # ---- Run: prefetcher (async) -> matmul (consumes via gcb) -> stop drains ----
    ttnn.experimental.start_tensor_prefetcher(device)
    # Fence the prefetcher against the weight write so it never reads stale DRAM.
    ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, 0)
    ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight, ring_size)], global_cb=gcb)
    tt_out = ttnn.linear(
        tt_act,
        tt_weight,
        program_config=program_config,
        memory_config=output_mem_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
        global_cb=gcb,
    )
    ttnn.experimental.stop_tensor_prefetcher(device)

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
# via the matmul-aware factory instead of calling create_global_circular_buffer_for_tensor_prefetcher
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

    # ---- Run: prefetcher (async) -> matmul (consumes via gcb) -> stop drains ----
    output_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, N // ring_size),
        core_grid=receiver_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )
    ttnn.experimental.start_tensor_prefetcher(device)
    ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight, ring_size)], global_cb=gcb)
    tt_out = ttnn.linear(
        tt_act,
        tt_weight,
        program_config=program_config,
        memory_config=output_mem_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
        global_cb=gcb,
    )
    ttnn.experimental.stop_tensor_prefetcher(device)

    out_torch = ttnn.to_torch(tt_out)
    expected = pt_act.float() @ pt_weight.float()
    passing, output_str = comp_pcc(expected, out_torch, 0.999)
    logger.info(f"[factory layers_buffered={layers_buffered}] {output_str}")
    assert passing, f"[factory layers_buffered={layers_buffered}] PCC check failed: {output_str}"


def test_create_global_circular_buffer_for_matmul_1d_rejects_undersized(device, expect_error):
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
    with expect_error(RuntimeError, "must be at least num_blocks"):
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
# P100 (7 banks) and P150/P300 (8 banks). The Tensor prefetcher auto-derives K-sub
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
def test_tensor_prefetcher_multi_tensor(device, num_tensors, num_layers):
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

    # The Tensor prefetcher auto-derives k_block_w_tiles = ceil(k_tiles / ring_size)
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
    gcb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(device, bank_to_receivers, gcb_size)

    # Per receiver per (layer, tensor): ring_size pushes.
    num_iters_total = num_layers * num_tensors * ring_size
    # Sender: push all `num_tensors` weights through the prefetcher, num_layers times.
    # The prefetcher has no num_layers replay count anymore, so flatten the list to
    # num_layers * num_tensors entries (layout dedup keeps the wire compact). With
    # num_layers > 1 this also exercises the multi-page request split.
    ttnn.experimental.start_tensor_prefetcher(device)
    ttnn.experimental.queue_tensor_prefetcher_request(
        device,
        [(w, ring_size) for w in weights] * num_layers,
        global_cb=gcb,
    )
    # Receiver: discard all pushed data.
    ttnn.experimental.test_dram_prefetcher_consumer(
        device,
        num_iters=num_iters_total,
        page_size_bytes=push_page_size,
        global_cb=gcb,
    )
    ttnn.experimental.stop_tensor_prefetcher(device)
    ttnn.synchronize_device(device)


# ---------------------------------------------------------------------------
# Receiver-contiguous DRAM layout smoke
# ---------------------------------------------------------------------------
# Allocates DRAM weight via NdShardSpec with `num_shards = ring_size`
# (over-subscribed: more shards than DRAM banks). The manager auto-detects
# this from buffer_distribution_spec().num_shards() and dispatches to the
# receiver-contiguous compute_tensor_geom path. The GCB topology is built
# so shard index == ring position, matching the round-robin shard-to-bank
# placement. Receiver is the discard consumer (smoke: no PCC check yet).


@pytest.mark.parametrize("num_tensors,num_layers", [(1, 1), (2, 1), (2, 5)])
def test_tensor_prefetcher_recv_contig_smoke(device, num_tensors, num_layers):
    if device.dram_grid_size().x != 8:
        pytest.skip("Receiver-contiguous smoke test expects 8 unharvested DRAM banks")
    num_dram_banks = device.dram_grid_size().x
    num_recv_per_bank = _MT_NUM_RECV_PER_BANK
    num_receivers = num_dram_banks * num_recv_per_bank
    ring_size = num_receivers

    K = _MT_K
    N = _MT_N
    n_per_recv = N // ring_size

    weights = []
    for i in range(num_tensors):
        torch.manual_seed(0xC400 + i)
        pt = torch.randn(1, 1, K, N)
        weights.append(_make_recv_contig_weight(device, pt, num_dram_banks, ring_size, ttnn.bfloat8_b))

    k_tiles = K // ttnn.TILE_SIZE
    k_block_w_tiles = (k_tiles + ring_size - 1) // ring_size
    n_tiles_per_recv = n_per_recv // ttnn.TILE_SIZE
    push_page_size = k_block_w_tiles * n_tiles_per_recv * _MT_TILE_BYTES
    per_recv_bytes_per_tensor = ring_size * push_page_size
    gcb_size = _round_up(per_recv_bytes_per_tensor, push_page_size)

    bank_to_receivers = [
        (b, _bank_receivers_strided(b, num_recv_per_bank, num_dram_banks, ring_cols=num_dram_banks))
        for b in range(num_dram_banks)
    ]
    gcb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(device, bank_to_receivers, gcb_size)

    num_iters_total = num_layers * num_tensors * ring_size

    ttnn.experimental.start_tensor_prefetcher(device)
    ttnn.experimental.queue_tensor_prefetcher_request(
        device,
        [(w, ring_size) for w in weights] * num_layers,
        global_cb=gcb,
    )
    ttnn.experimental.test_dram_prefetcher_consumer(
        device,
        num_iters=num_iters_total,
        page_size_bytes=push_page_size,
        global_cb=gcb,
    )
    ttnn.experimental.stop_tensor_prefetcher(device)
    ttnn.synchronize_device(device)


# ---------------------------------------------------------------------------
# Trace capture / replay
# ---------------------------------------------------------------------------
# `queue_tensor_prefetcher_request` does not go through the command queue: it
# serializes a request into socket pages and a host worker thread fans them out to
# the DRAM sender cores over NOC. Whether a request is *captured* into a trace — held
# back, then re-sent on every `execute_trace` of that trace — or sent immediately is a
# host-side routing decision with two inputs: the request's `capture_into_trace` flag,
# and whether the calling thread's current command queue is mid trace-capture.


class _TraceCase:
    """One gcb-fed matmul, shared by the tests in this section: a persistent activation,
    one DRAM weight, the gcb the prefetcher fills, and `num_weight_values` sets of weight
    values the test can write into that weight with `write_weight`, each with its own
    torch reference.

    Rewriting the weight is how the tests tell a captured request from one that was sent
    immediately. A request reads DRAM when the DRISC processes it, so a request held for
    replay reads the weight as it stands at replay time, while one sent during capture
    read it as it stood then. Overwrite the weight after recording the trace, and the
    matmul output says which happened — no reliance on a hang.
    """

    def __init__(self, device, num_weight_values=1, seed_tag=b"trace_replay", write_cq=0):
        self.device = device

        # ---- Topology (qkv_small shape; adapts to harvested DRAM bank counts) ----
        num_dram_banks = device.dram_grid_size().x
        num_receivers_per_bank = 1
        ring_size = num_dram_banks * num_receivers_per_bank
        ring_cols = num_dram_banks
        ring_rows = num_receivers_per_bank
        dtype = ttnn.bfloat16
        self.ring_size = ring_size
        self.dtype = dtype

        receiver_core_range_set = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ring_cols - 1, ring_rows - 1))}
        )

        M = 32
        k_tiles_per_shard = 8
        n_tiles_per_receiver = 1
        K = k_tiles_per_shard * ring_size * ttnn.TILE_SIZE
        N = ring_size * n_tiles_per_receiver * ttnn.TILE_SIZE

        torch.manual_seed(zlib.crc32(seed_tag))
        pt_weights = [torch.randn(1, 1, K, N) for _ in range(num_weight_values)]
        pt_act = torch.randn(1, 1, M, K)

        # ---- Weight (B): width-sharded in DRAM across the banks ----
        dram_core_range_set = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
        )
        weight_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_core_range_set, [K, N // num_dram_banks], ttnn.ShardOrientation.ROW_MAJOR),
        )

        # ---- Activation (A): width-sharded on receiver cores; persistent across replays ----
        K_per_shard = _round_up(math.ceil(K / ring_size), ttnn.TILE_SIZE)
        act_mem_config = ttnn.create_sharded_memory_config(
            shape=(M, K_per_shard),
            core_grid=receiver_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Host copies of each value set, kept so write_weight can push them into the one
        # device weight later.
        self.host_weights = [ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT) for w in pt_weights]

        # `write_cq` is the queue these host-to-device writes go out on — what
        # wait_for_cq_on_tensor_prefetcher is asked to fence against.
        self.write_cq = write_cq
        with ttnn.command_queue(write_cq):
            self.weight = ttnn.as_tensor(
                pt_weights[0], device=device, dtype=dtype, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
            )
            self.act = ttnn.from_torch(
                pt_act, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=act_mem_config
            )

        # ---- Matmul program config (1D-mcast gather_in0) ----
        out_block_h = M // ttnn.TILE_SIZE
        out_block_w = N // ring_size // ttnn.TILE_SIZE
        out_subblock_w = min(out_block_w, 8)
        while out_subblock_w > 1 and out_block_w % out_subblock_w != 0:
            out_subblock_w -= 1
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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

        # ---- DRAM-sender GlobalCircularBuffer: one layer deep ----
        tile_bytes = _bytes_per_tile(dtype)
        in1_block_size_bytes = k_tiles_per_shard * n_tiles_per_receiver * tile_bytes
        gcb_size = ring_size * in1_block_size_bytes
        bank_to_receivers = [
            (b, _bank_receivers_row_major(b, num_receivers_per_bank, ring_cols)) for b in range(num_dram_banks)
        ]
        self.gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
            device, [self.program_config], [self.weight], bank_to_receivers=bank_to_receivers, size=gcb_size
        )

        self.out_mem_config = ttnn.create_sharded_memory_config(
            shape=(M, N // ring_size),
            core_grid=receiver_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        )

        self.expected = [pt_act.float() @ w.float() for w in pt_weights]

    def write_weight(self, value_index):
        """Overwrite the DRAM weight in place with value set `value_index`, and wait for
        the write to land. In place matters: a captured request holds the addresses it was
        queued with, so this is what changes what a replay of it reads."""
        # A request that already went out may still be reading this weight: the DRISC DMAs
        # from DRAM when it processes the request, and nothing host-side reports that as
        # finished. Sleep first so the overwrite cannot land under an in-flight read — 50 ms
        # against a per-layer DMA measured in microseconds.
        time.sleep(0.05)
        ttnn.copy_host_to_device_tensor(self.host_weights[value_index], self.weight, cq_id=self.write_cq)
        ttnn.synchronize_device(self.device)

    def queue(self, **kwargs):
        """Queue one gcb layer of the weight. Passes no `capture_into_trace` of its own
        unless a test names one, so the binding default applies."""
        ttnn.experimental.queue_tensor_prefetcher_request(
            self.device, [(self.weight, self.ring_size)], global_cb=self.gcb, **kwargs
        )

    def linear(self, **kwargs):
        """Run the matmul over one gcb layer. in1 comes from the gcb, so the output
        reflects whatever the prefetcher put there, not the weight tensor passed here."""
        return ttnn.linear(
            self.act,
            self.weight,
            program_config=self.program_config,
            global_cb=self.gcb,
            memory_config=self.out_mem_config,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype,
            **kwargs,
        )

    def prefetch_and_linear(self, **kwargs):
        """Queue a layer and run the matmul over it, through the paired call production
        uses. Unlike `queue`, this one asks to be captured (`capture_into_trace=True`)."""
        return ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
            self.act,
            self.weight,
            global_cb=self.gcb,
            program_config=self.program_config,
            memory_config=self.out_mem_config,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype,
            **kwargs,
        )

    def check(self, torch_out, value_index):
        """(passing, message) for a readback against act @ weight-value-set value_index."""
        return comp_pcc(self.expected[value_index], torch_out, 0.999)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("replay_count", [1, 3])
def test_tensor_prefetcher_trace_replay(device, replay_count):
    """Capture a (prefetcher-request + linear) pair into a trace and replay it
    `replay_count` times; each replay must refill the GCB and produce the right
    matmul output."""
    case = _TraceCase(device)

    with tensor_prefetcher_session(device):
        # ---- Warmup: compile kernels + balance the GCB before trace capture. The
        # queue here is sent immediately (cq 0 not capturing) and consumed by the matmul. ----
        logger.info("Warmup (compile) run")
        tt_out = case.prefetch_and_linear()
        ttnn.synchronize_device(device)
        passing, msg = case.check(ttnn.to_torch(tt_out), 0)
        assert passing, f"warmup PCC failed: {msg}"

        # ---- Capture: the queue request is captured into the trace (cq 0 mid-capture),
        # NOT sent now. The linear is captured too. ----
        logger.info("Capturing trace")
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        tt_out = case.prefetch_and_linear()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        # ---- Replay: each execute_trace must replay the captured prefetcher request
        # (refilling the GCB) and the matmul. ----
        for i in range(replay_count):
            logger.info(f"Replay {i + 1}/{replay_count}")
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            passing, msg = case.check(ttnn.to_torch(tt_out), 0)
            assert passing, f"replay {i + 1} PCC failed: {msg}"

        ttnn.release_trace(device, trace_id)


# The next two tests both work by overwriting the DRAM weight after recording the trace,
# then replaying once. A request reads DRAM when the DRISC processes it, so the matmul
# output names the moment the request went out: the post-overwrite values mean it was held
# and replayed, the pre-overwrite values mean it went out during capture. They are each
# other's failure signature, and neither can regress into a hang — one layer is queued and
# one matmul runs it either way.
#
# The overwrite assumes a request queued during capture has already read DRAM by the time
# the new values land; `write_weight` sleeps to guarantee it, since there is no host-visible
# request-completion signal to wait on instead.
@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize("cq_id", [0, 1])
def test_tensor_prefetcher_trace_capture_on_queue(device, cq_id):
    """A request asking to be captured must land in the trace being recorded on the queue
    the caller named — including a queue that is not the default 0.

    Which queue is consulted follows ttnn's usual convention: the `queue_id` keyword is
    consumed by the operation wrapper and applied by making that queue the thread's current
    one, so the C++ layer never sees it. A keyword dropped anywhere along that chain leaves
    the prefetch resolving against some other queue, which is not recording, and the request
    goes out during capture instead of being held."""
    case = _TraceCase(device, num_weight_values=2, seed_tag=b"trace_capture_on_queue")

    with tensor_prefetcher_session(device):
        # Warmup on the same queue: trace capture needs the matmul program already cached,
        # and this leaves the gcb empty.
        tt_out = case.prefetch_and_linear(queue_id=cq_id)
        ttnn.synchronize_device(device)
        passing, msg = case.check(ttnn.to_torch(tt_out), 0)
        assert passing, f"warmup PCC failed: {msg}"

        trace_id = ttnn.begin_trace_capture(device, cq_id=cq_id)
        tt_out = case.prefetch_and_linear(queue_id=cq_id)
        ttnn.end_trace_capture(device, trace_id, cq_id=cq_id)

        case.write_weight(1)

        ttnn.execute_trace(device, trace_id, cq_id=cq_id, blocking=True)
        replayed = ttnn.to_torch(tt_out)
        ttnn.release_trace(device, trace_id)

    passing, msg = case.check(replayed, 1)
    assert passing, (
        f"replay on cq {cq_id} did not match the weight written after recording: {msg}. "
        f"Matching the pre-recording weight instead means the request went out during "
        f"capture, i.e. queue_id={cq_id} never reached the prefetch half."
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
def test_tensor_prefetcher_request_not_captured_by_default(device):
    """Without `capture_into_trace`, a request goes out immediately even when the current
    queue is mid trace-capture.

    This is the tri-state the flag exists for: `capture_into_trace=False` means "never
    captured", which is not the same as "no queue is recording". A caller on a thread that
    happens to be inside a capture region can still ask for an immediate send, and that
    request must not be re-sent on every `execute_trace`."""
    case = _TraceCase(device, num_weight_values=2, seed_tag=b"not_captured_by_default")

    with tensor_prefetcher_session(device):
        tt_out = case.prefetch_and_linear()
        ttnn.synchronize_device(device)
        passing, msg = case.check(ttnn.to_torch(tt_out), 0)
        assert passing, f"warmup PCC failed: {msg}"

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        # Mid-capture, at the binding default: goes out now, reading the weight as it
        # currently stands. The linear below is captured, as any op in this region is.
        case.queue()
        tt_out = case.linear()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        case.write_weight(1)

        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        replayed = ttnn.to_torch(tt_out)
        ttnn.release_trace(device, trace_id)

    passing, msg = case.check(replayed, 0)
    assert passing, (
        f"replay did not match the weight as it stood during recording: {msg}. Matching the "
        f"weight written afterwards means the request was captured and re-read DRAM on "
        f"replay, which capture_into_trace=False must not do."
    )


# ---------------------------------------------------------------------------
# Fencing against a command queue
# ---------------------------------------------------------------------------
# `wait_for_cq_on_tensor_prefetcher`'s queue argument reaches the C++ layer only in its
# positional form: a `cq_id=` keyword is consumed by the operation wrapper and applied
# by making that queue the thread's current one. That is why the binding defaults to
# None (resolve whatever queue is current) rather than to 0 — a `uint8_t = 0` default
# would silently fence queue 0 for the keyword and context forms.
@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize("cq_id", [0, 1])
@pytest.mark.parametrize("form", ["positional", "keyword", "context"])
def test_tensor_prefetcher_wait_for_cq_queue_forms(device, cq_id, form):
    """All three spellings must fence the queue the weight was written on, on a
    non-default queue as much as on queue 0."""
    case = _TraceCase(device, seed_tag=b"wait_for_cq_forms", write_cq=cq_id)

    with tensor_prefetcher_session(device):
        if form == "positional":
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id)
        elif form == "keyword":
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=cq_id)
        else:
            with ttnn.command_queue(cq_id):
                ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device)

        tt_out = case.prefetch_and_linear(queue_id=cq_id)
        ttnn.synchronize_device(device)
        out_torch = ttnn.to_torch(tt_out)

    passing, msg = case.check(out_torch, 0)
    assert passing, f"PCC failed after fencing cq {cq_id} in its {form} form: {msg}"


@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
def test_tensor_prefetcher_wait_for_cq_rejects_unknown_queue(device, expect_error):
    """A queue id the device does not have must raise, not fall back to queue 0."""
    _TraceCase(device, seed_tag=b"wait_for_cq_unknown")

    with tensor_prefetcher_session(device):
        # device_params above opens 2 command queues, so 2 is one past the last.
        with expect_error(RuntimeError, "is out of range"):
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, 2)


# ---------------------------------------------------------------------------
# Streaming recv-contig matmul (PCC vs torch.matmul)
# ---------------------------------------------------------------------------
# End-to-end PCC check of the streaming path: the prefetcher delivers each
# receiver's weight blocks in ring-rotated FIFO order (queue streaming=True) and
# the matmul consumes them block-by-block (program_config.stream_in1=True),
# starting before the whole tensor lands. This lets the GCB hold only a small
# live window (`window_blocks`) instead of the full ring_size blocks/receiver.
#
# Both receiver-contiguous topologies stream: ring position P always sits at grid cell
# (P % ring_cols, P // ring_cols), so the matmul's fixed row-major ring order maps each grid
# cell to column block P for either topology — only which DRAM bank feeds P differs (strided:
# P % num_banks; contiguous: P // recv_per_bank). ROUND_ROBIN_1D weight pairs with the strided
# GCB arc, CONTIGUOUS_1D with the contiguous arc; identity rotation makes ring position P lead
# with K-block P in both. `window_blocks is None` exercises streaming at full depth; smaller
# windows exercise the GCB shrink. The validator test covers byte-for-byte delivery parity; here
# we assert the streamed matmul output PCC-matches torch for both shard distributions.
@pytest.mark.parametrize(
    "distribution_strategy",
    [ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D, ttnn.ShardDistributionStrategy.CONTIGUOUS_1D],
    ids=["strided", "contiguous"],
)
@pytest.mark.parametrize(
    "name,k_tiles_per_shard,n_tiles_per_receiver,recv_per_bank,dtype",
    [
        ("qkv_small_bf16", 1, 1, 2, ttnn.bfloat16),  # ring=16
        ("ff1_bf8", 2, 7, 8, ttnn.bfloat8_b),  # FF1 ring=64
        ("ring32_bf8", 2, 6, 4, ttnn.bfloat8_b),  # ring=32
    ],
    ids=["qkv_small_bf16", "ff1_bf8", "ring32_bf8"],
)
@pytest.mark.parametrize("window_blocks", [2, 4, None], ids=["win2", "win4", "winfull"])
def test_tensor_prefetcher_streaming_matmul(
    device, name, k_tiles_per_shard, n_tiles_per_receiver, recv_per_bank, dtype, window_blocks, distribution_strategy
):
    num_dram_banks = device.dram_grid_size().x
    num_receivers_per_bank = recv_per_bank
    ring_size = num_dram_banks * num_receivers_per_bank
    ring_cols = num_dram_banks
    ring_rows = num_receivers_per_bank
    is_contiguous = distribution_strategy == ttnn.ShardDistributionStrategy.CONTIGUOUS_1D

    receiver_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ring_cols - 1, ring_rows - 1))}
    )

    M = 32
    K = k_tiles_per_shard * ring_size * ttnn.TILE_SIZE
    N = ring_size * n_tiles_per_receiver * ttnn.TILE_SIZE

    # ---- Weight (B): receiver-contiguous ND-sharded (num_shards = ring_size); shard distribution
    # (ROUND_ROBIN_1D strided / CONTIGUOUS_1D contiguous) matched by the GCB arc below ----
    torch.manual_seed(zlib.crc32(name.encode()))
    pt_weight = torch.randn(1, 1, K, N)
    tt_weight = _make_recv_contig_weight(
        device,
        pt_weight,
        num_dram_banks=num_dram_banks,
        ring_size=ring_size,
        dtype=dtype,
        distribution_strategy=distribution_strategy,
    )

    # ---- Activation (A): width-sharded across the receiver grid; K split across the ring ----
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

    # ---- Matmul program config: gather_in0 + stream_in1 ----
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
        stream_in1=True,
    )

    # ---- Shallow GCB: window_blocks (or full ring_size) blocks/receiver ----
    tile_bytes = _bytes_per_tile(dtype)
    in1_block_size_bytes = k_tiles_per_shard * n_tiles_per_receiver * tile_bytes
    blocks = window_blocks if window_blocks is not None else ring_size
    gcb_size = blocks * in1_block_size_bytes

    if is_contiguous:
        bank_to_receivers = [
            (b, _bank_receivers_contiguous(b, num_receivers_per_bank, ring_cols=ring_cols))
            for b in range(num_dram_banks)
        ]
    else:
        bank_to_receivers = [
            (b, _bank_receivers_strided(b, num_receivers_per_bank, num_dram_banks, ring_cols=ring_cols))
            for b in range(num_dram_banks)
        ]
    # The recv-contig matmul factory validates the (program_config, weight, bank_to_receivers) triple
    # and, because program_config.stream_in1 is set, relaxes its size floor from a full layer down to a
    # double-buffer window -- so the shallow streaming GCB is accepted here instead of only via the raw
    # create_global_circular_buffer_for_tensor_prefetcher path.
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
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    # program_config.stream_in1 selects streaming, so prefetch_and_linear queues the
    # matching identity-rotation request (natural ring order) and runs the consuming
    # matmul itself -- the test never spells out the rotation table or block_count.
    with tensor_prefetcher_session(device):
        tt_out = ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
            tt_act,
            tt_weight,
            global_cb=gcb,
            program_config=program_config,
            memory_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

    out_torch = ttnn.to_torch(tt_out)
    expected = pt_act.float() @ pt_weight.float()
    pcc_threshold = 0.999 if dtype == ttnn.bfloat16 else 0.99
    passing, output_str = comp_pcc(expected, out_torch, pcc_threshold)
    logger.info(f"[{name} win={window_blocks} {distribution_strategy}] {output_str}")
    assert passing, f"[{name} win={window_blocks} {distribution_strategy}] PCC check failed: {output_str}"


@pytest.mark.parametrize(
    "distribution_strategy",
    [ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D, ttnn.ShardDistributionStrategy.CONTIGUOUS_1D],
    ids=["strided", "contiguous"],
)
def test_tensor_prefetcher_streaming_mcast_in0(device, distribution_strategy):
    """Mcast-in0 consumes receiver-contiguous in1 K-blocks in natural FIFO order."""
    num_dram_banks = device.dram_grid_size().x
    recv_per_bank = 2
    receiver_count = num_dram_banks * recv_per_bank
    ring_cols = num_dram_banks
    ring_rows = recv_per_bank
    receiver_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ring_cols - 1, ring_rows - 1))}
    )

    dtype = ttnn.bfloat16
    M = ttnn.TILE_SIZE
    k_tiles = 2 * receiver_count
    K = k_tiles * ttnn.TILE_SIZE
    N = receiver_count * ttnn.TILE_SIZE
    in0_block_w = 1
    block_count = k_tiles // in0_block_w
    assert block_count != receiver_count

    torch.manual_seed(zlib.crc32(f"streaming_mcast_in0_{distribution_strategy}".encode()))
    pt_weight = torch.randn(1, 1, K, N)
    tt_weight = _make_recv_contig_weight(
        device,
        pt_weight,
        num_dram_banks=num_dram_banks,
        ring_size=receiver_count,
        dtype=dtype,
        distribution_strategy=distribution_strategy,
    )

    pt_act = torch.randn(1, 1, M, K)
    act_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, K // receiver_count),
        core_grid=receiver_cores,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_act = ttnn.from_torch(pt_act, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=act_mem_config)

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(ring_cols, ring_rows),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        gather_in0=False,
        hop_cores=ttnn.CoreRangeSet([]),
        num_global_cb_receivers=recv_per_bank,
        untilize_out=False,
        stream_in1=False,
    )

    if distribution_strategy == ttnn.ShardDistributionStrategy.CONTIGUOUS_1D:
        bank_to_receivers = [
            (b, _bank_receivers_contiguous(b, recv_per_bank, ring_cols=ring_cols)) for b in range(num_dram_banks)
        ]
    else:
        bank_to_receivers = [
            (b, _bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols=ring_cols))
            for b in range(num_dram_banks)
        ]

    page_size_bytes = in0_block_w * program_config.per_core_N * _bytes_per_tile(dtype)
    gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
        device,
        [program_config],
        [tt_weight],
        bank_to_receivers=bank_to_receivers,
        size=2 * page_size_bytes,
    )
    output_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, N // receiver_count),
        core_grid=receiver_cores,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    expected = pt_act.float() @ pt_weight.float()

    # Run twice so the second invocation hits the program cache and exercises
    # override_mcast_in0_program_parameters. The receiver-contiguous weight is
    # ND-sharded, so a naive tensor-backed CB update would TT_FATAL on the
    # GCB-backed in1 CB during the cache-hit path.
    cache_entries_after_first = None
    with tensor_prefetcher_session(device):
        for run in range(2):
            tt_out = ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
                tt_act,
                tt_weight,
                global_cb=gcb,
                program_config=program_config,
                memory_config=output_mem_config,
                compute_kernel_config=compute_kernel_config,
                dtype=dtype,
            )
            if run == 0:
                cache_entries_after_first = device.num_program_cache_entries()

            out_torch = ttnn.to_torch(tt_out)
            passing, output_str = comp_pcc(expected, out_torch, 0.999)
            logger.info(f"[streaming_mcast_in0 {distribution_strategy} run={run}] {output_str}")
            assert passing, f"streaming_mcast_in0 run={run} PCC failed: {output_str}"

    # The second run must reuse the cached program (no new entries), i.e. it took
    # the override path rather than rebuilding.
    assert (
        device.num_program_cache_entries() == cache_entries_after_first
    ), "second mcast_in0 invocation did not hit the program cache"
