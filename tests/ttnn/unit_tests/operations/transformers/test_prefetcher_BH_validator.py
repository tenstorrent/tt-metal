# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Validator-receiver smoke test for the prefetcher pipeline.

Replaces the matmul consumer with a diagnostic kernel that DPRINTs progress
and detects over/under-counts. Drives both the worker-core sender
(`ttnn.dram_prefetcher`) and the DRAM-core sender
(`ttnn.experimental.start_dram_core_prefetcher`) against the same validator so any
divergence between the two paths surfaces immediately.

See tt_metal/impl/buffers/prefetcher_matmul_design.md for the contract being validated.
"""

import os
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.prefetcher_common import round_up as _round_up


pytestmark = [
    run_for_blackhole("DRAM-core prefetcher requires Blackhole"),
    pytest.mark.skipif(
        os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0") != "1",
        reason="TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set",
    ),
]


_GCB_DEPTH_PAGES = 4  # small ring so the validator stresses reserve_back/wait_front handshakes


_TILE_BYTES_BF16 = 2048  # 32*32*2
_TILE_BYTES_BF8 = 1088  # 32*32 (data) + 64 (exponent header)


def _bank_receivers_row_major(bank_idx: int, recv_per_bank: int, ring_cols: int, row_offset: int = 0):
    """CoreRangeSet for bank `bank_idx`'s receivers, laid out row-major on a
    ring_cols-wide grid. Matches the bench's gather_in0 receiver layout."""
    cores = []
    for k in range(recv_per_bank):
        ring_pos = bank_idx * recv_per_bank + k
        col = ring_pos % ring_cols
        row = ring_pos // ring_cols + row_offset
        cores.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
    return ttnn.CoreRangeSet(cores)


def _setup_weight_and_gcb_dram_sender_at_row(device, K, N, dtype, recv_per_bank, num_layers, row_offset):
    """Variant of _setup_weight_and_gcb_dram_sender that places the receiver grid
    starting at row `row_offset`. Used by the multi-GCB switching test so two
    GCBs can share the prefetcher without overlapping receivers."""
    tile_bytes = _TILE_BYTES_BF16 if dtype == ttnn.bfloat16 else _TILE_BYTES_BF8
    num_dram_banks = device.dram_grid_size().x
    ring_size = num_dram_banks * recv_per_bank
    ring_cols = max(c for c in range(min(num_dram_banks, ring_size), 0, -1) if ring_size % c == 0)

    K_padded = _round_up(K, ring_size * ttnn.TILE_SIZE)
    k_tiles = K_padded // ttnn.TILE_SIZE
    k_block_w_tiles = k_tiles // ring_size
    n_per_recv_tiles = (N // num_dram_banks // recv_per_bank) // ttnn.TILE_SIZE
    push_page_size = k_block_w_tiles * n_per_recv_tiles * tile_bytes

    pt_weight = torch.zeros(1, 1, K_padded, N)
    pt_weight[:, :, :K, :] = torch.randn(1, 1, K, N)

    dram_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
    )
    weight_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [K_padded, N // num_dram_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_weight = ttnn.as_tensor(
        pt_weight, device=device, dtype=dtype, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
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

    bank_to_receivers = [
        (b, _bank_receivers_row_major(b, recv_per_bank, ring_cols=ring_cols, row_offset=row_offset))
        for b in range(num_dram_banks)
    ]
    gcb_size = _GCB_DEPTH_PAGES * push_page_size
    gcb = ttnn.experimental.create_global_circular_buffer_with_dram_senders(device, bank_to_receivers, gcb_size)
    num_iters_total = num_layers * ring_size
    return tt_weight, addrs, gcb, num_iters_total, push_page_size, ring_size


def _setup_weight_and_gcb_dram_sender(device, K, N, dtype, recv_per_bank, num_layers):
    """Build a DRAM-sender GCB sized to the prefetcher's per-receiver-per-block push.

    Returns (tt_weight, addrs, gcb, num_iters_total, push_page_size_bytes, ring_size).
    """
    tile_bytes = _TILE_BYTES_BF16 if dtype == ttnn.bfloat16 else _TILE_BYTES_BF8
    num_dram_banks = device.dram_grid_size().x
    ring_size = num_dram_banks * recv_per_bank
    ring_cols = max(c for c in range(min(num_dram_banks, ring_size), 0, -1) if ring_size % c == 0)

    K_padded = _round_up(K, ring_size * ttnn.TILE_SIZE)
    k_tiles = K_padded // ttnn.TILE_SIZE
    k_block_w_tiles = k_tiles // ring_size
    n_per_recv_tiles = (N // num_dram_banks // recv_per_bank) // ttnn.TILE_SIZE
    push_page_size = k_block_w_tiles * n_per_recv_tiles * tile_bytes

    torch.manual_seed(0xC0FFEE)
    pt_weight = torch.zeros(1, 1, K_padded, N)
    pt_weight[:, :, :K, :] = torch.randn(1, 1, K, N)

    dram_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
    )
    weight_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [K_padded, N // num_dram_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_weight = ttnn.as_tensor(
        pt_weight, device=device, dtype=dtype, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
    )

    # addrs tensor (unused on DRAM-core path but required by op contract).
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

    bank_to_receivers = [
        (b, _bank_receivers_row_major(b, recv_per_bank, ring_cols=ring_cols, row_offset=0))
        for b in range(num_dram_banks)
    ]
    gcb_size = _GCB_DEPTH_PAGES * push_page_size
    gcb = ttnn.experimental.create_global_circular_buffer_with_dram_senders(device, bank_to_receivers, gcb_size)

    num_iters_total = num_layers * ring_size
    logger.info(
        f"[validator-dram] K={K} K_padded={K_padded} N={N} banks={num_dram_banks} ring={ring_size} "
        f"k_block_w_tiles={k_block_w_tiles} push_page={push_page_size} gcb_size={gcb_size} "
        f"num_layers={num_layers} num_iters_total={num_iters_total}"
    )
    return tt_weight, addrs, gcb, num_iters_total, push_page_size, ring_size


def _setup_weight_and_gcb_worker_sender(device, K, N, dtype, recv_per_bank, num_layers):
    """Build a worker-sender GCB and addrs tensor for ttnn.dram_prefetcher.

    Returns (tt_weight, tt_addrs, gcb, num_iters_total, push_page_size_bytes, ring_size).
    """
    tile_bytes = _TILE_BYTES_BF16 if dtype == ttnn.bfloat16 else _TILE_BYTES_BF8
    num_dram_banks = device.dram_grid_size().x
    ring_size = num_dram_banks * recv_per_bank
    ring_cols = max(c for c in range(min(num_dram_banks, ring_size), 0, -1) if ring_size % c == 0)
    ring_rows = ring_size // ring_cols

    K_padded = _round_up(K, ring_size * ttnn.TILE_SIZE)
    k_tiles = K_padded // ttnn.TILE_SIZE
    k_block_w_tiles = k_tiles // ring_size
    n_per_recv_tiles = (N // num_dram_banks // recv_per_bank) // ttnn.TILE_SIZE
    push_page_size = k_block_w_tiles * n_per_recv_tiles * tile_bytes

    torch.manual_seed(0xC0FFEE)
    pt_weight = torch.zeros(1, 1, K_padded, N)
    pt_weight[:, :, :K, :] = torch.randn(1, 1, K, N)

    dram_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
    )
    weight_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [K_padded, N // num_dram_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_weight = ttnn.as_tensor(
        pt_weight, device=device, dtype=dtype, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
    )

    # Worker-sender layout: senders on row `ring_rows`, receivers on rows 0..ring_rows-1.
    sender_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, ring_rows), ttnn.CoreCoord(num_dram_banks - 1, ring_rows))}
    )
    sender_receiver_mapping = [
        (ttnn.CoreCoord(b, ring_rows), _bank_receivers_row_major(b, recv_per_bank, ring_cols=ring_cols))
        for b in range(num_dram_banks)
    ]
    # Worker prefetcher validates max_tensor_size <= gcb.size() — a full layer's per-recv bytes.
    gcb_size = ring_size * push_page_size
    gcb = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, gcb_size)

    # Worker-sender expects num_layers address rows per sender.
    addr_row = torch.tensor([tt_weight.buffer_address()] * num_layers, dtype=torch.int32).reshape(1, num_layers)
    tensor_addrs = addr_row.repeat(num_dram_banks, 1)
    tt_addrs = ttnn.as_tensor(
        tensor_addrs,
        device=device,
        dtype=ttnn.uint32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(sender_core_range_set, [1, num_layers], ttnn.ShardOrientation.ROW_MAJOR),
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    num_iters_total = num_layers * ring_size
    logger.info(
        f"[validator-worker] K={K} K_padded={K_padded} N={N} banks={num_dram_banks} ring={ring_size} "
        f"k_block_w_tiles={k_block_w_tiles} push_page={push_page_size} gcb_size={gcb_size} "
        f"num_layers={num_layers} num_iters_total={num_iters_total}"
    )
    return tt_weight, tt_addrs, gcb, num_iters_total, push_page_size, ring_size


@pytest.mark.parametrize(
    "K,N,dtype,recv_per_bank,num_layers",
    [
        (448, 1792, ttnn.bfloat16, 1, 2),  # default bench shape, fast path (full block fits)
        (2048, 3584, ttnn.bfloat8_b, 2, 1),  # multi-tensor shape, K-sub path (M=1)
        (4096, 14336, ttnn.bfloat8_b, 8, 1),  # FF1 production shape, K-sub+M path
    ],
    ids=["bench_default", "multi_ksub", "ff1_ksub_mchunk"],
)
def test_validator_dram_sender(device, K, N, dtype, recv_per_bank, num_layers):
    tt_weight, addrs, gcb, num_iters_total, push_page_size, ring_size = _setup_weight_and_gcb_dram_sender(
        device, K, N, dtype, recv_per_bank, num_layers
    )

    ttnn.experimental.start_dram_core_prefetcher(device)
    ttnn.experimental.queue_dram_core_prefetcher_request(device, [tt_weight], num_layers=num_layers, global_cb=gcb)
    ttnn.experimental.test_dram_prefetcher_validator(
        device,
        tt_weight,
        num_layers=num_layers,
        print_stride=max(1, ring_size // 4),
        global_cb=gcb,
    )
    ttnn.experimental.stop_dram_core_prefetcher(device)
    ttnn.synchronize_device(device)
    logger.info(f"[validator-dram] K={K} N={N} recv_per_bank={recv_per_bank} num_layers={num_layers} OK")


@pytest.mark.parametrize("K,N,dtype,recv_per_bank", [(448, 1792, ttnn.bfloat16, 1)])
def test_validator_dram_sender_multi_gcb_switching(device, K, N, dtype, recv_per_bank):
    """Two DRAM-sender GCBs share one prefetcher; interleave Queue(A) → Queue(B) → Queue(A)
    and verify both GCBs deliver the right data on every request. Catches regressions
    where the per-GCB ring state (fifo_wr_ptr, etc.) gets clobbered when the kernel
    switches GCBs mid-stream.

    Receivers are stacked vertically so the two GCBs use disjoint cores; the prefetcher
    runs on the same DRAM sender cores for both.
    """
    # GCB A: receiver grid on rows [0, recv_per_bank).
    tt_weight_a, addrs_a, gcb_a, _, _, ring_size = _setup_weight_and_gcb_dram_sender_at_row(
        device, K, N, dtype, recv_per_bank, num_layers=1, row_offset=0
    )
    # GCB B: receiver grid on rows [recv_per_bank, 2*recv_per_bank). Disjoint from A.
    tt_weight_b, addrs_b, gcb_b, _, _, _ = _setup_weight_and_gcb_dram_sender_at_row(
        device, K, N, dtype, recv_per_bank, num_layers=1, row_offset=recv_per_bank
    )

    logger.info(
        f"[validator-dram-multi-gcb] K={K} N={N} recv_per_bank={recv_per_bank} ring={ring_size} "
        f"A=row[0,{recv_per_bank}) B=row[{recv_per_bank},{2 * recv_per_bank})"
    )

    ttnn.experimental.start_dram_core_prefetcher(device)

    # Interleave A → B → A so the third request hits A's persistent ring state
    # established by the first A request and skipped over by the B request.
    ttnn.experimental.queue_dram_core_prefetcher_request(device, [tt_weight_a], num_layers=1, global_cb=gcb_a)
    ttnn.experimental.test_dram_prefetcher_validator(
        device, tt_weight_a, num_layers=1, print_stride=max(1, ring_size // 4), global_cb=gcb_a
    )
    ttnn.experimental.queue_dram_core_prefetcher_request(device, [tt_weight_b], num_layers=1, global_cb=gcb_b)
    ttnn.experimental.test_dram_prefetcher_validator(
        device, tt_weight_b, num_layers=1, print_stride=max(1, ring_size // 4), global_cb=gcb_b
    )
    ttnn.experimental.queue_dram_core_prefetcher_request(device, [tt_weight_a], num_layers=1, global_cb=gcb_a)
    ttnn.experimental.test_dram_prefetcher_validator(
        device, tt_weight_a, num_layers=1, print_stride=max(1, ring_size // 4), global_cb=gcb_a
    )

    ttnn.experimental.stop_dram_core_prefetcher(device)
    ttnn.synchronize_device(device)
    logger.info("[validator-dram-multi-gcb] A→B→A interleaved OK")


def test_validator_dram_sender_mixed_num_receivers(device):
    """Two DRAM-sender GCBs with DIFFERENT num_receivers share one prefetcher.
    Catches regressions if num_receivers ever drifts back to a kernel compile-time
    arg or a per-prefetcher constant: a single Queue against a GCB whose receiver
    count doesn't match the kernel's expectation would walk the wrong NOC table.
    """
    K, N, dtype = 448, 1792, ttnn.bfloat16
    # GCB A: recv_per_bank=1 (rows [0, 1)).
    tt_weight_a, addrs_a, gcb_a, _, _, ring_a = _setup_weight_and_gcb_dram_sender_at_row(
        device, K, N, dtype, recv_per_bank=1, num_layers=1, row_offset=0
    )
    # GCB B: recv_per_bank=2 (rows [1, 3)) — different receiver count.
    tt_weight_b, addrs_b, gcb_b, _, _, ring_b = _setup_weight_and_gcb_dram_sender_at_row(
        device, K, N, dtype, recv_per_bank=2, num_layers=1, row_offset=1
    )

    logger.info(f"[validator-dram-mixed] A: recv=1 ring={ring_a} rows[0,1) | " f"B: recv=2 ring={ring_b} rows[1,3)")

    ttnn.experimental.start_dram_core_prefetcher(device)

    ttnn.experimental.queue_dram_core_prefetcher_request(device, [tt_weight_a], num_layers=1, global_cb=gcb_a)
    ttnn.experimental.test_dram_prefetcher_validator(
        device, tt_weight_a, num_layers=1, print_stride=max(1, ring_a // 4), global_cb=gcb_a
    )
    ttnn.experimental.queue_dram_core_prefetcher_request(device, [tt_weight_b], num_layers=1, global_cb=gcb_b)
    ttnn.experimental.test_dram_prefetcher_validator(
        device, tt_weight_b, num_layers=1, print_stride=max(1, ring_b // 4), global_cb=gcb_b
    )
    ttnn.experimental.queue_dram_core_prefetcher_request(device, [tt_weight_a], num_layers=1, global_cb=gcb_a)
    ttnn.experimental.test_dram_prefetcher_validator(
        device, tt_weight_a, num_layers=1, print_stride=max(1, ring_a // 4), global_cb=gcb_a
    )

    ttnn.experimental.stop_dram_core_prefetcher(device)
    ttnn.synchronize_device(device)
    logger.info("[validator-dram-mixed] mixed-num_receivers interleave OK")


@pytest.mark.parametrize(
    "K,N,dtype,recv_per_bank,num_layers",
    [
        (448, 1792, ttnn.bfloat16, 1, 2),  # baseline
        (2048, 3584, ttnn.bfloat8_b, 2, 1),
        (4096, 14336, ttnn.bfloat8_b, 8, 1),  # FF1
    ],
    ids=["bench_default", "multi", "ff1"],
)
def test_validator_worker_sender(device, K, N, dtype, recv_per_bank, num_layers):
    if device.dram_grid_size().x != 8:
        # Worker prefetcher's reader_dram.cpp uses a fixed per-row TRID layout that hits
        # known issues on harvested cards (P100, 7 banks). Tracked separately from the
        # DRAM-core refactor.
        pytest.skip("Worker-sender validator requires an unharvested 8-bank device")

    tt_weight, tt_addrs, gcb, num_iters_total, push_page_size, ring_size = _setup_weight_and_gcb_worker_sender(
        device, K, N, dtype, recv_per_bank, num_layers
    )

    # Sub-device isolation: ttnn.dram_prefetcher's writer_l1 ends in remote_cb_sender_barrier
    # waiting for receiver acks, so the prefetcher program never completes until receivers run.
    # Without sub-devices, fast dispatch serializes on the prefetcher and the validator never
    # launches (deadlock). Stall-group set to the worker sub-device after enqueueing the
    # prefetcher lets the validator dispatch concurrently. Mirrors prefetcher_common.run_op.
    prefetcher_sub_device = ttnn.SubDevice([gcb.sender_cores()])
    worker_sub_device = ttnn.SubDevice([gcb.receiver_cores()])
    sub_device_manager = device.create_sub_device_manager([prefetcher_sub_device, worker_sub_device], 0)
    device.load_sub_device_manager(sub_device_manager)
    worker_sub_device_id = ttnn.SubDeviceId(1)
    try:
        ttnn.dram_prefetcher([tt_weight, tt_addrs], num_layers=num_layers, global_cb=gcb)
        device.set_sub_device_stall_group([worker_sub_device_id])
        ttnn.experimental.test_dram_prefetcher_validator(
            device,
            tt_weight,
            num_layers=num_layers,
            print_stride=max(1, ring_size // 4),
            global_cb=gcb,
        )
        ttnn.synchronize_device(device)
        device.reset_sub_device_stall_group()
    finally:
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(sub_device_manager)
    logger.info(f"[validator-worker] K={K} N={N} recv_per_bank={recv_per_bank} num_layers={num_layers} OK")
