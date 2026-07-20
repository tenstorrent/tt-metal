# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Validator-receiver smoke test for the prefetcher pipeline.

Replaces the matmul consumer with a diagnostic kernel that DPRINTs progress
and detects over/under-counts. Drives both the worker-core sender
(`ttnn.dram_prefetcher`) and the DRAM-core sender
(`ttnn.experimental.start_tensor_prefetcher`) against the same validator so any
divergence between the two paths surfaces immediately.

See tt_metal/impl/buffers/prefetcher_matmul_design.md for the contract being validated.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    round_up as _round_up,
    bytes_per_tile as _bytes_per_tile,
    ring_grid_cols as _ring_grid_cols,
    bank_receivers_strided as _bank_receivers_strided,
    bank_receivers_contiguous as _bank_receivers_contiguous,
    make_recv_contig_weight as _make_recv_contig_weight,
    tensor_prefetcher_session,
)


pytestmark = run_for_blackhole("Tensor prefetcher requires Blackhole")


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    """Skip unless programmable DRAM cores are available on this device."""
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip("programmable DRAM cores unavailable (need Blackhole and firmware >= 19.12.0.0)")


_GCB_DEPTH_PAGES = 4  # small ring so the validator stresses reserve_back/wait_front handshakes


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


def _setup_weight_and_gcb_dram_sender(device, K, N, dtype, recv_per_bank, num_layers, row_offset=0, dual_senders=False):
    """Build a DRAM-sender GCB sized to the prefetcher's per-receiver-per-block push.

    `row_offset` shifts the receiver grid down; the multi-GCB switching test uses
    it to place two GCBs on disjoint receiver rows for the same prefetcher.

    `dual_senders` requests the two-sender-per-bank GCB variant. With recv_per_bank=1
    each bank has a single receiver that cannot split, so every bank falls back to its
    primary sender — the mapping is identical to a single-sender GCB and stays
    K-row-major compatible.

    Returns (tt_weight, addrs, gcb, num_iters_total, push_page_size_bytes, ring_size).
    """
    tile_bytes = _bytes_per_tile(dtype)
    num_dram_banks = device.dram_grid_size().x
    ring_size = num_dram_banks * recv_per_bank
    ring_cols = _ring_grid_cols(num_dram_banks, ring_size)

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
        (b, _bank_receivers_row_major(b, recv_per_bank, ring_cols=ring_cols, row_offset=row_offset))
        for b in range(num_dram_banks)
    ]
    gcb_size = _GCB_DEPTH_PAGES * push_page_size
    gcb = ttnn.experimental.create_global_circular_buffer_with_dram_senders(
        device, bank_to_receivers, gcb_size, dual_senders_per_bank=dual_senders
    )

    num_iters_total = num_layers * ring_size
    return tt_weight, addrs, gcb, num_iters_total, push_page_size, ring_size


def _setup_weight_and_gcb_worker_sender(device, K, N, dtype, recv_per_bank, num_layers):
    """Build a worker-sender GCB and addrs tensor for ttnn.dram_prefetcher.

    Returns (tt_weight, tt_addrs, gcb, num_iters_total, push_page_size_bytes, ring_size).
    """
    tile_bytes = _bytes_per_tile(dtype)
    num_dram_banks = device.dram_grid_size().x
    ring_size = num_dram_banks * recv_per_bank
    ring_cols = _ring_grid_cols(num_dram_banks, ring_size)
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

    with tensor_prefetcher_session(device):
        # Flattened: repeat the tensor num_layers times (the prefetcher no longer has a
        # num_layers replay count). Dedup keeps this to 1 layout + num_layers entries.
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight, ring_size)] * num_layers, global_cb=gcb)
        ttnn.experimental.test_dram_prefetcher_validator(
            device,
            tt_weight,
            num_layers=num_layers,
            print_stride=max(1, ring_size // 4),
            global_cb=gcb,
        )


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
    tt_weight_a, addrs_a, gcb_a, _, _, ring_size = _setup_weight_and_gcb_dram_sender(
        device, K, N, dtype, recv_per_bank, num_layers=1
    )
    # GCB B: receiver grid on rows [recv_per_bank, 2*recv_per_bank). Disjoint from A.
    tt_weight_b, addrs_b, gcb_b, _, _, _ = _setup_weight_and_gcb_dram_sender(
        device, K, N, dtype, recv_per_bank, num_layers=1, row_offset=recv_per_bank
    )

    with tensor_prefetcher_session(device):
        # Interleave A → B → A so the third request hits A's persistent ring state
        # established by the first A request and skipped over by the B request.
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight_a, ring_size)], global_cb=gcb_a)
        ttnn.experimental.test_dram_prefetcher_validator(
            device, tt_weight_a, num_layers=1, print_stride=max(1, ring_size // 4), global_cb=gcb_a
        )
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight_b, ring_size)], global_cb=gcb_b)
        ttnn.experimental.test_dram_prefetcher_validator(
            device, tt_weight_b, num_layers=1, print_stride=max(1, ring_size // 4), global_cb=gcb_b
        )
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight_a, ring_size)], global_cb=gcb_a)
        ttnn.experimental.test_dram_prefetcher_validator(
            device, tt_weight_a, num_layers=1, print_stride=max(1, ring_size // 4), global_cb=gcb_a
        )


def test_validator_dram_sender_mixed_num_receivers(device):
    """Two DRAM-sender GCBs with DIFFERENT num_receivers share one prefetcher.
    Catches regressions if num_receivers ever drifts back to a kernel compile-time
    arg or a per-prefetcher constant: a single Queue against a GCB whose receiver
    count doesn't match the kernel's expectation would walk the wrong NOC table.
    """
    K, dtype = 448, ttnn.bfloat16
    # GCB B uses recv_per_bank=2, so n_per_bank (= N / num_dram_banks / TILE_SIZE,
    # in tiles) must be divisible by 2. Derive N from the device bank count so this
    # holds regardless of how many DRAM banks the part has (P150=8, P100=7) —
    # n_per_bank is then 2 * n_tiles_per_recv (even). A hardcoded N (e.g. 1792)
    # gives n_per_bank=7 on an 8-bank part, which 2 doesn't divide.
    num_dram_banks = device.dram_grid_size().x
    n_tiles_per_recv = 4
    N = num_dram_banks * 2 * n_tiles_per_recv * ttnn.TILE_SIZE
    # GCB A: recv_per_bank=1 (rows [0, 1)).
    tt_weight_a, addrs_a, gcb_a, _, _, ring_a = _setup_weight_and_gcb_dram_sender(
        device, K, N, dtype, recv_per_bank=1, num_layers=1
    )
    # GCB B: recv_per_bank=2 (rows [1, 3)) — different receiver count.
    tt_weight_b, addrs_b, gcb_b, _, _, ring_b = _setup_weight_and_gcb_dram_sender(
        device, K, N, dtype, recv_per_bank=2, num_layers=1, row_offset=1
    )

    with tensor_prefetcher_session(device):
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight_a, ring_a)], global_cb=gcb_a)
        ttnn.experimental.test_dram_prefetcher_validator(
            device, tt_weight_a, num_layers=1, print_stride=max(1, ring_a // 4), global_cb=gcb_a
        )
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight_b, ring_b)], global_cb=gcb_b)
        ttnn.experimental.test_dram_prefetcher_validator(
            device, tt_weight_b, num_layers=1, print_stride=max(1, ring_b // 4), global_cb=gcb_b
        )
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight_a, ring_a)], global_cb=gcb_a)
        ttnn.experimental.test_dram_prefetcher_validator(
            device, tt_weight_a, num_layers=1, print_stride=max(1, ring_a // 4), global_cb=gcb_a
        )


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


def _setup_weight_and_gcb_recv_contig(
    device,
    K,
    N,
    dtype,
    recv_per_bank,
    num_layers,
    dual_senders=False,
    distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
):
    """Build a DRAM-sender GCB + NdShardSpec-allocated weight for the
    receiver-contiguous DRAM-core path. num_shards = ring_size > num_dram_banks
    triggers the manager's recv-contig detection.

    The weight's shard distribution and the GCB sender->receiver pairing must
    agree: round-robin shards pair with strided arcs, shard-contiguous (CONTIGUOUS_1D)
    shards pair with contiguous arcs. Either pairing yields shard index == ring
    position, so the validator's per-receiver column slice is unchanged."""
    is_shard_contiguous = distribution_strategy == ttnn.ShardDistributionStrategy.CONTIGUOUS_1D
    tile_bytes = _bytes_per_tile(dtype)
    num_dram_banks = device.dram_grid_size().x
    ring_size = num_dram_banks * recv_per_bank
    ring_cols = _ring_grid_cols(num_dram_banks, ring_size)

    K_padded = _round_up(K, ring_size * ttnn.TILE_SIZE)
    k_tiles = K_padded // ttnn.TILE_SIZE
    k_block_w_tiles = k_tiles // ring_size
    n_per_recv_tiles = N // ring_size // ttnn.TILE_SIZE
    push_page_size = k_block_w_tiles * n_per_recv_tiles * tile_bytes

    torch.manual_seed(0xC0FFEE)
    pt_weight = torch.zeros(1, 1, K_padded, N)
    pt_weight[:, :, :K, :] = torch.randn(1, 1, K, N)

    tt_weight = _make_recv_contig_weight(
        device, pt_weight, num_dram_banks, ring_size, dtype, distribution_strategy=distribution_strategy
    )

    if is_shard_contiguous:
        bank_to_receivers = [
            (b, _bank_receivers_contiguous(b, recv_per_bank, ring_cols=ring_cols)) for b in range(num_dram_banks)
        ]
    else:
        bank_to_receivers = [
            (b, _bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols=ring_cols))
            for b in range(num_dram_banks)
        ]
    gcb_size = _GCB_DEPTH_PAGES * push_page_size
    gcb = ttnn.experimental.create_global_circular_buffer_with_dram_senders(
        device, bank_to_receivers, gcb_size, dual_senders_per_bank=dual_senders
    )
    num_iters_total = num_layers * ring_size
    return tt_weight, gcb, num_iters_total, push_page_size, ring_size


@pytest.mark.parametrize("streaming", [False, True], ids=["batched", "streaming"])
@pytest.mark.parametrize(
    "K,N,dtype,recv_per_bank,num_layers,dual_senders",
    [
        (2048, 3584, ttnn.bfloat8_b, 2, 1, False),  # ring=16, num_shards=16 > num_banks=8
        (4096, 14336, ttnn.bfloat8_b, 8, 1, False),  # FF1 ring=64
        (2048, 7168, ttnn.bfloat8_b, 4, 1, False),  # ring=32, single-sender nr=4 (discriminator)
        # Dual-sender: each bank's receivers split ceil/floor across two DRISC cores.
        (2048, 3584, ttnn.bfloat8_b, 2, 1, True),  # ring=16, even split 1/1 per bank
        (4096, 14336, ttnn.bfloat8_b, 8, 1, True),  # FF1 ring=64, even split 4/4 per bank
        (2304, 5376, ttnn.bfloat8_b, 3, 1, True),  # ring=24, odd split 2/1 per bank (ceil/floor)
    ],
    ids=["multi_ksub", "ff1", "single_r4", "multi_ksub_dual", "ff1_dual", "odd_dual"],
)
def test_validator_dram_sender_recv_contig(device, K, N, dtype, recv_per_bank, num_layers, dual_senders, streaming):
    # streaming=True exercises the ring-rotated delivery order: the prefetcher reads each
    # receiver's slab circularly from its ring index g_r, and the validator expects FIFO
    # position p to hold physical block (ring_pos + p) mod ring_size. Same byte content as
    # batched, reordered per receiver — a byte mismatch localizes a g_r/rotation bug here,
    # before the matmul.
    tt_weight, gcb, num_iters_total, push_page_size, ring_size = _setup_weight_and_gcb_recv_contig(
        device, K, N, dtype, recv_per_bank, num_layers, dual_senders=dual_senders
    )
    # Non-identity rotation (cyclic shift by 1): receiver at ring position g leads at block
    # (g + 1) % ring_size, not g. Unlike identity (rotation[g] == g), this only validates if the
    # prefetcher actually slices by the supplied rotation values rather than the bare ring index.
    # Empty == batched. The same rotation is handed to the validator so it expects the matching order.
    rotation = [(g + 1) % ring_size for g in range(ring_size)] if streaming else []
    with tensor_prefetcher_session(device):
        ttnn.experimental.queue_tensor_prefetcher_request(
            device, [(tt_weight, ring_size, rotation)] * num_layers, global_cb=gcb
        )
        ttnn.experimental.test_dram_prefetcher_validator(
            device,
            tt_weight,
            num_layers=num_layers,
            print_stride=max(1, ring_size // 4),
            global_cb=gcb,
            streaming=streaming,
            rotation=rotation,
        )


@pytest.mark.parametrize("K,N,dtype,num_layers", [(448, 1792, ttnn.bfloat16, 2)])
def test_validator_dram_sender_dual_single_receiver_bank(device, K, N, dtype, num_layers):
    """dual_senders_per_bank=True with a single receiver per bank (recv_per_bank=1).

    A single receiver cannot split across two senders, so every bank falls back to its
    primary sender and the mapping is identical to a single-sender GCB — previously this
    was rejected with a TT_FATAL. num_blocks == num_dram_banks keeps it on the K-row-major
    path (which the validator addresses uniformly), so this validates the fallback end to
    end. Same shape as test_validator_dram_sender[bench_default], only dual_senders flipped.
    """
    tt_weight, addrs, gcb, num_iters_total, push_page_size, ring_size = _setup_weight_and_gcb_dram_sender(
        device, K, N, dtype, recv_per_bank=1, num_layers=num_layers, dual_senders=True
    )
    with tensor_prefetcher_session(device):
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight, ring_size)] * num_layers, global_cb=gcb)
        ttnn.experimental.test_dram_prefetcher_validator(
            device,
            tt_weight,
            num_layers=num_layers,
            print_stride=max(1, ring_size // 4),
            global_cb=gcb,
        )


@pytest.mark.parametrize("streaming", [False, True], ids=["batched", "streaming"])
@pytest.mark.parametrize(
    "K,N,dtype,recv_per_bank,num_layers,dual_senders",
    [
        (2048, 3584, ttnn.bfloat8_b, 2, 1, False),  # ring=16, contiguous arcs
        (4096, 14336, ttnn.bfloat8_b, 8, 1, False),  # FF1 ring=64
        (2048, 7168, ttnn.bfloat8_b, 4, 1, False),  # ring=32, single-sender nr=4
        (4096, 14336, ttnn.bfloat8_b, 8, 1, True),  # FF1 ring=64, dual-sender split 4/4
    ],
    ids=["multi_ksub", "ff1", "single_r4", "ff1_dual"],
)
def test_validator_dram_sender_recv_contig_shard_contiguous(
    device, K, N, dtype, recv_per_bank, num_layers, dual_senders, streaming
):
    """Shard-contiguous (CONTIGUOUS_1D) recv-contig: adjacent shards share a bank, so each bank
    feeds a contiguous ring arc. Exercises the shard-contiguous shard->bank placement (host BDS)
    and the distribution-aware TensorAccessor the validator reads the source through.

    streaming=True additionally exercises the CONTIGUOUS_1D streaming path: the host must slice the
    rotation table by the *contiguous* global receiver position (bank*receivers_per_bank + slab),
    not the strided one. A strided-only slice (the pre-generalization bug) mismatches even for
    identity rotation, so the non-identity cyclic rotation below is a strict check."""
    tt_weight, gcb, num_iters_total, push_page_size, ring_size = _setup_weight_and_gcb_recv_contig(
        device,
        K,
        N,
        dtype,
        recv_per_bank,
        num_layers,
        dual_senders=dual_senders,
        distribution_strategy=ttnn.ShardDistributionStrategy.CONTIGUOUS_1D,
    )
    # Cyclic shift by 1 (empty == batched); same rotation handed to the validator so it expects the
    # matching contiguous-order delivery.
    rotation = [(g + 1) % ring_size for g in range(ring_size)] if streaming else []
    with tensor_prefetcher_session(device):
        ttnn.experimental.queue_tensor_prefetcher_request(
            device, [(tt_weight, ring_size, rotation)] * num_layers, global_cb=gcb
        )
        ttnn.experimental.test_dram_prefetcher_validator(
            device,
            tt_weight,
            num_layers=num_layers,
            print_stride=max(1, ring_size // 4),
            global_cb=gcb,
            streaming=streaming,
            rotation=rotation,
        )


def test_validator_dram_sender_recv_contig_page_size_switch(device):
    """A single recv-contig GCB can serve tensors with different page sizes.

    The first tensor leaves the FIFO pointer at an address that is not aligned to
    the second tensor's larger page size. Both sender and receiver must credit the
    skipped padding during resize, otherwise the second validator waits forever.
    """
    dtype = ttnn.bfloat8_b
    recv_per_bank = 2
    tile_bytes = _bytes_per_tile(dtype)
    num_dram_banks = device.dram_grid_size().x
    ring_size = num_dram_banks * recv_per_bank
    ring_cols = _ring_grid_cols(num_dram_banks, ring_size)

    bank_to_receivers = [
        (b, _bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols=ring_cols))
        for b in range(num_dram_banks)
    ]

    def make_weight(K, n_per_recv_tiles, seed):
        K_padded = _round_up(K, ring_size * ttnn.TILE_SIZE)
        N = ring_size * n_per_recv_tiles * ttnn.TILE_SIZE
        k_block_w_tiles = (K_padded // ttnn.TILE_SIZE) // ring_size
        push_page_size = k_block_w_tiles * n_per_recv_tiles * tile_bytes

        torch.manual_seed(seed)
        pt_weight = torch.zeros(1, 1, K_padded, N)
        pt_weight[:, :, :K, :] = torch.randn(1, 1, K, N)

        tt_weight = _make_recv_contig_weight(device, pt_weight, num_dram_banks, ring_size, dtype)
        return tt_weight, push_page_size

    # page_a = 1 * 3 * 1088 = 3264 B; page_b = 2 * 5 * 1088 = 10880 B.
    # After one ring of A pages, the FIFO pointer advances by 52224 B, which is
    # not page_b-aligned and forces resize padding before B.
    weight_a, page_a = make_weight(K=512, n_per_recv_tiles=3, seed=0xA11CE)
    weight_b, page_b = make_weight(K=1024, n_per_recv_tiles=5, seed=0xB0B)
    assert (ring_size * page_a) % page_b != 0

    gcb_size = ring_size * max(page_a, page_b)
    gcb = ttnn.experimental.create_global_circular_buffer_with_dram_senders(device, bank_to_receivers, gcb_size)

    with tensor_prefetcher_session(device):
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight_a, ring_size)], global_cb=gcb)
        ttnn.experimental.test_dram_prefetcher_validator(
            device,
            weight_a,
            num_layers=1,
            print_stride=max(1, ring_size // 4),
            global_cb=gcb,
        )
        ttnn.synchronize_device(device)

        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight_b, ring_size)], global_cb=gcb)
        ttnn.experimental.test_dram_prefetcher_validator(
            device,
            weight_b,
            num_layers=1,
            print_stride=max(1, ring_size // 4),
            global_cb=gcb,
        )
