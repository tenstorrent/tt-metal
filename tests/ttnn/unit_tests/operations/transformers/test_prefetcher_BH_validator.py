# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Validator-receiver smoke test for the prefetcher pipeline.

Replaces the matmul consumer with a diagnostic kernel that DPRINTs progress
and detects over/under-counts. Drives both the worker-core sender
(`ttnn.dram_prefetcher`, into a GlobalCircularBuffer) and the DRAM-core sender
(`ttnn.experimental.start_tensor_prefetcher`, into PrefetcherPipes) against
validators that re-derive the expected bytes the same way, so any divergence
between the two paths surfaces immediately.

See tt_metal/impl/buffers/prefetcher_matmul_design.md for the contract being validated.
"""

import gc
import time

import pytest
import torch
import ttnn

from tt_py_test_utils_common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    round_up as _round_up,
    bytes_per_tile as _bytes_per_tile,
    ring_grid_cols as _ring_grid_cols,
    bank_receivers_strided as _bank_receivers_strided,
    bank_receivers_contiguous as _bank_receivers_contiguous,
    make_recv_contig_weight as _make_recv_contig_weight,
    make_krow_major_weight as _make_krow_major_weight,
    make_tensor_prefetcher_pipes as _make_tensor_prefetcher_pipes,
    tensor_prefetcher_session,
    require_tensor_prefetcher,
)

pytestmark = run_for_blackhole("Tensor prefetcher requires Blackhole")


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    require_tensor_prefetcher(device)


_RING_DEPTH_PAGES = 4  # small ring so the validator stresses the credit handshakes


def _receiver_rows(num_dram_banks: int, recv_per_bank: int) -> int:
    """Rows the receiver grid for `recv_per_bank` receivers per bank occupies. A second delivery
    target on the same prefetcher must be offset by this much to keep the two receiver sets
    disjoint."""
    ring_size = num_dram_banks * recv_per_bank
    return ring_size // _ring_grid_cols(num_dram_banks, ring_size)


def _krow_major_weight_and_bank_map(device, K, N, dtype, recv_per_bank, row_offset=0):
    """Legacy K-row-major weight + bank->receiver pairing.

    One WIDTH_SHARDED shard per bank, K padded to a whole number of ring blocks; bank b's
    receivers sit at ring positions [b*recv_per_bank, (b+1)*recv_per_bank) on a row-major grid,
    in the order the sender slices the bank's shard. `row_offset` shifts the receiver grid down.

    Returns (tt_weight, bank_to_receivers, push_page_size, ring_size)."""
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

    tt_weight = _make_krow_major_weight(device, pt_weight, num_dram_banks, dtype)
    bank_to_receivers = [
        (b, _bank_receivers_contiguous(b, recv_per_bank, ring_cols=ring_cols, row_offset=row_offset))
        for b in range(num_dram_banks)
    ]
    return tt_weight, bank_to_receivers, push_page_size, ring_size


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
        (ttnn.CoreCoord(b, ring_rows), _bank_receivers_contiguous(b, recv_per_bank, ring_cols=ring_cols))
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
def test_validator_pipe_krow_major(device, K, N, dtype, recv_per_bank, num_layers):
    """Legacy K-row-major weight -> PrefetcherPipes, validated byte-for-byte against the source tensor.

    One sender per bank reads K-rows of the bank's whole shard and slices each across that bank's
    receivers, so a receiver's pipe sees the same per-block
    pages a receiver-contiguous weight would give it. ff1 is the case whose K-row does not fit a stage
    half, so the sender also chunks N across the receivers (M > 1). With one receiver per bank the two
    layouts name the same bytes, and bench_default goes through the receiver-contiguous loop instead.
    """
    tt_weight, pipes, bank_to_receivers, _push_page_size, ring_size = _setup_weight_and_pipes_krow_major(
        device, K, N, dtype, recv_per_bank
    )
    with tensor_prefetcher_session(device):
        _queue_and_validate_pipes(device, tt_weight, pipes, bank_to_receivers, ring_size, num_layers=num_layers)


def test_validator_pipe_krow_major_rejects_dual_senders(device, expect_error):
    """One read of a K-row-major bank shard feeds every receiver on the bank, so a bank split across two
    senders cannot deliver it."""
    tt_weight, bank_to_receivers, push_page_size, ring_size = _krow_major_weight_and_bank_map(
        device, 2048, 3584, ttnn.bfloat8_b, recv_per_bank=2
    )
    _space, pipes = _make_tensor_prefetcher_pipes(device, bank_to_receivers, _RING_DEPTH_PAGES * push_page_size)
    with tensor_prefetcher_session(device):
        with expect_error(RuntimeError, "requires exactly one primary sender per DRAM bank"):
            ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight, ring_size)], prefetcher_pipes=pipes)


@pytest.mark.parametrize(
    "weights",
    [(8, 1, 5), (0, 8, 5), (0, 1, 8)],
    ids=["free_sender", "noc1_sender", "ordinary"],
)
def test_tensor_prefetcher_rejects_invalid_mpfe_weight(device, weights, expect_error):
    with expect_error(RuntimeError, r"MPFE weights must be in \[0, 7\]"):
        ttnn.experimental.start_tensor_prefetcher(
            device,
            free_sender_mpfe_weight=weights[0],
            noc1_sender_mpfe_weight=weights[1],
            ordinary_mpfe_weight=weights[2],
        )


def test_validator_pipe_static_mpfe_restart(device):
    """Run two complete 0/1/5 lifetimes; shutdown must leave the next start clean."""
    tt_weight, pipes, bank_to_receivers, _push_page_size, ring_size = _setup_weight_and_pipes_krow_major(
        device, 448, 1792, ttnn.bfloat16, recv_per_bank=1
    )

    for _ in range(2):
        ttnn.experimental.start_tensor_prefetcher(
            device,
            free_sender_mpfe_weight=0,
            noc1_sender_mpfe_weight=1,
            ordinary_mpfe_weight=5,
        )
        try:
            _queue_and_validate_pipes(device, tt_weight, pipes, bank_to_receivers, ring_size)
        finally:
            ttnn.experimental.stop_tensor_prefetcher(device)
            ttnn.synchronize_device(device)


def test_validator_pipe_dynamic_mpfe(device):
    """Validate one dynamic 5/5/5 idle -> 0/1/5 active request."""
    tt_weight, pipes, bank_to_receivers, _push_page_size, ring_size = _setup_weight_and_pipes_krow_major(
        device, 448, 1792, ttnn.bfloat16, recv_per_bank=1
    )

    ttnn.experimental.start_tensor_prefetcher(
        device,
        free_sender_mpfe_weight=0,
        noc1_sender_mpfe_weight=1,
        ordinary_mpfe_weight=5,
        dynamic_mpfe_weighting=True,
    )
    try:
        _queue_and_validate_pipes(device, tt_weight, pipes, bank_to_receivers, ring_size)
    finally:
        ttnn.experimental.stop_tensor_prefetcher(device)
        ttnn.synchronize_device(device)


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


def _recv_contig_weight_and_bank_map(
    device,
    K,
    N,
    dtype,
    recv_per_bank,
    distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    row_offset=0,
):
    """Receiver-contiguous weight + bank->receiver pairing. num_shards = ring_size > num_dram_banks
    triggers the manager's recv-contig detection.

    The weight's shard distribution and the sender->receiver pairing must agree: round-robin shards
    pair with strided arcs, shard-contiguous (CONTIGUOUS_1D) shards pair with contiguous arcs.
    Either pairing yields shard index == ring position, so the validator's per-receiver column slice
    is unchanged.

    ``row_offset`` shifts the receiver grid down, which is how the multi-target tests place two
    delivery targets on disjoint receivers for one prefetcher.

    Returns (tt_weight, bank_to_receivers, push_page_size, ring_size)."""
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

    if distribution_strategy == ttnn.ShardDistributionStrategy.CONTIGUOUS_1D:
        bank_to_receivers = [
            (b, _bank_receivers_contiguous(b, recv_per_bank, ring_cols=ring_cols, row_offset=row_offset))
            for b in range(num_dram_banks)
        ]
    else:
        bank_to_receivers = [
            (b, _bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols=ring_cols, row_offset=row_offset))
            for b in range(num_dram_banks)
        ]
    return tt_weight, bank_to_receivers, push_page_size, ring_size


def _setup_weight_and_pipes_recv_contig(
    device,
    K,
    N,
    dtype,
    recv_per_bank,
    dual_senders=False,
    distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    row_offset=0,
    num_entries=_RING_DEPTH_PAGES,
    entry_divisor=1,
    return_space=False,
):
    """A receiver-contiguous weight and the pipes that deliver it (see _recv_contig_weight_and_bank_map
    for the bank->receiver pairing). The space ring is sized as `num_entries` blocks of this tensor's
    block size divided by `entry_divisor`; a later request may push any other aligned block size the ring holds, so a
    divisor above 1 is how a caller gets a ring the pushed block size does not divide.

    Returns (tt_weight, pipes, bank_to_receivers, push_page_size, ring_size[, pipe_space]); the
    validator takes bank_to_receivers to know which slab each receiver expects."""
    tt_weight, bank_to_receivers, push_page_size, ring_size = _recv_contig_weight_and_bank_map(
        device, K, N, dtype, recv_per_bank, distribution_strategy=distribution_strategy, row_offset=row_offset
    )
    pipe_space, pipes = _make_tensor_prefetcher_pipes(
        device, bank_to_receivers, (push_page_size // entry_divisor) * num_entries, single_sender=not dual_senders
    )
    result = (tt_weight, pipes, bank_to_receivers, push_page_size, ring_size)
    return (*result, pipe_space) if return_space else result


def _setup_weight_and_pipes_krow_major(device, K, N, dtype, recv_per_bank, row_offset=0, num_entries=_RING_DEPTH_PAGES):
    """A K-row-major weight and the pipes that deliver it (see _krow_major_weight_and_bank_map for the
    bank->receiver pairing), one pipe per bank, since one read of a bank's shard feeds all of that
    bank's receivers. The ring holds `num_entries` blocks.

    Returns (tt_weight, pipes, bank_to_receivers, push_page_size, ring_size)."""
    tt_weight, bank_to_receivers, push_page_size, ring_size = _krow_major_weight_and_bank_map(
        device, K, N, dtype, recv_per_bank, row_offset=row_offset
    )
    _space, pipes = _make_tensor_prefetcher_pipes(
        device, bank_to_receivers, push_page_size * num_entries, single_sender=True
    )
    return tt_weight, pipes, bank_to_receivers, push_page_size, ring_size


@pytest.mark.parametrize("streaming", [False, True], ids=["batched", "streaming"])
@pytest.mark.parametrize(
    "K,N,dtype,recv_per_bank,num_layers,dual_senders",
    [
        (2048, 3584, ttnn.bfloat8_b, 2, 1, False),  # ring=16, num_shards=16 > num_banks=8
        (4096, 14336, ttnn.bfloat8_b, 8, 1, False),  # FF1 ring=64
        (2048, 7168, ttnn.bfloat8_b, 4, 1, False),  # ring=32, single-sender nr=4 (discriminator)
        (2048, 3584, ttnn.bfloat8_b, 2, 2, False),  # two layers through one ring
        (448, 1792, ttnn.bfloat16, 1, 1, False),  # ring=8, bfloat16 tiles, one receiver per bank
        # Dual-sender: each bank's receivers split ceil/floor across two DRISC cores.
        (2048, 3584, ttnn.bfloat8_b, 2, 1, True),  # ring=16, even split 1/1 per bank
        (4096, 14336, ttnn.bfloat8_b, 8, 1, True),  # FF1 ring=64, even split 4/4 per bank
        (2304, 5376, ttnn.bfloat8_b, 3, 1, True),  # ring=24, odd split 2/1 per bank (ceil/floor)
    ],
    ids=["multi_ksub", "ff1", "single_r4", "multi_layer", "bf16_r1", "multi_ksub_dual", "ff1_dual", "odd_dual"],
)
def test_validator_pipe_recv_contig(device, K, N, dtype, recv_per_bank, num_layers, dual_senders, streaming):
    """Prefetcher -> PrefetcherPipe delivery, validated byte-for-byte against the source tensor.

    In the streaming case the rotation changes which DRAM block feeds a receiver at each push step,
    not how much any receiver is credited, so a pipe sender's write cursors stay in lockstep across receivers exactly as they
    do for batched delivery. A byte mismatch here means the rotation reached the wrong receiver."""
    tt_weight, pipes, bank_to_receivers, _push_page_size, ring_size = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank, dual_senders=dual_senders
    )
    # Non-identity rotation (cyclic shift by 1), so this only passes if the prefetcher slices by the
    # supplied rotation rather than the bare ring index. Empty == batched.
    rotation = [(g + 1) % ring_size for g in range(ring_size)] if streaming else []
    with tensor_prefetcher_session(device):
        _queue_and_validate_pipes(
            device, tt_weight, pipes, bank_to_receivers, ring_size, num_layers=num_layers, rotation=rotation
        )


@pytest.mark.parametrize("streaming", [False, True], ids=["batched", "streaming"])
@pytest.mark.parametrize(
    "K,N,dtype,recv_per_bank,dual_senders",
    [
        (2048, 3584, ttnn.bfloat8_b, 2, False),  # ring=16, contiguous arcs
        (4096, 14336, ttnn.bfloat8_b, 8, False),  # FF1 ring=64
        (2048, 7168, ttnn.bfloat8_b, 4, False),  # ring=32, single-sender nr=4
        (4096, 14336, ttnn.bfloat8_b, 8, True),  # FF1 ring=64, dual-sender split 4/4
    ],
    ids=["multi_ksub", "ff1", "single_r4", "ff1_dual"],
)
def test_validator_pipe_recv_contig_shard_contiguous(device, K, N, dtype, recv_per_bank, dual_senders, streaming):
    """CONTIGUOUS_1D shards paired with contiguous ring arcs, rather than round-robin/strided: adjacent
    shards share a bank, so each bank feeds a contiguous arc of the ring. Exercises the shard-contiguous
    shard->bank placement (host BDS) and the distribution-aware TensorAccessor the validator reads the
    source through.

    streaming=True additionally exercises the CONTIGUOUS_1D streaming path: the host must slice the
    rotation table by the *contiguous* global receiver position (bank*receivers_per_bank + slab), not
    the strided one. A strided-only slice mismatches even for identity rotation, so the non-identity
    cyclic rotation below is a strict check."""
    tt_weight, pipes, bank_to_receivers, _push_page_size, ring_size = _setup_weight_and_pipes_recv_contig(
        device,
        K,
        N,
        dtype,
        recv_per_bank,
        dual_senders=dual_senders,
        distribution_strategy=ttnn.ShardDistributionStrategy.CONTIGUOUS_1D,
    )
    # Cyclic shift by 1 (empty == batched); the same rotation is handed to the validator so it expects
    # the matching contiguous-order delivery.
    rotation = [(g + 1) % ring_size for g in range(ring_size)] if streaming else []
    with tensor_prefetcher_session(device):
        _queue_and_validate_pipes(device, tt_weight, pipes, bank_to_receivers, ring_size, rotation=rotation)


@pytest.mark.parametrize("K,N,dtype,recv_per_bank", [(2048, 3584, ttnn.bfloat8_b, 2)])
def test_validator_pipe_cursor_persists_across_requests(device, K, N, dtype, recv_per_bank):
    """Two separate queue calls of one layer each, drained by one two-layer validator.

    A PrefetcherPipe sender keeps each receiver's write cursor beside that receiver's credit
    counters, in the durable config page. The second request must therefore resume where the first
    stopped, which this only detects because the validator expects block b of layer 2 in the slot
    following layer 1's."""
    tt_weight, pipes, bank_to_receivers, _push_page_size, ring_size = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank
    )
    with tensor_prefetcher_session(device):
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight, ring_size)], prefetcher_pipes=pipes)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight, ring_size)], prefetcher_pipes=pipes)
        ttnn.experimental.test_tensor_prefetcher_pipe_validator(
            device,
            tt_weight,
            num_layers=2,
            print_stride=max(1, ring_size // 4),
            prefetcher_pipes=pipes,
            bank_to_receivers=bank_to_receivers,
        )


@pytest.mark.parametrize("K,N,dtype,recv_per_bank", [(2048, 3584, ttnn.bfloat8_b, 2)])
def test_pipe_consumer_cache_uses_pipe_identity(device, K, N, dtype, recv_per_bank):
    """The same pipe hits the consumer cache, while a same-address recarve misses it.

    Dropping and recarving from one space deliberately reproduces every piece of geometry and
    every allocation address. The replacement still has fresh persistent sender state, so a
    compiled program whose pipe slots are sticky must not be reused for it.
    """
    tt_weight, pipes, _bank_to_receivers, page_size, ring_size, pipe_space = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank, return_space=True
    )
    # CoreRangeSet is bound reference-internal from a pipe, so make an owning copy before dropping
    # the handles. Otherwise this reconstruction data itself keeps every original pipe alive.
    bank_to_receivers = [
        (pipe.sender_core().x, ttnn.CoreRangeSet(set(pipe.receiver_cores().ranges()))) for pipe in pipes
    ]
    original_addresses = [(pipe.buffer_address(), pipe.config_address()) for pipe in pipes]

    device.enable_program_cache()
    with tensor_prefetcher_session(device):
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight, ring_size)], prefetcher_pipes=pipes)
        before_first = device.num_program_cache_entries()
        ttnn.experimental.test_tensor_prefetcher_pipe_consumer(
            device, num_iters=ring_size, page_size_bytes=page_size, prefetcher_pipes=pipes
        )
        ttnn.synchronize_device(device)
        after_first = device.num_program_cache_entries()
        assert after_first == before_first + 1

        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight, ring_size)], prefetcher_pipes=pipes)
        before_second = device.num_program_cache_entries()
        ttnn.experimental.test_tensor_prefetcher_pipe_consumer(
            device, num_iters=ring_size, page_size_bytes=page_size, prefetcher_pipes=pipes
        )
        ttnn.synchronize_device(device)
        assert device.num_program_cache_entries() == before_second

    del pipes
    gc.collect()
    replacement = ttnn.experimental.create_prefetcher_pipes_for_tensor_prefetcher(
        pipe_space, bank_to_receivers, support_multi_receiver_shards=True
    )
    assert [(pipe.buffer_address(), pipe.config_address()) for pipe in replacement] == original_addresses

    with tensor_prefetcher_session(device):
        ttnn.experimental.queue_tensor_prefetcher_request(
            device, [(tt_weight, ring_size)], prefetcher_pipes=replacement
        )
        before_replacement = device.num_program_cache_entries()
        ttnn.experimental.test_tensor_prefetcher_pipe_consumer(
            device, num_iters=ring_size, page_size_bytes=page_size, prefetcher_pipes=replacement
        )
        ttnn.synchronize_device(device)
        assert device.num_program_cache_entries() == before_replacement + 1


def test_pipe_consumer_trace_replay(device):
    """A cached pipe consumer can be captured and replayed while its sender runs persistently."""
    num_dram_banks = device.dram_grid_size().x
    K = 448
    N = num_dram_banks * 2 * 2 * ttnn.TILE_SIZE
    tt_weight, pipes, _bank_to_receivers, page_size, ring_size = _setup_weight_and_pipes_recv_contig(
        device, K, N, ttnn.bfloat8_b, recv_per_bank=2
    )
    trace_repeats = 2

    with tensor_prefetcher_session(device):
        ttnn.experimental.queue_tensor_prefetcher_request(
            device, [(tt_weight, ring_size)] * (trace_repeats + 1), prefetcher_pipes=pipes
        )
        ttnn.experimental.test_tensor_prefetcher_pipe_consumer(
            device, num_iters=ring_size, page_size_bytes=page_size, prefetcher_pipes=pipes
        )
        ttnn.synchronize_device(device)

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(trace_repeats):
            ttnn.experimental.test_tensor_prefetcher_pipe_consumer(
                device, num_iters=ring_size, page_size_bytes=page_size, prefetcher_pipes=pipes
            )
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        ttnn.release_trace(device, trace_id)


def test_stop_drains_every_target_not_just_the_last(device):
    """Stop waits for the receivers of every target a sender has loaded, not only the last one.

    One prefetcher serves two pipe sets on the same DRAM senders, with disjoint receivers. The second
    set's receivers are drained to completion, so its acks are all in, while the first set's receivers
    hold every pop back for a fixed wall-clock delay. The first set's ring holds the whole tensor, so
    the sender finishes both requests without waiting for either. Stop must then wait out the delay for
    the first set's acks: if it returned after the last target's alone, those acks would land in DRISC L1
    the next program is free to reuse.
    """
    num_dram_banks = device.dram_grid_size().x
    held_recv_per_bank = 2
    K = 448
    N = num_dram_banks * held_recv_per_bank * 2 * ttnn.TILE_SIZE
    held_ring_blocks = num_dram_banks * held_recv_per_bank
    tt_weight_held, held_pipes, _bank_to_receivers, page_size, ring_size = _setup_weight_and_pipes_recv_contig(
        device, K, N, ttnn.bfloat8_b, held_recv_per_bank, num_entries=held_ring_blocks
    )
    tt_weight_last, last_pipes, last_bank_to_receivers, _last_page, last_ring_size = _setup_weight_and_pipes_krow_major(
        device,
        K,
        num_dram_banks * 2 * ttnn.TILE_SIZE,
        ttnn.bfloat16,
        recv_per_bank=1,
        row_offset=_receiver_rows(num_dram_banks, held_recv_per_bank),
    )
    # About 0.75 s at Blackhole's 1.35 GHz AICLK, and longer at any lower clock. Stop without the held
    # set's acks returns in milliseconds, so half a second separates the two outcomes.
    hold_cycles = 1_000_000_000
    min_stop_wait_s = 0.5

    ttnn.experimental.start_tensor_prefetcher(device)
    held_from = None
    try:
        ttnn.experimental.queue_tensor_prefetcher_request(
            device, [(tt_weight_held, ring_size)], prefetcher_pipes=held_pipes
        )
        _queue_and_validate_pipes(device, tt_weight_last, last_pipes, last_bank_to_receivers, last_ring_size)
        ttnn.synchronize_device(device)

        held_from = time.monotonic()
        ttnn.experimental.test_tensor_prefetcher_pipe_consumer(
            device, num_iters=ring_size, page_size_bytes=page_size, prefetcher_pipes=held_pipes, hold_cycles=hold_cycles
        )
    finally:
        ttnn.experimental.stop_tensor_prefetcher(device)
        stop_returned = time.monotonic()
        ttnn.synchronize_device(device)
    stop_waited_s = stop_returned - held_from
    assert stop_waited_s >= min_stop_wait_s, (
        f"stop_tensor_prefetcher returned {stop_waited_s:.3f} s after the held pipe consumer started holding its "
        f"acks, before the {min_stop_wait_s} s hold could have ended: it did not wait for those receivers"
    )


def test_stop_skips_targets_freed_mid_session(device):
    """Stop does not drain a target whose memory was freed while the prefetcher was running.

    A pipe set is sent a tensor nobody consumes, so its credits stay outstanding, and then it and its
    space are destroyed before stop. Its DRISC config pages go back to the arena, where anything may
    reuse them, so stop must not read them as counters; draining them would spin forever on acks that
    never come. A second pipe set queued afterwards on the same senders is still drained normally.
    """
    num_dram_banks = device.dram_grid_size().x
    freed_recv_per_bank = 2
    K = 448
    N = num_dram_banks * freed_recv_per_bank * 2 * ttnn.TILE_SIZE
    (
        tt_weight_freed,
        freed_pipes,
        _bank_to_receivers,
        _page_size,
        ring_size,
        freed_space,
    ) = _setup_weight_and_pipes_recv_contig(
        device,
        K,
        N,
        ttnn.bfloat8_b,
        freed_recv_per_bank,
        num_entries=num_dram_banks * freed_recv_per_bank,
        return_space=True,
    )
    tt_weight_kept, kept_pipes, kept_bank_to_receivers, _kept_page, kept_ring_size = _setup_weight_and_pipes_krow_major(
        device,
        K,
        num_dram_banks * 2 * ttnn.TILE_SIZE,
        ttnn.bfloat16,
        recv_per_bank=1,
        row_offset=_receiver_rows(num_dram_banks, freed_recv_per_bank),
    )

    with tensor_prefetcher_session(device):
        ttnn.experimental.queue_tensor_prefetcher_request(
            device, [(tt_weight_freed, ring_size)], prefetcher_pipes=freed_pipes
        )
        # The kept set's data arrives only after the freed set's request is fully sent, so this also
        # waits out those sends.
        _queue_and_validate_pipes(device, tt_weight_kept, kept_pipes, kept_bank_to_receivers, kept_ring_size)
        ttnn.synchronize_device(device)
        del freed_pipes, freed_space
        gc.collect()


@pytest.mark.parametrize("K,N,dtype,recv_per_bank", [(2048, 3584, ttnn.bfloat8_b, 2)])
@pytest.mark.parametrize("num_entries", [5, 7], ids=["ring_2.5_blocks", "ring_3.5_blocks"])
def test_validator_pipe_block_size_not_dividing_ring(device, K, N, dtype, recv_per_bank, num_entries):
    """A pushed block size need not divide the ring: the remainder becomes a trailing gap.

    Pipes are created at half this tensor's block size and an odd depth, so the ring is N-and-a-half
    blocks. The DRAM sender stops at the page-aligned usable limit and credits the leftover half
    block as padding at each wrap, so a lap credits the whole ring and its cursor comes back to the
    ring base on a block boundary. Two layers so the ring wraps at least once with a gap in play.
    """
    tt_weight, gapped_pipes, bank_to_receivers, _push_page_size, ring_size = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank, num_entries=num_entries, entry_divisor=2
    )
    with tensor_prefetcher_session(device):
        _queue_and_validate_pipes(device, tt_weight, gapped_pipes, bank_to_receivers, ring_size, num_layers=2)


@pytest.mark.parametrize("K,N,dtype,recv_per_bank", [(2048, 3584, ttnn.bfloat8_b, 2)])
@pytest.mark.parametrize("entry_divisor,num_entries", [(1, 1), (2, 3)], ids=["ring_1_block", "ring_1.5_blocks"])
def test_validator_pipe_ring_holds_one_block(device, K, N, dtype, recv_per_bank, entry_divisor, num_entries):
    """A ring with room for a single block: every push is a whole lap of that receiver's ring.

    The transport only needs somewhere to put one block -- it is a consumer that wants a second one
    for lookahead -- so this is a supported (if unpipelined) geometry. The 1.5-block variant puts a
    half-block trailing gap after that one block, so each push credits payload plus gap and the
    cursor wraps back to zero every time.
    """
    tt_weight, shallow_pipes, bank_to_receivers, _push_page_size, ring_size = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank, num_entries=num_entries, entry_divisor=entry_divisor
    )
    with tensor_prefetcher_session(device):
        _queue_and_validate_pipes(device, tt_weight, shallow_pipes, bank_to_receivers, ring_size, num_layers=1)


def _queue_and_validate_pipes(device, tt_weight, pipes, bank_to_receivers, ring_size, num_layers=1, rotation=()):
    """One PrefetcherPipe request plus the validator that drains it. A non-empty ``rotation`` streams the
    request and has the validator expect that rotation's delivery order."""
    rotation = list(rotation)
    ttnn.experimental.queue_tensor_prefetcher_request(
        device, [(tt_weight, ring_size, rotation)] * num_layers, prefetcher_pipes=pipes
    )
    ttnn.experimental.test_tensor_prefetcher_pipe_validator(
        device,
        tt_weight,
        num_layers=num_layers,
        print_stride=max(1, ring_size // 4),
        prefetcher_pipes=pipes,
        bank_to_receivers=bank_to_receivers,
        streaming=bool(rotation),
        rotation=rotation,
    )


@pytest.mark.parametrize("K,N,dtype", [(448, 1792, ttnn.bfloat8_b)])
def test_validator_pipe_dual_single_receiver_bank(device, K, N, dtype):
    """Dual senders requested (support_multi_receiver_shards=False) with one receiver per bank.

    Two things at once. A single receiver cannot be split across two senders, so every bank falls
    back to its primary sender: one pipe per bank, each with slab base 0. And one shard per bank is
    where the two DRAM layouts coincide -- ttnn hands such an NdShardSpec back as a legacy
    WIDTH_SHARDED ShardSpec, so the manager reads them as receiver-contiguous, which names the same
    bytes as K-row-major there.
    """
    tt_weight, pipes, bank_to_receivers, _push_page_size, ring_size = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank=1, dual_senders=True
    )
    assert len({p.sender_core().x for p in pipes}) == len(pipes), "a single-receiver bank must fall back to one sender"
    with tensor_prefetcher_session(device):
        _queue_and_validate_pipes(device, tt_weight, pipes, bank_to_receivers, ring_size)


@pytest.mark.parametrize("K,N,dtype,recv_per_bank", [(2048, 3584, ttnn.bfloat8_b, 2)])
def test_validator_pipe_multi_set_switching(device, K, N, dtype, recv_per_bank):
    """Two pipe sets share one prefetcher; interleave A -> B -> A.

    Every pipe owns its own DRISC config page, and a sender's write cursors live in that page
    beside its credit counters, so switching sets mid-stream has to leave each set's cursors where
    they were. The third request is the one that catches a set whose state the B request disturbed.
    Receivers are stacked vertically so the sets are disjoint; both run on the same DRAM senders.
    """
    num_dram_banks = device.dram_grid_size().x
    tt_weight_a, pipes_a, bank_to_receivers_a, _push_a, ring_size = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank
    )
    tt_weight_b, pipes_b, bank_to_receivers_b, _push_b, _ring_b = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank, row_offset=_receiver_rows(num_dram_banks, recv_per_bank)
    )
    with tensor_prefetcher_session(device):
        _queue_and_validate_pipes(device, tt_weight_a, pipes_a, bank_to_receivers_a, ring_size)
        _queue_and_validate_pipes(device, tt_weight_b, pipes_b, bank_to_receivers_b, ring_size)
        _queue_and_validate_pipes(device, tt_weight_a, pipes_a, bank_to_receivers_a, ring_size)


@pytest.mark.parametrize("dual_senders", [False, True], ids=["single_sender", "dual_senders"])
def test_validator_pipe_mixed_num_receivers(device, dual_senders):
    """Two pipe sets with DIFFERENT receiver counts share one prefetcher.

    A sender reads num_receivers out of the config page each request names, so a request against
    the three-receiver set must not walk the two-receiver set's NOC XY table, or vice versa. Sizes
    are derived from the bank count so both sets work whatever the part's DRAM harvest.

    With dual senders the two sets also split their banks differently -- 1/1 against 2/1 -- so a
    bank's trailing sender holds a different bank-local slab base in each set. Those bases travel
    on the pipes, and each request stamps the ones belonging to the set it names, so the same
    DRISC core serving both sets in turn must pick up a different base each time.
    """
    K, dtype = 448, ttnn.bfloat8_b
    num_dram_banks = device.dram_grid_size().x
    # Receivers per bank of 2 and 3: both leave num_shards > num_dram_banks, which is what keeps the
    # weight an NdShardSpec (one shard per bank canonicalizes to legacy WIDTH_SHARDED, and this
    # transport takes receiver-contiguous only). N covers 6 = lcm(2, 3) receivers per bank so both
    # ring sizes divide it.
    N = num_dram_banks * 6 * 2 * ttnn.TILE_SIZE
    tt_weight_a, pipes_a, bank_to_receivers_a, _push_a, ring_a = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank=2, dual_senders=dual_senders
    )
    tt_weight_b, pipes_b, bank_to_receivers_b, _push_b, ring_b = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank=3, row_offset=_receiver_rows(num_dram_banks, 2), dual_senders=dual_senders
    )
    with tensor_prefetcher_session(device):
        _queue_and_validate_pipes(device, tt_weight_a, pipes_a, bank_to_receivers_a, ring_a)
        _queue_and_validate_pipes(device, tt_weight_b, pipes_b, bank_to_receivers_b, ring_b)
        _queue_and_validate_pipes(device, tt_weight_a, pipes_a, bank_to_receivers_a, ring_a)


@pytest.mark.parametrize("K,N,dtype,recv_per_bank", [(2048, 3584, ttnn.bfloat8_b, 2)])
def test_pipe_list_order_is_not_semantic(device, K, N, dtype, recv_per_bank, expect_error):
    """A pipe carries the bank-local slabs its sender owns, so the request list's order does not.

    What the list must still be is every pipe of a single
    create_prefetcher_pipes_for_tensor_prefetcher call, each appearing once. Every call numbers a
    bank's slabs from 0, so two calls' pipes for one bank claim the same slabs however they are
    ordered; a repeated pipe delivers to a receiver twice; and a subset leaves a pipe's slab base
    pointing past the slabs a tensor laid out for that subset has on its bank.
    """
    num_dram_banks = device.dram_grid_size().x
    tt_weight, pipes, bank_to_receivers, _push_page_size, ring_size = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank, dual_senders=True
    )
    assert len(pipes) > len({p.sender_core().x for p in pipes}), "this case needs a bank with two senders"
    # A second set on its own receiver rows, so its pipes clash with the first set's only over
    # slabs -- their receivers are disjoint.
    _tt_weight_b, pipes_b, _bank_to_receivers_b, _push_b, _ring_b = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank, dual_senders=True, row_offset=_receiver_rows(num_dram_banks, recv_per_bank)
    )

    def queue(pipe_list):
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(tt_weight, ring_size)], prefetcher_pipes=pipe_list)

    with tensor_prefetcher_session(device):
        # Every pipe listed twice, so every receiver is delivered to twice.
        with expect_error(RuntimeError, "disjoint receiver sets"):
            queue(pipes + pipes)
        # Each set's leading pipe for bank 0: disjoint receivers, but the request may only combine
        # pipes stamped by one factory call.
        with expect_error(RuntimeError, "one CreatePrefetcherPipesForTensorPrefetcher call"):
            queue([pipes[0], pipes_b[0]])
        # Only a bank's trailing sender: its slab base is past the leading sender's slabs.
        with expect_error(RuntimeError, "in any order, but"):
            queue(pipes[1:])
        # Interleaved: no bank's two pipes are adjacent any more. Each keeps its own slab base, so
        # delivery is unchanged.
        _queue_and_validate_pipes(device, tt_weight, pipes[::2] + pipes[1::2], bank_to_receivers, ring_size)
        # Bank 0's two senders swapped, for both the request and the validator. Delivery follows the
        # pipes' own bases and the validator's expectation follows bank_to_receivers, so neither
        # depends on list order.
        swapped = list(pipes)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        _queue_and_validate_pipes(device, tt_weight, swapped, bank_to_receivers, ring_size)
        # The list as the factory returned it still works, on the same prefetcher.
        _queue_and_validate_pipes(device, tt_weight, pipes, bank_to_receivers, ring_size)


@pytest.mark.parametrize("K,N,dtype,recv_per_bank", [(448, 1792, ttnn.bfloat8_b, 1)])
def test_pipe_outlives_the_list_it_came_in(device, K, N, dtype, recv_per_bank):
    """A pipe kept out of the returned list still holds its device, and the rest still deliver.

    A pipe's destructor frees its ring out of the device's persistent L1, so each pipe keeps the
    device alive on its own rather than relying on the list that carried it.
    """
    tt_weight, pipes, bank_to_receivers, push_page_size, ring_size = _setup_weight_and_pipes_recv_contig(
        device, K, N, dtype, recv_per_bank
    )
    kept = pipes[0]
    assert kept.sender_core_type() == "dram"
    assert kept.ring_size() == push_page_size * _RING_DEPTH_PAGES
    with tensor_prefetcher_session(device):
        _queue_and_validate_pipes(device, tt_weight, pipes, bank_to_receivers, ring_size)
    # Reading through the surviving handle after the list is gone must not touch freed device state.
    del pipes
    assert kept.config_address() > 0


def test_validator_pipe_recv_contig_page_size_switch(device):
    """A single recv-contig pipe set can serve tensors with different page sizes.

    The first tensor leaves the write cursor at an offset that is not aligned to the second tensor's
    larger page size. Both sender and receiver must credit the skipped padding when they re-grid the
    ring, otherwise the second validator waits forever.
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
    # After one ring of A pages, the write cursor advances by 52224 B, which is
    # not page_b-aligned and forces padding before B.
    weight_a, page_a = make_weight(K=512, n_per_recv_tiles=3, seed=0xA11CE)
    weight_b, page_b = make_weight(K=1024, n_per_recv_tiles=5, seed=0xB0B)
    assert (ring_size * page_a) % page_b != 0

    _space, pipes = _make_tensor_prefetcher_pipes(
        device, bank_to_receivers, ring_size * max(page_a, page_b), single_sender=True
    )

    with tensor_prefetcher_session(device):
        _queue_and_validate_pipes(device, weight_a, pipes, bank_to_receivers, ring_size)
        ttnn.synchronize_device(device)
        _queue_and_validate_pipes(device, weight_b, pipes, bank_to_receivers, ring_size)
