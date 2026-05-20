# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Multi-tensor smoke test for the DRAM-core prefetcher.

The DRAM-core kernel's main loop iterates `for layer in num_layers: for t in num_tensors:`
and prior versions were documented as leaving the DRISC cores in reset state when
num_tensors > 1. After the chunked-DMA + receiver-layout fixes this test re-verifies
the multi-tensor path end-to-end against a discard receiver.

Setup:
- 8 banks * 2 receivers/bank = 16 receivers (matches BW-bench small shape).
- 2 weight tensors of identical shape (so num_blocks matches across tensors).
- Prefetcher pushes [w0, w1] with num_layers=1.
- Discard receiver consumes num_iters = num_tensors * num_blocks per receiver.
- Test passes if both ops complete without hang/OOM/PCC failure.
"""

import os
import pytest
import torch
import ttnn
from loguru import logger


_NUM_RECV_PER_BANK = 2
_K = 2048
# N must be divisible by `num_dram_banks * _NUM_RECV_PER_BANK * TILE_SIZE` for any
# supported BH bank count. lcm(7, 8) * 2 * 32 = 56 * 64 = 3584, so 3584 works on both
# P100 (7 banks) and P150/P300 (8 banks).
_N = 3584
_TILE_BYTES = 1088  # bf8_b


def _dram_programmable_enabled() -> bool:
    return os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0") == "1"


def _round_up(n, m):
    return ((n + m - 1) // m) * m


def _bank_receivers_row_major(bank_idx: int, recv_per_bank: int, ring_cols: int):
    cores = []
    for k in range(recv_per_bank):
        ring_pos = bank_idx * recv_per_bank + k
        col = ring_pos % ring_cols
        row = ring_pos // ring_cols
        cores.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
    return ttnn.CoreRangeSet(cores)


def _make_weight(device, seed: int, num_dram_banks: int) -> ttnn.Tensor:
    torch.manual_seed(seed)
    pt_weight = torch.randn(1, 1, _K, _N)
    dram_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
    )
    weight_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [_K, _N // num_dram_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.as_tensor(
        pt_weight, device=device, dtype=ttnn.bfloat8_b, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
    )


@pytest.mark.skipif(
    not _dram_programmable_enabled(), reason="TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set"
)
@pytest.mark.parametrize("num_tensors,num_layers", [(1, 1), (2, 1), (3, 1), (2, 5), (3, 10)])
def test_dram_core_prefetcher_multi_tensor(device, num_tensors, num_layers):
    arch = getattr(device, "arch", lambda: None)()
    if arch is not None and "BLACKHOLE" not in str(arch).upper():
        pytest.skip("DRAM-core prefetcher requires Blackhole")

    ttnn.device.enable_asynchronous_slow_dispatch(device)

    # Query the chip's actual DRAM bank count so the test adapts to harvested BHs
    # (P100 has 7, P150/P300 have 8). The receiver grid is laid out as
    # `num_dram_banks` columns × `_NUM_RECV_PER_BANK` rows.
    num_dram_banks = device.dram_grid_size().x
    num_receivers = num_dram_banks * _NUM_RECV_PER_BANK

    # Build `num_tensors` weight tensors, all same shape (so num_blocks matches).
    weights = [_make_weight(device, seed=0xB100 + i, num_dram_banks=num_dram_banks) for i in range(num_tensors)]
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

    # GCB sized to hold one tensor's data per receiver. Push page = kbw * n_tiles_per_recv
    # * tile_bytes (1 * (N/num_dram_banks/recv_per_bank/TILE) tiles per push, bf8_b).
    n_tiles_per_recv = (_N // num_dram_banks // _NUM_RECV_PER_BANK) // ttnn.TILE_SIZE
    push_page_size = n_tiles_per_recv * _TILE_BYTES
    k_tiles = _K // ttnn.TILE_SIZE  # 64
    per_recv_bytes_per_tensor = k_tiles * push_page_size
    # Round up to push-page multiple. Single-tensor's worth fits the worker-core validation
    # (max_tensor_size <= gcb.size()); for multi-tensor the prefetcher just streams data
    # through the same fifo, so one-tensor's worth is the right minimum.
    gcb_size = _round_up(per_recv_bytes_per_tensor, push_page_size)

    bank_to_receivers = [
        (b, _bank_receivers_row_major(b, _NUM_RECV_PER_BANK, ring_cols=num_dram_banks)) for b in range(num_dram_banks)
    ]
    gcb = ttnn.create_global_circular_buffer_with_dram_senders(device, bank_to_receivers, gcb_size)

    # Per receiver per layer: 1 push per K-block per tensor.
    num_iters_total = num_layers * num_tensors * k_tiles
    logger.info(
        f"[multi_tensor] num_tensors={num_tensors} num_layers={num_layers} K={_K} N={_N} "
        f"banks={num_dram_banks} ring={num_receivers} push_page={push_page_size} gcb_size={gcb_size} "
        f"num_iters_total={num_iters_total}"
    )

    # Sender: push all `num_tensors` weights through the prefetcher, num_layers times.
    ttnn.start_dram_core_prefetcher(
        device,
        weights + [addrs],
        num_layers=num_layers,
        global_cb=gcb,
    )
    # Receiver: discard all pushed data.
    ttnn.dram_prefetcher_consumer(
        device,
        num_iters=num_iters_total,
        page_size_bytes=push_page_size,
        global_cb=gcb,
    )
    ttnn.stop_dram_core_prefetcher(device)
    ttnn.synchronize_device(device)
    logger.info(f"[multi_tensor] num_tensors={num_tensors} num_layers={num_layers} completed cleanly")
