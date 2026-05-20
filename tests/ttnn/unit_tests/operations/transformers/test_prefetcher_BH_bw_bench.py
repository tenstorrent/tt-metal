# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Bandwidth bench: DRAM-core prefetcher vs worker-core prefetcher push BW.

Setup:
  - 8 senders, 2 receivers per sender = 16 receivers total.
  - Receivers run gcb_bench_discard_receiver.cpp: wait_front(1)+pop_front(1) in a
    loop, discard data. No matmul, so we isolate the prefetcher's push BW from
    receiver-side compute.
  - Same tensor pushed num_layers times.

Shape: K=2048, N=4096, bf8_b weights, 8-way DRAM bank split.
  - N_per_bank = 4096/8 = 512 elems = 16 tiles
  - N_per_recv = 512/2 = 256 elems = 8 tiles
  - max in-flight at receiver: K_tiles * N_per_recv_tiles * 1088 = 64 * 8 * 1088
    = ~544 KB. Fits comfortably in 1.5 MB worker L1.
  - Llama-3.1-8B FF1 (K=4096 N=14336) per-receiver tensor at 16 receivers is
    ~3.7 MB at bf8_b, which doesn't fit per-receiver L1. Using this smaller
    shape keeps the same FF-ratio (K:N = 1:2) and topology.

Push counts per layer (different for the two paths because they push at
different granularities; bytes-per-layer match):
  - DRAM-core (kInBlockWTiles=1):
      num_iters_per_layer = K_tiles = 64
      page_size = 1 * N_per_recv_tiles * 1088 = 8704 bytes
  - Worker-core (num_blocks = num_senders * num_recv_per_sender):
      num_blocks = 8 * 2 = 16
      block_tiles = K_tiles * N_per_bank_tiles / num_blocks = 64 * 16 / 16 = 64
      page_size = block_tiles * tile_bytes / num_recv_per_sender = 64 * 1088 / 2 = 34816 bytes
      num_iters_per_layer = num_blocks = 16

Total bytes pushed per receiver per layer = 64 * 8704 = 16 * 34816 = ~544 KB
(same for both paths).

Use BENCH_LAYERS to tune num_layers (default 1000).
"""

import math
import os
import time
import pytest
import torch
import ttnn
from loguru import logger


def _dram_programmable_enabled() -> bool:
    return os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0") == "1"


def _bench_layers() -> int:
    return int(os.environ.get("BENCH_LAYERS", "1000"))


def _round_up(n, m):
    return ((n + m - 1) // m) * m


# Workload constants. Override via env vars for cross-shape experiments.
_M = 32  # matmul batch dim; unused on the receiver side (no matmul) but needed by tensor shape
_K = int(os.environ.get("BENCH_K", "2048"))
_N = int(os.environ.get("BENCH_N", "4096"))
_NUM_BANKS = 8
_NUM_RECV_PER_BANK = int(os.environ.get("BENCH_RECV_PER_BANK", "1"))
_NUM_RECEIVERS = _NUM_BANKS * _NUM_RECV_PER_BANK
_DTYPE_NAME = os.environ.get("BENCH_DTYPE", "bfloat8_b")  # "bfloat8_b" or "bfloat16"
_DTYPE = {"bfloat8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16}[_DTYPE_NAME]
_TILE_BYTES = {"bfloat8_b": 1088, "bfloat16": 2048}[_DTYPE_NAME]
_BF8_TILE_BYTES = _TILE_BYTES  # kept for backward-compat in derivations below


_DRAM_CORE_K_BLOCK_W_TILES = int(os.environ.get("BENCH_DRAM_CORE_K_BLOCK_W", "1"))


def _expected_push_geometry(path: str, num_dram_banks: int):
    """Returns (num_iters_per_layer, page_size_bytes) for each prefetcher path."""
    n_tiles_per_bank = (_N // num_dram_banks) // ttnn.TILE_SIZE
    n_tiles_per_recv = n_tiles_per_bank // _NUM_RECV_PER_BANK
    k_tiles = _K // ttnn.TILE_SIZE
    if path == "dram_core":
        kbw = _DRAM_CORE_K_BLOCK_W_TILES
        assert k_tiles % kbw == 0, f"k_tiles ({k_tiles}) must be divisible by kbw ({kbw})"
        num_iters = k_tiles // kbw
        page_size = kbw * n_tiles_per_recv * _TILE_BYTES
    elif path == "worker_core":
        num_blocks = num_dram_banks * _NUM_RECV_PER_BANK  # = num_senders * num_recv_per_sender
        block_tiles = k_tiles * n_tiles_per_bank // num_blocks
        num_iters = num_blocks
        page_size = block_tiles * _TILE_BYTES // _NUM_RECV_PER_BANK
    else:
        raise ValueError(path)
    return num_iters, page_size


def _make_weight(device, num_dram_banks: int) -> ttnn.Tensor:
    torch.manual_seed(0xBE7)
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
        pt_weight, device=device, dtype=_DTYPE, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
    )


def _build_gcb_size(page_size_bytes: int) -> int:
    """GCB fifo: hold a full per-receiver tensor's worth of in-flight pages.

    Worker-core prefetcher's program factory validates `max_tensor_size <= gcb.size()`, where
    max_tensor_size = K_tiles * N_per_recv * tile_bytes (the full per-receiver bank shard).
    For our K=2048 N=4096 16-receiver config at bf8_b, that's 64 * 8 * 1088 = ~544 KB.
    Round up + a little headroom."""
    k_tiles = _K // ttnn.TILE_SIZE
    n_tiles_per_recv = (_N // _NUM_BANKS) // ttnn.TILE_SIZE // _NUM_RECV_PER_BANK
    max_tensor_bytes = k_tiles * n_tiles_per_recv * _BF8_TILE_BYTES
    # Round up to page_size to stay page-aligned, then to 4 KB.
    pages_to_hold = max(1, (max_tensor_bytes + page_size_bytes - 1) // page_size_bytes)
    return _round_up(pages_to_hold * page_size_bytes, 4096)


def _time_one_run(device, sender_fn, consumer_fn, warmup_runs: int = 1, timed_runs: int = 3) -> float:
    """Run (sender + consumer) and time it. Returns mean elapsed seconds across timed_runs."""
    for _ in range(warmup_runs):
        sender_fn()
        consumer_fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(timed_runs):
        sender_fn()
        consumer_fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / timed_runs


def _gbps(bytes_per_run: int, elapsed_s: float) -> float:
    return bytes_per_run / elapsed_s / 1e9


@pytest.mark.skipif(
    not _dram_programmable_enabled(), reason="TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set"
)
def test_bw_dram_core_prefetcher(device):
    arch = getattr(device, "arch", lambda: None)()
    if arch is not None and "BLACKHOLE" not in str(arch).upper():
        pytest.skip("DRAM-core prefetcher requires Blackhole")

    ttnn.device.enable_asynchronous_slow_dispatch(device)

    # Query DRAM bank count so the bench runs on harvested BHs (P100 has 7 banks).
    num_dram_banks = device.dram_grid_size().x
    num_receivers = num_dram_banks * _NUM_RECV_PER_BANK

    num_layers = _bench_layers()
    num_iters_per_layer, page_size = _expected_push_geometry("dram_core", num_dram_banks)
    num_iters_total = num_layers * num_iters_per_layer

    tt_weight = _make_weight(device, num_dram_banks)

    # tensor_addrs: unused on DRAM-core path but required by op contract.
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

    # GCB: receivers laid out as num_dram_banks columns × _NUM_RECV_PER_BANK rows
    # (col=bank, rows 0..recv_per_bank-1). On P100 banks=7, on P150/P300 banks=8.
    bank_to_receivers = []
    for b in range(num_dram_banks):
        bank_to_receivers.append(
            (b, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, _NUM_RECV_PER_BANK - 1))}))
        )
    gcb_size = _build_gcb_size(page_size)
    gcb = ttnn.create_global_circular_buffer_with_dram_senders(device, bank_to_receivers, gcb_size)

    logger.info(
        f"[dram_core_bw] K={_K} N={_N} {_DTYPE_NAME} banks={num_dram_banks} ring={num_receivers} num_layers={num_layers} "
        f"k_block_w={_DRAM_CORE_K_BLOCK_W_TILES} iters/layer={num_iters_per_layer} "
        f"page_size={page_size} gcb_size={gcb_size}"
    )

    def sender_fn():
        ttnn.start_dram_core_prefetcher(
            device,
            [tt_weight, addrs],
            num_layers=num_layers,
            global_cb=gcb,
            dram_core_k_block_w_tiles=_DRAM_CORE_K_BLOCK_W_TILES,
        )

    def consumer_fn():
        ttnn.dram_prefetcher_consumer(device, num_iters=num_iters_total, page_size_bytes=page_size, global_cb=gcb)
        ttnn.stop_dram_core_prefetcher(device)

    elapsed = _time_one_run(device, sender_fn, consumer_fn)
    bytes_per_run = num_iters_total * page_size * num_receivers  # aggregate across all receivers
    bw = _gbps(bytes_per_run, elapsed)
    per_recv_bw = bw / num_receivers
    logger.info(
        f"[dram_core_bw] elapsed={elapsed * 1e3:.2f}ms bytes={bytes_per_run / 1e6:.1f}MB "
        f"aggregate_bw={bw:.2f} GB/s per_recv_bw={per_recv_bw:.3f} GB/s"
    )


def test_bw_workercore_prefetcher(device):
    arch = getattr(device, "arch", lambda: None)()
    if arch is not None and "BLACKHOLE" not in str(arch).upper():
        pytest.skip("Bench tuned for Blackhole topology")

    ttnn.device.enable_asynchronous_slow_dispatch(device)

    # Worker-core path is out of scope for the harvested-card adaptation; keep
    # the unharvested _NUM_BANKS=8 baseline so the bench numbers stay comparable
    # to historical runs.
    num_dram_banks = _NUM_BANKS
    num_layers = _bench_layers()
    num_iters_per_layer, page_size = _expected_push_geometry("worker_core", num_dram_banks)
    num_iters_total = num_layers * num_iters_per_layer

    tt_weight = _make_weight(device, num_dram_banks)

    # Worker-core senders: row 2 cols 0..7 (sender per bank). Receivers: rows 0..1 cols 0..7.
    sender_core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(_NUM_BANKS - 1, 2))})

    sender_receiver_mapping = []
    for b in range(_NUM_BANKS):
        sender_receiver_mapping.append(
            (
                ttnn.CoreCoord(b, 2),
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, _NUM_RECV_PER_BANK - 1))}),
            )
        )
    gcb_size = _build_gcb_size(page_size)
    gcb = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, gcb_size)

    # tensor_addrs: real for worker-core path.
    tensor_addrs = torch.tensor([tt_weight.buffer_address()] * num_layers, dtype=torch.int32).reshape(1, num_layers)
    tensor_addrs = tensor_addrs.repeat(_NUM_BANKS, 1)
    tensor_addrs_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sender_core_range_set, [1, num_layers], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_addrs = ttnn.as_tensor(
        tensor_addrs,
        device=device,
        dtype=ttnn.uint32,
        memory_config=tensor_addrs_mem_config,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    logger.info(
        f"[workercore_bw] K={_K} N={_N} bf8_b ring={_NUM_RECEIVERS} num_layers={num_layers} "
        f"iters/layer={num_iters_per_layer} page_size={page_size} gcb_size={gcb_size}"
    )

    def sender_fn():
        ttnn.dram_prefetcher([tt_weight, tt_addrs], num_layers=num_layers, global_cb=gcb, enable_performance_mode=True)

    def consumer_fn():
        ttnn.dram_prefetcher_consumer(device, num_iters=num_iters_total, page_size_bytes=page_size, global_cb=gcb)

    elapsed = _time_one_run(device, sender_fn, consumer_fn)
    bytes_per_run = num_iters_total * page_size * _NUM_RECEIVERS
    bw = _gbps(bytes_per_run, elapsed)
    per_recv_bw = bw / _NUM_RECEIVERS
    logger.info(
        f"[workercore_bw] elapsed={elapsed * 1e3:.2f}ms bytes={bytes_per_run / 1e6:.1f}MB "
        f"aggregate_bw={bw:.2f} GB/s per_recv_bw={per_recv_bw:.3f} GB/s"
    )
