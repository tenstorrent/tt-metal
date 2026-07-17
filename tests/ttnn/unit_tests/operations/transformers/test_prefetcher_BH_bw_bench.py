# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Push-bandwidth bench: Tensor prefetcher vs worker-core prefetcher.

Mirrors `test_prefetcher_BH_bench.py`'s shape parametrization, but swaps the
matmul receiver for `ttnn.experimental.test_dram_prefetcher_consumer` — a discard receiver
that does `wait_front(1) + pop_front(1)` in a loop and throws the data
away. This isolates the prefetcher's push bandwidth from receiver-side
compute (no matmul, no L1 budget for matmul CBs).

Methodology (identical shape on both paths):
- Prefetcher launched/enqueued **once** outside the trace with
  `num_layers = trace_repeats + 1` (1 warmup + N traced layers).
- One warmup consumer call outside the trace, draining one layer
  (= ring_size pages). This also primes the cached MeshWorkload so the
  kernel-binary write happens here.
- Trace captures `trace_repeats` consumer ops (each draining one layer)
  and is replayed once. We time the replay + sync.

The consumer op caches its MeshWorkload across calls
(see `dram_prefetcher_consumer.cpp`). The first call writes the kernel
binary to DRAM (incompatible with trace capture); subsequent calls with
the same args reuse the cached workload and skip that write — so the
warmup call outside the trace is what makes capture-then-replay legal.

For the worker-core path the GCB sender/receiver mapping comes from the
production prefetcher_config.yaml, and a sub-device pins the consumer to
the receivers so it doesn't collide with dispatch cores.

Page size matches what each path pushes per receiver per ring-block:
`page_size = (K_padded / ring_size) * (N_padded / ring_size) * tile_bytes`.
Per-layer per receiver = ring_size pages = K_padded * (N_padded / ring_size)
* tile_bytes = the full per-receiver weight bytes.

Shapes are the production Llama prefetcher matmul shapes at the smallest
device count where the worker prefetcher fits — same set as the matmul
bench. See docs/tensor_prefetcher_bench_results.md for the matmul
companion numbers and docs/tensor_prefetcher_bw_results.md for the BW
results table.
"""

import os
import time
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    round_up as _round_up,
    make_recv_contig_weight as _make_recv_contig_weight,
)


pytestmark = run_for_blackhole("Tensor prefetcher requires Blackhole")


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    """Skip unless programmable DRAM cores are available on this device."""
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip(
            "programmable DRAM cores unavailable (need Blackhole, firmware >= 19.12.0.0, and either no harvested DRAM channels or a single device)"
        )


_NUM_DRAM_BANKS = 8

# Mutable shape globals populated by _apply_shape() at the top of each test.
_K = 512
_K_ORIG = 512
_N = 1024
_N_ORIG = 1024
_NUM_RECV_PER_BANK = 1
_RING_SIZE = _NUM_DRAM_BANKS * _NUM_RECV_PER_BANK
_RING_COLS = 8
_DTYPE_NAME = "bfloat8_b"
_DTYPE = ttnn.bfloat8_b
_DTYPE_BYTES = 1088 / 1024.0  # bfloat8_b per-element bytes including per-tile header

_DTYPE_FROM_NAME = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}
_DTYPE_BYTES_FROM_NAME = {"bfloat16": 2, "bfloat8_b": 1088 / 1024.0}


# Same set of shapes as test_prefetcher_BH_bench.py::LLAMA_SHAPES. Kept duplicated rather
# than imported to keep this file self-contained; if you add a row here add it there too.
LLAMA_SHAPES = [
    # Llama-3.1-8B @ 1 dev: K-row-major lands on M>1 (one K-row > ring_half);
    # receiver-contiguous's advertised win is on these shapes.
    pytest.param("8B_FF1_1d", dict(K=4096, N=14336, dtype="bfloat8_b", recv_per_bank=8), id="8B_FF1_1d"),
    pytest.param("8B_QKV_1d", dict(K=4096, N=12288, dtype="bfloat8_b", recv_per_bank=8), id="8B_QKV_1d"),
    pytest.param("8B_WO_1d", dict(K=4096, N=4096, dtype="bfloat8_b", recv_per_bank=8), id="8B_WO_1d"),
    # Llama-3.2-1B @ 1 dev
    pytest.param("1B_QKV", dict(K=2048, N=3072, dtype="bfloat8_b", recv_per_bank=8), id="1B_QKV"),
    pytest.param("1B_WO", dict(K=2048, N=2048, dtype="bfloat8_b", recv_per_bank=8), id="1B_WO"),
    pytest.param("1B_FF1", dict(K=2048, N=8192, dtype="bfloat8_b", recv_per_bank=8), id="1B_FF1"),
    pytest.param("1B_FF2", dict(K=8192, N=2048, dtype="bfloat8_b", recv_per_bank=8), id="1B_FF2"),
    # Llama-3.2-3B @ 1 dev
    pytest.param("3B_QKV", dict(K=3072, N=5120, dtype="bfloat8_b", recv_per_bank=8), id="3B_QKV"),
    pytest.param("3B_WO", dict(K=3072, N=3072, dtype="bfloat8_b", recv_per_bank=8), id="3B_WO"),
    pytest.param("3B_FF1", dict(K=3072, N=8192, dtype="bfloat8_b", recv_per_bank=8), id="3B_FF1"),
    pytest.param("3B_FF2", dict(K=8192, N=3072, dtype="bfloat8_b", recv_per_bank=8), id="3B_FF2"),
    # Llama-3.1-8B @ 2 dev (per-device shapes)
    pytest.param("8B_QKV_2d", dict(K=4096, N=3072, dtype="bfloat8_b", recv_per_bank=8), id="8B_QKV_2d"),
    pytest.param("8B_WO_2d", dict(K=2048, N=4096, dtype="bfloat8_b", recv_per_bank=8), id="8B_WO_2d"),
    pytest.param("8B_FF1_2d", dict(K=4096, N=7168, dtype="bfloat8_b", recv_per_bank=8), id="8B_FF1_2d"),
    pytest.param("8B_FF2_2d", dict(K=7168, N=4096, dtype="bfloat8_b", recv_per_bank=8), id="8B_FF2_2d"),
    # Llama-3.3-70B @ 8 dev (per-device shapes)
    pytest.param("70B_QKV_8d", dict(K=8192, N=1280, dtype="bfloat8_b", recv_per_bank=8), id="70B_QKV_8d"),
    pytest.param("70B_WO_8d", dict(K=1024, N=8192, dtype="bfloat8_b", recv_per_bank=8), id="70B_WO_8d"),
    pytest.param("70B_FF1_8d", dict(K=8192, N=3584, dtype="bfloat8_b", recv_per_bank=8), id="70B_FF1_8d"),
    pytest.param("70B_FF2_8d", dict(K=3584, N=8192, dtype="bfloat8_b", recv_per_bank=8), id="70B_FF2_8d"),
]


def _bench_trace_repeats() -> int:
    return int(os.environ.get("BENCH_TRACE_REPEATS", "100"))


def _apply_shape(shape: dict) -> None:
    global _K, _K_ORIG, _N, _N_ORIG, _NUM_RECV_PER_BANK, _RING_SIZE
    global _DTYPE_NAME, _DTYPE, _DTYPE_BYTES
    _K_ORIG = int(os.environ.get("BENCH_K", shape["K"]))
    _N_ORIG = int(os.environ.get("BENCH_N", shape["N"]))
    _NUM_RECV_PER_BANK = int(os.environ.get("BENCH_RECV_PER_BANK", shape["recv_per_bank"]))
    _RING_SIZE = _NUM_DRAM_BANKS * _NUM_RECV_PER_BANK
    assert _RING_SIZE % _RING_COLS == 0, f"ring_size {_RING_SIZE} not a clean rect on {_RING_COLS}-col grid"
    _K = _round_up(_K_ORIG, _RING_SIZE * ttnn.TILE_SIZE)
    _N = _round_up(_N_ORIG, _RING_SIZE * ttnn.TILE_SIZE)
    _DTYPE_NAME = os.environ.get("BENCH_DTYPE", shape["dtype"])
    _DTYPE = _DTYPE_FROM_NAME[_DTYPE_NAME]
    _DTYPE_BYTES = _DTYPE_BYTES_FROM_NAME[_DTYPE_NAME]


def _per_receiver_page_size_bytes() -> int:
    """One ring-block's bytes on a single receiver — matches what both prefetchers push
    per block (DRAM-core: k_block_w_tiles * n_per_recv_tiles * tile_bytes; worker-core:
    same value via the matmul writer's block_size_per_receiver derivation).
    """
    k_block_w_tiles = (_K // _RING_SIZE) // ttnn.TILE_SIZE
    n_per_recv_tiles = (_N // _RING_SIZE) // ttnn.TILE_SIZE
    return k_block_w_tiles * n_per_recv_tiles * int(_DTYPE_BYTES * ttnn.TILE_SIZE * ttnn.TILE_SIZE)


def _gcb_size_bytes(page_size: int, pages_per_layer: int) -> int:
    """Per-receiver GCB size = one full layer's worth of pages. Matches what the worker
    path passes (`pages_per_layer * page_size`) so the two paths have symmetric buffer
    depth. Previously this was `4 * page_size` for the DRAM-core path, which left only
    4 pages of in-flight headroom and gave `reserve_back` a ~16× backpressure tax vs the
    worker — making the comparison unfair (see docs/tensor_prefetcher_drisc_profile.md).
    """
    return pages_per_layer * page_size


def _build_weight(device, num_dram_banks: int) -> ttnn.Tensor:
    """DRAM-width-sharded weight tensor (the prefetcher pulls from this)."""
    torch.manual_seed(0xBED)
    pt_weight = torch.randn(1, 1, _K, _N)
    dram_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [_K, _N // num_dram_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.as_tensor(pt_weight, device=device, dtype=_DTYPE, memory_config=mem_config, layout=ttnn.TILE_LAYOUT)


def _build_weight_recv_contig(device, num_dram_banks: int) -> ttnn.Tensor:
    """DRAM-sharded weight in receiver-contiguous layout: NdShardSpec with
    `num_shards = ring_size` round-robin across `num_dram_banks` DRAM cores.
    Each shard is (K, n_per_recv); bank b holds `_NUM_RECV_PER_BANK` slabs
    stacked vertically. The manager auto-detects this from
    BufferDistributionSpec::num_shards() and dispatches to the recv-contig
    compute_tensor_geom path.
    """
    torch.manual_seed(0xBED)
    pt_weight = torch.randn(1, 1, _K, _N)
    return _make_recv_contig_weight(device, pt_weight, num_dram_banks, _RING_SIZE, _DTYPE)


def _build_dummy_addrs(device) -> ttnn.Tensor:
    """tensor_addrs is unused on the DRAM-core path but required by op contract."""
    return ttnn.from_torch(
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


def _gbps(bytes_total: float, elapsed_s: float) -> float:
    return bytes_total / elapsed_s / 1e9


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("op_name,shape", LLAMA_SHAPES)
def test_bw_tensor_prefetcher(device, op_name, shape):
    """Tensor prefetcher → discard receiver. Prefetcher launched out-of-band with
    num_layers = trace_repeats + 1 (1 warmup + N traced). A trace captures N consumer
    ops (each draining one layer worth = ring_size pages) and is replayed once. Same
    methodology as the matmul bench (test_bench_dram_core_repeats).
    """
    _apply_shape(shape)
    if device.dram_grid_size().x != 8:
        pytest.skip("DRAM-core bench expects 8 unharvested DRAM banks")

    trace_repeats = _bench_trace_repeats()
    num_prefetch_layers = trace_repeats + 1  # 1 warmup + trace_repeats inside the trace

    num_dram_banks = device.dram_grid_size().x
    num_receivers = num_dram_banks * _NUM_RECV_PER_BANK
    page_size = _per_receiver_page_size_bytes()
    pages_per_layer = num_receivers  # = ring_size

    tt_weight = _build_weight(device, num_dram_banks)
    addrs = _build_dummy_addrs(device)

    # DRAM-sender GCB with a simple grid receiver layout (banks 0..N-1 in row 0;
    # receivers stacked in rows 0..recv_per_bank-1). Works for DRAM-core because senders
    # are DRAM cores, not workers — no dispatch-core collision.
    bank_to_receivers = [
        (
            b,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, _NUM_RECV_PER_BANK - 1))}),
        )
        for b in range(num_dram_banks)
    ]
    gcb_size = _gcb_size_bytes(page_size, pages_per_layer)
    gcb = ttnn.experimental.create_global_circular_buffer_with_dram_senders(device, bank_to_receivers, gcb_size)

    ttnn.experimental.start_tensor_prefetcher(device)
    ttnn.experimental.queue_tensor_prefetcher_request(
        device, [(tt_weight, num_receivers)] * num_prefetch_layers, global_cb=gcb
    )

    # Warmup: drain 1 layer's worth of pages — this also primes the cached MeshWorkload
    # so the binary write happens here, not inside the trace.
    ttnn.experimental.test_dram_prefetcher_consumer(
        device, num_iters=pages_per_layer, page_size_bytes=page_size, global_cb=gcb
    )
    ttnn.synchronize_device(device)

    # Trace: trace_repeats consumer ops, each draining one layer (= pages_per_layer pages).
    bench_trace = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(trace_repeats):
        ttnn.experimental.test_dram_prefetcher_consumer(
            device, num_iters=pages_per_layer, page_size_bytes=page_size, global_cb=gcb
        )
    ttnn.end_trace_capture(device, bench_trace, cq_id=0)

    t0 = time.perf_counter()
    ttnn.execute_trace(device, bench_trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0
    ttnn.release_trace(device, bench_trace)
    ttnn.experimental.stop_tensor_prefetcher(device)

    if os.environ.get("TT_METAL_WATCHER", "0") == "1":
        time.sleep(3)  # let watcher dump DRISC ring buffers before device close

    bytes_per_recv = trace_repeats * pages_per_layer * page_size
    bytes_total = bytes_per_recv * num_receivers
    bw_total = _gbps(bytes_total, elapsed)
    bw_per_recv = bw_total / num_receivers
    logger.info(
        f"[dram_core_bw][{op_name}] trace_elapsed={elapsed * 1e3:.2f}ms "
        f"bytes_per_recv={bytes_per_recv / 1e6:.1f}MB total_bytes={bytes_total / 1e9:.2f}GB "
        f"aggregate_bw={bw_total:.2f} GB/s per_recv_bw={bw_per_recv:.3f} GB/s"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("op_name,shape", LLAMA_SHAPES)
def test_bw_tensor_prefetcher_recv_contig(device, op_name, shape):
    """Tensor prefetcher in **receiver-contiguous** layout → discard receiver.

    Identical methodology to test_bw_tensor_prefetcher: prefetcher launched
    out-of-band with num_layers = trace_repeats + 1, one warmup consumer drain,
    then trace captures `trace_repeats` consumer ops and is replayed once.

    The only differences from the K-row-major bench:
    - Weight allocated via NdShardSpec(Shape([K, n_per_recv]), ROUND_ROBIN_1D)
      with num_shards = ring_size > num_dram_banks. Bank b stacks
      num_recv_per_bank slabs of (K, n_per_recv) vertically. Manager auto-
      detects this and takes the recv-contig main loop in the DRISC kernel.
    - GCB topology is the same column-of-receivers layout the K-row-major
      bench uses (bank b -> col b, rows 0..R-1). Under BDS round-robin this
      makes bank b's slot r feed the receiver assigned ring position
      `b + r * num_banks` — a "strided" assignment. For a discard-receiver
      BW bench this is fine; byte correctness is exercised by
      test_validator_dram_sender_recv_contig.
    """
    _apply_shape(shape)
    if device.dram_grid_size().x != 8:
        pytest.skip("DRAM-core bench expects 8 unharvested DRAM banks")

    trace_repeats = _bench_trace_repeats()
    num_prefetch_layers = trace_repeats + 1  # 1 warmup + trace_repeats inside the trace

    num_dram_banks = device.dram_grid_size().x
    num_receivers = num_dram_banks * _NUM_RECV_PER_BANK
    page_size = _per_receiver_page_size_bytes()
    pages_per_layer = num_receivers  # = ring_size

    tt_weight = _build_weight_recv_contig(device, num_dram_banks)

    bank_to_receivers = [
        (
            b,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, _NUM_RECV_PER_BANK - 1))}),
        )
        for b in range(num_dram_banks)
    ]
    gcb_size = _gcb_size_bytes(page_size, pages_per_layer)
    dual_senders = os.environ.get("BENCH_DUAL_SENDERS", "0") == "1"
    gcb = ttnn.experimental.create_global_circular_buffer_with_dram_senders(
        device, bank_to_receivers, gcb_size, dual_senders_per_bank=dual_senders
    )

    ttnn.experimental.start_tensor_prefetcher(device)
    ttnn.experimental.queue_tensor_prefetcher_request(
        device, [(tt_weight, num_receivers)] * num_prefetch_layers, global_cb=gcb
    )

    # Warmup: drain 1 layer's worth of pages — primes the cached MeshWorkload.
    ttnn.experimental.test_dram_prefetcher_consumer(
        device, num_iters=pages_per_layer, page_size_bytes=page_size, global_cb=gcb
    )
    ttnn.synchronize_device(device)

    bench_trace = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(trace_repeats):
        ttnn.experimental.test_dram_prefetcher_consumer(
            device, num_iters=pages_per_layer, page_size_bytes=page_size, global_cb=gcb
        )
    ttnn.end_trace_capture(device, bench_trace, cq_id=0)

    t0 = time.perf_counter()
    ttnn.execute_trace(device, bench_trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0
    ttnn.release_trace(device, bench_trace)
    ttnn.experimental.stop_tensor_prefetcher(device)

    if os.environ.get("TT_METAL_WATCHER", "0") == "1":
        time.sleep(3)

    bytes_per_recv = trace_repeats * pages_per_layer * page_size
    bytes_total = bytes_per_recv * num_receivers
    bw_total = _gbps(bytes_total, elapsed)
    bw_per_recv = bw_total / num_receivers
    logger.info(
        f"[dram_core_bw_rc][{op_name}] dual_senders={dual_senders} trace_elapsed={elapsed * 1e3:.2f}ms "
        f"bytes_per_recv={bytes_per_recv / 1e6:.1f}MB total_bytes={bytes_total / 1e9:.2f}GB "
        f"aggregate_bw={bw_total:.2f} GB/s per_recv_bw={bw_per_recv:.3f} GB/s"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("op_name,shape", LLAMA_SHAPES)
def test_bw_tensor_prefetcher_streaming(device, op_name, shape):
    """DRAM-core prefetcher in **streaming** receiver-contiguous mode → discard receiver.

    Same recv-contig weight + topology as test_bw_tensor_prefetcher_recv_contig, but
    the request is queued with a per-receiver rotation table (each receiver's blocks read
    circularly from its ring index) and the GCB is sized to a small window of
    BENCH_GCB_WINDOW_BLOCKS blocks/receiver (default 4) instead of a full layer. This measures
    the prefetcher-side cost of streaming — the two-DMA-read source split on wrapping receivers
    plus the shallower GCB backpressure — against the batched recv-contig and K-row-major push-BW
    numbers.
    The discard consumer drains FIFO order regardless of block content, so the rotated
    delivery is BW-neutral; byte correctness is covered by the validator + matmul PCC tests.
    """
    _apply_shape(shape)

    trace_repeats = _bench_trace_repeats()
    num_prefetch_layers = trace_repeats + 1

    num_dram_banks = device.dram_grid_size().x
    num_receivers = num_dram_banks * _NUM_RECV_PER_BANK
    page_size = _per_receiver_page_size_bytes()
    pages_per_layer = num_receivers  # = ring_size

    tt_weight = _build_weight_recv_contig(device, num_dram_banks)

    bank_to_receivers = [
        (
            b,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, _NUM_RECV_PER_BANK - 1))}),
        )
        for b in range(num_dram_banks)
    ]
    # Shallow GCB: a small live window instead of a full layer (the streaming win).
    window_blocks = int(os.environ.get("BENCH_GCB_WINDOW_BLOCKS", "4"))
    window_blocks = min(window_blocks, pages_per_layer)
    gcb_size = window_blocks * page_size
    dual_senders = os.environ.get("BENCH_DUAL_SENDERS", "0") == "1"
    gcb = ttnn.experimental.create_global_circular_buffer_with_dram_senders(
        device, bank_to_receivers, gcb_size, dual_senders_per_bank=dual_senders
    )

    logger.info(
        f"[dram_core_bw_stream][{op_name}] dual_senders={dual_senders} window_blocks={window_blocks} "
        f"K={_K_ORIG} K_padded={_K} N={_N_ORIG} N_padded={_N} ring={num_receivers} page_size={page_size} "
        f"pages_per_layer={pages_per_layer} gcb_size={gcb_size} trace_repeats={trace_repeats}"
    )

    ttnn.experimental.start_tensor_prefetcher(device)
    # Identity rotation (rotation[r] = r) = natural topology ring order (reproduces old streaming=True).
    ttnn.experimental.queue_tensor_prefetcher_request(
        device, [(tt_weight, num_receivers, list(range(num_receivers)))] * num_prefetch_layers, global_cb=gcb
    )

    ttnn.experimental.test_dram_prefetcher_consumer(
        device, num_iters=pages_per_layer, page_size_bytes=page_size, global_cb=gcb
    )
    ttnn.synchronize_device(device)

    bench_trace = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(trace_repeats):
        ttnn.experimental.test_dram_prefetcher_consumer(
            device, num_iters=pages_per_layer, page_size_bytes=page_size, global_cb=gcb
        )
    ttnn.end_trace_capture(device, bench_trace, cq_id=0)

    t0 = time.perf_counter()
    ttnn.execute_trace(device, bench_trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0
    ttnn.release_trace(device, bench_trace)
    ttnn.experimental.stop_tensor_prefetcher(device)

    if os.environ.get("TT_METAL_WATCHER", "0") == "1":
        time.sleep(3)

    bytes_per_recv = trace_repeats * pages_per_layer * page_size
    bytes_total = bytes_per_recv * num_receivers
    bw_total = _gbps(bytes_total, elapsed)
    bw_per_recv = bw_total / num_receivers
    logger.info(
        f"[dram_core_bw_stream][{op_name}] dual_senders={dual_senders} window_blocks={window_blocks} "
        f"trace_elapsed={elapsed * 1e3:.2f}ms bytes_per_recv={bytes_per_recv / 1e6:.1f}MB "
        f"total_bytes={bytes_total / 1e9:.2f}GB aggregate_bw={bw_total:.2f} GB/s "
        f"per_recv_bw={bw_per_recv:.3f} GB/s"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("op_name,shape", LLAMA_SHAPES)
def test_bw_workercore_prefetcher(device, op_name, shape):
    """Worker-core prefetcher → discard receiver. Same shape as test_bw_dram_core:
    prefetcher enqueued ONCE outside the trace with `num_layers = trace_repeats + 1`,
    one warmup consumer call (drains 1 layer, primes the cached workload), then trace
    captures `trace_repeats` consumer ops (each drains one layer). Sub-device pins the
    consumer to the GCB receivers to avoid dispatch-core collision. Sender/receiver
    layout from `models/tt_transformers/tt/prefetcher/prefetcher_config.yaml`.
    """
    _apply_shape(shape)
    if device.dram_grid_size().x != 8:
        pytest.skip("Worker-sender bench expects 8 unharvested DRAM banks")

    trace_repeats = _bench_trace_repeats()
    num_prefetch_layers = trace_repeats + 1  # 1 warmup + trace_repeats inside the trace
    num_senders = _NUM_DRAM_BANKS
    page_size = _per_receiver_page_size_bytes()
    pages_per_layer = _RING_SIZE  # one page per ring position per layer per tensor

    # Production sender/receiver layout (cols 0/7 senders, scattered receivers) from the
    # YAML config. The naive row-major grid collides with dispatch cores on this hardware.
    from models.tt_transformers.tt.prefetcher import generate_sender_receiver_mapping, ARCH_CONFIG

    bh_cfg = ARCH_CONFIG["blackhole"]
    raw_mapping = generate_sender_receiver_mapping(num_receivers_per_sender=_NUM_RECV_PER_BANK)
    left_col = bh_cfg["sender_cols"]["left"]
    right_col = bh_cfg["sender_cols"]["right"]
    ordered_senders = [(left_col, y) for y in bh_cfg["bank_ordered_y_coords"]["left"]] + [
        (right_col, y) for y in bh_cfg["bank_ordered_y_coords"]["right"]
    ]
    assert len(ordered_senders) == num_senders, f"want {num_senders} senders, got {len(ordered_senders)}"
    sender_receiver_mapping = [
        (
            ttnn.CoreCoord(sx, sy),
            ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(rx, ry), ttnn.CoreCoord(rx, ry)) for rx, ry in raw_mapping[(sx, sy)]]
            ),
        )
        for sx, sy in ordered_senders
    ]
    sender_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(sx, sy)) for sx, sy in ordered_senders]
    )
    receiver_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(rx, ry), ttnn.CoreCoord(rx, ry))
            for sx, sy in ordered_senders
            for rx, ry in raw_mapping[(sx, sy)]
        ]
    )
    # Pin the workload to senders/receivers via sub-devices; without this the consumer
    # would land on the (0..7, 0..7) logical rect and collide with dispatch cores.
    sender_sub_device = ttnn.SubDevice([sender_core_range_set])
    receiver_sub_device = ttnn.SubDevice([receiver_core_range_set])
    sub_device_manager = device.create_sub_device_manager([sender_sub_device, receiver_sub_device], 0)
    device.load_sub_device_manager(sub_device_manager)
    receiver_sub_device_id = ttnn.SubDeviceId(1)
    device.set_sub_device_stall_group([receiver_sub_device_id])

    # Worker prefetcher validates max_tensor_size <= gcb.size(); size GCB to one full
    # per-receiver tensor (= pages_per_layer * page_size).
    gcb_size = pages_per_layer * page_size
    gcb = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, gcb_size)

    tt_weight = _build_weight(device, num_senders)

    # tensor_addrs: per-sender per-layer addresses (worker reader reads from addrs_cb).
    addr_row = torch.tensor([tt_weight.buffer_address()] * num_prefetch_layers, dtype=torch.int32).reshape(
        1, num_prefetch_layers
    )
    tensor_addrs = addr_row.repeat(num_senders, 1)
    tt_addrs = ttnn.as_tensor(
        tensor_addrs,
        device=device,
        dtype=ttnn.uint32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(sender_core_range_set, [1, num_prefetch_layers], ttnn.ShardOrientation.ROW_MAJOR),
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Launch the prefetcher exactly once for the whole bench (mirrors the DRAM-core
    # path's `start_tensor_prefetcher` call). One op invocation pushes
    # num_prefetch_layers tensors; the consumer drains them in pages_per_layer chunks.
    ttnn.dram_prefetcher(
        [tt_weight, tt_addrs], num_layers=num_prefetch_layers, global_cb=gcb, enable_performance_mode=True
    )

    # Warmup: drain 1 layer's worth of pages (also primes the cached consumer workload).
    ttnn.experimental.test_dram_prefetcher_consumer(
        device, num_iters=pages_per_layer, page_size_bytes=page_size, global_cb=gcb
    )
    ttnn.synchronize_device(device)

    bench_trace = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(trace_repeats):
        ttnn.experimental.test_dram_prefetcher_consumer(
            device, num_iters=pages_per_layer, page_size_bytes=page_size, global_cb=gcb
        )
    ttnn.end_trace_capture(device, bench_trace, cq_id=0)

    t0 = time.perf_counter()
    ttnn.execute_trace(device, bench_trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0
    ttnn.release_trace(device, bench_trace)

    bytes_per_recv = trace_repeats * pages_per_layer * page_size
    bytes_total = bytes_per_recv * _RING_SIZE
    bw_total = _gbps(bytes_total, elapsed)
    bw_per_recv = bw_total / _RING_SIZE
    logger.info(
        f"[workercore_bw][{op_name}] trace_elapsed={elapsed * 1e3:.2f}ms "
        f"bytes_per_recv={bytes_per_recv / 1e6:.1f}MB total_bytes={bytes_total / 1e9:.2f}GB "
        f"aggregate_bw={bw_total:.2f} GB/s per_recv_bw={bw_per_recv:.3f} GB/s"
    )
