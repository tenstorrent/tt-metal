# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor prefetcher (DRAM-core DRISC sender) feeding a Python generic-op receiver, against the
same bytes pulled from DRAM by the worker cores themselves.

Shape defaults to the Kimi K2.6 gate weight, W[K=emb=7168, N=hidden=2048] bfp8, receiver-
contiguous ND-sharded over the 8 DRAM banks with ring = 8 * RECV_PER_BANK receivers (col b, rows
0..R-1 for bank b). One layer = the whole weight = 15.6 MB.

  GCB_K / GCB_N / GCB_RECV_PER_BANK / GCB_LAYERS / GCB_DUAL (1 = two DRISC senders per bank)
  GCB_INFLIGHT (baseline staging depth in tiles)

Run through the device lock:  scripts/run_safe_pytest.sh routed_expert_work/gcb_recv/test_gcb_generic_op.py -q -s
"""
import os
import time

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.prefetcher_common import make_recv_contig_weight
from tests.ttnn.utils_for_testing import comp_pcc

TILE = 32
TILE_BYTES = 1088  # bfp8
REMOTE_CB = 24
STAGE_CB = 0
NUM_BANKS = 8


def _env(name, default):
    return int(os.environ.get(name, default))


def _recv_cores(R):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NUM_BANKS - 1, R - 1))})


def _output_tensor(device, R, k_block_w, n_per_recv_tiles):
    """L1 HEIGHT_SHARDED, one page-sized shard per receiver core (row-major core order)."""
    ring = NUM_BANKS * R
    shard = [k_block_w * TILE, n_per_recv_tiles * TILE]
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_recv_cores(R), shard, ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        torch.zeros(1, 1, ring * shard[0], shard[1]),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
    )


def _expected_last_block(pt_w, R, k_block_w, n_per_recv_tiles, block_count):
    """Core (x, y) is bank x, slab y => ring position x + 8y under ROUND_ROBIN_1D; the last page of a
    batched layer is block block_count-1 of that receiver's columns."""
    ring = NUM_BANKS * R
    rows = k_block_w * TILE
    cols = n_per_recv_tiles * TILE
    out = torch.zeros(ring * rows, cols)
    for y in range(R):
        for x in range(NUM_BANKS):
            core_idx = y * NUM_BANKS + x
            pos = x + NUM_BANKS * y
            out[core_idx * rows : (core_idx + 1) * rows] = pt_w[
                (block_count - 1) * rows : block_count * rows, pos * cols : (pos + 1) * cols
            ]
    return out


def _time_trace(device, fn):
    fn()  # warmup: JIT + cached workload (kernel binary write is illegal inside a trace)
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    t0 = time.perf_counter()
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    dt = time.perf_counter() - t0
    ttnn.release_trace(device, tid)
    return dt


@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 << 20}], indirect=True)
def test_gcb_generic_op(device):
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip("programmable DRAM cores unavailable")
    if device.dram_grid_size().x != NUM_BANKS:
        pytest.skip("expects 8 DRAM banks")
    K, N, R, layers = _env("GCB_K", 7168), _env("GCB_N", 2048), _env("GCB_RECV_PER_BANK", 4), _env("GCB_LAYERS", 8)
    ring = NUM_BANKS * R
    k_tiles, n_tiles = K // TILE, N // TILE
    assert k_tiles % ring == 0 and n_tiles % ring == 0, (k_tiles, n_tiles, ring)
    k_block_w = k_tiles // ring
    n_per_recv_tiles = n_tiles // ring
    page = k_block_w * n_per_recv_tiles * TILE_BYTES
    pages_per_layer = ring  # block_count == ring in batched mode
    layer_bytes = k_tiles * n_tiles * TILE_BYTES
    recv_cores = _recv_cores(R)

    torch.manual_seed(0)
    pt_w = torch.randn(1, 1, K, N)
    w_nd = make_recv_contig_weight(device, pt_w, NUM_BANKS, ring, ttnn.bfloat8_b)
    pt_w_q = ttnn.to_torch(w_nd)[0, 0]  # bfp8-quantised reference
    out = _output_tensor(device, R, k_block_w, n_per_recv_tiles)
    expected = _expected_last_block(pt_w_q, R, k_block_w, n_per_recv_tiles, pages_per_layer)

    def rt_args(vals):
        rt = ttnn.RuntimeArgs()
        for x in range(NUM_BANKS):
            for y in range(R):
                rt[x][y] = vals(x, y)
        return rt

    # ---------------- prefetcher -> generic-op receiver ----------------
    bank_to_receivers = [
        (b, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, R - 1))}))
        for b in range(NUM_BANKS)
    ]
    dual = _env("GCB_DUAL", 1) == 1
    gcb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(
        device, bank_to_receivers, pages_per_layer * page, support_multi_receiver_shards=not dual
    )

    def consumer_pd(num_layers):
        cb = ttnn.CBDescriptor()
        cb.total_size = page
        cb.core_ranges = recv_cores
        cb.remote_format_descriptors = [
            ttnn.CBFormatDescriptor(buffer_index=REMOTE_CB, data_format=ttnn.bfloat8_b, page_size=page)
        ]
        cb.set_global_circular_buffer(gcb)
        kern = ttnn.KernelDescriptor(
            kernel_source="routed_expert_work/gcb_recv/gcb_recv_kernel.cpp",
            core_ranges=recv_cores,
            compile_time_args=[REMOTE_CB, num_layers * pages_per_layer, page, 1],
            runtime_args=rt_args(lambda x, y: [out.buffer_address()]),
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=[cb])

    ttnn.experimental.start_tensor_prefetcher(device)
    try:
        # 1 layer for the correctness drain, `layers` to warm the bench program (a trace may not
        # load new binaries), `layers` for the traced drain.
        ttnn.experimental.queue_tensor_prefetcher_request(
            device, [(w_nd, pages_per_layer)] * (1 + 2 * layers), global_cb=gcb
        )
        pd_warm = consumer_pd(1)
        pd_bench = consumer_pd(layers)
        ttnn.generic_op([w_nd, out], pd_warm)
        ttnn.synchronize_device(device)
        got = ttnn.to_torch(out)[0, 0]
        ok, pcc = comp_pcc(expected, got)
        logger.info(f"prefetcher last-page check: pcc={pcc} max_abs={(expected - got).abs().max().item():.4g}")
        assert ok, "prefetcher delivered a page that does not match block_count-1 of the receiver's columns"

        # Timed: ONE op draining `layers` layers (no per-op dispatch in the measurement).
        ttnn.generic_op([w_nd, out], pd_bench)  # warm the bench program's binaries
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        ttnn.generic_op([w_nd, out], pd_bench)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        dt_pf = time.perf_counter() - t0
        ttnn.release_trace(device, tid)
    finally:
        ttnn.experimental.stop_tensor_prefetcher(device)
    bw_pf = layers * layer_bytes / dt_pf / 1e9
    logger.info(f"PREFETCHER  ring={ring} dual={dual} layers={layers}: {dt_pf*1e6:.0f} us  {bw_pf:.1f} GB/s aggregate")

    # ---------------- baseline: worker cores read their own slice from DRAM ----------------
    inflight = _env("GCB_INFLIGHT", 32)
    slice_tiles = k_tiles * n_per_recv_tiles

    def reader_pd(weight, num_layers):
        stage = ttnn.CBDescriptor(
            total_size=inflight * TILE_BYTES,
            core_ranges=recv_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=STAGE_CB, data_format=ttnn.bfloat8_b, page_size=TILE_BYTES)
            ],
        )
        ct = [slice_tiles, TILE_BYTES, inflight, num_layers, STAGE_CB] + ttnn.TensorAccessorArgs(
            weight
        ).get_compile_time_args()
        # core (x, y) reads the columns of ring position x + 8y (same slice the prefetcher delivers to it)
        kern = ttnn.KernelDescriptor(
            kernel_source="routed_expert_work/gcb_recv/dram_read_kernel.cpp",
            core_ranges=recv_cores,
            compile_time_args=ct,
            runtime_args=rt_args(
                lambda x, y: [
                    weight.buffer_address(),
                    (x + NUM_BANKS * y) * n_per_recv_tiles,
                    out.buffer_address(),
                    k_block_w * n_per_recv_tiles,
                    n_tiles,
                    n_per_recv_tiles,
                ]
            ),
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=[stage])

    w_il = ttnn.from_torch(
        pt_w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    results = {}
    for name, weight in (("nd_sharded", w_nd), ("interleaved", w_il)):
        ttnn.generic_op([weight, out], reader_pd(weight, 1))
        ttnn.synchronize_device(device)
        got = ttnn.to_torch(out)[0, 0]
        ok, pcc = comp_pcc(expected, got)
        assert ok, f"baseline {name} slice mismatch pcc={pcc}"
        dt = _time_trace(device, lambda w=weight: ttnn.generic_op([w, out], reader_pd(w, layers)))
        results[name] = dt
        logger.info(
            f"DRAM-READ {name:11s} inflight={inflight} layers={layers}: {dt*1e6:.0f} us  {layers*layer_bytes/dt/1e9:.1f} GB/s aggregate"
        )
    print(
        f"RESULT K={K} N={N} ring={ring} layer={layer_bytes/1e6:.1f}MB x{layers}: prefetcher {bw_pf:.1f} GB/s | "
        + " | ".join(f"dram-read {k} {layers*layer_bytes/v/1e9:.1f} GB/s" for k, v in results.items()),
        flush=True,
    )
