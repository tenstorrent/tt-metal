# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Broadcast Tensor prefetcher vs. plain mcast-in1, same matmul.

Both arms run the identical `mcast_in1` matmul on the identical worker line with the identical
program config; only where in1 comes from differs:

- **baseline**: in1 is INTERLEAVED in DRAM. The first worker reads each K-block over the NoC into
  its own CB and multicasts it on-grid to the rest. The read sits on the critical path.
- **prefetcher**: in1 is a single-shard NdShardSpec on one DRAM bank. That bank's DRISC reads it
  from GDDR and multicasts each K-block straight into every worker's GCB slot, off the command
  queue, so the read overlaps compute.

Timing follows test_prefetcher_BH_bench.py: capture a trace of `trace_repeats` cached matmul
dispatches, replay it once, and wall-clock the replay. That takes host dispatch out of the
measurement and reports steady-state device time per matmul.
"""

import time

import pytest
import torch
import ttnn
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    bytes_per_tile as _bytes_per_tile,
    make_broadcast_weight as _make_broadcast_weight,
    tensor_prefetcher_session,
)

# A broadcast page is `in0_block_w` K-rows of the FULL weight width, and the DRISC fit ladder
# cannot split below one K-row, so `N_tiles * tile_bytes` must fit one stage third (~23.6 KB).
# That caps N at ~11 tiles for bf16 and ~21 for bf8_b -- see the module docstring note.
#
# (name, M_tiles_per_worker, K_tiles, N_tiles, dtype)
BENCH_SHAPES = [
    ("m2_k32_n8_bf16", 2, 32, 8, ttnn.bfloat16),
    ("m6_k32_n8_bf16", 6, 32, 8, ttnn.bfloat16),
    ("m2_k64_n8_bf16", 2, 64, 8, ttnn.bfloat16),
    ("m4_k32_n16_bf8", 4, 32, 16, ttnn.bfloat8_b),
    # Weight-size sweep at fixed M/K, to separate the baseline's in1 read cost from
    # whatever fixed cost it carries.
    ("m2_k32_n1_bf16", 2, 32, 1, ttnn.bfloat16),
    ("m2_k32_n2_bf16", 2, 32, 2, ttnn.bfloat16),
    ("m2_k32_n4_bf16", 2, 32, 4, ttnn.bfloat16),
]

NUM_WORKERS = 8
TRACE_REPEATS = 64


def _flops(M, K, N):
    return 2.0 * M * K * N


def _build_common(device, per_core_M_tiles, k_tiles, n_tiles, in0_block_w, dtype, seed):
    """Shapes, activation and program config shared by both arms."""
    M = NUM_WORKERS * per_core_M_tiles * ttnn.TILE_SIZE
    K = k_tiles * ttnn.TILE_SIZE
    N = n_tiles * ttnn.TILE_SIZE
    worker_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NUM_WORKERS - 1, 0))})

    torch.manual_seed(seed)
    pt_weight = torch.randn(1, 1, K, N)
    pt_act = torch.randn(1, 1, M, K)

    act_mem_config = ttnn.create_sharded_memory_config(
        shape=(per_core_M_tiles * ttnn.TILE_SIZE, K),
        core_grid=worker_cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_act = ttnn.from_torch(pt_act, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=act_mem_config)
    out_mem_config = ttnn.create_sharded_memory_config(
        shape=(per_core_M_tiles * ttnn.TILE_SIZE, N),
        core_grid=worker_cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out_subblock_w = min(n_tiles, 4)
    while n_tiles % out_subblock_w != 0:
        out_subblock_w -= 1
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(NUM_WORKERS, 1),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_M_tiles,
        out_block_w=n_tiles,
        per_core_M=per_core_M_tiles,
        per_core_N=n_tiles,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=False,
        hop_cores=ttnn.CoreRangeSet([]),
        num_global_cb_receivers=1,
        untilize_out=False,
        stream_in1=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )
    return (
        M,
        K,
        N,
        worker_cores,
        pt_weight,
        pt_act,
        tt_act,
        out_mem_config,
        program_config,
        compute_kernel_config,
    )


TRACE_EXECUTIONS = 4


def _time_trace(device, run_one, trace_repeats, executions=TRACE_EXECUTIONS):
    """Capture `trace_repeats` cached dispatches and time steady-state replay.

    The FIRST execute_trace of a given trace carries a large one-off cost (several ms here), so
    timing a single replay charges that to the matmuls and wildly inflates per-matmul time at small
    repeat counts. Execute once to warm, then time `executions` further replays.
    """
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    out = None
    for _ in range(trace_repeats):
        out = run_one()  # named reference keeps the output buffer alive
    ttnn.end_trace_capture(device, trace, cq_id=0)
    assert out is not None

    # Warm: pay the one-off first-replay cost outside the measurement.
    ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(executions):
        ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed = (time.perf_counter() - t0) / executions
    ttnn.release_trace(device, trace)
    return elapsed


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("op_name,per_core_M_tiles,k_tiles,n_tiles,dtype", BENCH_SHAPES)
@pytest.mark.parametrize("in0_block_w", [1, 2, 4, 8], ids=lambda w: f"bw{w}")
@pytest.mark.parametrize("gcb_pages", [2, 8], ids=lambda n: f"depth{n}")
def test_broadcast_mcast_in1_vs_plain(
    device, op_name, per_core_M_tiles, k_tiles, n_tiles, dtype, in0_block_w, gcb_pages
):
    if k_tiles % in0_block_w != 0:
        pytest.skip(f"K ({k_tiles} tiles) not divisible by in0_block_w={in0_block_w}")
    # depth 2 is the minimum window and matches the baseline's double-buffered in1 CB byte for
    # byte -- the equal-L1 point. Deeper trades L1 for round batching. Cap the total so the GCB
    # cannot crowd out in0/out.
    if gcb_pages * in0_block_w * n_tiles * _bytes_per_tile(dtype) > 256 * 1024:
        pytest.skip("GCB would exceed the 256 KiB budget for this shape")
    (
        M,
        K,
        N,
        worker_cores,
        pt_weight,
        pt_act,
        tt_act,
        out_mem_config,
        program_config,
        compute_kernel_config,
    ) = _build_common(device, per_core_M_tiles, k_tiles, n_tiles, in0_block_w, dtype, seed=len(op_name))

    expected = pt_act.float() @ pt_weight.float()
    weight_bytes = k_tiles * n_tiles * _bytes_per_tile(dtype)

    # ---- Arm A: plain mcast-in1, in1 interleaved in DRAM, relayed on-grid by worker 0. ----
    tt_weight_dram = ttnn.from_torch(
        pt_weight, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    def plain_linear(out=None):
        return ttnn.linear(
            tt_act,
            tt_weight_dram,
            program_config=program_config,
            memory_config=out_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
            optional_output_tensor=out,
        )

    # Allocate the reusable output, then warm the *exact* variant the trace will replay -- passing
    # optional_output_tensor changes the program hash, so warming without it leaves the traced
    # variant uncached and trace capture refuses to load new binaries.
    plain_out = plain_linear()
    plain_linear(plain_out)
    passing, out_str = comp_pcc(expected, ttnn.to_torch(plain_out), 0.99)
    assert passing, f"[{op_name}] baseline PCC failed: {out_str}"
    plain_elapsed = _time_trace(device, lambda: plain_linear(plain_out), TRACE_REPEATS)
    # Every traced iteration rewrites the same buffer, so a correct replay leaves a correct result.
    # Without this a silently-degenerate arm (e.g. consuming stale pages) would just look fast.
    passing, out_str = comp_pcc(expected, ttnn.to_torch(plain_out), 0.99)
    assert passing, f"[{op_name}] baseline post-trace PCC failed: {out_str}"

    # ---- Arm B: broadcast Tensor prefetcher, in1 a single shard on one bank, DRISC-multicast. ----
    tt_weight_bcast = _make_broadcast_weight(device, pt_weight, dtype=dtype)
    page_bytes = in0_block_w * n_tiles * _bytes_per_tile(dtype)
    gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
        device,
        [program_config],
        [tt_weight_bcast],
        bank_to_receivers=[(0, worker_cores)],
        size=gcb_pages * page_bytes,
    )
    block_count = k_tiles // in0_block_w

    def gcb_linear(out=None):
        return ttnn.linear(
            tt_act,
            tt_weight_bcast,
            program_config=program_config,
            global_cb=gcb,
            memory_config=out_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
            optional_output_tensor=out,
        )

    with tensor_prefetcher_session(device):
        # One long-lived stream: 2 warmup layers + one layer per traced matmul per replay, queued
        # up front so the prefetcher runs ahead rather than being driven by the matmuls.
        ttnn.experimental.queue_tensor_prefetcher_request(
            device,
            [(tt_weight_bcast, block_count)] * (2 + (1 + TRACE_EXECUTIONS) * TRACE_REPEATS),
            global_cb=gcb,
        )
        gcb_out = gcb_linear()
        gcb_linear(gcb_out)  # warm the traced variant (see the baseline note above)
        passing, out_str = comp_pcc(expected, ttnn.to_torch(gcb_out), 0.99)
        assert passing, f"[{op_name}] prefetcher PCC failed: {out_str}"
        gcb_elapsed = _time_trace(device, lambda: gcb_linear(gcb_out), TRACE_REPEATS)
        passing, out_str = comp_pcc(expected, ttnn.to_torch(gcb_out), 0.99)
        assert passing, f"[{op_name}] prefetcher post-trace PCC failed: {out_str}"

    plain_us = plain_elapsed / TRACE_REPEATS * 1e6
    gcb_us = gcb_elapsed / TRACE_REPEATS * 1e6
    flops = _flops(M, K, N)
    logger.info(
        f"[bcast_bench][{op_name} bw={in0_block_w} depth={gcb_pages}] M={M} K={K} N={N} "
        f"workers={NUM_WORKERS} "
        f"weight={weight_bytes / 1024:.0f}KiB\n"
        f"    plain mcast_in1 : {plain_us:8.2f} us/matmul  "
        f"{flops * TRACE_REPEATS / plain_elapsed / 1e12:6.3f} TFLOP/s  "
        f"{weight_bytes * TRACE_REPEATS / plain_elapsed / 1e9:6.2f} GB/s in1\n"
        f"    prefetcher bcast: {gcb_us:8.2f} us/matmul  "
        f"{flops * TRACE_REPEATS / gcb_elapsed / 1e12:6.3f} TFLOP/s  "
        f"{weight_bytes * TRACE_REPEATS / gcb_elapsed / 1e9:6.2f} GB/s in1\n"
        f"    speedup         : {plain_us / gcb_us:6.3f}x"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 90000000}],
    indirect=True,
)
@pytest.mark.parametrize("trace_repeats", [1, 4, 16, 64, 128], ids=lambda n: f"rep{n}")
def test_repeat_count_amortization(device, trace_repeats):
    """Is one trace of N matmuls enough to amortize per-dispatch launch cost?

    If per-matmul time keeps falling as N grows, the headline numbers are polluted by a fixed
    per-trace/per-launch cost. If it flattens, the measured time really is per matmul.
    """
    op_name, per_core_M_tiles, k_tiles, n_tiles, dtype = "m2_k32_n8_bf16", 2, 32, 8, ttnn.bfloat16
    in0_block_w = 8
    (
        M,
        K,
        N,
        worker_cores,
        pt_weight,
        pt_act,
        tt_act,
        out_mem_config,
        program_config,
        compute_kernel_config,
    ) = _build_common(device, per_core_M_tiles, k_tiles, n_tiles, in0_block_w, dtype, seed=7)
    expected = pt_act.float() @ pt_weight.float()

    tt_weight_dram = ttnn.from_torch(
        pt_weight, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    def plain_linear(out=None):
        return ttnn.linear(
            tt_act,
            tt_weight_dram,
            program_config=program_config,
            memory_config=out_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
            optional_output_tensor=out,
        )

    plain_out = plain_linear()
    plain_linear(plain_out)
    plain_elapsed = _time_trace(device, lambda: plain_linear(plain_out), trace_repeats)
    passing, out_str = comp_pcc(expected, ttnn.to_torch(plain_out), 0.99)
    assert passing, f"baseline post-trace PCC failed: {out_str}"

    tt_weight_bcast = _make_broadcast_weight(device, pt_weight, dtype=dtype)
    page_bytes = in0_block_w * n_tiles * _bytes_per_tile(dtype)
    gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
        device,
        [program_config],
        [tt_weight_bcast],
        bank_to_receivers=[(0, worker_cores)],
        size=2 * page_bytes,
    )

    def gcb_linear(out=None):
        return ttnn.linear(
            tt_act,
            tt_weight_bcast,
            program_config=program_config,
            global_cb=gcb,
            memory_config=out_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
            optional_output_tensor=out,
        )

    with tensor_prefetcher_session(device):
        ttnn.experimental.queue_tensor_prefetcher_request(
            device,
            [(tt_weight_bcast, k_tiles // in0_block_w)] * (2 + (1 + TRACE_EXECUTIONS) * trace_repeats),
            global_cb=gcb,
        )
        gcb_out = gcb_linear()
        gcb_linear(gcb_out)
        gcb_elapsed = _time_trace(device, lambda: gcb_linear(gcb_out), trace_repeats)
        passing, out_str = comp_pcc(expected, ttnn.to_torch(gcb_out), 0.99)
        assert passing, f"prefetcher post-trace PCC failed: {out_str}"

    logger.info(
        f"[amortization] repeats={trace_repeats:4d}  "
        f"plain={plain_elapsed / trace_repeats * 1e6:8.2f} us/matmul  "
        f"prefetcher={gcb_elapsed / trace_repeats * 1e6:7.2f} us/matmul  "
        f"(trace totals: plain={plain_elapsed * 1e3:.2f}ms prefetcher={gcb_elapsed * 1e3:.2f}ms)"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 90000000}],
    indirect=True,
)
@pytest.mark.parametrize("gcb_pages", [2, 4, 8, 16], ids=lambda n: f"depth{n}")
@pytest.mark.parametrize("n_tiles", [1, 8], ids=lambda n: f"n{n}")
def test_gcb_depth_sensitivity(device, gcb_pages, n_tiles):
    """How much of the small-in0_block_w deficit is just a shallow GCB?

    The kernel batches B blocks per round, clamped by free space in the GCB. At the two-page
    minimum B is 2, so a 32-block tensor costs 16 rounds -- and each round pays a poll, a write
    barrier and a per-receiver credit update. A deeper window should collapse the round count.
    """
    per_core_M_tiles, k_tiles, in0_block_w, dtype = 2, 32, 1, ttnn.bfloat16
    (
        M,
        K,
        N,
        worker_cores,
        pt_weight,
        pt_act,
        tt_act,
        out_mem_config,
        program_config,
        compute_kernel_config,
    ) = _build_common(device, per_core_M_tiles, k_tiles, n_tiles, in0_block_w, dtype, seed=11)
    expected = pt_act.float() @ pt_weight.float()

    tt_weight_bcast = _make_broadcast_weight(device, pt_weight, dtype=dtype)
    page_bytes = in0_block_w * n_tiles * _bytes_per_tile(dtype)
    gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
        device,
        [program_config],
        [tt_weight_bcast],
        bank_to_receivers=[(0, worker_cores)],
        size=gcb_pages * page_bytes,
    )

    def gcb_linear(out=None):
        return ttnn.linear(
            tt_act,
            tt_weight_bcast,
            program_config=program_config,
            global_cb=gcb,
            memory_config=out_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
            optional_output_tensor=out,
        )

    with tensor_prefetcher_session(device):
        ttnn.experimental.queue_tensor_prefetcher_request(
            device,
            [(tt_weight_bcast, k_tiles // in0_block_w)] * (2 + (1 + TRACE_EXECUTIONS) * TRACE_REPEATS),
            global_cb=gcb,
        )
        out = gcb_linear()
        gcb_linear(out)
        elapsed = _time_trace(device, lambda: gcb_linear(out), TRACE_REPEATS)
        passing, out_str = comp_pcc(expected, ttnn.to_torch(out), 0.99)
        assert passing, f"depth={gcb_pages} PCC failed: {out_str}"

    logger.info(
        f"[gcb_depth] n_tiles={n_tiles} pages={gcb_pages:3d} ({gcb_pages * page_bytes / 1024:.0f}KiB)  "
        f"prefetcher={elapsed / TRACE_REPEATS * 1e6:7.2f} us/matmul"
    )
