# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Benchmark: DRAM-core prefetcher + matmul throughput on Blackhole.

Uses the `num_kernel_repeats` knob (default 1) on
`MatmulMultiCoreReuseMultiCast1DProgramConfig` to collapse N matmuls into a single op
invocation: the three gather_in0 matmul kernels (in0 reader, in1 reader, compute) wrap
their per-batch loop in an outer `for (r = 0; r < num_kernel_repeats; ++r)` loop, and the
DRAM-core prefetcher's `num_layers` is set to the same value so it pushes the weight N
times. One op launch -> N matmuls -> op-launch overhead amortized.

Slow dispatch only (the DRAM-core prefetcher doesn't run under fast dispatch yet, so
trace replay isn't an option for the prototype).

Shape matches the proven `ff_widest` case from `test_prefetcher_BH_dram_core_large`:
K=512, N=1024, bf16, ring=8 (8 DRAM banks * 1 receiver/bank). Llama-3.1-8B's FF1 shape
won't fit per-receiver L1 at ring=8; production runs it at ring=24+.

Use BENCH_REPEATS to set the repeat count (default 1000).
"""

import math
import os
import time
import pytest
import torch
import ttnn
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def _dram_programmable_enabled() -> bool:
    return os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0") == "1"


def _bench_repeats() -> int:
    return int(os.environ.get("BENCH_REPEATS", "1000"))


def _round_up(n, m):
    return ((n + m - 1) // m) * m


_M = 32
_K = 512  # K_tiles=16
_N = 1024  # N_tiles_per_receiver = 4 (1024/8/32)
_RING_SIZE = 8


def _build_program_config(num_kernel_repeats: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    in0_block_w = 1  # The DRISC prefetcher factory hard-codes in0_block_w_tiles=1.
    out_block_h = _M // ttnn.TILE_SIZE
    out_block_w = _N // _RING_SIZE // ttnn.TILE_SIZE
    out_subblock_w = min(out_block_w, 8)
    while out_subblock_w > 1 and out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(_RING_SIZE, 1),
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
        num_global_cb_receivers=1,
        untilize_out=False,
        num_kernel_repeats=num_kernel_repeats,
    )


def _flops_per_matmul() -> int:
    return 2 * _M * _K * _N


@pytest.mark.skipif(
    not _dram_programmable_enabled(), reason="TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set"
)
def test_bench_dram_core_repeats(device):
    """Time `num_kernel_repeats` matmuls collapsed into one op invocation."""
    arch = getattr(device, "arch", lambda: None)()
    if arch is not None and "BLACKHOLE" not in str(arch).upper():
        pytest.skip("DRAM-core prefetcher requires Blackhole")

    ttnn.device.enable_asynchronous_slow_dispatch(device)

    num_kernel_repeats = _bench_repeats()
    num_dram_banks = _RING_SIZE
    num_receivers_per_bank = 1

    receiver_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_RING_SIZE - 1, 0))}
    )

    torch.manual_seed(0xBE7)
    pt_weight = torch.randn(1, 1, _K, _N)
    pt_act = torch.randn(1, 1, _M, _K)

    dram_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
    )
    weight_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [_K, _N // num_dram_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_weight = ttnn.as_tensor(
        pt_weight, device=device, dtype=ttnn.bfloat16, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
    )

    K_per_shard = _round_up(math.ceil(_K / _RING_SIZE), ttnn.TILE_SIZE)
    act_mem_config = ttnn.create_sharded_memory_config(
        shape=(_M, K_per_shard),
        core_grid=receiver_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_act = ttnn.from_torch(
        pt_act, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=act_mem_config
    )

    # tensor_addrs is unused by the DRAM-core path but required by op contract.
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

    block_size_bytes = (_K * (_N // num_dram_banks) // num_receivers_per_bank) * 2  # bf16
    gcb_size = _round_up(block_size_bytes * 4, 4096)
    bank_to_receivers = [
        (b, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0))}))
        for b in range(num_dram_banks)
    ]
    gcb = ttnn.create_dram_sender_global_circular_buffer(device, bank_to_receivers, gcb_size)

    program_config = _build_program_config(num_kernel_repeats)
    output_mem_config = ttnn.create_sharded_memory_config(
        shape=(_M, _N // _RING_SIZE),
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

    logger.info(f"[bench] K={_K} N={_N} ring={_RING_SIZE} gcb_size={gcb_size} repeats={num_kernel_repeats}")

    # Correctness: single-repeat config first.
    cc_config = _build_program_config(num_kernel_repeats=1)
    ttnn.dram_prefetcher([tt_weight, addrs], num_layers=1, run_on_dram_cores=True, dram_sender_global_cb=gcb)
    cc_out = ttnn.linear(
        tt_act,
        tt_weight,
        program_config=cc_config,
        memory_config=output_mem_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
        dram_sender_global_cb=gcb,
    )
    cc_torch = ttnn.to_torch(cc_out)
    expected = pt_act.float() @ pt_weight.float()
    passing, output_str = comp_pcc(expected, cc_torch, 0.99)
    logger.info(f"[bench] PCC (repeats=1): {output_str}")
    assert passing, f"[bench] PCC failed: {output_str}"

    def run_once():
        ttnn.dram_prefetcher(
            [tt_weight, addrs], num_layers=num_kernel_repeats, run_on_dram_cores=True, dram_sender_global_cb=gcb
        )
        return ttnn.linear(
            tt_act,
            tt_weight,
            program_config=program_config,
            memory_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            dram_sender_global_cb=gcb,
        )

    # Warmup + 3 timed runs.
    run_once()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(3):
        run_once()
    ttnn.synchronize_device(device)
    elapsed = (time.perf_counter() - t0) / 3
    per_matmul_us = elapsed / num_kernel_repeats * 1e6
    tflops = _flops_per_matmul() * num_kernel_repeats / elapsed / 1e12
    logger.info(
        f"[bench] elapsed={elapsed * 1e3:.2f}ms/op repeats={num_kernel_repeats} "
        f"per_matmul={per_matmul_us:.2f}us -> {tflops:.4f} TFLOP/s"
    )
