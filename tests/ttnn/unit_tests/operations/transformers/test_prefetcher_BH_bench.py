# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Benchmark: DRAM-core prefetcher vs worker-core prefetcher on Blackhole.

Both paths share the same gather_in0 matmul kernels. Uses the `num_kernel_repeats` knob
(default 1) on `MatmulMultiCoreReuseMultiCast1DProgramConfig` to collapse N matmuls into
a single op invocation: the three gather_in0 matmul kernels wrap their per-batch loop in
an outer `for (r = 0; r < num_kernel_repeats; ++r)` loop, and the prefetcher's
`num_layers` is set to the same value so it pushes the weight N times. One op launch ->
N matmuls -> op-launch overhead amortized.

Slow dispatch only (the DRAM-core prefetcher doesn't run under fast dispatch yet, so
trace replay isn't an option). SubDevice managers also don't work under slow dispatch,
so the worker-core case runs senders + receivers on the default sub-device.

Shape: K=512, N=1024, bf16, ring=8 (matches the proven `ff_widest` case from
`test_prefetcher_BH_dram_core_large`). Llama-3.1-8B's FF1 shape doesn't fit per-receiver
L1 at ring=8; production runs it at ring=24+.

Use BENCH_REPEATS to set the repeat count (default 1000). For useful asymptotic numbers,
1000+ is recommended.

Measured (M=32, K=512, N=1024, ring=8, bf16, BH):
                    DRAM-core           Worker-core
  N=10:    188us, 0.18 TFLOP/s     190us, 0.18 TFLOP/s
  N=100:    29us, 1.17 TFLOP/s      25us, 1.34 TFLOP/s
  N=1000:   12us, 2.70 TFLOP/s     8.5us, 3.96 TFLOP/s
  N=10000:  11us, 3.10 TFLOP/s     6.8us, 4.95 TFLOP/s

Worker-core is ~60% faster at the asymptote on this shape. Op-launch overhead per
invocation is ~1.8 ms for both paths (the per-matmul cost at N=10 vs N=10000).
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
_K = int(os.environ.get("BENCH_K", "512"))
_N = int(os.environ.get("BENCH_N", "1024"))
_NUM_DRAM_BANKS = 8
_NUM_RECV_PER_BANK = int(os.environ.get("BENCH_RECV_PER_BANK", "1"))
_RING_SIZE = _NUM_DRAM_BANKS * _NUM_RECV_PER_BANK
_RING_COLS = 8
_RING_ROWS = (_RING_SIZE + _RING_COLS - 1) // _RING_COLS
assert _RING_SIZE == _RING_COLS * _RING_ROWS, f"ring_size {_RING_SIZE} not a clean rect on 8-col grid"
_DTYPE_NAME = os.environ.get("BENCH_DTYPE", "bfloat16")  # "bfloat16" or "bfloat8_b"
_DTYPE = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}[_DTYPE_NAME]
_DTYPE_BYTES = {"bfloat16": 2, "bfloat8_b": 1088 / 1024.0}[_DTYPE_NAME]  # avg bytes/elem (bf8_b includes header)


# Cap GCB to leave headroom in L1 for the matmul CBs (in1 double-buffered, in2 ring history,
# in0, out, interm0). The worker-core path also requires gcb_size >= one full per-receiver
# tensor, so we clamp from below to that.
_L1_BANK_HEADROOM_BYTES = 256 * 1024  # leave 256 KB for matmul + activation + output L1


def _gcb_size_capped(block_size_bytes: int, max_buffered_blocks: int = 4) -> int:
    # Matmul receiver needs ~block_size_bytes of L1 for in1_CB (the per-receiver weight slice
    # fully buffered for gather_in0). To fit GCB + in1_CB + other matmul CBs in ~1.4 MB L1,
    # the GCB itself can't exceed ~(1.4 MB - block_size_bytes - matmul overhead).
    upper_l1 = max(block_size_bytes, 1_400_000 - block_size_bytes - _L1_BANK_HEADROOM_BYTES)
    upper_cb_pages_bytes = 65000 * 16  # 16 B page * <65535 pages = ~1 MB
    upper = min(upper_l1, upper_cb_pages_bytes)
    desired = block_size_bytes * max_buffered_blocks
    sized = max(block_size_bytes, min(desired, upper))
    sized = _round_up(sized, 4096)
    if sized // 16 >= 65535:
        sized = (65000 * 16 // 4096) * 4096
    return sized


def _bank_receivers_row_major(bank_idx: int, recv_per_bank: int, row_offset: int = 0):
    """Return the CoreRangeSet for bank `bank_idx`'s receivers, laid out as the
    contiguous row-major slice of the ring at positions [b*recv_per_bank, (b+1)*recv_per_bank).
    `row_offset` shifts the whole rectangle down (used by worker-core to put senders on row 0)."""
    cores = []
    for k in range(recv_per_bank):
        ring_pos = bank_idx * recv_per_bank + k
        col = ring_pos % _RING_COLS
        row = ring_pos // _RING_COLS + row_offset
        cores.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
    return ttnn.CoreRangeSet(cores)


def _build_program_config(num_kernel_repeats: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    in0_block_w = 1  # The DRISC prefetcher factory hard-codes in0_block_w_tiles=1.
    out_block_h = _M // ttnn.TILE_SIZE
    out_block_w = _N // _RING_SIZE // ttnn.TILE_SIZE
    out_subblock_w = min(out_block_w, 8)
    while out_subblock_w > 1 and out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(_RING_COLS, _RING_ROWS),
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
    num_dram_banks = _NUM_DRAM_BANKS
    num_receivers_per_bank = _NUM_RECV_PER_BANK

    # Receivers laid out as a row-major rectangle (_RING_COLS x _RING_ROWS) on worker rows
    # 0..(_RING_ROWS-1). gather_in0 walks worker_cores_vec in row-major order, so bank b's
    # receivers must be at row-major-adjacent ring positions [b*recv_per_bank, (b+1)*recv_per_bank).
    receiver_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_RING_COLS - 1, _RING_ROWS - 1))}
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
        pt_weight, device=device, dtype=_DTYPE, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
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

    # bytes/elem: bf16=2, bf8_b≈1.0625 (1088B/tile / 1024 elems/tile)
    block_size_bytes = int((_K * (_N // num_dram_banks) // num_receivers_per_bank) * _DTYPE_BYTES)
    gcb_size = _gcb_size_capped(block_size_bytes)
    bank_to_receivers = [(b, _bank_receivers_row_major(b, num_receivers_per_bank)) for b in range(num_dram_banks)]
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
        f"[dram_core] elapsed={elapsed * 1e3:.2f}ms/op repeats={num_kernel_repeats} "
        f"per_matmul={per_matmul_us:.2f}us -> {tflops:.4f} TFLOP/s"
    )


def test_bench_workercore_repeats(device):
    """A/B baseline: same matmul + num_kernel_repeats but with the worker-core prefetcher
    (BRISC+NCRISC senders on worker cores instead of DRISC on DRAM cores). Same shape,
    same dtype, same num_kernel_repeats as the DRAM-core bench. SubDevice managers aren't
    supported under slow dispatch so we run without sub_device_id; senders + receivers
    share the default sub-device.
    """
    arch = getattr(device, "arch", lambda: None)()
    if arch is not None and "BLACKHOLE" not in str(arch).upper():
        pytest.skip("Bench tuned for Blackhole topology")

    ttnn.device.enable_asynchronous_slow_dispatch(device)

    num_kernel_repeats = _bench_repeats()
    num_senders = _NUM_DRAM_BANKS

    # Receivers as row-major _RING_COLS x _RING_ROWS rectangle on rows 0..(_RING_ROWS-1).
    # Senders on row _RING_ROWS, one per DRAM bank, in column order.
    receiver_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_RING_COLS - 1, _RING_ROWS - 1))}
    )
    sender_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, _RING_ROWS), ttnn.CoreCoord(num_senders - 1, _RING_ROWS))}
    )

    # Sender s sends to receivers at row-major-adjacent ring positions [s*recv_per, (s+1)*recv_per).
    sender_receiver_mapping = [
        (ttnn.CoreCoord(s, _RING_ROWS), _bank_receivers_row_major(s, _NUM_RECV_PER_BANK)) for s in range(num_senders)
    ]
    # Per-sender, per-block bytes (each sender pushes its share of the weight).
    block_size_bytes = int((_K * (_N // num_senders) // _NUM_RECV_PER_BANK) * _DTYPE_BYTES)
    gcb_size = _gcb_size_capped(block_size_bytes)
    gcb = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, gcb_size)

    torch.manual_seed(0xBE8)
    pt_weight = torch.randn(1, 1, _K, _N)
    pt_act = torch.randn(1, 1, _M, _K)

    # Weight: width-sharded in DRAM across `num_senders` banks.
    dram_core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_senders - 1, 0))})
    weight_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [_K, _N // num_senders], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_weight = ttnn.as_tensor(
        pt_weight, device=device, dtype=_DTYPE, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
    )

    # tensor_addrs: worker-core BRISC reader expects num_layers addresses per sender (= same
    # tt_weight repeated for each iteration since the bench reuses the same matmul).
    def _build_tensor_addrs(n_addrs: int) -> ttnn.Tensor:
        addr_row = torch.tensor([tt_weight.buffer_address()] * n_addrs, dtype=torch.int32).reshape(1, n_addrs)
        tensor_addrs = addr_row.repeat(num_senders, 1)  # one row per sender, all rows identical
        return ttnn.as_tensor(
            tensor_addrs,
            device=device,
            dtype=ttnn.uint32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(sender_core_range_set, [1, n_addrs], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    tt_addrs_cc = _build_tensor_addrs(1)
    tt_addrs = _build_tensor_addrs(num_kernel_repeats)

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

    logger.info(f"[workercore] K={_K} N={_N} ring={_RING_SIZE} gcb_size={gcb_size} repeats={num_kernel_repeats}")

    # Correctness: single-repeat config first.
    cc_config = _build_program_config(num_kernel_repeats=1)
    ttnn.dram_prefetcher([tt_weight, tt_addrs_cc], num_layers=1, global_cb=gcb)
    cc_out = ttnn.linear(
        tt_act,
        tt_weight,
        program_config=cc_config,
        memory_config=output_mem_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
        global_cb=gcb,
    )
    cc_torch = ttnn.to_torch(cc_out)
    expected = pt_act.float() @ pt_weight.float()
    passing, output_str = comp_pcc(expected, cc_torch, 0.99)
    logger.info(f"[workercore] PCC (repeats=1): {output_str}")
    assert passing, f"[workercore] PCC failed: {output_str}"

    def run_once():
        ttnn.dram_prefetcher([tt_weight, tt_addrs], num_layers=num_kernel_repeats, global_cb=gcb)
        return ttnn.linear(
            tt_act,
            tt_weight,
            program_config=program_config,
            memory_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            global_cb=gcb,
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
        f"[workercore] elapsed={elapsed * 1e3:.2f}ms/op repeats={num_kernel_repeats} "
        f"per_matmul={per_matmul_us:.2f}us -> {tflops:.4f} TFLOP/s"
    )
