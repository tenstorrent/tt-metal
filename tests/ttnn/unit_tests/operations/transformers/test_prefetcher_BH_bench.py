# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Benchmark: Tensor prefetcher vs worker-core prefetcher on Blackhole.

Parametrized over Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Llama-3.3-70B production
prefetcher matmuls (QKV / WO / FF1 / FF2). For models that need multi-device tensor
parallelism in production (`is_prefetcher_supported` returns False at lower device
counts), tensor shapes are computed at the smallest device count where the worker
prefetcher fits: 1B/3B at 1 device, 8B at 2 devices, 70B at 8 devices.

Both paths share the same gather_in0 matmul kernels and the same production scattered
receiver layout / receiver sub-device / stall group setup. The only difference is
which prefetcher is used:
- DRAM-core: DRISC kernel on DRAM cores, started once via
  `start_tensor_prefetcher(num_layers=N+1)` before the trace, stopped after.
- Worker-core: BRISC+NCRISC kernels on worker cores, dispatched once via
  `ttnn.dram_prefetcher(num_layers=N+1)` before the trace.

Both push N+1 layers (1 warmup + N traced); both traces contain only N matmul
launches. Sender/receiver layout from `models/tt_transformers/tt/prefetcher/prefetcher_config.yaml`
(production col-0/col-7 senders, scattered receivers); matmul pinned to receivers
via `sub_device_id`.

Shape envs `BENCH_K / BENCH_N / BENCH_DTYPE / BENCH_RECV_PER_BANK` override the
parametrize values for ad-hoc runs. `BENCH_TRACE_REPEATS` (default 100) controls the
trace replay length for both paths. All shapes use ring=64 (8 banks × 8
receivers/bank) and bfloat8_b.

Per-shape results: see `docs/tensor_prefetcher_bench_results.md`. The
table there was measured with the previous worker-core trace shape and is
pending re-measurement after the move to N discrete `ttnn.linear` launches.

Caveats:
- Both paths trace exactly N single matmul dispatches; the prefetcher runs once
  out-of-band ahead of the trace and is consumed by the warmup matmul + the N
  traced matmuls (production-faithful: each decoder layer is a separate dispatch).
- TFLOP/s uses unpadded (K, N) for both paths (the formula honors the actual matmul
  work; padding for ring divisibility doesn't count as useful flops).
"""

import math
import os
import time
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    round_up as _round_up,
    ring_grid_cols as _ring_grid_cols,
    make_recv_contig_weight as _make_recv_contig_weight,
)


pytestmark = run_for_blackhole("Tensor prefetcher requires Blackhole")


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    """Skip unless programmable DRAM cores are available on this device."""
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip("programmable DRAM cores unavailable (need Blackhole and firmware >= 19.12.0.0)")


def _select_num_dram_banks(available_banks: int) -> int:
    n_tiles = _N // ttnn.TILE_SIZE
    candidates = [b for b in range(available_banks, 0, -1) if n_tiles % b == 0 and _N % b == 0]
    return candidates[0]


def _select_num_receivers_per_bank(num_dram_banks: int) -> int:
    n_tiles_per_bank = (_N // num_dram_banks) // ttnn.TILE_SIZE
    candidates = [r for r in range(_NUM_RECV_PER_BANK, n_tiles_per_bank + 1) if n_tiles_per_bank % r == 0]
    return candidates[0]


_M = 32
_NUM_DRAM_BANKS = 8

# Module-level shape vars. Mutable: rewritten by `_apply_shape()` at the top of each test
# from the pytest parameter (with BENCH_* env overrides for ad-hoc runs). Helpers below
# read these as ambient module state so we don't have to thread `K, N, dtype, recv_per_bank`
# through every helper signature. `_N` is the **padded** N (up to ring*TILE_SIZE) used by
# all sizing math; `_N_ORIG` is the original N used only in the TFLOP/s formula.
_K = 512
_K_ORIG = 512
_N = 1024
_N_ORIG = 1024
_NUM_RECV_PER_BANK = 1
_RING_SIZE = _NUM_DRAM_BANKS * _NUM_RECV_PER_BANK
_RING_COLS = 8
_RING_ROWS = 1
_DTYPE_NAME = "bfloat16"
_DTYPE = ttnn.bfloat16
_DTYPE_BYTES = 2

_DTYPE_FROM_NAME = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}
_DTYPE_BYTES_FROM_NAME = {"bfloat16": 2, "bfloat8_b": 1088 / 1024.0}  # bf8_b includes per-tile header


# Production Llama prefetcher-fed matmul shapes on Blackhole (ring=64, bf8_b).
# Per-device shapes from tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH.py:
# QKV is N-sharded (qkv_size = head_dim*(2*n_kv_heads + n_heads)), WO is K-sharded
# (n_heads*head_dim), FF1/FF3 are N-sharded (hidden_dim), FF2 is K-sharded (hidden_dim).
# For models that don't fit single-device (Llama-3.1-8B, Llama-3.3-70B), shapes are
# computed for tensor parallelism = 2 devices (per the goal).
#
# Llama-3.2-1B  (1 dev): dim=2048, hidden_dim=8192,  n_heads=32, n_kv_heads=8, head_dim=64,  qkv=3072
# Llama-3.2-3B  (1 dev): dim=3072, hidden_dim=8192,  n_heads=24, n_kv_heads=8, head_dim=128, qkv=5120
# Llama-3.1-8B  (2 dev): dim=4096, hidden_dim=14336, n_heads=32, n_kv_heads=8, head_dim=128, qkv=6144
# Llama-3.3-70B (2 dev): dim=8192, hidden_dim=28672, n_heads=64, n_kv_heads=8, head_dim=128, qkv=10240
#
# All shapes use ring=64 (8 banks × 8 receivers/bank); N gets padded up to ring*TILE_SIZE in
# the test body when not already a multiple.
LLAMA_SHAPES = [
    # Llama-3.2-1B, single-device per-op shapes (no TP).
    pytest.param("1B_QKV", dict(K=2048, N=3072, dtype="bfloat8_b", recv_per_bank=8), id="1B_QKV"),
    pytest.param("1B_WO", dict(K=2048, N=2048, dtype="bfloat8_b", recv_per_bank=8), id="1B_WO"),
    pytest.param("1B_FF1", dict(K=2048, N=8192, dtype="bfloat8_b", recv_per_bank=8), id="1B_FF1"),
    pytest.param("1B_FF2", dict(K=8192, N=2048, dtype="bfloat8_b", recv_per_bank=8), id="1B_FF2"),
    # Llama-3.2-3B, single-device.
    pytest.param("3B_QKV", dict(K=3072, N=5120, dtype="bfloat8_b", recv_per_bank=8), id="3B_QKV"),
    pytest.param("3B_WO", dict(K=3072, N=3072, dtype="bfloat8_b", recv_per_bank=8), id="3B_WO"),
    pytest.param("3B_FF1", dict(K=3072, N=8192, dtype="bfloat8_b", recv_per_bank=8), id="3B_FF1"),
    pytest.param("3B_FF2", dict(K=8192, N=3072, dtype="bfloat8_b", recv_per_bank=8), id="3B_FF2"),
    # Llama-3.1-8B at 2 devices (single-device doesn't fit worker L1 budget).
    pytest.param("8B_QKV_2d", dict(K=4096, N=3072, dtype="bfloat8_b", recv_per_bank=8), id="8B_QKV_2d"),
    pytest.param("8B_WO_2d", dict(K=2048, N=4096, dtype="bfloat8_b", recv_per_bank=8), id="8B_WO_2d"),
    pytest.param("8B_FF1_2d", dict(K=4096, N=7168, dtype="bfloat8_b", recv_per_bank=8), id="8B_FF1_2d"),
    pytest.param("8B_FF2_2d", dict(K=7168, N=4096, dtype="bfloat8_b", recv_per_bank=8), id="8B_FF2_2d"),
    # Llama-3.3-70B at 8 devices (2/4 don't fit is_prefetcher_supported's 1 MB/core).
    # dim=8192, hidden=28672, n_heads=64, n_kv_heads=8, head_dim=128, qkv_size=10240.
    # Per-device: qkv N=10240/8=1280, wo K=64*128/8=1024, ff1 N=28672/8=3584, ff2 K=28672/8=3584.
    pytest.param("70B_QKV_8d", dict(K=8192, N=1280, dtype="bfloat8_b", recv_per_bank=8), id="70B_QKV_8d"),
    pytest.param("70B_WO_8d", dict(K=1024, N=8192, dtype="bfloat8_b", recv_per_bank=8), id="70B_WO_8d"),
    pytest.param("70B_FF1_8d", dict(K=8192, N=3584, dtype="bfloat8_b", recv_per_bank=8), id="70B_FF1_8d"),
    pytest.param("70B_FF2_8d", dict(K=3584, N=8192, dtype="bfloat8_b", recv_per_bank=8), id="70B_FF2_8d"),
]


def _apply_shape(shape: dict) -> None:
    """Set module globals from the parametrize dict, allowing BENCH_* env vars to override.

    Tests call this at the top of their body so all helpers below see the shape via ambient
    module state. Run with `BENCH_K=... BENCH_N=...` to override the parametrize values for
    ad-hoc one-off runs.

    `_N` is the padded N (up to ring*TILE_SIZE) used by all sizing math; `_N_ORIG` is the
    original N from the shape, used in the TFLOP/s formula so headline numbers reflect the
    actual matmul work (not the padded extra).
    """
    global _K, _K_ORIG, _N, _N_ORIG, _NUM_RECV_PER_BANK, _RING_SIZE, _RING_ROWS
    global _DTYPE_NAME, _DTYPE, _DTYPE_BYTES
    _K_ORIG = int(os.environ.get("BENCH_K", shape["K"]))
    _N_ORIG = int(os.environ.get("BENCH_N", shape["N"]))
    _NUM_RECV_PER_BANK = int(os.environ.get("BENCH_RECV_PER_BANK", shape["recv_per_bank"]))
    _RING_SIZE = _NUM_DRAM_BANKS * _NUM_RECV_PER_BANK
    assert _RING_SIZE % _RING_COLS == 0, f"ring_size {_RING_SIZE} not a clean rect on 8-col grid"
    _RING_ROWS = _RING_SIZE // _RING_COLS
    # Pad K and N to multiples of (ring_size * TILE_SIZE) so every ring receiver gets at
    # least one tile in each direction. Matches production's pad_n_to_ring_size() in
    # test_prefetcher_BH.py and the h_tiles_padded round-up in is_prefetcher_supported.
    _K = _round_up(_K_ORIG, _RING_SIZE * ttnn.TILE_SIZE)
    _N = _round_up(_N_ORIG, _RING_SIZE * ttnn.TILE_SIZE)
    _DTYPE_NAME = os.environ.get("BENCH_DTYPE", shape["dtype"])
    _DTYPE = _DTYPE_FROM_NAME[_DTYPE_NAME]
    _DTYPE_BYTES = _DTYPE_BYTES_FROM_NAME[_DTYPE_NAME]


# Cap GCB to leave headroom in L1 for the matmul CBs (in1 double-buffered, in2 ring history,
# in0, out, interm0). The worker-core path also requires gcb_size >= one full per-receiver
# tensor, so we clamp from below to that.
_L1_BANK_HEADROOM_BYTES = 256 * 1024  # leave 256 KB for matmul + activation + output L1


def _matmul_in1_block_size_bytes(k: int, ring_size: int) -> int:
    """Matmul's receiver fifo_page_size = in0_block_w * per_core_N * tile_bytes.
    in0_block_w (matmul-computed) = K_per_shard_tiles, per_core_N = N_per_recv_tiles."""
    k_per_shard_tiles = (k // ring_size) // ttnn.TILE_SIZE
    n_per_recv_tiles = (_N // ring_size) // ttnn.TILE_SIZE
    return k_per_shard_tiles * n_per_recv_tiles * int(_DTYPE_BYTES * ttnn.TILE_SIZE * ttnn.TILE_SIZE)


def _gcb_size_capped(block_size_bytes: int, k: int, ring_size: int, max_buffered_blocks: int = 4) -> int:
    # Matmul receiver needs ~block_size_bytes of L1 for in1_CB (the per-receiver weight slice
    # fully buffered for gather_in0). To fit GCB + in1_CB + other matmul CBs in ~1.4 MB L1,
    # the GCB itself can't exceed ~(1.4 MB - block_size_bytes - matmul overhead).
    # gcb_size is snapped (below) to a multiple of the matmul's receiver fifo_page_size
    # (= in1_block_size_bytes) so the per-receiver fifo holds a whole number of pages.
    in1_block_size = _matmul_in1_block_size_bytes(k, ring_size)
    upper_l1 = max(block_size_bytes, 1_400_000 - block_size_bytes - _L1_BANK_HEADROOM_BYTES)
    upper_cb_pages_bytes = 65000 * 16  # 16 B page * <65535 pages = ~1 MB
    upper = min(upper_l1, upper_cb_pages_bytes)
    desired = block_size_bytes * max_buffered_blocks
    sized = max(block_size_bytes, min(desired, upper))
    # Snap to a multiple of in1_block_size (rounding down so we stay under the L1 cap).
    sized = max(in1_block_size, (sized // in1_block_size) * in1_block_size)
    if sized // 16 >= 65535:
        sized = (65000 * 16 // in1_block_size) * in1_block_size
    return sized


def _bank_receivers_row_major(bank_idx: int, recv_per_bank: int, ring_cols: int, row_offset: int = 0):
    """Return the CoreRangeSet for bank `bank_idx`'s receivers, laid out as the
    contiguous row-major slice of the ring at positions [b*recv_per_bank, (b+1)*recv_per_bank).
    `row_offset` shifts the whole rectangle down (used by worker-core to put senders on row 0)."""
    cores = []
    for k in range(recv_per_bank):
        ring_pos = bank_idx * recv_per_bank + k
        col = ring_pos % ring_cols
        row = ring_pos // ring_cols + row_offset
        cores.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
    return ttnn.CoreRangeSet(cores)


def _build_program_config(
    ring_size: int,
    ring_cols: int,
    ring_rows: int,
    num_global_cb_receivers: int,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    in0_block_w = 1  # The DRISC prefetcher factory hard-codes in0_block_w_tiles=1.
    out_block_h = _M // ttnn.TILE_SIZE
    out_block_w = _N // ring_size // ttnn.TILE_SIZE
    out_subblock_w = min(out_block_w, 8)
    while out_subblock_w > 1 and out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(ring_cols, ring_rows),
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
        # Each bank's N-width is split across `_NUM_RECV_PER_BANK` matmul receivers.
        # Matmul sizes in1_CB as `N_per_bank_tiles / num_global_cb_receivers`, so this
        # must match the actual receiver count per bank, otherwise in1_CB is oversized
        # (causing OOM or sender stall via mismatched cb sizes).
        num_global_cb_receivers=num_global_cb_receivers,
        untilize_out=False,
    )


def _flops_per_matmul(k: int) -> int:
    # Use the unpadded K and N: padded zeros aren't useful flops. (Callers pass _K or
    # _K_padded depending on what they have at hand; either way swap in _K_ORIG.)
    return 2 * _M * _K_ORIG * _N_ORIG


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("op_name,shape", LLAMA_SHAPES)
def test_bench_dram_core_repeats(device, op_name, shape):
    """Trace replay of N single-matmul launches fed by the DRISC prefetcher.

    Parametrized over Llama-3.1-8B prefetcher-fed matmul shapes (FF1, QKV, O, FF2).
    BENCH_K / BENCH_N / BENCH_DTYPE / BENCH_RECV_PER_BANK env vars override per-parameter.
    """
    _apply_shape(shape)

    trace_repeats = int(os.environ.get("BENCH_TRACE_REPEATS", "100"))
    num_prefetch_layers = trace_repeats + 1  # 1 warmup + trace_repeats inside the trace
    if device.dram_grid_size().x != 8:
        pytest.skip("Production receiver layout requires 8 unharvested DRAM banks")
    num_dram_banks = _select_num_dram_banks(device.dram_grid_size().x)
    num_receivers_per_bank = _select_num_receivers_per_bank(num_dram_banks)
    ring_size = num_dram_banks * num_receivers_per_bank
    ring_cols = _ring_grid_cols(_NUM_DRAM_BANKS, ring_size)
    ring_rows = ring_size // ring_cols
    k_padded = _round_up(_K, ring_size * ttnn.TILE_SIZE)
    if num_dram_banks != _NUM_DRAM_BANKS or num_receivers_per_bank != _NUM_RECV_PER_BANK:
        pytest.skip(
            f"Production receiver layout requires {_NUM_DRAM_BANKS} banks x {_NUM_RECV_PER_BANK} recv/bank, "
            f"got {num_dram_banks} x {num_receivers_per_bank}"
        )

    # Production scattered receiver layout (matches test_bench_workercore_repeats) so the
    # matmul receiver/in0-gather NoC paths are identical between the two paths.
    from models.tt_transformers.tt.prefetcher import generate_sender_receiver_mapping, ARCH_CONFIG

    bh_cfg = ARCH_CONFIG["blackhole"]
    raw_mapping = generate_sender_receiver_mapping(num_receivers_per_sender=num_receivers_per_bank)
    left_y = bh_cfg["bank_ordered_y_coords"]["left"]
    right_y = bh_cfg["bank_ordered_y_coords"]["right"]
    left_col = bh_cfg["sender_cols"]["left"]
    right_col = bh_cfg["sender_cols"]["right"]
    ordered_senders = [(left_col, y) for y in left_y] + [(right_col, y) for y in right_y]
    # Group production receivers by y-row, then sort rows so each group of 8 receivers
    # occupies contiguous ring positions in the matmul's row-major walk. Then pair DRAM
    # bank i with the i-th sorted y-row: bank i's N-slice lands at ring positions
    # [i*recv_per_bank, (i+1)*recv_per_bank) — matching the matmul's expected layout.
    receivers_by_y: dict = {}
    for sx, sy in ordered_senders:
        receivers_by_y.setdefault(sy, []).extend(raw_mapping[(sx, sy)])
    sorted_ys = sorted(receivers_by_y.keys())
    assert len(sorted_ys) == num_dram_banks, f"want {num_dram_banks} y-rows, got {len(sorted_ys)}"
    receivers_per_y = [sorted(receivers_by_y[y]) for y in sorted_ys]  # row-major-sortable
    receiver_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(rx, ry), ttnn.CoreCoord(rx, ry)) for row in receivers_per_y for rx, ry in row]
    )

    # Pin the matmul to the receiver sub-device + stall group (matches worker-core test).
    receiver_sub_device = ttnn.SubDevice([receiver_core_range_set])
    sub_device_manager = device.create_sub_device_manager([receiver_sub_device], 0)
    device.load_sub_device_manager(sub_device_manager)
    receiver_sub_device_id = ttnn.SubDeviceId(0)
    device.set_sub_device_stall_group([receiver_sub_device_id])

    torch.manual_seed(0xBE7)
    pt_weight = torch.zeros(1, 1, k_padded, _N)
    pt_weight[:, :, :_K, :] = torch.randn(1, 1, _K, _N)
    pt_act = torch.zeros(1, 1, _M, k_padded)
    pt_act[:, :, :, :_K] = torch.randn(1, 1, _M, _K)

    dram_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
    )
    weight_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [k_padded, _N // num_dram_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_weight = ttnn.as_tensor(
        pt_weight, device=device, dtype=_DTYPE, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
    )

    K_per_shard = k_padded // ring_size
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
    block_size_bytes = int((k_padded * (_N // num_dram_banks) // num_receivers_per_bank) * _DTYPE_BYTES)
    # max_buffered_blocks=1 keeps GCB ~= one fifo-page, leaving room for matmul receiver
    # CBs in worker L1 (1.4 MB). Larger shapes (70B FF1/FF2) overflow at deeper GCBs.
    gcb_size = _gcb_size_capped(block_size_bytes, k=k_padded, ring_size=ring_size, max_buffered_blocks=1)
    # Pair DRAM bank i with the receivers of the i-th sorted y-row so bank i pushes to
    # ring positions [i*recv_per_bank, (i+1)*recv_per_bank) — required by the gather_in0
    # matmul's bank-to-receivers ordering assertion.
    bank_to_receivers = [
        (
            bank_idx,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(rx, ry), ttnn.CoreCoord(rx, ry)) for rx, ry in row]),
        )
        for bank_idx, row in enumerate(receivers_per_y)
    ]
    cc_program_config = _build_program_config(
        ring_size=ring_size,
        ring_cols=ring_cols,
        ring_rows=ring_rows,
        num_global_cb_receivers=num_receivers_per_bank,
    )
    gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
        device,
        [cc_program_config],
        [tt_weight],
        bank_to_receivers=bank_to_receivers,
        size=gcb_size,
    )
    output_mem_config = ttnn.create_sharded_memory_config(
        shape=(_M, _N // ring_size),
        core_grid=receiver_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    optional_output_tensor = ttnn.from_torch(
        torch.zeros(1, 1, _M, _N),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_mem_config,
    )

    def single_linear():
        return ttnn.linear(
            tt_act,
            tt_weight,
            program_config=cc_program_config,
            memory_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            global_cb=gcb,
            optional_output_tensor=optional_output_tensor,
            sub_device_id=receiver_sub_device_id,
        )

    # One long-lived DRISC stream: 1 warmup/correctness layer + trace_repeats traced layers.
    ttnn.experimental.start_tensor_prefetcher(device)
    ttnn.experimental.queue_tensor_prefetcher_request(
        device,
        [(tt_weight, ring_size)] * num_prefetch_layers,
        global_cb=gcb,
    )

    # Correctness + program-cache warmup: one real one-matmul launch.
    cc_out = single_linear()
    cc_torch = ttnn.to_torch(cc_out)
    expected = pt_act.float() @ pt_weight.float()
    passing, output_str = comp_pcc(expected, cc_torch, 0.99)
    logger.info(f"[bench] PCC: {output_str}")
    assert passing, f"[bench] PCC failed: {output_str}"

    # Capture a trace of `trace_repeats` cached single-matmul dispatches and replay it once.
    # The prefetcher feeds one layer per matmul, so num_prefetch_layers = 1 + trace_repeats.
    bench_trace = ttnn.begin_trace_capture(device, cq_id=0)
    bench_output = None
    for _ in range(trace_repeats):
        bench_output = single_linear()  # Keep a named reference so the output buffer stays alive.
    ttnn.end_trace_capture(device, bench_trace, cq_id=0)
    assert bench_output is not None  # trace_repeats > 0, so the loop ran at least once.

    t0 = time.perf_counter()
    ttnn.execute_trace(device, bench_trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0
    ttnn.release_trace(device, bench_trace)

    ttnn.experimental.stop_tensor_prefetcher(device)

    per_matmul_us = elapsed / trace_repeats * 1e6
    # Use unpadded K in the TFLOP/s formula so it's comparable across paths (worker-core uses
    # _K directly; padding to ring-aligned K is wasted work that doesn't count as useful flops).
    tflops = _flops_per_matmul(_K) * trace_repeats / elapsed / 1e12
    logger.info(
        f"[dram_core][{op_name}] trace_elapsed={elapsed * 1e3:.2f}ms repeats={trace_repeats} "
        f"per_matmul={per_matmul_us:.2f}us -> {tflops:.4f} TFLOP/s"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("op_name,shape", LLAMA_SHAPES)
@pytest.mark.parametrize(
    "distribution",
    [ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D, ttnn.ShardDistributionStrategy.CONTIGUOUS_1D],
    ids=["round_robin", "shard_contiguous"],
)
def test_bench_dram_core_repeats_recv_contig(device, op_name, shape, distribution):
    """Same matmul trace-replay bench as test_bench_dram_core_repeats, but the weight
    is in **receiver-contiguous** DRAM layout (NdShardSpec, num_shards = ring_size).

    Parametrized on the shard distribution to A/B the ring-routing effect:
      - round_robin: bank b feeds the STRIDED ring set [b, b+num_banks, ...] (column b
        of receivers_per_y) -- each bank's writes fan out across the whole ring.
      - shard-contiguous (CONTIGUOUS_1D): bank b feeds the CONTIGUOUS arc [b*R, b*R+R-1] (row b of
        receivers_per_y) -- each bank's writes stay on a local ring segment.
    The matmul topology (receiver cores, ring order, program config) is identical; only
    the shard->bank placement and the GCB pairing change, so PCC holds for both. Compare
    the logged GB/s across the two ids for the same shape.

    Why this is matmul-correct: in a gather_in0 matmul, ring core r computes output
    N-cols [r*per_core_N, (r+1)*per_core_N) and therefore needs weight[all K, those
    N-cols] = recv-contig shard r, streamed as ring_size K-blocks in order. The
    recv-contig prefetcher delivers exactly that. The only wiring difference vs the
    K-row-major bench:
      - Weight allocated via NdShardSpec(Shape([K_padded, n_per_recv]), ROUND_ROBIN_1D).
      - bank_to_receivers is the STRIDED transpose of the row-grouping: bank b feeds
        ring positions [b, b+num_banks, b+2*num_banks, ...]. Under BDS round-robin,
        bank b's slab s == shard (b + s*num_banks), so this delivers shard r to ring
        position r (PCC-verified below).
      - GCB built via create_global_circular_buffer_for_matmul_1d, which auto-detects the
        weight's DRAM layout (here receiver-contiguous NdShardSpec) and sizes/builds the GCB
        accordingly.
    """
    _apply_shape(shape)

    trace_repeats = int(os.environ.get("BENCH_TRACE_REPEATS", "100"))
    num_prefetch_layers = trace_repeats + 1
    if device.dram_grid_size().x != 8:
        pytest.skip("Production receiver layout requires 8 unharvested DRAM banks")
    num_dram_banks = _select_num_dram_banks(device.dram_grid_size().x)
    num_receivers_per_bank = _select_num_receivers_per_bank(num_dram_banks)
    ring_size = num_dram_banks * num_receivers_per_bank
    ring_cols = _ring_grid_cols(_NUM_DRAM_BANKS, ring_size)
    ring_rows = ring_size // ring_cols
    k_padded = _round_up(_K, ring_size * ttnn.TILE_SIZE)
    if num_dram_banks != _NUM_DRAM_BANKS or num_receivers_per_bank != _NUM_RECV_PER_BANK:
        pytest.skip(
            f"Production receiver layout requires {_NUM_DRAM_BANKS} banks x {_NUM_RECV_PER_BANK} recv/bank, "
            f"got {num_dram_banks} x {num_receivers_per_bank}"
        )

    # Production scattered receiver layout (identical to the K-row-major bench) so the
    # matmul receiver/in0-gather NoC paths match exactly.
    from models.tt_transformers.tt.prefetcher import generate_sender_receiver_mapping, ARCH_CONFIG

    bh_cfg = ARCH_CONFIG["blackhole"]
    raw_mapping = generate_sender_receiver_mapping(num_receivers_per_sender=num_receivers_per_bank)
    left_y = bh_cfg["bank_ordered_y_coords"]["left"]
    right_y = bh_cfg["bank_ordered_y_coords"]["right"]
    left_col = bh_cfg["sender_cols"]["left"]
    right_col = bh_cfg["sender_cols"]["right"]
    ordered_senders = [(left_col, y) for y in left_y] + [(right_col, y) for y in right_y]
    receivers_by_y: dict = {}
    for sx, sy in ordered_senders:
        receivers_by_y.setdefault(sy, []).extend(raw_mapping[(sx, sy)])
    sorted_ys = sorted(receivers_by_y.keys())
    assert len(sorted_ys) == num_dram_banks, f"want {num_dram_banks} y-rows, got {len(sorted_ys)}"
    receivers_per_y = [sorted(receivers_by_y[y]) for y in sorted_ys]  # row-major-sortable
    # Flattened row-major == ring position order (ring pos r = receivers_per_y[r//rpb][r%rpb]).
    receiver_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(rx, ry), ttnn.CoreCoord(rx, ry)) for row in receivers_per_y for rx, ry in row]
    )

    receiver_sub_device = ttnn.SubDevice([receiver_core_range_set])
    sub_device_manager = device.create_sub_device_manager([receiver_sub_device], 0)
    device.load_sub_device_manager(sub_device_manager)
    receiver_sub_device_id = ttnn.SubDeviceId(0)
    device.set_sub_device_stall_group([receiver_sub_device_id])

    torch.manual_seed(0xBE7)
    pt_weight = torch.zeros(1, 1, k_padded, _N)
    pt_weight[:, :, :_K, :] = torch.randn(1, 1, _K, _N)
    pt_act = torch.zeros(1, 1, _M, k_padded)
    pt_act[:, :, :, :_K] = torch.randn(1, 1, _M, _K)

    # Receiver-contiguous weight: NdShardSpec, num_shards = ring_size, each shard (K_padded, n_per_recv).
    n_per_recv = _N // ring_size
    is_shard_contiguous = distribution == ttnn.ShardDistributionStrategy.CONTIGUOUS_1D
    tt_weight = _make_recv_contig_weight(
        device, pt_weight, num_dram_banks, ring_size, _DTYPE, distribution_strategy=distribution
    )

    K_per_shard = k_padded // ring_size
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

    block_size_bytes = int((k_padded * n_per_recv) * _DTYPE_BYTES)
    gcb_size = _gcb_size_capped(block_size_bytes, k=k_padded, ring_size=ring_size, max_buffered_blocks=1)

    cc_program_config = _build_program_config(
        ring_size=ring_size,
        ring_cols=ring_cols,
        ring_rows=ring_rows,
        num_global_cb_receivers=num_receivers_per_bank,
    )

    # bank_to_receivers pairing must match the BDS placement so ring position r receives
    # shard r (full K, N-cols [r*npr, ...)):
    #   round_robin -> STRIDED: bank b feeds [b, b+num_banks, ...] = column b of receivers_per_y.
    #   shard-contiguous -> CONTIGUOUS arc: bank b feeds [b*R, b*R+R-1] = row b of receivers_per_y.
    if is_shard_contiguous:
        bank_to_receivers = [
            (
                b,
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(*receivers_per_y[b][s]), ttnn.CoreCoord(*receivers_per_y[b][s]))
                        for s in range(num_receivers_per_bank)
                    ]
                ),
            )
            for b in range(num_dram_banks)
        ]
    else:
        bank_to_receivers = [
            (
                b,
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(*receivers_per_y[s][b]), ttnn.CoreCoord(*receivers_per_y[s][b]))
                        for s in range(num_receivers_per_bank)
                    ]
                ),
            )
            for b in range(num_dram_banks)
        ]
    dual_senders = os.environ.get("BENCH_DUAL_SENDERS", "0") == "1"
    # Centralized recv-contig GCB builder: validates the (program_config, weight, bank_to_receivers)
    # triple (num_shards == ring_size, K % ring_size == 0, per_core_N == per-receiver N) and sizes/builds
    # the GCB in one place.
    gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
        device,
        [cc_program_config],
        [tt_weight],
        bank_to_receivers,
        gcb_size,
        support_multi_receiver_shards=not dual_senders,
    )
    output_mem_config = ttnn.create_sharded_memory_config(
        shape=(_M, _N // ring_size),
        core_grid=receiver_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    optional_output_tensor = ttnn.from_torch(
        torch.zeros(1, 1, _M, _N),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_mem_config,
    )

    def single_linear():
        return ttnn.linear(
            tt_act,
            tt_weight,
            program_config=cc_program_config,
            memory_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            global_cb=gcb,
            optional_output_tensor=optional_output_tensor,
            sub_device_id=receiver_sub_device_id,
        )

    # Centralized recv-contig param + cross-check: returns the validated block_count
    # (== ring_size) and TT_FATALs on a weight/program_config/gcb mismatch.
    block_count = ttnn.experimental.tensor_prefetcher_block_count_for_matmul_1d(cc_program_config, tt_weight, gcb)
    ttnn.experimental.start_tensor_prefetcher(device)
    ttnn.experimental.queue_tensor_prefetcher_request(
        device,
        [(tt_weight, block_count)] * num_prefetch_layers,
        global_cb=gcb,
    )

    # Correctness + program-cache warmup: one real one-matmul launch.
    cc_out = single_linear()
    cc_torch = ttnn.to_torch(cc_out)
    expected = pt_act.float() @ pt_weight.float()
    passing, output_str = comp_pcc(expected, cc_torch, 0.99)
    logger.info(f"[dram_core_rc] PCC: {output_str}")
    assert passing, f"[dram_core_rc] PCC failed: {output_str}"

    bench_trace = ttnn.begin_trace_capture(device, cq_id=0)
    bench_output = None
    for _ in range(trace_repeats):
        bench_output = single_linear()
    ttnn.end_trace_capture(device, bench_trace, cq_id=0)
    assert bench_output is not None

    t0 = time.perf_counter()
    ttnn.execute_trace(device, bench_trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0
    ttnn.release_trace(device, bench_trace)

    ttnn.experimental.stop_tensor_prefetcher(device)

    per_matmul_us = elapsed / trace_repeats * 1e6
    tflops = _flops_per_matmul(_K) * trace_repeats / elapsed / 1e12
    # Effective weight-streaming bandwidth: the full weight (K_padded x N) crosses the
    # ring once per matmul. This is the metric shard-contiguous distribution aims to improve.
    weight_bytes = k_padded * _N * _DTYPE_BYTES
    gbps = weight_bytes * trace_repeats / elapsed / 1e9
    dist_id = "shard_contiguous" if is_shard_contiguous else "round_robin"
    logger.info(
        f"[dram_core_rc][{op_name}] dist={dist_id} dual_senders={dual_senders} trace_elapsed={elapsed * 1e3:.2f}ms "
        f"repeats={trace_repeats} per_matmul={per_matmul_us:.2f}us -> {tflops:.4f} TFLOP/s, {gbps:.1f} GB/s"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("op_name,shape", LLAMA_SHAPES)
def test_bench_workercore_repeats(device, op_name, shape):
    """A/B baseline: same shape under trace replay with the worker-core prefetcher
    (BRISC+NCRISC senders on worker cores instead of DRISC on DRAM cores). Both the
    prefetcher op and the matmuls execute under FD inside the trace.

    Parametrized over the same Llama shapes as the DRAM-core test. FF2 skipped because
    it exceeds the worker prefetcher's L1 budget (matches production behavior — the
    full-model `Prefetcher` class in models/tt_transformers/tt/prefetcher.py routes FF2
    to a DRAM-sharded matmul instead of the prefetcher path).

    Uses dispatch_core_axis=COL like the canonical `test_prefetcher_BH` to keep the
    sender/receiver core grid (logical rows 0..ring_rows) clear of dispatch cores.
    """
    _apply_shape(shape)
    if device.dram_grid_size().x != 8:
        pytest.skip("Worker-sender bench expects 8 unharvested DRAM banks")

    trace_repeats = int(os.environ.get("BENCH_TRACE_REPEATS", "100"))
    num_prefetch_layers = trace_repeats + 1  # 1 warmup + trace_repeats inside the trace
    num_senders = _NUM_DRAM_BANKS
    ring_size = _RING_SIZE
    ring_cols = _RING_COLS
    ring_rows = _RING_ROWS

    # Use the production sender/receiver layout from
    # models/tt_transformers/tt/prefetcher/prefetcher_config.yaml. The naive row-major
    # grid in the original bench collides with dispatch cores at physical workers
    # 14-2/14-3 on Blackhole P150; the production layout avoids them by placing
    # senders on cols 0 (left) and 7 (right) with bank-ordered rows.
    from models.tt_transformers.tt.prefetcher import generate_sender_receiver_mapping, ARCH_CONFIG

    bh_cfg = ARCH_CONFIG["blackhole"]
    raw_mapping = generate_sender_receiver_mapping(num_receivers_per_sender=_NUM_RECV_PER_BANK)
    left_y = bh_cfg["bank_ordered_y_coords"]["left"]
    right_y = bh_cfg["bank_ordered_y_coords"]["right"]
    left_col = bh_cfg["sender_cols"]["left"]
    right_col = bh_cfg["sender_cols"]["right"]
    # Senders in bank-ID order: left first, then right.
    ordered_senders = [(left_col, y) for y in left_y] + [(right_col, y) for y in right_y]
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
    # gather_in0 matmul ignores compute_with_storage_grid_size and runs on the cores given
    # by sub_device_id. Set up: sub_device 0 = senders, sub_device 1 = receivers; matmul
    # runs on the second sub-device.
    sender_sub_device = ttnn.SubDevice([sender_core_range_set])
    receiver_sub_device = ttnn.SubDevice([receiver_core_range_set])
    sub_device_manager = device.create_sub_device_manager([sender_sub_device, receiver_sub_device], 0)
    device.load_sub_device_manager(sub_device_manager)
    receiver_sub_device_id = ttnn.SubDeviceId(1)
    device.set_sub_device_stall_group([receiver_sub_device_id])

    # Per-sender, per-block bytes (each sender pushes its share of the weight).
    block_size_bytes = int((_K * (_N // num_senders) // _NUM_RECV_PER_BANK) * _DTYPE_BYTES)
    gcb_size = _gcb_size_capped(block_size_bytes, k=_K, ring_size=ring_size)
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

    # One dram_prefetcher invocation with num_layers=num_prefetch_layers pushes that many
    # copies of the weight; 1 warmup matmul + N traced matmul launches each consume one.
    tt_addrs = _build_tensor_addrs(num_prefetch_layers)

    K_per_shard = _round_up(math.ceil(_K / ring_size), ttnn.TILE_SIZE)
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

    program_config = _build_program_config(
        ring_size=ring_size,
        ring_cols=ring_cols,
        ring_rows=ring_rows,
        num_global_cb_receivers=_NUM_RECV_PER_BANK,
    )
    output_mem_config = ttnn.create_sharded_memory_config(
        shape=(_M, _N // ring_size),
        core_grid=receiver_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    def single_linear():
        return ttnn.linear(
            tt_act,
            tt_weight,
            program_config=program_config,
            memory_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            global_cb=gcb,
            sub_device_id=receiver_sub_device_id,
        )

    # One bulk dram_prefetcher dispatch before the trace, mirroring the DRAM-core path's
    # start_tensor_prefetcher call. The kernel runs async on device and pushes
    # num_prefetch_layers worth of pages; the warmup matmul + N traced matmul launches
    # drain them as they arrive (GCB credit gates the producer/consumer pipeline).
    ttnn.dram_prefetcher([tt_weight, tt_addrs], num_layers=num_prefetch_layers, global_cb=gcb)

    # Correctness + program-cache warmup: one real one-matmul launch.
    cc_out = single_linear()
    cc_torch = ttnn.to_torch(cc_out)
    expected = pt_act.float() @ pt_weight.float()
    passing, output_str = comp_pcc(expected, cc_torch, 0.99)
    logger.info(f"[workercore] PCC: {output_str}")
    assert passing, f"[workercore] PCC failed: {output_str}"

    # Capture a trace of `trace_repeats` cached single-matmul dispatches and replay it once.
    bench_trace = ttnn.begin_trace_capture(device, cq_id=0)
    bench_output = None
    for _ in range(trace_repeats):
        bench_output = single_linear()  # Keep named so the output buffer stays alive.
    ttnn.end_trace_capture(device, bench_trace, cq_id=0)
    assert bench_output is not None  # trace_repeats > 0, so the loop ran at least once.

    t0 = time.perf_counter()
    ttnn.execute_trace(device, bench_trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0
    ttnn.release_trace(device, bench_trace)

    per_matmul_us = elapsed / trace_repeats * 1e6
    tflops = _flops_per_matmul(_K) * trace_repeats / elapsed / 1e12
    logger.info(
        f"[workercore][{op_name}] trace_elapsed={elapsed * 1e3:.2f}ms repeats={trace_repeats} "
        f"per_matmul={per_matmul_us:.2f}us -> {tflops:.4f} TFLOP/s"
    )
