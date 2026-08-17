# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Regression test: a 1D-config matmul (modeling Llama lm_head) runs ~20% slower when it executes
immediately after a DRAM-sharded-config matmul, because of a NOC virtual-channel (VC) state leak
between the two kernels.  Both matmuls are individually correct and fast; only the back-to-back
ordering is affected, which is why a single-op test would never catch this.

Root cause: the DRAM-sharded matmul reader (reader_bmm_tile_layout_in1_sender_dram_sharded.cpp)
calls set_async_read_state<NocOptions::CUSTOM_VC, ...> to write a per-bank virtual channel into
NCRISC_RD_CMD_BUF NOC_CTRL without restoring it at kernel end.  Subsequent kernels that rely on
NOC_CTRL being at the firmware default (VC=1, set in noc_init) -- such as the 1D matmul's reader
reader_bmm_tile_layout_in1_sender_writer_padding.cpp in DM_DEDICATED_NOC mode -- inherit the
stale VC and achieve ~20% lower DRAM read bandwidth.

Fix: reset NOC_CTRL to VC=1 at the end of reader_bmm_tile_layout_in1_sender_dram_sharded.cpp.

This test measures lm_head execution time solo vs immediately after a DRAM-sharded matmul
and asserts the ratio does not exceed DEGRADATION_THRESHOLD.

  Without fix: ratio ~1.20x  -> test FAILS
  With fix:    ratio ~1.00x  -> test PASSES

Important: the two phases are kept strictly separate so that DS never runs during the solo
warm-up.  If DS were mixed into the warm-up the NOC VC state on the shared cores would
already be corrupted when we measure solo, masking the degradation.
"""

import math
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

SEQ = 32  # M: decode batch x sequence length
DIM = 4096  # K: model hidden dimension
DS_N = 4096  # N for DRAM-sharded (DS) matmul
LM_HEAD_N = 131072  # N for lm_head 1D matmul (= 64 cores x 64 tiles x 32 elements)

# ---------------------------------------------------------------------------
# Compute grids
# ---------------------------------------------------------------------------

# DS matmul: 4x8 = 32 compute cores.  DIM % (TILE_SIZE * 32) = 4096 % 1024 = 0
_DS_GRID_X = 8
_DS_GRID_Y = 4
_DS_NUM_CORES = _DS_GRID_X * _DS_GRID_Y  # 32

# lm_head matmul: 8x8 = 64 compute cores.
_LM_GRID_X = 8
_LM_GRID_Y = 8
_LM_NUM_CORES = _LM_GRID_X * _LM_GRID_Y  # 64

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

_WARMUP_ITERS = 3  # kernel compilation + cache warm-up, not measured
_MEASURE_ITERS = 7  # odd count so median is the middle value
_DEGRADATION_THRESHOLD = 1.10  # lm_head after DS must be < 1.10x solo time


# ---------------------------------------------------------------------------
# Program / compute configs
# ---------------------------------------------------------------------------


def _ds_program_config():
    per_core_N = DIM // (32 * _DS_NUM_CORES)  # 4
    in0_block_w = DIM // (32 * _DS_NUM_CORES)  # 4
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=math.ceil(SEQ / 32),  # 1
        per_core_N=per_core_N,
        fused_activation=None,
    )


def _lm_head_program_config():
    per_core_N = LM_HEAD_N // (32 * _LM_NUM_CORES)  # 64
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(_LM_GRID_X, _LM_GRID_Y),
        in0_block_w=2,
        per_core_M=1,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=8,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _compute_config():
    cls = ttnn.WormholeComputeKernelConfig if not is_blackhole() else ttnn.types.BlackholeComputeKernelConfig
    return cls(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def _make_ds_tensors(device):
    """
    in0  (1,1,32,4096)   BF16  L1 width-sharded on 32 compute cores (4x8)
    in1  (1,1,4096,4096) BFP8  DRAM width-sharded across DRAM banks
    """
    dram_grid_size = device.dram_grid_size()
    dram_num_banks = dram_grid_size.x * dram_grid_size.y

    n_padded = math.ceil(DS_N / (32 * dram_num_banks)) * (32 * dram_num_banks)
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
            )
        }
    )
    in1_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (DIM, n_padded // dram_num_banks), ttnn.ShardOrientation.ROW_MAJOR),
    )
    in0_mem = ttnn.create_sharded_memory_config(
        shape=(SEQ, DIM // _DS_NUM_CORES),
        core_grid=ttnn.CoreGrid(y=_DS_GRID_Y, x=_DS_GRID_X),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    in0 = ttnn.from_torch(
        torch.randn(1, 1, SEQ, DIM, dtype=torch.bfloat16),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_mem,
    )
    in1 = ttnn.from_torch(
        torch.randn(1, 1, DIM, DS_N, dtype=torch.bfloat16),
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_mem,
    )
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
    return in0, in1, out_mem


def _make_lm_head_tensors(device):
    """
    in0  (1,1,32,4096)     BF16  L1 interleaved
    in1  (1,1,4096,131072) BFP8  DRAM interleaved  (~537 MB)

    torch_ref covers only the first _PCC_COLS columns to avoid a 4096x131072 CPU matmul.
    """
    torch_in0 = torch.randn(1, 1, SEQ, DIM, dtype=torch.bfloat16)
    torch_in1 = torch.randn(1, 1, DIM, LM_HEAD_N, dtype=torch.bfloat16)
    in0 = ttnn.from_torch(
        torch_in0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    in1 = ttnn.from_torch(
        torch_in1,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    # Reference only over a small slice — the full 4096x131072 matmul is too expensive on CPU.
    _PCC_COLS = 1024
    torch_ref = (torch_in0.float() @ torch_in1[..., :_PCC_COLS].float()).to(torch.bfloat16)
    return in0, in1, out_mem, torch_ref, _PCC_COLS


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_noc_vc_state_not_leaked_after_dram_sharded_matmul(device):
    """Assert lm_head is not slowed down by residual NOC VC state from a DRAM-sharded matmul.

    reader_bmm_tile_layout_in1_sender_dram_sharded.cpp calls
    set_async_read_state<NocOptions::CUSTOM_VC, ...> and must reset NOC_CTRL to the firmware
    default (VC=1) before returning.  Without the reset, the subsequent lm_head reader
    (reader_bmm_tile_layout_in1_sender_writer_padding.cpp) inherits a stale custom VC and
    reads the ~537 MB weight at ~149 GB/s instead of ~178 GB/s -- a persistent 20% slowdown.

    Measurement strategy:
      Phase 1 -- warm up and measure lm_head with NO DS ever having run.
                 NOC_CTRL is at firmware default (vc=1) on all cores.
      Phase 2 -- warm up DS kernel, then measure lm_head immediately after each DS run.
                 Without the fix, DS leaves vc=custom on its 32 cores (which overlap with
                 lm_head's 64-core grid), degrading lm_head reads by ~20%.

    Fails if:  median(phase2_lm_times) / median(phase1_lm_times) >= DEGRADATION_THRESHOLD
    Passes if: the ratio is below the threshold (NOC_CTRL correctly reset by the fix).
    """
    grid = device.compute_with_storage_grid_size()
    if grid.x < _LM_GRID_X or grid.y < _LM_GRID_Y:
        pytest.skip(f"Device compute grid {grid.x}x{grid.y} smaller than required {_LM_GRID_X}x{_LM_GRID_Y}")

    ds_cfg = _ds_program_config()
    lm_cfg = _lm_head_program_config()
    compute_cfg = _compute_config()

    ds_in0, ds_in1, ds_out_mem = _make_ds_tensors(device)
    lm_in0, lm_in1, lm_out_mem, lm_ref, pcc_cols = _make_lm_head_tensors(device)

    def run_ds():
        out = ttnn.matmul(
            ds_in0,
            ds_in1,
            program_config=ds_cfg,
            memory_config=ds_out_mem,
            dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=compute_cfg,
        )
        out.deallocate(True)

    def run_lm_head():
        return ttnn.matmul(
            lm_in0,
            lm_in1,
            program_config=lm_cfg,
            memory_config=lm_out_mem,
            dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=compute_cfg,
        )

    # PCC check against the first pcc_cols columns only (full-width reference is too expensive)
    out_tt = run_lm_head()
    out_torch = ttnn.to_torch(out_tt)[..., :pcc_cols]
    out_tt.deallocate(True)
    pcc = torch.corrcoef(torch.stack([lm_ref.flatten().float(), out_torch.flatten().float()]))[0, 1].item()
    logger.info(f"lm_head PCC: {pcc:.6f}")
    assert pcc > 0.97, f"lm_head PCC too low: {pcc:.6f}"

    # -----------------------------------------------------------------------
    # Phase 1: warm up lm_head kernel ONLY, then measure solo.
    # DS never runs here, so NOC_CTRL stays at the firmware default (vc=1).
    # -----------------------------------------------------------------------
    for _ in range(_WARMUP_ITERS):
        out = run_lm_head()
        out.deallocate(True)
    ttnn.device.synchronize_device(device)

    solo_times = []
    for _ in range(_MEASURE_ITERS):
        t0 = time.perf_counter()
        out = run_lm_head()
        ttnn.device.synchronize_device(device)
        solo_times.append(time.perf_counter() - t0)
        out.deallocate(True)

    # -----------------------------------------------------------------------
    # Phase 2: warm up DS kernel, then measure lm_head after each DS run.
    # DS leaves vc=custom on its 32 cores (without the fix), which are a
    # subset of lm_head's 64-core grid -- degrading lm_head DRAM reads.
    # -----------------------------------------------------------------------
    for _ in range(_WARMUP_ITERS):
        run_ds()
    ttnn.device.synchronize_device(device)

    seq_times = []
    for _ in range(_MEASURE_ITERS):
        run_ds()
        ttnn.device.synchronize_device(device)  # DS completes; stale VC now in NOC_CTRL
        t0 = time.perf_counter()
        out = run_lm_head()
        ttnn.device.synchronize_device(device)
        seq_times.append(time.perf_counter() - t0)
        out.deallocate(True)

    solo_median = sorted(solo_times)[_MEASURE_ITERS // 2]
    seq_median = sorted(seq_times)[_MEASURE_ITERS // 2]
    ratio = seq_median / solo_median

    logger.info(f"lm_head solo     median: {solo_median * 1e6:.0f} us")
    logger.info(f"lm_head after DS median: {seq_median * 1e6:.0f} us")
    logger.info(f"ratio: {ratio:.3f}x  (threshold: < {_DEGRADATION_THRESHOLD}x)")

    assert ratio < _DEGRADATION_THRESHOLD, (
        f"lm_head is {ratio:.2f}x slower after DRAM-sharded matmul "
        f"(threshold {_DEGRADATION_THRESHOLD}x). "
        f"NOC_CTRL VC not restored at end of "
        f"reader_bmm_tile_layout_in1_sender_dram_sharded.cpp."
    )
