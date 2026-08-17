# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# block_C_conv_to_conv  —  TTNN IR op-by-op reproduction
#
# Source: /proj_sw/user_dev/pchandrasekaran/tt-metal/new.mlir
# Module: block_C_conv_to_conv
#
# forward() data-flow summary:
#   %0-%4  : const_eval weights (YUV, DW, conv2 bias/weight)
#   %6-%16 : YUV adapter  [1,3,1280,2304] RM DRAM
#   %17-%21: Y-path  pixel_unshuffle(4) → [1,16,320,576] L1  (ALIVE through %34)
#   %22-%25: UV-path  reshape+permute  → [1,1,184320,32] L1
#   %26    : dw conv2d  → [1,1,46080,32]  HEIGHT_SHARDED L1
#   %27-%33: post-DW reshape → pixel_unshuffle(2) → [1,8,320,576] L1
#   %34-%38: permute Y+UV2 → reshape → concat → [1,1,184320,24] DRAM
#   %39-%40: HEIGHT_SHARDED round-trip (DRAM→L1 HS→DRAM)
#   %41    : 1×1 conv2d in_ch=24 out_ch=64 → [1,1,184320,64]
#   %42-%44: reshape+permute+DRAM → return [1,64,320,576]

import pytest
import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT

# ---------------------------------------------------------------------------
# %12 linear: MatmulMultiCoreReuseMultiCast1D program config
# MLIR attrs: compute_with_storage_grid_size=(8,8), in0_block_w=3,
#             out_subblock_h=1, out_subblock_w=3, out_block_h=45, out_block_w=3,
#             per_core_m=45, per_core_n=3, fuse_batch=true, mcast_in0=false,
#             gather_in0=false, untilize_out=false
# ---------------------------------------------------------------------------
_MATMUL_PROG_CFG = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    in0_block_w=3,
    out_subblock_h=1,
    out_subblock_w=3,
    out_block_h=45,
    out_block_w=3,
    per_core_M=45,
    per_core_N=3,
    fuse_batch=True,
    mcast_in0=False,
    gather_in0=False,
    untilize_out=False,
    num_global_cb_receivers=0,
)

# ---------------------------------------------------------------------------
# %12 linear output: HEIGHT_SHARDED L1 TILE (layout29)
# 64 cores (0,0)-(7,7); per-core shard = 45 tile-rows × 3 tile-cols
# = [45*32, 3*32] = [1440, 96] elements
# ---------------------------------------------------------------------------
_LINEAR_HS_SHARD = ttnn.ShardSpec(
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
    [1440, 96],
    ttnn.ShardOrientation.ROW_MAJOR,
)
_LINEAR_HS_CFG = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    _LINEAR_HS_SHARD,
)


# ===========================================================================
# Weight-packing helpers  (CPU-only)
# ===========================================================================


def _pack_yuv_weight(w: torch.Tensor) -> torch.Tensor:
    """[3,3,1,1] → [1,1,96,96] block-diagonal (kron K=32)."""
    K = 32
    w2d = w.float().squeeze()
    w_bc = w2d.unsqueeze(-1).unsqueeze(-1).expand(3, 3, K, K)
    k = torch.arange(K, dtype=torch.int64)
    diag = (k.unsqueeze(0) == k.unsqueeze(1)).float()
    out = (w_bc * diag).permute(1, 2, 0, 3).reshape(3 * K, 3 * K)
    return out.to(torch.bfloat16).reshape(1, 1, 96, 96)


def _pack_yuv_bias(b: torch.Tensor) -> torch.Tensor:
    """[1,1,1,3] → [1,1,1,96] via repeat_interleave(32)."""
    return b.reshape(3).repeat_interleave(32).to(torch.bfloat16).reshape(1, 1, 1, 96)


def _pack_dw_weight(w: torch.Tensor) -> torch.Tensor:
    """[2,1,2,1] → [32,1,2,1] via repeat_interleave(16, dim=0)."""
    return w.repeat_interleave(16, dim=0)


# ===========================================================================
# YUV adapter  (MLIR %6 – %16)
#
# Dealloc rule: reshape = zero-cost view; dealloc original only AFTER the
# first copy-op that consumes the view (Python TTNN has no ref-counting).
# ===========================================================================


def _yuv_adapter(device, tt_in, tt_w, tt_b, compute):
    """
    MLIR %6–%16: spatial-pack linear YUV color transform.
    Input : tt_in  [1,3,1280,2304]  RM DRAM  (layout22)
    Output: tt16   [1,3,1280,2304]  RM DRAM  (layout22)
    """
    # %6: reshape [1,3,1280,2304]→[1,96,40,2304]  zero-cost view (layout24)
    tt6 = ttnn.reshape(tt_in, (1, 96, 40, 2304))
    # %7: permute (0,2,3,1) → [1,40,2304,96]  RM DRAM  (layout25)
    tt7 = ttnn.permute(tt6, (0, 2, 3, 1), memory_config=DRAM)
    # MLIR deallocates %arg0 before %7 and %6 after %7; Python defers both until
    # tt7 is independent (tt_in/tt6 share a buffer)
    ttnn.deallocate(tt_in)

    # %8: reshape [1,40,2304,96]→[1,1,92160,96]  zero-cost view (layout26)
    tt8 = ttnn.reshape(tt7, (1, 1, 92160, 96))
    # %9: to_layout(tile) → TILE DRAM  (layout27)
    tt9 = ttnn.to_layout(tt8, TILE, memory_config=DRAM)
    ttnn.deallocate(tt7)  # safe: tt9 independent from tt7+tt8 shared buf

    # %10/%11: DRAM→DRAM re-tile ops (layout27→layout28, 2880×3 tile packing for matmul)
    # Python DRAM_MEMORY_CONFIG cannot express 2880×3 tile packing — to_memory_config
    # returns the same buffer (no-op), so deallocating tt9 would corrupt tt10/tt11.
    # Skip these ops and pass tt9 directly to linear.

    # %12: linear → HEIGHT_SHARDED L1 TILE  (layout29: 64 cores, 45×3 tiles/core)
    # MLIR: transpose_a=false, transpose_b=false (explicit; defaults in Python)
    tt12 = ttnn.linear(
        tt9,
        tt_w,
        bias=tt_b,
        memory_config=_LINEAR_HS_CFG,  # L1 HEIGHT_SHARDED (layout29)
        dtype=ttnn.bfloat16,
        program_config=_MATMUL_PROG_CFG,
        compute_kernel_config=compute,
        transpose_a=False,
        transpose_b=False,
    )
    # Dealloc order matches MLIR: input, %3 (bias), %0 (weight)
    ttnn.deallocate(tt9)
    ttnn.deallocate(tt_b)
    ttnn.deallocate(tt_w)

    # %13: reshape [1,1,92160,96]→[1,40,2304,96]  zero-cost view (layout30)
    #      inherits HEIGHT_SHARDED L1 TILE from tt12
    tt13 = ttnn.reshape(tt12, (1, 40, 2304, 96))
    # MLIR deallocates %12 here; Python must wait (tt13 is a view of tt12)

    # %14: to_memory_config(DRAM) → RM DRAM  (layout25)
    #      HEIGHT_SHARDED L1 TILE → RM DRAM interleaved
    #      Python to_layout handles TILE→RM + L1→DRAM in one call
    tt14 = ttnn.to_layout(tt13, RM, memory_config=DRAM)
    ttnn.deallocate(tt12)  # safe: tt14 independent from tt12+tt13 shared buf

    # %15: permute (0,3,1,2) → [1,96,40,2304]  RM DRAM  (layout24)
    tt15 = ttnn.permute(tt14, (0, 3, 1, 2), memory_config=DRAM)
    ttnn.deallocate(tt14)

    # %16: reshape [1,96,40,2304]→[1,3,1280,2304]  zero-cost view (layout22)
    tt16 = ttnn.reshape(tt15, (1, 3, 1280, 2304))
    # Do NOT deallocate tt15 — same buffer as tt16; freed via tt16/tt19
    return tt16


# ===========================================================================
# pixel_unshuffle → L1 (the op under test)
#
#   pu_to_l1 = True  (AFTER / FIX):   pixel_unshuffle writes its output DIRECTLY
#                                     to L1 (TILE). No DRAM→L1 round-trip.
#   pu_to_l1 = False (BEFORE / BUG):  pixel_unshuffle → TILE DRAM, then a SEPARATE
#                                     to_memory_config(L1) copy. This two-step
#                                     leaves the allocator in a state that makes
#                                     the downstream 1×1 conv's internal reshard
#                                     clash with its matmul static circular buffers.
#
# The caller owns `x` and deallocates it after this returns.
# ===========================================================================


def _pixel_unshuffle_to_l1(x, factor, pu_to_l1):
    if pu_to_l1:
        # AFTER (fix): output straight to L1 TILE
        return ttnn.pixel_unshuffle(x, downscale_factor=factor, memory_config=L1, output_layout=TILE)
    # BEFORE (bug): DRAM output, then a separate copy into L1
    tmp = ttnn.pixel_unshuffle(x, downscale_factor=factor)  # → TILE DRAM
    out = ttnn.to_memory_config(tmp, L1)  # → TILE L1 (separate copy)
    ttnn.deallocate(tmp)
    return out


# ===========================================================================
# Y-path + UV-path  (MLIR %17 – %25)
# ===========================================================================


def _y_and_uv_paths(device, tt16, pu_to_l1):
    """
    Y-path  (%17–%21): pixel_unshuffle(4) → [1,16,320,576] L1 interleaved.
                       Kept alive until %34 (after DW conv2d).
    UV-path (%22–%25): → [1,1,184320,32] L1 interleaved.
                       %25 is a zero-cost view of %24; caller must NOT
                       deallocate %24 separately.
    Returns (tt21, tt25).
    """
    # Y-path
    # %17: slice_static → [1,1,1280,2304]  TILE L1 interleaved (layout31)
    #      MLIR slice_static implicitly tilizes; Python slice gives RM output
    tt17 = ttnn.slice(tt16, [0, 0, 0, 0], [1, 1, 1280, 2304])
    # %18: to_memory_config(DRAM) → TILE DRAM  (layout32)
    tt18 = ttnn.to_layout(tt17, TILE, memory_config=DRAM)
    ttnn.deallocate(tt17)

    # %19: to_memory_config(%16, DRAM) → TILE DRAM  (layout27)
    tt19 = ttnn.to_layout(tt16, TILE, memory_config=DRAM)
    ttnn.deallocate(tt16)

    # %20-%21: pixel_unshuffle(4) → [1,16,320,576] L1 TILE.  KEEP ALIVE (consumed at %34).
    tt21 = _pixel_unshuffle_to_l1(tt18, 4, pu_to_l1)
    ttnn.deallocate(tt18)

    # UV-path
    # %22: slice_static → [1,2,1280,2304]  TILE L1 interleaved  (layout35)
    tt22 = ttnn.slice(tt19, [0, 1, 0, 0], [1, 3, 1280, 2304])
    ttnn.deallocate(tt19)

    # %23: reshape [1,2,1280,2304]→[1,32,80,2304]  non-zero-cost copy  (layout36 L1)
    tt23 = ttnn.reshape(tt22, (1, 32, 80, 2304), memory_config=L1)
    ttnn.deallocate(tt22)

    # %24: permute (0,2,3,1) → [1,80,2304,32]  TILE L1 interleaved  (layout37)
    tt24 = ttnn.permute(tt23, (0, 2, 3, 1), memory_config=L1)
    ttnn.deallocate(tt23)

    # %25: reshape [1,80,2304,32]→[1,1,184320,32]  zero-cost view  (layout38 L1)
    tt25 = ttnn.reshape(tt24, (1, 1, 184320, 32), memory_config=L1)
    # MLIR deallocates %24 here; Python must NOT (tt25 is a view of tt24)
    # deallocate_activation=True in DW conv will free tt25 (and tt24 buffer)

    return tt21, tt25


# ===========================================================================
# Shared pipeline
# ===========================================================================


def _run_pipeline(device, pu_to_l1):
    """
    Op-by-op reproduction of block_C_conv_to_conv forward() from new.mlir.
    Pipeline: YUV adapter → pixel_unshuffle Y+UV → DW conv2d →
    pixel_unshuffle UV2 → concat(Y,UV2) → 1×1 conv2d → [1,64,320,576].

    pu_to_l1 selects the pixel_unshuffle output path (see _pixel_unshuffle_to_l1):
      False → BEFORE (reproduces the CB clash at the 1×1 conv)
      True  → AFTER  (fix: pixel_unshuffle writes straight to L1)
    """
    torch.manual_seed(42)

    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )

    arg1 = torch.randn(2, 1, 2, 1, dtype=torch.bfloat16)
    arg2 = torch.randn(3, 3, 1, 1, dtype=torch.bfloat16)
    arg3 = torch.randn(1, 1, 1, 3, dtype=torch.bfloat16)
    arg4 = torch.randn(64, 24, 1, 1, dtype=torch.bfloat16)
    arg5 = torch.randn(1, 1, 1, 64, dtype=torch.bfloat16)

    # const_eval_0: YUV weight [1,1,96,96] TILE DRAM
    tt_yuv_w = ttnn.from_torch(
        _pack_yuv_weight(arg2),
        dtype=ttnn.bfloat16,
        layout=TILE,
        device=device,
        memory_config=DRAM,
    )
    # const_eval_3: YUV bias [1,1,1,96] TILE DRAM
    tt_yuv_b = ttnn.from_torch(
        _pack_yuv_bias(arg3),
        dtype=ttnn.bfloat16,
        layout=TILE,
        device=device,
        memory_config=DRAM,
    )

    # const_eval_1: DW conv weight — prepare_conv_weights mirrors forward_const_eval_1
    dw_cfg = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        deallocate_activation=True,
        act_block_h_override=0,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
        output_layout=TILE,
    )
    tt_dw_w = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(_pack_dw_weight(arg1), dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=L1,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=32,
        out_channels=32,
        batch_size=1,
        input_height=80,
        input_width=2304,
        kernel_size=(2, 1),
        stride=(2, 2),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=False,
        groups=32,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=dw_cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    # const_eval_4: conv2 weight; const_eval_2: conv2 bias
    # Note: no relu6 activation (removed in new IR vs old 14_ops.mlir)
    conv2_cfg = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        deallocate_activation=True,
        act_block_h_override=0,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
        output_layout=TILE,
    )
    tt_conv2_w = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(arg4, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=24,
        out_channels=64,
        batch_size=1,
        input_height=320,
        input_width=576,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv2_cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_conv2_b = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(arg5, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        in_channels=24,
        out_channels=64,
        batch_size=1,
        input_height=320,
        input_width=576,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv2_cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    # arg0  [1,3,1280,2304]  RM DRAM  (layout22)
    tt_arg0 = ttnn.from_torch(
        torch.randn(1, 3, 1280, 2304, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=RM,
        device=device,
        memory_config=DRAM,
    )

    # %6–%16 YUV adapter
    tt16 = _yuv_adapter(device, tt_arg0, tt_yuv_w, tt_yuv_b, compute)

    # %17–%25 Y-path + UV-path
    tt21, tt25 = _y_and_uv_paths(device, tt16, pu_to_l1)

    # ── %26 DW conv2d ──────────────────────────────────────────────────────────
    #   in: [1,1,184320,32] L1 interleaved (layout38)
    #   out: [1,1,46080,32] HEIGHT_SHARDED L1 (layout39, 63 cores)
    tt26 = ttnn.conv2d(
        input_tensor=tt25,
        weight_tensor=tt_dw_w,
        bias_tensor=None,
        in_channels=32,
        out_channels=32,
        device=device,
        kernel_size=(2, 1),
        stride=(2, 2),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=1,
        input_height=80,
        input_width=2304,
        groups=32,
        dtype=ttnn.bfloat16,
        conv_config=dw_cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    # Dealloc order matches MLIR: %1 (dw weight) after conv2d
    ttnn.deallocate(tt_dw_w)

    # ── %27–%28 reshape + to_memory_config(L1) ─────────────────────────────────
    #   %27: zero-cost view [1,1,46080,32]→[1,40,1152,32]  (layout39→layout40)
    tt27 = ttnn.reshape(tt26, (1, 40, 1152, 32))
    #   %28: to_memory_config(L1) → L1 interleaved  (layout41, 8×8)
    tt28 = ttnn.to_memory_config(tt27, L1)
    ttnn.deallocate(tt26)  # safe: tt28 independent; frees tt26+tt27 shared buf

    # ── %29 permute (0,3,1,2) → [1,32,40,1152]  L1  (layout42) ────────────────
    tt29 = ttnn.permute(tt28, (0, 3, 1, 2), memory_config=L1)
    ttnn.deallocate(tt28)

    # ── %30 reshape [1,32,40,1152]→[1,2,640,1152]  L1  (layout43) ──────────────
    #   Non-zero-cost (40→64 tile-pad) → copy
    tt30 = ttnn.reshape(tt29, (1, 2, 640, 1152), memory_config=L1)
    ttnn.deallocate(tt29)

    # ── %31 to_memory_config(DRAM) → [1,2,640,1152]  TILE DRAM  (layout44) ─────
    tt31 = ttnn.to_memory_config(tt30, DRAM)
    ttnn.deallocate(tt30)

    # %32-%33: pixel_unshuffle(2) → [1,8,320,576] L1 TILE.
    tt33 = _pixel_unshuffle_to_l1(tt31, 2, pu_to_l1)
    ttnn.deallocate(tt31)

    # ── %34 permute(%21,(0,2,3,1)) → [1,320,576,16]  L1  (layout47) ────────────
    tt34 = ttnn.permute(tt21, (0, 2, 3, 1), memory_config=L1)
    ttnn.deallocate(tt21)

    # ── %35 permute(%33,(0,2,3,1)) → [1,320,576,8]  L1  (layout47) ─────────────
    tt35 = ttnn.permute(tt33, (0, 2, 3, 1), memory_config=L1)
    ttnn.deallocate(tt33)

    # ── %36 reshape [1,320,576,16]→[1,1,184320,16]  L1  (layout38) ─────────────
    #   Zero-cost view: last=16 unchanged, 576/32=18 (no tile padding on dim-2)
    tt36 = ttnn.reshape(tt34, (1, 1, 184320, 16), memory_config=L1)
    ttnn.deallocate(tt34)  # matches MLIR deallocate(%34); force=False keeps tt36 alive

    # ── %37 reshape [1,320,576,8]→[1,1,184320,8]  L1  (layout38) ───────────────
    #   Zero-cost view: last=8 unchanged, 576/32=18
    tt37 = ttnn.reshape(tt35, (1, 1, 184320, 8), memory_config=L1)
    ttnn.deallocate(tt35)  # matches MLIR deallocate(%35); force=False keeps tt37 alive

    # ── %38 concat([%36,%37], dim=3) → [1,1,184320,24]  TILE DRAM  (layout48) ──
    tt38 = ttnn.concat([tt36, tt37], dim=3, memory_config=DRAM)
    # originals already freed → these drops refcount to 0 → buffers actually freed
    ttnn.deallocate(tt37)  # refcount 0 → tt35/tt37 buffer freed
    ttnn.deallocate(tt36)  # refcount 0 → tt34/tt36 buffer freed

    # ── %39 to_memory_config(L1 HEIGHT_SHARDED) ────────────────────────────────
    #   shard_spec: 64 cores (0,0)-(7,7), shard_shape=(2880,32) = 90×1 tiles/core
    tt_hs_shard_39 = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        [2880, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_hs_cfg_39 = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        tt_hs_shard_39,
    )
    tt39 = ttnn.to_memory_config(tt38, tt_hs_cfg_39)
    ttnn.deallocate(tt38)

    # ── %40 to_memory_config(DRAM) → [1,1,184320,24]  TILE DRAM  (layout48) ────
    tt40 = ttnn.to_memory_config(tt39, DRAM)
    ttnn.deallocate(tt39)

    # ── %41 1×1 conv2d (in_ch=24, out_ch=64) → [1,1,184320,64]  (layout50) ─────
    #   HEIGHT_SHARDED L1, 64 cores, 90×2 tiles/core
    tt41 = ttnn.conv2d(
        input_tensor=tt40,
        weight_tensor=tt_conv2_w,
        bias_tensor=tt_conv2_b,
        in_channels=24,
        out_channels=64,
        device=device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=1,
        input_height=320,
        input_width=576,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=conv2_cfg,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    # Dealloc order matches MLIR: %40 (auto-freed by deallocate_activation=True),
    # %4 (conv2 weight), %2 (conv2 bias)
    # tt40 already freed by deallocate_activation; skip to avoid double-free
    ttnn.deallocate(tt_conv2_w)  # %4
    ttnn.deallocate(tt_conv2_b)  # %2

    # ── %42–%44 reshape+permute+DRAM → [1,64,320,576] ─────────────────────────
    #   %42: zero-cost view [1,1,184320,64]→[1,320,576,64]  (layout51, H_SHARDED L1)
    tt42 = ttnn.reshape(tt41, (1, 320, 576, 64))
    ttnn.deallocate(tt41)  # matches MLIR deallocate(%41); force=False keeps tt42 alive
    #   %43: permute (0,3,1,2) → [1,64,320,576]  TILE L1 interleaved  (layout52)
    tt43 = ttnn.permute(tt42, (0, 3, 1, 2), memory_config=L1)
    ttnn.deallocate(tt42)  # matches MLIR deallocate(%42); refcount 0 → tt41/tt42 buf freed
    #   %44: to_memory_config(DRAM) → [1,64,320,576]  TILE DRAM  (layout23)
    tt44 = ttnn.to_memory_config(tt43, DRAM)
    ttnn.deallocate(tt43)

    out_shape = tuple(tt44.shape)
    print(f"output: shape={tt44.shape}  memory={tt44.memory_config()}  layout={tt44.layout}")
    ttnn.deallocate(tt44)
    return out_shape


# ===========================================================================
# Tests — BEFORE (reproduces CB clash) and AFTER (fixed)
# ===========================================================================


def test_block_C_conv_to_conv_before_fix(device):
    """
    BEFORE THE FIX — this test is EXPECTED TO FAIL.

    pixel_unshuffle writes to DRAM, then a SEPARATE to_memory_config(L1) copies it.
    This two-step leaves the L1 allocator in a state where the downstream 1×1
    conv2d's internal DRAM→L1 reshard places its buffers inside the matmul's
    static circular-buffer region → CB clash:

        "Statically allocated circular buffers in program 66 clash with L1
         buffers ... L1 buffer allocated at 761856 and static circular buffer
         region ends at 849312"

    The clash propagates as a test failure (no guard).
    """
    _run_pipeline(device, pu_to_l1=False)


def test_block_C_conv_to_conv_after_fix(device):
    """
    AFTER THE FIX.

    pixel_unshuffle writes its output DIRECTLY to L1 (memory_config=L1,
    output_layout=TILE), eliminating the DRAM→L1 round-trip. The downstream 1×1
    conv2d's internal reshard then lands its buffers above the matmul static-CB
    region → no clash. The full pipeline runs to completion.
    """
    out_shape = _run_pipeline(device, pu_to_l1=True)
    assert out_shape == (1, 64, 320, 576), f"unexpected output shape {out_shape}"
