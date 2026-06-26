# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Full forward pass reproduction of:
#   yuv_concat_conv_cam0_block_A (1×3×1536×1536)
#
# IR source:
#   BLOCK_A_AND_C/yuv_concat_minimal_ir/ttnn_yuv_concat_conv_cam0_block_A.mlir
#
# Pipeline (matches IR forward() function line-by-line):
#
#   Input (1,3,1536,1536) ROW_MAJOR DRAM
#     │
#     ├── [YUV Adapter — Method 2 spatial packing + ttnn.linear]
#     │   reshape  → (1,96,48,1536)          K=32, C×K=96=TILE_WIDTH, 0% waste
#     │   permute  → (1,48,1536,96) NHWC DRAM
#     │   reshape  → (1,1,73728,96) DRAM
#     │   to_layout TILE → TILE DRAM
#     │   linear   (96×96 packed weight, bias 96)
#     │   reshape  → (1,48,1536,96)
#     │   permute  → (1,96,48,1536) NCHW DRAM
#     │   reshape  → (1,3,1536,1536) ROW_MAJOR DRAM
#     │   to_layout TILE → (1,3,1536,1536) TILE DRAM
#     │
#     ├── [Y path — pixel_unshuffle r=4]  ★ BOTTLENECK R1 + P1
#     │   slice     ch=0  → (1,1,1536,1536) TILE L1
#     │   to_memory DRAM  → (1,1,1536,1536) TILE DRAM
#     │   reshape   → (1,1,384,4,384,4) TILE DRAM  ← R1 BOTTLENECK (X_PAD=32[4])
#     │   permute {0,3,5,1,2,4} → (1,4,4,1,384,384) TILE L1  ← P1
#     │   reshape   → (1,16,384,384) TILE L1
#     │
#     ├── [UV path — spatial-packed avgpool + pixel_unshuffle r=2]  ★ BOTTLENECK R2 + P2
#     │   slice     ch=1:3 → (1,2,1536,1536) TILE L1
#     │   reshape   → (1,32,96,1536)  K=16 spatial pack for depthwise avgpool
#     │   permute   → (1,96,1536,32) NHWC DRAM
#     │   reshape   → (1,1,147456,32) DRAM
#     │   conv2d    groups=32, k=[2,1], s=[2,2] → (1,1,36864,32) TILE L1 HS
#     │   reshape   → (1,48,768,32)
#     │   to_memory L1 interleaved
#     │   permute   → (1,32,48,768) TILE L1
#     │   to_memory DRAM TILE  → (1,32,48,768) TILE DRAM
#     │   reshape   → (1,2,384,2,384,2) TILE DRAM  ← R2 BOTTLENECK (X_PAD=32[2])
#     │   permute {0,3,5,1,2,4} → (1,2,2,2,384,384) TILE L1  ← P2
#     │   reshape   → (1,8,384,384) TILE L1
#     │
#     └── [Concat + Final Conv2d]
#         permute Y  → (1,384,384,16) NHWC DRAM
#         permute UV → (1,384,384,8)  NHWC DRAM
#         reshape    → (1,1,147456,16), (1,1,147456,8)
#         concat dim=3 → (1,1,147456,24) DRAM
#         conv2d IC=24, OC=64, 1×1, stride=1
#         reshape + permute → (1,64,384,384) TILE DRAM

import math

import pytest
import torch
import torch.nn.functional as F
import ttnn

# ---------------------------------------------------------------------------
# Weight packing helpers (torch only — no TTNN ops)
# ---------------------------------------------------------------------------

TILE_WIDTH = 32


def _make_packed_weight_pointwise(
    torch_weight: torch.Tensor, in_channels: int, out_channels: int, K: int
) -> torch.Tensor:
    """
    Method 2 block-diagonal weight for the YUV adapter 1×1 pointwise conv.
    Input:  [OC, IC, 1, 1]  e.g. [3, 3, 1, 1]
    Output: [IC*K, OC*K]    e.g. [96, 96]
    """
    w_bc = torch_weight.float().expand(out_channels, in_channels, K, K)
    k_range = torch.arange(K, dtype=torch.int64)
    k_row = k_range.reshape(1, 1, K, 1).expand(1, 1, K, K)
    k_col = k_range.reshape(1, 1, 1, K).expand(1, 1, K, K)
    i_k = (k_row == k_col).to(torch.float32)
    w_perm = (w_bc * i_k).permute(1, 2, 0, 3)
    return w_perm.reshape(in_channels * K, out_channels * K).to(torch.bfloat16)


def _make_packed_bias_pointwise(torch_bias: torch.Tensor, out_channels: int, K: int) -> torch.Tensor:
    """
    Packed bias for the YUV adapter: [OC] → [OC*K] via repeat_interleave(K).
    """
    return torch_bias.reshape(out_channels).repeat_interleave(K)


def _make_packed_weight_depthwise(original_weight: torch.Tensor, K: int) -> torch.Tensor:
    """
    Packed depthwise avgpool weight: [groups, IC/groups, kH, kW] → [groups*K, IC/groups, kH, kW]
    via repeat_interleave(K, dim=0).
    """
    return original_weight.repeat_interleave(K, dim=0)


# ---------------------------------------------------------------------------
# Layout check helper
# ---------------------------------------------------------------------------


def _check(step: str, tt: ttnn.Tensor, shape: list, layout, mem: str) -> None:
    """
    Assert shape and layout (TILE/ROW_MAJOR) match the IR exactly.
    Memory (DRAM/L1) is checked softly — a mismatch prints a note but does not
    fail, because TTNN Python auto-placement sometimes differs from the Forge IR.
    The bottleneck steps (R1, R2) are asserted strictly since they are TILE DRAM.
    """
    actual_shape = list(tt.shape)
    assert actual_shape == shape, f"{step}: shape {actual_shape} != {shape}"
    assert tt.layout == layout, f"{step}: layout {tt.layout} != {layout}"
    is_dram = tt.memory_config().buffer_type == ttnn.BufferType.DRAM
    actual_mem = "DRAM" if is_dram else "L1"
    if actual_mem != mem:
        print(f"  NOTE {step}: IR expects {mem}, got {actual_mem} (non-critical)")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG = pytest.param(1, 3, 3, 1536, 1536, id="cam0_block_A_1x3x1536x1536")

# YUV adapter spatial-packing factor: K = TILE_WIDTH // gcd(3, 32) = 32
_YUV_K = TILE_WIDTH // math.gcd(3, TILE_WIDTH)  # 32

# Depthwise avgpool spatial-packing factor: K = TILE_WIDTH // gcd(2, 32) = 16
_DW_K = TILE_WIDTH // math.gcd(2, TILE_WIDTH)  # 16


@pytest.mark.parametrize("batch, yuv_ic, yuv_oc, input_h, input_w", [_CONFIG])
def test_yuv_concat_conv_cam0_block_A(device, batch, yuv_ic, yuv_oc, input_h, input_w):
    """
    Full forward pass of yuv_concat_conv_cam0_block_A, matching the TTNN IR
    op-by-op with layout/shape verification at each step.

    Bottleneck ops verified:
      R1: reshape (1,1,1536,1536)→(1,1,384,4,384,4)  TILE DRAM  X_PAD=32[4]  87.5% waste
      P1: permute {0,3,5,1,2,4}  reads 37.75 MB (18432×1 tile grid)
      R2: reshape (1,32,48,768)→(1,2,384,2,384,2)    TILE DRAM  X_PAD=32[2]  93.75% waste
      P2: permute {0,3,5,1,2,4}  reads 37.75 MB (18432×1 tile grid)
    """
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    L1 = ttnn.L1_MEMORY_CONFIG
    RM = ttnn.ROW_MAJOR_LAYOUT
    TILE = ttnn.TILE_LAYOUT

    yuv_K = _YUV_K  # 32 — YUV adapter spatial packing
    dw_K = _DW_K  # 16 — depthwise avgpool spatial packing
    yuv_packed_ic = yuv_ic * yuv_K  # 96
    yuv_packed_sp = (input_h // yuv_K) * input_w  # 73728
    dw_packed_ic = 2 * dw_K  # 32 (UV channels=2, K=16)
    uv_c = 2

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=True,
    )
    conv_config_dw = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=TILE,
        deallocate_activation=True,
        act_block_h_override=0,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )
    conv_config_final = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=TILE,
        deallocate_activation=True,
        act_block_h_override=0,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )

    # ── Random torch tensors (exact shapes from the IR) ──────────────────────
    torch.manual_seed(42)
    torch_input = torch.randn(batch, yuv_ic, input_h, input_w, dtype=torch.bfloat16)

    # YUV adapter weights: arg2=[3,3,1,1], arg3=[1,1,1,3]
    torch_yuv_w = torch.randn(yuv_oc, yuv_ic, 1, 1, dtype=torch.bfloat16)
    torch_yuv_b = torch.randn(1, 1, 1, yuv_oc, dtype=torch.bfloat16)

    # Depthwise avgpool weight: arg1=[2,1,2,1] OIHW groups=2
    torch_dw_w = torch.full((uv_c, 1, 2, 1), 0.5, dtype=torch.bfloat16)

    # Final 1×1 conv: arg4=[64,24,1,1], arg5=[1,1,1,64]
    final_ic, final_oc = 24, 64
    final_h, final_w = input_h // 4, input_w // 4  # 384×384
    torch_final_w = torch.randn(final_oc, final_ic, 1, 1, dtype=torch.bfloat16)
    torch_final_b = torch.randn(1, 1, 1, final_oc, dtype=torch.bfloat16)

    # ── Torch golden (full pipeline, matching device op sequence exactly) ─────
    # 1. YUV adapter (1×1 pointwise conv — Method 2 is equivalent to this)
    yuv_out = F.conv2d(
        torch_input.float(),
        torch_yuv_w.reshape(yuv_oc, yuv_ic, 1, 1).float(),
        bias=torch_yuv_b.reshape(yuv_oc).float(),
    )  # [1,3,1536,1536]

    # 2. Slice Y and UV
    y_torch = yuv_out[:, 0:1, :, :]  # [1,1,1536,1536]
    uv_torch = yuv_out[:, 1:3, :, :]  # [1,2,1536,1536]

    # 3. Y pixel_unshuffle r=4 — matches device R1+P1 (confirmed channel ordering)
    #    R1: [1,1,1536,1536]→[1,1,384,4,384,4], P1: permute{0,3,5,1,2,4}→[1,4,4,1,384,384]
    #    Final reshape→[1,16,384,384] with channel k = r_H*4 + r_W
    #    PyTorch PixelUnshuffle(4) gives the same channel ordering ✓
    y_us = torch.nn.PixelUnshuffle(4)(y_torch)  # [1,16,384,384]

    # 4. UV path: avgpool then match device's R2+P2 exact reshape/permute sequence
    #    Device represents avgpool output as [1,32,48,768] via spatial packing K=16.
    #    uv_avg[1,2,768,768] → reshape[1,32,48,768] is valid (same flat ordering):
    #      c32*48*768 + h48*768 + w  ≡  c_uv*768*768 + h*768 + w  ✓
    uv_avg = F.avg_pool2d(uv_torch.float(), kernel_size=(2, 1), stride=(2, 2))  # [1,2,768,768]
    uv_32_48_768 = uv_avg.reshape(1, 32, 48, 768)  # mirror device [1,32,48,768]
    uv_6d = uv_32_48_768.reshape(1, 2, 384, 2, 384, 2)  # R2: same 6D split
    uv_perm = uv_6d.permute(0, 3, 5, 1, 2, 4).contiguous()  # P2: same permutation
    uv_us = uv_perm.reshape(1, 8, 384, 384)  # [1,8,384,384]
    # Note: device channel ordering is k = r_H*4 + r_W*2 + c_uv
    #       PyTorch PixelUnshuffle(2) gives k = c_uv*4 + r_H*2 + r_W — DIFFERENT
    #       Using device ordering ensures final conv weights align.

    # 5. Concat: permute Y and UV to NHWC, concat dim=3, back to NCHW
    y_nhwc = y_us.permute(0, 2, 3, 1)  # [1,384,384,16]
    uv_nhwc = uv_us.permute(0, 2, 3, 1)  # [1,384,384,8]
    cat_nhwc = torch.cat([y_nhwc, uv_nhwc], dim=3)  # [1,384,384,24]
    cat_nchw = cat_nhwc.permute(0, 3, 1, 2)  # [1,24,384,384]

    # 6. Final 1×1 conv
    golden = F.conv2d(
        cat_nchw,
        torch_final_w.reshape(final_oc, final_ic, 1, 1).float(),
        bias=torch_final_b.reshape(final_oc).float(),
    )  # [1,64,384,384]

    # ── Packed weights (torch only) ───────────────────────────────────────────
    # YUV adapter packed weight [96,96] and bias [96] (Method 2)
    torch_yuv_w_packed = _make_packed_weight_pointwise(torch_yuv_w, yuv_ic, yuv_oc, yuv_K)
    torch_yuv_b_packed = _make_packed_bias_pointwise(torch_yuv_b, yuv_oc, yuv_K)

    # Depthwise avgpool packed weight [32,1,2,1]
    torch_dw_w_packed = _make_packed_weight_depthwise(torch_dw_w, dw_K)

    # ── Prepare conv weights on device ────────────────────────────────────────
    # Depthwise avgpool weight  (IR: forward_const_eval_1)
    tt_dw_w = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(torch_dw_w_packed, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=dw_packed_ic,
        out_channels=dw_packed_ic,
        batch_size=batch,
        input_height=input_h // dw_K,  # 96
        input_width=input_w,  # 1536
        kernel_size=(2, 1),
        stride=(2, 2),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=False,
        groups=dw_packed_ic,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config_dw,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    # Final conv weight + bias  (IR: forward_const_eval_4, const_eval_2)
    tt_final_w = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(torch_final_w, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=final_ic,
        out_channels=final_oc,
        batch_size=batch,
        input_height=final_h,
        input_width=final_w,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config_final,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_final_b = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(torch_final_b, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        in_channels=final_ic,
        out_channels=final_oc,
        batch_size=batch,
        input_height=final_h,
        input_width=final_w,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config_final,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    # YUV adapter packed weight+bias on device (TILE DRAM)  — IR: forward_const_eval_0, const_eval_3
    tt_yuv_w = ttnn.from_torch(
        torch_yuv_w_packed.reshape(1, 1, yuv_packed_ic, yuv_packed_ic),
        dtype=ttnn.bfloat16,
        layout=TILE,
        device=device,
        memory_config=DRAM,
    )
    tt_yuv_b = ttnn.from_torch(
        torch_yuv_b_packed.reshape(1, 1, 1, yuv_packed_ic),
        dtype=ttnn.bfloat16,
        layout=TILE,
        device=device,
        memory_config=DRAM,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # FORWARD PASS — exact op sequence from IR forward() lines 126-210
    #
    # Deallocation rule: ttnn.reshape returns a FREE VIEW sharing the source's
    # buffer (for both ROW_MAJOR and TILE layouts). The source must NOT be
    # deallocated until the first non-reshape op has consumed the view.
    # ═══════════════════════════════════════════════════════════════════════════

    yuv_h = input_h // yuv_K  # 48
    r = 4  # Y pixel_unshuffle factor
    uv_r = 2  # UV pixel_unshuffle factor
    out_h_y = input_h // r  # 384
    out_w_y = input_w // r  # 384
    dw_packed_h = input_h // dw_K  # 96
    dw_packed_sp = dw_packed_h * input_w  # 147456
    dw_out_h = dw_packed_h // 2  # 48
    dw_out_w = input_w // 2  # 768
    dw_out_sp = dw_out_h * dw_out_w  # 36864
    uv_ps_out_h = final_h  # 384
    uv_ps_out_w = final_w  # 384
    concat_sp = final_h * final_w  # 147456

    # Input  (#ttnn_layout22: ROW_MAJOR DRAM)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)
    _check("input", tt_input, [batch, yuv_ic, input_h, input_w], RM, "DRAM")

    # %6 reshape [1,3,1536,1536] → [1,96,48,1536]  — FREE VIEW of tt_input
    tt6 = ttnn.reshape(tt_input, (batch, yuv_packed_ic, yuv_h, input_w))
    _check("%6  reshape→[1,96,48,1536]", tt6, [batch, yuv_packed_ic, yuv_h, input_w], RM, "DRAM")

    # %7 permute {0,2,3,1} → [1,48,1536,96]  — consumes tt6 (view of tt_input)
    tt7 = ttnn.permute(tt6, dims=(0, 2, 3, 1), memory_config=DRAM)
    _check("%7  permute→[1,48,1536,96]", tt7, [batch, yuv_h, input_w, yuv_packed_ic], RM, "DRAM")
    ttnn.deallocate(tt_input)  # safe: permute finished reading tt6
    ttnn.deallocate(tt6)

    # %8 reshape → [1,1,73728,96]  — FREE VIEW of tt7
    tt8 = ttnn.reshape(tt7, (batch, 1, yuv_packed_sp, yuv_packed_ic))
    _check("%8  reshape→[1,1,73728,96]", tt8, [batch, 1, yuv_packed_sp, yuv_packed_ic], RM, "DRAM")

    # %9 to_layout TILE  — consumes tt8 (view of tt7)
    tt9 = ttnn.to_layout(tt8, TILE, memory_config=DRAM)
    _check("%9  to_layout TILE", tt9, [batch, 1, yuv_packed_sp, yuv_packed_ic], TILE, "DRAM")
    ttnn.deallocate(tt7)  # safe: to_layout finished reading tt8
    ttnn.deallocate(tt8)

    # %10/%11 — to_memory_config(DRAM) on already-DRAM tensor: skip (no-op)

    # %12 linear [1,1,73728,96] @ [1,1,96,96] + bias
    tt12 = ttnn.linear(tt9, tt_yuv_w, bias=tt_yuv_b, memory_config=DRAM)
    _check("%12 linear", tt12, [batch, 1, yuv_packed_sp, yuv_packed_ic], TILE, "DRAM")
    ttnn.deallocate(tt9)
    ttnn.deallocate(tt_yuv_w)
    ttnn.deallocate(tt_yuv_b)

    # %13 reshape → [1,48,1536,96]  — FREE VIEW of tt12
    tt13 = ttnn.reshape(tt12, (batch, yuv_h, input_w, yuv_packed_ic))
    _check("%13 reshape→[1,48,1536,96]", tt13, [batch, yuv_h, input_w, yuv_packed_ic], TILE, "DRAM")

    # %14 to_layout ROW_MAJOR  — consumes tt13 (view of tt12)
    tt14 = ttnn.to_layout(tt13, RM, memory_config=DRAM)
    _check("%14 to_layout ROW_MAJOR", tt14, [batch, yuv_h, input_w, yuv_packed_ic], RM, "DRAM")
    ttnn.deallocate(tt12)  # safe: to_layout finished reading tt13
    ttnn.deallocate(tt13)

    # %15 permute {0,3,1,2} NHWC→NCHW → [1,96,48,1536]
    tt15 = ttnn.permute(tt14, dims=(0, 3, 1, 2), memory_config=DRAM)
    _check("%15 permute→[1,96,48,1536]", tt15, [batch, yuv_packed_ic, yuv_h, input_w], RM, "DRAM")
    ttnn.deallocate(tt14)

    # %16 reshape → [1,3,1536,1536]  — FREE VIEW of tt15
    tt16 = ttnn.reshape(tt15, (batch, yuv_ic, input_h, input_w))
    _check("%16 reshape→[1,3,1536,1536]", tt16, [batch, yuv_ic, input_h, input_w], RM, "DRAM")

    # %17 to_layout TILE  — consumes tt16 (view of tt15)
    tt17 = ttnn.to_layout(tt16, TILE, memory_config=DRAM)
    _check("%17 to_layout TILE", tt17, [batch, yuv_ic, input_h, input_w], TILE, "DRAM")
    ttnn.deallocate(tt15)  # safe: to_layout finished reading tt16
    ttnn.deallocate(tt16)

    # ── Y path ────────────────────────────────────────────────────────────────

    # %18 slice Y ch=0  → [1,1,1536,1536] TILE
    tt18 = ttnn.slice(tt17, [0, 0, 0, 0], [batch, 1, input_h, input_w])
    assert list(tt18.shape) == [batch, 1, input_h, input_w], "%18 slice Y shape"
    assert tt18.layout == TILE, "%18 slice Y not TILE"

    # %19 to_memory_config DRAM (alias if already DRAM — same storage)
    tt19 = tt18 if tt18.memory_config().buffer_type == ttnn.BufferType.DRAM else ttnn.to_memory_config(tt18, DRAM)
    _check("%19 to_memory DRAM", tt19, [batch, 1, input_h, input_w], TILE, "DRAM")

    # %20 ★ BOTTLENECK R1 ★  reshape (1,1,1536,1536)→(1,1,384,4,384,4) TILE DRAM
    #   memref<18432×1×tile>  X_PAD=32[4]  87.5% waste  37.75 MB DRAM write
    tt20 = ttnn.reshape(tt19, [batch, 1, out_h_y, r, out_w_y, r])
    assert list(tt20.shape) == [batch, 1, out_h_y, r, out_w_y, r], "R1 shape"
    assert tt20.layout == TILE and tt20.memory_config().buffer_type == ttnn.BufferType.DRAM, "R1 must be TILE DRAM"

    # %21 ★ BOTTLENECK P1 ★  permute {0,3,5,1,2,4} → (1,4,4,1,384,384) L1
    #   reads 37.75 MB (18,432 tiles, 87.5% zero-padding per tile)
    tt21 = ttnn.permute(tt20, dims=(0, 3, 5, 1, 2, 4), memory_config=L1)
    _check("%21 BOTTLENECK P1 permute→6D", tt21, [batch, r, r, 1, out_h_y, out_w_y], TILE, "L1")
    ttnn.deallocate(tt19)  # safe: permute finished reading tt20 (view of tt19)
    ttnn.deallocate(tt20)

    # %22 reshape 6D→4D → [1,16,384,384] TILE L1  — FREE VIEW of tt21
    #   tt21 kept alive until %35 (permute consumes tt22)
    tt22 = ttnn.reshape(tt21, [batch, r * r, out_h_y, out_w_y])
    _check("%22 reshape→[1,16,384,384]", tt22, [batch, r * r, out_h_y, out_w_y], TILE, "L1")

    # ── UV path ───────────────────────────────────────────────────────────────

    # %23 slice UV ch=1:3  → [1,2,1536,1536] TILE
    tt23 = ttnn.slice(tt17, [0, 1, 0, 0], [batch, 3, input_h, input_w])
    assert list(tt23.shape) == [batch, uv_c, input_h, input_w], "%23 slice UV shape"
    assert tt23.layout == TILE, "%23 slice UV not TILE"
    ttnn.deallocate(tt17)

    # %24 reshape → [1,32,96,1536]  — FREE VIEW of tt23
    tt24 = ttnn.reshape(tt23, (batch, dw_packed_ic, dw_packed_h, input_w))
    _check("%24 reshape→[1,32,96,1536]", tt24, [batch, dw_packed_ic, dw_packed_h, input_w], TILE, "L1")

    # to_layout ROW_MAJOR  — consumes tt24 (view of tt23)
    tt24_rm = ttnn.to_layout(tt24, RM, memory_config=DRAM)
    ttnn.deallocate(tt23)  # safe
    ttnn.deallocate(tt24)

    # %25 permute {0,2,3,1} → [1,96,1536,32] ROW_MAJOR DRAM
    tt25 = ttnn.permute(tt24_rm, dims=(0, 2, 3, 1), memory_config=DRAM)
    _check("%25 permute→[1,96,1536,32]", tt25, [batch, dw_packed_h, input_w, dw_packed_ic], RM, "DRAM")
    ttnn.deallocate(tt24_rm)

    # %26 reshape → [1,1,147456,32]  — FREE VIEW of tt25
    tt26 = ttnn.reshape(tt25, (batch, 1, dw_packed_sp, dw_packed_ic))
    _check("%26 reshape→[1,1,147456,32]", tt26, [batch, 1, dw_packed_sp, dw_packed_ic], RM, "DRAM")

    # %27 conv2d (deallocate_activation=True frees tt26 internally)
    tt27 = ttnn.conv2d(
        input_tensor=tt26,
        weight_tensor=tt_dw_w,
        in_channels=dw_packed_ic,
        out_channels=dw_packed_ic,
        device=device,
        bias_tensor=None,
        kernel_size=(2, 1),
        stride=(2, 2),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=batch,
        input_height=dw_packed_h,
        input_width=input_w,
        groups=dw_packed_ic,
        dtype=ttnn.bfloat16,
        conv_config=conv_config_dw,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    _check("%27 conv2d dw", tt27, [batch, 1, dw_out_sp, dw_packed_ic], TILE, "L1")
    ttnn.deallocate(tt25)  # safe: conv2d finished reading tt26 (view of tt25)
    ttnn.deallocate(tt_dw_w)

    # %28 reshape → [1,48,768,32] TILE L1  — FREE VIEW of tt27
    tt28 = ttnn.reshape(tt27, (batch, dw_out_h, dw_out_w, dw_packed_ic))
    _check("%28 reshape→[1,48,768,32]", tt28, [batch, dw_out_h, dw_out_w, dw_packed_ic], TILE, "L1")

    # %29 to_memory_config L1  — consumes tt28 (view of tt27)
    tt29 = ttnn.to_memory_config(tt28, L1)
    _check("%29 to_memory L1", tt29, [batch, dw_out_h, dw_out_w, dw_packed_ic], TILE, "L1")
    ttnn.deallocate(tt27)  # safe
    ttnn.deallocate(tt28)

    # %30 permute {0,3,1,2} → [1,32,48,768] TILE L1
    tt30 = ttnn.permute(tt29, dims=(0, 3, 1, 2), memory_config=L1)
    _check("%30 permute→[1,32,48,768]", tt30, [batch, dw_packed_ic, dw_out_h, dw_out_w], TILE, "L1")
    ttnn.deallocate(tt29)

    # %31 to_memory_config DRAM TILE  (#ttnn_layout44: H=48 padded to 64 in TILE)
    tt31 = ttnn.to_memory_config(tt30, DRAM)
    _check("%31 to_memory DRAM TILE", tt31, [batch, dw_packed_ic, dw_out_h, dw_out_w], TILE, "DRAM")
    ttnn.deallocate(tt30)

    # %32 ★ BOTTLENECK R2 ★  reshape (1,32,48,768)→(1,2,384,2,384,2) TILE DRAM
    #   memref<18432×1×tile>  X_PAD=32[2]  93.75% waste  37.75 MB DRAM write
    tt32 = ttnn.reshape(tt31, [batch, uv_c, uv_ps_out_h, uv_r, uv_ps_out_w, uv_r])
    assert list(tt32.shape) == [batch, uv_c, uv_ps_out_h, uv_r, uv_ps_out_w, uv_r], "R2 shape"
    assert tt32.layout == TILE and tt32.memory_config().buffer_type == ttnn.BufferType.DRAM, "R2 must be TILE DRAM"

    # %33 ★ BOTTLENECK P2 ★  permute {0,3,5,1,2,4} → (1,2,2,2,384,384) L1
    #   reads 37.75 MB (18,432 tiles, 93.75% zero-padding per tile)
    tt33 = ttnn.permute(tt32, dims=(0, 3, 5, 1, 2, 4), memory_config=L1)
    _check("%33 BOTTLENECK P2 permute→6D", tt33, [batch, uv_r, uv_r, uv_c, uv_ps_out_h, uv_ps_out_w], TILE, "L1")
    ttnn.deallocate(tt31)  # safe: permute finished reading tt32 (view of tt31)
    ttnn.deallocate(tt32)

    # %34 reshape 6D→4D → [1,8,384,384] TILE L1  — FREE VIEW of tt33
    #   tt33 kept alive until %36 (permute consumes tt34)
    tt34 = ttnn.reshape(tt33, [batch, uv_r * uv_r * uv_c, uv_ps_out_h, uv_ps_out_w])
    _check("%34 reshape→[1,8,384,384]", tt34, [batch, uv_r * uv_r * uv_c, uv_ps_out_h, uv_ps_out_w], TILE, "L1")

    # ── Concat path ───────────────────────────────────────────────────────────

    # %35 permute Y [1,16,384,384]→[1,384,384,16] NHWC DRAM
    #   consumes tt22 (FREE VIEW of tt21)
    tt35_tile = ttnn.permute(tt22, dims=(0, 2, 3, 1), memory_config=DRAM)
    _check("%35 permute Y→NHWC TILE", tt35_tile, [batch, final_h, final_w, r * r], TILE, "DRAM")
    ttnn.deallocate(tt21)  # safe: permute finished reading tt22 (view of tt21)
    ttnn.deallocate(tt22)
    # Untilize: reshape on TILE is NOT a free view — must convert to ROW_MAJOR first
    tt35 = ttnn.to_layout(tt35_tile, RM, memory_config=DRAM)
    ttnn.deallocate(tt35_tile)

    # %36 permute UV [1,8,384,384]→[1,384,384,8] NHWC DRAM
    #   consumes tt34 (FREE VIEW of tt33)
    tt36_tile = ttnn.permute(tt34, dims=(0, 2, 3, 1), memory_config=DRAM)
    _check("%36 permute UV→NHWC TILE", tt36_tile, [batch, final_h, final_w, uv_r * uv_r * uv_c], TILE, "DRAM")
    ttnn.deallocate(tt33)  # safe: permute finished reading tt34 (view of tt33)
    ttnn.deallocate(tt34)
    tt36 = ttnn.to_layout(tt36_tile, RM, memory_config=DRAM)
    ttnn.deallocate(tt36_tile)

    # %37 reshape Y → [1,1,147456,16] ROW_MAJOR  — FREE VIEW of tt35
    tt37 = ttnn.reshape(tt35, (batch, 1, concat_sp, r * r))
    _check("%37 reshape Y→flat", tt37, [batch, 1, concat_sp, r * r], RM, "DRAM")

    # %38 reshape UV → [1,1,147456,8] ROW_MAJOR  — FREE VIEW of tt36
    tt38 = ttnn.reshape(tt36, (batch, 1, concat_sp, uv_r * uv_r * uv_c))
    _check("%38 reshape UV→flat", tt38, [batch, 1, concat_sp, uv_r * uv_r * uv_c], RM, "DRAM")

    # %39 concat dim=3 → [1,1,147456,24] ROW_MAJOR  — consumes tt37 and tt38
    tt39 = ttnn.concat([tt37, tt38], dim=3, memory_config=DRAM)
    _check("%39 concat→[1,1,147456,24]", tt39, [batch, 1, concat_sp, final_ic], RM, "DRAM")
    ttnn.deallocate(tt35)  # safe: concat finished reading tt37 (view of tt35)
    ttnn.deallocate(tt37)
    ttnn.deallocate(tt36)  # safe: concat finished reading tt38 (view of tt36)
    ttnn.deallocate(tt38)

    # %40/%41 L1 HEIGHT_SHARDED → DRAM round-trip (IR redundancy — skip)

    # %42 conv2d IC=24, OC=64, 1×1, stride=1
    tt42 = ttnn.conv2d(
        input_tensor=tt39,
        weight_tensor=tt_final_w,
        in_channels=final_ic,
        out_channels=final_oc,
        device=device,
        bias_tensor=tt_final_b,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=batch,
        input_height=final_h,
        input_width=final_w,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=conv_config_final,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    _check("%42 conv2d final", tt42, [batch, 1, concat_sp, final_oc], TILE, "L1")
    ttnn.deallocate(tt39)
    ttnn.deallocate(tt_final_w)
    ttnn.deallocate(tt_final_b)

    # %43 reshape → [1,384,384,64] TILE L1  — FREE VIEW of tt42
    tt43 = ttnn.reshape(tt42, (batch, final_h, final_w, final_oc))
    _check("%43 reshape→[1,384,384,64]", tt43, [batch, final_h, final_w, final_oc], TILE, "L1")

    # %44 permute NHWC→NCHW → [1,64,384,384] TILE L1  — consumes tt43 (view of tt42)
    tt44 = ttnn.permute(tt43, dims=(0, 3, 1, 2), memory_config=L1)
    _check("%44 permute→NCHW [1,64,384,384]", tt44, [batch, final_oc, final_h, final_w], TILE, "L1")
    ttnn.deallocate(tt42)  # safe: permute finished reading tt43 (view of tt42)
    ttnn.deallocate(tt43)

    # %45 to_memory_config DRAM
    tt45 = ttnn.to_memory_config(tt44, DRAM)
    _check("%45 output [1,64,384,384]", tt45, [batch, final_oc, final_h, final_w], TILE, "DRAM")
    ttnn.deallocate(tt44)

    # ── Extract result and verify ─────────────────────────────────────────────
    result = ttnn.to_torch(ttnn.to_layout(tt45, RM, memory_config=DRAM))
    ttnn.deallocate(tt45)

    assert result.shape == torch.Size(
        [batch, final_oc, final_h, final_w]
    ), f"Output shape mismatch: got {tuple(result.shape)}"

    pcc = torch.corrcoef(torch.stack([result.float().flatten(), golden.float().flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99"
