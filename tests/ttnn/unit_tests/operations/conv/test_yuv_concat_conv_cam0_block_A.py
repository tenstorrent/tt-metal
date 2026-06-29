# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# yuv_concat_conv_cam0_block_A  —  baseline IR reproduction
#
# Source: BLOCK_A_AND_C/yuv_concat_minimal_ir/ttnn_yuv_concat_conv_cam0_block_A.mlir
#
# Pipeline:
#
#   Input  [1, 3, 1536, 1536]  ROW_MAJOR  DRAM  bfloat16
#      │
#   ┌──┴────────────────────────────────────────────────────┐
#   │  YUV Adapter  (spatial pack K=32, ttnn.linear)        │
#   │  Output: [1, 3, 1536, 1536]  TILE  DRAM  bfloat16    │
#   └──┬─────────────────────────────────────────────────────┘
#      │                              │
#   ┌──┴───────────────┐   ┌──────────┴────────────────────────────────┐
#   │  Y Path          │   │  UV Path                                  │
#   │  slice ch=0      │   │  slice ch=1:3                             │
#   │  ★ R1 reshape 6D │   │  spatial-packed depthwise avgpool         │
#   │  ★ P1 permute    │   │  ★ R2 reshape 6D                         │
#   │  [1,16,384,384]  │   │  ★ P2 permute                            │
#   │  TILE  L1        │   │  [1, 8, 384, 384]  TILE  L1              │
#   └──┬───────────────┘   └──────────┬────────────────────────────────┘
#      │                              │
#   ┌──┴──────────────────────────────┴────────────────────────┐
#   │  Concat + Final Conv2d                                   │
#   │  [1, 64, 384, 384]  TILE  DRAM  bfloat16                │
#   └──────────────────────────────────────────────────────────┘
#
# Bottlenecks (★):
#   R1: reshape → [1,1,384,4,384,4]  X_PAD=32[4]  87.5% waste  ~2.24 ms
#   P1: permute {0,3,5,1,2,4}        reads 37.75 MB padded     ~0.32 ms
#   R2: reshape → [1,2,384,2,384,2]  X_PAD=32[2]  93.75% waste ~1.71 ms
#   P2: permute {0,3,5,1,2,4}        reads 37.75 MB padded     ~0.26 ms

import math
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
import ttnn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_WIDTH = 32
_YUV_K = TILE_WIDTH // math.gcd(3, TILE_WIDTH)  # 32  YUV spatial-pack factor
_DW_K = TILE_WIDTH // math.gcd(2, TILE_WIDTH)  # 16  avgpool spatial-pack factor
_FINAL_CONV_K = 4  # final 1×1 conv spatial-pack: IC=24 → IC×K=96=3×TILE_WIDTH (0% waste)
_CONFIG = pytest.param(1, 3, 3, 1536, 1536, id="cam0_block_A_1x3x1536x1536")


# ---------------------------------------------------------------------------
# Derived dimensions (all in one place so every function uses consistent names)
# ---------------------------------------------------------------------------


@dataclass
class Dims:
    # Input
    batch: int
    yuv_ic: int  # 3
    yuv_oc: int  # 3
    input_h: int  # 1536
    input_w: int  # 1536

    # YUV adapter
    yuv_K: int  # 32  — C×K=TILE_WIDTH, 0% tile waste in NHWC permute
    yuv_packed_ic: int  # 96
    yuv_h: int  # 48  — input_h // K

    # Y pixel_unshuffle  (r=4)
    r_y: int = 4
    out_h: int = 0  # 384
    out_w: int = 0  # 384

    # UV avgpool spatial pack  (K=16)
    dw_K: int = 0  # 16
    dw_packed_ic: int = 0  # 32  — 2 UV channels × K=16
    dw_packed_h: int = 0  # 96  — input_h // dw_K
    dw_out_h: int = 0  # 48  — after avgpool stride=2 on H
    dw_out_w: int = 0  # 768 — after avgpool stride=2 on W

    # UV pixel_unshuffle  (r=2)
    r_uv: int = 2
    uv_c: int = 2  # UV channels in input

    # Final conv
    final_ic: int = 24  # Y(16) + UV(8)
    final_oc: int = 64
    concat_sp: int = 0  # 384×384 = 147456

    def __post_init__(self):
        self.out_h = self.input_h // self.r_y  # 384
        self.out_w = self.input_w // self.r_y  # 384
        self.dw_K = _DW_K  # 16
        self.dw_packed_ic = self.uv_c * self.dw_K  # 32
        self.dw_packed_h = self.input_h // self.dw_K  # 96
        self.dw_out_h = self.dw_packed_h // 2  # 48
        self.dw_out_w = self.input_w // 2  # 768
        self.concat_sp = self.out_h * self.out_w  # 147456


def make_dims(batch, yuv_ic, yuv_oc, input_h, input_w):
    return Dims(
        batch=batch,
        yuv_ic=yuv_ic,
        yuv_oc=yuv_oc,
        input_h=input_h,
        input_w=input_w,
        yuv_K=_YUV_K,
        yuv_packed_ic=yuv_ic * _YUV_K,
        yuv_h=input_h // _YUV_K,
    )


# ---------------------------------------------------------------------------
# Weight-packing helpers (CPU only, no TTNN ops)
# ---------------------------------------------------------------------------


def _pack_weight_pointwise(w: torch.Tensor, ic: int, oc: int, K: int) -> torch.Tensor:
    """Block-diagonal Method-2 weight for the YUV 1×1 adapter.
    Input  [OC, IC, 1, 1] → Output [IC*K, OC*K]  e.g. [96, 96]
    """
    w_bc = w.float().expand(oc, ic, K, K)
    k = torch.arange(K, dtype=torch.int64)
    k_row = k.reshape(1, 1, K, 1).expand(1, 1, K, K)
    k_col = k.reshape(1, 1, 1, K).expand(1, 1, K, K)
    return (w_bc * (k_row == k_col).float()).permute(1, 2, 0, 3).reshape(ic * K, oc * K).to(torch.bfloat16)


def _pack_bias_pointwise(b: torch.Tensor, oc: int, K: int) -> torch.Tensor:
    """[OC] → [OC*K] via repeat_interleave(K)."""
    return b.reshape(oc).repeat_interleave(K)


def _pack_weight_depthwise(w: torch.Tensor, K: int) -> torch.Tensor:
    """[G, IC/G, kH, kW] → [G*K, IC/G, kH, kW] via repeat_interleave(K, dim=0)."""
    return w.repeat_interleave(K, dim=0)


# ---------------------------------------------------------------------------
# Layout / shape assertion helper  (soft memory check, strict shape+layout)
# ---------------------------------------------------------------------------


def _check(label: str, tt: ttnn.Tensor, shape: list, layout, mem: str) -> None:
    """Assert shape and layout match the IR. Memory placement is checked softly."""
    assert list(tt.shape) == shape, f"{label}: shape {list(tt.shape)} != {shape}"
    assert tt.layout == layout, f"{label}: layout {tt.layout} != {layout}"
    actual = "DRAM" if tt.memory_config().buffer_type == ttnn.BufferType.DRAM else "L1"
    if actual != mem:
        print(f"  NOTE {label}: IR expects {mem}, got {actual}")


# ---------------------------------------------------------------------------
# Conv configs (shared across all device stages)
# ---------------------------------------------------------------------------


def _make_conv_configs(device):
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=True,
    )
    base = dict(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.TILE_LAYOUT,
        deallocate_activation=True,
        act_block_h_override=0,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )
    return compute, ttnn.Conv2dConfig(**base), ttnn.Conv2dConfig(**base)


# ===========================================================================
# Stage 1 — YUV Adapter
# ===========================================================================
#
# Implements a 1×1 pointwise conv on [N, 3, H, W] using Method-2 spatial
# packing: reshape C=3 → C×K=96, permute to NHWC, run ttnn.linear with a
# 96×96 block-diagonal weight, unpack back to NCHW.
#
# C×K = 3×32 = 96 = TILE_WIDTH  →  0% tile-column padding waste in permute.
#
# Input:  [N, 3, H, W]      ROW_MAJOR  DRAM  bfloat16
# Output: [N, 3, H, W]      TILE       DRAM  bfloat16


def run_yuv_adapter(
    device,
    tt_input: ttnn.Tensor,
    tt_yuv_w: ttnn.Tensor,
    tt_yuv_b: ttnn.Tensor,
    d: Dims,
) -> ttnn.Tensor:
    """YUV adapter: spatial-packed 1×1 conv via ttnn.linear."""
    DRAM, RM, TILE = ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
    sp = d.yuv_h * d.input_w  # 73728 = (H/K) × W

    # spatial pack: [N, 3, H, W] → [N, 96, H/32, W]
    t = ttnn.reshape(tt_input, (d.batch, d.yuv_packed_ic, d.yuv_h, d.input_w))
    _check("yuv_pack  ", t, [d.batch, d.yuv_packed_ic, d.yuv_h, d.input_w], RM, "DRAM")

    # NCHW → NHWC  [N, H/32, W, 96]  — C=96=TILE_WIDTH → 0% waste
    t = ttnn.permute(t, (0, 2, 3, 1), memory_config=DRAM)
    ttnn.deallocate(tt_input)
    _check("yuv_nhwc  ", t, [d.batch, d.yuv_h, d.input_w, d.yuv_packed_ic], RM, "DRAM")

    # flatten spatial → [N, 1, 73728, 96] + tilize for linear
    t = ttnn.to_layout(ttnn.reshape(t, (d.batch, 1, sp, d.yuv_packed_ic)), TILE, memory_config=DRAM)
    _check("yuv_flat  ", t, [d.batch, 1, sp, d.yuv_packed_ic], TILE, "DRAM")

    # linear: 96×96 block-diagonal weight  [N, 1, 73728, 96]
    t = ttnn.linear(t, tt_yuv_w, bias=tt_yuv_b, memory_config=DRAM)
    ttnn.deallocate(tt_yuv_w)
    ttnn.deallocate(tt_yuv_b)
    _check("yuv_linear", t, [d.batch, 1, sp, d.yuv_packed_ic], TILE, "DRAM")

    # untilize + permute NHWC → NCHW → unpack spatial
    t = ttnn.to_layout(ttnn.reshape(t, (d.batch, d.yuv_h, d.input_w, d.yuv_packed_ic)), RM, memory_config=DRAM)
    t = ttnn.permute(t, (0, 3, 1, 2), memory_config=DRAM)
    t = ttnn.to_layout(ttnn.reshape(t, (d.batch, d.yuv_ic, d.input_h, d.input_w)), TILE, memory_config=DRAM)
    _check("yuv_out   ", t, [d.batch, d.yuv_ic, d.input_h, d.input_w], TILE, "DRAM")
    return t  # [N, 3, 1536, 1536]  TILE  DRAM


# ===========================================================================
# Stage 2 — Y Path  (pixel_unshuffle r=4 via 6D reshape + permute)
# ===========================================================================
#
# ★ BOTTLENECK R1: reshape → [1,1,384,4,384,4]  TILE DRAM
#     X_PAD=32[4]  87.5% tile-column waste  37.75 MB write  ~2.24 ms
# ★ BOTTLENECK P1: permute {0,3,5,1,2,4}
#     reads 18432 padded tiles (87.5% zero-pad per tile)    ~0.32 ms
#
# Input:  [N, 3, H, W]       TILE  DRAM  bfloat16  (slices ch=0)
# Output: [N, 16, H/4, W/4]  TILE  L1   bfloat16


def run_y_path(device, tt17: ttnn.Tensor, d: Dims) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Y channel pixel_unshuffle r=4. Returns (tt21_src, tt22_view)."""
    DRAM, L1, TILE = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.TILE_LAYOUT
    r = d.r_y  # 4

    # slice Y channel  [N, 1, 1536, 1536]  TILE
    tt_y = ttnn.slice(tt17, [0, 0, 0, 0], [d.batch, 1, d.input_h, d.input_w])
    _check("y_slice", tt_y, [d.batch, 1, d.input_h, d.input_w], TILE, "L1")

    # ensure DRAM for R1 write
    if tt_y.memory_config().buffer_type != ttnn.BufferType.DRAM:
        tt_y = ttnn.to_memory_config(tt_y, DRAM)

    # ★ R1  [N, 1, 1536, 1536] → [N, 1, 384, 4, 384, 4]  TILE DRAM
    #        X_PAD=32[4]  87.5% tile waste  37.75 MB
    tt_r1 = ttnn.reshape(tt_y, [d.batch, 1, d.out_h, r, d.out_w, r])
    assert tt_r1.layout == TILE and tt_r1.memory_config().buffer_type == ttnn.BufferType.DRAM, "R1 must be TILE DRAM"
    _check("y_R1 ★ ", tt_r1, [d.batch, 1, d.out_h, r, d.out_w, r], TILE, "DRAM")

    # ★ P1  permute {0,3,5,1,2,4} → [N, 4, 4, 1, 384, 384]  TILE L1
    #        reads 18432×1 tile grid (87.5% zero-padding per tile)
    tt21 = ttnn.permute(tt_r1, (0, 3, 5, 1, 2, 4), memory_config=L1)
    ttnn.deallocate(tt_y)
    ttnn.deallocate(tt_r1)
    _check("y_P1 ★ ", tt21, [d.batch, r, r, 1, d.out_h, d.out_w], TILE, "L1")

    # reshape 6D → 4D  (free view of tt21 — tt21 kept alive by caller)
    tt22 = ttnn.reshape(tt21, [d.batch, r * r, d.out_h, d.out_w])
    _check("y_out  ", tt22, [d.batch, r * r, d.out_h, d.out_w], TILE, "L1")

    # Return tt21 (source) alongside tt22 (view) so caller manages lifetime.
    # tt21 must stay alive until the permute in the concat stage has consumed tt22.
    return tt21, tt22  # [N, 16, 384, 384]  TILE  L1


# ===========================================================================
# Stage 3a — UV Depthwise Avgpool
# ===========================================================================
#
# Spatial packing K=16 makes C×K = 2×16 = 32 = TILE_WIDTH for the depthwise conv.
# Implements avg_pool2d(kernel=(2,1), stride=(2,2)) via a packed depthwise conv2d.
# At the end the spatial packing is undone: [N, 32, 48, 768] → [N, 2, 768, 768].
#
# Input:  [N, 3,  H,   W]      TILE  DRAM  (slices ch=1:3)
# Output: [N, 2, 768, 768]     ROW_MAJOR  DRAM  bfloat16
#         (tt_rm, tt_unpack) — tt_unpack is a FREE VIEW of tt_rm;
#         caller must keep tt_rm alive until it finishes reading tt_unpack.


def run_uv_avgpool(
    device,
    tt17: ttnn.Tensor,
    tt_dw_w: ttnn.Tensor,
    d: Dims,
    compute_config,
    conv_cfg_dw,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """UV spatial-packed depthwise avgpool + unpack. Consumes tt17."""
    DRAM, L1, RM, TILE = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
    dw_sp = d.dw_packed_h * d.input_w  # 147456

    # slice UV channels [1:3]  [N, 2, 1536, 1536]  TILE
    tt_uv = ttnn.slice(tt17, [0, 1, 0, 0], [d.batch, 3, d.input_h, d.input_w])
    ttnn.deallocate(tt17)
    _check("uv_slice", tt_uv, [d.batch, d.uv_c, d.input_h, d.input_w], TILE, "L1")

    # spatial pack K=16: [N, 2, 1536, 1536] → [N, 32, 96, 1536]  TILE
    tt_pack = ttnn.reshape(tt_uv, (d.batch, d.dw_packed_ic, d.dw_packed_h, d.input_w))
    _check("uv_pack ", tt_pack, [d.batch, d.dw_packed_ic, d.dw_packed_h, d.input_w], TILE, "L1")

    # NCHW → NHWC  [N, 96, 1536, 32]  TILE L1  — C=32=TILE_WIDTH → 0% waste
    tt_nhwc = ttnn.permute(tt_pack, (0, 2, 3, 1), memory_config=L1)
    ttnn.deallocate(tt_uv)
    ttnn.deallocate(tt_pack)
    _check("uv_nhwc ", tt_nhwc, [d.batch, d.dw_packed_h, d.input_w, d.dw_packed_ic], TILE, "L1")

    # flatten  [N, 1, 147456, 32]  TILE L1
    tt_flat = ttnn.reshape(tt_nhwc, (d.batch, 1, dw_sp, d.dw_packed_ic))
    _check("uv_flat ", tt_flat, [d.batch, 1, dw_sp, d.dw_packed_ic], TILE, "L1")

    # depthwise conv2d: groups=32, k=[2,1], stride=[2,2]  →  [N, 1, 36864, 32]  TILE L1 HS
    tt_avg = ttnn.conv2d(
        input_tensor=tt_flat,
        weight_tensor=tt_dw_w,
        in_channels=d.dw_packed_ic,
        out_channels=d.dw_packed_ic,
        device=device,
        bias_tensor=None,
        kernel_size=(2, 1),
        stride=(2, 2),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=d.batch,
        input_height=d.dw_packed_h,
        input_width=d.input_w,
        groups=d.dw_packed_ic,
        dtype=ttnn.bfloat16,
        conv_config=conv_cfg_dw,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    ttnn.deallocate(tt_nhwc)
    ttnn.deallocate(tt_dw_w)
    dw_out_sp = d.dw_out_h * d.dw_out_w
    _check("uv_avg  ", tt_avg, [d.batch, 1, dw_out_sp, d.dw_packed_ic], TILE, "L1")

    # NHWC unpack + permute → [N, 32, 48, 768]  TILE DRAM
    tt_nhwc2 = ttnn.reshape(tt_avg, (d.batch, d.dw_out_h, d.dw_out_w, d.dw_packed_ic))
    tt_nhwc2 = ttnn.to_memory_config(tt_nhwc2, L1)
    ttnn.deallocate(tt_avg)
    tt_nchw = ttnn.permute(tt_nhwc2, (0, 3, 1, 2), memory_config=L1)
    ttnn.deallocate(tt_nhwc2)
    tt31 = ttnn.to_memory_config(tt_nchw, DRAM)
    ttnn.deallocate(tt_nchw)

    # Unpack spatial packing K=16: [N, 32, 48, 768] → [N, 2, 768, 768]  ROW_MAJOR DRAM
    # flat: c32*48*768+h48*768+w = (c32//16*768 + c32%16*48+h48)*768+w  ✓
    # C=2 UV channels × H=768 × W=768 — standard avgpool NCHW output form
    uv_h_out = (d.dw_packed_ic // d.uv_c) * d.dw_out_h  # 16 × 48 = 768
    tt_rm = ttnn.to_layout(tt31, RM, memory_config=DRAM)
    ttnn.deallocate(tt31)
    # tt_unpack is a FREE VIEW of tt_rm — caller keeps tt_rm alive until consuming tt_unpack
    tt_unpack = ttnn.reshape(tt_rm, (d.batch, d.uv_c, uv_h_out, d.dw_out_w))
    _check("uv_unpack", tt_unpack, [d.batch, d.uv_c, uv_h_out, d.dw_out_w], RM, "DRAM")
    return tt_rm, tt_unpack  # [N, 2, 768, 768]  ROW_MAJOR  DRAM


# ===========================================================================
# Stage 3b — UV Pixel_Unshuffle  (r=2 via 6D reshape + permute — baseline bottleneck)
# ===========================================================================
#
# Accepts [N, 2, 768, 768] RM DRAM from run_uv_avgpool (unpacked form).
# Converts to TILE, then applies the same 6D R2+P2 as the original IR.
#
# ★ BOTTLENECK R2: reshape → [N,2,384,2,384,2]  TILE DRAM
#     X_PAD=32[2]  93.75% tile-column waste  37.75 MB write  ~1.71 ms
# ★ BOTTLENECK P2: permute {0,3,5,1,2,4}
#     reads 18432 padded tiles (93.75% zero-pad per tile)    ~0.26 ms
#
# Input:  (tt_rm, tt_unpack) = [N, 2, 768, 768]  ROW_MAJOR  DRAM  bfloat16
# Output: [N, 8, 384, 384]   TILE  L1  bfloat16


def run_uv_pixel_unshuffle(
    device, tt_rm: ttnn.Tensor, tt_unpack: ttnn.Tensor, d: Dims
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """UV pixel_unshuffle r=2 via 6D R2+P2. Returns (tt33_src, tt34_view)."""
    DRAM, L1, TILE = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.TILE_LAYOUT
    r = d.r_uv  # 2

    # to TILE  [N, 2, 768, 768] RM → TILE DRAM  (last dims 768,768 → 0% tile waste)
    tt31 = ttnn.to_layout(tt_unpack, TILE, memory_config=DRAM)
    ttnn.deallocate(tt_rm)  # safe: to_layout finished reading tt_unpack (view of tt_rm)
    ttnn.deallocate(tt_unpack)
    _check("uv_tile ", tt31, [d.batch, d.uv_c, d.dw_out_w, d.dw_out_w], TILE, "DRAM")

    # ★ R2  [N, 2, 768, 768] → [N, 2, 384, 2, 384, 2]  TILE DRAM
    #        valid: h768=2h'+rh, w768=2w'+rw  →  same 6D split as baseline ✓
    #        X_PAD=32[2]  93.75% tile waste  37.75 MB
    tt_r2 = ttnn.reshape(tt31, [d.batch, d.uv_c, d.out_h, r, d.out_w, r])
    assert tt_r2.layout == TILE and tt_r2.memory_config().buffer_type == ttnn.BufferType.DRAM, "R2 must be TILE DRAM"
    _check("uv_R2 ★", tt_r2, [d.batch, d.uv_c, d.out_h, r, d.out_w, r], TILE, "DRAM")

    # ★ P2  permute {0,3,5,1,2,4} → [N, 2, 2, 2, 384, 384]  TILE L1
    #        reads 18432×1 tile grid (93.75% zero-padding per tile)
    tt33 = ttnn.permute(tt_r2, (0, 3, 5, 1, 2, 4), memory_config=L1)
    ttnn.deallocate(tt31)
    ttnn.deallocate(tt_r2)
    _check("uv_P2 ★", tt33, [d.batch, r, r, d.uv_c, d.out_h, d.out_w], TILE, "L1")

    # reshape 6D → 4D  (free view — tt33 kept alive by caller)
    tt34 = ttnn.reshape(tt33, [d.batch, r * r * d.uv_c, d.out_h, d.out_w])
    _check("uv_out  ", tt34, [d.batch, r * r * d.uv_c, d.out_h, d.out_w], TILE, "L1")

    return tt33, tt34  # [N, 8, 384, 384]  TILE  L1


# ===========================================================================
# Stage 4 — Concat + Final 1×1 Conv2d
# ===========================================================================
#
# Permutes Y and UV to NHWC, concatenates along channel dim, runs 1×1 conv.
#
# Input Y:  [N, 16, 384, 384]  TILE  L1   bfloat16
# Input UV: [N,  8, 384, 384]  TILE  L1   bfloat16
# Output:   [N, 64, 384, 384]  TILE  DRAM bfloat16


def run_concat_and_conv(
    device,
    tt21: ttnn.Tensor,
    tt22: ttnn.Tensor,  # Y:  source + view
    tt33: ttnn.Tensor,
    tt34: ttnn.Tensor,  # UV: source + view
    tt_final_w: ttnn.Tensor,
    tt_final_b: ttnn.Tensor,
    d: Dims,
    compute_config,
    conv_cfg_final,
) -> ttnn.Tensor:
    """Concat Y+UV and run final 1×1 conv2d."""
    DRAM, L1, TILE = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.TILE_LAYOUT
    r = d.r_y  # 4

    # permute Y NCHW → NHWC  [N, 384, 384, 16]  TILE L1
    tt_y_nhwc = ttnn.permute(tt22, (0, 2, 3, 1), memory_config=L1)
    ttnn.deallocate(tt21)
    ttnn.deallocate(tt22)
    _check("y_nhwc ", tt_y_nhwc, [d.batch, d.out_h, d.out_w, r * r], TILE, "L1")

    # permute UV NCHW → NHWC  [N, 384, 384, 8]  TILE L1
    tt_uv_nhwc = ttnn.permute(tt34, (0, 2, 3, 1), memory_config=L1)
    ttnn.deallocate(tt33)
    ttnn.deallocate(tt34)
    _check("uv_nhwc", tt_uv_nhwc, [d.batch, d.out_h, d.out_w, d.r_uv * d.r_uv * d.uv_c], TILE, "L1")

    # flatten  [N, 1, 147456, 16]  and  [N, 1, 147456, 8]
    tt_yf = ttnn.reshape(tt_y_nhwc, (d.batch, 1, d.concat_sp, r * r))
    tt_uvf = ttnn.reshape(tt_uv_nhwc, (d.batch, 1, d.concat_sp, d.r_uv * d.r_uv * d.uv_c))

    # concat dim=3 → [N, 1, 147456, 24]  TILE DRAM
    tt_cat = ttnn.concat([tt_yf, tt_uvf], dim=3, memory_config=DRAM)
    ttnn.deallocate(tt_y_nhwc)
    ttnn.deallocate(tt_yf)
    ttnn.deallocate(tt_uv_nhwc)
    ttnn.deallocate(tt_uvf)
    _check("concat ", tt_cat, [d.batch, 1, d.concat_sp, d.final_ic], TILE, "DRAM")

    # final conv2d  IC=24, OC=64, k=1×1, stride=1
    tt_out = ttnn.conv2d(
        input_tensor=tt_cat,
        weight_tensor=tt_final_w,
        in_channels=d.final_ic,
        out_channels=d.final_oc,
        device=device,
        bias_tensor=tt_final_b,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=d.batch,
        input_height=d.out_h,
        input_width=d.out_w,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=conv_cfg_final,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    ttnn.deallocate(tt_cat)
    ttnn.deallocate(tt_final_w)
    ttnn.deallocate(tt_final_b)
    _check("conv   ", tt_out, [d.batch, 1, d.concat_sp, d.final_oc], TILE, "L1")

    # unpack NHWC → NCHW  [N, 64, 384, 384]  TILE DRAM
    tt_out = ttnn.to_memory_config(
        ttnn.permute(ttnn.reshape(tt_out, (d.batch, d.out_h, d.out_w, d.final_oc)), (0, 3, 1, 2), memory_config=L1),
        DRAM,
    )
    _check("output ", tt_out, [d.batch, d.final_oc, d.out_h, d.out_w], TILE, "DRAM")
    return tt_out  # [N, 64, 384, 384]  TILE  DRAM


# ===========================================================================
# Weight preparation helpers
# ===========================================================================


def _prep_yuv_weights(device, torch_yuv_w, torch_yuv_b, d: Dims, DRAM, RM, TILE):
    """Prepare YUV adapter packed weight+bias tensors on device."""
    w_packed = _pack_weight_pointwise(torch_yuv_w, d.yuv_ic, d.yuv_oc, d.yuv_K)
    b_packed = _pack_bias_pointwise(torch_yuv_b, d.yuv_oc, d.yuv_K)
    tt_w = ttnn.from_torch(
        w_packed.reshape(1, 1, d.yuv_packed_ic, d.yuv_packed_ic),
        dtype=ttnn.bfloat16,
        layout=TILE,
        device=device,
        memory_config=DRAM,
    )
    tt_b = ttnn.from_torch(
        b_packed.reshape(1, 1, 1, d.yuv_packed_ic), dtype=ttnn.bfloat16, layout=TILE, device=device, memory_config=DRAM
    )
    return tt_w, tt_b


def _prep_dw_weights(device, torch_dw_w, d: Dims, DRAM, RM, TILE, compute_config, conv_cfg_dw):
    """Prepare depthwise avgpool weight on device."""
    w_packed = _pack_weight_depthwise(torch_dw_w, d.dw_K)
    return ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w_packed, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=d.dw_packed_ic,
        out_channels=d.dw_packed_ic,
        batch_size=d.batch,
        input_height=d.dw_packed_h,
        input_width=d.input_w,
        kernel_size=(2, 1),
        stride=(2, 2),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=False,
        groups=d.dw_packed_ic,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_cfg_dw,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )


def _prep_final_weights(device, torch_final_w, torch_final_b, d: Dims, DRAM, RM, TILE, compute_config, conv_cfg_final):
    """Prepare final 1×1 conv weight+bias on device."""
    tt_w = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(torch_final_w, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=d.final_ic,
        out_channels=d.final_oc,
        batch_size=d.batch,
        input_height=d.out_h,
        input_width=d.out_w,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_cfg_final,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_b = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(torch_final_b, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        in_channels=d.final_ic,
        out_channels=d.final_oc,
        batch_size=d.batch,
        input_height=d.out_h,
        input_width=d.out_w,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_cfg_final,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    return tt_w, tt_b


# ===========================================================================
# TEST 0 — Baseline (6D reshape bottleneck)
# ===========================================================================


@pytest.mark.parametrize("batch, yuv_ic, yuv_oc, input_h, input_w", [_CONFIG])
def test_yuv_concat_conv_cam0_block_A(device, batch, yuv_ic, yuv_oc, input_h, input_w):
    """
    Full forward pass of yuv_concat_conv_cam0_block_A matching the TTNN IR.

    Input:  [1, 3, 1536, 1536]  ROW_MAJOR  DRAM  bfloat16
    Output: [1, 64,  384,  384]  TILE       DRAM  bfloat16

    Bottleneck ops reproduced exactly:
      R1: reshape → [1,1,384,4,384,4]  TILE DRAM  X_PAD=32[4]  87.5% waste  ~2.24 ms
      P1: permute {0,3,5,1,2,4}        reads 37.75 MB padded                ~0.32 ms
      R2: reshape → [1,2,384,2,384,2]  TILE DRAM  X_PAD=32[2]  93.75% waste ~1.71 ms
      P2: permute {0,3,5,1,2,4}        reads 37.75 MB padded                ~0.26 ms
    """
    DRAM, RM, TILE = ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
    d = make_dims(batch, yuv_ic, yuv_oc, input_h, input_w)
    compute, conv_dw, conv_final = _make_conv_configs(device)

    # ── Torch tensors ──────────────────────────────────────────────────────
    torch.manual_seed(42)
    x = torch.randn(batch, yuv_ic, input_h, input_w, dtype=torch.bfloat16)  # [1,3,1536,1536]
    torch_yw = torch.randn(yuv_oc, yuv_ic, 1, 1, dtype=torch.bfloat16)  # [3,3,1,1]
    torch_yb = torch.randn(1, 1, 1, yuv_oc, dtype=torch.bfloat16)  # [1,1,1,3]
    torch_dw = torch.full((d.uv_c, 1, 2, 1), 0.5, dtype=torch.bfloat16)  # [2,1,2,1]
    torch_fw = torch.randn(d.final_oc, d.final_ic, 1, 1, dtype=torch.bfloat16)  # [64,24,1,1]
    torch_fb = torch.randn(1, 1, 1, d.final_oc, dtype=torch.bfloat16)  # [1,1,1,64]

    # ── CPU golden ─────────────────────────────────────────────────────────
    yuv_out = F.conv2d(
        x.float(), torch_yw.reshape(yuv_oc, yuv_ic, 1, 1).float(), bias=torch_yb.reshape(yuv_oc).float()
    )  # [1,3,1536,1536]
    y_us = torch.nn.PixelUnshuffle(4)(yuv_out[:, 0:1])  # [1,16,384,384]
    uv_avg = F.avg_pool2d(yuv_out[:, 1:3], kernel_size=(2, 1), stride=(2, 2))  # [1,2,768,768]
    uv_6d = uv_avg.reshape(batch, d.uv_c, d.out_h, d.r_uv, d.out_w, d.r_uv)
    uv_us = (
        uv_6d.permute(0, 3, 5, 1, 2, 4).contiguous().reshape(batch, d.r_uv * d.r_uv * d.uv_c, d.out_h, d.out_w)
    )  # [1,8,384,384]
    cat = torch.cat([y_us.permute(0, 2, 3, 1), uv_us.permute(0, 2, 3, 1)], dim=3).permute(0, 3, 1, 2)
    golden = F.conv2d(
        cat, torch_fw.reshape(d.final_oc, d.final_ic, 1, 1).float(), bias=torch_fb.reshape(d.final_oc).float()
    )  # [1,64,384,384]

    # ── Device weights ──────────────────────────────────────────────────────
    tt_yw, tt_yb = _prep_yuv_weights(device, torch_yw, torch_yb, d, DRAM, RM, TILE)
    tt_dw = _prep_dw_weights(device, torch_dw, d, DRAM, RM, TILE, compute, conv_dw)
    tt_fw, tt_fb = _prep_final_weights(device, torch_fw, torch_fb, d, DRAM, RM, TILE, compute, conv_final)

    # ── Forward pass ────────────────────────────────────────────────────────
    tt_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)

    # Stage 1: YUV Adapter  →  [1, 3, 1536, 1536]  TILE  DRAM
    tt17 = run_yuv_adapter(device, tt_input, tt_yw, tt_yb, d)

    # Stage 2: Y Path  →  [1, 16, 384, 384]  TILE  L1  (★ R1+P1 bottleneck)
    # run_y_path only slices ch=0; tt17 is NOT deallocated inside — it stays valid.
    tt21, tt22 = run_y_path(device, tt17, d)

    # Stage 3a: UV Avgpool + unpack  →  [N, 2, 768, 768]  ROW_MAJOR  DRAM
    # run_uv_avgpool slices ch=1:3, deallocates tt17, and unpacks spatial-pack K=16.
    tt_rm_uv, tt_unpack_uv = run_uv_avgpool(device, tt17, tt_dw, d, compute, conv_dw)

    # Stage 3b: UV Pixel_Unshuffle  →  [1, 8, 384, 384]  TILE  L1  (★ R2+P2 bottleneck)
    tt33, tt34 = run_uv_pixel_unshuffle(device, tt_rm_uv, tt_unpack_uv, d)

    # Stage 4: Concat + Final Conv  →  [1, 64, 384, 384]  TILE  DRAM
    tt_out = run_concat_and_conv(device, tt21, tt22, tt33, tt34, tt_fw, tt_fb, d, compute, conv_final)

    # ── Verify ──────────────────────────────────────────────────────────────
    result = ttnn.to_torch(ttnn.to_layout(tt_out, RM, memory_config=DRAM))
    ttnn.deallocate(tt_out)

    assert list(result.shape) == [batch, d.final_oc, d.out_h, d.out_w]
    pcc = torch.corrcoef(torch.stack([result.float().flatten(), golden.flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99"


# ===========================================================================
# Pixel_Unshuffle stage functions  (replace 6D reshape bottleneck with ttnn op)
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 2 (PS) — Y Path  via ttnn.pixel_unshuffle
# ---------------------------------------------------------------------------
#
# Replaces the 6D reshape (R1) + 6D permute (P1) bottleneck (~2.56 ms)
# with a single ttnn.pixel_unshuffle call that reads NCHW DRAM directly.
#
# Channel ordering: k = rh*4+rw  — identical to torch.nn.PixelUnshuffle(4) ✓
#
# Input:  [N,  1, 1536, 1536]  TILE  DRAM  bfloat16  (slice of ch=0)
# Output: [N, 16,  384,  384]  ROW_MAJOR  DRAM  bfloat16


def run_y_path_pixel_unshuffle(device, tt17: ttnn.Tensor, d: Dims) -> ttnn.Tensor:
    """Y pixel_unshuffle via ttnn.pixel_unshuffle (no 6D reshape)."""
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    TILE = ttnn.TILE_LAYOUT

    # slice Y channel  [N, 1, 1536, 1536]  TILE
    tt_y = ttnn.slice(tt17, [0, 0, 0, 0], [d.batch, 1, d.input_h, d.input_w])
    if tt_y.memory_config().buffer_type != ttnn.BufferType.DRAM:
        tt_y = ttnn.to_memory_config(tt_y, DRAM)
    _check("y_ps_in ", tt_y, [d.batch, 1, d.input_h, d.input_w], TILE, "DRAM")

    # ttnn.pixel_unshuffle: [N, 1, 1536, 1536] → [N, 16, 384, 384]  ROW_MAJOR DRAM
    # kernel reads NCHW directly — no padded 6D intermediate tensor
    tt22 = ttnn.pixel_unshuffle(tt_y, downscale_factor=d.r_y)
    ttnn.deallocate(tt_y)
    _check("y_ps_out", tt22, [d.batch, d.r_y * d.r_y, d.out_h, d.out_w], ttnn.ROW_MAJOR_LAYOUT, "DRAM")
    return tt22  # [N, 16, 384, 384]  ROW_MAJOR  DRAM


# ---------------------------------------------------------------------------
# Stage 3b (PS) — UV Pixel_Unshuffle  via ttnn.pixel_unshuffle
# ---------------------------------------------------------------------------
#
# Receives [N, 2, 768, 768] from run_uv_avgpool (already unpacked by avgpool).
# Applies pixel_unshuffle(r=2) directly — no extra reshape needed.
#
# Input:  (tt_rm, tt_unpack) = [N, 2, 768, 768]  ROW_MAJOR  DRAM  bfloat16
# Output: [N, 8, 384, 384]   ROW_MAJOR  DRAM  bfloat16
#         channel ordering: k = c_uv*4 + rh*2 + rw  (PyTorch standard) ✓


def run_uv_pixel_unshuffle_op(device, tt_rm: ttnn.Tensor, tt_unpack: ttnn.Tensor, d: Dims) -> ttnn.Tensor:
    """UV pixel_unshuffle(r=2) on unpacked [N,2,768,768] from avgpool."""
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    RM = ttnn.ROW_MAJOR_LAYOUT

    # pixel_unshuffle(r=2): [N, 2, 768, 768] → [N, 8, 384, 384]
    # C_out = 2×r² = 8,  H_out = 768/2 = 384,  W_out = 768/2 = 384
    # channel ordering: k = c_uv*4 + rh*2 + rw  (PyTorch standard) ✓
    tt_uv = ttnn.pixel_unshuffle(tt_unpack, downscale_factor=d.r_uv)
    ttnn.deallocate(tt_rm)  # safe: pixel_unshuffle finished reading tt_unpack (view of tt_rm)
    ttnn.deallocate(tt_unpack)
    _check("uv_ps_out", tt_uv, [d.batch, d.uv_c * d.r_uv * d.r_uv, d.out_h, d.out_w], RM, "DRAM")
    return tt_uv  # [N, 8, 384, 384]  ROW_MAJOR  DRAM  — concrete allocation


# ===========================================================================
# TEST 2 — pixel_unshuffle Y + UV  (ttnn.pixel_unshuffle replaces 6D bottleneck)
# ===========================================================================


@pytest.mark.parametrize("batch, yuv_ic, yuv_oc, input_h, input_w", [_CONFIG])
def test_yuv_concat_conv_cam0_block_A_pixel_unshuffle(device, batch, yuv_ic, yuv_oc, input_h, input_w):
    """
    Full forward pass using ttnn.pixel_unshuffle for Y and UV paths.

    Replaces the 6D reshape bottlenecks:
      ★ R1+P1  (Y path, ~2.56 ms)  →  ttnn.pixel_unshuffle(r=4) on [1,1,1536,1536]
      ★ R2+P2  (UV path, ~1.97 ms) →  unpack spatial-pack + ttnn.pixel_unshuffle(r=2)

    Input:  [1, 3, 1536, 1536]  ROW_MAJOR  DRAM  bfloat16
    Output: [1, 64,  384,  384]  TILE       DRAM  bfloat16

    Better concat+conv vs baseline:
      Baseline: 2 separate NCHW→NHWC permutes + concat_dim3 + conv2d  (7 ops)
      This:     concat_dim1 NCHW + 1 permute + flatten + conv2d        (6 ops, -1 permute)

    UV channel ordering: k = c_uv*4 + rh*2 + rw  (PyTorch PixelUnshuffle standard)
    PCC >= 0.99 vs CPU golden applying identical ops.
    """
    DRAM, RM, TILE = ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
    d = make_dims(batch, yuv_ic, yuv_oc, input_h, input_w)
    compute, conv_dw, conv_final = _make_conv_configs(device)

    # ── Torch tensors ──────────────────────────────────────────────────────────
    torch.manual_seed(42)
    x = torch.randn(batch, yuv_ic, input_h, input_w, dtype=torch.bfloat16)  # [1,3,1536,1536]
    torch_yw = torch.randn(yuv_oc, yuv_ic, 1, 1, dtype=torch.bfloat16)  # [3,3,1,1]
    torch_yb = torch.randn(1, 1, 1, yuv_oc, dtype=torch.bfloat16)  # [1,1,1,3]
    torch_dw = torch.full((d.uv_c, 1, 2, 1), 0.5, dtype=torch.bfloat16)  # [2,1,2,1]
    torch_fw = torch.randn(d.final_oc, d.final_ic, 1, 1, dtype=torch.bfloat16)  # [64,24,1,1]
    torch_fb = torch.randn(1, 1, 1, d.final_oc, dtype=torch.bfloat16)  # [1,1,1,64]

    # ── CPU golden (pixel_unshuffle path) ──────────────────────────────────────
    yuv_out = F.conv2d(
        x.float(), torch_yw.reshape(yuv_oc, yuv_ic, 1, 1).float(), bias=torch_yb.reshape(yuv_oc).float()
    )  # [1,3,1536,1536]
    y_us = torch.nn.PixelUnshuffle(4)(yuv_out[:, 0:1])  # [1, 16, 384, 384]
    uv_avg = F.avg_pool2d(yuv_out[:, 1:3], kernel_size=(2, 1), stride=(2, 2))  # [1, 2, 768, 768]
    uv_us = torch.nn.PixelUnshuffle(d.r_uv)(uv_avg)  # [1, 8, 384, 384]
    cat_nchw = torch.cat([y_us, uv_us], dim=1)  # [1, 24, 384, 384]
    golden = F.conv2d(
        cat_nchw.float(),
        torch_fw.reshape(d.final_oc, d.final_ic, 1, 1).float(),
        bias=torch_fb.reshape(d.final_oc).float(),
    )  # [1, 64, 384, 384]

    # ── Device weights ──────────────────────────────────────────────────────────
    tt_yw, tt_yb = _prep_yuv_weights(device, torch_yw, torch_yb, d, DRAM, RM, TILE)
    tt_dw = _prep_dw_weights(device, torch_dw, d, DRAM, RM, TILE, compute, conv_dw)
    tt_fw, tt_fb = _prep_final_weights(device, torch_fw, torch_fb, d, DRAM, RM, TILE, compute, conv_final)

    # ── Forward pass ────────────────────────────────────────────────────────────
    tt_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)

    # Stage 1: YUV Adapter  →  [1, 3, 1536, 1536]  TILE  DRAM
    tt17 = run_yuv_adapter(device, tt_input, tt_yw, tt_yb, d)

    # Stage 2 (PS): Y pixel_unshuffle  →  [1, 16, 384, 384]  ROW_MAJOR  DRAM
    tt_y = run_y_path_pixel_unshuffle(device, tt17, d)

    # Stage 3a: UV Avgpool + unpack  →  [N, 2, 768, 768]  ROW_MAJOR  DRAM
    tt_rm_uv, tt_unpack_uv = run_uv_avgpool(device, tt17, tt_dw, d, compute, conv_dw)

    # Stage 3b (PS): UV pixel_unshuffle(r=2)  →  [N, 8, 384, 384]  ROW_MAJOR  DRAM
    tt_uv = run_uv_pixel_unshuffle_op(device, tt_rm_uv, tt_unpack_uv, d)

    # Stage 4: Concat + Final Conv  →  [1, 64, 384, 384]  TILE  DRAM
    # concat NCHW dim=1  →  [N, 24, 384, 384]
    tt_cat_nchw = ttnn.concat([tt_y, tt_uv], dim=1, memory_config=DRAM)
    ttnn.deallocate(tt_y)
    ttnn.deallocate(tt_uv)

    # single permute NCHW → NHWC  →  [N, 384, 384, 24]
    tt_cat_nhwc = ttnn.permute(tt_cat_nchw, (0, 2, 3, 1), memory_config=DRAM)
    ttnn.deallocate(tt_cat_nchw)

    # flatten for conv2d  →  [N, 1, 147456, 24]  (free view)
    tt_cat_flat = ttnn.reshape(tt_cat_nhwc, (d.batch, 1, d.concat_sp, d.final_ic))

    # final conv2d IC=24, OC=64, k=1×1
    tt_out = ttnn.conv2d(
        input_tensor=tt_cat_flat,
        weight_tensor=tt_fw,
        in_channels=d.final_ic,
        out_channels=d.final_oc,
        device=device,
        bias_tensor=tt_fb,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=d.batch,
        input_height=d.out_h,
        input_width=d.out_w,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=conv_final,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    ttnn.deallocate(tt_cat_nhwc)  # safe: conv2d finished reading tt_cat_flat (view of tt_cat_nhwc)
    ttnn.deallocate(tt_fw)
    ttnn.deallocate(tt_fb)

    # reshape NHWC unpack + permute NHWC→NCHW  →  [N, 64, 384, 384]  TILE  DRAM
    tt_out = ttnn.to_memory_config(
        ttnn.permute(
            ttnn.reshape(tt_out, (d.batch, d.out_h, d.out_w, d.final_oc)),
            (0, 3, 1, 2),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        ),
        DRAM,
    )

    # ── Verify ──────────────────────────────────────────────────────────────────
    result = ttnn.to_torch(ttnn.to_layout(tt_out, RM, memory_config=DRAM))
    ttnn.deallocate(tt_out)

    assert list(result.shape) == [batch, d.final_oc, d.out_h, d.out_w]
    pcc = torch.corrcoef(torch.stack([result.float().flatten(), golden.flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99"


# ===========================================================================
# TEST 3 — pixel_unshuffle Y + UV + spatial-packed final conv K=4  (DRAM)
# ===========================================================================
#
# Copies Test 2's code exactly, replacing only the concat+conv section with
# spatial packing K=4: IC=24 → IC×K=96=3×TILE_WIDTH (0% tile-column waste vs 25%).
#
# Pack:   reshape[N,96,96,384] → permute→NHWC → flatten[N,1,36864,96]  (DRAM)
# Conv:   IC=96 OC=256, Conv2dL1FullSliceConfig  (DRAM)
# Unpack: reshape→permute{0,3,1,2}→reshape  →  [N,64,384,384]  (DRAM)
#         256×96×384 = 64×384×384  ✓  (no 5D permute)


@pytest.mark.parametrize("batch, yuv_ic, yuv_oc, input_h, input_w", [_CONFIG])
def test_yuv_concat_conv_cam0_block_A_pixel_unshuffle_sp_conv(device, batch, yuv_ic, yuv_oc, input_h, input_w):
    """pixel_unshuffle Y+UV, spatial-packed final conv K=4, all ops in DRAM."""
    DRAM, RM, TILE = ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
    K = _FINAL_CONV_K  # 4
    d = make_dims(batch, yuv_ic, yuv_oc, input_h, input_w)
    compute, conv_dw, _ = _make_conv_configs(device)

    ic_p = d.final_ic * K
    oc_p = d.final_oc * K  # 96, 256
    h_pk = d.out_h // K
    sp_pk = h_pk * d.out_w  # 96, 36864

    conv_sp = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=TILE,
        deallocate_activation=True,
        act_block_h_override=0,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )

    # ── Torch tensors ──────────────────────────────────────────────────────────
    torch.manual_seed(42)
    x = torch.randn(batch, yuv_ic, input_h, input_w, dtype=torch.bfloat16)
    torch_yw = torch.randn(yuv_oc, yuv_ic, 1, 1, dtype=torch.bfloat16)
    torch_yb = torch.randn(1, 1, 1, yuv_oc, dtype=torch.bfloat16)
    torch_dw = torch.full((d.uv_c, 1, 2, 1), 0.5, dtype=torch.bfloat16)
    torch_fw = torch.randn(d.final_oc, d.final_ic, 1, 1, dtype=torch.bfloat16)
    torch_fb = torch.randn(1, 1, 1, d.final_oc, dtype=torch.bfloat16)

    # ── CPU golden ──────────────────────────────────────────────────────────────
    yuv_out = F.conv2d(x.float(), torch_yw.reshape(yuv_oc, yuv_ic, 1, 1).float(), bias=torch_yb.reshape(yuv_oc).float())
    y_us = torch.nn.PixelUnshuffle(4)(yuv_out[:, 0:1])
    uv_avg = F.avg_pool2d(yuv_out[:, 1:3], kernel_size=(2, 1), stride=(2, 2))
    uv_us = torch.nn.PixelUnshuffle(d.r_uv)(uv_avg)
    cat_nchw = torch.cat([y_us, uv_us], dim=1)
    golden = F.conv2d(
        cat_nchw.float(),
        torch_fw.reshape(d.final_oc, d.final_ic, 1, 1).float(),
        bias=torch_fb.reshape(d.final_oc).float(),
    )

    # ── Device weights ──────────────────────────────────────────────────────────
    w_2d = _pack_weight_pointwise(torch_fw, d.final_ic, d.final_oc, K)
    b_1d = _pack_bias_pointwise(torch_fb, d.final_oc, K)
    w_sp = w_2d.T.contiguous().reshape(oc_p, ic_p, 1, 1)
    b_sp = b_1d.reshape(1, 1, 1, oc_p)

    tt_yw, tt_yb = _prep_yuv_weights(device, torch_yw, torch_yb, d, DRAM, RM, TILE)
    tt_dw = _prep_dw_weights(device, torch_dw, d, DRAM, RM, TILE, compute, conv_dw)
    tt_fw_sp = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w_sp, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=ic_p,
        out_channels=oc_p,
        batch_size=batch,
        input_height=h_pk,
        input_width=d.out_w,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_sp,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_fb_sp = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(b_sp, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        in_channels=ic_p,
        out_channels=oc_p,
        batch_size=batch,
        input_height=h_pk,
        input_width=d.out_w,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_sp,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    # ── Forward pass ────────────────────────────────────────────────────────────
    tt_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)
    tt17 = run_yuv_adapter(device, tt_input, tt_yw, tt_yb, d)
    tt_y = run_y_path_pixel_unshuffle(device, tt17, d)
    tt_rm, tt_unp = run_uv_avgpool(device, tt17, tt_dw, d, compute, conv_dw)
    tt_uv = run_uv_pixel_unshuffle_op(device, tt_rm, tt_unp, d)

    # concat NCHW dim=1  →  [N, 24, 384, 384]  DRAM
    tt_cat = ttnn.concat([tt_y, tt_uv], dim=1, memory_config=DRAM)
    ttnn.deallocate(tt_y)
    ttnn.deallocate(tt_uv)

    # PACK: reshape→permute→reshape  (all DRAM)
    tt = ttnn.reshape(tt_cat, (batch, ic_p, h_pk, d.out_w))  # [N,96,96,384] free view
    tt = ttnn.permute(tt, (0, 2, 3, 1), memory_config=DRAM)  # [N,96,384,96]
    tt_pack = ttnn.reshape(tt, (batch, 1, sp_pk, ic_p))  # [N,1,36864,96]
    ttnn.deallocate(tt_cat)

    tt_conv = ttnn.conv2d(
        input_tensor=tt_pack,
        weight_tensor=tt_fw_sp,
        in_channels=ic_p,
        out_channels=oc_p,
        device=device,
        bias_tensor=tt_fb_sp,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=batch,
        input_height=h_pk,
        input_width=d.out_w,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=conv_sp,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    ttnn.deallocate(tt_fw_sp)
    ttnn.deallocate(tt_fb_sp)

    # UNPACK: reshape→permute→reshape  (all DRAM)
    tt = ttnn.reshape(tt_conv, (batch, h_pk, d.out_w, oc_p))  # [N,96,384,256]
    tt = ttnn.permute(tt, (0, 3, 1, 2), memory_config=DRAM)  # [N,256,96,384]
    ttnn.deallocate(tt_conv)
    tt_out = ttnn.reshape(tt, (batch, d.final_oc, d.out_h, d.out_w))  # [N,64,384,384] ✓
    tt_out = ttnn.to_layout(tt_out, RM, memory_config=DRAM)

    result = ttnn.to_torch(tt_out)
    ttnn.deallocate(tt_out)

    assert list(result.shape) == [batch, d.final_oc, d.out_h, d.out_w]
    pcc = torch.corrcoef(torch.stack([result.float().flatten(), golden.float().flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99"


# ===========================================================================
# TEST 4 — pixel_unshuffle Y + UV + spatial-packed final conv K=4  (max L1)
# ===========================================================================
#
# Copies Test 3's code exactly, adding L1 optimizations:
#   tt17 → L1 after YUV adapter  (UV permute reads L1 not DRAM, -0.151 ms)
#   concat → L1                  (pack permute reads L1, -0.056 ms)
#   pack permute → L1            (L1→L1 faster)
#   tilize in L1 → shard L1_IL→L1_HS for conv
#   conv in L1_HS                (matmul IC=96 tile-aligned ~0.045 ms)
#   un-shard L1_HS→L1_IL
#   unpack permute → L1          (L1→L1 faster)
#   final to_layout RM in L1, then to_memory_config DRAM


@pytest.mark.parametrize("batch, yuv_ic, yuv_oc, input_h, input_w", [_CONFIG])
def test_yuv_concat_conv_cam0_block_A_pixel_unshuffle_sp_conv_l1(device, batch, yuv_ic, yuv_oc, input_h, input_w):
    """pixel_unshuffle Y+UV, spatial-packed K=4 final conv, pack/conv/unpack in L1."""
    DRAM, L1, RM, TILE = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
    K = _FINAL_CONV_K  # 4
    d = make_dims(batch, yuv_ic, yuv_oc, input_h, input_w)
    compute, conv_dw, _ = _make_conv_configs(device)

    ic_p = d.final_ic * K
    oc_p = d.final_oc * K  # 96, 256
    h_pk = d.out_h // K
    sp_pk = h_pk * d.out_w  # 96, 36864

    conv_sp = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=TILE,
        deallocate_activation=True,
        act_block_h_override=0,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )

    # ── Torch tensors ──────────────────────────────────────────────────────────
    torch.manual_seed(42)
    x = torch.randn(batch, yuv_ic, input_h, input_w, dtype=torch.bfloat16)
    torch_yw = torch.randn(yuv_oc, yuv_ic, 1, 1, dtype=torch.bfloat16)
    torch_yb = torch.randn(1, 1, 1, yuv_oc, dtype=torch.bfloat16)
    torch_dw = torch.full((d.uv_c, 1, 2, 1), 0.5, dtype=torch.bfloat16)
    torch_fw = torch.randn(d.final_oc, d.final_ic, 1, 1, dtype=torch.bfloat16)
    torch_fb = torch.randn(1, 1, 1, d.final_oc, dtype=torch.bfloat16)

    # ── CPU golden ──────────────────────────────────────────────────────────────
    yuv_out = F.conv2d(x.float(), torch_yw.reshape(yuv_oc, yuv_ic, 1, 1).float(), bias=torch_yb.reshape(yuv_oc).float())
    y_us = torch.nn.PixelUnshuffle(4)(yuv_out[:, 0:1])
    uv_avg = F.avg_pool2d(yuv_out[:, 1:3], kernel_size=(2, 1), stride=(2, 2))
    uv_us = torch.nn.PixelUnshuffle(d.r_uv)(uv_avg)
    cat_nchw = torch.cat([y_us, uv_us], dim=1)
    golden = F.conv2d(
        cat_nchw.float(),
        torch_fw.reshape(d.final_oc, d.final_ic, 1, 1).float(),
        bias=torch_fb.reshape(d.final_oc).float(),
    )

    # ── Device weights ──────────────────────────────────────────────────────────
    w_2d = _pack_weight_pointwise(torch_fw, d.final_ic, d.final_oc, K)
    b_1d = _pack_bias_pointwise(torch_fb, d.final_oc, K)
    w_sp = w_2d.T.contiguous().reshape(oc_p, ic_p, 1, 1)
    b_sp = b_1d.reshape(1, 1, 1, oc_p)

    tt_yw, tt_yb = _prep_yuv_weights(device, torch_yw, torch_yb, d, DRAM, RM, TILE)
    tt_dw = _prep_dw_weights(device, torch_dw, d, DRAM, RM, TILE, compute, conv_dw)
    tt_fw_sp = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(w_sp, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        weights_format="OIHW",
        in_channels=ic_p,
        out_channels=oc_p,
        batch_size=batch,
        input_height=h_pk,
        input_width=d.out_w,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_sp,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    tt_fb_sp = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(b_sp, dtype=ttnn.bfloat16, layout=RM),
        input_memory_config=DRAM,
        input_layout=TILE,
        in_channels=ic_p,
        out_channels=oc_p,
        batch_size=batch,
        input_height=h_pk,
        input_width=d.out_w,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_sp,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    tt_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=DRAM)
    _tt17d = run_yuv_adapter(device, tt_input, tt_yw, tt_yb, d)
    # OPT-D: tt17 to L1 — UV permute reads L1 not DRAM  14.16MB/64=221KB/bank ✓
    tt17 = ttnn.to_memory_config(_tt17d, L1)
    ttnn.deallocate(_tt17d)
    # Y pixel_unshuffle (DRAM) → move output to L1  [N,16,384,384] RM  4.72MB/64=73.7KB/bank ✓
    tt_y_dram = run_y_path_pixel_unshuffle(device, tt17, d)
    tt_y = ttnn.to_memory_config(tt_y_dram, L1)
    ttnn.deallocate(tt_y_dram)

    # UV avgpool+pixel_unshuffle stay in DRAM (pixel_unshuffle kernel requires DRAM input)
    # → move final UV output to L1  [N,8,384,384] RM  2.36MB/64=36.9KB/bank ✓
    # peak during concat: tt_y(4.72MB) + tt_uv(2.36MB) = 7.08MB/64=110.6KB/bank ✓
    tt_rm, tt_unp = run_uv_avgpool(device, tt17, tt_dw, d, compute, conv_dw)  # tt17 deallocated inside
    tt_uv_dram = run_uv_pixel_unshuffle_op(device, tt_rm, tt_unp, d)
    tt_uv = ttnn.to_memory_config(tt_uv_dram, L1)
    ttnn.deallocate(tt_uv_dram)

    # concat L1→L1  [N,24,384,384] RM L1  — no DRAM reads for either input
    tt_cat = ttnn.concat([tt_y, tt_uv], dim=1, memory_config=L1)
    ttnn.deallocate(tt_y)
    ttnn.deallocate(tt_uv)

    # PACK — L1→L1 throughout (no DRAM reads)
    tt = ttnn.reshape(tt_cat, (batch, ic_p, h_pk, d.out_w))  # [N,96,96,384] free view
    tt = ttnn.permute(tt, (0, 2, 3, 1), memory_config=L1)  # [N,96,384,96] L1
    tt_pack = ttnn.reshape(tt, (batch, 1, sp_pk, ic_p))  # [N,1,36864,96] L1 free view
    ttnn.deallocate(tt_cat)

    # Shard to L1_HS for conv: 36864/64=576 rows × 96 ch = 110 KB/core ✓
    cgrid = device.compute_with_storage_grid_size()
    ncores = cgrid.x * cgrid.y
    shard_h = math.ceil(sp_pk / ncores)  # 576
    grid_rs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cgrid.x - 1, cgrid.y - 1))})
    shard_spec = ttnn.ShardSpec(grid_rs, [shard_h, ic_p], ttnn.ShardOrientation.ROW_MAJOR)
    l1_hs_in = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    # tilize in L1 → shard L1_IL→L1_HS  (intra-L1, fast)
    tt_tile_l1 = ttnn.to_layout(tt_pack, TILE, memory_config=L1)
    tt_pack_l1 = ttnn.to_memory_config(tt_tile_l1, l1_hs_in)
    ttnn.deallocate(tt_pack)
    ttnn.deallocate(tt_tile_l1)

    # conv2d in L1_HS — matmul IC=96 (tile-aligned) → L1_HS output
    tt_conv = ttnn.conv2d(
        input_tensor=tt_pack_l1,
        weight_tensor=tt_fw_sp,
        in_channels=ic_p,
        out_channels=oc_p,
        device=device,
        bias_tensor=tt_fb_sp,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=batch,
        input_height=h_pk,
        input_width=d.out_w,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=conv_sp,
        compute_config=compute,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )
    ttnn.deallocate(tt_fw_sp)
    ttnn.deallocate(tt_fb_sp)

    # un-shard L1_HS → L1_IL  (18.87 MB / 64 = 295 KB/core ✓)
    tt_conv_l1 = ttnn.to_memory_config(tt_conv, L1)
    ttnn.deallocate(tt_conv)

    # UNPACK — all in L1  (L1→L1, fast; 295 KB/core ✓)
    tt = ttnn.reshape(tt_conv_l1, (batch, h_pk, d.out_w, oc_p))  # [N,96,384,256] L1 free view
    tt = ttnn.permute(tt, (0, 3, 1, 2), memory_config=L1)  # [N,256,96,384] L1
    ttnn.deallocate(tt_conv_l1)
    tt = ttnn.reshape(tt, (batch, d.final_oc, d.out_h, d.out_w))  # [N,64,384,384] L1 free view
    tt_out = ttnn.to_memory_config(ttnn.to_layout(tt, RM, memory_config=L1), DRAM)

    result = ttnn.to_torch(tt_out)
    ttnn.deallocate(tt_out)

    assert list(result.shape) == [batch, d.final_oc, d.out_h, d.out_w]
    pcc = torch.corrcoef(torch.stack([result.float().flatten(), golden.float().flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99"
