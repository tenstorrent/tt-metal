# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn.functional as F
import ttnn


# ---------------------------------------------------------------------------
# UV AveragePool DRAM Bottleneck
#
# In the BEV model the UV channels (C=2) are downsampled 2× via an ONNX
# AveragePool(kernel=[2,1], stride=[2,2]) which the compiler lowers to a
# depthwise conv2d.  TTNN conv2d requires NHWC input, so the compiler inserts:
#
#   to_layout TILE          [N,C,H,W]  ROW_MAJOR → TILE       no waste (C in row-dim)
#   permute {0,2,3,1}  ★   [N,C,H,W]  TILE → [N,H,W,C] TILE  C=2→32 padded  16× inflation
#   reshape [1,1,H*W,C]     flatten spatial for conv2d
#   conv2d(groups=C, k=[2,1], s=[2,2], dram_height_slice)
#   reshape [1,out_H,out_W,C]
#   permute {0,3,1,2}       NHWC → NCHW output  (C=2 back to row-dim, no waste)
#   to_memory_config DRAM
#   to_layout ROW_MAJOR
#
# Bottleneck — the first permute:
#   Block A (1×2×1536×1536):  9.4 MB real → 150.9 MB DRAM write  FPU=8.7%  ~6.5 ms
#   Block C (1×2×1280×2304): 11.8 MB real → 188.7 MB DRAM write  FPU=9.9%  ~7.1 ms
#
# IR sources:
#   uv_avgpool_block_A_ir/ttnn_uv_avgpool_block_A.mlir
#   uv_avgpool_block_C_ir/ttnn_uv_avgpool_block_C.mlir
# ---------------------------------------------------------------------------

_CONFIGS = [
    pytest.param(1, 2, 2, 1536, 1536, id="avgpool_1_1x2x1536x1536"),
    pytest.param(1, 2, 2, 1280, 2304, id="avgpool_2_1x2x1280x2304"),
]

# AveragePool ONNX attributes — identical for both blocks
_KERNEL_SIZE = (2, 1)  # kH=2, kW=1
_STRIDE = (2, 2)  # 2× downsample in both spatial dims
_PADDING = (0, 0, 0, 0)
_DILATION = (1, 1)


@pytest.mark.parametrize(
    "batch, in_channels, out_channels, input_height, input_width",
    _CONFIGS,
)
def test_avgpool_dram_bottleneck(
    device,
    batch,
    in_channels,
    out_channels,
    input_height,
    input_width,
):
    groups = in_channels  # depthwise: each input channel is its own group

    # Exact compute and conv config from IR:
    #   compute_config = hifi3, fp32_dest_acc_en=true
    #   conv2d_config  = weights_dtype=bf16, deallocate_activation=true,
    #                    act_block_h_override=0, enable_kernel_stride_folding=false,
    #                    config_tensors_in_dram=true
    #   slice_config   = dram_height (Conv2dDRAMSliceHeight)
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=True,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.TILE_LAYOUT,
        deallocate_activation=True,
        act_block_h_override=0,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )
    dram_interleaved = ttnn.DRAM_MEMORY_CONFIG

    # Output spatial dimensions:
    #   out_h = floor((H - kH) / sH) + 1 = (H - 2) // 2 + 1 = H // 2  (even H)
    #   out_w = floor((W - kW) / sW) + 1 = (W - 1) // 2 + 1 = W // 2  (even W)
    out_h = input_height // _STRIDE[0]
    out_w = input_width // _STRIDE[1]
    spatial = input_height * input_width

    # Avgpool weight: OIHW [out_C, in_C/groups, kH, kW] = [2, 1, 2, 1]
    # Each of the 2 groups averages 2×1 pixels → weight = 1/(kH×kW) = 0.5
    torch_input = torch.randn(batch, in_channels, input_height, input_width, dtype=torch.bfloat16)
    torch_weight = torch.full(
        (out_channels, in_channels // groups, _KERNEL_SIZE[0], _KERNEL_SIZE[1]),
        1.0 / (_KERNEL_SIZE[0] * _KERNEL_SIZE[1]),
        dtype=torch.bfloat16,
    )

    # Prepare depthwise conv2d weight — has_bias=False (no bias in IR)
    tt_weight = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        input_memory_config=dram_interleaved,
        input_layout=ttnn.TILE_LAYOUT,
        weights_format="OIHW",
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        kernel_size=_KERNEL_SIZE,
        stride=_STRIDE,
        padding=_PADDING,
        dilation=_DILATION,
        has_bias=False,
        groups=groups,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceConfig.SliceTypeEnum.DRAMSliceHeight),
    )

    # ── Exact TTNN IR op sequence ────────────────────────────────────────────

    # IR: %arg0 arrives as ROW_MAJOR DRAM (ttnn_layout2)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=dram_interleaved,
    )

    # IR: %2 = ttnn.to_layout(tile)  →  TilizeDeviceOperation
    #   [N,C,H,W] ROW_MAJOR → TILE DRAM (ttnn_layout4)
    #   C=2 is in the ROW dimension — no tile-column padding, no waste
    #   Block A: 96×48 tile grid = 11.8 MB   Block C: 80×72 tile grid = 11.8 MB
    tt_nchw_tile = ttnn.to_layout(tt_input, layout=ttnn.TILE_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_input)

    # IR: %3 = ttnn.permute {0,2,3,1}  →  ★ BOTTLENECK ★ PermuteDeviceOperation
    #   [N,C,H,W] TILE → [N,H,W,C] TILE DRAM (ttnn_layout5)
    #   C=2 moves to the COLUMN dimension → padded to TILE_WIDTH=32 → 30/32 = 93.8% zeros
    #   Block A: 9.4 MB real data → 150.9 MB DRAM write (73728×1 tile grid)
    #   Block C: 11.8 MB real data → 188.7 MB DRAM write (92160×1 tile grid)
    #   X_PAD[LOGICAL] = 32[2]  FPU%≈9%  BRISC≈FW (pure data-movement)
    tt_nhwc = ttnn.permute(tt_nchw_tile, dims=(0, 2, 3, 1), memory_config=dram_interleaved)
    ttnn.deallocate(tt_nchw_tile)

    # IR: %4 = ttnn.reshape [N,H,W,C] → [1,1,H*W,C]  (ttnn_layout6, same padded TILE)
    tt_flat = ttnn.reshape(tt_nhwc, shape=(batch, 1, spatial, in_channels))
    ttnn.deallocate(tt_nhwc)

    # IR: %5 = ttnn.conv2d(groups=2, k=[2,1], s=[2,2], dram_height_slice)
    #   dram_height slice emits per-slice:
    #     PaddedSliceDeviceOperation   — loads padded NHWC slice from DRAM into L1
    #     HaloDeviceOperation          — no-op for kH=2, kW=1 (no halo needed)
    #     MoveDeviceOperation          — L1 reshape for conv2d kernel
    #     Conv2dDeviceOperation        — actual avgpool compute (FPU%≈0.04%)
    #     SliceWriteDeviceOperation    — writes output slice to DRAM
    #   Block A: 2 slices   Block C: 3 slices
    tt_out = ttnn.conv2d(
        input_tensor=tt_flat,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=None,
        kernel_size=_KERNEL_SIZE,
        stride=_STRIDE,
        padding=_PADDING,
        dilation=_DILATION,
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        groups=groups,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceConfig.SliceTypeEnum.DRAMSliceHeight),
    )
    ttnn.deallocate(tt_flat)
    ttnn.deallocate(tt_weight)

    # IR: %6 = ttnn.reshape [1,1,out_H*out_W,C] → [N,out_H,out_W,C]  (ttnn_layout8)
    tt_out_nhwc = ttnn.reshape(tt_out, shape=(batch, out_h, out_w, out_channels))
    ttnn.deallocate(tt_out)

    # IR: %7 = ttnn.permute {0,3,1,2}  →  PermuteDeviceOperation
    #   [N,out_H,out_W,C] TILE DRAM → [N,C,out_H,out_W] TILE L1  (ttnn_layout9)
    #   C=2 back to row-dim → out_W is tile-aligned → no waste, FPU%≈67%
    tt_nchw_out = ttnn.permute(tt_out_nhwc, dims=(0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(tt_out_nhwc)

    # IR: %8 = ttnn.to_memory_config DRAM  →  CopyDeviceOperation
    #   [N,C,out_H,out_W] TILE L1 → TILE DRAM  (ttnn_layout3)
    tt_output_tile = ttnn.to_memory_config(tt_nchw_out, memory_config=dram_interleaved)
    ttnn.deallocate(tt_nchw_out)

    # UntilizeDeviceOperation (appears inside signposts in Tracy CSV)
    tt_output = ttnn.to_layout(tt_output_tile, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_output_tile)

    result = ttnn.to_torch(tt_output)
    ttnn.deallocate(tt_output)

    assert result.shape == torch.Size(
        [batch, out_channels, out_h, out_w]
    ), f"Shape mismatch: got {tuple(result.shape)}, expected ({batch},{out_channels},{out_h},{out_w})"

    # Golden: depthwise conv2d with avgpool weights — equivalent to F.avg_pool2d
    golden = F.conv2d(
        torch_input.float(),
        torch_weight.float(),
        bias=None,
        stride=_STRIDE,
        padding=(0, 0),
        dilation=_DILATION,
        groups=groups,
    )

    pcc = torch.corrcoef(torch.stack([result.float().flatten(), golden.float().flatten()]))[0, 1].item()
    assert pcc >= 0.99, (
        f"PCC {pcc:.6f} < 0.99 for config " f"({batch}, {in_channels}, {out_channels}, {input_height}, {input_width})"
    )
