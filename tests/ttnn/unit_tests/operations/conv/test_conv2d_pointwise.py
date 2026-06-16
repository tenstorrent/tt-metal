# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import math
import pytest
import torch
import torch.nn.functional as F
import ttnn


_CONFIGS = [
    pytest.param(1, 3, 3, 1536, 1536, id="conv2d_1_1x3x1536x1536"),
    pytest.param(1, 3, 3, 1280, 2304, id="conv2d_2_1x3x1280x2304"),
]


def _make_compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=True,
    )


def _make_conv_config():
    return ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.TILE_LAYOUT,
        deallocate_activation=True,
        act_block_h_override=0,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )


@pytest.mark.parametrize(
    "batch, in_channels, out_channels, input_height, input_width",
    _CONFIGS,
)
def test_conv2d_dram_bottleneck(
    device,
    batch,
    in_channels,
    out_channels,
    input_height,
    input_width,
):
    compute_config = _make_compute_config(device)
    conv_config = _make_conv_config()
    spatial = input_height * input_width
    dram_interleaved = ttnn.DRAM_MEMORY_CONFIG

    torch_input = torch.randn(batch, in_channels, input_height, input_width, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)

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
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    tt_bias = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        input_memory_config=dram_interleaved,
        input_layout=ttnn.TILE_LAYOUT,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=dram_interleaved
    )
    tt_nchw_tile = ttnn.to_layout(tt_input, layout=ttnn.TILE_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_input)
    tt_nhwc = ttnn.permute(tt_nchw_tile, dims=(0, 2, 3, 1), memory_config=dram_interleaved)
    ttnn.deallocate(tt_nchw_tile)
    tt_flat = ttnn.reshape(tt_nhwc, shape=(batch, 1, spatial, in_channels))
    ttnn.deallocate(tt_nhwc)

    [tt_out, [out_h, out_w], [d_w, d_b]] = ttnn.conv2d(
        input_tensor=tt_flat,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=tt_bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    ttnn.deallocate(tt_flat)
    ttnn.deallocate(tt_weight)
    ttnn.deallocate(tt_bias)

    tt_out = ttnn.reshape(tt_out, shape=(batch, out_h, out_w, out_channels))
    tt_nchw_out = ttnn.permute(tt_out, dims=(0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(tt_out)
    tt_output = ttnn.to_memory_config(tt_nchw_out, memory_config=dram_interleaved)
    ttnn.deallocate(tt_nchw_out)

    result = ttnn.to_torch(tt_output)
    assert result.shape == torch.Size((batch, out_channels, input_height, input_width)), (
        f"Shape mismatch: got {tuple(result.shape)}, "
        f"expected ({batch}, {out_channels}, {input_height}, {input_width})"
    )
    ttnn.deallocate(tt_output)


@pytest.mark.parametrize(
    "batch, in_channels, out_channels, input_height, input_width",
    _CONFIGS,
)
def test_conv2d_only(
    device,
    batch,
    in_channels,
    out_channels,
    input_height,
    input_width,
):
    compute_config = _make_compute_config(device)
    conv_config = _make_conv_config()
    spatial = input_height * input_width
    dram_interleaved = ttnn.DRAM_MEMORY_CONFIG

    torch_weight = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)
    torch_input_flat = torch.randn(batch, 1, spatial, in_channels, dtype=torch.bfloat16)

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
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    tt_bias = ttnn.prepare_conv_bias(
        bias_tensor=ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        input_memory_config=dram_interleaved,
        input_layout=ttnn.TILE_LAYOUT,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )

    tt_input = ttnn.from_torch(
        torch_input_flat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_interleaved,
    )

    [tt_out, [out_h, out_w], [d_w, d_b]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=tt_bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        groups=1,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=ttnn.Conv2dL1FullSliceConfig,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_weight)
    ttnn.deallocate(tt_bias)

    result = ttnn.to_torch(tt_out)
    assert result.shape == torch.Size((batch, 1, spatial, out_channels)), (
        f"Shape mismatch: got {tuple(result.shape)}, " f"expected ({batch}, 1, {spatial}, {out_channels})"
    )
    ttnn.deallocate(tt_out)


# ---------------------------------------------------------------------------
# Method 2 — Spatial Packing
#
# For a 1x1 / stride=1 / pad=0 pointwise conv with C < TILE_WIDTH (32):
#   C=3 in NHWC TILE pads each tile row to 32, wasting 29/32 = 90.6% of every
#   tile read.  Packing K=32 adjacent spatial pixels into one "super-pixel" with
#   C*K=96 channels fills tile rows completely (96 = 3 x TILE_WIDTH, 0% waste),
#   reducing activation DRAM reads from 188.7 MB to 17.7 MB while keeping the
#   single-kernel matmul path (no DRAM slicing overhead).
#
# Packing factor K satisfies: C * K ≡ 0 (mod TILE_WIDTH)
#   K = TILE_WIDTH // gcd(C, TILE_WIDTH) = 32 // gcd(3,32) = 32
#   C * K = 96  →  3 full tile columns, zero padding waste.
#
# Weight: block-diagonal [C*K, OC*K] with K copies of the original [C, OC]
#   weight on the diagonal — preserves spatial independence (no pixel mixing).
#
# Bias: K-tiled repetition of original bias.
# ---------------------------------------------------------------------------

TILE_WIDTH = 32


def _spatial_pack_factor(in_channels: int) -> int:
    return TILE_WIDTH // math.gcd(in_channels, TILE_WIDTH)


def _make_packed_weight(torch_weight: torch.Tensor, in_channels: int, out_channels: int, K: int) -> torch.Tensor:
    # Row-group packing: [N,C,H,W] → [N,C*K,H/K,W].
    # Packed channel c'=c*K+k holds row-group k of original channel c.
    # Required: W_packed[c*K+k, oc*K+k] = W[oc,c]  for all k, 0 elsewhere.
    # Efficiently set via stride-K slicing: W_block[k::K, k::K] = W_orig.T
    import numpy as np

    W_orig = torch_weight.reshape(out_channels, in_channels).float().numpy()  # [OC, IC]
    W_block = np.zeros((in_channels * K, out_channels * K), dtype=np.float32)
    for k in range(K):
        W_block[k::K, k::K] = W_orig.T  # W_orig.T[c, oc] = W_orig[oc, c] = W[oc, c]
    return torch.from_numpy(W_block).to(torch.bfloat16)


def _make_packed_bias(torch_bias: torch.Tensor, out_channels: int, K: int) -> torch.Tensor:
    # bias_packed[oc*K+k] = bias[oc]  →  repeat_interleave, not repeat
    return torch_bias.reshape(out_channels).repeat_interleave(K)


# ---------------------------------------------------------------------------
# Method 2 — Approach 2: Row-Packing via on-device reshape + permute
#
# Instead of packing on the host and sending a flat [H*W/K, C*K] tensor,
# this variant does the spatial packing entirely on device:
#
#   [N, C, H, W]  ROW_MAJOR
#     → reshape  [N, C*K, H/K, W]   (ROW_MAJOR — free view, no data copy)
#     → permute  [N, H/K, W, C*K]   (NHWC, 17.7 MB — vs 188.7 MB baseline permute)
#     → reshape  [1, 1, N*H/K*W, C*K]  (flatten spatial — free view)
#     → to_layout TILE               (17.7 MB, C*K=96 fills tiles 100%)
#     → linear   [C*K → OC*K]        (reads 17.7 MB activation)
#
# Host unpack: [1,1,N*H/K*W,OC*K] → reshape [N,H/K,W,OC*K]
#            → permute [N,OC*K,H/K,W] → reshape [N,OC,H,W]
#
# Key difference from Method 2 Approach 1:
#   - Approach 1: host pack (torch views) + from_torch packed tensor
#   - Approach 2: from_torch NCHW + on-device reshape (view) + 17.7 MB permute
#
# The on-device permute [N,C*K,H/K,W] → [N,H/K,W,C*K] writes only 17.7 MB
# (C*K=96 = 3 full tile widths, 0% padding waste) vs the baseline permute
# which writes 188.7 MB (C=3 → 32, 10.7× inflation).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch, in_channels, out_channels, input_height, input_width",
    _CONFIGS,
)
def test_conv2d_method2_approach2_dram_bottleneck(
    device,
    batch,
    in_channels,
    out_channels,
    input_height,
    input_width,
):
    dram_interleaved = ttnn.DRAM_MEMORY_CONFIG
    K = _spatial_pack_factor(in_channels)
    assert input_height % K == 0, f"input_height={input_height} must be divisible by K={K}"

    packed_ic = in_channels * K
    packed_oc = out_channels * K
    packed_h = input_height // K
    packed_spatial = packed_h * input_width

    torch_input = torch.randn(batch, in_channels, input_height, input_width, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)

    torch_w_packed = _make_packed_weight(torch_weight, in_channels, out_channels, K)
    torch_b_packed = _make_packed_bias(torch_bias, out_channels, K)

    # Weight [1,1,C*K,OC*K] and bias [1,1,1,OC*K] prepared on host, moved to device TILE
    tt_weight = ttnn.from_torch(
        torch_w_packed.reshape(1, 1, packed_ic, packed_oc),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_interleaved,
    )
    tt_bias = ttnn.from_torch(
        torch_b_packed.reshape(1, 1, 1, packed_oc),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_interleaved,
    )

    # Pack on device — full on-device pipeline.
    # Same deallocation fix: never free a reshape source while its view is in use.
    #   [N,C,H,W] ROW_MAJOR
    #     → reshape [N,C*K,H/K,W]   ROW_MAJOR view  (deallocate AFTER permute)
    #     → permute [N,H/K,W,C*K]   ROW_MAJOR       (17.7 MB)
    #     → reshape [1,1,sp,C*K]    ROW_MAJOR view  (free)
    #     → to_layout TILE                           (17.7 MB, C*K=96 fills tiles 100%)

    tt_nchw = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=dram_interleaved
    )

    # reshape → view of tt_nchw's buffer
    tt_packed_nchw = ttnn.reshape(tt_nchw, (batch, packed_ic, packed_h, input_width))
    # permute — consume the view; deallocate tt_nchw AFTER this op
    tt_nhwc = ttnn.permute(tt_packed_nchw, dims=(0, 2, 3, 1), memory_config=dram_interleaved)
    ttnn.deallocate(tt_nchw)  # safe: permute has finished reading tt_packed_nchw (view)
    ttnn.deallocate(tt_packed_nchw)

    # reshape [N,H/K,W,C*K] → [1,1,packed_sp,C*K]: view of tt_nhwc's buffer
    tt_flat = ttnn.reshape(tt_nhwc, (batch, 1, packed_spatial, packed_ic))
    # tilize — consumes the view; deallocate tt_nhwc AFTER to_layout finishes
    tt_tile = ttnn.to_layout(tt_flat, layout=ttnn.TILE_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_nhwc)  # safe: to_layout has read tt_flat (view of tt_nhwc)
    ttnn.deallocate(tt_flat)

    # Step 6: single matmul — reads only 17.7 MB (vs 188.7 MB baseline)
    tt_out_packed = ttnn.linear(tt_tile, tt_weight, bias=tt_bias, memory_config=dram_interleaved)
    ttnn.deallocate(tt_tile)
    ttnn.deallocate(tt_weight)
    ttnn.deallocate(tt_bias)

    # Unpack on device — full on-device pipeline.
    # Key fix: ttnn.reshape on ROW_MAJOR returns a VIEW sharing the same buffer.
    # Never deallocate the reshape source before the view is consumed by the next op.
    #   [1,1,packed_sp,OC*K] TILE
    #     → untilize ROW_MAJOR                           (17.7 MB)
    #     → reshape  [N,H/K,W,OC*K]    ROW_MAJOR view   (deallocate AFTER permute)
    #     → permute  [N,OC*K,H/K,W]    ROW_MAJOR        (14.2 MB)
    #     → reshape  [N,OC,H,W]        ROW_MAJOR view

    # Step 7: untilize
    tt_out_rm = ttnn.to_layout(tt_out_packed, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_out_packed)

    # Step 8: reshape → view of tt_out_rm's buffer
    tt_out_nhwc = ttnn.reshape(tt_out_rm, (batch, packed_h, input_width, packed_oc))
    # Step 9: permute — consume the view; deallocate tt_out_rm AFTER this op
    tt_out_nchw_packed = ttnn.permute(tt_out_nhwc, dims=(0, 3, 1, 2), memory_config=dram_interleaved)
    ttnn.deallocate(tt_out_rm)  # safe: permute has finished reading tt_out_nhwc (view of tt_out_rm)
    ttnn.deallocate(tt_out_nhwc)

    # Step 10: reshape → view of tt_out_nchw_packed's buffer
    tt_output = ttnn.reshape(tt_out_nchw_packed, (batch, out_channels, input_height, input_width))
    result = ttnn.to_torch(tt_output)  # consume the view before deallocating
    ttnn.deallocate(tt_out_nchw_packed)  # safe: to_torch has read tt_output (view of this)
    ttnn.deallocate(tt_output)

    assert result.shape == torch.Size(
        (batch, out_channels, input_height, input_width)
    ), f"Shape mismatch: got {tuple(result.shape)}, expected ({batch}, {out_channels}, {input_height}, {input_width})"

    # Golden comparison: reference 1×1 conv2d using PyTorch
    golden = F.conv2d(
        torch_input.float(),
        torch_weight.float().reshape(out_channels, in_channels, 1, 1),
        bias=torch_bias.float().reshape(out_channels),
        stride=1,
        padding=0,
    )  # [N, OC, H, W]

    result_flat = result.float().flatten()
    golden_flat = golden.float().flatten()
    pcc = torch.corrcoef(torch.stack([result_flat, golden_flat]))[0, 1].item()
    assert pcc >= 0.99, (
        f"PCC {pcc:.6f} is below threshold 0.99 for config "
        f"({batch}, {in_channels}, {out_channels}, {input_height}, {input_width})"
    )


# ---------------------------------------------------------------------------
# Method 2 Approach 2 + Opportunity 3: L1 HEIGHT_SHARDED matmul
#
# Same spatial packing as Approach 2 but adds a reshard step after tilize:
#   DRAM INTERLEAVED TILE [1,1,packed_sp,C*K]
#     → to_memory_config L1 HEIGHT_SHARDED [per_core_rows, C*K] per core
#     → linear on L1 sharded input → DRAM output
#
# Hypothesis: L1 sharded input eliminates DRAM broadcast overhead in the
# matmul, reducing MatmulDeviceOperation from 0.43 ms.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch, in_channels, out_channels, input_height, input_width",
    _CONFIGS,
)
def test_conv2d_method2_dram_sharded_matmul(
    device,
    batch,
    in_channels,
    out_channels,
    input_height,
    input_width,
):
    dram_interleaved = ttnn.DRAM_MEMORY_CONFIG
    K = _spatial_pack_factor(in_channels)
    assert input_height % K == 0

    packed_ic = in_channels * K
    packed_oc = out_channels * K
    packed_h = input_height // K
    packed_spatial = packed_h * input_width

    torch_input = torch.randn(batch, in_channels, input_height, input_width, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)

    torch_w_packed = _make_packed_weight(torch_weight, in_channels, out_channels, K)
    torch_b_packed = _make_packed_bias(torch_bias, out_channels, K)

    tt_weight = ttnn.from_torch(
        torch_w_packed.reshape(1, 1, packed_ic, packed_oc),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_interleaved,
    )
    tt_bias = ttnn.from_torch(
        torch_b_packed.reshape(1, 1, 1, packed_oc),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_interleaved,
    )

    # Pack (identical to Approach 2)
    tt_nchw = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=dram_interleaved
    )
    tt_packed_nchw = ttnn.reshape(tt_nchw, (batch, packed_ic, packed_h, input_width))
    tt_nhwc = ttnn.permute(tt_packed_nchw, dims=(0, 2, 3, 1), memory_config=dram_interleaved)
    ttnn.deallocate(tt_nchw)
    ttnn.deallocate(tt_packed_nchw)
    tt_flat = ttnn.reshape(tt_nhwc, (batch, 1, packed_spatial, packed_ic))
    tt_tile = ttnn.to_layout(tt_flat, layout=ttnn.TILE_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_nhwc)
    ttnn.deallocate(tt_flat)

    # Opportunity 3: reshard to L1 HEIGHT_SHARDED before matmul
    # Each core gets [packed_spatial/64, packed_ic] activation in L1
    num_cores = 64
    sharded_l1 = ttnn.create_sharded_memory_config(
        shape=(packed_spatial, packed_ic),
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_tile_l1 = ttnn.to_memory_config(tt_tile, sharded_l1)
    ttnn.deallocate(tt_tile)

    # Matmul: each core reads its own L1 shard, no DRAM broadcast
    tt_out_packed = ttnn.linear(tt_tile_l1, tt_weight, bias=tt_bias, memory_config=dram_interleaved)
    ttnn.deallocate(tt_tile_l1)
    ttnn.deallocate(tt_weight)
    ttnn.deallocate(tt_bias)

    # Unpack (identical to Approach 2)
    tt_out_rm = ttnn.to_layout(tt_out_packed, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_out_packed)
    tt_out_nhwc = ttnn.reshape(tt_out_rm, (batch, packed_h, input_width, packed_oc))
    tt_out_nchw_packed = ttnn.permute(tt_out_nhwc, dims=(0, 3, 1, 2), memory_config=dram_interleaved)
    ttnn.deallocate(tt_out_rm)
    ttnn.deallocate(tt_out_nhwc)
    tt_output = ttnn.reshape(tt_out_nchw_packed, (batch, out_channels, input_height, input_width))
    result = ttnn.to_torch(tt_output)
    ttnn.deallocate(tt_out_nchw_packed)
    ttnn.deallocate(tt_output)

    assert result.shape == torch.Size((batch, out_channels, input_height, input_width))

    golden = F.conv2d(
        torch_input.float(),
        torch_weight.float().reshape(out_channels, in_channels, 1, 1),
        bias=torch_bias.float().reshape(out_channels),
        stride=1,
        padding=0,
    )
    pcc = torch.corrcoef(torch.stack([result.float().flatten(), golden.float().flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99"


# ---------------------------------------------------------------------------
# Method 2 Approach 2 + Opportunity 4: HEIGHT_SHARDED Tilize
#
# Combines Opportunity 3 + 4: tilizes DIRECTLY to L1 HEIGHT_SHARDED,
# eliminating the separate InterleavedToShardedDeviceOperation step.
# The matmul then reads from L1 (not DRAM).
#
#   ROW_MAJOR → to_layout(TILE, L1 HEIGHT_SHARDED)  ← one op: tilize + shard
#   → linear (reads from L1 per-core shard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch, in_channels, out_channels, input_height, input_width",
    _CONFIGS,
)
def test_conv2d_method2_height_sharded_tilize(
    device,
    batch,
    in_channels,
    out_channels,
    input_height,
    input_width,
):
    dram_interleaved = ttnn.DRAM_MEMORY_CONFIG
    K = _spatial_pack_factor(in_channels)
    assert input_height % K == 0

    packed_ic = in_channels * K
    packed_oc = out_channels * K
    packed_h = input_height // K
    packed_spatial = packed_h * input_width

    torch_input = torch.randn(batch, in_channels, input_height, input_width, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, 1, 1, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)

    torch_w_packed = _make_packed_weight(torch_weight, in_channels, out_channels, K)
    torch_b_packed = _make_packed_bias(torch_bias, out_channels, K)

    tt_weight = ttnn.from_torch(
        torch_w_packed.reshape(1, 1, packed_ic, packed_oc),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_interleaved,
    )
    tt_bias = ttnn.from_torch(
        torch_b_packed.reshape(1, 1, 1, packed_oc),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_interleaved,
    )

    # Pack (identical to Approach 2)
    tt_nchw = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=dram_interleaved
    )
    tt_packed_nchw = ttnn.reshape(tt_nchw, (batch, packed_ic, packed_h, input_width))
    tt_nhwc = ttnn.permute(tt_packed_nchw, dims=(0, 2, 3, 1), memory_config=dram_interleaved)
    ttnn.deallocate(tt_nchw)
    ttnn.deallocate(tt_packed_nchw)
    tt_flat = ttnn.reshape(tt_nhwc, (batch, 1, packed_spatial, packed_ic))
    ttnn.deallocate(tt_nhwc)

    # Opportunity 4: tilize DIRECTLY to L1 HEIGHT_SHARDED
    # Reads 17.7 MB from DRAM ROW_MAJOR, writes to L1 per-core shards.
    # No separate InterleavedToShardedDeviceOperation needed.
    sharded_l1 = ttnn.create_sharded_memory_config(
        shape=(packed_spatial, packed_ic),
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_tile_l1 = ttnn.to_layout(tt_flat, ttnn.TILE_LAYOUT, memory_config=sharded_l1)
    ttnn.deallocate(tt_flat)

    # Matmul reads from L1 (each core uses its local shard, no DRAM read)
    tt_out_packed = ttnn.linear(tt_tile_l1, tt_weight, bias=tt_bias, memory_config=dram_interleaved)
    ttnn.deallocate(tt_tile_l1)
    ttnn.deallocate(tt_weight)
    ttnn.deallocate(tt_bias)

    # Unpack (identical to Approach 2)
    tt_out_rm = ttnn.to_layout(tt_out_packed, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram_interleaved)
    ttnn.deallocate(tt_out_packed)
    tt_out_nhwc = ttnn.reshape(tt_out_rm, (batch, packed_h, input_width, packed_oc))
    tt_out_nchw_packed = ttnn.permute(tt_out_nhwc, dims=(0, 3, 1, 2), memory_config=dram_interleaved)
    ttnn.deallocate(tt_out_rm)
    ttnn.deallocate(tt_out_nhwc)
    tt_output = ttnn.reshape(tt_out_nchw_packed, (batch, out_channels, input_height, input_width))
    result = ttnn.to_torch(tt_output)
    ttnn.deallocate(tt_out_nchw_packed)
    ttnn.deallocate(tt_output)

    assert result.shape == torch.Size((batch, out_channels, input_height, input_width))

    golden = F.conv2d(
        torch_input.float(),
        torch_weight.float().reshape(out_channels, in_channels, 1, 1),
        bias=torch_bias.float().reshape(out_channels),
        stride=1,
        padding=0,
    )
    pcc = torch.corrcoef(torch.stack([result.float().flatten(), golden.float().flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99"
