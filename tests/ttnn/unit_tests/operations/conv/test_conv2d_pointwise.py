# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
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
