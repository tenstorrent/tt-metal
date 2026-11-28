# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, WS, BS
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config",
    (
        (353, 384, 8, 8, WS, None),
        (128, 128, 32, 32, BS, None),
        (16, 16, 256, 256, HS, {"act_block_h": 32}),
    ),
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "input_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [True, False],
)
@pytest.mark.parametrize(
    "packer_l1_acc",
    [True, False],
)
@pytest.mark.parametrize(
    "filter, padding",
    [
        [3, (1, 2, 2, 3)],
        [5, (2, 4, 3, 5)],
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_conv_features(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    shard_layout,
    config,
    filter,
    stride,
    padding,
    output_layout,
    fp32_accum,
    packer_l1_acc,
    input_dtype,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        None,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter,
        filter,
        stride,
        stride,
        padding,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        run_twice=True,
        input_layout=ttnn.TILE_LAYOUT if input_dtype == ttnn.bfloat8_b else None,
        input_dtype=input_dtype,
    )


@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config",
    (
        (353, 384, 8, 8, WS, None),
        (128, 128, 32, 32, BS, None),
        (16, 16, 256, 256, HS, {"act_block_h": 32}),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "input_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "filter, padding",
    [
        [3, (1, 2, 2, 3)],
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv_dram_config(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    shard_layout,
    config,
    filter,
    stride,
    padding,
    output_layout,
    input_dtype,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter,
        filter,
        stride,
        stride,
        padding,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=True,
        run_twice=True,
        input_layout=ttnn.TILE_LAYOUT if input_dtype == ttnn.bfloat8_b else None,
        input_dtype=input_dtype,
        config_tensors_in_dram=True,
    )


SliceHeight = ttnn.Conv2dDRAMSliceHeight
SliceWidth = ttnn.Conv2dDRAMSliceWidth


@pytest.mark.parametrize(
    "input_layout, dtype",
    [[ttnn.TILE_LAYOUT, ttnn.bfloat8_b], [ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16]],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, weights_dtype, kernel, stride, padding, dilation, act_block_h_override,  math_fidelity",
    # fmt: off
    (
        (1, 400,  528,  192,   192,    ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1),      0,  ttnn.MathFidelity.HiFi4 ),
        (2,  13,   31,  313,    71,    ttnn.bfloat8_b, (5, 5), (1, 1), (2, 2), (2, 2), 32 * 4,  ttnn.MathFidelity.LoFi  ),
        (2,  63,  129,  981,    39,    ttnn.bfloat8_b, (3, 3), (2, 2), (2, 2), (1, 1),      0,  ttnn.MathFidelity.LoFi  ),
        (2, 512,  512,  128,   128,    ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 32 * 8,  ttnn.MathFidelity.LoFi  ),
        (2, 64,   64,   384,   64,     ttnn.bfloat8_b, (4, 4), (2, 2), (1, 1), (1, 1),      0,  ttnn.MathFidelity.LoFi  ),
        (1, 4,    32,   1024,  1024,   ttnn.bfloat8_b, (5, 5), (1, 1), (0, 0), (1, 1),     32,  ttnn.MathFidelity.LoFi  ),
        (1, 2904, 2904,   48,    48,   ttnn.bfloat8_b, (3, 3), (1, 1), (0, 0), (1, 1),     32,  ttnn.MathFidelity.HiFi4 ),
    )
    # fmt: on
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[True, True, False]],
)
def test_conv_dram(
    device,
    torch_tensor_map,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    has_bias,
    weights_dtype,
    dtype,
    kernel,
    stride,
    padding,
    dilation,
    act_block_h_override,
    math_fidelity,
    fp32_accum,
    input_layout,
    packer_l1_acc,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")
    config = {
        "act_block_h": act_block_h_override,
    }

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding,
        config,
        dilation_h=dilation[0],
        dilation_w=dilation[1],
        has_bias=has_bias,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        input_dtype=dtype,
        input_layout=input_layout,
        output_layout=input_layout,
        run_twice=True,
        fast_compare=True,
        use_dram_slicing=True,
    )


_CONV_ZERO_PARAM_CASES = [
    ({"in_channels": 0}, "in_channels"),
    ({"out_channels": 0}, "out_channels"),
    ({"batch_size": 0}, "batch_size"),
    ({"kernel_size": (0, 3)}, "kernel_size"),
    ({"stride": (1, 0)}, "stride"),
    ({"groups": 0}, "groups"),
]
_CONV_ZERO_PARAM_IDS = ["in_channels", "out_channels", "batch_size", "kernel_size", "stride", "groups"]


def _conv_zero_param_kwargs(device, override):
    kwargs = dict(
        in_channels=32,
        out_channels=32,
        batch_size=1,
        input_height=8,
        input_width=8,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        device=device,
    )
    kwargs.update(override)
    return kwargs


@pytest.mark.parametrize("override, field", _CONV_ZERO_PARAM_CASES, ids=_CONV_ZERO_PARAM_IDS)
def test_conv2d_rejects_zero_params(device, override, field, expect_error):
    tt_input = ttnn.from_torch(torch.randn(1, 1, 64, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, device=device)
    tt_weight = ttnn.from_torch(torch.randn(32, 32, 3, 3, dtype=torch.bfloat16), dtype=ttnn.bfloat16)
    with expect_error(RuntimeError, f"{field}.*must be greater than 0"):
        ttnn.conv2d(input_tensor=tt_input, weight_tensor=tt_weight, **_conv_zero_param_kwargs(device, override))


@pytest.mark.parametrize("override, field", _CONV_ZERO_PARAM_CASES, ids=_CONV_ZERO_PARAM_IDS)
def test_prepare_conv_weights_rejects_zero_params(device, override, field, expect_error):
    tt_weight = ttnn.from_torch(torch.randn(32, 32, 3, 3, dtype=torch.bfloat16), dtype=ttnn.bfloat16)
    with expect_error(RuntimeError, f"{field}.*must be greater than 0"):
        ttnn.prepare_conv_weights(
            weight_tensor=tt_weight,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            has_bias=False,
            input_dtype=ttnn.bfloat16,
            **_conv_zero_param_kwargs(device, override),
        )


_DIRTY_MARKER = 777.0


def _dirty_dram(device, num_elements):
    # Leave a recognisable value in freed DRAM so rows the op never writes cannot pass by reading back as zero.
    rows = max(1, num_elements // 32)
    for layout in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
        dirty = ttnn.from_torch(
            torch.full((1, 1, rows, 32), _DIRTY_MARKER, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(dirty)


# (id, H, W, kernel, padding, slice_type, num_slices): output lines that receive no input (rows lying entirely in
# padding) must still be computed, so that every later slice is written too.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["rm", "tile"])
@pytest.mark.parametrize(
    "height, width, kernel, padding, slice_type, num_slices",
    [
        (8, 8, (1, 1), (1, 1), ttnn.Conv2dDRAMSliceHeight, 10),
        (8, 8, (3, 3), (3, 3), ttnn.Conv2dDRAMSliceHeight, 12),
        (8, 8, (1, 1), (1, 1), ttnn.Conv2dDRAMSliceWidth, 10),
        (8, 8, (3, 3), (3, 3), ttnn.Conv2dDRAMSliceWidth, 12),
        # controls: no slice is empty
        (8, 8, (3, 3), (3, 3), ttnn.Conv2dDRAMSliceHeight, 2),
        (8, 8, (3, 3), (1, 1), ttnn.Conv2dDRAMSliceHeight, 8),
    ],
    ids=["1x1p1_h10", "3x3p3_h12", "1x1p1_w10", "3x3p3_w12", "ctrl_3x3p3_h2", "ctrl_3x3p1_h8"],
)
def test_conv2d_dram_slice_rows_without_input(
    device, height, width, kernel, padding, slice_type, num_slices, output_layout
):
    if slice_type == ttnn.Conv2dDRAMSliceWidth and output_layout == ttnn.TILE_LAYOUT:
        pytest.skip("width slicing a TILE output allows only one slice")
    cin = cout = 32
    torch.manual_seed(0)
    torch_input = torch.randn(1, cin, height, width).bfloat16().float()
    torch_weight = torch.randn(cout, cin, *kernel).bfloat16().float()
    torch_bias = torch.randn(1, 1, 1, cout).bfloat16().float()
    torch_out = torch.nn.functional.conv2d(torch_input, torch_weight, torch_bias.reshape(-1), padding=padding)
    out_h, out_w = torch_out.shape[2], torch_out.shape[3]

    _dirty_dram(device, 4 * out_h * out_w * cout * 2)
    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).contiguous(),
        ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out, (got_h, got_w) = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=ttnn.from_torch(torch_weight, ttnn.bfloat16),
        bias_tensor=ttnn.from_torch(torch_bias, ttnn.bfloat16),
        in_channels=cin,
        out_channels=cout,
        device=device,
        kernel_size=kernel,
        stride=(1, 1),
        padding=padding,
        dilation=(1, 1),
        batch_size=1,
        input_height=height,
        input_width=width,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, output_layout=output_layout),
        compute_config=ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=slice_type, num_slices=num_slices),
        groups=1,
        return_output_dim=True,
        dtype=ttnn.bfloat16,
    )
    assert (got_h, got_w) == (out_h, out_w)
    result = ttnn.to_torch(tt_out.cpu()).float().reshape(1, got_h, got_w, -1)[..., :cout].permute(0, 3, 1, 2)
    assert not (result == _DIRTY_MARKER).any(), "output holds values the op never wrote"
    torch.testing.assert_close(result, torch_out, atol=0.1 + 0.05 * torch_out.abs().max().item(), rtol=0.05)
