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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "filter, stride, input_height, input_width",
    [
        ((2, 2), (2, 2), 34, 34),
        ((1, 2), (1, 2), 34, 34),
        ((2, 1), (2, 1), 34, 34),
        ((4, 4), (4, 4), 32, 32),
        ((2, 2), (1, 1), 34, 34),
    ],
    ids=["k2s2", "k1x2s2", "k2x1s2", "k4s4", "k2s1"],
)
@pytest.mark.parametrize("shard_layout", [None, HS, BS, WS], ids=["auto", "h", "b", "w"])
def test_conv2d_kernel_stride_folding_explicit_shard_layouts(
    device, filter, stride, input_height, input_width, shard_layout
):
    """A k == s conv folds into a pointwise op, so an explicit shard layout must match auto.

    The folded activation carries K = kernel window * channels while the weights were prepared in
    the conv layout (channel padding to the shard boundary), so the HEIGHT/BLOCK sharded matmul
    raised "K dimension in tiles must match between input A (4) and input B (8)" and the
    WIDTH_SHARDED conv raised "The width of tensor a needs to match the height of tensor b".
    Auto sharding was already correct. k2s1 does not fold, so it kept working and must keep working.
    """
    torch.manual_seed(0)
    input_channels, output_channels, batch_size = 32, 64, 1
    torch_input = torch.randn(batch_size, input_channels, input_height, input_width).to(torch.bfloat16).float()
    torch_weight = (torch.randn(output_channels, input_channels, *filter) * 0.2).to(torch.bfloat16).float()
    reference = torch.nn.functional.conv2d(torch_input, torch_weight, stride=stride, padding=0)

    # Host input in the layout the reported repro used: the op folds the activation itself.
    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).reshape(1, 1, batch_size * input_height * input_width, input_channels),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_layout=shard_layout,
        enable_kernel_stride_folding=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    conv_args = dict(
        input_tensor=tt_input,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        kernel_size=filter,
        stride=stride,
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=1,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat16,
    )
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_output, [out_height, out_width], [tt_weight_prepared, _] = ttnn.conv2d(
        weight_tensor=tt_weight, bias_tensor=None, **conv_args
    )
    out = ttnn.to_torch(tt_output).float().reshape(batch_size, out_height, out_width, -1)[..., :output_channels]
    out = out.permute(0, 3, 1, 2)
    assert list(out.shape) == list(reference.shape), f"{list(out.shape)} vs {list(reference.shape)}"
    assert (out - reference).abs().max() < 0.05 * reference.abs().max(), "first call differs from torch"

    # The prepared device weights have to be reusable for the same conv.
    tt_output, _, _ = ttnn.conv2d(weight_tensor=tt_weight_prepared, bias_tensor=None, **conv_args)
    out = ttnn.to_torch(tt_output).float().reshape(batch_size, out_height, out_width, -1)[..., :output_channels]
    out = out.permute(0, 3, 1, 2)
    assert (out - reference).abs().max() < 0.05 * reference.abs().max(), "reused weights differ from torch"
