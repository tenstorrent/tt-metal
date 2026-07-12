# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout
import ttnn


def run_conv(
    device,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    use_1d_systolic_array,
    config_override,
    transpose_mcast=True,
    enable_auto_formatting=False,
    padded_input_channels=None,
    fp32_accum=False,
    packer_l1_acc=False,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=True,
    groups=1,
    auto_shard=False,
    shard_layout=None,
):
    # has_bias = False
    has_bias = False
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_length]
    conv_weight_shape = [output_channels, input_channels // groups, kernel_size]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_ncl = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_ncl, (0, 2, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if has_bias else None
    torch_out_golden_tensor = torch.nn.functional.conv1d(
        torch_input_tensor_ncl,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=stride,
        padding=padding,
        groups=groups,
    )

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    if shard_layout is None:
        shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if use_1d_systolic_array else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
    if auto_shard:
        shard_layout = None

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=weights_dtype,
        shard_layout=shard_layout,
        deallocate_activation=deallocate_activation,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]
        print("Setting Act Block H to ", conv_config.act_block_h_override)
    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
            print("Setting num_cores_nhw to 98")

    [tt_output_tensor_on_device, out_length, [weights_device, bias_device]] = ttnn.conv1d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        batch_size=batch_size,
        input_length=input_length,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        dtype=output_dtype,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = torch.Tensor(ttnn.to_torch(tt_output_tensor))

    # torch_output_tensor is in row major layout and NLC shape
    # NLC to NCL
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_length, output_channels)

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 2, 1))

    if not fp32_accum:
        pcc = 0.995
    elif math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.9969
    else:
        pcc = 0.998

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    assert passing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, groups, use_1d_systolic_array, config_override",
    (
        (1, 5120, 5120, 32, 4, 1, 3, 5120, True, None),
        (1, 5120, 5120, 1024, 4, 1, 3, 5120, True, None),
        (1, 2560, 2560, 1027, 4, 1, 0, 2560, True, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv1d_mamba(
    device,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups,
    use_1d_systolic_array,
    config_override,
    output_layout,
):
    if activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if groups > 5120 or input_channels > 5120 or output_channels > 5120:
        pytest.skip("OOM")
    if (input_channels > 2560 or output_channels > 2560) and output_dtype == ttnn.bfloat16:
        pytest.skip("OOM")

    run_conv(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        output_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        use_1d_systolic_array,
        config_override,
        transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
        padded_input_channels=None,
        output_layout=output_layout,
        groups=groups,
        auto_shard=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, groups, use_1d_systolic_array, config_override",
    (
        # (1, 32, 3, 32, 3, 1, 1, 1, True, None),  # workaround for https://github.com/tenstorrent/tt-metal/issues/49393
        (1, 128, 32, 1024, 5, 1, 2, 1, True, None),
        (1, 512, 32, 5120, 3, 1, 1, 1, True, None),
        (1, 64, 64, 2560, 3, 1, 1, 32, True, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv1d(
    device,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups,
    use_1d_systolic_array,
    config_override,
    output_layout,
):
    if activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if groups > 5120 or input_channels > 5120 or output_channels > 5120:
        pytest.skip("OOM")
    if (input_channels > 2560 or output_channels > 2560) and output_dtype == ttnn.bfloat16:
        pytest.skip("OOM")

    run_conv(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        output_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        use_1d_systolic_array,
        config_override,
        transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
        padded_input_channels=None,
        output_layout=output_layout,
        groups=groups,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, groups, shard_layout, config_override",
    (
        # (8, 768, 768, 384, 1, 1, 0, 4, True, None, False), #Pass
        (8, 768, 3072, 384, 1, 1, 0, 4, ttnn.TensorMemoryLayout.WIDTH_SHARDED, {"act_block_h": 1536}),
        (8, 3072, 768, 384, 1, 1, 0, 4, ttnn.TensorMemoryLayout.WIDTH_SHARDED, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_squeezebert_conv1d(
    device,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups,
    shard_layout,
    config_override,
    output_layout,
):
    if activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    # if groups > 5120 or input_channels > 5120 or output_channels > 5120:
    #     pytest.skip("OOM")
    # if (input_channels > 2560 or output_channels > 2560) and output_dtype == ttnn.bfloat16:
    #     pytest.skip("OOM")

    run_conv(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        output_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        False,
        config_override,
        shard_layout=shard_layout,
        padded_input_channels=None,
        output_layout=output_layout,
        groups=groups,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, groups",
    (
        (1, 32, 32, 1024, 3, 1, 1, 1),
        (2, 512, 512, 1024, 7, 1, 3, 512),
    ),
)
@pytest.mark.parametrize("prepare_weights", [True, False])
def test_with_prepare_weights(
    device,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups,
    prepare_weights,
):
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_length]
    conv_weight_shape = [output_channels, input_channels // groups, kernel_size]
    torch_input_tensor_ncl = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()

    torch_input_tensor = torch.permute(torch_input_tensor_ncl, (0, 2, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_out_golden_tensor = torch.nn.functional.conv1d(
        torch_input_tensor_ncl,
        torch_weight_tensor,
        stride=stride,
        padding=padding,
        groups=groups,
    )

    tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, dtype=ttnn.bfloat16)

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    if prepare_weights:
        tt_weight_tensor = ttnn.prepare_conv_weights(
            weight_tensor=tt_weight_tensor,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            in_channels=input_channels,
            out_channels=output_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=input_length,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            dilation=(1, 1),
            has_bias=False,
            groups=groups,
            device=device,
            input_dtype=ttnn.bfloat16,
        )

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=False,
    )

    tt_output_tensor_on_device, out_length = ttnn.conv1d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        batch_size=batch_size,
        input_length=input_length,
        conv_config=conv_config,
        groups=groups,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
    )

    torch_output_tensor = ttnn.to_torch(tt_output_tensor_on_device)

    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_length, output_channels)

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 2, 1))

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.995)
    print(pcc_msg)
    assert passing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_length, kernel_size, stride, padding, dilation",
    (
        (1, 128, 128, 8192, 3, 1, 3, 3),
        (1, 128, 128, 8192, 3, 1, 2, 2),
    ),
)
@pytest.mark.parametrize(
    "shard_layout",
    [ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED],
)
def test_conv1d_dilation(
    device,
    batch_size,
    input_channels,
    output_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    dilation,
    shard_layout,
):
    """Regression test for #37716: block-sharded conv1d with dilation>1 produced wrong results
    due to missing act_block_w_extra_align_bytes in read_dilated_channels path."""
    torch.manual_seed(0)
    torch_input = torch.randn(batch_size, input_channels, input_length, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(output_channels, input_channels, kernel_size, dtype=torch.bfloat16).float()

    golden = torch.nn.functional.conv1d(
        torch_input,
        torch_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    input_tt = ttnn.from_torch(
        torch_input.permute(0, 2, 1),
        layout=ttnn.Layout.ROW_MAJOR,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    weight_tt = ttnn.from_torch(torch_weight, layout=ttnn.Layout.ROW_MAJOR)

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=shard_layout,
        deallocate_activation=True,
        config_tensors_in_dram=True,
    )

    tt_out, out_len = ttnn.conv1d(
        input_tensor=input_tt,
        weight_tensor=weight_tt,
        device=device,
        in_channels=input_channels,
        out_channels=output_channels,
        batch_size=batch_size,
        input_length=input_length,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias_tensor=None,
        conv_config=conv_config,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
    )

    tt_output = ttnn.to_torch(tt_out).reshape(batch_size, out_len, output_channels).permute(0, 2, 1)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(tt_output, golden, pcc=0.999)
    print(pcc_msg)
    assert passing


def run_conv1d_replicate_pad(
    device,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups=1,
    dilation=1,
    has_bias=True,
    weights_dtype=ttnn.bfloat16,
    output_dtype=ttnn.bfloat16,
):
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_length]
    conv_weight_shape = [output_channels, input_channels // groups, kernel_size]
    torch_input_tensor_ncl = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()

    # Set edge values to large numbers so replicate padding has a measurable effect on output.
    # Without this, random values near 0 make replicate vs zero padding hard to distinguish via PCC.
    torch_input_tensor_ncl[:, :, 0] = 10.0
    torch_input_tensor_ncl[:, :, -1] = -10.0

    torch_input_tensor = torch.permute(torch_input_tensor_ncl, (0, 2, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn([output_channels], dtype=torch.bfloat16).float() if has_bias else None

    # Golden: explicit replicate pad + conv1d with no padding
    if isinstance(padding, (list, tuple)) and len(padding) == 2:
        pad_left, pad_right = padding[0], padding[1]
    else:
        pad_left = pad_right = padding

    torch_padded_input = torch.nn.functional.pad(
        torch_input_tensor_ncl,
        (pad_left, pad_right),
        mode="replicate",
    )
    torch_out_golden = torch.nn.functional.conv1d(
        torch_padded_input,
        torch_weight_tensor,
        bias=torch_bias_tensor,
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=groups,
    )

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor.reshape(1, 1, 1, -1), weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=weights_dtype,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=True,
        padding_mode=ttnn.PaddingMode.Replicate,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )

    [tt_output_tensor_on_device, out_length, [weights_device, bias_device]] = ttnn.conv1d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        batch_size=batch_size,
        input_length=input_length,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        dtype=output_dtype,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = torch.Tensor(ttnn.to_torch(tt_output_tensor))
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_length, output_channels)
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 2, 1))

    pcc = 0.995
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden, pcc=pcc)
    assert passing, pcc_msg


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, dilation, groups",
    (
        (1, 64, 32, 1024, 3, 1, 1, 1, 1),
        (1, 64, 32, 1024, 5, 1, 2, 1, 1),
        (1, 128, 64, 512, 3, 1, 1, 1, 1),
        (1, 64, 32, 1024, 3, 2, 1, 1, 1),
        (1, 64, 32, 512, 5, 1, 4, 1, 1),
        (2, 64, 32, 512, 3, 1, 1, 1, 1),
        (1, 64, 32, 512, 3, 1, (1, 2), 1, 1),
        # dilation > 1: padding = dilation for 3x3 kernel keeps output length equal.
        (1, 64, 32, 512, 3, 1, 2, 2, 1),
        # depthwise (groups == in_channels == out_channels) with replicate pad.
        (1, 32, 32, 512, 3, 1, 1, 1, 32),
    ),
)
def test_conv1d_replicate_pad(
    device, batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, dilation, groups
):
    run_conv1d_replicate_pad(
        device,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_length=input_length,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


# ---------------------------------------------------------------------------
# DRAM slicing
#
# conv1d reshapes its 1D input to a [N, 1, input_length, C] 4D tensor and delegates to
# conv2d. Because the height dimension is always 1, only width slicing (slicing along
# input_length) is meaningful. The tests below cover the L1_FULL OOM baseline, manual and
# auto DRAM width slicing, the channel-bound case width slicing cannot relieve, and
# DRAM_HEIGHT slicing being rejected (degenerate for conv1d).
# ---------------------------------------------------------------------------


def run_conv1d_slice(
    device,
    batch_size,
    in_channels,
    out_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    slice_config,
    pcc=0.99,
    weights_dtype=ttnn.bfloat16,
    activations_dtype=ttnn.bfloat16,
    output_dtype=ttnn.bfloat16,
):
    torch.manual_seed(0)
    torch_input_ncl = torch.randn(batch_size, in_channels, input_length, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.bfloat16).float()

    golden = torch.nn.functional.conv1d(
        torch_input_ncl,
        torch_weight,
        bias=None,
        stride=stride,
        padding=padding,
    )

    # DRAM slicing requires the input to live in DRAM (interleaved). Layout is NLC for conv1d.
    input_tt = ttnn.from_torch(
        torch_input_ncl.permute(0, 2, 1),
        dtype=activations_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weight_tt = ttnn.from_torch(torch_weight, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    # DRAM slicing requires an explicit shard layout (no auto-shard).
    conv_config = ttnn.Conv1dConfig(
        weights_dtype=weights_dtype,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=False,
    )

    tt_out, out_length = ttnn.conv1d(
        input_tensor=input_tt,
        weight_tensor=weight_tt,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_length=input_length,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        conv_config=conv_config,
        slice_config=slice_config,
        dtype=output_dtype,
        return_output_dim=True,
    )

    out = ttnn.to_torch(tt_out).reshape(batch_size, out_length, out_channels).permute(0, 2, 1)
    passing, pcc_msg = check_with_pcc_without_tensor_printout(out, golden, pcc=pcc)
    assert passing, pcc_msg
    return out


# A 1D conv whose L1 footprint is dominated by the (long) sequence dimension, so it
# overflows L1 in the L1_FULL path but is relieved by slicing along input_length (width).
# input_length=32768 with 256 channels: the width-independent weight block is small enough
# that width slicing brings the per-slice footprint under the L1 budget.
_SLICE_OOM_SHAPE = dict(
    batch_size=1,
    in_channels=256,
    out_channels=256,
    input_length=32768,
    kernel_size=3,
    stride=1,
    padding=1,
)

# A 1D conv whose L1 footprint is dominated by the channel dimension (the weight block does
# not shrink with width slicing). Width slicing - the only kind conv1d supports - cannot
# relieve this, so even the auto-slicer (up to one tile per slice) fails to find a fit.
_SLICE_CHANNEL_BOUND_SHAPE = dict(
    batch_size=1,
    in_channels=512,
    out_channels=512,
    input_length=32768,
    kernel_size=3,
    stride=1,
    padding=1,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
def test_conv1d_no_slicing_oom(device, expect_error):
    """Baseline: the large conv1d should run out of L1 when forced into the L1_FULL path.
    The match string asserts the failure is an L1 capacity error, not something unrelated."""
    with expect_error(RuntimeError, "circular buffer|beyond max L1|Out of Memory|allocat"):
        run_conv1d_slice(
            device,
            **_SLICE_OOM_SHAPE,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
@pytest.mark.parametrize("num_slices", [4, 8])
def test_conv1d_manual_dram_width_slicing(device, num_slices):
    """The same conv1d should succeed with manual DRAM width slicing."""
    run_conv1d_slice(
        device,
        **_SLICE_OOM_SHAPE,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth,
            num_slices=num_slices,
        ),
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
def test_conv1d_auto_dram_width_slicing(device):
    """The same conv1d should succeed with auto-determined DRAM width slicing (num_slices=0)."""
    run_conv1d_slice(
        device,
        **_SLICE_OOM_SHAPE,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth,
            num_slices=0,
        ),
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
def test_conv1d_channel_bound_slicing_insufficient(device, expect_error):
    """Width slicing cannot relieve a channel-bound OOM (the weight block is width-independent).

    Documents the boundary of conv1d DRAM slicing: when L1 pressure comes from the channel
    dimension rather than the sequence length, even maximal width slicing fails to find a fit.
    """
    with expect_error(RuntimeError, "could not find valid slice configuration"):
        run_conv1d_slice(
            device,
            **_SLICE_CHANNEL_BOUND_SHAPE,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dDRAMSliceWidth,
                num_slices=0,
            ),
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
def test_conv1d_dram_height_slicing_rejected(device, expect_error):
    """DRAM_HEIGHT slicing is degenerate for conv1d (height==1) and must be rejected."""
    with expect_error(RuntimeError, "DRAM_HEIGHT"):
        run_conv1d_slice(
            device,
            batch_size=1,
            in_channels=64,
            out_channels=64,
            input_length=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dDRAMSliceHeight,
                num_slices=4,
            ),
        )


# ---------------------------------------------------------------------------
# Default-routing regressions (no slice_config)
#
# Three fixes let stock ttnn.conv1d (including depthwise groups==C) handle the
# long-sequence / narrow-channel shapes that previously OOMed or hung when called
# with no slice_config:
#   1. DRAM slicing by default. A missing slice_config now forwards nullopt so conv2d
#      auto-routes by input location - inputs already in L1 stay in L1, while DRAM/host
#      inputs are width-sliced through DRAM (height is always 1, so the auto slice type
#      is always DRAM_WIDTH) - instead of forcing L1_FULL and OOMing on long sequences.
#      The DRAM path auto-determines a shard layout when shard_layout is unset, so
#      auto_shard / shard_layout=None callers keep working.
#   2. 4D weight reshape. The 3D conv1d weight [out, in/groups, K] is reinterpreted as
#      4D [.., 1, K] up front, because the DRAM auto-shard path reads the kernel width
#      (weight.logical_shape()[3]) before weights are prepared.
#   3. Depthwise CB deadlock. The depthwise compute kernel now tilizes the full
#      activation block height; tilizing only in0_num_subblocks tile-rows under-produced
#      (and deadlocked the activation CB) whenever out_subblock_h_ntiles > 1, which
#      happens at long depthwise sequence lengths.
# ---------------------------------------------------------------------------


def run_conv1d_route(
    device,
    batch_size,
    in_channels,
    out_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups=1,
    shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    slice_config=None,
    act_block_h=None,
    input_in_dram=True,
    weights_dtype=ttnn.bfloat16,
    activations_dtype=ttnn.bfloat16,
    output_dtype=ttnn.bfloat16,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_accum=False,
    packer_l1_acc=False,
    pcc=0.99,
    config_tensors_in_dram=False,
    fused_activation=None,
    golden_activation=None,
    dilation=1,
):
    """Run ttnn.conv1d and check it against the torch golden.

    slice_config=None exercises the default routing path (a DRAM/host input must auto-route
    through DRAM width slicing rather than forcing L1_FULL). shard_layout=None selects
    auto-shard. input_in_dram places the input in DRAM; set False to keep it in L1.
    act_block_h sets act_block_h_override (used to force out_subblock_h_ntiles > 1).
    """
    torch.manual_seed(0)
    torch_input_ncl = torch.randn(batch_size, in_channels, input_length, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(out_channels, in_channels // groups, kernel_size, dtype=torch.bfloat16).float()

    golden = torch.nn.functional.conv1d(
        torch_input_ncl,
        torch_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if golden_activation is not None:
        golden = golden_activation(golden)

    input_tt = ttnn.from_torch(
        torch_input_ncl.permute(0, 2, 1),  # NLC for conv1d
        dtype=activations_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if input_in_dram else ttnn.L1_MEMORY_CONFIG,
    )
    weight_tt = ttnn.from_torch(torch_weight, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=weights_dtype,
        shard_layout=shard_layout,  # None == auto-shard
        deallocate_activation=False,
        config_tensors_in_dram=config_tensors_in_dram,
        activation=fused_activation,
    )
    if act_block_h is not None:
        conv_config.act_block_h_override = act_block_h
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )

    tt_out, out_length = ttnn.conv1d(
        input_tensor=input_tt,
        weight_tensor=weight_tt,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_length=input_length,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=slice_config,
        dtype=output_dtype,
        return_output_dim=True,
    )

    out = ttnn.to_torch(tt_out).reshape(batch_size, out_length, out_channels).permute(0, 2, 1)
    passing, pcc_msg = check_with_pcc_without_tensor_printout(out, golden, pcc=pcc)
    assert passing, pcc_msg
    return out


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
@pytest.mark.parametrize("config_tensors_in_dram", [False, True])
def test_conv1d_depthwise_reader_indices_storage(device, config_tensors_in_dram):
    channels = 512
    run_conv1d_route(
        device,
        batch_size=1,
        in_channels=channels,
        out_channels=channels,
        input_length=7,
        kernel_size=4,
        stride=1,
        padding=0,
        groups=channels,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_in_dram=False,
        config_tensors_in_dram=config_tensors_in_dram,
        pcc=0.995,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
def test_conv1d_depthwise_dilation(device):
    channels = 512
    run_conv1d_route(
        device,
        batch_size=1,
        in_channels=channels,
        out_channels=channels,
        input_length=10,
        kernel_size=4,
        stride=1,
        padding=0,
        dilation=2,
        groups=channels,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_in_dram=False,
        pcc=0.995,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
@pytest.mark.parametrize(
    "fused_activation,golden_activation",
    [
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), torch.nn.functional.silu),
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), torch.nn.functional.relu),
    ],
    ids=["silu", "relu"],
)
@pytest.mark.parametrize("channels,kernel_size", [(512, 4), (1280, 7)], ids=["coalesced", "non_coalesced"])
def test_conv1d_depthwise_fused_activation(device, fused_activation, golden_activation, channels, kernel_size):
    run_conv1d_route(
        device,
        batch_size=1,
        in_channels=channels,
        out_channels=channels,
        input_length=kernel_size + 3,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        groups=channels,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_in_dram=False,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fused_activation=fused_activation,
        golden_activation=golden_activation,
        pcc=0.995,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
def test_conv1d_default_route_long_seq(device):
    """Fix 1: a long-sequence conv1d in DRAM with no slice_config must auto-route through
    DRAM width slicing instead of forcing L1_FULL. Same shape as the L1_FULL OOM baseline
    (test_conv1d_no_slicing_oom); without the fix this OOMs ("circular buffers grow beyond
    max L1"), with it the input is width-sliced through DRAM and matches golden."""
    run_conv1d_route(device, **_SLICE_OOM_SHAPE)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
def test_conv1d_default_route_auto_shard(device):
    """Fix 1: auto-shard (shard_layout=None) with no slice_config must keep working - the
    DRAM routing path auto-determines a shard layout when none is given. Without the fix
    this OOMs in L1_FULL."""
    run_conv1d_route(device, **_SLICE_OOM_SHAPE, shard_layout=None)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
def test_conv1d_default_route_l1_input_stays_l1(device):
    """Fix 1 guard: an input already in L1 with no slice_config must stay in L1 (L1_FULL),
    not get pushed through the DRAM slicing path. Small enough to fit L1; passes with and
    without the fix - it guards against the new default wrongly re-routing L1 inputs."""
    run_conv1d_route(
        device,
        batch_size=1,
        in_channels=64,
        out_channels=64,
        input_length=1024,
        kernel_size=3,
        stride=1,
        padding=1,
        input_in_dram=False,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1 << 15}], indirect=True)
def test_conv1d_depthwise_dram_slice_auto_shard(device):
    """Fix 2 (4D weight reshape): depthwise (groups == C) conv1d through the DRAM-slicing
    auto-shard path (explicit DRAM width slice + shard_layout=None). That path reads the
    kernel width as weight.logical_shape()[3] before weights are prepared; without the
    up-front 3D->4D reshape this raises "ShapeBase[] index out of range. 3 not in [-4, 3)".
    An explicit shard_layout (non-auto) does NOT hit this, so auto-shard is required."""
    C = 256
    run_conv1d_route(
        device,
        batch_size=1,
        in_channels=C,
        out_channels=C,
        input_length=2921,
        kernel_size=12,
        stride=1,
        padding=0,
        groups=C,
        shard_layout=None,  # auto-shard: triggers the logical_shape()[3] read
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=8),
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_conv1d_depthwise_subblock_deadlock(device):
    """Fix 3 (depthwise CB deadlock): a depthwise (groups == C) conv1d whose act_block_h is
    forced large enough that out_subblock_h_ntiles > 1. The depthwise compute kernel tilized
    only in0_num_subblocks tile-rows while mul_and_accumulate_block consumed the full block;
    they mismatch when out_subblock_h_ntiles > 1, so tilize under-produces and the activation
    CB deadlocks (the op hangs). With the full-block-height tilize this completes and matches
    golden. Fits L1 (no slicing) so the deadlock is the only failure mode under test."""
    C = 64
    run_conv1d_route(
        device,
        batch_size=1,
        in_channels=C,
        out_channels=C,
        input_length=2048,
        kernel_size=12,
        stride=1,
        padding=0,
        groups=C,
        act_block_h=256,  # 8 tile-rows -> out_subblock_h_ntiles > 1
        input_in_dram=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_conv1d_depthwise_default_route_long_seq(device):
    """Regression for #46395 (fixes 1, 2, 3 end-to-end): the real vocoder STAGE_C upsample
    tail - depthwise groups==C at ~29k sequence length - previously OOMed/deadlocked through
    the stock conv1d path. Ported from the experimental conv1d_depthwise repro to exercise
    only the stock conv1d path. Without the fix this OOMs in L1_FULL (~4.5 MB CB vs 1.5 MB
    L1). l1_small_size matches the vocoder's setting."""
    C = 64
    run_conv1d_route(
        device,
        batch_size=1,
        in_channels=C,
        out_channels=C,
        input_length=28841,
        kernel_size=12,
        stride=1,
        padding=0,
        groups=C,
        weights_dtype=ttnn.float32,
        activations_dtype=ttnn.float32,
        output_dtype=ttnn.float32,
        fp32_accum=True,
        packer_l1_acc=True,
        pcc=0.999,
    )
