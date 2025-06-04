# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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
        dtype=output_dtype,
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
        (1, 32, 3, 32, 3, 1, 1, 1, True, None),
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
    ((1, 32, 32, 1024, 3, 1, 1, 1),),
)
@pytest.mark.parametrize("prepare_weights", [True, False])
@pytest.mark.parametrize(
    "preprocess_weights_on_device",
    [True, False],
)
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
    preprocess_weights_on_device,
):
    if prepare_weights and preprocess_weights_on_device:
        pytest.skip("preprocess_weights_on_device isn't needed when prepare_weights is True")

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
            input_height=input_length,
            input_width=1,
            kernel_size=(kernel_size, 1),
            stride=(1, 1),
            padding=(1, 0),
            dilation=(1, 1),
            has_bias=False,
            groups=1,
            device=device,
        )

    conv_config = ttnn.Conv1dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=False,
        preprocess_weights_on_device=preprocess_weights_on_device,
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
        return_output_dim=True,
    )

    torch_output_tensor = ttnn.to_torch(tt_output_tensor_on_device)

    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_length, output_channels)

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 2, 1))

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.995)
    print(pcc_msg)
    assert passing
