import ttnn
import pytest
from collections import ChainMap
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout
import torch
from loguru import logger

default_parameters = {
    "input": None,
    "kernel": None,
    "stride": (1, 1),
    "pad": (0, 0),
    "dilation": (1, 1),
    "bias": False,
    "groups": 1,
    "act_dtype": ttnn.bfloat16,
    "weights_dtype": ttnn.bfloat16,
    "shard": None,
    "in_layout": ttnn.ROW_MAJOR_LAYOUT,
    "out_layout": ttnn.TILE_LAYOUT,
    "math_fi": ttnn.MathFidelity.HiFi4,
    "packer_acc": True,
    "fp32_acc": True,
    "split_reader": False,
    "act_x2_buffer": False,
    "weights_x2_buffer": False,
}


def run_conv(
    device,
    batch_size,
    input,
    kernel,
    stride,
    pad,
    dilation,
    bias,
    groups,
    act_dtype,
    weights_dtype,
    shard,
    in_layout,
    out_layout,
    math_fi,
    packer_acc,
    fp32_acc,
    split_reader,
    act_x2_buffer,
    weights_x2_buffer,
):
    input_height = input[0]
    input_width = input[1]
    input_channels = input[2]

    output_channels = kernel[0]
    filter_height = kernel[1]
    filter_width = kernel[2]

    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_nchw = torch.ones(conv_input_shape, dtype=torch.bfloat16).float()

    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.ones(conv_weight_shape, dtype=torch.bfloat16).float()

    torch_bias_tensor = torch.zeros(conv_bias_shape, dtype=torch.bfloat16).float() if bias else None
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if bias else None,
        stride=stride,
        padding=pad,
        dilation=dilation,
        groups=groups,
    )
    output_shape_nhwc = [
        torch_out_golden_tensor.shape[0],
        torch_out_golden_tensor.shape[2],
        torch_out_golden_tensor.shape[3],
        torch_out_golden_tensor.shape[1],
    ]

    reader_patterns_cache = {}

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor,
        weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
    )
    tt_bias_tensor = None
    if bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor,
            weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
        )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        dtype=act_dtype,
        weights_dtype=weights_dtype,
        shard_layout=shard,
        enable_act_double_buffer=act_x2_buffer,
        enable_weights_double_buffer=weights_x2_buffer,
        enable_split_reader=split_reader,
        output_layout=out_layout,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fi,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_acc,
    )

    [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=stride,
        padding=pad,
        dilation=dilation,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        return_weights_and_bias=True,
        return_output_dim=True,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    reader_patterns_cache.clear()

    if not fp32_acc:
        pcc = 0.985
    elif math_fi == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.996
    else:
        pcc = 0.997

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing


from .models.resnet50_test_data import test_data as resnet50_test_data


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("params", resnet50_test_data)
def test_resnet50(device, params):
    run_conv(device, batch_size=2, **ChainMap(params, default_parameters))


from .standalone.width_sharded_test_data import test_data as width_sharded_test_data


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("params", width_sharded_test_data)
def test_width_sharded(device, params):
    run_conv(device, batch_size=2, **ChainMap(params, default_parameters))
