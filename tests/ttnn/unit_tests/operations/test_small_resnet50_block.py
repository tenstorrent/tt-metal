# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import ttnn
import torch
import time
from models.utility_functions import (
    is_grayskull,
    is_wormhole_b0,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


# utility functions
def do_nothing_op(x):
    return x


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class resnet50Bottleneck:
    expansion: int = 4

    def __init__(self, parameters, downsample, model_config) -> None:
        super().__init__()
        # init is just to pre-process pytorch weights and bias tensors
        torch_identity_conv_weight_tensor = torch.zeros(
            [parameters.conv1.weight.shape[1], parameters.conv1.weight.shape[1], 1, 1], dtype=torch.bfloat16
        ).float()
        for k in range(parameters.conv1.weight.shape[1]):
            for c in range(parameters.conv1.weight.shape[1]):
                if k == c:
                    torch_identity_conv_weight_tensor[k][c][0][0] = 1
        self.identity_conv_weight_tensor = ttnn.from_torch(
            torch_identity_conv_weight_tensor,
            dtype=model_config["WEIGHTS_DTYPE"] if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b else ttnn.float32,
        )
        self.conv1_weight_tensor = ttnn.from_torch(
            parameters.conv1.weight,
            dtype=model_config["WEIGHTS_DTYPE"] if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b else ttnn.float32,
        )
        self.conv1_bias_tensor = ttnn.from_torch(
            parameters.conv1.bias,
            dtype=model_config["WEIGHTS_DTYPE"] if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b else ttnn.float32,
        )
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 1

        self.conv2_weight_tensor = ttnn.from_torch(
            parameters.conv2.weight,
            dtype=model_config["WEIGHTS_DTYPE"] if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b else ttnn.float32,
        )
        self.conv2_bias_tensor = ttnn.from_torch(
            parameters.conv2.bias,
            dtype=model_config["WEIGHTS_DTYPE"] if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b else ttnn.float32,
        )
        self.conv2_input_channels = self.conv2_weight_tensor.shape[1]
        self.conv2_output_channels = self.conv2_weight_tensor.shape[0]
        self.conv2_stride = 2 if downsample else 1
        assert self.conv2_weight_tensor.shape[2] == 3

        self.conv3_weight_tensor = ttnn.from_torch(
            parameters.conv3.weight,
            dtype=model_config["WEIGHTS_DTYPE"] if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b else ttnn.float32,
        )
        self.conv3_bias_tensor = ttnn.from_torch(
            parameters.conv3.bias,
            dtype=model_config["WEIGHTS_DTYPE"] if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b else ttnn.float32,
        )
        self.conv3_input_channels = self.conv3_weight_tensor.shape[1]
        self.conv3_output_channels = self.conv3_weight_tensor.shape[0]
        assert self.conv3_weight_tensor.shape[2] == 1

        self.downsample = downsample
        if downsample:
            self.ds_conv_weight_tensor = ttnn.from_torch(
                parameters.ds_conv.weight,
                dtype=model_config["WEIGHTS_DTYPE"]
                if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b
                else ttnn.float32,
            )
            self.ds_conv_bias_tensor = ttnn.from_torch(
                parameters.ds_conv.bias,
                dtype=model_config["WEIGHTS_DTYPE"]
                if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b
                else ttnn.float32,
            )
            self.ds_conv_input_channels = self.ds_conv_weight_tensor.shape[1]
            self.ds_conv_output_channels = self.ds_conv_weight_tensor.shape[0]
            assert self.ds_conv_weight_tensor.shape[2] == 1
        self.model_config = model_config
        return

    def __call__(self, x, device, batch_size, input_height, input_width, conv_op_cache):
        # logger.info("This module input shape - ", self.module_input_shape)
        # conv1 is 1x1 conv
        # print("Running conv1")
        x, input_height, input_width, self.identity_conv_weight_tensor, _ = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.identity_conv_weight_tensor,
            in_channels=self.conv1_input_channels,
            out_channels=self.conv1_input_channels,
            device=device,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=ttnn.Conv2dConfig(
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                math_fidelity=self.model_config["MATH_FIDELITY"],
            ),
            conv_op_cache=conv_op_cache,
        )

        out, input_height, input_width, self.conv1_weight_tensor, self.conv1_bias_tensor = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_tensor,
            in_channels=self.conv1_input_channels,
            out_channels=self.conv1_output_channels,
            device=device,
            bias_tensor=self.conv1_bias_tensor,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=ttnn.Conv2dConfig(
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                math_fidelity=self.model_config["MATH_FIDELITY"],
                activation="relu",
            ),
            conv_op_cache=conv_op_cache,
        )

        if self.downsample:
            ds_out, _, _, self.ds_conv_weight_tensor, self.ds_conv_bias_tensor = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.ds_conv_weight_tensor,
                in_channels=self.ds_conv_input_channels,
                out_channels=self.ds_conv_output_channels,
                device=device,
                bias_tensor=self.ds_conv_bias_tensor,
                kernel_size=(1, 1),
                stride=(2, 2),
                padding=(0, 0),
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                conv_config=ttnn.Conv2dConfig(
                    dtype=self.model_config["ACTIVATIONS_DTYPE"],
                    weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                    math_fidelity=self.model_config["MATH_FIDELITY"],
                ),
                conv_op_cache=conv_op_cache,
            )
            ttnn.deallocate(x)
        else:
            ds_out = x

        # print("Running conv2")
        out, input_height, input_width, self.conv2_weight_tensor, self.conv2_bias_tensor = ttnn.conv2d(
            input_tensor=out,
            weight_tensor=self.conv2_weight_tensor,
            in_channels=self.conv2_input_channels,
            out_channels=self.conv2_output_channels,
            device=device,
            bias_tensor=self.conv2_bias_tensor,
            kernel_size=(3, 3),
            stride=(2, 2) if self.downsample else (1, 1),
            padding=(1, 1),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=ttnn.Conv2dConfig(
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                math_fidelity=self.model_config["MATH_FIDELITY"],
                activation="relu",
            ),
            conv_op_cache=conv_op_cache,
        )

        # conv3 is 1x1 conv
        # print("Running conv3")
        out, _, _, self.conv3_weight_tensor, self.conv3_bias_tensor = ttnn.conv2d(
            input_tensor=out,
            weight_tensor=self.conv3_weight_tensor,
            in_channels=self.conv3_input_channels,
            out_channels=self.conv3_output_channels,
            device=device,
            bias_tensor=self.conv3_bias_tensor,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=ttnn.Conv2dConfig(
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                math_fidelity=self.model_config["MATH_FIDELITY"],
            ),
            conv_op_cache=conv_op_cache,
        )

        # underscore version is in_place = True
        out = ttnn.add_and_apply_activation_(out, ds_out, activation="relu", memory_config=ttnn.get_memory_config(out))

        ttnn.deallocate(ds_out)

        return out


class resnet50BottleneckPytorch:
    expansion: int = 4

    def __init__(
        self,
        parameters,
        downsample,
    ) -> None:
        # init is just to collect weight and bias parameters and variables to be used during runtime
        # for code readability
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 1

        self.conv2_weight_tensor = parameters.conv2.weight
        self.conv2_bias_tensor = parameters.conv2.bias
        self.conv2_input_channels = self.conv2_weight_tensor.shape[1]
        self.conv2_output_channels = self.conv2_weight_tensor.shape[0]
        self.conv2_stride = 2 if downsample else 1
        assert self.conv2_weight_tensor.shape[2] == 3

        self.conv3_weight_tensor = parameters.conv3.weight
        self.conv3_bias_tensor = parameters.conv3.bias
        self.conv3_input_channels = self.conv3_weight_tensor.shape[1]
        self.conv3_output_channels = self.conv3_weight_tensor.shape[0]
        assert self.conv3_weight_tensor.shape[2] == 1

        self.downsample = downsample
        if downsample:
            self.ds_conv_weight_tensor = parameters.ds_conv.weight
            self.ds_conv_bias_tensor = parameters.ds_conv.bias
            self.ds_conv_input_channels = self.ds_conv_weight_tensor.shape[1]
            self.ds_conv_output_channels = self.ds_conv_weight_tensor.shape[0]
            assert self.ds_conv_weight_tensor.shape[2] == 1

        return

    # Pytorch resnet50 block forward call
    def __call__(self, x):
        # print("Running conv1")
        out = torch.nn.functional.conv2d(
            x, self.conv1_weight_tensor, bias=self.conv1_bias_tensor.reshape(-1), stride=1, padding=0
        )
        out = torch.nn.functional.relu(out)

        # print("Running conv2")
        out = torch.nn.functional.conv2d(
            out, self.conv2_weight_tensor, bias=self.conv2_bias_tensor.reshape(-1), stride=self.conv2_stride, padding=1
        )
        out = torch.nn.functional.relu(out)
        # conv3 is 1x1 conv
        # print("Running conv3")
        out = torch.nn.functional.conv2d(
            out, self.conv3_weight_tensor, bias=self.conv3_bias_tensor.reshape(-1), stride=1, padding=0
        )

        if self.downsample:
            ds_out = torch.nn.functional.conv2d(
                x, self.ds_conv_weight_tensor, bias=self.ds_conv_bias_tensor.reshape(-1), stride=2, padding=0
            )
        else:
            ds_out = do_nothing_op(x)
        out = torch.add(out, ds_out)
        out = torch.nn.functional.relu(out)
        return out


def build_run_and_validate_ttnn_model_new(
    device,
    batch_size,
    input_height,
    input_width,
    input_channels,
    downsample,
    is_1d_systolic,
    act_dtype,
    weight_dtype,
    math_fidelity,
    parameters,
    torch_input_tensor_nchw,
    torch_golden_out_tensor_nchw,
):
    model_config = {
        "MATH_FIDELITY": math_fidelity,
        "WEIGHTS_DTYPE": weight_dtype,
        "ACTIVATIONS_DTYPE": act_dtype,
    }

    # Test new API i.e. "JIT" conv
    ttnn_model = resnet50Bottleneck(parameters, downsample, model_config)

    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    ## reshape to [1, 1, N*H*W, C]
    torch_input_tensor = torch.reshape(torch_input_tensor, (1, 1, -1, torch_input_tensor.shape[-1]))
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run 2 iterations. First iteration is warm-up i.e. W/B preprocessing and conv object caching
    conv_op_cache = {}
    for i in range(2):
        start_time = time.time()
        # Run ttnn model (1 resnet50 block)
        ttnn_out_tensor = ttnn_model(
            ttnn_input_tensor, device, batch_size, input_height, input_width, conv_op_cache=conv_op_cache
        )
        print("--- Execution time for this iteration - %s seconds ---" % (time.time() - start_time))
        # output post-processing
        ttnn_out_tensor = ttnn.to_memory_config(ttnn_out_tensor, ttnn.L1_MEMORY_CONFIG)
        ttnn_out_tensor = ttnn.to_layout(ttnn_out_tensor, ttnn.ROW_MAJOR_LAYOUT)
        torch_out_tensor = ttnn.to_torch(ttnn_out_tensor)
        output_height = input_height // (2 if downsample else 1)
        output_width = input_width // (2 if downsample else 1)
        torch_out_tensor = torch.reshape(
            torch_out_tensor, (batch_size, output_height, output_width, torch_out_tensor.shape[-1])
        )
        torch_out_tensor = torch.permute(torch_out_tensor, (0, 3, 1, 2))
        pcc_passed, pcc_message = assert_with_pcc(torch_golden_out_tensor_nchw, torch_out_tensor, pcc=0.99)


@pytest.mark.skip("Needs testing!")
@pytest.mark.parametrize(
    "batch_size, input_height, input_width, input_channels, downsample, is_1d_systolic, act_dtype, weight_dtype, math_fidelity",
    (
        (8, 56, 56, 64, False, True, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi4),  ## pass
        (8, 56, 56, 64, True, True, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi4),  ## pass
    ),
)
def test_small_resnet50_block(
    device,
    use_program_cache,
    batch_size,
    input_height,
    input_width,
    input_channels,
    downsample,
    is_1d_systolic,
    act_dtype,
    weight_dtype,
    math_fidelity,
):
    torch.manual_seed(0)
    conv1_weight_shape = [input_channels, input_channels, 1, 1]
    conv1_bias_shape = [1, 1, 1, input_channels]

    conv2_weight_shape = [input_channels, input_channels, 3, 3]
    conv2_bias_shape = [1, 1, 1, input_channels]

    conv3_weight_shape = [input_channels, input_channels, 1, 1]
    conv3_bias_shape = [1, 1, 1, input_channels]

    conv1_input_shape = [batch_size, input_channels, input_height, input_width]
    torch_input_tensor_nchw = torch.randn(conv1_input_shape, dtype=torch.bfloat16).float()

    torch_conv1_weight_tensor = torch.randn(conv1_weight_shape, dtype=torch.bfloat16).float()
    torch_conv1_bias_tensor = torch.randn(conv1_bias_shape, dtype=torch.bfloat16).float()

    torch_conv2_weight_tensor = torch.randn(conv2_weight_shape, dtype=torch.bfloat16).float()
    torch_conv2_bias_tensor = torch.randn(conv2_bias_shape, dtype=torch.bfloat16).float()

    torch_conv3_weight_tensor = torch.randn(conv3_weight_shape, dtype=torch.bfloat16).float()
    torch_conv3_bias_tensor = torch.randn(conv3_bias_shape, dtype=torch.bfloat16).float()

    parameters = {
        "conv1": {"weight": torch_conv1_weight_tensor, "bias": torch_conv1_bias_tensor},
        "conv2": {"weight": torch_conv2_weight_tensor, "bias": torch_conv2_bias_tensor},
        "conv3": {"weight": torch_conv3_weight_tensor, "bias": torch_conv3_bias_tensor},
    }
    parameters = dotdict(parameters)
    parameters.conv1 = dotdict(parameters.conv1)
    parameters.conv2 = dotdict(parameters.conv2)
    parameters.conv3 = dotdict(parameters.conv3)

    if downsample:
        ds_conv_weight_shape = [input_channels, input_channels, 1, 1]
        ds_conv_bias_shape = [1, 1, 1, input_channels]
        torch_ds_conv_weight_tensor = torch.randn(ds_conv_weight_shape, dtype=torch.bfloat16).float()
        torch_ds_conv_bias_tensor = torch.randn(ds_conv_bias_shape, dtype=torch.bfloat16).float()
        parameters.ds_conv = {"weight": torch_ds_conv_weight_tensor, "bias": torch_ds_conv_bias_tensor}
        parameters.ds_conv = dotdict(parameters.ds_conv)

    torch_model = resnet50BottleneckPytorch(parameters, downsample)
    torch_golden_out_tensor_nchw = torch_model(torch_input_tensor_nchw)
    build_run_and_validate_ttnn_model_new(
        device,
        batch_size,
        input_height,
        input_width,
        input_channels,
        downsample,
        is_1d_systolic,
        act_dtype,
        weight_dtype,
        math_fidelity,
        parameters,
        torch_input_tensor_nchw,
        torch_golden_out_tensor_nchw,
    )
