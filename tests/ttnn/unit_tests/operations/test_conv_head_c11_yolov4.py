# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
import tt_lib
import pytest
from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d, preprocess_model

from tests.ttnn.utils_for_testing import assert_with_pcc


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256
    ttnn_module_args["use_shallow_conv_variant"] = False


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        print("ttnn_module_args", ttnn_module_args)
        parameters = {}
        parameters = TtHead.custom_preprocessor(device, model, name, ttnn_module_args)
        return parameters

    return custom_preprocessor


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.c11 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.b11 = nn.BatchNorm2d(512)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, inputs):
        x11 = self.c11(inputs)
        x11 = self.b11(x11)
        x11 = self.relu(x11)
        return x11


class TtHead:
    def custom_preprocessor(device, model, name, ttnn_module_args):
        print("We do reach here!")
        parameters = {}
        if isinstance(model, Head):
            ttnn_module_args.c11["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c11["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c11["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c11["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c11["activation"] = None  # Fuse relu with conv1
            ttnn_module_args.c11["deallocate_activation"] = False
            ttnn_module_args.c11["conv_blocking_and_parallelization_config_override"] = None
            conv11_weight, conv11_bias = fold_batch_norm2d_into_conv2d(model.c11, model.b11)
            update_ttnn_module_args(ttnn_module_args.c11)
            parameters["c11"], c11_parallel_config = preprocess_conv2d(
                conv11_weight, conv11_bias, ttnn_module_args.c11, return_parallel_config=True
            )

    def __init__(self, device, parameters) -> None:
        self.device = device
        print("keys in parameters in TtHead are: ", parameters.keys())
        self.c11 = parameters.c11
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def __call__(self, inputs):
        inputs = tt_lib.tensor.interleaved_to_sharded(inputs, self.c11.conv.input_sharded_memory_config)
        output_tensor = self.c11(inputs)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.relu(output_tensor)

        return output_tensor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unit_head_c11(device, model_location_generator):
    model_path = model_location_generator("models", model_subdir="Yolo")
    if model_path == "models":
        weights_pth = "tests/ttnn/integration_tests/yolov4/yolov4.pth"
    else:
        weights_pth = str(model_path / "yolov4.pth")
    state_dict = torch.load(weights_pth)
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(("head.conv11")))}

    torch_model = Head()

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
    print("new_state_dict", new_state_dict.keys())
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    reader_patterns_cache = {}

    input = torch.load("tests/ttnn/integration_tests/yolov4/head_c11_input.pt")
    # input = torch.randn(1, 1, 400, 256)
    torch_input = torch.reshape(input.float(), (1, 20, 20, 256))
    torch_input = torch.permute(torch_input, (0, 3, 1, 2))

    ttnn_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtHead(device, parameters)

    torch_output = torch_model(torch_input)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = torch.reshape(ttnn_output, (1, 10, 10, 512))
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))

    assert_with_pcc(torch_output, ttnn_output)
