# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
import tt_lib as ttl
import tt_lib.profiler as profiler

from models.experimental.functional_yolov4.reference.downsamples import DownSamples
from models.experimental.functional_yolov4.tt.ttnn_downsamples import TtDownSamples

import tests.ttnn.integration_tests.yolov4.custom_preprocessor_d1 as D1
import tests.ttnn.integration_tests.yolov4.custom_preprocessor_d2 as D2
import tests.ttnn.integration_tests.yolov4.custom_preprocessor_d3 as D3
import tests.ttnn.integration_tests.yolov4.custom_preprocessor_d4 as D4
import tests.ttnn.integration_tests.yolov4.custom_preprocessor_d5 as D5

import ttnn
import tt_lib
from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d
import ttnn


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        parameters["downsample1"] = D1.custom_preprocessor(
            device, model.downsample1, name, ttnn_module_args["downsample1"]
        )
        parameters["downsample2"] = D2.custom_preprocessor(
            device, model.downsample2, name, ttnn_module_args["downsample2"]
        )
        parameters["downsample3"] = D3.custom_preprocessor(
            device, model.downsample3, name, ttnn_module_args["downsample3"]
        )
        parameters["downsample4"] = D4.custom_preprocessor(
            device, model.downsample4, name, ttnn_module_args["downsample4"]
        )
        parameters["downsample5"] = D5.custom_preprocessor(
            device, model.downsample5, name, ttnn_module_args["downsample5"]
        )
        return parameters

    return custom_preprocessor


@skip_for_wormhole_b0()
def test_downsample1(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/yolov4/yolov4.pth")
    ds_state_dict = {
        k: v for k, v in state_dict.items() if (k.startswith(("down1.", "down2.", "down3.", "down4.", "down5.")))
    }
    torch_model = DownSamples()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 3, 320, 320)  # Batch size of 1, 128 input channels, 160x160 height and width
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )
    ttnn_model = TtDownSamples(parameters)

    # Tensor Preprocessing
    #
    input_shape = torch_input_tensor.shape
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    # output_tensor = ttnn_model(device, input_tensor)
    with ttnn.tracer.trace():
        output_tensor = ttnn_model(device, input_tensor)
    ttnn.tracer.visualize(output_tensor, file_name="downsamples.svg")
    #
    # Tensor Postprocessing
    #
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.reshape(1, 10, 10, 1024)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
