# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.experimental.functional_yolov4.reference.neck import Neck
from models.experimental.functional_yolov4.tt.ttnn_neck import TtNeck
import tests.ttnn.integration_tests.yolov4.custom_preprocessor_neck as neck
import ttnn
import tt_lib


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        parameters = neck.custom_preprocessor(device, model, name, ttnn_module_args)
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_downsample1(device, reset_seeds, model_location_generator):
    model_path = model_location_generator("models", model_subdir="Yolo")
    weights_pth = str(model_path / "yolov4.pth")
    state_dict = torch.load(weights_pth)
    state_dict = torch.load(weights_pth)
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("neek."))}

    torch_model = Neck()

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor1 = torch.randn(1, 1024, 10, 10)  # Batch size of 1, 128 input channels, 160x160 height and width
    torch_input_tensor2 = torch.randn(1, 512, 20, 20)
    torch_input_tensor3 = torch.randn(1, 256, 40, 40)
    torch_input_tensor = [torch_input_tensor1, torch_input_tensor2, torch_input_tensor3]
    torch_output_tensor1, torch_output_tensor2, torch_output_tensor3 = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtNeck(device, parameters)

    # Tensor Preprocessing
    #
    input_tensor1 = torch.permute(torch_input_tensor1, (0, 2, 3, 1))
    input_tensor1 = input_tensor1.reshape(
        input_tensor1.shape[0], 1, input_tensor1.shape[1] * input_tensor1.shape[2], input_tensor1.shape[3]
    )
    input_tensor1 = ttnn.from_torch(input_tensor1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    input_tensor2 = torch.permute(torch_input_tensor2, (0, 2, 3, 1))
    input_tensor2 = input_tensor2.reshape(
        input_tensor2.shape[0], 1, input_tensor2.shape[1] * input_tensor2.shape[2], input_tensor2.shape[3]
    )
    input_tensor2 = ttnn.from_torch(input_tensor2, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    input_tensor3 = torch.permute(torch_input_tensor3, (0, 2, 3, 1))
    input_tensor3 = input_tensor3.reshape(
        input_tensor3.shape[0], 1, input_tensor3.shape[1] * input_tensor3.shape[2], input_tensor3.shape[3]
    )
    input_tensor3 = ttnn.from_torch(input_tensor3, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    input_tensor = [input_tensor1, input_tensor2, input_tensor3]
    output_tensor1, output_tensor2, output_tensor3 = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #
    output_tensor1 = ttnn.to_torch(output_tensor1)
    output_tensor1 = output_tensor1.reshape(1, 40, 40, 128)
    output_tensor1 = torch.permute(output_tensor1, (0, 3, 1, 2))
    output_tensor1 = output_tensor1.to(torch_input_tensor1.dtype)

    output_tensor2 = ttnn.to_torch(output_tensor2)
    output_tensor2 = output_tensor2.reshape(1, 10, 10, 512)
    output_tensor2 = torch.permute(output_tensor2, (0, 3, 1, 2))
    output_tensor2 = output_tensor2.to(torch_input_tensor2.dtype)

    output_tensor3 = ttnn.to_torch(output_tensor3)
    output_tensor3 = output_tensor3.reshape(1, 20, 20, 256)
    output_tensor3 = torch.permute(output_tensor3, (0, 3, 1, 2))
    output_tensor3 = output_tensor3.to(torch_input_tensor3.dtype)

    assert_with_pcc(torch_output_tensor1, output_tensor1, pcc=0.93)  # PCC = 0.9399853493196958
    assert_with_pcc(torch_output_tensor2, output_tensor2, pcc=0.93)  # PCC = 0.9391142122379718
    assert_with_pcc(torch_output_tensor3, output_tensor3, pcc=0.92)  # PCC = 0.9299983109764207
