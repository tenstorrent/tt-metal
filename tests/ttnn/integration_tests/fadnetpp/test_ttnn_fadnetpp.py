# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import tests.ttnn.integration_tests.fadnetpp.custom_preprocessor_dispnetc as c_dispnetc
import tests.ttnn.integration_tests.fadnetpp.custom_preprocessor_dispnetres as c_dispnetres
from models.experimental.functional_fadnetpp.reference.fadnetpp import FadNetPP
from models.experimental.functional_fadnetpp.tt.tt_fadnetpp import TtFadNetPP
from models.utility_functions import skip_for_wormhole_b0
from ttnn.model_preprocessing import preprocess_model
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        parameters["dispnetc"] = c_dispnetc.custom_preprocessor(
            device, model.dispnetc, name, ttnn_module_args["dispnetc"]
        )
        parameters["dispnetres"] = c_dispnetres.custom_preprocessor(
            device, model.dispnetres, name, ttnn_module_args["dispnetres"]
        )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@skip_for_wormhole_b0()
def test_fadnetpp_model(device, reset_seeds, model_location_generator):
    in_planes = 3 * 3 + 1 + 1
    torch_model = FadNetPP(in_planes)
    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    ds_state_dict = {k: v for k, v in torch_model.state_dict().items()}
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 6, 960, 576)  # Batch size of 1, 6 input channels, 960x576 height and width
    (torch_output_tensor0, torch_output_tensor1) = torch_model(torch_input_tensor)
    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )
    ttnn_model = TtFadNetPP(parameters, device, in_planes, torch_model)
    #
    # Tensor Preprocessing
    #
    imgs = torch.chunk(torch_input_tensor, 2, dim=1)
    img_left = imgs[0]
    img_right = imgs[1]
    img_left = torch.permute(img_left, (0, 2, 3, 1))
    img_right = torch.permute(img_right, (0, 2, 3, 1))
    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = torch_input_tensor.reshape(
        torch_input_tensor.shape[0],
        1,
        torch_input_tensor.shape[1] * torch_input_tensor.shape[2],
        torch_input_tensor.shape[3],
    )
    input_tensor1 = img_left.reshape(img_left.shape[0], 1, img_left.shape[1] * img_left.shape[2], img_left.shape[3])
    input_tensor2 = img_right.reshape(
        img_right.shape[0], 1, img_right.shape[1] * img_right.shape[2], img_right.shape[3]
    )

    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_tensor1 = ttnn.from_torch(input_tensor1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_tensor2 = ttnn.from_torch(input_tensor2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    (output_tensor0, output_tensor1) = ttnn_model(device, input_tensor, input_tensor1, input_tensor2)

    #
    # Tensor Postprocessing
    #

    output_tensor0 = ttnn.to_torch(output_tensor0)
    output_tensor0 = output_tensor0.reshape(1, 960, 576, 1)
    output_tensor0 = torch.permute(output_tensor0, (0, 3, 1, 2))
    output_tensor0 = output_tensor0.to(torch_input_tensor.dtype)

    output_tensor1 = ttnn.to_torch(output_tensor1)
    output_tensor1 = output_tensor1.reshape(1, 960, 576, 1)
    output_tensor1 = torch.permute(output_tensor1, (0, 3, 1, 2))
    output_tensor1 = output_tensor1.to(torch_input_tensor.dtype)

    assert_with_pcc(torch_output_tensor0, output_tensor0, pcc=0.99)
    assert_with_pcc(torch_output_tensor1, output_tensor1, pcc=0.99)
