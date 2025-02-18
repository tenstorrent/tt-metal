# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import skip_for_grayskull
from models.experimental.functional_vanilla_unet.reference.unet import UNet
from models.experimental.functional_vanilla_unet.ttnn.ttnn_unet import TtUnet
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_vanilla_unet.ttnn.model_preprocesser import create_custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@skip_for_grayskull()
def test_unet(device, reset_seeds, model_location_generator):
    state_dict = torch.load(
        "models/experimental/functional_vanilla_unet/unet.pt",
        map_location=torch.device("cpu"),
    )
    ds_state_dict = {k: v for k, v in state_dict.items()}

    reference_model = UNet()

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_input_tensor = torch.randn(1, 3, 480, 640)
    torch_output_tensor = reference_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    ttnn_model = TtUnet(device=device, parameters=parameters, model=reference_model)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1), device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    ttnn_output = ttnn_model(device, ttnn_input_tensor)
    print("test")

    print(ttnn_output.shape, torch_output_tensor.shape)
    # ttnn_output = ttnn.to_torch(ttnn_output)
    # ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    # ttnn_output = ttnn_output.reshape(torch_output_tensor.shape)

    assert_with_pcc(torch_output_tensor, ttnn_output, pcc=1.0)
