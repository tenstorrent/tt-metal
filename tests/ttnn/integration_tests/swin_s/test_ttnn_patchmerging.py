# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
import pytest
import ttnn
from ttnn.model_preprocessing import preprocess_model, preprocess_linear_weight, preprocess_layernorm_parameter
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_swin_s.reference.patchmerging import PatchMerging
from models.experimental.functional_swin_s.tt.tt_patchmerging import TtPatchMerging


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_patchmerging(device, reset_seeds):
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = state_dict = model.state_dict()
    patchmerging_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("features.2."))}

    if not patchmerging_state_dict:
        raise ValueError("No parameters found in resblock_state_dict")

    torch_model = PatchMerging(96)
    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in patchmerging_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    # Input tensor for testing
    torch_input_tensor = torch.randn(8, 128, 128, 96)  # Sample input tensor
    torch_output_tensor = torch_model(torch_input_tensor)

    # Preprocess the model for TTNN
    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        # custom_preprocessor=create_custom_preprocessor(ttnn.device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    # Convert the model to TTNN
    ttnn_model = TtPatchMerging(device, parameters, 96)

    # Convert input tensor to TTNN format
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Apply TTNN model
    output_tensor = ttnn_model(input_tensor)

    # Convert output tensor back to Torch format
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
