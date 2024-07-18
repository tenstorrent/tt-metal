# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
from models.experimental.functional_swin_s.reference.swin_transformer_block import SwinTransformerBlock
from models.experimental.functional_swin_s.tt.tt_swin_transformer_block import TtSwinTransformerBlock
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
from ttnn.model_preprocessing import preprocess_model
import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_shifted_window_attention(device, reset_seeds):
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = model.state_dict()
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("features.3.0."))}

    torch_model = SwinTransformerBlock(192, 6, [7, 7], [0, 0])

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(8, 64, 64, 192)
    torch_output = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        # custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtSwinTransformerBlock(device, parameters, 192, 6, [7, 7], [0, 0])

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_output = ttnn_model(input_tensor)

    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.99)
