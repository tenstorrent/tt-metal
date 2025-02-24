# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import torch.nn as nn
import ttnn
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_rms_norm import (
    ttnn_RMSNorm as tt_module,
)
from models.experimental.functional_stable_diffusion3_5.reference.rms_norm import RMSNorm
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.utility_functions import skip_for_grayskull


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name):
        parameters = {}
        if isinstance(model, RMSNorm):
            parameters["rms_norm"] = {}
            parameters["rms_norm"]["weight"] = ttnn.from_torch(
                model.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "init_inputs,fwd_inputs",
    [
        ((64, 1e-06, True), (2, 24, 333, 64)),
    ],
)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_rms_norm(init_inputs, fwd_inputs, device, reset_seeds):
    torch_sub_module = RMSNorm(dim=init_inputs[0], eps=init_inputs[1], elementwise_affine=init_inputs[2])
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_sub_module, device=device, custom_preprocessor=create_custom_preprocessor(device)
    )
    hidden_states = torch.randn(fwd_inputs, dtype=torch.bfloat16)
    tt_input_hidden_states = ttnn.from_torch(
        hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_sub_module = tt_module(
        dim=init_inputs[0], eps=init_inputs[1], elementwise_affine=init_inputs[2], parameters=parameters.rms_norm
    )
    tt_out = tt_sub_module(hidden_states=tt_input_hidden_states, device=device)
    torch_out = torch_sub_module(hidden_states)
    tt_out_in_torch = ttnn.to_torch(tt_out)
    assert_with_pcc(torch_out, tt_out_in_torch, 0.99)
