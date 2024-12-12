# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_time_steps import ttnn_Timesteps as tt_module
from models.experimental.functional_stable_diffusion3_5.reference.time_steps import Timesteps
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.utility_functions import skip_for_grayskull


@pytest.mark.parametrize(
    "init_inputs",
    [
        (256, True, 0, 1),
    ],
)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_time_steps(init_inputs, device, reset_seeds):
    torch_sub_module = Timesteps(init_inputs[0], init_inputs[1], init_inputs[2], init_inputs[3])
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_sub_module,
        device=device,
    )
    time_stamps = torch.tensor([100, 100], dtype=torch.int32)
    tt_input = ttnn.from_torch(
        time_stamps, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_input = ttnn.squeeze(tt_input, dim=0)
    tt_sub_module = tt_module(init_inputs[0], init_inputs[1], init_inputs[2], init_inputs[3])
    tt_out = tt_sub_module(tt_input, device)
    torch_out = torch_sub_module(time_stamps)
    tt_out_in_torch = ttnn.to_torch(tt_out)
    assert_with_pcc(torch_out, tt_out_in_torch, 0.96)
