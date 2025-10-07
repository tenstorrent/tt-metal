# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.efficientnetb0.tt.model_preprocessing import (
    create_efficientnetb0_input_tensors,
    create_efficientnetb0_model_parameters,
)
from models.experimental.efficientnetb0.tt import efficientnetb0 as ttnn_efficientnetb0
from models.experimental.efficientnetb0.common import load_torch_model, EFFICIENTNETB0_L1_SMALL_SIZE


@pytest.mark.parametrize("device_params", [{"l1_small_size": EFFICIENTNETB0_L1_SMALL_SIZE}], indirect=True)
def test_efficientnetb0_model(device, reset_seeds, model_location_generator):
    torch_model = load_torch_model(model_location_generator)

    torch_input, ttnn_input = create_efficientnetb0_input_tensors(device)
    torch_output = torch_model(torch_input)
    conv_params, parameters = create_efficientnetb0_model_parameters(torch_model, torch_input, device=device)

    ttnn_model = ttnn_efficientnetb0.Efficientnetb0(device, parameters, conv_params)

    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.92)
