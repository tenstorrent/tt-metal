# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import check_pcc_conv


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_unet_model(batch, groups, device, use_program_cache, reset_seeds):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=True)
    model = unet_shallow_torch.UNet.from_random_weights(groups=1)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    torch_output_tensor = model(torch_input)
    output_tensor = ttnn_model(ttnn_input)

    check_pcc_conv(torch_output_tensor, output_tensor, 0.97)
