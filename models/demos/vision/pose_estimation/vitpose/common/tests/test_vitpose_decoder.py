# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.vision.pose_estimation.vitpose.common.common import load_torch_model
from models.demos.vision.pose_estimation.vitpose.common.reference.vitpose_reference import (
    extract_reference_parameters,
    vitpose_simple_decoder,
)
from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_decoder import (
    VitPoseSimpleDecoder,
    preprocess_decoder_parameters,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_vitpose_simple_decoder(device, batch_size):
    torch.manual_seed(0)

    model = load_torch_model()
    state_dict = model.state_dict()
    ref_params = extract_reference_parameters(model)

    hidden_states = torch.randn(batch_size, 192, 768, dtype=torch.bfloat16)
    torch_output = vitpose_simple_decoder(hidden_states, parameters=ref_params["head"])

    tt_params = preprocess_decoder_parameters(state_dict, dtype=ttnn.bfloat16)
    decoder = VitPoseSimpleDecoder(tt_params, device, batch_size=batch_size)

    tt_input = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device)

    tt_output = decoder(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    tt_output_reshaped = tt_output.reshape(batch_size, 64, 48, 17).permute(0, 3, 1, 2)
    assert_with_pcc(torch_output.float(), tt_output_reshaped.float(), 0.99)
