# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.vision.pose_estimation.vitpose.common.common import load_torch_model
from models.demos.vision.pose_estimation.vitpose.common.reference.vitpose_reference import (
    extract_reference_parameters,
    vitpose_attention,
)
from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_attention import (
    preprocess_attention_parameters,
    vitpose_attention as ttnn_vitpose_attention,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("layer_idx", [0])
def test_vitpose_attention(device, batch_size, layer_idx):
    torch.manual_seed(0)

    model = load_torch_model()
    state_dict = model.state_dict()
    ref_params = extract_reference_parameters(model)

    hidden_states = torch.randn(batch_size, 192, 768, dtype=torch.bfloat16)
    torch_output = vitpose_attention(hidden_states, parameters=ref_params["backbone"]["encoder"][layer_idx])

    tt_params = preprocess_attention_parameters(state_dict, layer_idx, dtype=ttnn.bfloat16)
    tt_params = {k: ttnn.to_device(v, device) for k, v in tt_params.items()}

    tt_input = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device)

    tt_output = ttnn_vitpose_attention(tt_input, parameters=tt_params)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output.float(), tt_output.float(), 0.99)
