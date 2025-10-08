# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import ttnn
from models.demos.mobilenetv2.common import MOBILENETV2_BATCH_SIZE, MOBILENETV2_L1_SMALL_SIZE, load_torch_model
from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.demos.mobilenetv2.tt import ttnn_mobilenetv2
from models.demos.mobilenetv2.tt.model_preprocessing import (
    create_mobilenetv2_input_tensors,
    create_mobilenetv2_model_parameters,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
    ids=[
        "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [
        MOBILENETV2_BATCH_SIZE,
    ],
)
def test_mobilenetv2(device, use_pretrained_weight, batch_size, reset_seeds, model_location_generator):
    if use_pretrained_weight:
        torch_model = Mobilenetv2()
        torch_model = load_torch_model(torch_model, model_location_generator)
    else:
        torch_model = Mobilenetv2()
        state_dict = torch_model.state_dict()

    torch_model.eval()

    torch_input_tensor, ttnn_input_tensor = create_mobilenetv2_input_tensors(
        batch=batch_size, input_height=224, input_width=224
    )
    torch_output_tensor = torch_model(torch_input_tensor)

    model_parameters = create_mobilenetv2_model_parameters(torch_model, device=device)

    ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(model_parameters, device, batchsize=batch_size)
    output_tensor = ttnn_model(ttnn_input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.944 if use_pretrained_weight else 0.999)
