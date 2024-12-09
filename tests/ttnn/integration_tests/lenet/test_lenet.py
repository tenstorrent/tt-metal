# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import is_grayskull
from models.demos.lenet.tt import tt_lenet
from models.demos.lenet import lenet_utils


@pytest.mark.parametrize(
    "batch_size",
    [64],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_lenet(device, batch_size, model_location_generator, reset_seeds):
    num_classes = 10
    test_input, images, outputs = lenet_utils.get_test_data(batch_size)

    pt_model_path = model_location_generator("model.pt", model_subdir="LeNet")
    torch_lenet, state_dict = lenet_utils.load_torch_lenet(pt_model_path, num_classes)
    model = torch_lenet.float()
    torch_output = model(test_input)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=lenet_utils.custom_preprocessor,
    )
    parameters = lenet_utils.custom_preprocessor_device(parameters, device)
    x = test_input.permute(0, 2, 3, 1)
    x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_output = tt_lenet.lenet(x, device, parameters)

    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.997 if is_grayskull() else 0.9993)
