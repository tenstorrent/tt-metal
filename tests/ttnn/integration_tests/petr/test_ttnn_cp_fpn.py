# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.utility_functions import skip_for_grayskull
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_petr.tt.ttnn_cp_fpn import ttnn_CPFPN
from models.experimental.functional_petr.reference.cp_fpn import CPFPN


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, CPFPN):
            parameters["lateral_convs"] = {}
            for i, child in enumerate(model.lateral_convs):
                parameters["lateral_convs"][i] = {}
                parameters["lateral_convs"][i]["conv"] = {}
                parameters["lateral_convs"][i]["conv"]["weight"] = ttnn.from_torch(
                    child.conv.weight, dtype=ttnn.bfloat16
                )
                parameters["lateral_convs"][i]["conv"]["bias"] = ttnn.from_torch(
                    torch.reshape(child.conv.bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )
            parameters["fpn_convs"] = {}
            for i, child in enumerate(model.fpn_convs):
                parameters["fpn_convs"][i] = {}
                parameters["fpn_convs"][i]["conv"] = {}
                parameters["fpn_convs"][i]["conv"]["weight"] = ttnn.from_torch(child.conv.weight, dtype=ttnn.bfloat16)
                parameters["fpn_convs"][i]["conv"]["bias"] = ttnn.from_torch(
                    torch.reshape(child.conv.bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
def test_cp_fpn(device, reset_seeds):
    torch_model = CPFPN(in_channels=[768, 1024], out_channels=256, num_outs=2)
    torch_model.eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    input_a = torch.randn(6, 768, 20, 50)
    input_b = torch.randn(6, 1024, 10, 25)
    torch_output = torch_model([input_a, input_b])
    ttnn_model = ttnn_CPFPN(in_channels=[768, 1024], out_channels=256, num_outs=2, parameters=parameters)

    ttnn_input_1 = ttnn.from_torch(input_a.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input_2 = ttnn.from_torch(input_b.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn_model(device, [ttnn_input_1, ttnn_input_2])

    for i in range(len(ttnn_output)):
        ttnn_output_check = ttnn.to_torch(ttnn_output[i])
        ttnn_output_check = ttnn_output_check.permute(0, 3, 1, 2)
        assert_with_pcc(ttnn_output_check, torch_output[i], pcc=0.99)
