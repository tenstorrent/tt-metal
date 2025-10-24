# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from loguru import logger
from models.experimental.functional_petr.tt.ttnn_cp_fpn import ttnn_CPFPN
from models.experimental.functional_petr.reference.cp_fpn import CPFPN
from models.experimental.functional_petr.tt.common import create_custom_preprocessor_cpfpn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_cp_fpn(device, reset_seeds):
    torch_model = CPFPN(in_channels=[768, 1024], out_channels=256, num_outs=2)
    torch_model.eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor_cpfpn(None), device=None
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
        pcc_threshold = 0.99
        passed, msg = check_with_pcc(torch_output[i], ttnn_output_check, pcc=pcc_threshold)
        assert_with_pcc(ttnn_output_check, torch_output[i], pcc=0.99)
        logger.info(f"cp_fpn layer  passed: " f"PCC={msg}")
