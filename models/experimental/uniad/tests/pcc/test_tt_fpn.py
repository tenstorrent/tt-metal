# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.experimental.uniad.reference.fpn import FPN

from models.experimental.uniad.tt.ttnn_fpn import TtFPN
from models.experimental.uniad.tt.model_preprocessing_encoder import (
    create_uniad_FPN_parameters,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.uniad.common import load_torch_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_uniad_fpn(
    device,
    reset_seeds,
    model_location_generator,
):
    torch_model = FPN(
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        add_extra_convs="on_output",
    )

    torch_model = load_torch_model(
        torch_model=torch_model, layer="img_neck", model_location_generator=model_location_generator
    )

    tensor1 = torch.randn(6, 512, 80, 45)
    tensor2 = torch.randn(6, 1024, 40, 23)
    tensor3 = torch.randn(6, 2048, 20, 12)

    input_tensors = [tensor1, tensor2, tensor3]
    parameter = create_uniad_FPN_parameters(torch_model, input_tensors=input_tensors, device=device)
    torch_output = torch_model(input_tensors)

    tensor1 = torch.permute(tensor1, (0, 2, 3, 1))
    tensor1 = torch.reshape(tensor1, [1, 1, tensor1.shape[0] * tensor1.shape[1] * tensor1.shape[2], tensor1.shape[-1]])

    tensor2 = torch.permute(tensor2, (0, 2, 3, 1))
    tensor2 = torch.reshape(tensor2, [1, 1, tensor2.shape[0] * tensor2.shape[1] * tensor2.shape[2], tensor2.shape[-1]])

    tensor3 = torch.permute(tensor3, (0, 2, 3, 1))
    tensor3 = torch.reshape(tensor3, [1, 1, tensor3.shape[0] * tensor3.shape[1] * tensor3.shape[2], tensor3.shape[-1]])

    ttnn_input_tensors = [
        ttnn.from_torch(tensor1, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT),
        ttnn.from_torch(tensor2, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT),
        ttnn.from_torch(tensor3, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT),
    ]

    tt_model = TtFPN(
        conv_args=parameter.model_args,
        conv_pth=parameter,
        device=device,
    )

    ttnn_outputs = tt_model(ttnn_input_tensors)

    for i in range(4):
        n, c, h, w = torch_output[i].shape
        ttnn_output = ttnn.to_torch(ttnn_outputs[i])
        ttnn_output = ttnn_output.reshape(n, h, w, c)
        ttnn_output = ttnn_output.permute(0, 3, 1, 2)
        assert_with_pcc(ttnn_output, torch_output[i], 0.99)
