# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from models.experimental.detr3d.reference.detr3d_model import PositionEmbeddingCoordsSine as ref_model
from models.experimental.detr3d.source.detr3d.models.position_embedding import PositionEmbeddingCoordsSine as org_model
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "temperature, normalize, scale, pos_type, d_pos, d_in, gauss_scale, xyz_shape,num_channels, input_range_shape",
    [
        # Case 1
        (10000, True, None, "fourier", 256, 3, 1.0, (1, 128, 3), None, (2, 1, 3)),
        # Case 2
        (
            10000,
            True,
            None,
            "fourier",
            256,
            3,
            1.0,
            (1, 1024, 3),
            None,
            (2, 1, 3),
        ),
    ],
)
def test_position_embedding(
    temperature, normalize, scale, pos_type, d_pos, d_in, gauss_scale, xyz_shape, input_range_shape, num_channels
):
    org_module = org_model(temperature, normalize, scale, pos_type, d_pos, d_in, gauss_scale)  # .to(torch.bfloat16)
    org_module.eval()
    ref_module = ref_model(temperature, normalize, scale, pos_type, d_pos, d_in, gauss_scale)  # .to(torch.bfloat16)
    ref_module.eval()
    print("sd", org_module.gauss_B.shape)
    ref_module.load_state_dict(org_module.state_dict())
    xyz = torch.randn(xyz_shape, dtype=torch.float32)  # bfloat16
    input_range_tensors = [torch.randn(input_range_shape[1:]) for _ in range(input_range_shape[0])]
    org_out = org_module(xyz, num_channels, input_range_tensors)
    ref_out = ref_module(xyz, num_channels, input_range_tensors)
    assert_with_pcc(org_out, ref_out, 1.0)
