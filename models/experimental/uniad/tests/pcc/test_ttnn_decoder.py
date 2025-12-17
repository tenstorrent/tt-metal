# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

import pytest
import torch
from torch import nn

from models.experimental.uniad.reference.decoder import DetectionTransformerDecoder, FFN

from models.experimental.uniad.tt.ttnn_decoder import TtDetectionTransformerDecoder

from models.experimental.uniad.tt.model_preprocessing_perception_transformer import extract_sequential_branch

from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight, preprocess_linear_bias

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.uniad.common import load_torch_model


class DotDict(dict):
    def __getattr__(self, key):
        return self[key]


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, FFN):
        parameters = {
            "ffn": {},
        }
        parameters["ffn"][f"ffn0"] = {
            "linear1": {
                "weight": preprocess_linear_weight(model.layers[0][0].weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.layers[0][0].bias, dtype=ttnn.bfloat16),
            },
            "linear2": {
                "weight": preprocess_linear_weight(model.layers[1].weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.layers[1].bias, dtype=ttnn.bfloat16),
            },
        }
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_uniad_decoder(device, reset_seeds, model_location_generator):
    reference_model = DetectionTransformerDecoder(num_layers=6, embed_dim=256, num_heads=8)

    reference_model = load_torch_model(
        torch_model=reference_model,
        layer="pts_bbox_head.transformer.decoder",
        model_location_generator=model_location_generator,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )

    query = torch.randn(901, 1, 256)
    kwargs = {}
    kwargs["key"] = None
    kwargs["value"] = torch.randn(2500, 1, 256)
    kwargs["query_pos"] = torch.randn(901, 1, 256)
    kwargs["spatial_shapes"] = torch.Tensor([[50, 50]]).to(dtype=torch.int64)
    reference_points = torch.randn(1, 901, 3)

    def create_block():
        return nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))

    reg_branches = nn.ModuleList([create_block() for _ in range(6)])

    parameters_branches = {}
    parameters_branches = extract_sequential_branch(reg_branches, dtype=ttnn.bfloat16, device=device)

    parameters_branches = DotDict(parameters_branches)

    ttnn_model = TtDetectionTransformerDecoder(6, 256, 8, parameters, device)

    output1, output2 = reference_model(
        query=query,
        key=kwargs["key"],
        value=kwargs["value"],
        query_pos=kwargs["query_pos"],
        reference_points=reference_points,
        spatial_shapes=kwargs["spatial_shapes"],
        reg_branches=reg_branches,
    )

    ttnn_output1, ttnn_output2 = ttnn_model(
        query=ttnn.from_torch(query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        key=None,
        value=ttnn.from_torch(kwargs["value"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        query_pos=ttnn.from_torch(kwargs["query_pos"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        reference_points=ttnn.from_torch(reference_points, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        spatial_shapes=ttnn.from_torch(
            kwargs["spatial_shapes"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
        reg_branches=parameters_branches,
    )

    pcc1, x = assert_with_pcc(output1, ttnn.to_torch(ttnn_output1), 0.99)
    pcc2, y = assert_with_pcc(output2, ttnn.to_torch(ttnn_output2), 0.99)
