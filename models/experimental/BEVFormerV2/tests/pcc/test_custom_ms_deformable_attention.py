# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.BEVFormerV2.reference.multihead_attention import CustomMSDeformableAttention
from models.experimental.BEVFormerV2.tt.ttnn_custom_ms_deformable_attention import TtCustomMSDeformableAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.experimental.BEVFormerV2.common import download_bevformerv2_weights


from models.experimental.BEVFormerV2.tests.pcc.custom_preprocessors import (
    custom_preprocessor_custom_ms_deformable_attention as custom_preprocessor,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_custom_ms_deformable_attention(
    device,
    reset_seeds,
    model_location_generator,
):
    embed_dims = 256
    num_heads = 8
    num_levels = 1  # Decoder cross-attention uses 1 level
    num_points = 4
    batch_first = False

    torch_model = CustomMSDeformableAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        batch_first=batch_first,
    )

    weights_path = download_bevformerv2_weights()
    checkpoint = torch.load(weights_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    custom_state = {}
    for key, value in state_dict.items():
        if "decoder.layers.0.attentions.1" in key:
            new_key = key.replace("pts_bbox_head.transformer.decoder.layers.0.attentions.1.", "")
            custom_state[new_key] = value

    torch_model.load_state_dict(custom_state, strict=False)
    torch_model.eval()

    bs = 1
    num_query = 900
    spatial_shapes = torch.tensor([[100, 100]])
    num_value = int(torch.prod(spatial_shapes, dim=1).sum().item())

    query = torch.randn(num_query, bs, embed_dims)
    value = torch.randn(num_value, bs, embed_dims)
    reference_points = torch.rand(bs, num_query, num_levels, 2)
    level_start_index = torch.tensor([0])

    torch_output = torch_model(
        query,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    ttnn_model = TtCustomMSDeformableAttention(
        params=parameters.custom_ms_deformable_attention,
        device=device,
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        batch_first=batch_first,
    )

    query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    value = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    reference_points = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16)
    spatial_shapes = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.uint32)
    level_start_index = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    ttnn_output = ttnn_model(
        query,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)
    pcc_passed, pcc_message = assert_with_pcc(torch_output, ttnn_output, 0.97)
    logger.info(pcc_message)
