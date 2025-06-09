# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.maptr.reference import temporal_self_attention
from models.experimental.maptr.ttnn import ttnn_temporal_self_attention
from models.experimental.maptr.ttnn.model_preprocessing import (
    create_maptr_model_parameters_tsa,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_temporal_self_attention(
    device,
    reset_seeds,
    use_program_cache,
):
    weights_path = "models/experimental/maptr/maptr_weights_sd.pth"
    torch_model = temporal_self_attention.TemporalSelfAttention(embed_dims=256, num_levels=1)

    torch_dict = torch.load(weights_path)

    state_dict = {
        k: v
        for k, v in torch_dict.items()
        if (k.startswith("pts_bbox_head.transformer.encoder.layers.0.attentions.0."))
    }
    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    query = torch.randn(1, 3200, 256)
    query_pos = torch.randn(1, 3200, 256)
    reference_points = torch.randn(2, 3200, 1, 2)
    spatial_shapes = torch.tensor([[80, 40]])
    level_start_index = torch.tensor([0])

    torch_output = torch_model(
        query,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    parameter = create_maptr_model_parameters_tsa(
        torch_model, [query, query_pos, reference_points, spatial_shapes, level_start_index]
    )

    ttnn_model = ttnn_temporal_self_attention.TemporalSelfAttentionTT(embed_dims=256, num_levels=1, device=device)

    query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    query_pos = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)
    reference_points = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16)
    spatial_shapes = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    level_start_index = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    ttnn_output = ttnn_model(
        query,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        parameter=parameter.temporal_self_attention,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.999)
    logger.info(pcc_message)
