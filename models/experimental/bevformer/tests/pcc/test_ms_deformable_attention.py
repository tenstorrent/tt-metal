# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from models.experimental.bevformer.reference.ms_deformable_attention import MSDeformableAttention
from models.experimental.bevformer.tt.tt_ms_deformable_attention import TTMSDeformableAttention

from models.experimental.bevformer.model_config import DeformableAttentionConfig

from models.experimental.bevformer.tests.common import PRESETS, assert_finite, assert_pcc

from models.experimental.bevformer.tt.model_preprocessing import (
    create_ms_deformable_attention_parameters,
)

# Module-scoped device: opens once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device({"l1_small_size": 10 * 1024})


@pytest.mark.parametrize(
    "config_name, batch_size, num_queries, expected_pcc",
    [
        ("nuscenes_tiny", 1, 900, 0.999),
        ("nuscenes_base", 1, 10000, 0.999),
        ("carla_base", 1, 12000, 0.999),
    ],
)
# The module hands the reference points to the fused op unconverted; the encoder and decoder
# pass float32 ones.
@pytest.mark.parametrize("reference_points_dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16-refs", "fp32-refs"])
@pytest.mark.parametrize("seed", [0])
def test_ms_deformable_attention_forward(
    device,
    reference_points_dtype,
    config_name,
    batch_size,
    num_queries,
    expected_pcc,
    seed,
):
    """The module against the reference on each preset, with bfloat16 and float32 reference points."""
    torch.manual_seed(seed)

    preset = PRESETS[config_name]

    embed_dims = preset.embed_dims
    num_heads = preset.num_heads
    num_levels = preset.num_levels
    num_points = preset.num_points

    spatial_shapes = torch.tensor(preset.spatial_shapes, dtype=torch.long)
    total_key_length = [h * w for h, w in spatial_shapes.tolist()]

    query = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)
    value = torch.randn(batch_size, sum(total_key_length), embed_dims, dtype=torch.float32)

    reference_points = torch.rand(batch_size, num_queries, num_levels, 2, dtype=torch.float32)

    tt_query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_value = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_reference_points = ttnn.from_torch(
        reference_points, device=device, dtype=reference_points_dtype, layout=ttnn.TILE_LAYOUT
    )

    config = DeformableAttentionConfig(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
    )

    ref_model = MSDeformableAttention(config)
    ref_model.eval()

    tt_parameters = create_ms_deformable_attention_parameters(ref_model, device=device, dtype=ttnn.float32)

    tt_model = TTMSDeformableAttention(
        config=config,
        device=device,
        params=tt_parameters,
        spatial_shapes=spatial_shapes,
    )

    ref_model_output = ref_model(
        query,
        value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
    )

    tt_model_output = tt_model(
        query=tt_query,
        value=tt_value,
        reference_points=tt_reference_points,
    )
    tt_model_output = ttnn.to_torch(tt_model_output)

    assert (
        tt_model_output.shape == ref_model_output.shape
    ), f"{tuple(tt_model_output.shape)} != {tuple(ref_model_output.shape)}"
    assert_finite("the MSDA output", tt_model_output)
    assert_pcc(ref_model_output, tt_model_output, expected_pcc)
