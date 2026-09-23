# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""BEVFormer's MSDA module against the PyTorch reference.

`TTMSDeformableAttention` computes its core attention with a single device op,
`ttnn.experimental.fused_msda_from_offsets`, which generates the sampling
locations inside the reader. This file is what says that swap did not change the
model's numerics: same weights, same inputs, compared against the reference
module rather than against another TT implementation.

Complements `test_ms_deformable_attention.py`, which checks the same module with
the repo's usual tolerance bundle; here the gate is a tighter PCC across several
seeds, because a location-generation bug shows up as a seed-dependent drift
rather than a uniform offset.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.bevformer.config import DeformableAttentionConfig
from models.experimental.bevformer.config.encoder_config import get_preset_config
from models.experimental.bevformer.reference.ms_deformable_attention import MSDeformableAttention
from models.experimental.bevformer.tests.test_utils import check_with_pcc
from models.experimental.bevformer.tt.model_preprocessing import create_ms_deformable_attention_parameters
from models.experimental.bevformer.tt.tt_ms_deformable_attention import TTMSDeformableAttention

# Module-scoped device: opens once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device({"l1_small_size": 10 * 1024})


def _build(device, config_name, batch_size, num_queries, seed):
    """Shared setup: reference module, preprocessed TT params, and the inputs."""
    torch.manual_seed(seed)

    preset_config = get_preset_config(config_name)
    if preset_config is None:
        pytest.fail(f"Configuration '{config_name}' not found")

    model_config = preset_config.model_config
    num_levels = model_config.num_levels
    spatial_shapes = torch.tensor(preset_config.dataset_config.spatial_shapes[:num_levels], dtype=torch.long)
    num_keys = int(spatial_shapes.prod(dim=1).sum().item())

    config = DeformableAttentionConfig(
        embed_dims=model_config.embed_dims,
        num_heads=model_config.num_heads,
        num_levels=num_levels,
        num_points=model_config.num_points,
    )

    query = torch.randn(batch_size, num_queries, config.embed_dims, dtype=torch.float32)
    value = torch.randn(batch_size, num_keys, config.embed_dims, dtype=torch.float32)
    reference_points = torch.rand(batch_size, num_queries, num_levels, 2, dtype=torch.float32)

    ref_model = MSDeformableAttention(config)
    ref_model.eval()

    tt_parameters = create_ms_deformable_attention_parameters(
        torch_model=ref_model, device=device, config=config, dtype=ttnn.float32
    )

    torch_inputs = (query, value, reference_points, spatial_shapes)
    return config, ref_model, tt_parameters, torch_inputs


def _run_tt(device, config, tt_parameters, torch_inputs):
    query, value, reference_points, spatial_shapes = torch_inputs

    tt_query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_value = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_reference_points = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_model = TTMSDeformableAttention(
        config=config,
        device=device,
        params=tt_parameters,
        spatial_shapes=spatial_shapes,
    )
    out = tt_model(query=tt_query, value=tt_value, reference_points=tt_reference_points)
    return ttnn.to_torch(out).to(torch.float32)


@pytest.mark.parametrize(
    "config_name, batch_size, num_queries",
    [
        ("nuscenes_tiny", 1, 900),
        ("nuscenes_base", 1, 2500),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_msda_matches_reference(device, config_name, batch_size, num_queries, seed):
    config, ref_model, tt_parameters, torch_inputs = _build(device, config_name, batch_size, num_queries, seed)
    query, value, reference_points, spatial_shapes = torch_inputs

    expected = ref_model(query, value, reference_points=reference_points, spatial_shapes=spatial_shapes)
    got = _run_tt(device, config, tt_parameters, torch_inputs)

    assert got.shape == expected.shape, f"{tuple(got.shape)} != {tuple(expected.shape)}"
    logger.info(f"{config_name} q={num_queries} seed={seed}: output {tuple(got.shape)}")

    passed, message = check_with_pcc(expected, got, pcc=0.999)
    assert passed, f"msda_output[{config_name}, seed={seed}]: {message}"


def test_msda_output_is_deterministic(device):
    """Two identical calls must agree exactly.

    The op distributes (batch, head, query-block) work units across cores and
    accumulates each unit's `levels * points` reduction locally, so the summation
    order is fixed and the result should not vary run to run. A mismatch here
    would point at uninitialized L1 being read -- the reader explicitly zeroes
    the input rows whose bilinear corner it did not gather, so a nonzero
    difference here points straight at that zeroing.
    """
    config, _, tt_parameters, torch_inputs = _build(device, "nuscenes_tiny", 1, 900, 0)

    first = _run_tt(device, config, tt_parameters, torch_inputs)
    second = _run_tt(device, config, tt_parameters, torch_inputs)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
