# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.bevformer.model_config import GRID_DTYPE
from models.experimental.bevformer.reference.point_sampling_3d_2d import bev_reference_points
from models.experimental.bevformer.tests.common import (
    BEV_SHAPES,
    assert_pcc,
    build_reference_encoder,
    ego_shift,
    random_bev,
    random_encoder_inputs,
)
from models.experimental.bevformer.tt.model_preprocessing import create_temporal_self_attention_parameters
from models.experimental.bevformer.tt.tt_ms_deformable_attention import fp32_grid_sample_config
from models.experimental.bevformer.tt.tt_temporal_self_attention import TTTemporalSelfAttention, tsa_grid_bias

CASES = [
    # (name, bev_shape, batch_size, with_previous_bev)
    ("tiny-first-frame", BEV_SHAPES["tiny"], 1, False),
    ("tiny-previous-bev", BEV_SHAPES["tiny"], 1, True),
    ("base-previous-bev", BEV_SHAPES["base"], 1, True),
    # bs=2 with a per-sample shift, so a batch mix-up in the stacked (previous, current) maps or
    # the shifted reference points shows.
    ("tiny-bs2-previous-bev", BEV_SHAPES["tiny"], 2, True),
    # Non-square, so a swapped (h, w) in the grid scale or the reference points shows.
    ("50x100-previous-bev", (50, 100), 1, True),
]


def _to_device(tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device)


@torch.no_grad()
@pytest.mark.parametrize("name, bev_shape, batch_size, with_previous_bev", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_temporal_self_attention(device, reset_seeds, name, bev_shape, batch_size, with_previous_bev):
    """The first layer's self-attention, on the encoder's inputs: the BEV queries and, for a
    later frame, the previous BEV stacked with them and the shifted reference points."""
    bev_h, bev_w = bev_shape
    torch_model = build_reference_encoder(num_layers=1).layers[0].attentions[0]
    inputs = random_encoder_inputs(bev_shape, batch_size)
    query, pos = inputs["bev_query"].permute(1, 0, 2), inputs["bev_pos"].permute(1, 0, 2)
    num_query = bev_h * bev_w

    ref_2d = bev_reference_points(bev_h, bev_w, batch_size)
    if with_previous_bev:
        previous = random_bev(bev_shape, batch_size, torch.Generator().manual_seed(1)).permute(1, 0, 2)
        value = torch.stack([previous, query], 1).reshape(batch_size * 2, num_query, -1)
        shifted = ref_2d + ego_shift(batch_size)[:, None, None, :]
        reference_points = torch.stack([shifted, ref_2d], 1).reshape(batch_size * 2, num_query, 1, 2)
    else:
        value = None
        reference_points = torch.stack([ref_2d, ref_2d], 1).reshape(batch_size * 2, num_query, 1, 2)

    torch_output = torch_model(
        query,
        value=value,
        query_pos=pos,
        reference_points=reference_points,
        spatial_shapes=torch.tensor([[bev_h, bev_w]]),
    )

    tt_model = TTTemporalSelfAttention(
        create_temporal_self_attention_parameters(torch_model, device),
        device,
        bev_shape=bev_shape,
        grid_sample_compute_config=fp32_grid_sample_config(device),
    )
    grid_bias = tsa_grid_bias(
        _to_device(reference_points, device, GRID_DTYPE, ttnn.ROW_MAJOR_LAYOUT),
        tt_model.num_heads,
        tt_model.num_points,
        GRID_DTYPE,
    )
    tt_output = tt_model(
        _to_device(query, device),
        None if value is None else _to_device(value, device),
        _to_device(pos, device),
        grid_bias,
    )
    tt_output = ttnn.to_torch(tt_output).float().reshape(torch_output.shape)
    # comp_pcc zeroes NaN and Inf before correlating, so they must be ruled out here.
    assert torch.isfinite(tt_output).all(), "non-finite values in the self-attention output"
    assert_pcc(torch_output, tt_output, 0.999)
    # The attended part alone: the residual would carry the PCC on its own.
    assert_pcc(torch_output - query, tt_output - query, 0.999)
