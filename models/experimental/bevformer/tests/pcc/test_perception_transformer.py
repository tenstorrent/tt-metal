# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.bevformer.tests.backbone_common import assert_pcc
from models.experimental.bevformer.tests.encoder_common import BEV_SHAPES, NUM_LAYERS, SPATIAL_SHAPES, EMBED_DIMS
from models.experimental.bevformer.tests.perception_common import (
    build_reference_transformer,
    fpn_rows,
    frame_metas,
    random_fpn_levels,
)
from models.experimental.bevformer.tt.model_preprocessing import create_perception_transformer_parameters
from models.experimental.bevformer.tt.tt_perception_transformer import TTPerceptionTransformer

CASES = [
    # (name, bev_shape, num_layers, batch_size, yaw_step_deg)
    ("base", BEV_SHAPES["base"], NUM_LAYERS, 1, 0.0),
    # bs=2: each sample has its own CAN bus, rotation and shift, and the second sample's rig is
    # turned, so a batch mix-up in any of them shows.
    ("tiny-bs2", BEV_SHAPES["tiny"], NUM_LAYERS, 2, 40.0),
]


def _to_device(tensor, device):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@torch.no_grad()
@pytest.mark.parametrize(
    "name, bev_shape, num_layers, batch_size, yaw_step_deg", CASES, ids=[case[0] for case in CASES]
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_perception_transformer(device, reset_seeds, name, bev_shape, num_layers, batch_size, yaw_step_deg):
    """Two consecutive frames through the encoder's glue: the CAN-bus MLP, the camera and level
    embeddings, and on the second frame the previous BEV's rotation and the ego shift, with each
    side's own first-frame BEV as its previous BEV."""
    bev_h, bev_w = bev_shape
    generator = torch.Generator().manual_seed(0)
    torch_model = build_reference_transformer(num_layers)
    levels = random_fpn_levels(batch_size, generator)
    bev_queries = torch.randn(bev_h * bev_w, EMBED_DIMS, generator=generator)
    bev_pos = torch.randn(bev_h * bev_w, batch_size, EMBED_DIMS, generator=generator)
    frames = frame_metas(batch_size, 2, generator, yaw_step_deg)

    tt_model = TTPerceptionTransformer(
        create_perception_transformer_parameters(torch_model, device),
        device,
        bev_h=bev_h,
        bev_w=bev_w,
        spatial_shapes=SPATIAL_SHAPES,
    )
    tt_levels = [_to_device(fpn_rows(level), device) for level in levels]
    tt_queries = _to_device(bev_queries[None], device)
    tt_pos = _to_device(bev_pos.permute(1, 0, 2), device)

    torch_prev = tt_prev = frame = None
    for i, metas in enumerate(frames):
        torch_output = torch_model.get_bev_features(
            levels, bev_queries, bev_h, bev_w, bev_pos, metas, prev_bev=torch_prev
        )
        frame = tt_model.prepare_frame(metas, frame)
        tt_prev = tt_model(tt_levels, tt_queries, tt_pos, frame, prev_bev=tt_prev)
        tt_output = ttnn.to_torch(tt_prev).float().reshape(torch_output.shape)
        # comp_pcc zeroes NaN and Inf before correlating, so they must be ruled out here.
        assert torch.isfinite(tt_output).all(), f"non-finite values in frame {i}'s BEV"
        assert_pcc(torch_output, tt_output, 0.997)
        torch_prev = torch_output
