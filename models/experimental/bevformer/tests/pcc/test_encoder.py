# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.bevformer.model_config import ENCODER_NUM_LAYERS, GRID_DTYPE, SPATIAL_SHAPES
from models.experimental.bevformer.tests.common import (
    BEV_SHAPES,
    assert_pcc,
    build_reference_encoder,
    camera_rows,
    ego_shift,
    random_encoder_inputs,
)
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_encoder_parameters
from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerEncoder

CASES = [
    # (name, bev_shape, num_layers, batch_size, yaw_step_deg)
    ("base", BEV_SHAPES["base"], ENCODER_NUM_LAYERS, 1, 0.0),
    ("base-1-layer", BEV_SHAPES["base"], 1, 1, 0.0),
    ("tiny", BEV_SHAPES["tiny"], ENCODER_NUM_LAYERS, 1, 0.0),
    # bs=2 with a per-sample shift and the second sample's rig turned, so a batch mix-up in the
    # stacked previous BEV, the reference points, the shift or the rebatch shows.
    ("tiny-bs2", BEV_SHAPES["tiny"], ENCODER_NUM_LAYERS, 2, 40.0),
    # Non-square, so a swapped (h, w) in the BEV grid, its reference points or the shift shows.
    ("50x100", (50, 100), ENCODER_NUM_LAYERS, 1, 0.0),
]


def _to_device(tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device)


def _batch_first(tensor):
    return tensor.permute(1, 0, 2).contiguous()


@torch.no_grad()
@pytest.mark.parametrize(
    "name, bev_shape, num_layers, batch_size, yaw_step_deg", CASES, ids=[case[0] for case in CASES]
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_encoder(device, reset_seeds, name, bev_shape, num_layers, batch_size, yaw_step_deg):
    """Two consecutive frames: the first without a previous BEV, the second with each side's
    own first-frame output as its previous BEV and an ego shift, so the temporal path carries
    the device's own error forward as it does in the detector."""
    bev_h, bev_w = bev_shape
    torch_model = build_reference_encoder(num_layers)
    tt_model = TTBEVFormerEncoder(
        create_bevformer_encoder_parameters(torch_model, device),
        device,
        bev_h=bev_h,
        bev_w=bev_w,
        spatial_shapes=SPATIAL_SHAPES,
    )
    inputs = random_encoder_inputs(bev_shape, batch_size, yaw_step_deg=yaw_step_deg)
    spatial_shapes = torch.tensor(SPATIAL_SHAPES)
    shift = ego_shift(batch_size)

    def reference(prev_bev, frame_shift):
        return torch_model(
            inputs["bev_query"],
            inputs["value"],
            bev_h,
            bev_w,
            inputs["bev_pos"],
            spatial_shapes,
            inputs["img_metas"],
            prev_bev=prev_bev,
            shift=frame_shift,
        )

    torch_first = reference(None, None)
    torch_second = reference(_batch_first(torch_first), shift)

    plan = tt_model.prepare_frame(inputs["img_metas"])
    tt_query = _to_device(_batch_first(inputs["bev_query"]), device)
    tt_pos = _to_device(_batch_first(inputs["bev_pos"]), device)
    tt_value = _to_device(camera_rows(inputs["value"]), device)
    tt_shift = _to_device(shift.view(batch_size, 1, 1, 2), device, GRID_DTYPE, ttnn.ROW_MAJOR_LAYOUT)
    tt_first = tt_model(tt_query, tt_value, tt_pos, plan)
    tt_second = tt_model(tt_query, tt_value, tt_pos, plan, prev_bev=tt_first, shift=tt_shift)

    for torch_output, tt_output in ((torch_first, tt_first), (torch_second, tt_second)):
        tt_output = ttnn.to_torch(tt_output).float().reshape(torch_output.shape)
        # comp_pcc zeroes NaN and Inf before correlating, so they must be ruled out here.
        assert torch.isfinite(tt_output).all(), "non-finite values in the encoder output"
        assert_pcc(torch_output, tt_output, 0.997)
