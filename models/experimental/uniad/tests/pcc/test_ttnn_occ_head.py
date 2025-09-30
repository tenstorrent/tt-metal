# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.uniad.reference.occ_head import OccHead
from models.experimental.uniad.tt.ttnn_occ_head import TtOccHead
from models.experimental.uniad.common import load_torch_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_occhead(reset_seeds, device, model_location_generator):
    occflow_grid_conf = {
        "xbound": [-50.0, 50.0, 0.5],
        "ybound": [-50.0, 50.0, 0.5],
        "zbound": [-10.0, 10.0, 20.0],
    }
    occ_head = OccHead(
        grid_conf=occflow_grid_conf,
        ignore_index=255,
        bev_proj_dim=256,
        bev_proj_nlayers=4,
        # Transformer
        attn_mask_thresh=0.3,
    )

    occ_head = load_torch_model(
        torch_model=occ_head, layer="occ_head", model_location_generator=model_location_generator
    )

    bev_feat = torch.randn(2500, 1, 256)
    no_query = True
    gt_segmentation = []
    gt_segmentation.append(torch.randn(1, 7, 200, 200))
    gt_instance = []
    gt_instance.append(torch.randn(1, 7, 200, 200))
    gt_img_is_valid = []
    gt_img_is_valid.append(torch.randn(1, 9))

    all_traj_scores = torch.randn(3, 1, 1, 6)
    all_traj_preds = torch.randn(3, 1, 1, 6, 12, 5)
    valid_traj_masks = torch.randn(1, 1)
    traj_query = torch.randn(3, 1, 0, 6, 256)
    track_query = torch.randn(1, 0, 256)
    track_query_pos = torch.randn(1, 0, 256)
    track_scores = torch.randn(1, 0)
    sdc_traj_query = torch.randn(3, 1, 6, 256)
    sdc_track_query = torch.randn(1, 256)
    sdc_track_query_pos = torch.randn(1, 256)
    bev_pos = torch.randn(1, 256, 50, 50)

    outs_dict = {}
    outs_dict["all_traj_scores"] = all_traj_scores
    outs_dict["all_traj_preds"] = all_traj_preds
    outs_dict["valid_traj_masks"] = valid_traj_masks
    outs_dict["traj_query"] = traj_query
    outs_dict["track_query"] = track_query
    outs_dict["track_query_pos"] = track_query_pos
    outs_dict["track_scores"] = track_scores
    outs_dict["sdc_traj_query"] = sdc_traj_query
    outs_dict["sdc_track_query"] = sdc_track_query
    outs_dict["sdc_track_query_pos"] = sdc_track_query_pos
    outs_dict["bev_pos"] = bev_pos

    torch_out = occ_head(
        bev_feat=bev_feat,
        no_query=no_query,
        gt_segmentation=gt_segmentation,
        gt_instance=gt_instance,
        gt_img_is_valid=gt_img_is_valid,
        outs_dict=outs_dict,
    )

    ttnn_model = TtOccHead(device=device)

    tt_bev_feat = ttnn.from_torch(bev_feat, device=device, dtype=ttnn.bfloat16)
    tt_gt_segmentation = []
    tt_gt_segmentation.append(ttnn.from_torch(gt_segmentation[0], device=device, dtype=ttnn.bfloat16))
    tt_gt_instance = []
    tt_gt_instance.append(ttnn.from_torch(gt_instance[0], device=device, dtype=ttnn.bfloat16))
    tt_gt_img_is_valid = []
    tt_gt_img_is_valid.append(ttnn.from_torch(gt_img_is_valid[0], device=device, dtype=ttnn.bfloat16))

    tt_output = ttnn_model(
        bev_feat=tt_bev_feat,
        no_query=no_query,
        gt_segmentation=tt_gt_segmentation,
        gt_instance=tt_gt_instance,
        gt_img_is_valid=tt_gt_img_is_valid,
        outs_dict=outs_dict,
    )

    tt_output["seg_gt"] = ttnn.to_torch(tt_output["seg_gt"])
    tt_output["ins_seg_gt"] = ttnn.to_torch(tt_output["ins_seg_gt"])
    tt_output["seg_out"] = ttnn.to_torch(tt_output["seg_out"])
    tt_output["ins_seg_out"] = ttnn.to_torch(tt_output["ins_seg_out"])

    _, pcc = assert_with_pcc(torch_out["seg_gt"], tt_output["seg_gt"], 0.99)
    logger.info(f"pcc: {pcc}")
    _, pcc = assert_with_pcc(torch_out["ins_seg_gt"], tt_output["ins_seg_gt"], 0.99)
    logger.info(f"pcc: {pcc}")
    _, pcc = assert_with_pcc(torch_out["seg_out"], tt_output["seg_out"], 0.99)
    logger.info(f"pcc: {pcc}")
    _, pcc = assert_with_pcc(torch_out["ins_seg_out"], tt_output["ins_seg_out"], 0.99)
    logger.info(f"pcc: {pcc}")
