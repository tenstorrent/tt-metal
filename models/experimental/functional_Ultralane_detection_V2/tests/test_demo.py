import torch
import pytest
from models.experimental.functional_Ultralane_detection_V2.reference.tu_simple_model import Tu_Simple
from models.experimental.functional_Ultralane_detection_V2.Ultra_Fast_Lane_Detection_v2_forked.model.model_culane import (
    parsingNet,
)
from models.experimental.functional_Ultralane_detection_V2.Ultra_Fast_Lane_Detection_v2_forked.configs import (
    tusimple_res34 as cfg,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_Ultralane_detection_V2.reference.tu_simple_model_utils import (
    run_test_tusimple,
    LaneEval,
)
import json
import numpy as np
import os


@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 3, 320, 800),
    ],
)
def test_tu_simple_res34_inference(batch_size, input_channels, height, width):
    reference_model = Tu_Simple(input_height=height, input_width=width)
    repo_model = parsingNet(
        pretrained=True,
        backbone=cfg.backbone,
        num_grid_row=cfg.num_cell_row,
        num_cls_row=cfg.num_row,
        num_grid_col=cfg.num_cell_col,
        num_cls_col=cfg.num_col,
        num_lane_on_row=cfg.num_lanes,
        num_lane_on_col=cfg.num_lanes,
        use_aux=cfg.use_aux,
        input_height=cfg.train_height,
        input_width=cfg.train_width,
        fc_norm=cfg.fc_norm,
    )
    state_dict = torch.load("models/experimental/functional_Ultralane_detection_V2/reference/tusimple_res34.pth")
    reference_model.load_state_dict(state_dict["model"])
    repo_model.load_state_dict(state_dict["model"])
    cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
    cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    run_test_tusimple(
        reference_model,
        cfg.data_root,
        cfg.test_work_dir,
        "tusimple_reference",
        False,
        cfg.crop_ratio,
        cfg.train_width,
        cfg.train_height,
        row_anchor=cfg.row_anchor,
        col_anchor=cfg.col_anchor,
    )
    run_test_tusimple(
        repo_model,
        cfg.data_root,
        cfg.test_work_dir,
        "tusimple_repo",
        False,
        cfg.crop_ratio,
        cfg.train_width,
        cfg.train_height,
        row_anchor=cfg.row_anchor,
        col_anchor=cfg.col_anchor,
    )
    gt_file_path = (
        "/home/ubuntu/venkatesh/tt-metal/models/experimental/functional_Ultralane_detection_V2/demo/GT_test_labels.json"
    )

    res = LaneEval.bench_one_submit(os.path.join(cfg.test_work_dir, "tusimple_reference" + ".txt"), gt_file_path)
    res = json.loads(res)
    for r in res:
        if r["name"] == "F1":
            print("F1 score for reference is", r["value"])

    res1 = LaneEval.bench_one_submit(os.path.join(cfg.test_work_dir, "tusimple_repo" + ".txt"), gt_file_path)
    res1 = json.loads(res1)
    for r in res1:
        if r["name"] == "F1":
            print("F1 score for repo is", r["value"])
