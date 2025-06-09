# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import numpy as np
import pytest
import torch
from loguru import logger

from models.demos.ufld_v2.demo import model_config as cfg
from models.demos.ufld_v2.demo.demo_utils import LaneEval, run_test_tusimple
from models.demos.ufld_v2.reference.ufld_v2_model import TuSimple34
from models.demos.ufld_v2.tests.ufld_v2_e2e_performant import UFLDPerformantRunner


@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 3, 320, 800),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        # False,
        True
    ],
    ids=[
        # "pretrained_weight_false",
        "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
def test_ufld_v2_demo(
    batch_size, input_channels, height, width, device, use_pretrained_weight, use_program_cache, reset_seeds
):
    reference_model = TuSimple34(input_height=height, input_width=width)
    if use_pretrained_weight:
        logger.info(f"Demo Inference using Pre-trained Weights")
        weights_path = "models/demos/ufld_v2/tusimple_res34.pth"
        if not os.path.exists(weights_path):
            os.system("bash models/demos/ufld_v2/weights_download.sh")
        state_dict = torch.load(weights_path)
        new_state_dict = {}
        for key, value in state_dict["model"].items():
            new_key = key.replace("model.", "res_model.")
            new_state_dict[new_key] = value
        reference_model.load_state_dict(new_state_dict)
    else:
        logger.info(f"Demo Inference using Random Weights")
    cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
    cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    run_test_tusimple(
        reference_model,
        cfg.data_root,
        cfg.data_root,
        "reference_model_results",
        False,
        cfg.crop_ratio,
        cfg.train_width,
        cfg.train_height,
        batch_size=batch_size,
        row_anchor=cfg.row_anchor,
        col_anchor=cfg.col_anchor,
        device=None,
    )
    run_test_tusimple(
        UFLDPerformantRunner,
        cfg.data_root,
        cfg.data_root,
        "ttnn_model_results",
        False,
        cfg.crop_ratio,
        cfg.train_width,
        cfg.train_height,
        batch_size=batch_size,
        row_anchor=cfg.row_anchor,
        col_anchor=cfg.col_anchor,
        device=device,
    )

    gt_file_path = os.path.join(cfg.data_root, "ground_truth_labels" + ".json")
    res = LaneEval.bench_one_submit(os.path.join(cfg.data_root, "reference_model_results" + ".txt"), gt_file_path)
    res = json.loads(res)
    for r in res:
        if r["name"] == "F1":
            logger.info(f"F1 Score for Reference Model is {r['value']}")

    res1 = LaneEval.bench_one_submit(os.path.join(cfg.data_root, "ttnn_model_results" + ".txt"), gt_file_path)
    res1 = json.loads(res1)
    for r in res1:
        if r["name"] == "F1":
            logger.info(f"F1 Score for ttnn Model is {r['value']}")
