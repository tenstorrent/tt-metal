# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import json
import numpy as np
import os
from loguru import logger
from models.demos.ufld_v2.reference.ufld_v2_model import TuSimple34
from models.demos.ufld_v2.demo import model_config as cfg
from models.demos.ufld_v2.demo.demo_utils import (
    run_test_tusimple,
    LaneEval,
)
from models.demos.ufld_v2.ttnn.ttnn_ufld_v2 import TtnnUFLDv2
from tests.ttnn.integration_tests.ufld_v2.test_ttnn_ufld_v2 import custom_preprocessor_whole_model
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args


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
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_UFLD_v2_demo(batch_size, input_channels, height, width, device, use_pretrained_weight):
    torch_input_tensor = torch.randn((batch_size, input_channels, height, width))
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

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=custom_preprocessor_whole_model,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=reference_model, run_model=lambda model: reference_model(torch_input_tensor), device=device
    )
    ttnn_model = TtnnUFLDv2(conv_args=parameters.conv_args, conv_pth=parameters, device=device)
    run_test_tusimple(
        ttnn_model,
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
