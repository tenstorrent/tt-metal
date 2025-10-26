# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess

import numpy as np
import pytest
from loguru import logger

from models.demos.ufld_v2.demo import model_config as cfg
from models.demos.ufld_v2.demo.demo_utils import LaneEval, run_test_tusimple
from models.demos.ufld_v2.reference.ufld_v2_model import TuSimple34
from models.demos.ufld_v2.runner.performant_runner import UFLDPerformantRunner
from models.demos.ufld_v2.runner.performant_runner_infra import load_torch_model


def run_ufld_v2_dataset_inference(
    batch_size_per_device,
    input_channels,
    height,
    width,
    num_of_images,
    is_overlay,
    device,
    use_pretrained_weight,
    reset_seeds,
    exp_name_1,
    exp_name_2,
    model_location_generator=None,
    tolerance=0.2,
):
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    reference_model = TuSimple34(input_height=height, input_width=width)
    if use_pretrained_weight:
        logger.info(f"Demo Inference using Pre-trained Weights")
        reference_model = load_torch_model(model_location_generator, use_pretrained_weight)
    else:
        logger.info(f"Demo Inference using Random Weights")
    cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
    cfg.col_anchor = np.linspace(0, 1, cfg.num_col)

    dataset_path = "models/demos/ufld_v2/demo/image_data"
    if not os.path.exists(dataset_path):
        subprocess.run(["python3", "models/demos/ufld_v2/demo/data_download.py"], check=True)
    cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
    cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    run_test_tusimple(
        reference_model,
        cfg.data_root,
        cfg.data_root,
        exp_name_1,
        False,
        cfg.crop_ratio,
        cfg.train_width,
        cfg.train_height,
        batch_size=batch_size,
        row_anchor=cfg.row_anchor,
        col_anchor=cfg.col_anchor,
        device=None,
        n_images=num_of_images,
        is_overlay=is_overlay,
        is_eval=True,
        model_location_generator=model_location_generator,
    )
    run_test_tusimple(
        UFLDPerformantRunner,
        cfg.data_root,
        cfg.data_root,
        exp_name_2,
        False,
        cfg.crop_ratio,
        cfg.train_width,
        cfg.train_height,
        batch_size=batch_size,
        row_anchor=cfg.row_anchor,
        col_anchor=cfg.col_anchor,
        device=device,
        n_images=num_of_images,
        is_overlay=is_overlay,
        is_eval=True,
        model_location_generator=model_location_generator,
    )
    gt_file_path = "models/demos/ufld_v2/demo/image_data/test_label_till_nimages.json"
    os.makedirs(os.path.dirname(gt_file_path), exist_ok=True)
    input_file = "models/demos/ufld_v2/demo/image_data/test_label.json"
    with open(input_file, "r") as infile, open(gt_file_path, "w") as outfile:
        for i, line in enumerate(infile):
            if i >= num_of_images:
                break
            outfile.write(line)

    res = LaneEval.bench_one_submit(os.path.join(cfg.data_root, exp_name_1 + ".txt"), gt_file_path)
    res = json.loads(res)
    for r in res:
        if r["name"] == "F1":
            reference_f1 = r["value"]
            logger.info(f"F1 Score for Reference Model is {r['value']}")

    res1 = LaneEval.bench_one_submit(os.path.join(cfg.data_root, exp_name_2 + ".txt"), gt_file_path)
    res1 = json.loads(res1)
    for r in res1:
        if r["name"] == "F1":
            ttnn_f1 = r["value"]
            logger.info(f"F1 Score for ttnn Model is {r['value']}")

    assert (
        abs(reference_f1 - ttnn_f1) <= tolerance
    ), f"F1 score difference {abs(reference_f1 - ttnn_f1):.4f} exceeds tolerance of {tolerance}"


@pytest.mark.parametrize(
    "batch_size,input_channels,height,width,num_of_images,is_overlay,exp_name_1,exp_name_2",
    [
        (1, 3, 320, 800, 100, False, "reference_model_results_dataset", "ttnn_model_results_dataset"),
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
def test_ufld_v2_dataset_inference(
    batch_size,
    input_channels,
    height,
    width,
    num_of_images,
    is_overlay,
    use_pretrained_weight,
    exp_name_1,
    exp_name_2,
    device,
    reset_seeds,
    model_location_generator,
):
    run_ufld_v2_dataset_inference(
        batch_size,
        input_channels,
        height,
        width,
        num_of_images,
        is_overlay,
        device,
        use_pretrained_weight,
        reset_seeds,
        exp_name_1,
        exp_name_2,
        model_location_generator,
    )


@pytest.mark.parametrize(
    "batch_size_per_device,input_channels,height,width,num_of_images,is_overlay,exp_name_1,exp_name_2",
    [
        (1, 3, 320, 800, 100, False, "reference_model_results_dataset_dp", "ttnn_model_results_dataset_dp"),
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
def test_ufld_v2_dataset_inference_dp(
    batch_size_per_device,
    input_channels,
    height,
    width,
    num_of_images,
    is_overlay,
    use_pretrained_weight,
    exp_name_1,
    exp_name_2,
    mesh_device,
    reset_seeds,
    model_location_generator,
):
    run_ufld_v2_dataset_inference(
        batch_size_per_device,
        input_channels,
        height,
        width,
        num_of_images,
        is_overlay,
        mesh_device,
        use_pretrained_weight,
        reset_seeds,
        exp_name_1,
        exp_name_2,
        model_location_generator,
    )
