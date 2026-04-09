# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov8l.common import load_torch_model, yolov8l_l1_small_size_for_res
from models.demos.yolov8l.runner.performant_runner_infra import yolov8l_dram_sharded_input_from_torch
from models.demos.yolov8l.tt.tt_yolov8l_utils import custom_preprocessor
from models.demos.yolov8l.tt.ttnn_yolov8l import TtYolov8lModel
from tests.ttnn.utils_for_testing import assert_with_pcc


def _run_yolov8l_pcc(device, model_location_generator, input_res, use_pretrained_weights=True):
    input_tensor = torch.rand((1, 3, input_res, input_res))
    inp_h, inp_w = input_tensor.shape[2], input_tensor.shape[3]
    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)
        state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict, inp_h=inp_h, inp_w=inp_w)
    ttnn_model = TtYolov8lModel(device=device, parameters=parameters, res=(inp_h, inp_w))

    torch_input_ch16 = F.pad(input_tensor, (0, 0, 0, 0, 0, 13))
    ttnn_input = yolov8l_dram_sharded_input_from_torch(device, torch_input_ch16)

    with torch.inference_mode():
        ttnn_model_output = ttnn_model(ttnn_input)[0]
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    with torch.inference_mode():
        torch_model_output = torch_model(input_tensor)[0]

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize(
    "use_pretrained_weights",
    [True],
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": yolov8l_l1_small_size_for_res(1280, 1280)}],
    indirect=True,
    ids=["l1_1280_for_all_res"],
)
@pytest.mark.parametrize("input_res", [640, 1280], ids=["640", "1280"])
def test_yolov8l(device, use_pretrained_weights, model_location_generator, input_res):
    _run_yolov8l_pcc(device, model_location_generator, input_res, use_pretrained_weights=use_pretrained_weights)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": yolov8l_l1_small_size_for_res(640, 640)}],
    indirect=True,
    ids=["l1_640"],
)
def test_yolov8l_640(device, model_location_generator):
    _run_yolov8l_pcc(device, model_location_generator, 640)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": yolov8l_l1_small_size_for_res(1280, 1280)}],
    indirect=True,
    ids=["l1_1280"],
)
def test_yolov8l_1280(device, model_location_generator):
    _run_yolov8l_pcc(device, model_location_generator, 1280)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 8)],
    indirect=True,
    ids=["t3k_1x8"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": yolov8l_l1_small_size_for_res(1280, 1280)}],
    indirect=True,
    ids=["l1_1280_for_all_res"],
)
@pytest.mark.parametrize("input_hw", [640, 1280], ids=["640", "1280"])
def test_yolov8l_dp_batch8(mesh_device, model_location_generator, input_hw):
    batch_size = mesh_device.get_num_devices()
    input_tensor = torch.rand((batch_size, 3, input_hw, input_hw))
    inp_h, inp_w = input_tensor.shape[2], input_tensor.shape[3]
    torch_model = load_torch_model(model_location_generator)
    state_dict = torch_model.state_dict()
    _, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    parameters = custom_preprocessor(mesh_device, state_dict, inp_h=inp_h, inp_w=inp_w, mesh_mapper=weights_mesh_mapper)
    ttnn_model = TtYolov8lModel(device=mesh_device, parameters=parameters, res=(inp_h, inp_w))

    torch_input_ch16 = F.pad(input_tensor, (0, 0, 0, 0, 0, 13))
    ttnn_input = yolov8l_dram_sharded_input_from_torch(mesh_device, torch_input_ch16)

    with torch.inference_mode():
        ttnn_model_output = ttnn_model(ttnn_input)[0]
        ttnn_model_output = ttnn.to_torch(ttnn_model_output, mesh_composer=output_mesh_composer)

    with torch.inference_mode():
        torch_model_output = torch_model(input_tensor)[0]

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")
