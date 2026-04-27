# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn

# from models.common.utility_functions import disable_persistent_kernel_cache
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov8s.common import YOLOV8S_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov8s.tt.tt_yolov8s_utils import create_custom_mesh_preprocessor, custom_preprocessor
from models.demos.yolov8s.tt.ttnn_yolov8s import TtYolov8sModel
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE}], indirect=True, ids=["0"])
@pytest.mark.parametrize(
    "input_tensor",
    [torch.rand((1, 3, 640, 640))],
    ids=["input_tensor1"],
)
@pytest.mark.parametrize(
    "use_pretrained_weights",
    [True],
)
def test_yolov8s_640(device, input_tensor, use_pretrained_weights, model_location_generator):
    # disable_persistent_kernel_cache()

    inp_h, inp_w = input_tensor.shape[2], input_tensor.shape[3]
    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)
        state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict, inp_h=inp_h, inp_w=inp_w)
    ttnn_model = TtYolov8sModel(device=device, parameters=parameters, res=(inp_h, inp_w))

    n, c, h, w = input_tensor.shape
    if c == 3:
        c = 16
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input = ttnn_input.to(device, input_mem_config)

    with torch.inference_mode():
        ttnn_model_output = ttnn_model(ttnn_input)[0]
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    with torch.inference_mode():
        torch_model_output = torch_model(input_tensor)[0]

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


def _run_yolov8s_dp_mesh_pcc(mesh_device, use_pretrained_weights, model_location_generator, min_channels=16):
    num_devices = mesh_device.get_num_devices()
    batch_size = num_devices

    torch_model = load_torch_model(model_location_generator)

    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_device, weights_mesh_mapper),
        device=mesh_device,
    )
    ttnn_model = TtYolov8sModel(device=mesh_device, parameters=parameters, res=(640, 640))

    torch_input_tensor = torch.rand((batch_size, 3, 640, 640))

    n, c, h, w = torch_input_tensor.shape
    n_per_device = n // num_devices if n // num_devices != 0 else n
    c_padded = min_channels if c < min_channels else c

    input_mem_config = ttnn.create_sharded_memory_config(
        [n_per_device, c_padded, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )

    ttnn_input = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
    )
    ttnn_input = ttnn_input.to(mesh_device, input_mem_config)

    with torch.inference_mode():
        ttnn_model_output = ttnn_model(ttnn_input)[0]
        ttnn_model_output = ttnn.to_torch(ttnn_model_output, mesh_composer=output_mesh_composer)

    with torch.inference_mode():
        torch_model_output = torch_model(torch_input_tensor)[0]

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],
    indirect=True,
    ids=["n300_1x2"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("use_pretrained_weights", [True])
def test_yolov8s_dp_batch2(mesh_device, use_pretrained_weights, model_location_generator):
    _run_yolov8s_dp_mesh_pcc(mesh_device, use_pretrained_weights, model_location_generator)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 4)],
    indirect=True,
    ids=["wh_1x4"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("use_pretrained_weights", [True])
def test_yolov8s_dp_batch4(mesh_device, use_pretrained_weights, model_location_generator):
    _run_yolov8s_dp_mesh_pcc(mesh_device, use_pretrained_weights, model_location_generator)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 8)],
    indirect=True,
    ids=["t3k_1x8"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("use_pretrained_weights", [True])
def test_yolov8s_dp_batch8(mesh_device, use_pretrained_weights, model_location_generator):
    _run_yolov8s_dp_mesh_pcc(mesh_device, use_pretrained_weights, model_location_generator)
