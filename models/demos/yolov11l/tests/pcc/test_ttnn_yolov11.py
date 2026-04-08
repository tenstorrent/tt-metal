# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.yolov11l.common import (
    load_torch_model,
    yolov11_l1_small_size_for_res,
    yolov11_trace_region_size_e2e_for_res,
)
from models.demos.yolov11l.reference import yolov11
from models.demos.yolov11l.tt import ttnn_yolov11
from models.demos.yolov11l.tt.common import get_mesh_mappers
from models.demos.yolov11l.tt.model_preprocessing import create_yolov11_input_tensors, create_yolov11_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "resolution",
    [
        ([1, 3, 1280, 1280]),
        ([1, 3, 640, 640]),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weights",
    [
        True,
        # False
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": yolov11_l1_small_size_for_res(1280, 1280),
            "trace_region_size": yolov11_trace_region_size_e2e_for_res(1280, 1280),
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
def test_yolov11(device, reset_seeds, resolution, use_pretrained_weights, model_location_generator, min_channels=8):
    torch_model = yolov11.YoloV11()
    torch_model.eval()

    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=resolution[0],
        input_channels=resolution[1],
        input_height=resolution[2],
        input_width=resolution[3],
        is_sub_module=False,
    )
    n, c, h, w = ttnn_input.shape
    if c == 3:  # for sharding config of padded input
        c = min_channels
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input = ttnn_input.to(device, input_mem_config)
    torch_output = torch_model(torch_input)
    parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_yolov11.TtnnYoloV11(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 8)],
    indirect=True,
    ids=["t3k_1x8"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": yolov11_l1_small_size_for_res(1280, 1280),
            "trace_region_size": 23887872,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("use_pretrained_weights", [True])
@pytest.mark.parametrize("input_hw", [640, 1280], ids=["640", "1280"])
# @pytest.mark.parametrize("input_hw", [1280], ids=["1280"])
def test_yolov11_dp_batch8(
    mesh_device, reset_seeds, use_pretrained_weights, model_location_generator, input_hw, min_channels=8
):
    # create_yolov11_input_tensors multiplies batch by num_devices internally.
    # For a 1x8 mesh, batch=1 gives global batch 8 (one sample per device).
    resolution = [1, 3, input_hw, input_hw]
    torch_model = yolov11.YoloV11()
    torch_model.eval()

    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    torch_input, ttnn_input = create_yolov11_input_tensors(
        mesh_device,
        batch=resolution[0],
        input_channels=resolution[1],
        input_height=resolution[2],
        input_width=resolution[3],
        is_sub_module=False,
    )
    n, c, h, w = ttnn_input.shape
    if c == 3:  # for sharding config of padded input
        c = min_channels
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input = ttnn_input.to(mesh_device, input_mem_config)
    torch_output = torch_model(torch_input)
    parameters = create_yolov11_model_parameters(torch_model, torch_input, device=mesh_device)
    ttnn_model = ttnn_yolov11.TtnnYoloV11(mesh_device, parameters)
    ttnn_output = ttnn_model(ttnn_input)
    _, _, output_mesh_composer = get_mesh_mappers(mesh_device)
    ttnn_output = ttnn.to_torch(ttnn_output, mesh_composer=output_mesh_composer)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
