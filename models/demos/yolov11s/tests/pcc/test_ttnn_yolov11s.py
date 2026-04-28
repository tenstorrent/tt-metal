# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov11s.common import YOLOV11S_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov11s.reference import yolov11s
from models.demos.yolov11s.tests.pcc.pcc_logging import log_assert_with_pcc
from models.demos.yolov11s.tt import ttnn_yolov11s
from models.demos.yolov11s.tt.model_preprocessing import create_yolov11s_input_tensors, create_yolov11s_model_parameters


@pytest.mark.parametrize(
    "resolution",
    [
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
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV11S_L1_SMALL_SIZE}], indirect=True)
def test_yolov11s(device, reset_seeds, resolution, use_pretrained_weights, model_location_generator, min_channels=8):
    torch_model = yolov11s.YoloV11s()
    torch_model.eval()

    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    torch_input, ttnn_input = create_yolov11s_input_tensors(
        device,
        batch=resolution[0],
        input_channels=resolution[1],
        input_height=resolution[2],
        input_width=resolution[3],
        is_sub_module=False,
    )
    n, c, h, w = ttnn_input.shape
    if c == 3:
        c = min_channels
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input = ttnn_input.to(device, input_mem_config)
    torch_output = torch_model(torch_input)
    parameters = create_yolov11s_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_yolov11s.TtnnYoloV11s(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    log_assert_with_pcc(
        f"YOLOv11s full model (pretrained_weights={use_pretrained_weights})", torch_output, ttnn_output, 0.99
    )


def _run_yolov11s_dp_mesh_pcc(mesh_device, use_pretrained_weights, model_location_generator, min_channels=8):
    """
    Data-parallel: global batch == mesh device count (one 640×640 sample per device), same as YOLOv8s DP PCC.

    Conv op args must be inferred with **per-device batch 1** (proxy tensor). Using the full global batch
    for ``infer_ttnn_module_args`` bakes batch_size=2 into every conv while each chip only holds one
    activation shard, which blows up first conv2d (L1 / wrong batch_size).
    """
    num_devices = mesh_device.get_num_devices()
    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(mesh_device)

    torch_model = yolov11s.YoloV11s()
    torch_model.eval()

    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    torch_input_tensor = torch.randn((num_devices, 3, 640, 640))
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

    torch_output = torch_model(torch_input_tensor)
    torch_param_proxy = torch_input_tensor[0:1]
    parameters = create_yolov11s_model_parameters(torch_model, torch_param_proxy, device=mesh_device)
    ttnn_model = ttnn_yolov11s.TtnnYoloV11s(mesh_device, parameters)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output, mesh_composer=output_mesh_composer)
    log_assert_with_pcc(
        f"YOLOv11s DP mesh (pretrained_weights={use_pretrained_weights})", torch_output, ttnn_output, 0.99
    )


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],
    indirect=True,
    ids=["n300_1x2"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV11S_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("use_pretrained_weights", [True])
def test_yolov11s_dp_batch2(mesh_device, reset_seeds, use_pretrained_weights, model_location_generator):
    _run_yolov11s_dp_mesh_pcc(mesh_device, use_pretrained_weights, model_location_generator)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 4)],
    indirect=True,
    ids=["wh_1x4"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV11S_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("use_pretrained_weights", [True])
def test_yolov11s_dp_batch4(mesh_device, reset_seeds, use_pretrained_weights, model_location_generator):
    _run_yolov11s_dp_mesh_pcc(mesh_device, use_pretrained_weights, model_location_generator)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 8)],
    indirect=True,
    ids=["t3k_1x8"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV11S_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("use_pretrained_weights", [True])
def test_yolov11s_dp_batch8(mesh_device, reset_seeds, use_pretrained_weights, model_location_generator):
    _run_yolov11s_dp_mesh_pcc(mesh_device, use_pretrained_weights, model_location_generator)
