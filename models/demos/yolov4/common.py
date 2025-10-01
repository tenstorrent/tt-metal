# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import cv2
import numpy as np
import torch

import ttnn
from models.demos.yolov4.reference.downsample1 import DownSample1
from models.demos.yolov4.reference.downsample2 import DownSample2
from models.demos.yolov4.reference.downsample3 import DownSample3
from models.demos.yolov4.reference.downsample4 import DownSample4
from models.demos.yolov4.reference.downsample5 import DownSample5
from models.demos.yolov4.reference.head import Head
from models.demos.yolov4.reference.neck import Neck
from models.demos.yolov4.reference.yolov4 import Yolov4
from models.tt_cnn.tt.builder import Conv2dConfiguration

YOLOV4_BOXES_PCC = 0.99
YOLOV4_CONFS_PCC = 0.9
YOLOV4_BOXES_PCC_BLACKHOLE = 0.96

YOLOV4_L1_SMALL_SIZE = 10960


def load_image(image_path, resolution):
    image = cv2.imread(image_path)
    image = cv2.resize(image, resolution)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def image_to_tensor(image):
    if type(image) == np.ndarray and len(image.shape) == 3:
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(image) == np.ndarray and len(image.shape) == 4:
        image = torch.from_numpy(image.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        raise TypeError("Unsupported image type")

    return torch.autograd.Variable(image)


def load_torch_model(model_location_generator, module=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        model_path = "models"
    else:
        model_path = model_location_generator("vision-models/yolov4", model_subdir="", download_if_ci_v2=True)

    if model_path == "models":
        if not os.path.exists("models/demos/yolov4/tests/pcc/yolov4.pth"):  # check if yolov4.th is availble
            os.system(
                "models/demos/yolov4/tests/pcc/yolov4_weights_download.sh"
            )  # execute the yolov4_weights_download.sh file
        weights_pth = "models/demos/yolov4/tests/pcc/yolov4.pth"
    else:
        weights_pth = os.path.join(model_path, "yolov4.pth")

    torch_dict = torch.load(weights_pth)
    state_dict = torch_dict

    if module is None:
        torch_model = Yolov4()
    elif module == "down1":
        torch_model = DownSample1()
        state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("down1."))}
    elif module == "down2":
        torch_model = DownSample2()
        state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("down2."))}
    elif module == "down3":
        torch_model = DownSample3()
        state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("down3."))}
    elif module == "down4":
        torch_model = DownSample4()
        state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("down4."))}
    elif module == "down5":
        torch_model = DownSample5()
        state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("down5."))}
    elif module == "head":
        torch_model = Head()
        state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("head."))}
    elif module == "neck":
        torch_model = Neck()
        state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("neek."))}
    else:
        raise ValueError("Invalid module name")

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    return torch_model


def get_model_result(ttnn_output_tensor, resolution, mesh_composer=None):
    result_boxes_padded = ttnn.to_torch(ttnn_output_tensor[0], mesh_composer=mesh_composer)
    result_confs = ttnn.to_torch(ttnn_output_tensor[1], mesh_composer=mesh_composer)

    result_boxes_padded = result_boxes_padded.permute(0, 2, 1, 3)
    result_boxes_list = []
    # Unpadding
    # That ttnn tensor is the concat output of 3 padded tensors
    # As a perf workaround I'm doing the unpadding on the torch output here.
    # TODO: cleaner ttnn code when ttnn.untilize() is fully optimized
    if resolution == (320, 320):
        box_1_start_i = 0
        box_1_end_i = 6100
        box_2_start_i = 6128
        box_2_end_i = 6228
        box_3_start_i = 6256
        box_3_end_i = 6356
    elif resolution == (640, 640):
        box_1_start_i = 0
        box_1_end_i = 24400
        box_2_start_i = 24428
        box_2_end_i = 24828
        box_3_start_i = 24856
        box_3_end_i = 25256
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    result_boxes_list.append(result_boxes_padded[:, box_1_start_i:box_1_end_i])
    result_boxes_list.append(result_boxes_padded[:, box_2_start_i:box_2_end_i])
    result_boxes_list.append(result_boxes_padded[:, box_3_start_i:box_3_end_i])
    result_boxes = torch.cat(result_boxes_list, dim=1)

    return [result_boxes.to(torch.float16), result_confs.to(torch.float16)]


def create_sharding_strategy(conv2d_args):
    """Create appropriate sharding strategy from conv2d_args"""
    from models.tt_cnn.tt.builder import (
        AutoShardedStrategyConfiguration,
        BlockShardedStrategyConfiguration,
        HeightShardedStrategyConfiguration,
        WidthShardedStrategyConfiguration,
    )

    shard_layout = getattr(conv2d_args, "shard_layout", None)
    if shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        return HeightShardedStrategyConfiguration(
            reshard_if_not_optimal=getattr(conv2d_args, "reshard_if_not_optimal", False),
            act_block_h_override=getattr(conv2d_args, "act_block_h", 0) or 0,
        )
    elif shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        return BlockShardedStrategyConfiguration(
            reshard_if_not_optimal=getattr(conv2d_args, "reshard_if_not_optimal", False),
        )
    elif shard_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return WidthShardedStrategyConfiguration(
            reshard_if_not_optimal=getattr(conv2d_args, "reshard_if_not_optimal", False),
        )
    else:
        return AutoShardedStrategyConfiguration()


def create_conv2d_config(conv_args, weight, bias, activation=None):
    """Helper to create Conv2dConfiguration with proper sharding parameters"""
    return Conv2dConfiguration.from_model_args(
        conv_args,
        weight,
        bias,
        weights_dtype=ttnn.bfloat8_b,
        sharding_strategy=create_sharding_strategy(conv_args),
        deallocate_activation=getattr(conv_args, "deallocate_activation", False),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        enable_act_double_buffer=True,
        output_layout=ttnn.TILE_LAYOUT,
        activation=activation,
    )
