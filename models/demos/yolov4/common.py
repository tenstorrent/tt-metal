# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

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

YOLOV4_BOXES_PCC = 0.99
YOLOV4_CONFS_PCC = 0.9
YOLOV4_BOXES_PCC_BLACKHOLE = 0.96


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
    if model_location_generator == None:
        model_path = "models"
    else:
        model_path = model_location_generator("models", model_subdir="Yolo")

    if model_path == "models":
        if not os.path.exists("models/demos/yolov4/tests/pcc/yolov4.pth"):  # check if yolov4.th is availble
            os.system(
                "models/demos/yolov4/tests/pcc/yolov4_weights_download.sh"
            )  # execute the yolov4_weights_download.sh file
        weights_pth = "models/demos/yolov4/tests/pcc/yolov4.pth"
    else:
        weights_pth = str(model_path / "yolov4.pth")

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


def get_mesh_mappers(device):
    if device.get_num_devices() > 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = None
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer
