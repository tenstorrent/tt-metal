# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import cv2
import numpy as np
import torch

import ttnn
from models.experimental.yolov10.reference.yolov10 import YOLOv10
from ultralytics import YOLO

YOLOV10_BOXES_PCC = 0.99
YOLOV10_CONFS_PCC = 0.9
YOLOV10_BOXES_PCC_BLACKHOLE = 0.96


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


def load_torch_model(use_weights_from_ultralytics=True, module=None):
    state_dict = None
    model = YOLOv10()
    if use_weights_from_ultralytics:
        pretrained_model = YOLO("yolov10x.pt")
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

    new_state_dict = {name: param for name, param in model.state_dict().items() if isinstance(param, torch.FloatTensor)}

    model.load_state_dict(new_state_dict)
    model.eval()

    return model


def get_model_result(ttnn_output_tensor, resolution=(640, 640)):
    result_boxes_padded = ttnn.to_torch(ttnn_output_tensor[0])
    result_confs = ttnn.to_torch(ttnn_output_tensor[1])

    result_boxes_padded = result_boxes_padded.permute(0, 2, 1, 3)
    result_boxes_list = []
    # Unpadding workaround
    # The 'ttnn' tensor is a concatenation of three padded tensors.
    # For performance reasons, unpadding is currently handled on the torch output.
    # TODO: Replace this with cleaner TTNN logic once ttnn.utilize() is fully optimized.
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
