# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import cv2
import pytest
import torch

import ttnn
from models.demos.yolov4.common import image_to_tensor, load_image, load_torch_model
from models.demos.yolov4.post_processing import load_class_names, plot_boxes_cv2, post_processing
from models.demos.yolov4.runner.runner import YOLOv4Runner
from models.demos.yolov4.tt.model_preprocessing import create_yolov4_model_parameters
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull()
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 320),
        (640, 640),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov4(device, reset_seeds, model_location_generator, resolution):
    torch.manual_seed(0)

    imgfile = "models/demos/yolov4/resources/giraffe_320.jpg"
    img = load_image(imgfile, resolution)
    torch_input = image_to_tensor(img)

    input_tensor = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16)

    torch_model = load_torch_model(model_location_generator)
    parameters = create_yolov4_model_parameters(torch_model, torch_input, resolution, device)

    ttnn_model_runner = YOLOv4Runner(device, parameters, resolution)

    ## Giraffe image detection
    conf_thresh = 0.3
    nms_thresh = 0.4
    output = ttnn_model_runner.run(ttnn_input)

    boxes = post_processing(img, conf_thresh, nms_thresh, output)
    namesfile = "models/demos/yolov4/resources/coco.names"
    class_names = load_class_names(namesfile)
    img = cv2.imread(imgfile)
    plot_boxes_cv2(img, boxes[0], "ttnn_yolov4_prediction_demo.jpg", class_names)
