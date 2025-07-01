# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import cv2
import pytest

import ttnn
from models.demos.yolov4.common import image_to_tensor, load_image
from models.demos.yolov4.post_processing import load_class_names, plot_boxes_cv2, post_processing
from models.demos.yolov4.runner.performant_runner import YOLOv4PerformantRunner
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 40960, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 320),
        (640, 640),
    ],
)
@pytest.mark.parametrize(
    "imgfile",
    [
        "models/demos/yolov4/resources/giraffe_320.jpg",
    ],
)
def test_yolov4(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    imgfile,
):
    disable_persistent_kernel_cache()

    yolov4_trace_2cq = YOLOv4PerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=None,
    )

    img = load_image(imgfile, resolution)
    torch_input_tensor = image_to_tensor(img)
    # torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)

    preds = yolov4_trace_2cq.run(torch_input_tensor)

    conf_thresh = 0.3
    nms_thresh = 0.4
    boxes = post_processing(img, conf_thresh, nms_thresh, preds)

    namesfile = "models/demos/yolov4/resources/coco.names"
    class_names = load_class_names(namesfile)

    img_cv = cv2.imread(imgfile)

    # Create a unique output file using image name and resolution
    output_dir = "yolov4_predictions"
    os.makedirs(output_dir, exist_ok=True)
    img_base = os.path.splitext(os.path.basename(imgfile))[0]
    output_filename = f"ttnn_yolov4_{img_base}_{resolution[0]}x{resolution[1]}.jpg"
    output_path = os.path.join(output_dir, output_filename)

    plot_boxes_cv2(img_cv, boxes[0], output_path, class_names)

    yolov4_trace_2cq.release()
