# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov9c.demo.demo_utils import (
    load_coco_class_names,
    load_torch_model,
    postprocess,
    save_seg_predictions_by_model,
)
from models.demos.yolov9c.runner.performant_runner import YOLOv9PerformantRunner
from models.experimental.yolo_eval.evaluate import save_yolo_predictions_by_model
from models.experimental.yolo_eval.utils import LoadImages
from models.experimental.yolo_eval.utils import postprocess as obj_postprocess
from models.experimental.yolo_eval.utils import preprocess
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "source",
    [
        "models/demos/yolov9c/demo/image.png",
        # "models/sample_data/huggingface_cat_image.jpg", # Uncomment to run the demo with another image.
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",  # To run the demo with pre-trained weights
        # "False", # Uncomment to the run demo with random weights
    ],
)
@pytest.mark.parametrize(
    "model_task",
    [
        "segment",  # To run the demo for instance segmentation
        "detect",  # To run the demo for Object Detection
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
def test_demo(
    device,
    source,
    use_weights_from_ultralytics,
    model_type,
    model_task,
    reset_seeds,
):
    disable_persistent_kernel_cache()

    enable_segment = model_task == "segment"
    save_dir = "models/demos/yolov9c/demo/runs"
    dataset = LoadImages(path=source)
    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, s = batch
        im = preprocess(im0s, res=(640, 640))

        if model_type == "torch_model":
            model = load_torch_model(use_weights_from_ultralytics=use_weights_from_ultralytics, model_task=model_task)
            logger.info("Inferencing [Torch] Model")
            preds = model(im)
            if enable_segment:
                results = postprocess(preds, im, im0s, batch)
                save_seg_predictions_by_model(results[0], save_dir, source, model_type)
            else:
                results = obj_postprocess(preds, im, im0s, batch, names)
                save_yolo_predictions_by_model(results[0], save_dir, source, model_type)

        else:
            performant_runner = YOLOv9PerformantRunner(
                device,
                1,
                ttnn.bfloat8_b,
                ttnn.bfloat8_b,
                model_task=model_task,
                resolution=(640, 640),
                model_location_generator=None,
                torch_input_tensor=im,
            )
            performant_runner._capture_yolov9_trace_2cqs()
            logger.info("Inferencing [TTNN] Model")

            preds = performant_runner.run(torch_input_tensor=im)
            preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32)
            if enable_segment:
                detect1_out, detect2_out, detect3_out = [
                    ttnn.to_torch(tensor, dtype=torch.float32) for tensor in preds[1][0]
                ]
                mask = ttnn.to_torch(preds[1][1], dtype=torch.float32)
                proto = ttnn.to_torch(preds[1][2], dtype=torch.float32)
                proto = proto.reshape((1, 160, 160, 32)).permute((0, 3, 1, 2))
                preds[1] = [[detect1_out, detect2_out, detect3_out], mask, proto]
                results = postprocess(preds, im, im0s, batch)
                for i in range(len(results)):
                    save_seg_predictions_by_model(results[i], save_dir, paths[i], "tt_model")
            else:
                results = obj_postprocess(preds[0], im, im0s, batch, names)[0]
                save_yolo_predictions_by_model(results, save_dir, source, "tt_model")
            performant_runner.release()

    logger.info("Inference done")
