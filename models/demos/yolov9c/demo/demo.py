# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import cv2
import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.demos.yolov9c.demo.demo_seg_utils import LoadImages, postprocess, preprocess
from models.demos.yolov9c.demo.demo_utils import load_coco_class_names
from models.demos.yolov9c.reference import yolov9c
from models.demos.yolov9c.tt import ttnn_yolov9c
from models.demos.yolov9c.tt.model_preprocessing import create_yolov9c_input_tensors, create_yolov9c_model_parameters
from models.experimental.yolo_evaluation.yolo_common_evaluation import save_yolo_predictions_by_model
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0


def get_consistent_color(index):
    cmap = plt.get_cmap("tab20")
    color = cmap(index % 20)[:3]  # RGB values (0-1)
    return tuple(int(c * 255) for c in color)


def save_seg_predictions_by_model(result, save_dir, image_path, model_name):
    os.makedirs(os.path.join(save_dir, model_name), exist_ok=True)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = image.copy()

    if model_name == "torch_model":
        box_color = (0, 255, 0)
    else:
        box_color = (255, 0, 0)
    boxes = result.boxes.xyxy.cpu().detach().numpy().astype(int)
    scores = result.boxes.conf.cpu().detach().numpy()
    masks = result.masks.data.cpu().detach().numpy()

    for i in range(len(masks)):
        mask = masks[i]
        box = boxes[i]
        score = scores[i]
        color = get_consistent_color(i)
        mask_rgb = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            mask_rgb[:, :, c] = mask * color[c]

        overlay[mask.astype(bool)] = (0.5 * overlay[mask.astype(bool)] + 0.5 * mask_rgb[mask.astype(bool)]).astype(
            np.uint8
        )
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
        label = f"{score:.2f}"
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(save_dir, model_name, f"segmentation_{timestamp}.jpg")
    cv2.imwrite(out_path, overlay_bgr)
    logger.info(f"Saved to {out_path}")


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "source",
    [
        # "models/sample_data/huggingface_cat_image.jpg",
        "models/demos/yolov9c/demo/image.png",  # Uncomment to run the demo with another image.
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        "tt_model",
        # "torch_model",  # Uncomment to run the demo with torch model.
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        # "False", # Uncomment to run the demo with random weights.
        "True",
    ],
)
@pytest.mark.parametrize(
    "model_task",
    [
        "segment",
        # "detection", # Uncomment to run the demo for detection.
    ],
)
def test_demo(device, source, model_type, use_weights_from_ultralytics, model_task, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()

    weights = "yolov9c-seg.pt" if model_task == "segment" else "yolov9c.pt"
    enable_segment = True if model_task == "segment" else False

    if model_type == "torch_model":
        state_dict = None
        if use_weights_from_ultralytics:
            torch_model = YOLO(weights)  # Use weights "yolov9c.pt" for object detection
            state_dict = torch_model.state_dict()
        model = yolov9c.YoloV9(enable_segment=enable_segment)
        state_dict = model.state_dict() if state_dict is None else state_dict

        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2
        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info("Inferencing [Torch] Model")
    else:
        torch_input, ttnn_input = create_yolov9c_input_tensors(device)
        state_dict = None
        if use_weights_from_ultralytics:
            torch_model = YOLO(weights)  # Use weights "yolov9c.pt" for object detection
            state_dict = torch_model.state_dict()

        torch_model = yolov9c.YoloV9()
        state_dict = torch_model.state_dict() if state_dict is None else state_dict
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2

        torch_model.load_state_dict(new_state_dict)
        torch_model.eval()
        parameters = create_yolov9c_model_parameters(torch_model, torch_input, device=device)
        model = ttnn_yolov9c.YoloV9(
            device, parameters, enable_segment=enable_segment
        )  # Set enable_segment to False for Object detection.
        logger.info("Inferencing [TTNN] Model")

    save_dir = "models/demos/yolov9c/demo/runs"
    dataset = LoadImages(path=source)
    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)
    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, s = batch
        im = preprocess(im0s, res=(640, 640))
        img = torch.permute(im, (0, 2, 3, 1))
        img = img.reshape(
            1,
            1,
            img.shape[0] * img.shape[1] * img.shape[2],
            img.shape[3],
        )
        ttnn_im = ttnn.from_torch(img, dtype=ttnn.bfloat16)

        if model_type == "torch_model":
            preds = model(im)
        else:
            preds = model(ttnn_im)
            if enable_segment:
                preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32)  # detects

                preds[1][1] = ttnn.to_torch(preds[1][1], dtype=torch.float32)  # masks

                preds[1][2] = ttnn.to_torch(preds[1][2], dtype=torch.float32)  # protos
                preds[1][2] = preds[1][2].reshape((1, 154, 154, 32))
                preds[1][2] = preds[1][2].permute((0, 3, 1, 2))
            else:
                preds = ttnn.to_torch(outputs, dtype=torch.float32)

        results = postprocess(preds, im, im0s, batch)

        if enable_segment:
            for i in range(len(results)):
                save_seg_predictions_by_model(results[i], save_dir, paths[i], model_type)
        else:
            save_yolo_predictions_by_model(results[0], save_dir, source, model_type)

    logger.info("Inference done")
