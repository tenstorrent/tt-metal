# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import cv2
import fiftyone
import numpy as np
import pytest
import torch
from loguru import logger
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn
from ultralytics import YOLO

import ttnn
from models.demos.yolov4.post_processing import gen_yolov4_boxes_confs
from models.experimental.yolo_evaluation.yolo_evaluation_utils import (
    LoadImages,
    postprocess,
    preprocess,
)
from models.utility_functions import disable_persistent_kernel_cache


def iou(pred_box, gt_box):
    """Calculate IoU (Intersection over Union) between two bounding boxes."""
    x1_pred, y1_pred, x2_pred, y2_pred = pred_box[:4]
    x1_gt, y1_gt, x2_gt, y2_gt = gt_box

    # Calculate the intersection area
    ix = max(0, min(x2_pred, x2_gt) - max(x1_pred, x1_gt))
    iy = max(0, min(y2_pred, y2_gt) - max(y1_pred, y1_gt))
    intersection = ix * iy

    # Calculate the union area
    union = (x2_pred - x1_pred) * (y2_pred - y1_pred) + (x2_gt - x1_gt) * (y2_gt - y1_gt) - intersection
    return intersection / union


def calculate_map(predictions, ground_truths, iou_threshold=0.5, num_classes=3):
    """Calculate mAP for object detection."""
    ap_scores = []

    # Iterate through each class
    for class_id in range(num_classes):
        y_true = []
        y_scores = []

        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = [p for p in pred if p[5] == class_id]
            gt_boxes = [g for g in gt if g[4] == class_id]

            for pred_box in pred_boxes:
                best_iou = 0
                matched_gt = None

                for gt_box in gt_boxes:
                    iou_score = iou(pred_box[:4], gt_box[:4])  # Compare the [x1, y1, x2, y2] part of the box
                    if iou_score > best_iou:
                        best_iou = iou_score
                        matched_gt = gt_box

                # If IoU exceeds threshold, consider it a true positive
                if best_iou >= iou_threshold:
                    y_true.append(1)  # True Positive
                    y_scores.append(pred_box[4])
                    gt_boxes.remove(matched_gt)  # Remove matched ground truth
                else:
                    y_true.append(0)  # False Positive
                    y_scores.append(pred_box[4])

            # Ground truth boxes that were not matched are false negatives
            for gt_box in gt_boxes:
                y_true.append(0)  # False Negative
                y_scores.append(0)  # No detection
        if len(y_true) == 0 or len(y_scores) == 0:
            continue

        # Calculate precision-recall and average precision for this class
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        ap_scores.append(ap)

    # Calculate mAP as the mean of the AP scores
    mAP = np.mean(ap_scores)
    return mAP


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_load(weights, model_path, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        w = model_path  # depends on model which we take
        ckpt = torch.load(w, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


def save_yolo_predictions_by_model(result, save_dir, image_path, model_name):
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if model_name == "torch_model":
        bounding_box_color, label_color = (0, 255, 0), (0, 255, 0)
    else:
        bounding_box_color, label_color = (255, 0, 0), (255, 255, 0)

    boxes = result["boxes"]["xyxy"]
    scores = result["boxes"]["conf"]
    classes = result["boxes"]["cls"]
    names = result["names"]

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {score.item():.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), bounding_box_color, 3)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    filename = os.path.basename(image_path)
    output_name = f"prediction_{filename}"
    output_path = os.path.join(model_save_dir, output_name)

    cv2.imwrite(output_path, image)

    logger.info(f"Predictions saved to {output_path}")


def evaluation(
    device,
    res,
    model_type,
    model,
    parameters,
    input_dtype,
    input_layout,
    save_dir,
    model_name=None,
    additional_layer=None,
):
    disable_persistent_kernel_cache()

    dataset_name = "coco-2017"
    dataset = fiftyone.zoo.load_zoo_dataset(
        dataset_name,
        split="validation",
        max_samples=500,
    )

    source_list = [i["filepath"] for i in dataset]
    data_set = LoadImages(path=[i["filepath"] for i in dataset])

    with open(os.path.expanduser("~") + "/fiftyone" + "/" + dataset_name + "/info.json", "r") as file:
        # Parse the JSON data
        data = json.load(file)
        classes = data["classes"]

    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)

    index = 0
    predicted_bbox = []
    preprocessed_images = []  # List to store preprocessed images

    # Preprocessing loop
    for batch in data_set:
        paths, im0s, s = batch
        if model_name == "YOLOv4":
            sized = cv2.resize(im0s[0], (res[0], res[1]))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            if type(sized) == np.ndarray and len(sized.shape) == 3:  # cv2 image
                img = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(sized) == np.ndarray and len(sized.shape) == 4:
                img = torch.from_numpy(sized.transpose(0, 3, 1, 2)).float().div(255.0)
            else:
                exit()
        else:
            im = preprocess(im0s, res=res)

        if model_name == "YOLOv4":
            input_shape = img.shape
            im = img.clone()
            img = torch.autograd.Variable(img)
            n, c, h, w = input_shape
            input_tensor = torch.permute(img, (0, 2, 3, 1))
            input_tensor = input_tensor.reshape(1, 1, h * w * n, c)
            ttnn_im = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn_im = ttnn.pad(ttnn_im, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
        else:
            ttnn_im = im.permute((0, 2, 3, 1))

        if model_name in ["YOLOv11", "YOLOv9c", "YOLOv8x", "YOLOv10"]:
            ttnn_im = ttnn_im.reshape(
                1,
                1,
                ttnn_im.shape[0] * ttnn_im.shape[1] * ttnn_im.shape[2],
                ttnn_im.shape[3],
            )
        if model_name != "YOLOv4":
            if model_name == "YOLOv8x":
                ttnn_im = ttnn.from_torch(ttnn_im, dtype=input_dtype, layout=input_layout)
            else:
                ttnn_im = ttnn.from_torch(ttnn_im, dtype=input_dtype, layout=input_layout, device=device)

        if model_type != "torch_model":
            preprocessed_images.append((ttnn_im, im, im0s))
        else:
            preprocessed_images.append((im, im, im0s))

    # Model inference loop
    for ttnn_im, im, im0s in preprocessed_images:
        if model_type == "torch_model":
            preds = model(im)
            if model_name == "YOLOv4":
                from models.demos.yolov4.post_processing import get_region_boxes

                y1, y2, y3 = gen_yolov4_boxes_confs(preds)
                output = get_region_boxes([y1, y2, y3])
        else:
            if model_name in ["YOLOv11", "YOLOv9c", "YOLOv10"]:
                preds = model(ttnn_im)
                preds = ttnn.to_torch(preds, dtype=torch.float32)
            elif model_name == "YOLOv8x":
                ttnn_im = ttnn.pad(ttnn_im, [1, 1, ttnn_im.shape[2], 16], [0, 0, 0, 0], 0)
                preds = model.execute_yolov8x_trace_2cqs_inference(ttnn_im)
                preds = ttnn.to_torch(preds, dtype=torch.float32)
            elif model_name == "YOLOv4":
                preds = model._execute_yolov4_trace_2cqs_inference(ttnn_im)
                result_boxes = preds[0]
                result_confs = preds[1]
                output = [result_boxes.to(torch.float16), result_confs.to(torch.float16)]
            else:
                preds = model(ttnn_im)
                preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32)

        if model_name == "YOLOv4":
            from models.demos.yolov4.post_processing import post_processing

            results = post_processing(img, 0.3, 0.4, output)
        elif model_name == "YOLOv10":
            results = postprocess(preds, im, im0s, batch, classes, model_name=model_name, conf=0.5)[0]
        else:
            results = postprocess(preds, im, im0s, batch, classes)[0]

        if model_name == "YOLOv4":
            predicted_temp = results[0]
            for i in predicted_temp:
                del i[5]
            predicted_bbox.append(predicted_temp)
            index += 1
        else:
            pred = results["boxes"]["xyxy"].tolist()
            h, w = results["orig_img"].shape[0], results["orig_img"].shape[1]

            for index_of_prediction, (conf, values) in enumerate(
                zip(results["boxes"]["conf"].tolist(), results["boxes"]["cls"].tolist())
            ):
                pred[index_of_prediction][0] /= w  # normalizing the output since groundtruth values are normalized
                pred[index_of_prediction][1] /= h  # normalizing the output since groundtruth values are normalized
                pred[index_of_prediction][2] /= w  # normalizing the output since groundtruth values are normalized
                pred[index_of_prediction][3] /= h  # normalizing the output since groundtruth values are normalized
                pred[index_of_prediction].append(conf)
                pred[index_of_prediction].append(int(values))

            predicted_bbox.append(pred)
            save_yolo_predictions_by_model(results, save_dir, source_list[index], model_type)
            index += 1
    if model_name in ["YOLOv8x"] and model_type == "tt_metal":
        model.release_yolov8x_trace_2cqs_inference()
    ground_truth = []
    for i in dataset:
        sample = []
        if i["ground_truth"] != None:
            for j in i["ground_truth"]["detections"]:
                bb_temp = j["bounding_box"]
                bb_temp[2] += bb_temp[0]
                bb_temp[3] += bb_temp[1]
                bb_temp.append(classes.index(j["label"]))
                sample.append(bb_temp)
        ground_truth.append(sample)

    class_indices = [box[5] for image in predicted_bbox for box in image]
    num_classes = max(class_indices) + 1

    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    mAPval_50_95 = []
    for iou_threshold in iou_thresholds:
        # Calculate mAP
        mAP = calculate_map(predicted_bbox, ground_truth, num_classes=num_classes, iou_threshold=iou_threshold)
        mAPval_50_95.append(mAP)

    logger.info("mAPval_50_95: {}", mAPval_50_95)
    mAPval50_95_value = sum(mAPval_50_95) / len(mAPval_50_95)

    logger.info(f"Mean Average Precision for val 50-95 (mAPval 50-95): {mAPval50_95_value:.4f}")


@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
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
def test_run_yolov4_eval(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    model_type,
):
    if model_type == "torch_model":
        from models.demos.yolov4.common import load_torch_model

        torch_model = load_torch_model(model_location_generator)
    else:
        from models.demos.yolov4.runner.performant_runner import YOLOv4PerformantRunner

        ttnn_model = YOLOv4PerformantRunner(
            device,
            batch_size,
            act_dtype,
            weight_dtype,
            resolution=resolution,
            model_location_generator=None,
        )

    save_dir = "models/demos/yolov4/demo/runs"
    model_name = "YOLOv4"
    input_dtype = ttnn.bfloat16
    input_layout = ttnn.ROW_MAJOR_LAYOUT

    evaluation(
        device=device,
        res=resolution,
        model_type=model_type,
        model=ttnn_model if model_type == "tt_model" else torch_model,
        parameters=None,
        input_dtype=input_dtype,
        input_layout=input_layout,
        save_dir=save_dir,
        model_name=model_name,
        additional_layer=None,
    )


@pytest.mark.parametrize(
    "model_type",
    [("tt_model"), ("torch_model")],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("res", [(640, 640)])
def test_yolov8s_world(device, model_type, res, reset_seeds):
    from models.experimental.yolov8s_world.reference import yolov8s_world
    from models.experimental.yolov8s_world.tt.ttnn_yolov8s_world import TtYOLOWorld
    from models.experimental.yolov8s_world.tt.ttnn_yolov8s_world_utils import (
        create_custom_preprocessor,
        attempt_load,
        move_to_device,
    )

    from ttnn.model_preprocessing import preprocess_model_parameters

    if model_type == "torch_model":
        model = attempt_load("yolov8s-world.pt", map_location="cpu")
    else:
        weights_torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
        torch_model = yolov8s_world.YOLOWorld(model_torch=weights_torch_model)

        state_dict = weights_torch_model.state_dict()
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
            new_state_dict[name1] = parameter2

        torch_model.load_state_dict(new_state_dict)
        torch_model = torch_model.model

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device)
        )

        for i in [12, 15, 19, 22]:
            parameters["model"][i]["attn"]["gl"]["weight"] = ttnn.to_device(
                parameters["model"][i]["attn"]["gl"]["weight"], device=device
            )
            parameters["model"][i]["attn"]["gl"]["bias"] = ttnn.to_device(
                parameters["model"][i]["attn"]["gl"]["bias"], device=device
            )
            parameters["model"][i]["attn"]["bias"] = ttnn.to_device(
                parameters["model"][i]["attn"]["bias"], device=device
            )

        parameters["model"][16] = move_to_device(parameters["model"][16], device)

        parameters["model"][23]["cv4"] = move_to_device(parameters["model"][23]["cv4"], device)

        model = TtYOLOWorld(device=device, parameters=parameters)

    input_dtype = ttnn.bfloat16
    input_layout = ttnn.ROW_MAJOR_LAYOUT

    save_dir = "models/experimental/yolov8s_world/demo/runs"

    evaluation(
        device=device,
        res=res,
        model_type=model_type,
        model=model,
        parameters=None,
        input_dtype=input_dtype,
        input_layout=input_layout,
        save_dir=save_dir,
    )


@pytest.mark.parametrize(
    "model_type",
    [("tt_model"), ("torch_model")],
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_yolov8x(device, model_type, res, use_program_cache, reset_seeds):
    from models.demos.yolov8x.tests.yolov8x_e2e_performant import Yolov8xTrace2CQ

    torch_model = YOLO("yolov8x.pt")
    torch_model = torch_model.model
    model = torch_model.eval()

    if model_type == "torch_model":
        torch_model = YOLO("yolov8x.pt")
        torch_model = torch_model.model
        model = torch_model.eval()
        logger.info("Inferencing using Torch Model")
    else:
        model = Yolov8xTrace2CQ()
        model.initialize_yolov8x_trace_2cqs_inference(
            device,
            1,
        )
        logger.info("Inferencing using ttnn Model")

    save_dir = "models/demos/yolov8x/demo/runs"

    input_dtype = ttnn.bfloat16
    input_layout = ttnn.ROW_MAJOR_LAYOUT

    evaluation(
        device=device,
        res=res,
        model_type=model_type,
        model=model,
        parameters=None,
        input_dtype=input_dtype,
        input_layout=input_layout,
        save_dir=save_dir,
        model_name="YOLOv8x",
    )
