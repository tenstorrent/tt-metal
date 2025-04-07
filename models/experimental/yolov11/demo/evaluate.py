# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import fiftyone
import os
import json
import torch
import cv2
from datetime import datetime
import ttnn
from functools import partial
from loguru import logger
import sys
from tqdm import tqdm
from models.utility_functions import disable_persistent_kernel_cache
import pytest
from models.experimental.yolov11.demo.demo_utils import LoadImages, preprocess, postprocess
from models.experimental.yolov11.reference.yolov11 import attempt_load
from torch import nn
import numpy as np
import shutil
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings

warnings.filterwarnings("ignore")


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
            # print(f"No predictions or ground truth for class {class_id}")
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"prediction_{timestamp}.jpg"
    output_path = os.path.join(model_save_dir, output_name)

    cv2.imwrite(output_path, image)

    print(f"Predictions saved to {output_path}")


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

    dataset = fiftyone.zoo.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=250,
    )

    source_list = [i["filepath"] for i in dataset]
    data_set = LoadImages(path=[i["filepath"] for i in dataset])

    with open("/home/ubuntu/fiftyone/coco-2017/info.json", "r") as file:
        # Parse the JSON data
        data = json.load(file)
        classes = data["classes"]

    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)

    index = 0
    predicted_bbox = []
    for batch in tqdm(data_set, desc="Processing dataset"):
        sample = []
        paths, im0s, s = batch
        if model_name == "YOLOv4":
            sized = cv2.resize(im0s[0], (res[0], res[1]))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            if type(sized) == np.ndarray and len(sized.shape) == 3:  # cv2 image
                img = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(sized) == np.ndarray and len(sized.shape) == 4:
                img = torch.from_numpy(sized.transpose(0, 3, 1, 2)).float().div(255.0)
        else:
            im = preprocess(im0s, resolution=res)

        if model_name == "YOLOv4":
            input_shape = img.shape
            input_tensor = torch.permute(img, (0, 2, 3, 1))
            # input_tensor = ttnn.from_torch(input_tensor, ttnn.bfloat16)
            input_tensor = torch.permute(img, (0, 2, 3, 1))  # put channel at the end
            input_tensor = torch.nn.functional.pad(
                input_tensor, (0, 13, 0, 0, 0, 0, 0, 0)
            )  # pad channel dim from 3 to 16
            N, H, W, C = input_tensor.shape
            input_tensor = torch.reshape(input_tensor, (N, 1, H * W, C))

            shard_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 7),
                    ),
                }
            )
            n_cores = 64
            shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)
            input_mem_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )
            ttnn_im = ttnn.from_torch(
                input_tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=input_mem_config,
            )
        else:
            ttnn_im = im.permute((0, 2, 3, 1))
        if model_name == "YOLOv11":  # only for yolov11
            ttnn_im = ttnn_im.reshape(
                1,
                1,
                ttnn_im.shape[0] * ttnn_im.shape[1] * ttnn_im.shape[2],
                ttnn_im.shape[3],
            )
        if model_name != "YOLOv4":
            ttnn_im = ttnn.from_torch(ttnn_im, dtype=input_dtype, layout=input_layout, device=device)

        if model_type == "torch_model":
            preds = model(im)
        else:
            preds = model(ttnn_im)
            if model_name == "YOLOv11":
                preds = ttnn.to_torch(preds, dtype=torch.float32)
            elif model_name == "YOLOv4":
                output_tensor1 = ttnn.to_torch(preds[0])
                output_tensor1 = output_tensor1.reshape(1, 40, 40, 255)
                output_tensor1 = torch.permute(output_tensor1, (0, 3, 1, 2))

                output_tensor2 = ttnn.to_torch(preds[1])
                output_tensor2 = output_tensor2.reshape(1, 20, 20, 255)
                output_tensor2 = torch.permute(output_tensor2, (0, 3, 1, 2))

                output_tensor3 = ttnn.to_torch(preds[2])
                output_tensor3 = output_tensor3.reshape(1, 10, 10, 255)
                output_tensor3 = torch.permute(output_tensor3, (0, 3, 1, 2))

                yolo1 = additional_layer(
                    anchor_mask=[0, 1, 2],
                    num_classes=len(classes),
                    anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                    num_anchors=9,
                    stride=8,
                )

                yolo2 = additional_layer(
                    anchor_mask=[3, 4, 5],
                    num_classes=len(classes),
                    anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                    num_anchors=9,
                    stride=16,
                )

                yolo3 = additional_layer(
                    anchor_mask=[6, 7, 8],
                    num_classes=len(classes),
                    anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                    num_anchors=9,
                    stride=32,
                )
                y1 = yolo1(output_tensor1)
                y2 = yolo2(output_tensor2)
                y3 = yolo3(output_tensor3)
                from models.demos.yolov4.demo.demo import get_region_boxes

                output = get_region_boxes([y1, y2, y3])

            else:
                preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32)
        if model_name == "YOLOv4":
            from models.demos.yolov4.demo.demo import post_processing

            results = post_processing(img, 0.3, 0.4, output)
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

    ground_truth = []
    for i in tqdm(dataset, desc="Processing dataset"):
        sample = []
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
        print(f"Mean Average Precision (mAP): {mAP:.4f} for IOU Threshold: {iou_threshold:.4f}")
        mAPval_50_95.append(mAP)

    print("mAPval_50_95", mAPval_50_95)
    mAPval50_95_value = sum(mAPval_50_95) / len(mAPval_50_95)

    print(f"Mean Average Precision for val 50-95 (mAPval 50-95): {mAPval50_95_value:.4f}")


@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("res", [(224, 224)])
def test_yolo11n(device, model_type, res, reset_seeds):
    from models.experimental.yolov11.tt import ttnn_yolov11  # depends on model which we take
    from models.experimental.yolov11.reference import yolov11  # depends on model which we take
    from models.experimental.yolov11.tt.model_preprocessing import (
        create_yolov11_input_tensors,
        create_yolov11_model_parameters,
    )

    try:
        sys.modules["ultralytics"] = yolov11
        sys.modules["ultralytics.nn.tasks"] = yolov11
        sys.modules["ultralytics.nn.modules.conv"] = yolov11
        sys.modules["ultralytics.nn.modules.block"] = yolov11
        sys.modules["ultralytics.nn.modules.head"] = yolov11

    except KeyError:
        print("models.experimental.yolov11.reference.yolov11_utils not found.")

    if model_type == "torch_model":
        state_dict = attempt_load("yolov11n.pt", map_location="cpu").state_dict()
        model = yolov11.YoloV11()
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2
        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info("Inferencing using Torch Model")
    else:
        torch_input, ttnn_input = create_yolov11_input_tensors(
            device, input_channels=3, input_height=224, input_width=224
        )
        state_dict = attempt_load("yolov11n.pt", map_location="cpu").state_dict()
        torch_model = yolov11.YoloV11()
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2
        torch_model.load_state_dict(new_state_dict)
        torch_model.eval()
        parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
        model = ttnn_yolov11.YoloV11(device, parameters)
        logger.info("Inferencing using ttnn Model")

    save_dir = "models/experimental/yolov11/demo/runs"
    model_name = "YOLOv11"

    model_path = "models/experimental/yolov11/reference/yolov11n.pt"

    input_layout = ttnn.TILE_LAYOUT
    input_dtype = ttnn.bfloat8_b

    evaluation(
        device=device,
        res=res,
        model_type=model_type,
        model=model,
        parameters=None,
        input_dtype=input_dtype,
        input_layout=input_layout,
        save_dir=save_dir,
        model_name=model_name,
    )
