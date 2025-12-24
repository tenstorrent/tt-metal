# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops
import torchvision.transforms as transforms

COCO_INSTANCE_CATEGORY_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "sheep",
    "horse",
    "dog",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sport ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wineglass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hotdog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def preprocess_image(
    image_path: str, target_size: Tuple[int, int] = (512, 512)
) -> Tuple[Tensor, List[Tuple[int, int]]]:
    """Preprocess image for inference."""
    image = Image.open(image_path).convert("RGB")
    og_size = [tuple(image.size)]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, og_size


def postprocess_detections(
    head_outputs: Dict[str, List[Tensor]],
    anchors: List[List[Tensor]],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    detections_per_img: int,
    topk_candidates: int,
    box_coder,
) -> List[Dict[str, Tensor]]:
    """Postprocess model outputs to generate final detections."""
    class_logits = head_outputs["cls_logits"]
    box_regression = head_outputs["bbox_regression"]
    num_images = len(image_shapes)
    detections: List[Dict[str, Tensor]] = []

    for index in range(num_images):
        box_regression_per_image = [br[index] for br in box_regression]
        logits_per_image = [cl[index] for cl in class_logits]
        anchors_per_image, image_shape = anchors[index], image_shapes[index]

        image_boxes = []
        image_scores = []
        image_labels = []

        for box_regression_per_level, logits_per_level, anchors_per_level in zip(
            box_regression_per_image, logits_per_image, anchors_per_image
        ):
            num_classes = logits_per_level.shape[-1]

            scores_per_level = torch.sigmoid(logits_per_level).flatten()
            keep_idxs = scores_per_level > score_thresh
            scores_per_level = scores_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            num_topk = det_utils._topk_min(topk_idxs, topk_candidates, 0)
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = topk_idxs % num_classes

            boxes_per_level = box_coder.decode_single(
                box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
            )
            boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        keep = box_ops.batched_nms(image_boxes.float(), image_scores.float(), image_labels, nms_thresh)
        keep = keep[:detections_per_img]

        detections.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
            }
        )

    return detections


def visualize_detections(
    image_path: str,
    detections: List[Dict[str, Tensor]],
    output_path: str,
    target_size: Tuple[int, int] = (512, 512),
) -> None:
    """Visualize detections on the image and save the result."""
    label_map = {i: name for i, name in enumerate(COCO_INSTANCE_CATEGORY_NAMES)}

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.CenterCrop(target_size)])
    image = transform(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    dets = detections[0]
    boxes = dets["boxes"]
    scores = dets["scores"]
    labels = dets["labels"]

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        class_name = label_map.get(label.item(), str(label.item()))
        logger.info(f"Detected: {class_name} (confidence: {score:.2f})")

        text = f"{class_name}: {score:.2f}"
        draw.text((x1 + 5, y1 + 5), text, fill="yellow", font=font)

    image.save(output_path)
    logger.info(f"Saved visualization to {output_path}")


def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    """Resize bounding boxes from original size to new size."""
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
