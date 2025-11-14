# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms


# COCO class names
COCO_CLASSES = [
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
    "",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "",
    "backpack",
    "umbrella",
    "",
    "",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "",
    "wine glass",
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
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "",
    "dining table",
    "",
    "",
    "toilet",
    "",
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
    "",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    """Resize image with aspect ratio preservation and padding."""
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, width, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return (
        canvas,
        new_w,
        new_h,
        old_w,
        old_h,
        padding_w,
        padding_h,
    )


def preprocess_image(image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Preprocess image for EfficientDet inference.

    Args:
        image_path: Path to input image
        max_size: Maximum size for resizing (default: 512)
        mean: Mean values for normalization (RGB order)
        std: Std values for normalization (RGB order)

    Returns:
        ori_img: Original image (BGR format)
        framed_img: Preprocessed image tensor (NCHW format)
        framed_meta: Metadata for coordinate transformation
    """
    # Load image
    ori_img = cv2.imread(image_path)
    if ori_img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert BGR to RGB and normalize
    normalized_img = (ori_img[..., ::-1] / 255.0 - mean) / std

    # Resize with aspect ratio preservation
    img_meta = aspectaware_resize_padding(normalized_img, max_size, max_size, means=None)
    framed_img = img_meta[0]
    framed_meta = img_meta[1:]

    # Convert to tensor and add batch dimension
    framed_img = torch.from_numpy(framed_img.transpose(2, 0, 1)).float()  # HWC to CHW
    framed_img = framed_img.unsqueeze(0)  # Add batch dimension

    return ori_img, framed_img, framed_meta


def postprocess(
    regression, classification, anchors, regressBoxes, clipBoxes, input_tensor, threshold=0.5, iou_threshold=0.5
):
    """
    Post-process model outputs to get bounding boxes.

    Args:
        regression: Regression outputs from model
        classification: Classification outputs from model
        anchors: Anchor boxes
        regressBoxes: BBoxTransform module
        clipBoxes: ClipBoxes module
        input_tensor: Input image tensor (for getting image dimensions)
        threshold: Score threshold for detections
        iou_threshold: IoU threshold for NMS

    Returns:
        List of detection dictionaries with 'rois', 'class_ids', and 'scores'
    """
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, input_tensor)

    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]

    out = []
    for i in range(regression.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append(
                {
                    "rois": np.array(()),
                    "class_ids": np.array(()),
                    "scores": np.array(()),
                }
            )
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append(
                {
                    "rois": boxes_.cpu().numpy(),
                    "class_ids": classes_.cpu().numpy(),
                    "scores": scores_.cpu().numpy(),
                }
            )
        else:
            out.append(
                {
                    "rois": np.array(()),
                    "class_ids": np.array(()),
                    "scores": np.array(()),
                }
            )

    return out


def invert_affine(metas, preds):
    """Transform bounding boxes back to original image coordinates."""
    for i in range(len(preds)):
        if len(preds[i]["rois"]) == 0:
            continue
        else:
            if isinstance(metas, float):
                preds[i]["rois"][:, [0, 2]] = preds[i]["rois"][:, [0, 2]] / metas
                preds[i]["rois"][:, [1, 3]] = preds[i]["rois"][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]["rois"][:, [0, 2]] = preds[i]["rois"][:, [0, 2]] / (new_w / old_w)
                preds[i]["rois"][:, [1, 3]] = preds[i]["rois"][:, [1, 3]] / (new_h / old_h)
    return preds


def draw_bounding_boxes(image, preds, class_names, color=(0, 255, 0), label_color=(255, 255, 255)):
    """
    Draw bounding boxes on image.

    Args:
        image: Input image (BGR format)
        preds: List of prediction dictionaries with 'rois', 'class_ids', 'scores'
        class_names: List of class names
        color: Bounding box color (BGR format)
        label_color: Label text color (BGR format)

    Returns:
        Image with bounding boxes drawn
    """
    img = image.copy()

    for pred in preds:
        if len(pred["rois"]) == 0:
            continue

        for j in range(len(pred["rois"])):
            x1, y1, x2, y2 = pred["rois"][j].astype(int)
            class_id = int(pred["class_ids"][j])
            score = float(pred["scores"][j])

            # Clamp coordinates to image boundaries
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            # Get class name
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            label = f"{class_name} {score:.2f}"

            # Draw bounding box
            thickness = max(1, int(0.001 * (h + w) / 2))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Draw label background and text
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
            label_y = max(y1, text_height + 5)
            cv2.rectangle(
                img,
                (x1, label_y - text_height - 5),
                (x1 + text_width, label_y + baseline),
                color,
                -1,
            )
            cv2.putText(
                img,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                label_color,
                thickness,
                cv2.LINE_AA,
            )

    return img
