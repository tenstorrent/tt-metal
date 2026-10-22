# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO26 Object Detection Demo.

Demonstrates inference on Tenstorrent hardware with visualization.
"""

import argparse
import torch
import ttnn
import cv2
import numpy as np
from loguru import logger

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
    "backpack",
    "umbrella",
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


STRIDES = [8, 16, 32]  # P3, P4, P5


def load_image(image_path: str, input_size: int = 640) -> tuple:
    """
    Load and letterbox an image for YOLO26.

    Args:
        image_path: Path to input image
        input_size: Model input size

    Returns:
        Tuple of (NHWC float tensor in [0, 1], RGB, original BGR image, scale_info)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_shape = image.shape[:2]  # (height, width)

    # Resize maintaining aspect ratio, pad with 114 (Ultralytics LetterBox)
    scale = min(input_size / original_shape[0], input_size / original_shape[1])
    new_h, new_w = round(original_shape[0] * scale), round(original_shape[1] * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_h = (input_size - new_h) // 2
    pad_w = (input_size - new_w) // 2
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    # BGR -> RGB, NHWC, [0, 1]
    tensor = torch.from_numpy(np.ascontiguousarray(padded[..., ::-1])).float() / 255.0
    tensor = tensor.unsqueeze(0)

    scale_info = {
        "original_shape": original_shape,
        "scale": scale,
        "pad_h": pad_h,
        "pad_w": pad_w,
    }

    return tensor, image, scale_info


def tt_outputs_to_torch(outputs: dict, batch_size: int, num_classes: int) -> tuple:
    """
    Convert TtYOLO26 outputs to torch.

    Returns:
        Tuple of (boxes [B, 4, A], scores [B, nc, A]) concatenated over P3/P4/P5 in Ultralytics anchor order,
        plus the per-scale (h, w)
    """
    boxes, scores, shapes = [], [], []
    for (box_t, h, w), (cls_t, _, _) in zip(outputs["boxes"], outputs["scores"]):
        per_scale = []
        for t, ch in ((box_t, 4), (cls_t, num_classes)):
            if t.memory_config().is_sharded():
                t = ttnn.sharded_to_interleaved(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            t = ttnn.to_torch(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)).float()
            per_scale.append(t.reshape(batch_size, h * w, -1)[..., :ch].permute(0, 2, 1))
        boxes.append(per_scale[0])
        scores.append(per_scale[1])
        shapes.append((h, w))
    return torch.cat(boxes, dim=-1), torch.cat(scores, dim=-1), shapes


def postprocess_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    shapes: list,
    conf_threshold: float = 0.25,
    max_det: int = 300,
) -> list:
    """
    NMS-free YOLO26 (one2one head) post-processing, as Ultralytics Detect with end2end=True.

    reg_max=1, so the box branch predicts (left, top, right, bottom) distances from the cell centre, in units of the
    stride. Class scores are logits.

    Args:
        boxes: [B, 4, A] ltrb distances
        scores: [B, nc, A] class logits
        shapes: per-scale (h, w), in the same order as STRIDES
        conf_threshold: Confidence threshold
        max_det: Maximum detections per image

    Returns:
        List (per image) of detections [(x1, y1, x2, y2, confidence, class_id), ...] in input-image pixels
    """
    anchors, strides = [], []
    for (h, w), stride in zip(shapes, STRIDES):
        sy, sx = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
        anchors.append(torch.stack((sx, sy), -1).view(-1, 2))
        strides.append(torch.full((h * w, 1), float(stride)))
    anchors = torch.cat(anchors).T  # [2, A]
    strides = torch.cat(strides).T  # [1, A]

    lt, rb = boxes.chunk(2, dim=1)
    xyxy = torch.cat((anchors - lt, anchors + rb), dim=1) * strides  # [B, 4, A]
    probs = scores.sigmoid()

    results = []
    for b in range(boxes.shape[0]):
        p = probs[b].T  # [A, nc]
        k = min(max_det, p.shape[0])
        # top-k anchors by best class, then top-k (anchor, class) pairs among them
        top_anchor = p.max(dim=-1).values.topk(k).indices
        conf, flat = p[top_anchor].flatten().topk(k)
        anchor_idx = top_anchor[flat // p.shape[1]]
        class_idx = flat % p.shape[1]
        keep = conf > conf_threshold
        dets = [
            [*xyxy[b, :, a].tolist(), c.item(), cls.item()]
            for a, c, cls in zip(anchor_idx[keep], conf[keep], class_idx[keep])
        ]
        results.append(dets)
    return results


def to_original_coords(det: list, scale_info: dict) -> list:
    """Undo letterbox: input-image pixels -> original-image pixels, clipped."""
    x1, y1, x2, y2, score, class_id = det
    h, w = scale_info["original_shape"]
    s, ph, pw = scale_info["scale"], scale_info["pad_h"], scale_info["pad_w"]
    x1, x2 = [min(max((v - pw) / s, 0), w) for v in (x1, x2)]
    y1, y2 = [min(max((v - ph) / s, 0), h) for v in (y1, y2)]
    return [x1, y1, x2, y2, score, int(class_id)]


def class_color(class_id: int) -> tuple:
    palette = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72)]
    return palette[class_id % len(palette)]


def draw_detections(image: np.ndarray, detections: list, title: str = "") -> np.ndarray:
    """
    Draw detection boxes (original-image coordinates) on a copy of the image.
    """
    output_image = image.copy()
    for x1, y1, x2, y2, score, class_id in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = class_color(class_id)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)

        name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class {class_id}"
        label = f"{name} {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        y_text = max(y1, label_h + 8)
        cv2.rectangle(output_image, (x1, y_text - label_h - 8), (x1 + label_w + 4, y_text), color, -1)
        cv2.putText(output_image, label, (x1 + 2, y_text - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if title:
        cv2.rectangle(output_image, (0, 0), (output_image.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(output_image, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return output_image


def run_reference(variant: str, image_path: str, conf_threshold: float, scale_info: dict, input_tensor) -> list:
    """Ultralytics reference on the same letterboxed input, one2one head, NMS-free (end2end=True)."""
    from ultralytics import YOLO

    model = YOLO(f"{variant}.pt").model.eval().float()
    detect = model.model[-1]
    detect.end2end = True
    with torch.no_grad():
        preds = model(input_tensor.permute(0, 3, 1, 2))
    preds = preds[0] if isinstance(preds, (tuple, list)) else preds  # [B, max_det, 6]: x1 y1 x2 y2 score cls
    dets = [p.tolist() for p in preds[0] if p[4] > conf_threshold]
    return [to_original_coords(d, scale_info) for d in dets]


def iou(a: list, b: list) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare_detections(tt_dets: list, ref_dets: list, iou_threshold: float = 0.7) -> tuple:
    """Greedy one-to-one match on class + IoU. Returns (matched pairs, unmatched tt, unmatched ref)."""
    unmatched_ref = list(range(len(ref_dets)))
    matched, unmatched_tt = [], []
    for i, t in sorted(enumerate(tt_dets), key=lambda x: -x[1][4]):
        best, best_iou = None, iou_threshold
        for j in unmatched_ref:
            if ref_dets[j][5] == t[5]:
                v = iou(t, ref_dets[j])
                if v >= best_iou:
                    best, best_iou = j, v
        if best is None:
            unmatched_tt.append(i)
        else:
            unmatched_ref.remove(best)
            matched.append((i, best, best_iou))
    return matched, unmatched_tt, unmatched_ref


def main():
    parser = argparse.ArgumentParser(description="YOLO26 Object Detection Demo")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image path")
    parser.add_argument("--output", "-o", type=str, default="output.jpg", help="Output image path")
    parser.add_argument("--variant", type=str, default="yolo26n", help="Model variant (yolo26n, yolo26s)")
    parser.add_argument("--input-size", type=int, default=640, help="Input size")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run the Ultralytics reference (NMS-free, one2one) and save a side-by-side image",
    )
    args = parser.parse_args()

    from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtYOLO26

    logger.info(f"Running YOLO26 demo: {args.variant}, input {args.input}, size {args.input_size}")

    input_tensor, original_image, scale_info = load_image(args.input, args.input_size)
    logger.info(f"Loaded image: {original_image.shape}")

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=YOLO26_L1_SMALL_SIZE)
    try:
        # Raw uint8 image in, [1, 1, H, W * 3]: conversion and 1/255 normalization run on device
        tt_model = TtYOLO26(device, args.variant, uint8_input=True)
        tt_model.load_weights_from_ultralytics(args.variant)

        _, h, w, c = input_tensor.shape
        image_u8 = (input_tensor * 255).round().to(torch.uint8).reshape(1, 1, h, w * c)
        tt_input = ttnn.from_torch(image_u8, dtype=ttnn.uint8, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        outputs = tt_model(tt_input)
        boxes, scores, shapes = tt_outputs_to_torch(outputs, 1, tt_model.num_classes)
    finally:
        ttnn.close_device(device)

    tt_dets = postprocess_detections(boxes, scores, shapes, conf_threshold=args.conf_threshold)[0]
    tt_dets = [to_original_coords(d, scale_info) for d in tt_dets]

    logger.info(f"TTNN {args.variant}: {len(tt_dets)} detections")
    for x1, y1, x2, y2, score, class_id in tt_dets:
        logger.info(f"  {COCO_CLASSES[class_id]:12s} {score:.2f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    tt_image = draw_detections(original_image, tt_dets, f"TTNN {args.variant} (Tenstorrent)")
    cv2.imwrite(args.output, tt_image)
    logger.info(f"Saved output to: {args.output}")

    if args.compare:
        ref_dets = run_reference(args.variant, args.input, args.conf_threshold, scale_info, input_tensor)
        logger.info(f"Ultralytics {args.variant} (end2end, CPU fp32): {len(ref_dets)} detections")
        for x1, y1, x2, y2, score, class_id in ref_dets:
            logger.info(f"  {COCO_CLASSES[class_id]:12s} {score:.2f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        matched, unmatched_tt, unmatched_ref = compare_detections(tt_dets, ref_dets)
        for i, j, v in matched:
            logger.info(
                f"  match {COCO_CLASSES[tt_dets[i][5]]:12s} IoU={v:.3f} "
                f"conf TTNN={tt_dets[i][4]:.3f} ref={ref_dets[j][4]:.3f}"
            )
        for i in unmatched_tt:
            logger.warning(f"  only in TTNN: {COCO_CLASSES[tt_dets[i][5]]} {tt_dets[i][4]:.2f}")
        for j in unmatched_ref:
            logger.warning(f"  only in reference: {COCO_CLASSES[ref_dets[j][5]]} {ref_dets[j][4]:.2f}")
        logger.info(f"Matched {len(matched)}/{len(ref_dets)} reference detections (same class, IoU >= 0.7)")

        ref_image = draw_detections(original_image, ref_dets, f"Ultralytics {args.variant} (CPU reference)")
        side_by_side = np.concatenate([tt_image, ref_image], axis=1)
        compare_path = args.output.rsplit(".", 1)[0] + "_vs_reference.jpg"
        cv2.imwrite(compare_path, side_by_side)
        logger.info(f"Saved comparison to: {compare_path}")


if __name__ == "__main__":
    main()
