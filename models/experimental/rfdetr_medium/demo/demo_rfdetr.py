# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
RF-DETR Medium Detection Demo — TTNN vs PyTorch reference.

Downloads a sample COCO image, runs detection through both TTNN and
PyTorch reference pipelines, draws bounding boxes side-by-side, and
saves the result.

Usage:
    python models/experimental/rfdetr_medium/demo/demo_rfdetr.py
    python models/experimental/rfdetr_medium/demo/demo_rfdetr.py --image path/to/image.jpg
    python models/experimental/rfdetr_medium/demo/demo_rfdetr.py --score-thr 0.5
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import requests
import torch
from PIL import Image

import ttnn

from models.experimental.rfdetr_medium.common import (
    RESOLUTION,
    RFDETR_MEDIUM_L1_SMALL_SIZE,
    load_torch_model,
)

COCO_ID_TO_NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

SAMPLE_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

COLORS = np.random.RandomState(42).randint(60, 255, size=(91, 3)).tolist()


def download_image(url, cache_dir="/tmp/rfdetr_demo"):
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(fname):
        print(f"Downloading {url} ...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(fname, "wb") as f:
            f.write(r.content)
    return fname


def preprocess_image(image_path):
    """Load and preprocess image to [1, 3, 576, 576] float32 tensor."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = img.resize((RESOLUTION, RESOLUTION), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor, (orig_h, orig_w)


def draw_detections(image, detections, orig_size, title="", score_thr=0.3):
    """Draw detection boxes on the image. Returns annotated image."""
    img = image.copy()
    h, w = img.shape[:2]
    scale_x = w / RESOLUTION
    scale_y = h / RESOLUTION

    if len(detections) == 0 or len(detections[0]["boxes"]) == 0:
        cv2.putText(img, f"{title}: No detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return img

    det = detections[0]
    boxes = det["boxes"]
    scores = det["scores"]
    labels = det["labels"]

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    for i in range(len(boxes)):
        if scores[i] < score_thr:
            continue

        x1, y1, x2, y2 = boxes[i]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        label_idx = int(labels[i])
        class_name = COCO_ID_TO_NAME.get(label_idx, f"class_{label_idx}")
        color = COLORS[label_idx % len(COLORS)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{class_name}: {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img


def run_pytorch_reference(torch_model, image_tensor, score_thr=0.3):
    """Run PyTorch reference pipeline."""
    from models.experimental.rfdetr_medium.reference.rfdetr_medium import full_reference_forward

    with torch.no_grad():
        result = full_reference_forward(torch_model, image_tensor, score_thr=score_thr)
    return result["detections"]


def run_ttnn_pipeline(torch_model, image_tensor, device, score_thr=0.3):
    """Run TTNN pipeline."""
    from models.experimental.rfdetr_medium.tt.tt_rfdetr import TtRFDETR
    from models.experimental.rfdetr_medium.tt.model_preprocessing import (
        load_backbone_weights,
        load_projector_weights,
        load_decoder_weights,
        load_detection_head_weights,
    )

    backbone_params = load_backbone_weights(torch_model, device)
    projector_params = load_projector_weights(torch_model, device)
    decoder_params = load_decoder_weights(torch_model, device)
    head_params = load_detection_head_weights(torch_model, device)

    tt_model = TtRFDETR(
        device=device,
        torch_model=torch_model,
        backbone_params=backbone_params,
        projector_params=projector_params,
        decoder_params=decoder_params,
        head_params=head_params,
    )

    result = tt_model.forward(image_tensor)
    return result["detections"]


def main():
    parser = argparse.ArgumentParser(description="RF-DETR Medium Detection Demo")
    parser.add_argument(
        "--image", type=str, default=None, help="Path to input image (downloads COCO sample if not provided)"
    )
    parser.add_argument("--score-thr", type=float, default=0.3, help="Detection score threshold (default: 0.3)")
    parser.add_argument(
        "--output",
        type=str,
        default="rfdetr_demo_output.png",
        help="Output image path (default: rfdetr_demo_output.png)",
    )
    parser.add_argument("--skip-ref", action="store_true", help="Skip PyTorch reference (only run TTNN)")
    args = parser.parse_args()

    if args.image is None:
        image_path = download_image(SAMPLE_IMAGE_URL)
    else:
        image_path = args.image

    print(f"Input image: {image_path}")
    print(f"Score threshold: {args.score_thr}")

    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Error: cannot read image {image_path}")
        sys.exit(1)
    orig_h, orig_w = orig_img.shape[:2]
    print(f"Original size: {orig_w}x{orig_h}")

    image_tensor, orig_size = preprocess_image(image_path)
    print(f"Preprocessed tensor: {image_tensor.shape}")

    print("\nLoading RF-DETR Medium model...")
    torch_model = load_torch_model()

    # --- PyTorch reference ---
    if not args.skip_ref:
        print("\n--- PyTorch Reference ---")
        t0 = time.time()
        ref_detections = run_pytorch_reference(torch_model, image_tensor, args.score_thr)
        t_ref = time.time() - t0
        n_ref = len(ref_detections[0]["boxes"]) if len(ref_detections) > 0 else 0
        print(f"  Detections: {n_ref}")
        if n_ref > 0:
            for i in range(min(n_ref, 10)):
                box = ref_detections[0]["boxes"][i].cpu().numpy()
                score = ref_detections[0]["scores"][i].item()
                label = int(ref_detections[0]["labels"][i].item())
                cls_name = COCO_ID_TO_NAME.get(label, f"cls_{label}")
                print(f"    [{cls_name}] score={score:.3f} box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")
        print(f"  Time: {t_ref:.2f}s")

    # --- TTNN ---
    print("\n--- TTNN Pipeline ---")
    device = ttnn.open_device(device_id=0, l1_small_size=RFDETR_MEDIUM_L1_SMALL_SIZE)

    t0 = time.time()
    tt_detections = run_ttnn_pipeline(torch_model, image_tensor, device, args.score_thr)
    t_tt = time.time() - t0
    n_tt = len(tt_detections[0]["boxes"]) if len(tt_detections) > 0 else 0
    print(f"  Detections: {n_tt}")
    if n_tt > 0:
        for i in range(min(n_tt, 10)):
            box = tt_detections[0]["boxes"][i].cpu().numpy()
            score = tt_detections[0]["scores"][i].item()
            label = int(tt_detections[0]["labels"][i].item())
            cls_name = COCO_ID_TO_NAME.get(label, f"cls_{label}")
            print(f"    [{cls_name}] score={score:.3f} box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")
    print(f"  Time: {t_tt:.2f}s (includes weight loading)")

    ttnn.close_device(device)

    # --- Draw results ---
    print("\nDrawing results...")
    if not args.skip_ref:
        img_ref = draw_detections(
            orig_img, ref_detections, orig_size, title="PyTorch Reference", score_thr=args.score_thr
        )
    img_tt = draw_detections(orig_img, tt_detections, orig_size, title="TTNN (Tenstorrent)", score_thr=args.score_thr)

    if not args.skip_ref:
        combined = np.hstack([img_ref, img_tt])
    else:
        combined = img_tt

    cv2.imwrite(args.output, combined)
    print(f"\nOutput saved to: {args.output}")

    # --- Compare detections ---
    if not args.skip_ref and n_ref > 0 and n_tt > 0:
        print("\n--- Detection Comparison ---")
        from rfdetr.util.box_ops import box_iou

        ref_boxes = ref_detections[0]["boxes"]
        tt_boxes = tt_detections[0]["boxes"]
        if isinstance(tt_boxes, ttnn.Tensor):
            tt_boxes = ttnn.to_torch(tt_boxes).float()

        iou_matrix = box_iou(ref_boxes, tt_boxes)[0]
        max_ious, matched_idx = iou_matrix.max(dim=1)

        ref_labels = ref_detections[0]["labels"]
        tt_labels = tt_detections[0]["labels"]

        print(f"  Reference: {n_ref} detections, TTNN: {n_tt} detections")
        for i in range(n_ref):
            j = matched_idx[i].item()
            iou = max_ious[i].item()
            ref_cls = COCO_ID_TO_NAME.get(int(ref_labels[i]), "?")
            tt_cls = COCO_ID_TO_NAME.get(int(tt_labels[j]), "?")
            match_str = "MATCH" if ref_cls == tt_cls and iou > 0.5 else "MISMATCH"
            print(f"    Ref[{ref_cls}] <-> TT[{tt_cls}]  IoU={iou:.4f}  {match_str}")

        avg_iou = max_ious.mean().item()
        print(f"\n  Average IoU: {avg_iou:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
