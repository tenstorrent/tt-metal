# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
RF-DETR Medium detection demo.

Runs TTNN and PyTorch side-by-side on sample COCO images,
draws detection boxes, and saves comparison images.

Usage:
    python models/experimental/rfdetr_medium/demo/demo.py
    python models/experimental/rfdetr_medium/demo/demo.py --image path/to/image.jpg
    python models/experimental/rfdetr_medium/demo/demo.py --score-thr 0.2
"""

import argparse
import os
import time
from pathlib import Path

import torch

from models.experimental.rfdetr_medium.common import (
    RESOLUTION,
    RFDETR_MEDIUM_L1_SMALL_SIZE,
    load_torch_model,
)

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


def download_sample_images(output_dir):
    """Download sample COCO images for demo."""
    import urllib.request

    urls = [
        ("https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg", "cats_remotes.jpg"),
        ("https://farm1.staticflickr.com/141/321965803_40bd4b1cef_z.jpg", "dog_beach.jpg"),
        ("https://farm5.staticflickr.com/4028/4679072348_48c7abce72_z.jpg", "street_scene.jpg"),
    ]

    images = []
    for url, filename in urls:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                continue
        images.append(filepath)

    return images


def preprocess_image(image_path, resolution=RESOLUTION):
    """Load and preprocess image for RF-DETR."""
    from PIL import Image
    from torchvision import transforms

    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)

    transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor = transform(image).unsqueeze(0)
    return tensor, original_size, image


def draw_detections(image, detections, class_names, title=""):
    """Draw detection boxes on image using PIL."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    colors = [
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#FF00FF",
        "#00FFFF",
        "#FF8000",
        "#8000FF",
        "#0080FF",
        "#FF0080",
        "#80FF00",
        "#00FF80",
    ]

    for i, det in enumerate(detections):
        boxes = det.get("boxes", torch.tensor([]))
        scores = det.get("scores", torch.tensor([]))
        labels = det.get("labels", torch.tensor([]))

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()

        for j in range(len(boxes)):
            box = boxes[j]
            score = scores[j]
            label = int(labels[j])

            color = colors[label % len(colors)]
            x1, y1, x2, y2 = box

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            class_name = class_names[label] if label < len(class_names) else f"cls_{label}"
            text = f"{class_name}: {score:.2f}"
            draw.text((x1, y1 - 16), text, fill=color, font=font)

    if title:
        draw.text((10, 10), title, fill="white", font=font)

    return image


def run_pytorch_reference(torch_model, image_tensor, score_thr=0.3):
    """Run PyTorch reference and time it."""
    from models.experimental.rfdetr_medium.reference.rfdetr_medium import full_reference_forward

    start = time.time()
    with torch.no_grad():
        result = full_reference_forward(torch_model, image_tensor, score_thr)
    elapsed = time.time() - start
    return result, elapsed


def run_ttnn_model(torch_model, image_tensor, device, score_thr=0.3):
    """Run TTNN model and time it."""
    from models.experimental.rfdetr_medium.tt.tt_rfdetr import TtRFDETR
    from models.experimental.rfdetr_medium.tt.model_preprocessing import (
        load_backbone_weights,
        load_decoder_weights,
        load_detection_head_weights,
    )

    backbone_params = load_backbone_weights(torch_model, device)
    decoder_params = load_decoder_weights(torch_model, device)
    head_params = load_detection_head_weights(torch_model, device)

    tt_model = TtRFDETR(
        device=device,
        torch_model=torch_model,
        backbone_params=backbone_params,
        decoder_params=decoder_params,
        head_params=head_params,
    )

    start = time.time()
    result = tt_model.forward(image_tensor)
    elapsed = time.time() - start
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description="RF-DETR Medium Detection Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold")
    parser.add_argument("--output-dir", type=str, default="models/experimental/rfdetr_medium/demo", help="Output dir")
    parser.add_argument("--skip-ttnn", action="store_true", help="Only run PyTorch reference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading RF-DETR Medium model...")
    torch_model = load_torch_model()
    print("Model loaded.")

    # Get images
    if args.image:
        image_paths = [args.image]
    else:
        print("Downloading sample images...")
        image_paths = download_sample_images(args.output_dir)

    if not image_paths:
        print("No images available. Please provide --image argument.")
        return

    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        image_tensor, original_size, pil_image = preprocess_image(image_path)

        # PyTorch reference
        ref_result, ref_time = run_pytorch_reference(torch_model, image_tensor, args.score_thr)
        ref_dets = ref_result["detections"]
        n_ref = sum(len(d["boxes"]) for d in ref_dets)
        print(f"  PyTorch: {n_ref} detections in {ref_time:.3f}s")

        # Draw PyTorch results
        ref_image = pil_image.copy().resize((RESOLUTION, RESOLUTION))
        ref_image = draw_detections(ref_image, ref_dets, COCO_CLASSES, "PyTorch Reference")
        ref_path = os.path.join(args.output_dir, f"ref_{Path(image_path).stem}.jpg")
        ref_image.save(ref_path)
        print(f"  Saved: {ref_path}")

        if not args.skip_ttnn:
            try:
                import ttnn

                device = ttnn.open_device(device_id=0, l1_small_size=RFDETR_MEDIUM_L1_SMALL_SIZE)

                tt_result, tt_time = run_ttnn_model(torch_model, image_tensor, device, args.score_thr)
                tt_dets = tt_result["detections"]
                n_tt = sum(len(d["boxes"]) for d in tt_dets)
                print(f"  TTNN:    {n_tt} detections in {tt_time:.3f}s")

                # Draw TTNN results
                tt_image = pil_image.copy().resize((RESOLUTION, RESOLUTION))
                tt_image = draw_detections(tt_image, tt_dets, COCO_CLASSES, "TTNN")
                tt_path = os.path.join(args.output_dir, f"ttnn_{Path(image_path).stem}.jpg")
                tt_image.save(tt_path)
                print(f"  Saved: {tt_path}")

                ttnn.close_device(device)
            except Exception as e:
                print(f"  TTNN inference failed: {e}")
                print("  Run with --skip-ttnn to only test PyTorch reference.")


if __name__ == "__main__":
    main()
