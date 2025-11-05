# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import ttnn
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.experimental.SSD512.reference.ssd import build_ssd
from models.experimental.SSD512.tt.tt_ssd import build_ssd512
from models.experimental.SSD512.tt.layers.detect import TtDetect
from models.experimental.SSD512.reference.data.voc0712 import VOC_CLASSES
from models.experimental.SSD512.reference.data.config import voc


def load_image(image_path, size=512):
    """Load and preprocess image for SSD512 input."""
    # Load image with OpenCV (BGR format by default)
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Store original size for box scaling
    original_height, original_width = image_bgr.shape[:2]

    # Convert to RGB for visualization (original image)
    original_img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_img = Image.fromarray(original_img_rgb)

    # Resize directly to 512x512 (no aspect ratio preservation, matching original demo)
    x = cv2.resize(image_bgr, (size, size)).astype(np.float32)

    # Subtract BGR mean (SSD standard: [104, 117, 123] for BGR channels)
    x -= (104.0, 117.0, 123.0)

    # Ensure contiguous array
    x = x.astype(np.float32)

    # Convert to tensor and permute from HWC to CHW
    img_tensor = torch.from_numpy(x).permute(2, 0, 1)  # HWC -> CHW

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    # Store original dimensions for later box scaling
    original_img.original_size = (original_width, original_height)

    return img_tensor, original_img


def draw_detections(image, detections, output_path, model_name):
    """Draw bounding boxes and labels on image."""
    draw = ImageDraw.Draw(image)

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    if len(detections) > 0:
        det = detections[0]  # Get first image detections
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]

        if hasattr(image, "original_size"):
            img_width, img_height = image.original_size
        else:
            img_width, img_height = image.size
        scale = torch.tensor([img_width, img_height, img_width, img_height], dtype=torch.float32)

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            box_scaled = (box * scale).cpu()
            x1 = box_scaled[0].item()
            y1 = box_scaled[1].item()
            x2 = box_scaled[2].item()
            y2 = box_scaled[3].item()

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))

            if x1 >= x2 or y1 >= y2:
                continue

            class_idx = label.item()
            if class_idx < len(VOC_CLASSES):
                class_name = VOC_CLASSES[class_idx]
            else:
                class_name = f"Class {class_idx}"

            color = colors[class_idx % len(colors)]

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw label with background
            label_text = f"{class_name}: {score.item():.2f}"
            bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 20), label_text, fill=(255, 255, 255), font=font)

    # Add title
    title = f"{model_name} Detections"
    bbox = draw.textbbox((10, 10), title, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0, 180))
    draw.text((10, 10), title, fill=(255, 255, 255), font=font)

    # Save image
    image.save(output_path)
    logger.info(f"Saved {model_name} output to: {output_path}")


def generate_prior_boxes(cfg, device=None):
    """Generate prior boxes for SSD512."""
    from models.experimental.SSD512.reference.layers.functions.prior_box import PriorBox

    # Use PyTorch PriorBox for simplicity
    prior_box = PriorBox(cfg)
    priors = prior_box.forward()

    # Ensure it's a torch tensor
    if not isinstance(priors, torch.Tensor):
        priors = torch.tensor(priors)

    return priors


def run_pytorch_detection(model, image_tensor, priors, conf_thresh=0.01, nms_thresh=0.45, top_k=200):
    """Run PyTorch model and get detections."""
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

        batch_size = output.shape[0]
        detections = []

        for b in range(batch_size):
            boxes_list = []
            scores_list = []
            labels_list = []

            for cls_idx in range(1, model.num_classes):
                cls_output = output[b, cls_idx]  # [top_k, 5]
                j = 0
                while j < top_k and cls_output[j, 0] >= conf_thresh:
                    score = cls_output[j, 0].item()
                    box = cls_output[j, 1:5]  # [x1, y1, x2, y2] in [0,1] normalized

                    box = torch.clamp(box, 0.0, 1.0)

                    boxes_list.append(box.unsqueeze(0))
                    scores_list.append(score)
                    labels_list.append(cls_idx)
                    j += 1

            if len(boxes_list) > 0:
                detections.append(
                    {
                        "boxes": torch.cat(boxes_list, 0),
                        "scores": torch.tensor(scores_list),
                        "labels": torch.tensor(labels_list, dtype=torch.long),
                    }
                )
            else:
                detections.append(
                    {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}
                )

        return detections


def run_ttnn_detection(model, image_tensor, priors, device, conf_thresh=0.01, nms_thresh=0.45, top_k=200):
    """Run TTNN model and get detections."""
    # Synchronize and clear memory before forward pass
    if device is not None:
        ttnn.synchronize_device(device)
        import gc

        gc.collect()
        ttnn.synchronize_device(device)

    # Forward pass
    loc, conf = model.forward(image_tensor, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Convert to torch tensors
    loc_torch = loc.float()
    conf_torch = conf.float()

    # Reshape to [batch, num_priors, 4] and [batch, num_priors, num_classes]
    batch_size = 1
    num_priors = loc_torch.shape[1] // 4
    loc_torch = loc_torch.view(batch_size, num_priors, 4)
    conf_torch = conf_torch.view(batch_size, num_priors, model.num_classes)

    # Apply softmax to confidence
    conf_torch = F.softmax(conf_torch, dim=-1)

    # Convert priors to TTNN tensor
    priors_ttnn = ttnn.from_torch(priors, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Convert loc and conf to TTNN tensors
    loc_ttnn = ttnn.from_torch(loc_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    conf_ttnn = ttnn.from_torch(conf_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Create detect layer
    detect = TtDetect(
        num_classes=model.num_classes, top_k=top_k, conf_thresh=conf_thresh, nms_thresh=nms_thresh, device=device
    )

    # Get detections
    detections = detect(loc_ttnn, conf_ttnn, priors_ttnn)

    # Cleanup TTNN tensors
    if device is not None:
        ttnn.deallocate(loc_ttnn)
        ttnn.deallocate(conf_ttnn)
        ttnn.deallocate(priors_ttnn)
        ttnn.synchronize_device(device)
        import gc

        gc.collect()

    return detections


def main():
    parser = argparse.ArgumentParser(description="SSD512 Demo with Real Images")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/experimental/SSD512/ssd512_outputs",
        help="Directory to save output images (default: ./models/experimental/SSD512/ssd512_outputs)",
    )
    parser.add_argument(
        "--conf_thresh", type=float, default=0.01, help="Confidence threshold for detections (default: 0.01)"
    )
    parser.add_argument("--nms_thresh", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--top_k", type=int, default=200, help="Top K detections per class (default: 200)")
    parser.add_argument("--device_id", type=int, default=0, help="TTNN device ID (default: 0)")
    parser.add_argument(
        "--restart_device",
        action="store_true",
        help="Restart device between images to free all memory (slower but avoids OOM)",
    )
    parser.add_argument(
        "--l1_small_size", type=int, default=None, help="L1 small size in bytes (default: device default)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get VOC configuration
    voc_cfg = voc["SSD512"]

    # Build PyTorch model (only need to do this once)
    logger.info("Building PyTorch model...")
    torch_model = build_ssd("test", size=512, num_classes=21)
    torch_model.eval()

    # Initialize with random weights (xavier uniform)
    logger.info("Initializing random weights...")
    for m in torch_model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    # Initialize device and model (will be reused unless --restart_device is set)
    device = None
    ttnn_model = None

    def init_device_and_model():
        """Initialize device and model. Can be called multiple times if restarting device."""
        nonlocal device, ttnn_model

        if device is not None:
            logger.info("Closing existing device...")
            ttnn.close_device(device)
            import gc

            gc.collect()

        # Initialize TTNN device
        logger.info("Initializing TTNN device...")
        if args.l1_small_size is not None:
            logger.info(f"  Using l1_small_size={args.l1_small_size} bytes")
            device = ttnn.open_device(device_id=args.device_id, l1_small_size=args.l1_small_size)
        else:
            device = ttnn.open_device(device_id=args.device_id)

        # Build TTNN model
        logger.info("Building TTNN model...")
        ttnn_model = build_ssd512(num_classes=21, device=device)

        # Load weights from PyTorch to TTNN (same random weights)
        # Use device=None (host) for weights to avoid OOM - conv2d will prepare them correctly
        logger.info("Loading weights from PyTorch to TTNN...")
        logger.info("  Using device=None (host) for weights to avoid L1 memory issues...")
        ttnn_model.load_weights_from_torch(torch_model, weight_device=None)

        # Synchronize device after weight loading to ensure weights are ready
        ttnn.synchronize_device(device)
        import gc

        gc.collect()
        ttnn.synchronize_device(device)

        return device, ttnn_model

    # Initialize device and model once
    device, ttnn_model = init_device_and_model()

    try:
        # Generate prior boxes
        logger.info("Generating prior boxes...")
        priors_torch = generate_prior_boxes(voc_cfg, device=device)

        # Get image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(args.input_dir).glob(f"*{ext}"))
            image_files.extend(Path(args.input_dir).glob(f"*{ext.upper()}"))

        if len(image_files) == 0:
            logger.error(f"No images found in {args.input_dir}")
            return

        logger.info(f"Found {len(image_files)} images to process")

        # Process each image
        for img_idx, img_path in enumerate(image_files):
            logger.info(f"\nProcessing image {img_idx+1}/{len(image_files)}: {img_path.name}")

            # Restart device if requested (frees all memory including L1)
            if args.restart_device and img_idx > 0:
                logger.info("  Restarting device to free all memory...")
                device, ttnn_model = init_device_and_model()

            # Load and preprocess image
            image_tensor, original_img = load_image(str(img_path))

            # Run PyTorch detection
            logger.info("Running PyTorch model...")
            torch_detections = run_pytorch_detection(
                torch_model,
                image_tensor,
                priors_torch,
                conf_thresh=args.conf_thresh,
                nms_thresh=args.nms_thresh,
                top_k=args.top_k,
            )

            if len(torch_detections) > 0:
                det = torch_detections[0]
                boxes = det["boxes"]
                scores = det["scores"]
                labels = det["labels"]

                class_counts = {}
                train_detections = []
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    class_idx = label.item()
                    class_name = VOC_CLASSES[class_idx] if class_idx < len(VOC_CLASSES) else f"Class {class_idx}"
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    if class_name == "train":
                        train_detections.append((i, score.item(), box))

            base_name = img_path.stem
            torch_output_path = os.path.join(args.output_dir, f"{base_name}_pytorch.jpg")
            display_thresh = 0.6
            train_thresh = 0.01
            max_detections_per_class = 5
            torch_detections_display = []
            if len(torch_detections) > 0:
                det = torch_detections[0]
                train_class_idx = VOC_CLASSES.index("train") if "train" in VOC_CLASSES else None

                mask = det["scores"] >= display_thresh
                if train_class_idx is not None:
                    train_mask = (det["labels"] == train_class_idx) & (det["scores"] >= train_thresh)
                    mask = mask | train_mask

                if mask.any():
                    filtered_boxes = det["boxes"][mask]
                    filtered_scores = det["scores"][mask]
                    filtered_labels = det["labels"][mask]

                    from torchvision.ops import nms

                    unique_labels = filtered_labels.unique()
                    final_boxes = []
                    final_scores = []
                    final_labels = []

                    for label in unique_labels:
                        label_mask = filtered_labels == label
                        if not label_mask.any():
                            continue
                        label_boxes = filtered_boxes[label_mask]
                        label_scores = filtered_scores[label_mask]

                        keep = nms(label_boxes, label_scores, 0.4)

                        if label.item() == train_class_idx:
                            max_for_class = max(max_detections_per_class, len(keep))
                        else:
                            max_for_class = max_detections_per_class

                        if len(keep) > max_for_class:
                            # Sort by score and keep top-K
                            sorted_indices = torch.argsort(label_scores[keep], descending=True)
                            keep = keep[sorted_indices[:max_for_class]]

                        if len(keep) > 0:
                            final_boxes.append(label_boxes[keep])
                            final_scores.append(label_scores[keep])
                            final_labels.append(torch.full((len(keep),), label.item(), dtype=torch.long))

                    if len(final_boxes) > 0:
                        torch_detections_display.append(
                            {
                                "boxes": torch.cat(final_boxes, 0),
                                "scores": torch.cat(final_scores, 0),
                                "labels": torch.cat(final_labels, 0),
                            }
                        )
                    else:
                        torch_detections_display.append(
                            {
                                "boxes": torch.zeros((0, 4)),
                                "scores": torch.zeros(0),
                                "labels": torch.zeros(0, dtype=torch.long),
                            }
                        )
                else:
                    torch_detections_display.append(
                        {
                            "boxes": torch.zeros((0, 4)),
                            "scores": torch.zeros(0),
                            "labels": torch.zeros(0, dtype=torch.long),
                        }
                    )
            else:
                torch_detections_display = torch_detections

            draw_detections(original_img.copy(), torch_detections_display, torch_output_path, "PyTorch")
            logger.info(f"  Saved PyTorch output to: {torch_output_path}")
            logger.info(
                f"  Total detections: {len(torch_detections[0]['boxes']) if len(torch_detections) > 0 else 0}, "
                f"Displayed (>= {display_thresh}): {len(torch_detections_display[0]['boxes']) if len(torch_detections_display) > 0 else 0}"
            )

            # Run TTNN detection
            logger.info("Running TTNN model...")
            ttnn.synchronize_device(device)
            import gc

            gc.collect()
            # Try to deallocate all buffers to free L1 memory
            try:
                # This may not be available in all TTNN versions, so wrap in try-except
                if hasattr(ttnn, "deallocate_buffers"):
                    ttnn.deallocate_buffers(device)
            except Exception as e:
                logger.debug(f"  Could not deallocate buffers: {e}")

            # Additional synchronization to ensure all operations complete
            ttnn.synchronize_device(device)
            gc.collect()
            ttnn.synchronize_device(device)

            # NOTE: OOM may occur with real images due to L1 memory constraints during weight preparation
            # This is a known limitation - the device may need more L1 memory or weights need different handling
            try:
                ttnn_detections = run_ttnn_detection(
                    ttnn_model,
                    image_tensor,
                    priors_torch,
                    device,
                    conf_thresh=args.conf_thresh,
                    nms_thresh=args.nms_thresh,
                    top_k=args.top_k,
                )
            except RuntimeError as e:
                if "Out of Memory" in str(e) or "L1" in str(e):
                    logger.error(f"OOM error on image {img_path.name}.")
                    continue
                else:
                    raise

            # Synchronize after TTNN detection to free memory
            ttnn.synchronize_device(device)
            gc.collect()
            ttnn.synchronize_device(device)

            # Save TTNN output
            ttnn_output_path = os.path.join(args.output_dir, f"{base_name}_ttnn.jpg")
            draw_detections(original_img.copy(), ttnn_detections, ttnn_output_path, "TTNN")
            logger.info(f"  Saved TTNN output to: {ttnn_output_path}")

            # Log detection counts
            if len(torch_detections) > 0:
                torch_count = len(torch_detections[0]["boxes"])
                logger.info(f"  PyTorch detections: {torch_count}")
            else:
                logger.info("  PyTorch detections: 0")

            if len(ttnn_detections) > 0:
                ttnn_count = len(ttnn_detections[0]["boxes"])
                logger.info(f"  TTNN detections: {ttnn_count}")
            else:
                logger.info("  TTNN detections: 0")

        logger.info(f"\n✓ All images processed. Outputs saved to: {args.output_dir}")

    finally:
        # Close device
        logger.info("Closing TTNN device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
