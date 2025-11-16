"""
COCO Validation Script for YOLOS-small
Evaluates model accuracy on COCO 2017 validation dataset
"""

import argparse
import json
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

import ttnn
from models.demos.yolos_small.demo import load_image, load_pretrained_weights, preprocess_image
from models.demos.yolos_small.reference.config import get_yolos_small_config
from models.demos.yolos_small.reference.modeling_yolos import YolosForObjectDetection as PyTorchYolos
from models.demos.yolos_small.yolos_ttnn.common import OptimizationConfig, convert_to_ttnn_tensor, get_dtype_for_stage
from models.demos.yolos_small.yolos_ttnn.modeling_yolos import YolosForObjectDetection as TtnnYolos


def convert_predictions_to_coco_format(predictions, image_id, original_size):
    """
    Convert model predictions to COCO format.

    Args:
        predictions: Model predictions dict
        image_id: COCO image ID
        original_size: Original image size (width, height)

    Returns:
        List of COCO-format detections
    """
    coco_results = []

    scores = predictions["scores"][0]
    labels = predictions["labels"][0]
    boxes = predictions["boxes"][0]
    keep = predictions["keep"][0]

    orig_w, orig_h = original_size

    for i in range(len(scores)):
        if keep[i]:
            score = scores[i].item()
            label = labels[i].item()
            box = boxes[i]  # [center_x, center_y, width, height] normalized

            # Convert to pixel coordinates
            cx = box[0].item() * orig_w
            cy = box[1].item() * orig_h
            w = box[2].item() * orig_w
            h = box[3].item() * orig_h

            # Convert to COCO format [x, y, width, height] (top-left corner)
            x1 = cx - w / 2
            y1 = cy - h / 2

            coco_results.append({"image_id": image_id, "category_id": label, "bbox": [x1, y1, w, h], "score": score})

    return coco_results


def validate_coco(
    model,
    coco_path,
    device=None,
    use_ttnn=False,
    threshold=0.05,
    max_images=None,
):
    """
    Validate model on COCO dataset.

    Args:
        model: PyTorch or TTNN model
        coco_path: Path to COCO dataset
        device: TTNN device (if using TTNN)
        use_ttnn: Whether using TTNN model
        threshold: Detection threshold
        max_images: Max number of images to evaluate (None for all)

    Returns:
        Dictionary with COCO metrics
    """
    coco_path = Path(coco_path)
    ann_file = coco_path / "annotations" / "instances_val2017.json"
    img_dir = coco_path / "val2017"

    # Load COCO annotations
    coco = COCO(str(ann_file))

    # Get image IDs
    img_ids = coco.getImgIds()
    if max_images:
        img_ids = img_ids[:max_images]

    print(f"Validating on {len(img_ids)} images...")

    # Collect predictions
    all_predictions = []

    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]

        # Load and preprocess image
        image = load_image(str(img_path))
        original_size = image.size  # (width, height)
        pixel_values = preprocess_image(image)

        # Run inference
        if use_ttnn:
            # Use dtype appropriate for the optimization stage when converting inputs.
            input_dtype = get_dtype_for_stage(model.opt_config) if hasattr(model, "opt_config") else ttnn.bfloat16
            pixel_values_ttnn = convert_to_ttnn_tensor(
                pixel_values,
                device,
                dtype=input_dtype,
            )
            predictions = model.predict(pixel_values_ttnn, threshold=threshold)
        else:
            with torch.no_grad():
                predictions = model.predict(pixel_values, threshold=threshold)

        # Convert to COCO format
        coco_preds = convert_predictions_to_coco_format(predictions, img_id, original_size)
        all_predictions.extend(coco_preds)

    # Save predictions
    pred_file = "coco_predictions.json"
    with open(pred_file, "w") as f:
        json.dump(all_predictions, f)

    print(f"Saved predictions to {pred_file}")

    # Evaluate using COCO API
    if len(all_predictions) > 0:
        coco_dt = coco.loadRes(pred_file)
        coco_eval = COCOeval(coco, coco_dt, "bbox")
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            "AP": coco_eval.stats[0],
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
            "APs": coco_eval.stats[3],
            "APm": coco_eval.stats[4],
            "APl": coco_eval.stats[5],
        }

        return metrics
    else:
        print("No predictions to evaluate!")
        return None


def main():
    parser = argparse.ArgumentParser(description="YOLOS COCO Validation")
    parser.add_argument("--coco-path", type=str, required=True, help="Path to COCO dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3], help="Optimization stage for TTNN")
    parser.add_argument("--threshold", type=float, default=0.05, help="Detection threshold")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to evaluate (None for all)")
    parser.add_argument("--pytorch-only", action="store_true", help="Evaluate PyTorch only")
    parser.add_argument("--device-id", type=int, default=0, help="Tenstorrent device ID")

    args = parser.parse_args()

    print("=" * 80)
    print("YOLOS-small COCO Validation")
    print("=" * 80)

    # Load config
    config = get_yolos_small_config()

    # Load PyTorch model
    print("\nLoading PyTorch model...")
    pytorch_model = PyTorchYolos(config)
    pytorch_model = load_pretrained_weights(pytorch_model, args.checkpoint)
    pytorch_model.eval()

    # Validate PyTorch
    print("\n" + "=" * 80)
    print("Validating PyTorch Model")
    print("=" * 80)

    pytorch_metrics = validate_coco(
        pytorch_model,
        args.coco_path,
        threshold=args.threshold,
        max_images=args.max_images,
        use_ttnn=False,
    )

    if pytorch_metrics:
        print("\nPyTorch Results:")
        print(f"  AP: {pytorch_metrics['AP']:.3f}")
        print(f"  AP50: {pytorch_metrics['AP50']:.3f}")
        print(f"  AP75: {pytorch_metrics['AP75']:.3f}")

    if args.pytorch_only:
        return

    # Validate TTNN
    print("\n" + "=" * 80)
    print(f"Validating TTNN Model - Stage {args.stage}")
    print("=" * 80)

    # Initialize device
    device = ttnn.open_device(device_id=args.device_id)

    # Get optimization config
    if args.stage == 1:
        opt_config = OptimizationConfig.stage1()
    elif args.stage == 2:
        opt_config = OptimizationConfig.stage2()
    else:
        opt_config = OptimizationConfig.stage3()

    # Create TTNN model
    ttnn_model = TtnnYolos(
        config=config,
        device=device,
        reference_model=pytorch_model,
        opt_config=opt_config,
    )

    ttnn_metrics = validate_coco(
        ttnn_model,
        args.coco_path,
        device=device,
        threshold=args.threshold,
        max_images=args.max_images,
        use_ttnn=True,
    )

    if ttnn_metrics:
        print(f"\nTTNN Stage {args.stage} Results:")
        print(f"  AP: {ttnn_metrics['AP']:.3f}")
        print(f"  AP50: {ttnn_metrics['AP50']:.3f}")
        print(f"  AP75: {ttnn_metrics['AP75']:.3f}")

        # Compare with PyTorch
        if pytorch_metrics:
            print(f"\nDifference from PyTorch:")
            print(f"  ΔAP: {ttnn_metrics['AP'] - pytorch_metrics['AP']:.3f}")
            print(f"  ΔAP50: {ttnn_metrics['AP50'] - pytorch_metrics['AP50']:.3f}")

    # Save results
    results = {
        "pytorch": pytorch_metrics,
        "ttnn": ttnn_metrics,
        "stage": args.stage,
        "threshold": args.threshold,
        "num_images": args.max_images or "all",
    }

    results_file = f"validation_results_stage{args.stage}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {results_file}")

    # Cleanup
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
"""
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""
