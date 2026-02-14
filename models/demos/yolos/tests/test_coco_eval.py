# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""COCO validation script for YOLOS-small TTNN implementation."""

import os
import sys
import torch
import json
from pathlib import Path
from PIL import Image
import numpy as np

# COCO evaluation tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# HuggingFace
import transformers
from transformers import YolosForObjectDetection, YolosImageProcessor

# TTNN
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from tt.ttnn_functional_yolos import (
    custom_preprocessor,
    yolos_for_object_detection,
)


def box_cxcywh_to_xyxy(boxes):
    """Convert center-x, center-y, width, height to x1, y1, x2, y2."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def rescale_boxes(boxes, orig_size):
    """Rescale boxes from [0,1] to image pixel coordinates."""
    img_h, img_w = orig_size
    boxes = box_cxcywh_to_xyxy(boxes)
    boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=boxes.dtype)
    return boxes


def convert_to_coco_format(image_id, scores, labels, boxes, threshold=0.0):
    """Convert predictions to COCO results format."""
    results = []
    keep = scores > threshold
    
    for score, label, box, k in zip(scores, labels, boxes, keep):
        if k:
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            results.append({
                "image_id": image_id,
                "category_id": label.item(),
                "bbox": [x1, y1, w, h],  # COCO format: [x, y, width, height]
                "score": score.item(),
            })
    
    return results


def preprocess_image_fixed_size(image, size=(512, 864)):
    """Preprocess image to fixed size with ImageNet normalization."""
    import torchvision.transforms as transforms
    
    # Resize to fixed size (H, W)
    image = image.resize((size[1], size[0]), Image.BILINEAR)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(image).unsqueeze(0)  # [1, 3, H, W]


def run_coco_evaluation(
    coco_root: str = "/workdir/coco",
    num_images: int = 100,  # Subset for faster evaluation
    threshold: float = 0.0,
    use_ttnn: bool = True,
    device_id: int = 0,
):
    """Run COCO evaluation on val2017."""
    
    print("=" * 80)
    print("YOLOS-small COCO Evaluation")
    print("=" * 80)
    
    # Paths
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
    img_dir = os.path.join(coco_root, "val2017")
    
    print(f"\nAnnotation file: {ann_file}")
    print(f"Image directory: {img_dir}")
    
    # Load COCO
    coco_gt = COCO(ann_file)
    image_ids = sorted(coco_gt.getImgIds())[:num_images]
    print(f"Evaluating on {len(image_ids)} images")
    
    # Load HuggingFace model 
    print("\nLoading HuggingFace YOLOS-small...")
    hf_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
    hf_model.eval()
    
    # Get config
    config = hf_model.config
    
    # Fixed image size (matching YOLOS-small training)
    IMG_SIZE = (512, 864)  # (H, W)
    
    # Prepare TTNN if needed
    ttnn_device = None
    parameters = None
    cls_token = None
    detection_tokens = None
    position_embeddings = None
    
    if use_ttnn:
        print(f"\nInitializing TTNN device {device_id}...")
        ttnn_device = ttnn.open_device(device_id=device_id)
        
        # Convert model to bfloat16
        model_bf16 = hf_model.to(torch.bfloat16)
        
        # Preprocess parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model_bf16,
            device=ttnn_device,
            custom_preprocessor=custom_preprocessor,
        )
        
        # Get state dict for embeddings
        state_dict = model_bf16.state_dict()
        
        # cls_token
        torch_cls_token = state_dict["vit.embeddings.cls_token"]
        cls_token = ttnn.from_torch(
            torch_cls_token,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=ttnn_device,
        )
        
        # detection_tokens
        torch_det_tokens = state_dict["vit.embeddings.detection_tokens"]
        detection_tokens = ttnn.from_torch(
            torch_det_tokens,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=ttnn_device,
        )
        
        # position_embeddings
        torch_pos_emb = state_dict["vit.embeddings.position_embeddings"]
        position_embeddings = ttnn.from_torch(
            torch_pos_emb,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=ttnn_device,
        )
        
        print("TTNN parameters loaded")
    
    # COCO category mapping
    id2label = hf_model.config.id2label
    
    # Run inference
    all_results = []
    
    print(f"\nRunning inference... (image size: {IMG_SIZE})")
    for i, img_id in enumerate(image_ids):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(image_ids)} images")
        
        # Load image
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_size = (img_info["height"], img_info["width"])
        
        # Preprocess to fixed size - produces [1, 3, H, W] tensor
        pixel_values = preprocess_image_fixed_size(image, IMG_SIZE)
        
        with torch.no_grad():
            if use_ttnn:
                # TTNN inference - need to convert NCHW to NHWC and pad to 4 channels
                pixel_values_bf16 = pixel_values.to(torch.bfloat16)
                
                # Convert NCHW to NHWC: [1, 3, 512, 864] -> [1, 512, 864, 3]
                pixel_values_nhwc = torch.permute(pixel_values_bf16, (0, 2, 3, 1))
                
                # Pad channels from 3 to 4: [1, 512, 864, 3] -> [1, 512, 864, 4]
                pixel_values_padded = torch.nn.functional.pad(
                    pixel_values_nhwc, (0, 1, 0, 0, 0, 0, 0, 0)
                )
                
                pixel_values_ttnn = ttnn.from_torch(
                    pixel_values_padded,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=ttnn_device,
                )
                
                logits_ttnn, boxes_ttnn = yolos_for_object_detection(
                    config,
                    pixel_values_ttnn,
                    cls_token,
                    detection_tokens,
                    position_embeddings,
                    attention_mask=None,
                    parameters=parameters,
                )
                
                logits = ttnn.to_torch(logits_ttnn).float()
                pred_boxes = ttnn.to_torch(boxes_ttnn).float()
            else:
                # HuggingFace inference  
                outputs = hf_model(pixel_values)
                logits = outputs.logits
                pred_boxes = outputs.pred_boxes
        
        # Post-process predictions
        probs = torch.softmax(logits[0], dim=-1)  # [100, num_classes]
        scores, labels = probs[..., :-1].max(-1)  # Exclude "no-object" class
        
        # Rescale boxes to original image coordinates
        boxes_scaled = rescale_boxes(pred_boxes[0], orig_size)
        
        # Convert labels to COCO category IDs
        coco_labels = []
        for label in labels:
            label_name = id2label.get(label.item(), None)
            if label_name:
                # Find COCO category ID by name
                for cat in coco_gt.cats.values():
                    if cat["name"] == label_name:
                        coco_labels.append(cat["id"])
                        break
                else:
                    coco_labels.append(label.item() + 1)  # Fallback
            else:
                coco_labels.append(label.item() + 1)
        coco_labels = torch.tensor(coco_labels)
        
        # Convert to COCO format
        img_results = convert_to_coco_format(
            img_id, scores, coco_labels, boxes_scaled, threshold
        )
        all_results.extend(img_results)
    
    print(f"\nTotal predictions: {len(all_results)}")
    
    # Run COCO evaluation
    if len(all_results) > 0:
        print("\nRunning COCO evaluation...")
        
        # Save results to temp file (required by COCOeval)
        results_file = "/tmp/yolos_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f)
        
        coco_dt = coco_gt.loadRes(results_file)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = image_ids
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        ap = coco_eval.stats[0]  # AP @ IoU=0.50:0.95
        ap50 = coco_eval.stats[1]  # AP @ IoU=0.50
        
        print(f"\n" + "=" * 80)
        print(f"Results Summary")
        print(f"=" * 80)
        print(f"AP @ IoU=0.50:0.95: {ap:.4f}")
        print(f"AP @ IoU=0.50:      {ap50:.4f}")
        print(f"Expected AP (HF): ~0.36-0.45")
        
        # Cleanup
        os.remove(results_file)
    else:
        print("No predictions generated!")
        ap = 0.0
    
    # Cleanup TTNN
    if ttnn_device:
        ttnn.close_device(ttnn_device)
    
    return ap


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=str, default="/workdir/coco")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--no-ttnn", action="store_true", help="Use HuggingFace only")
    parser.add_argument("--device-id", type=int, default=0)
    
    args = parser.parse_args()
    
    ap = run_coco_evaluation(
        coco_root=args.coco_root,
        num_images=args.num_images,
        threshold=args.threshold,
        use_ttnn=not args.no_ttnn,
        device_id=args.device_id,
    )
    
    print(f"\nFinal AP: {ap:.4f}")
