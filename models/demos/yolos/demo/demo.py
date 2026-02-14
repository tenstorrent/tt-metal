# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
YOLOS-small Object Detection Demo

Demonstrates object detection using TTNN implementation on Tenstorrent hardware.
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import YolosForObjectDetection

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from tt.ttnn_functional_yolos import custom_preprocessor, yolos_for_object_detection


# COCO class names
COCO_CLASSES = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def load_image(path_or_url: str, size=(512, 864)):
    """Load and resize image from path or URL."""
    if path_or_url.startswith("http"):
        response = requests.get(path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(path_or_url)
    
    image = image.convert("RGB")
    image = image.resize((size[1], size[0]), Image.BILINEAR)
    return image


def preprocess_image(image):
    """Preprocess image with ImageNet normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def visualize_detections(image, scores, labels, boxes, threshold=0.7, id2label=None):
    """Draw detections on image."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    for i, (score, label, box) in enumerate(zip(scores, labels, boxes)):
        if score < threshold:
            continue
        
        # Convert from [cx, cy, w, h] normalized to pixel coords
        cx, cy, w, h = box
        x1 = (cx - w/2) * width
        y1 = (cy - h/2) * height
        x2 = (cx + w/2) * width
        y2 = (cy + h/2) * height
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Draw label
        if id2label:
            class_name = id2label.get(label, f"class_{label}")
        elif label < len(COCO_CLASSES):
            class_name = COCO_CLASSES[label]
        else:
            class_name = f"class_{label}"
        
        text = f"{class_name}: {score:.2f}"
        draw.text((x1, max(0, y1-15)), text, fill="red")
    
    return image


def run_demo(args):
    """Run object detection demo."""
    print("=" * 70)
    print("YOLOS-small Object Detection Demo (TTNN)")
    print("=" * 70)
    
    # Load HuggingFace model
    print("\nLoading HuggingFace YOLOS-small...")
    hf_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
    hf_model.eval()
    config = hf_model.config
    id2label = config.id2label
    
    print(f"Model config: hidden={config.hidden_size}, layers={config.num_hidden_layers}")
    print(f"Image size: {config.image_size}, Patch size: {config.patch_size}")
    print(f"Detection tokens: {config.num_detection_tokens}")
    
    # Load and preprocess image
    print(f"\nLoading image: {args.image}")
    image = load_image(args.image, size=tuple(config.image_size))
    pixel_values = preprocess_image(image)
    print(f"Image shape: {pixel_values.shape}")
    
    # Run HuggingFace inference
    print("\n--- HuggingFace Reference ---")
    with torch.no_grad():
        start = time.perf_counter()
        hf_out = hf_model(pixel_values)
        hf_time = time.perf_counter() - start
    
    hf_logits = hf_out.logits[0]
    hf_boxes = hf_out.pred_boxes[0]
    hf_probs = torch.softmax(hf_logits, dim=-1)
    hf_scores, hf_labels = hf_probs[..., :-1].max(-1)
    
    hf_num_det = (hf_scores > args.threshold).sum().item()
    print(f"Inference time: {hf_time*1000:.1f} ms")
    print(f"Detections (score>{args.threshold}): {hf_num_det}")
    
    # Save HuggingFace result
    hf_vis = visualize_detections(
        image.copy(), hf_scores.numpy(), hf_labels.numpy(), hf_boxes.numpy(),
        args.threshold, id2label
    )
    hf_output = args.output.replace(".jpg", "_hf.jpg").replace(".png", "_hf.png")
    hf_vis.save(hf_output)
    print(f"Saved: {hf_output}")
    
    # Initialize TTNN
    print("\n--- TTNN Implementation ---")
    print(f"Opening device {args.device_id}...")
    device = ttnn.open_device(device_id=args.device_id)
    
    # Prepare TTNN model
    model_bf16 = hf_model.to(torch.bfloat16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model_bf16,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )
    
    state_dict = model_bf16.state_dict()
    cls_token = ttnn.from_torch(
        state_dict["vit.embeddings.cls_token"],
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    detection_tokens = ttnn.from_torch(
        state_dict["vit.embeddings.detection_tokens"],
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    position_embeddings = ttnn.from_torch(
        state_dict["vit.embeddings.position_embeddings"],
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    
    # Prepare input (NCHW -> NHWC + pad channels)
    pixel_values_bf16 = pixel_values.to(torch.bfloat16)
    pixel_values_nhwc = torch.permute(pixel_values_bf16, (0, 2, 3, 1))
    pixel_values_padded = torch.nn.functional.pad(pixel_values_nhwc, (0, 1, 0, 0, 0, 0, 0, 0))
    pixel_values_ttnn = ttnn.from_torch(
        pixel_values_padded, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    
    # Run TTNN inference
    print("Running TTNN inference...")
    start = time.perf_counter()
    ttnn_logits, ttnn_boxes = yolos_for_object_detection(
        config, pixel_values_ttnn, cls_token, detection_tokens, position_embeddings,
        attention_mask=None, parameters=parameters
    )
    ttnn_time = time.perf_counter() - start
    
    ttnn_logits = ttnn.to_torch(ttnn_logits).float()[0]
    ttnn_boxes = ttnn.to_torch(ttnn_boxes).float()[0]
    ttnn_probs = torch.softmax(ttnn_logits, dim=-1)
    ttnn_scores, ttnn_labels = ttnn_probs[..., :-1].max(-1)
    
    ttnn_num_det = (ttnn_scores > args.threshold).sum().item()
    print(f"Inference time: {ttnn_time*1000:.1f} ms")
    print(f"Detections (score>{args.threshold}): {ttnn_num_det}")
    print(f"Speedup vs HuggingFace: {hf_time/ttnn_time:.2f}x")
    
    # Save TTNN result
    ttnn_vis = visualize_detections(
        image.copy(), ttnn_scores.numpy(), ttnn_labels.numpy(), ttnn_boxes.numpy(),
        args.threshold, id2label
    )
    ttnn_output = args.output.replace(".jpg", "_ttnn.jpg").replace(".png", "_ttnn.png")
    ttnn_vis.save(ttnn_output)
    print(f"Saved: {ttnn_output}")
    
    # Benchmark if requested
    if args.benchmark:
        print("\n--- Benchmark (10 runs) ---")
        
        # Warmup
        for _ in range(3):
            yolos_for_object_detection(
                config, pixel_values_ttnn, cls_token, detection_tokens, position_embeddings,
                attention_mask=None, parameters=parameters
            )
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            yolos_for_object_detection(
                config, pixel_values_ttnn, cls_token, detection_tokens, position_embeddings,
                attention_mask=None, parameters=parameters
            )
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times)
        print(f"Average TTNN time: {avg_time*1000:.1f} ms")
        print(f"Throughput: {1/avg_time:.1f} FPS")
        
        # Save benchmark results
        results = {
            "hf_time_ms": hf_time * 1000,
            "ttnn_avg_time_ms": avg_time * 1000,
            "speedup": hf_time / avg_time,
            "throughput_fps": 1 / avg_time,
        }
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Saved: benchmark_results.json")
    
    # Cleanup
    ttnn.close_device(device)
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOS-small Demo")
    parser.add_argument(
        "--image", type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="Path or URL to input image"
    )
    parser.add_argument("--output", type=str, default="demo_result.jpg", help="Output image path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--device-id", type=int, default=0, help="Tenstorrent device ID")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    
    run_demo(parser.parse_args())
