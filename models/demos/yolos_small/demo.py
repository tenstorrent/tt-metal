"""
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import time
from io import BytesIO

import requests
import torch
from PIL import Image

import ttnn

# Import reference and TTNN implementations
from models.demos.yolos_small.reference.config import get_yolos_small_config
from models.demos.yolos_small.reference.modeling_yolos import YolosForObjectDetection as PyTorchYolos
from models.demos.yolos_small.yolos_ttnn.common import OptimizationConfig, convert_to_ttnn_tensor, get_dtype_for_stage
from models.demos.yolos_small.yolos_ttnn.modeling_yolos import YolosForObjectDetection as TtnnYolos

# COCO class names
COCO_CLASSES = [
    "N/A",
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
    "N/A",
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
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
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
    "N/A",
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
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
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
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def load_image(image_path_or_url, size=(512, 864)):
    """Load and preprocess image from path or URL."""
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to expected input size
    image = image.resize(size, Image.BILINEAR)

    return image


def preprocess_image(image):
    """
    Preprocess image for YOLOS input.
    Normalizes with ImageNet mean and std.
    """
    # Convert to tensor and normalize
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
    return pixel_values


def visualize_predictions(image, predictions, threshold=0.7):
    """Visualize predictions on image."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    width, height = image.size

    scores = predictions["scores"][0]  # First batch
    labels = predictions["labels"][0]
    boxes = predictions["boxes"][0]
    keep = predictions["keep"][0]

    # Draw bounding boxes
    for i in range(len(scores)):
        if keep[i]:
            score = scores[i].item()
            label = labels[i].item()
            box = boxes[i]  # [center_x, center_y, width, height] normalized

            # Convert to pixel coordinates
            cx = box[0].item() * width
            cy = box[1].item() * height
            w = box[2].item() * width
            h = box[3].item() * height

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Draw label
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
            text = f"{class_name}: {score:.2f}"
            draw.text((x1, y1 - 10), text, fill="red")

    return image


def load_pretrained_weights(model, checkpoint_path=None):
    """
    Load pretrained weights from HuggingFace or checkpoint.
    """
    if checkpoint_path:
        # Load from local checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        # Download from HuggingFace
        try:
            from transformers import YolosForObjectDetection as HFYolos

            hf_model = HFYolos.from_pretrained("hustvl/yolos-small")

            # Copy weights to our reference model
            # This is a simplified version - you may need to adjust key names
            model.load_state_dict(hf_model.state_dict(), strict=False)
            print("Loaded weights from HuggingFace: hustvl/yolos-small")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Using randomly initialized weights")

    return model


def benchmark_inference(model, pixel_values, num_runs=10, warmup_runs=3):
    """Benchmark inference time."""
    # Warmup
    for _ in range(warmup_runs):
        _ = model(pixel_values)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(pixel_values)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return {
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "throughput_fps": 1.0 / avg_time,
    }


def main():
    parser = argparse.ArgumentParser(description="YOLOS-small Object Detection Demo")
    parser.add_argument(
        "--image",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="Path or URL to input image",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (optional, will use HuggingFace if not provided)",
    )
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save output image with predictions")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold for detections")
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Optimization stage (1: basic, 2: optimized, 3: deep optimizations)",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--device-id", type=int, default=0, help="Tenstorrent device ID")
    parser.add_argument("--pytorch-only", action="store_true", help="Run PyTorch reference only (no TTNN)")

    args = parser.parse_args()

    print("=" * 80)
    print("YOLOS-small Object Detection Demo")
    print("=" * 80)

    # Load configuration
    config = get_yolos_small_config()
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Image size: {config.image_size}")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Detection tokens: {config.num_detection_tokens}")

    # Load and preprocess image
    print(f"\nLoading image from: {args.image}")
    image = load_image(args.image, size=tuple(config.image_size))
    pixel_values = preprocess_image(image)
    print(f"Image shape: {pixel_values.shape}")

    # === PyTorch Reference ===
    print("\n" + "=" * 80)
    print("Running PyTorch Reference Implementation")
    print("=" * 80)

    pytorch_model = PyTorchYolos(config)
    pytorch_model = load_pretrained_weights(pytorch_model, args.checkpoint)
    pytorch_model.eval()

    with torch.no_grad():
        start = time.perf_counter()
        pytorch_predictions = pytorch_model.predict(pixel_values, threshold=args.threshold)
        pytorch_time = time.perf_counter() - start

    print(f"PyTorch inference time: {pytorch_time * 1000:.2f} ms")
    print(f"Detected {pytorch_predictions['keep'][0].sum().item()} objects")

    # Visualize PyTorch predictions
    pytorch_output_image = visualize_predictions(image.copy(), pytorch_predictions, args.threshold)
    pytorch_output_path = args.output.replace(".jpg", "_pytorch.jpg")
    pytorch_output_image.save(pytorch_output_path)
    print(f"Saved PyTorch predictions to: {pytorch_output_path}")

    if args.pytorch_only:
        print("\nPyTorch-only mode enabled. Exiting.")
        return

    # === TTNN Implementation ===
    print("\n" + "=" * 80)
    print(f"Running TTNN Implementation - Stage {args.stage}")
    print("=" * 80)

    # Initialize TTNN device
    device = ttnn.open_device(device_id=args.device_id)
    print(f"Opened Tenstorrent device {args.device_id}")

    # Get optimization config for selected stage
    if args.stage == 1:
        opt_config = OptimizationConfig.stage1()
        print("Using Stage 1: Basic bring-up (no optimizations)")
    elif args.stage == 2:
        opt_config = OptimizationConfig.stage2()
        print("Using Stage 2: Basic optimizations (sharding, fusion, L1)")
    else:
        opt_config = OptimizationConfig.stage3()
        print("Using Stage 3: Deep optimizations (fused SDPA, bfloat8, max cores)")

    # Create TTNN model
    ttnn_model = TtnnYolos(
        config=config,
        device=device,
        reference_model=pytorch_model,
        opt_config=opt_config,
    )
    print("TTNN model initialized and weights loaded")

    # Convert input to TTNN using dtype appropriate for the chosen stage
    input_dtype = get_dtype_for_stage(opt_config)
    pixel_values_ttnn = convert_to_ttnn_tensor(
        pixel_values,
        device,
        dtype=input_dtype,
    )

    # Run inference
    print("\nRunning TTNN inference...")
    start = time.perf_counter()
    ttnn_predictions = ttnn_model.predict(pixel_values_ttnn, threshold=args.threshold)
    ttnn_time = time.perf_counter() - start

    print(f"TTNN inference time: {ttnn_time * 1000:.2f} ms")
    print(f"Speedup vs PyTorch: {pytorch_time / ttnn_time:.2f}x")
    print(f"Detected {ttnn_predictions['keep'][0].sum().item()} objects")

    # Visualize TTNN predictions
    ttnn_output_image = visualize_predictions(image.copy(), ttnn_predictions, args.threshold)
    ttnn_output_path = args.output.replace(".jpg", f"_ttnn_stage{args.stage}.jpg")
    ttnn_output_image.save(ttnn_output_path)
    print(f"Saved TTNN predictions to: {ttnn_output_path}")

    # Benchmark if requested
    if args.benchmark:
        print("\n" + "=" * 80)
        print("Running Performance Benchmark")
        print("=" * 80)

        print("\nPyTorch benchmark...")
        pytorch_bench = benchmark_inference(lambda x: pytorch_model(x), pixel_values)
        print(f"  Avg time: {pytorch_bench['avg_time_ms']:.2f} ± {pytorch_bench['std_time_ms']:.2f} ms")
        print(f"  Throughput: {pytorch_bench['throughput_fps']:.2f} FPS")

        print("\nTTNN benchmark...")
        ttnn_bench = benchmark_inference(lambda x: ttnn_model(x), pixel_values_ttnn)
        print(f"  Avg time: {ttnn_bench['avg_time_ms']:.2f} ± {ttnn_bench['std_time_ms']:.2f} ms")
        print(f"  Throughput: {ttnn_bench['throughput_fps']:.2f} FPS")
        print(f"  Speedup: {pytorch_bench['avg_time_ms'] / ttnn_bench['avg_time_ms']:.2f}x")

        # Save benchmark results
        benchmark_results = {
            "pytorch": pytorch_bench,
            "ttnn": ttnn_bench,
            "speedup": pytorch_bench["avg_time_ms"] / ttnn_bench["avg_time_ms"],
            "stage": args.stage,
            "config": {
                "hidden_size": config.hidden_size,
                "num_layers": config.num_hidden_layers,
                "image_size": config.image_size,
            },
        }

        benchmark_path = f"benchmark_stage{args.stage}.json"
        with open(benchmark_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"\nSaved benchmark results to: {benchmark_path}")

    # Cleanup
    ttnn.close_device(device)
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
