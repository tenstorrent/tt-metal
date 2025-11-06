# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import logging
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

# Global model variable
model = None
device_global = None

# Import postprocessing functions from the working demo
import sys

# Import TTNN here so it's available for the model
import ttnn

sys.path.insert(0, "/home/ubuntu/pose/tt-metal")
from ttnn_raw_output_demo import apply_pytorch_postprocessing


def apply_simple_nms(detections, iou_threshold=0.5):
    """Apply simple non-maximum suppression to remove overlapping detections"""
    if len(detections) <= 1:
        return detections

    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x[4], reverse=True)

    keep = []
    for i, det in enumerate(detections):
        if i in keep:
            continue

        keep.append(i)
        x1, y1, w1, h1 = det[:4]

        for j in range(i + 1, len(detections)):
            if j in keep:
                continue

            x2, y2, w2, h2 = detections[j][:4]

            # Calculate IoU
            iou = calculate_iou(x1, y1, w1, h1, x2, y2, w2, h2)
            if iou > iou_threshold:
                keep.remove(j) if j in keep else None

    return [detections[i] for i in sorted(keep)]


def calculate_iou(x1, y1, w1, h1, x2, y2, w2, h2):
    """Calculate intersection over union for two bounding boxes"""
    # Convert to x2, y2 coordinates
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    # Calculate intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


app = FastAPI(
    title="YOLOv11 pose estimation",
    description="Inference engine to detect human poses in image.",
    version="0.0",
)

# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()]
)

# Global model variable
model = None


@app.get("/")
async def root():
    return {"message": "Hello World", "status": "Server running", "model_loaded": model is not None}


@app.on_event("startup")
async def startup():
    global model
    print("=== FASTAPI SERVER STARTUP ===")
    print("TTNN model will be loaded on first pose request")
    print("✓ Server ready to accept requests")
    model = None  # Will be loaded lazily


@app.on_event("shutdown")
async def shutdown():
    global device_global
    if "device_global" in globals():
        try:
            device_global.disable_and_close()
            print("✓ Device closed")
        except:
            pass


@app.post("/pose_estimation_v2")
async def pose_estimation_v2(file: UploadFile = File(...)):
    global model, device_global
    global request_count
    if "request_count" not in globals():
        request_count = 0
    request_count += 1

    print(f"DEBUG: Received pose estimation request #{request_count}")
    # Check if we're getting different images
    import hashlib

    image_hash = hashlib.md5(contents).hexdigest()[:8]
    print(f"DEBUG: Processing image with hash: {image_hash}, size: {len(contents)} bytes")

    # Lazy loading of TTNN model
    if model is None:
        try:
            print("=== LAZY LOADING TTNN MODEL ===")

            print("Creating device...")
            device_id = 0
            # Use the exact same configuration as the working ttnn_raw_output_demo.py
            device_global = ttnn.open_device(device_id=device_id, l1_small_size=32768)

            print("Loading TTNN model...")
            # Use EXACTLY the same approach as the working ttnn_raw_output_demo.py
            from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
            from models.demos.yolov11.tt.model_preprocessing_pose import create_yolov11_pose_model_parameters
            from models.demos.yolov11.tt.ttnn_yolov11_pose_model import TtnnYoloV11Pose

            # Load PyTorch model (same as working demo)
            torch_model = YoloV11Pose()

            # Create sample input for preprocessing (same as working demo)
            img_tensor_batched = torch.randn(1, 3, 640, 640)  # Add batch dimension
            parameters = create_yolov11_pose_model_parameters(torch_model, img_tensor_batched, device=device_global)

            # Create TTNN model (same as working demo)
            model = TtnnYoloV11Pose(device=device_global, parameters=parameters)

            print("✓ TTNN model loaded successfully")

        except Exception as e:
            print(f"TTNN model loading failed: {e}")
            import traceback

            traceback.print_exc()
            return {"error": f"TTNN model loading failed: {str(e)}"}

    try:
        print(f"DEBUG: Received pose request for file: {file.filename}")
        print(f"DEBUG: File object type: {type(file)}")

        print("DEBUG: Reading image file...")
        # Read and process the uploaded image
        contents = await file.read()
        print(f"DEBUG: File size: {len(contents)} bytes")

        # Load image with PIL first
        try:
            image = Image.open(BytesIO(contents)).convert("RGB")
            print(f"DEBUG: PIL image loaded, size: {image.size}")
        except Exception as e:
            print(f"DEBUG: PIL image loading failed: {e}")
            return {"error": f"Image loading failed: {str(e)}"}

        # Use same preprocessing as working demo
        import cv2

        # Convert PIL image to numpy array
        try:
            image_array = np.array(image)
            print(f"DEBUG: Image shape: {image_array.shape}")
        except Exception as e:
            print(f"DEBUG: Numpy conversion failed: {e}")
            return {"error": f"Image conversion failed: {str(e)}"}

        # Preprocess like working demo: resize and pad to 640x640
        orig_h, orig_w = image_array.shape[:2]
        target_size = (640, 640)
        scale = min(target_size[0] / orig_w, target_size[1] / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        # Resize image
        resized_img = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        padded_img[:new_h, :new_w] = resized_img

        # Convert to tensor: CHW format, normalized
        image_tensor = torch.from_numpy(padded_img).float().permute(2, 0, 1) / 255.0
        print(f"DEBUG: Preprocessed tensor shape: {image_tensor.shape}")

        # Keep padded image shape for postprocessing
        padded_shape = padded_img.shape

        print("DEBUG: Starting TTNN model inference...")
        print(f"DEBUG: Model type: {type(model)}")

        # Convert input to TTNN format (same as working demo: add batch dimension)
        image_tensor_batched = image_tensor.unsqueeze(0)  # Add batch dimension
        print(f"DEBUG: Input tensor shape: {image_tensor_batched.shape}")
        tt_input = ttnn.from_torch(image_tensor_batched, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        print(f"DEBUG: TTNN input created")
        tt_input = ttnn.to_device(tt_input, device_global)
        print(f"DEBUG: TTNN input moved to device")

        # Run TTNN inference (core model only) - returns [batch, 56, num_anchors]
        try:
            response = model(tt_input)
            print("DEBUG: TTNN inference completed successfully")
        except Exception as e:
            print(f"DEBUG: TTNN inference failed: {e}")
            import traceback

            traceback.print_exc()
            return {"error": f"TTNN inference failed: {str(e)}"}

        # Convert back to torch (output is in TILE_LAYOUT from DRAM)
        raw_output = ttnn.to_torch(response)
        print(f"DEBUG: TTNN raw output shape: {raw_output.shape}")
        print(f"DEBUG: Raw output range: [{raw_output.min():.6f}, {raw_output.max():.6f}]")

        print("DEBUG: Applying postprocessing like offline demo...")
        # Apply postprocessing like the offline demo does
        # Create anchors_per_stride like the offline demo
        anchors_per_stride = None  # Will be created in apply_pytorch_postprocessing

        processed_output = apply_pytorch_postprocessing(raw_output, anchors_per_stride)
        print(f"DEBUG: Processed output shape: {processed_output.shape}")
        print(f"DEBUG: Processed output range: [{processed_output.min():.6f}, {processed_output.max():.6f}]")

        # Extract components from processed output
        bbox = processed_output[0, :4, :]  # [4, num_anchors] - center_x, center_y, w, h
        conf = processed_output[0, 4:5, :]  # [1, num_anchors] - confidence
        keypoints = processed_output[0, 5:, :]  # [51, num_anchors] - 17 keypoints * 3 values each

        print(f"DEBUG: bbox shape: {bbox.shape}, conf shape: {conf.shape}, keypoints shape: {keypoints.shape}")
        print(f"DEBUG: Raw bbox sample: {bbox[:, 0].tolist() if bbox.shape[1] > 0 else 'no bboxes'}")

        # Debug: Show confidence statistics
        max_conf = conf.max().item()
        mean_conf = conf.mean().item()
        print(f"DEBUG: Confidence stats - Max: {max_conf:.6f}, Mean: {mean_conf:.6f}")

        # Apply confidence threshold
        conf_threshold = 0.5
        conf_mask = conf[0, :] > conf_threshold
        valid_indices = torch.where(conf_mask)[0]

        print(f"DEBUG: Found {len(valid_indices)} detections above confidence threshold {conf_threshold}")

        # Limit detections to prevent server hang
        max_detections = 5  # Reduced for debugging
        if len(valid_indices) > max_detections:
            # Sort by confidence and take top N
            conf_values = conf[0, valid_indices]
            _, top_indices = torch.topk(conf_values, max_detections)
            valid_indices = valid_indices[top_indices]
            print(f"DEBUG: Limited to top {max_detections} detections")

        if len(valid_indices) == 0:
            # Debug: show max confidence found
            max_conf = conf.max().item()
            print(f"DEBUG: Max confidence in output: {max_conf:.6f}")
            # Show top 5 confidence values
            top_confs = torch.topk(conf.squeeze(), min(5, conf.numel())).values
            print(f"DEBUG: Top confidence values: {top_confs.tolist()}")

        # Extract valid detections
        detections = []
        print(f"DEBUG: Processing {len(valid_indices)} valid detections")
        for idx in valid_indices:
            # Get bbox in 640x640 coordinate system
            x, y, w, h = bbox[:, idx].tolist()
            confidence = conf[0, idx].item()

            # Get keypoints (17 keypoints × 3 values each = 51 values) in 640x640 coordinate system
            kpt_data = keypoints[:, idx]  # [51]

            # Convert coordinates from 640x640 back to original image coordinates
            # First, undo the padding and scaling
            orig_h, orig_w = image_array.shape[:2]

            # Calculate scaling and padding that was applied
            scale = min(640 / orig_w, 640 / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            pad_left = (640 - new_w) // 2
            pad_top = (640 - new_h) // 2

            # Convert bbox from 640x640 coords back to original image coords, then normalize to 640x640 display space
            x_orig = (x - pad_left) / scale
            y_orig = (y - pad_top) / scale
            w_orig = w / scale
            h_orig = h / scale

            # Normalize to [0,1] for 640x640 display space (what client uses)
            x_norm = x_orig / 640.0
            y_norm = y_orig / 640.0
            w_norm = w_orig / 640.0
            h_norm = h_orig / 640.0

            # Convert keypoints from 640x640 coords back to original image coords, then normalize
            kpt_normalized = []
            for i in range(17):  # 17 keypoints
                base_idx = i * 3
                kx_640 = float(kpt_data[base_idx])
                ky_640 = float(kpt_data[base_idx + 1])
                kv = float(kpt_data[base_idx + 2])

                # Convert back to original coordinates
                kx_orig = (kx_640 - pad_left) / scale
                ky_orig = (ky_640 - pad_top) / scale

                # Normalize to [0,1] for 640x640 display space
                kx_norm = kx_orig / 640.0
                ky_norm = ky_orig / 640.0

                kpt_normalized.extend([kx_norm, ky_norm, kv])

            # Combine: [x, y, w, h, conf, kpt_x1, kpt_y1, kpt_conf1, ...]
            detection = [x_norm, y_norm, w_norm, h_norm, confidence] + kpt_normalized
            detections.append(detection)

            # Debug: print first detection coordinates
            if len(detections) == 1:
                print(
                    f"DEBUG: First detection bbox: [{x_norm:.3f}, {y_norm:.3f}, {w_norm:.3f}, {h_norm:.3f}] conf={confidence:.3f}"
                )

        # Apply NMS to remove overlapping detections
        if len(detections) > 1:
            detections = apply_simple_nms(detections, iou_threshold=0.5)

        print(f"DEBUG: Final detections after NMS: {len(detections)}")

        print(f"DEBUG: Returning {len(detections)} detections")
        return {"detections": detections}

    except Exception as e:
        logging.error(f"Error in pose estimation: {e}")
        import traceback

        traceback.print_exc()
        return {"error": f"Pose estimation failed: {str(e)}"}
