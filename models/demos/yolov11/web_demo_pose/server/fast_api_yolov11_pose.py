# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import logging
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

# Global model variable
model = None
device_global = None

# Import TTNN here so it's available for the model
import ttnn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
for parent in SCRIPT_DIR.parents:
    potential_root = parent
    if (potential_root / "models" / "demos" / "yolov11").exists():
        REPO_ROOT = potential_root
        break

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def apply_pytorch_postprocessing(predictions, anchors_per_stride):
    """
    Apply the same post-processing as the PyTorch YoloV11PoseRaw model does
    TTNN returns raw concatenated tensor: [bbox_raw(64) + conf_raw(1) + keypoints_raw(51)]
    """
    # Handle TTNN tensor layout: [1, 1, anchors, channels] -> [batch, channels, anchors]
    print(f"Input predictions shape: {predictions.shape}")
    if len(predictions.shape) == 4 and predictions.shape[0] == 1 and predictions.shape[1] == 1:
        # TTNN output: [1, 1, anchors, channels] -> [anchors, channels]
        predictions = predictions.squeeze(0).squeeze(0)
        # Convert to PyTorch expected format: [channels, anchors] -> [batch, channels, anchors]
        predictions = predictions.permute(1, 0).unsqueeze(0)  # [1, channels, anchors]

    batch_size = predictions.shape[0]
    num_channels = predictions.shape[1]
    num_anchors = predictions.shape[2]
    print(f"After processing: batch_size={batch_size}, channels={num_channels}, num_anchors={num_anchors}")

    # Split predictions into components (TTNN returns raw concatenated format)
    # predictions: [batch, 116, num_anchors]
    # Format: [bbox_raw(64) + conf_raw(1) + keypoints_raw(51)] = 116

    bbox_raw = predictions[:, :64, :]  # [batch, 64, anchors] - raw DFL features
    conf_raw = predictions[:, 64:65, :]  # [batch, 1, anchors] - raw confidence (needs sigmoid)
    kpt_raw = predictions[:, 65:116, :]  # [batch, 51, anchors] - raw keypoint features

    # ===== Process bounding boxes with DFL =====
    # Reshape bbox_raw for DFL processing: [batch, 64, anchors] -> [batch, 4, 16, anchors]
    bbox_raw = bbox_raw.reshape(batch_size, 4, 16, -1)
    bbox_raw = torch.permute(bbox_raw, (0, 2, 1, 3))  # [batch, 16, 4, anchors]

    # Apply softmax for DFL
    bbox_softmax = torch.softmax(bbox_raw, dim=1)

    # Apply DFL weights (simple approximation)
    dfl_weights = torch.arange(16, dtype=torch.float32, device=bbox_raw.device).reshape(1, 16, 1, 1)
    bbox_decoded = torch.sum(bbox_softmax * dfl_weights, dim=1)  # [batch, 4, anchors]

    # Split into left-top and right-bottom coordinates
    lt = bbox_decoded[:, :2, :]  # [batch, 2, anchors]
    rb = bbox_decoded[:, 2:4, :]  # [batch, 2, anchors]

    # Create anchors exactly like YOLOv11 make_anchors function
    # For 640x640 input: strides [8,16,32] give grids [80x80, 40x40, 20x20]
    anchor_points = []
    stride_tensor = []
    strides = [8, 16, 32]

    for stride in strides:
        # Calculate grid size for this stride
        grid_h = 640 // stride  # 640/8=80, 640/16=40, 640/32=20
        grid_w = 640 // stride

        # Create grid coordinates (same as make_anchors)
        sx = torch.arange(end=grid_w, device=predictions.device, dtype=predictions.dtype) + 0.5
        sy = torch.arange(end=grid_h, device=predictions.device, dtype=predictions.dtype) + 0.5
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((grid_h * grid_w, 1), stride, dtype=predictions.dtype, device=predictions.device)
        )

    # Concatenate like YOLOv11 make_anchors
    anchors_cat = torch.cat(anchor_points, 0)  # [total_anchors, 2] - (x, y)
    strides_cat = torch.cat(stride_tensor, 0)  # [total_anchors, 1]

    # Transpose anchors to [2, total_anchors] like YOLOv11
    anchors_expanded = anchors_cat.t()  # [2, total_anchors] - (y, x) -> (x, y)
    strides_expanded = strides_cat.t()  # [1, total_anchors]

    print(f"Anchors expanded shape: {anchors_expanded.shape}, Strides expanded shape: {strides_expanded.shape}")
    print(f"Total predictions: {num_anchors}")

    # Apply anchor offsets like YOLOv11 does
    # anchors_expanded: [2, total_anchors] - (x, y)
    # lt, rb: [batch, 2, total_anchors] - (x, y) offsets

    # Remove batch dimension for calculation
    lt = lt.squeeze(0)  # [2, total_anchors]
    rb = rb.squeeze(0)  # [2, total_anchors]

    lt_anchored = anchors_expanded - lt  # [2, total_anchors]
    rb_anchored = anchors_expanded + rb  # [2, total_anchors]

    # Calculate center and size
    center_x = (lt_anchored[0, :] + rb_anchored[0, :]) / 2  # x coordinates
    center_y = (lt_anchored[1, :] + rb_anchored[1, :]) / 2  # y coordinates
    width = rb_anchored[0, :] - lt_anchored[0, :]
    height = rb_anchored[1, :] - lt_anchored[1, :]

    # Scale by strides
    center_x = center_x * strides_expanded.squeeze(0)
    center_y = center_y * strides_expanded.squeeze(0)
    width = width * strides_expanded.squeeze(0)
    height = height * strides_expanded.squeeze(0)

    # Convert to (x,y,w,h) format and add batch dimension
    bbox = torch.stack([center_x, center_y, width, height], dim=0).unsqueeze(0)

    # ===== Process confidence =====
    # Apply sigmoid to raw confidence scores
    conf = torch.sigmoid(conf_raw)

    # ===== Process keypoints =====
    # Reshape to [batch, 17, 3, anchors] for processing
    kpt_raw = kpt_raw.reshape(batch_size, 17, 3, -1)

    # Extract x, y, visibility
    kpt_x = kpt_raw[:, :, 0, :]  # [batch, 17, anchors]
    kpt_y = kpt_raw[:, :, 1, :]  # [batch, 17, anchors]
    kpt_v = kpt_raw[:, :, 2, :]  # [batch, 17, anchors]

    # Apply sigmoid to visibility
    kpt_v = torch.sigmoid(kpt_v)

    # Expand anchors and strides for keypoint processing [batch, 17, anchors]
    anchor_x_expanded = anchors_expanded[0:1, :].unsqueeze(0).expand(batch_size, 17, -1)  # [batch, 17, anchors]
    anchor_y_expanded = anchors_expanded[1:2, :].unsqueeze(0).expand(batch_size, 17, -1)  # [batch, 17, anchors]
    strides_expanded_kpt = strides_expanded.unsqueeze(0).expand(batch_size, 17, -1)  # [batch, 17, anchors]

    # Keypoint decoding - SAME as bbox: (kpt * 2 - 0.5 + anchor) * stride
    # Keypoints DO use anchors and strides like bounding boxes!
    kpt_x_decoded = (kpt_x * 2.0 - 0.5 + anchor_x_expanded) * strides_expanded_kpt
    kpt_y_decoded = (kpt_y * 2.0 - 0.5 + anchor_y_expanded) * strides_expanded_kpt

    # Stack back to [batch, 17, 3, anchors]
    keypoints_decoded = torch.stack([kpt_x_decoded, kpt_y_decoded, kpt_v], dim=2)

    # Reshape back to [batch, 51, anchors]
    keypoints_decoded = keypoints_decoded.reshape(batch_size, 51, -1)

    # Final output: [bbox(4) + conf(1) + keypoints(51)] = 56 channels
    output = torch.concat((bbox, conf, keypoints_decoded), 1)

    return output


# Standard YOLOv11 pose anchors
anchors_per_stride = [
    [[12, 16], [19, 36], [40, 28]],  # stride 8
    [[36, 75], [76, 55], [72, 146]],  # stride 16
    [[142, 110], [192, 243], [459, 401]],  # stride 32
]


def apply_simple_nms(detections, iou_threshold=0.5):
    """Apply simple non-maximum suppression to remove overlapping detections"""
    if len(detections) <= 1:
        return detections

    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x[4], reverse=True)

    keep = []
    for i, det in enumerate(detections):
        # Check if this detection overlaps too much with any already kept detection
        should_keep = True
        x1, y1, w1, h1 = det[:4]

        for kept_idx in keep:
            kept_det = detections[kept_idx]
            x2, y2, w2, h2 = kept_det[:4]

            # Calculate IoU
            iou = calculate_iou(x1, y1, w1, h1, x2, y2, w2, h2)
            if iou > iou_threshold:
                should_keep = False
                break

        # Only keep this detection if it doesn't overlap too much with kept detections
        if should_keep:
            keep.append(i)

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
    global model, device_global
    print("=== FASTAPI SERVER STARTUP ===")
    print("Loading TTNN YOLOv11 pose model...")
    try:
        from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
        from models.demos.yolov11.tt.model_preprocessing_pose import create_yolov11_pose_model_parameters
        from models.demos.yolov11.tt.ttnn_yolov11_pose_model import TtnnYoloV11Pose

        device_id = 0
        device_global = ttnn.CreateDevice(device_id, l1_small_size=32768)  # Increased for pose model

        # Load PyTorch model for parameter creation
        torch_model = YoloV11Pose()
        weights_path = REPO_ROOT / "models" / "demos" / "yolov11" / "reference" / "yolov11_pose_pretrained_correct.pth"
        if weights_path.exists():
            torch_model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
            print("✓ Loaded pretrained weights")
        else:
            print(f"⚠ Pretrained weights not found at {weights_path}")
            reference_dir = REPO_ROOT / "models" / "demos" / "yolov11" / "reference"
            if str(reference_dir) not in sys.path:
                sys.path.insert(0, str(reference_dir))
            try:
                from models.demos.yolov11.reference.load_weights_correct import load_weights as generate_pose_weights

                print("Attempting to auto-download YOLOv11 pose weights...")
                reference_dir.mkdir(parents=True, exist_ok=True)
                cwd = os.getcwd()
                try:
                    os.chdir(reference_dir)
                    generate_pose_weights()
                finally:
                    os.chdir(cwd)
            except ModuleNotFoundError as exc:
                logging.exception("Auto-download failed: unable to import weight generator: %s", exc)
            except Exception as exc:
                logging.exception("Auto-download failed while generating YOLOv11 pose weights: %s", exc)

            if weights_path.exists():
                torch_model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
                print("✓ Auto-downloaded and loaded pretrained weights")
            else:
                raise FileNotFoundError(
                    f"Unable to locate pretrained pose weights at {weights_path}. Please ensure the weights are available."
                )

        # Create TTNN model parameters
        img_tensor_batched = torch.randn(1, 3, 640, 640)  # Dummy tensor for parameter creation
        parameters = create_yolov11_pose_model_parameters(torch_model, img_tensor_batched, device=device_global)

        # Create TTNN model
        model = TtnnYoloV11Pose(device=device_global, parameters=parameters)
        print("✓ TTNN pose model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load TTNN pose model: {e}")
        import traceback

        traceback.print_exc()
        model = None
        device_global = None


@app.on_event("shutdown")
async def shutdown():
    global model, device_global
    if model:
        try:
            model.release()
            print("✓ Model released")
        except Exception as e:
            print(f"Warning: Failed to release model: {e}")
    if device_global:
        try:
            ttnn.close_device(device_global)
            print("✓ Device closed")
        except Exception as e:
            print(f"Warning: Failed to close device: {e}")


@app.post("/pose_estimation_v2")
async def pose_estimation_v2(file: UploadFile = File(...)):
    global model, device_global
    global request_count
    if "request_count" not in globals():
        request_count = 0
    request_count += 1

    print(f"DEBUG: Received pose estimation request #{request_count}")

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

        # Check if we're getting different images
        import hashlib

        image_hash = hashlib.md5(contents).hexdigest()[:8]
        print(f"DEBUG: Processing image with hash: {image_hash}, size: {len(contents)} bytes")

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

        # Run TTNN pose inference
        try:
            # Convert to TTNN tensor
            tt_input = ttnn.from_torch(image_tensor.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            tt_input = ttnn.to_device(tt_input, device_global)

            # Run inference
            tt_output = model(tt_input)

            # Convert back to torch
            processed_output = ttnn.to_torch(tt_output)

            # Decode raw TTNN outputs to match PyTorch pose head format
            processed_output = processed_output.cpu()
            processed_output = apply_pytorch_postprocessing(processed_output, anchors_per_stride)

            # Clean up TTNN tensors
            ttnn.deallocate(tt_input)
            ttnn.deallocate(tt_output)
            print("DEBUG: TTNN pose inference completed successfully")
        except Exception as e:
            print(f"DEBUG: TTNN pose inference failed: {e}")
            import traceback

            traceback.print_exc()
            return {"error": f"TTNN pose inference failed: {str(e)}"}

        print(f"DEBUG: TTNN model output shape after decoding: {processed_output.shape}")
        print(f"DEBUG: Model output range: [{processed_output.min():.6f}, {processed_output.max():.6f}]")
        if processed_output.shape[-1] > 0:
            print(f"DEBUG: Model output sample [0,:5,0]: {processed_output[0,:5,0].tolist()}")
        else:
            print("DEBUG: No detections returned!")

        # Extract components from processed output
        bbox = processed_output[0, :4, :]  # [4, num_anchors] - center_x, center_y, w, h
        conf = processed_output[0, 4:5, :]  # [1, num_anchors] - confidence
        keypoints = processed_output[0, 5:, :]  # [51, num_anchors] - 17 keypoints * 3 values each

        print(f"DEBUG: bbox[0,:3]: {bbox[0,:3].tolist()}")
        print(f"DEBUG: conf[0,:3]: {conf[0,:3].tolist()}")

        print(f"DEBUG: bbox shape: {bbox.shape}, conf shape: {conf.shape}, keypoints shape: {keypoints.shape}")
        print(f"DEBUG: Raw bbox sample: {bbox[:, 0].tolist() if bbox.shape[1] > 0 else 'no bboxes'}")

        # Debug: Show confidence statistics
        max_conf = conf.max().item()
        mean_conf = conf.mean().item()
        print(f"DEBUG: Confidence stats - Max: {max_conf:.6f}, Mean: {mean_conf:.6f}")

        # Apply confidence threshold
        conf_threshold = 0.01
        conf_mask = conf[0, :] > conf_threshold
        valid_indices = torch.where(conf_mask)[0]

        print(f"DEBUG: Found {len(valid_indices)} detections above confidence threshold {conf_threshold}")

        # Limit detections to prevent server hang
        max_detections = 10  # Reasonable limit
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

            # Convert bbox center coords from 640x640 back to original image coords
            x_center_orig = (x - pad_left) / scale
            y_center_orig = (y - pad_top) / scale
            w_orig = w / scale
            h_orig = h / scale

            # Convert to top-left corner + width/height (normalized)
            x1_orig = x_center_orig - (w_orig / 2.0)
            y1_orig = y_center_orig - (h_orig / 2.0)

            # Normalize to [0,1] relative to original image dimensions (what client expects)
            x_norm = x1_orig / orig_w
            y_norm = y1_orig / orig_h
            w_norm = w_orig / orig_w
            h_norm = h_orig / orig_h

            # Clamp to valid range
            x_norm = float(torch.clamp(torch.tensor(x_norm), 0.0, 1.0).item())
            y_norm = float(torch.clamp(torch.tensor(y_norm), 0.0, 1.0).item())
            w_norm = float(torch.clamp(torch.tensor(w_norm), 0.0, 1.0).item())
            h_norm = float(torch.clamp(torch.tensor(h_norm), 0.0, 1.0).item())

            # Convert keypoints from 640x640 coords back to original image coords, then normalize
            kpt_normalized = []
            for i in range(17):  # 17 keypoints
                base_idx = i * 3
                if base_idx + 2 >= len(kpt_data):
                    print(
                        f"DEBUG: Warning - not enough keypoint data. base_idx={base_idx}, len(kpt_data)={len(kpt_data)}"
                    )
                    break
                kx_640 = float(kpt_data[base_idx])
                ky_640 = float(kpt_data[base_idx + 1])
                kv = float(kpt_data[base_idx + 2])

                # Convert back to original coordinates
                kx_orig = (kx_640 - pad_left) / scale
                ky_orig = (ky_640 - pad_top) / scale

                # Normalize to [0,1] relative to original image dimensions
                kx_norm = kx_orig / orig_w
                ky_norm = ky_orig / orig_h

                kpt_normalized.extend([kx_norm, ky_norm, kv])

            # Combine: [x, y, w, h, conf, class_id, kpt_x1, kpt_y1, kpt_conf1, ...]
            # class_id = 0 for person
            detection = [x_norm, y_norm, w_norm, h_norm, confidence, 0] + kpt_normalized
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
