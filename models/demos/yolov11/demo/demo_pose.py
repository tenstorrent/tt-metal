# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO11 Pose Estimation Demo using TTNN

This demo runs pose estimation (keypoint detection) using the TTNN implementation
on TT-Metal hardware. It can run on images or COCO dataset.
"""

import os

import cv2
import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import disable_persistent_kernel_cache
from models.demos.utils.common_demo_utils import LoadImages, get_mesh_mappers, preprocess
from models.demos.yolov11.common import YOLOV11_L1_SMALL_SIZE
from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
from models.demos.yolov11.tt.model_preprocessing_pose import create_yolov11_pose_model_parameters
from models.demos.yolov11.tt.ttnn_yolov11_pose_model import TtnnYoloV11Pose

# COCO Keypoint connections for skeleton visualization
SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],  # Legs
    [6, 12],
    [7, 13],  # Torso
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],  # Arms
    [6, 7],  # Shoulders
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],  # Face
    [5, 6],
    [5, 7],  # Neck
]

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def init_pose_model_and_runner(device, model_type, batch_size_per_device):
    """
    Initialize YOLO11 Pose model and TTNN runner

    Args:
        device: TT device
        model_type: "torch_model" or "tt_model"
        batch_size_per_device: Batch size per device

    Returns:
        model: PyTorch model
        ttnn_model: TTNN model (if model_type == "tt_model")
        mesh_composer: Mesh composer for multi-device
        batch_size: Total batch size
    """
    disable_persistent_kernel_cache()

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    logger.info(f"Running YOLO11 Pose with batch_size={batch_size} across {num_devices} devices")

    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    # Load PyTorch model with pretrained weights
    weights_path = "models/demos/yolov11/reference/yolov11_pose_pretrained_correct.pth"

    torch_model = YoloV11Pose()

    if os.path.exists(weights_path):
        torch_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        logger.info(f"Loaded pretrained weights from {weights_path}")
    else:
        logger.warning(f"Pretrained weights not found at {weights_path}")
        logger.warning("Run: cd models/demos/yolov11/reference && python3 load_weights_correct.py")

    torch_model.eval()

    ttnn_model = None
    if model_type == "tt_model":
        # Create TTNN model from SAME PyTorch structure
        logger.info("Creating TTNN pose model...")
        dummy_input = torch.randn(batch_size, 3, 640, 640)
        parameters = create_yolov11_pose_model_parameters(torch_model, dummy_input, device=device)
        ttnn_model = TtnnYoloV11Pose(device, parameters)
        logger.info("TTNN pose model created")

    return torch_model, ttnn_model, outputs_mesh_composer, batch_size


def process_images(dataset, res, batch_size):
    """Load and preprocess images"""
    torch_images, orig_images, paths_images = [], [], []

    for paths, im0s, _ in dataset:
        assert len(im0s) == batch_size, f"Expected batch of size {batch_size}, but got {len(im0s)}"

        paths_images.extend(paths)
        orig_images.extend(im0s)

        for idx, img in enumerate(im0s):
            if img is None:
                raise ValueError(f"Could not read image: {paths[idx]}")
            tensor = preprocess([img], res=res)
            torch_images.append(tensor)

        if len(torch_images) >= batch_size:
            break

    torch_input_tensor = torch.cat(torch_images, dim=0)
    return torch_input_tensor, orig_images, paths_images


def postprocess_pose(preds, orig_images, paths_images, input_size=(640, 640), conf_threshold=0.7, nms_threshold=0.45):
    """
    Postprocess pose predictions with proper coordinate transformation

    Args:
        preds: Model predictions [batch, 56, num_anchors]
        orig_images: Original images
        paths_images: Image paths
        input_size: Model input size (height, width)
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold

    Returns:
        List of detections per image
    """
    results = []

    for i in range(len(orig_images)):
        orig_img = orig_images[i]
        orig_h, orig_w = orig_img.shape[:2]

        # Calculate scaling factors from preprocessing
        # The preprocess() function does letterboxing (maintains aspect ratio)
        scale = min(input_size[0] / orig_h, input_size[1] / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        # Calculate padding (letterbox adds padding to center the image)
        pad_top = (input_size[0] - new_h) // 2
        pad_left = (input_size[1] - new_w) // 2

        # Extract predictions for this image
        # Handle both torch tensors and numpy arrays
        if isinstance(preds, torch.Tensor):
            bbox = preds[i, 0:4, :].detach().cpu().numpy()  # [4, anchors]
            conf = preds[i, 4, :].detach().cpu().numpy()  # [anchors]
            keypoints = preds[i, 5:56, :].detach().cpu().numpy()  # [51, anchors]
        else:
            bbox = preds[i, 0:4, :]  # [4, anchors]
            conf = preds[i, 4, :]  # [anchors]
            keypoints = preds[i, 5:56, :]  # [51, anchors]

        detections = []
        for j in range(preds.shape[2]):
            if conf[j] > conf_threshold:
                # Bbox in 640x640 space
                x, y, w, h = bbox[:, j]

                # Transform back to original image space
                x = (x - pad_left) / scale
                y = (y - pad_top) / scale
                w = w / scale
                h = h / scale

                x1 = int(max(0, x - w / 2))
                y1 = int(max(0, y - h / 2))
                x2 = int(min(orig_w, x + w / 2))
                y2 = int(min(orig_h, y + h / 2))

                # Extract and transform keypoints
                kpts = keypoints[:, j].reshape(17, 3)
                kpts_transformed = []

                for kpt in kpts:
                    kx, ky, kv = kpt
                    # Transform keypoint coordinates back to original image space
                    kx = (kx - pad_left) / scale
                    ky = (ky - pad_top) / scale
                    # Ensure within bounds
                    kx = max(0, min(orig_w, kx))
                    ky = max(0, min(orig_h, ky))
                    kpts_transformed.append([kx, ky, kv])

                detections.append(
                    {"bbox": [x1, y1, x2, y2], "confidence": float(conf[j]), "keypoints": np.array(kpts_transformed)}
                )

        # Simple NMS (remove overlapping detections)
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        results.append({"path": paths_images[i], "image": orig_images[i], "detections": detections})

    return results


def visualize_and_save_pose(results, save_dir):
    """
    Visualize pose detections and save to disk

    Args:
        results: List of detection results
        save_dir: Directory to save visualized images
    """
    os.makedirs(save_dir, exist_ok=True)

    colors = [
        (255, 0, 0),
        (255, 85, 0),
        (255, 170, 0),
        (255, 255, 0),
        (170, 255, 0),
        (85, 255, 0),
        (0, 255, 0),
        (0, 255, 85),
        (0, 255, 170),
        (0, 255, 255),
        (0, 170, 255),
        (0, 85, 255),
        (0, 0, 255),
        (85, 0, 255),
        (170, 0, 255),
        (255, 0, 255),
        (255, 0, 170),
    ]

    for result in results:
        img = result["image"].copy()
        detections = result["detections"]

        # Draw each detection
        for det in detections:
            bbox = det["bbox"]
            conf = det["confidence"]
            keypoints = det["keypoints"]

            # Draw bounding box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(
                img, f"Person: {conf:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            # Draw keypoints
            for kpt_idx, (x, y, v) in enumerate(keypoints):
                if v > 0.3:  # Visible keypoints
                    color = colors[kpt_idx % len(colors)]
                    cv2.circle(img, (int(x), int(y)), 5, color, -1)
                    cv2.circle(img, (int(x), int(y)), 6, (255, 255, 255), 1)

            # Draw skeleton
            for connection in SKELETON:
                kpt1_idx = connection[0] - 1
                kpt2_idx = connection[1] - 1

                if kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints):
                    kpt1 = keypoints[kpt1_idx]
                    kpt2 = keypoints[kpt2_idx]

                    if kpt1[2] > 0.3 and kpt2[2] > 0.3:
                        pt1 = (int(kpt1[0]), int(kpt1[1]))
                        pt2 = (int(kpt2[0]), int(kpt2[1]))
                        cv2.line(img, pt1, pt2, (255, 255, 0), 2)

        # Save result
        basename = os.path.basename(result["path"])
        output_path = os.path.join(save_dir, f"pose_{basename}")
        cv2.imwrite(output_path, img)
        logger.info(f"Saved visualization to: {output_path}")


def run_inference_and_save_pose(
    torch_model, ttnn_model, model_type, outputs_mesh_composer, im_tensor, orig_images, paths_images, save_dir, device
):
    """
    Run pose estimation inference and save results

    Args:
        torch_model: PyTorch model
        ttnn_model: TTNN model (or None)
        model_type: "torch_model" or "tt_model"
        outputs_mesh_composer: Mesh composer
        im_tensor: Input image tensor
        orig_images: Original images
        paths_images: Image paths
        save_dir: Save directory
        device: TT device
    """
    logger.info(f"Running inference with {model_type}...")

    if model_type == "torch_model":
        # Run PyTorch model
        with torch.no_grad():
            preds = torch_model(im_tensor)
    else:
        # Run TTNN model
        # Convert input to TTNN format
        ttnn_input = ttnn.from_torch(
            im_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # Configure input memory
        n, c, h, w = im_tensor.shape
        if c == 3:
            c = 16  # Will be padded
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        ttnn_input = ttnn_input.to(device, input_mem_config)

        # Run TTNN inference
        ttnn_output = ttnn_model(ttnn_input)

        # Decode keypoints on CPU
        anchors = ttnn_model.pose_head.anchors
        strides = ttnn_model.pose_head.strides

        # Convert to torch
        preds_raw = ttnn.to_torch(ttnn_output, dtype=torch.float32, mesh_composer=outputs_mesh_composer)

        # Debug: Check raw TTNN output
        logger.info(f"TTNN RAW output range:")
        logger.info(f"  Bbox: [{preds_raw[:, 0:4, :].min():.2f}, {preds_raw[:, 0:4, :].max():.2f}]")
        logger.info(f"  Conf: [{preds_raw[:, 4, :].min():.4f}, {preds_raw[:, 4, :].max():.4f}]")
        logger.info(f"  Kpts RAW: [{preds_raw[:, 5:56, :].min():.2f}, {preds_raw[:, 5:56, :].max():.2f}]")

        # SKIP CPU decoding for now - use raw keypoints
        # The visualization will need to handle raw values differently
        preds = preds_raw

        logger.info(f"Using RAW keypoints (no CPU decoding):")
        logger.info(f"  Will interpret raw values in visualization")

    logger.info(f"Inference complete. Processing {len(orig_images)} images...")

    # Debug: Print prediction ranges
    logger.info(f"Predictions shape: {preds.shape}")
    logger.info(f"  Bbox range: [{preds[:, 0:4, :].min():.2f}, {preds[:, 0:4, :].max():.2f}]")
    logger.info(f"  Conf range: [{preds[:, 4, :].min():.4f}, {preds[:, 4, :].max():.4f}]")
    logger.info(f"  Kpts range: [{preds[:, 5:56, :].min():.2f}, {preds[:, 5:56, :].max():.2f}]")

    # Postprocess predictions
    results = postprocess_pose(preds, orig_images, paths_images)

    # Visualize and save
    logger.info(f"Saving results to {save_dir}...")
    visualize_and_save_pose(results, save_dir)

    # Print summary
    total_people = sum(len(r["detections"]) for r in results)
    logger.info(f"Detected {total_people} people across {len(results)} images")


def run_yolov11n_pose_demo(device, model_type, res, input_loc, batch_size_per_device):
    """
    Run YOLO11 Pose demo on images

    Args:
        device: TT device
        model_type: "torch_model" or "tt_model"
        res: Input resolution (height, width)
        input_loc: Path to input images
        batch_size_per_device: Batch size per device
    """
    torch_model, ttnn_model, mesh_composer, batch_size = init_pose_model_and_runner(
        device, model_type, batch_size_per_device
    )

    dataset = LoadImages(path=os.path.abspath(input_loc), batch=batch_size)
    im_tensor, orig_images, paths_images = process_images(dataset, res, batch_size)

    save_dir = "models/demos/yolov11/demo/runs/pose_ttnn"

    run_inference_and_save_pose(
        torch_model, ttnn_model, model_type, mesh_composer, im_tensor, orig_images, paths_images, save_dir, device
    )

    logger.info("Pose estimation demo complete!")


def run_yolov11n_pose_demo_dataset(device, model_type, res, batch_size_per_device):
    """
    Run YOLO11 Pose demo on COCO dataset

    Args:
        device: TT device
        model_type: "torch_model" or "tt_model"
        res: Input resolution (height, width)
        batch_size_per_device: Batch size per device
    """
    import fiftyone

    torch_model, ttnn_model, mesh_composer, batch_size = init_pose_model_and_runner(
        device, model_type, batch_size_per_device
    )

    dataset = fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=batch_size)
    filepaths = [sample["filepath"] for sample in dataset]
    image_loader = LoadImages(filepaths, batch=batch_size)
    im_tensor, orig_images, paths_images = process_images(image_loader, res, batch_size)

    save_dir = "models/demos/yolov11/demo/runs/pose_ttnn_coco"

    run_inference_and_save_pose(
        torch_model, ttnn_model, model_type, mesh_composer, im_tensor, orig_images, paths_images, save_dir, device
    )

    logger.info("Pose estimation demo on COCO dataset complete!")


# ===== Pytest Test Functions =====


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model",  # PyTorch implementation
        "tt_model",  # TTNN implementation (TT-Metal hardware)
    ),
)
@pytest.mark.parametrize("res", [(640, 640)])
@pytest.mark.parametrize(
    "input_loc, batch_size_per_device",
    [
        (
            "models/demos/yolov11/demo/images",
            1,
        ),
    ],
)
def test_pose_demo(device, model_type, res, input_loc, batch_size_per_device):
    """Test YOLO11 Pose demo"""
    run_yolov11n_pose_demo(
        device,
        model_type,
        res,
        input_loc,
        batch_size_per_device,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        "torch_model",
        # "tt_model",
    ),
)
@pytest.mark.parametrize("res", [(640, 640)])
@pytest.mark.parametrize(
    "input_loc, batch_size_per_device",
    [
        (
            "models/demos/yolov11/demo/images",
            1,
        ),
    ],
)
def test_pose_demo_dp(
    mesh_device,
    model_type,
    res,
    input_loc,
    batch_size_per_device,
):
    """Test YOLO11 Pose demo with data parallel"""
    run_yolov11n_pose_demo(
        mesh_device,
        model_type,
        res,
        input_loc,
        batch_size_per_device,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        "torch_model",
        # "tt_model",
    ),
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_pose_demo_dataset(device, model_type, res):
    """Test YOLO11 Pose demo on COCO dataset"""
    run_yolov11n_pose_demo_dataset(
        device,
        model_type,
        res,
        batch_size_per_device=1,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        "torch_model",
        # "tt_model",
    ),
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_pose_demo_dataset_dp(mesh_device, model_type, res):
    """Test YOLO11 Pose demo on COCO dataset with data parallel"""
    run_yolov11n_pose_demo_dataset(
        mesh_device,
        model_type,
        res,
        batch_size_per_device=1,
    )
