# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import fiftyone
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import disable_persistent_kernel_cache
from models.demos.utils.common_demo_utils import LoadImages, get_mesh_mappers, preprocess
from models.demos.yolov11.common import YOLOV11_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
from models.demos.yolov11.runner.performant_runner_pose import YOLOv11PosePerformantRunner

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


def init_pose_model_and_runner(
    device, model_type, use_weights_from_ultralytics, batch_size_per_device, model_location_generator
):
    """
    Initialize YOLO11 Pose model and TTNN runner

    Args:
        device: TT device
        model_type: "torch_model" or "tt_model"
        use_weights_from_ultralytics: Whether to load Ultralytics weights
        batch_size_per_device: Batch size per device
        model_location_generator: Function to generate model location

    Returns:
        model: PyTorch model
        performant_runner: TTNN runner (or None for torch_model)
        outputs_mesh_composer: Mesh composer
        batch_size: Total batch size
    """
    disable_persistent_kernel_cache()

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    logger.info(f"Running YOLO11 Pose with batch_size={batch_size} across {num_devices} devices")

    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    if use_weights_from_ultralytics:
        torch_model = load_torch_model(model_location_generator)
        model = torch_model.eval()
    else:
        model = YoloV11Pose()

    performant_runner = None
    if model_type == "tt_model":
        performant_runner = YOLOv11PosePerformantRunner(
            device,
            device_batch_size=batch_size_per_device,
            inputs_mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
            outputs_mesh_composer=outputs_mesh_composer,
        )

    return model, performant_runner, outputs_mesh_composer, batch_size


def process_images(dataset, res, batch_size):
    """Load and preprocess images for pose estimation"""
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


def postprocess_pose_predictions(
    preds,
    orig_images,
    paths_images,
    input_size=(640, 640),
    conf_threshold=0.5,
    nms_threshold=0.45,
):
    """
    Postprocess pose predictions with proper coordinate transformation

    Args:
        preds: Model predictions [batch, 56, num_anchors]
        orig_images: Original images
        paths_images: Image paths
        input_size: Model input size (height, width)
        conf_threshold: Confidence threshold for detections
        nms_threshold: NMS IoU threshold

    Returns:
        List of results with detections per image
    """
    results = []

    for i in range(len(orig_images)):
        orig_img = orig_images[i]
        orig_h, orig_w = orig_img.shape[:2]

        # Calculate scaling factors from preprocessing (letterboxing)
        scale = min(input_size[0] / orig_h, input_size[1] / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        # Calculate padding (letterbox centers the image)
        pad_top = (input_size[0] - new_h) // 2
        pad_left = (input_size[1] - new_w) // 2

        # Extract predictions for this image
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
                # Bbox in 640x640 space - transform back to original image space
                x, y, w, h = bbox[:, j]

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
                    {"bbox": [x1, y1, x2, y2], "confidence": float(conf[j]), "keypoints": kpts_transformed}
                )

        # Simple NMS (sort by confidence, remove overlapping detections)
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        if len(detections) > 1:
            # Simple overlap-based filtering
            filtered = [detections[0]]  # Keep highest confidence
            for det in detections[1:]:
                overlap = False
                for kept in filtered:
                    # Check if bounding boxes overlap significantly
                    ix1 = max(kept["bbox"][0], det["bbox"][0])
                    iy1 = max(kept["bbox"][1], det["bbox"][1])
                    ix2 = min(kept["bbox"][2], det["bbox"][2])
                    iy2 = min(kept["bbox"][3], det["bbox"][3])

                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        kept_area = (kept["bbox"][2] - kept["bbox"][0]) * (kept["bbox"][3] - kept["bbox"][1])
                        det_area = (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
                        union_area = kept_area + det_area - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0

                        if iou > nms_threshold:
                            overlap = True
                            break

                if not overlap:
                    filtered.append(det)

            detections = filtered

        results.append({"path": paths_images[i], "image": orig_images[i], "detections": detections})

    return results


def visualize_pose_detections(results, save_dir):
    """
    Visualize pose detections with keypoints and skeleton, save to disk

    Args:
        results: List of detection results per image
        save_dir: Directory to save visualized images
    """
    import cv2

    os.makedirs(save_dir, exist_ok=True)

    # Colors for different keypoints
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

        # Draw each person detection
        for person_idx, det in enumerate(detections):
            bbox = det["bbox"]
            conf = det["confidence"]
            keypoints = det["keypoints"]

            # Draw bounding box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"Person {person_idx+1}: {conf:.2f}",
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Draw keypoints
            visible_kpts = 0
            for kpt_idx, (x, y, v) in enumerate(keypoints):
                if v > 0.3:  # Keypoint is visible
                    visible_kpts += 1
                    color = colors[kpt_idx % len(colors)]
                    cv2.circle(img, (int(x), int(y)), 4, color, -1)
                    cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 1)

                    # Optional: Add keypoint labels for debugging
                    # cv2.putText(img, str(kpt_idx), (int(x) + 6, int(y) + 6),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            # Draw skeleton connections
            for connection in SKELETON:
                kpt1_idx = connection[0] - 1  # Convert to 0-based indexing
                kpt2_idx = connection[1] - 1

                if (
                    kpt1_idx < len(keypoints)
                    and kpt2_idx < len(keypoints)
                    and keypoints[kpt1_idx][2] > 0.3
                    and keypoints[kpt2_idx][2] > 0.3
                ):
                    pt1 = (int(keypoints[kpt1_idx][0]), int(keypoints[kpt1_idx][1]))
                    pt2 = (int(keypoints[kpt2_idx][0]), int(keypoints[kpt2_idx][1]))
                    cv2.line(img, pt1, pt2, (255, 255, 0), 2)

            logger.info(f"Person {person_idx+1}: {visible_kpts}/17 keypoints visible")

        # Save result
        basename = os.path.basename(result["path"])
        output_path = os.path.join(save_dir, f"pose_{basename}")
        cv2.imwrite(output_path, img)
        logger.info(f"Saved pose visualization: {output_path}")


def run_inference_and_save_pose(
    model, runner, model_type, outputs_mesh_composer, im_tensor, orig_images, paths_images, save_dir
):
    """
    Run pose estimation inference and save results

    Args:
        model: PyTorch model
        runner: TTNN runner (or None)
        model_type: "torch_model" or "tt_model"
        outputs_mesh_composer: Mesh composer
        im_tensor: Input tensor
        orig_images: Original images
        paths_images: Image paths
        save_dir: Save directory
    """
    if model_type == "torch_model":
        preds = model(im_tensor)
    else:
        preds = runner.run(im_tensor)
        preds = ttnn.to_torch(preds, dtype=torch.float32, mesh_composer=outputs_mesh_composer)

    # Debug: Print prediction statistics
    logger.info(f"Predictions shape: {preds.shape}")
    logger.info(f"  Bbox range: [{preds[:, 0:4, :].min():.2f}, {preds[:, 0:4, :].max():.2f}]")
    logger.info(f"  Conf range: [{preds[:, 4, :].min():.4f}, {preds[:, 4, :].max():.4f}]")
    logger.info(f"  Kpts range: [{preds[:, 5:56, :].min():.2f}, {preds[:, 5:56, :].max():.2f}]")

    # Postprocess predictions
    results = postprocess_pose_predictions(preds, orig_images, paths_images)

    # Visualize and save
    visualize_pose_detections(results, save_dir)

    # Print summary
    total_people = sum(len(r["detections"]) for r in results)
    logger.info(f"Detected {total_people} people across {len(results)} images")


def run_yolov11_pose_demo(
    device, model_type, use_weights_from_ultralytics, res, input_loc, batch_size_per_device, model_location_generator
):
    """
    Run YOLO11 Pose Estimation demo on images

    Args:
        device: TT device
        model_type: "torch_model" or "tt_model"
        use_weights_from_ultralytics: Load Ultralytics weights
        res: Input resolution (height, width)
        input_loc: Path to input images
        batch_size_per_device: Batch size per device
        model_location_generator: Model location generator
    """
    model, runner, mesh_composer, batch_size = init_pose_model_and_runner(
        device, model_type, use_weights_from_ultralytics, batch_size_per_device, model_location_generator
    )

    dataset = LoadImages(path=os.path.abspath(input_loc), batch=batch_size)
    im_tensor, orig_images, paths_images = process_images(dataset, res, batch_size)

    save_dir = "models/demos/yolov11/demo/runs/pose"

    run_inference_and_save_pose(
        model, runner, model_type, mesh_composer, im_tensor, orig_images, paths_images, save_dir
    )

    if runner:
        runner.release()
    logger.info("Pose estimation demo complete!")


def run_yolov11_pose_demo_dataset(
    device, model_type, use_weights_from_ultralytics, res, batch_size_per_device, model_location_generator
):
    """
    Run YOLO11 Pose Estimation demo on COCO dataset

    Args:
        device: TT device
        model_type: "torch_model" or "tt_model"
        use_weights_from_ultralytics: Load Ultralytics weights
        res: Input resolution (height, width)
        batch_size_per_device: Batch size per device
        model_location_generator: Model location generator
    """
    model, runner, mesh_composer, batch_size = init_pose_model_and_runner(
        device, model_type, use_weights_from_ultralytics, batch_size_per_device, model_location_generator
    )

    dataset = fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=batch_size)
    filepaths = [sample["filepath"] for sample in dataset]
    image_loader = LoadImages(filepaths, batch=batch_size)
    im_tensor, orig_images, paths_images = process_images(image_loader, res, batch_size)

    with open(os.path.expanduser("~") + "/fiftyone/coco-2017/info.json") as f:
        class_info = json.load(f)

    save_dir = "models/demos/yolov11/demo/runs/pose_coco"

    run_inference_and_save_pose(
        model, runner, model_type, mesh_composer, im_tensor, orig_images, paths_images, save_dir
    )

    if runner:
        runner.release()
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
        # "torch_model",  # Uncomment to run with PyTorch model
        "tt_model",  # TTNN implementation (TT-Metal hardware)
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [False],  # Use custom pose model weights to match TTNN implementation
)
@pytest.mark.parametrize("res", [(640, 640)])
@pytest.mark.parametrize(
    "input_loc, batch_size_per_device ",
    [
        (
            "models/demos/yolov11/demo/images",
            1,
        ),
    ],
)
def test_pose_demo(
    device, model_type, use_weights_from_ultralytics, res, input_loc, batch_size_per_device, model_location_generator
):
    """Test YOLO11 Pose Estimation demo"""
    run_yolov11_pose_demo(
        device,
        model_type,
        use_weights_from_ultralytics,
        res,
        input_loc,
        batch_size_per_device,
        model_location_generator,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model",  # Uncomment to run with PyTorch model
        "tt_model",  # TTNN implementation (TT-Metal hardware)
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [False],  # Use custom pose model weights to match TTNN implementation
)
@pytest.mark.parametrize("res", [(640, 640)])
@pytest.mark.parametrize(
    "input_loc, batch_size_per_device ",
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
    use_weights_from_ultralytics,
    res,
    input_loc,
    batch_size_per_device,
    model_location_generator,
):
    """Test YOLO11 Pose Estimation demo with data parallel"""
    run_yolov11_pose_demo(
        mesh_device,
        model_type,
        use_weights_from_ultralytics,
        res,
        input_loc,
        batch_size_per_device,
        model_location_generator,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model",  # Uncomment to run with PyTorch model
        "tt_model",  # TTNN implementation (TT-Metal hardware)
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [False],  # Use custom pose model weights to match TTNN implementation
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_pose_demo_dataset(device, model_type, use_weights_from_ultralytics, res, model_location_generator):
    """Test YOLO11 Pose Estimation demo on COCO dataset"""
    run_yolov11_pose_demo_dataset(
        device,
        model_type,
        use_weights_from_ultralytics,
        res,
        batch_size_per_device=1,
        model_location_generator=model_location_generator,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model",  # Uncomment to run with PyTorch model
        "tt_model",  # TTNN implementation (TT-Metal hardware)
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [False],  # Use custom pose model weights to match TTNN implementation
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_pose_demo_dataset_dp(mesh_device, model_type, use_weights_from_ultralytics, res, model_location_generator):
    """Test YOLO11 Pose Estimation demo on COCO dataset with data parallel"""
    run_yolov11_pose_demo_dataset(
        mesh_device,
        model_type,
        use_weights_from_ultralytics,
        res,
        batch_size_per_device=1,
        model_location_generator=model_location_generator,
    )
