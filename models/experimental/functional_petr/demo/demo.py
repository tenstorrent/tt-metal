# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone PETR Demo that works without MMDetection3D dependencies
This version uses mock/standalone implementations to avoid version conflicts
"""

import torch
import ttnn
import pytest
import numpy as np

# Import the TT-specific modules
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
    ParameterDict,
    ParameterList,
    fold_batch_norm2d_into_conv2d,
    infer_ttnn_module_args,
)

from torch.nn import Conv2d, Linear
from torch import nn
from models.experimental.functional_petr.reference.utils import LiDARInstance3DBoxes

import cv2

# ============= Mock Classes to Replace MMDet3D Dependencies =============
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless systems
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def transform_box(box, matrix):
    """
    Apply a 4x4 transformation matrix to a 3D bounding box.
    Args:
        box: numpy array of shape (7,) or (n,7) representing the 3D box parameters
             (x, y, z, w, l, h, yaw)
        matrix: numpy array of shape (4, 4) representing the transformation matrix
    Returns:
        box_transformed: numpy array of same shape as input box representing the transformed box
    """
    # Extract box parameters
    print(f"box shape: {box.shape}")
    print(f"box: {box}")
    # if box.shape == (7,):
    #     x, y, z, w, l, h, yaw = box
    # else:
    x, y, z, w, l, h, yaw = box[:7]

    # Create a 4x4 box transformation matrix
    box_to_lidar = np.eye(4)
    box_to_lidar[:3, :3] = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    box_to_lidar[:3, 3] = [x, y, z]

    # Multiply the box transformation matrix with the given matrix
    lidar_to_cam = np.matmul(matrix, box_to_lidar)

    # Extract transformed box parameters
    x, y, z = lidar_to_cam[:3, 3]
    yaw = np.arctan2(lidar_to_cam[1, 0], lidar_to_cam[0, 0])

    if box.shape == (7,):
        return np.array([x, y, z, w, l, h, yaw])
    else:
        return np.stack([x, y, z, w, l, h, yaw], axis=-1)


def project_3d_to_2d(box_3d, intrinsic):
    """
    Project a 3D bounding box to 2D using the camera intrinsic matrix.
    Args:
        box_3d: numpy array of shape (7,) representing the 3D box parameters
                (x, y, z, w, l, h, yaw) in the camera frame
        intrinsic: numpy array of shape (3, 3) representing the camera intrinsic matrix
    Returns:
        box_2d: tuple of (xmin, ymin, xmax, ymax) representing the 2D bounding box
    """
    # Extract box vertices in the camera frame
    x, y, z, w, l, h, yaw = box_3d
    vertices_3d = np.array(
        [
            [x - l / 2, y - w / 2, z - h / 2, 1],
            [x - l / 2, y + w / 2, z - h / 2, 1],
            [x + l / 2, y + w / 2, z - h / 2, 1],
            [x + l / 2, y - w / 2, z - h / 2, 1],
            [x - l / 2, y - w / 2, z + h / 2, 1],
            [x - l / 2, y + w / 2, z + h / 2, 1],
            [x + l / 2, y + w / 2, z + h / 2, 1],
            [x + l / 2, y - w / 2, z + h / 2, 1],
        ]
    )

    # Project vertices onto the image plane
    vertices_2d = np.matmul(intrinsic, vertices_3d[:, :3].T)

    # Check for zero values in the third row
    mask = vertices_2d[2, :] != 0
    vertices_2d = vertices_2d[:, mask]

    # Perform division only for non-zero values
    vertices_2d = vertices_2d[:2, :] / vertices_2d[2, :]
    vertices_2d = vertices_2d.T

    # Find the bounding box of the projected vertices
    if vertices_2d.shape[0] > 0:
        xmin, ymin = np.min(vertices_2d, axis=0)
        xmax, ymax = np.max(vertices_2d, axis=0)
        return xmin, ymin, xmax, ymax
    else:
        return None


def draw_box_on_image(img, box_2d, color=(0, 255, 0), thickness=2):
    """
    Draw a 2D bounding box on an image.
    Args:
        img: numpy array representing the image
        box_2d: tuple of (xmin, ymin, xmax, ymax) representing the 2D bounding box
        color: tuple of (B, G, R) values for the box color
        thickness: integer for the box line thickness
    Returns:
        None (the function modifies the input image in-place)
    """
    xmin, ymin, xmax, ymax = map(int, box_2d)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)


def save_petr_visualizations(ttnn_output, camera_images, img_metas, output_dir="./"):
    output_dict = ttnn_output[0]["pts_bbox"]
    boxes = output_dict["bboxes_3d"].tensor.cpu().numpy()
    scores = output_dict["scores_3d"].to(torch.float32).cpu().numpy()
    labels = output_dict["labels_3d"].to(torch.float32).cpu().numpy()

    # 1. Bird's Eye View (BEV) visualization
    fig, ax = plt.subplots(figsize=(10, 10))

    # Filter by confidence
    threshold = 0.14
    mask = scores > threshold
    filtered_boxes = boxes[mask][:50]  # Limit to 50 boxes
    cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    for cam_id, cam_name in enumerate(cam_names):
        img = camera_images[cam_id]

        # Get intrinsic and extrinsic matrices for this camera
        intrinsic = img_metas[0]["cam_intrinsic"][cam_id]
        lidar2cam = img_metas[0]["lidar2cam"][cam_id]

        for box in filtered_boxes:
            # Transform box from LiDAR to camera frame
            box_cam = transform_box(box, lidar2cam)

            # Project 3D box vertices to 2D
            box_2d = project_3d_to_2d(box_cam, intrinsic)
            if box_2d is not None:
                draw_box_on_image(img, box_2d)

            # Draw 2D box on image
            # draw_box_on_image(img, box_2d)

        # Save annotated image
        cv2.imwrite(f"{output_dir}/petr_{cam_name}.jpg", img)
    # Draw boxes
    for i, box in enumerate(filtered_boxes):
        x, y, z, w, l, h, yaw = box[:7]

        # Create rectangle
        rect = patches.Rectangle(
            (x - l / 2, y - w / 2), l, w, angle=np.degrees(yaw), linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        # Add confidence text
        ax.text(x, y, f"{scores[mask][i]:.2f}", fontsize=8)

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"PETR BEV Detection - {len(filtered_boxes)} objects")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.savefig(f"{output_dir}/petr_bev.jpg", dpi=100, bbox_inches="tight")
    plt.close()

    # 2. 3D scatter plot (side view)
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121)
    ax1.scatter(filtered_boxes[:, 0], filtered_boxes[:, 2], c="blue", s=20)
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Z (meters)")
    ax1.set_title("Side View (X-Z)")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(122)
    ax2.scatter(filtered_boxes[:, 1], filtered_boxes[:, 2], c="green", s=20)
    ax2.set_xlabel("Y (meters)")
    ax2.set_ylabel("Z (meters)")
    ax2.set_title("Side View (Y-Z)")
    ax2.grid(True, alpha=0.3)

    plt.savefig(f"{output_dir}/petr_3d_views.jpg", dpi=100, bbox_inches="tight")
    plt.close()

    print(f"Saved visualizations:")
    print(f"  - petr_bev.jpg")
    print(f"  - petr_3d_views.jpg")

    return filtered_boxes


class MockLiDARInstance3DBoxes:
    """Mock 3D bounding box class"""

    def __init__(self, tensor=None):
        if tensor is None:
            self.tensor = torch.zeros(1, 9)
        else:
            self.tensor = tensor


class MockDet3DDataPreprocessor:
    """
    Standalone implementation of Det3DDataPreprocessor
    Processes the saved data without requiring MMDetection3D
    """

    def __init__(
        self, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], bgr_to_rgb=False, pad_size_divisor=32
    ):
        self.mean = torch.tensor(mean).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 1, 3, 1, 1)
        self.bgr_to_rgb = bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor

    def __call__(self, data, training=False):
        """Process the input data"""

        # Handle different input formats
        if isinstance(data, str):
            data = torch.load(data)

        # Extract images - handle different formats
        if "imgs" in data:
            imgs = data["imgs"]
        elif "inputs" in data and "imgs" in data["inputs"]:
            imgs = data["inputs"]["imgs"]
        else:
            # Create dummy images
            print("Warning: No images found, creating dummy data")
            imgs = torch.zeros(1, 6, 3, 320, 800)

        # Ensure tensor and correct shape
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs).float()

        # Add batch dimension if needed
        if len(imgs.shape) == 4:  # [cameras, channels, height, width]
            imgs = imgs.unsqueeze(0)

        # Normalize if needed (check if already normalized)
        if imgs.max() > 10.0:  # Assume unnormalized if large values
            imgs = (imgs - self.mean) / self.std

        # Pad to divisible size
        b, c, ch, h, w = 1, 6, 3, 320, 800
        pad_h = (self.pad_size_divisor - h % self.pad_size_divisor) % self.pad_size_divisor
        pad_w = (self.pad_size_divisor - w % self.pad_size_divisor) % self.pad_size_divisor

        if pad_h > 0 or pad_w > 0:
            imgs = torch.nn.functional.pad(imgs, (0, pad_w, 0, pad_h))

        # Create output structure
        output = {"inputs": {"imgs": imgs}, "data_samples": []}

        # Create data samples with metadata
        batch_size = imgs.shape[0]
        for i in range(batch_size):
            sample = type("DataSample", (), {})()  # Create simple object

            # Extract or create metadata
            # print(f"data: {data}")
            if "img_metas" in data:
                if isinstance(data["img_metas"], list):
                    metainfo = data["img_metas"][i] if i < len(data["img_metas"]) else data["img_metas"][0]
                else:
                    metainfo = data["img_metas"]
                if "cam2img" not in metainfo:
                    metainfo["cam2img"] = [np.eye(4, dtype=np.float32) for _ in range(6)]
                if "lidar2img" not in metainfo:
                    metainfo["lidar2img"] = [np.eye(4, dtype=np.float32) for _ in range(6)]
                if "lidar2cam" not in metainfo:
                    metainfo["lidar2cam"] = [np.eye(4, dtype=np.float32) for _ in range(6)]
                if "cam2lidar" not in metainfo:
                    metainfo["cam2lidar"] = [np.eye(4, dtype=np.float32) for _ in range(6)]
                if "ego2global" not in metainfo:
                    metainfo["ego2global"] = np.eye(4, dtype=np.float32)
                if "cam_intrinsic" not in metainfo:
                    metainfo["cam_intrinsic"] = [
                        np.array([[1000, 0, 800], [0, 1000, 450], [0, 0, 1]]) for _ in range(6)
                    ]
                if "img_shape" in metainfo:
                    metainfo["img_shape"] = (h, w)
                if "pad_shape" in metainfo:
                    metainfo["pad_shape"] = (h, w)
                if "ori_shape" in metainfo:
                    metainfo["ori_shape"] = [(h, w)] * 6
                # if 'box_mode_3d' in metainfo:
                #     metainfo['box_mode_3d'] = 'LiDAR'
                # if 'box_type_3d' in metainfo:
                #     metainfo['box_type_3d'] = MockLiDARInstance3DBoxes
                # if 'sample_idx' in metainfo:
                #     metainfo['sample_idx'] = i
            else:
                # Create default metadata
                metainfo = {
                    "filename": [f"cam_{j}.jpg" for j in range(6)],
                    "ori_shape": [(h, w)] * 6,
                    "img_shape": (h, w),
                    "pad_shape": (h + pad_h, w + pad_w),
                    "box_mode_3d": "LiDAR",
                    "box_type_3d": MockLiDARInstance3DBoxes,
                    "sample_idx": i,
                    "lidar2img": [np.eye(4) for _ in range(6)],
                    "cam_intrinsic": [np.array([[1000, 0, 800], [0, 1000, 450], [0, 0, 1]]) for _ in range(6)],
                    "cam2img": [np.eye(4, dtype=np.float32) for _ in range(6)],
                    "lidar2cam": [np.eye(4, dtype=np.float32) for _ in range(6)],
                    "cam2lidar": [np.eye(4, dtype=np.float32) for _ in range(6)],
                    "ego2global": np.eye(4, dtype=np.float32),
                }

            # Ensure box_type_3d is set correctly
            metainfo["box_type_3d"] = LiDARInstance3DBoxes
            # print(f"metainfo: {metainfo}")

            sample.metainfo = metainfo
            output["data_samples"].append(sample)

        return output


# ============= Import PETR modules (assuming these exist) =============
try:
    from models.experimental.functional_petr.reference.petr import PETR
    from models.experimental.functional_petr.reference.petr_head import PETRHead, pos2posemb3d
    from models.experimental.functional_petr.reference.cp_fpn import CPFPN
    from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP, eSEModule, _OSA_module
    from models.experimental.functional_petr.tt.ttnn_petr import ttnn_PETR
except ImportError as e:
    print(f"Warning: Could not import PETR modules: {e}")
    print("Make sure the PETR reference implementation is available")


# ============= Helper Functions (same as original) =============


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["input_proj", "adapt_pos3d", "position_encoder"]:
                continue
            object[name] = move_to_device(value, device)
        return object
    elif isinstance(object, ParameterList):
        for index, element in enumerate(object):
            object[index] = move_to_device(element, device)
        return object
    elif isinstance(object, ttnn.Tensor):
        return ttnn.to_device(object, device)
    else:
        return object


def stem_parameters_preprocess(model):
    parameters = {}
    if isinstance(model, VoVNetCP):
        if hasattr(model, "stem"):
            layers = list(model.stem.named_children())
        for i, (name, layer) in enumerate(layers):
            if "conv" in name:
                conv_name, conv_layer = layers[i]
                norm_name, norm_layer = layers[i + 1]
                prefix = conv_name.split("/")[0]
                if prefix not in parameters:
                    parameters[prefix] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv_layer, norm_layer)
                parameters[prefix]["weight"] = conv_weight
                parameters[prefix]["bias"] = conv_bias
    return parameters


def create_custom_preprocessor_cpfpn(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, CPFPN):
            parameters["lateral_convs"] = {}
            for i, child in enumerate(model.lateral_convs):
                parameters["lateral_convs"][i] = {}
                parameters["lateral_convs"][i]["conv"] = {}
                parameters["lateral_convs"][i]["conv"]["weight"] = ttnn.from_torch(
                    child.conv.weight, dtype=ttnn.bfloat16
                )
                parameters["lateral_convs"][i]["conv"]["bias"] = ttnn.from_torch(
                    torch.reshape(child.conv.bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )
            parameters["fpn_convs"] = {}
            for i, child in enumerate(model.fpn_convs):
                parameters["fpn_convs"][i] = {}
                parameters["fpn_convs"][i]["conv"] = {}
                parameters["fpn_convs"][i]["conv"]["weight"] = ttnn.from_torch(child.conv.weight, dtype=ttnn.bfloat16)
                parameters["fpn_convs"][i]["conv"]["bias"] = ttnn.from_torch(
                    torch.reshape(child.conv.bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )
        return parameters

    return custom_preprocessor


def create_custom_preprocessor_vovnetcp(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, eSEModule):
            parameters["fc"] = {}
            parameters["fc"]["weight"] = ttnn.from_torch(model.fc.weight, dtype=ttnn.bfloat16)
            parameters["fc"]["bias"] = ttnn.from_torch(torch.reshape(model.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)
        if isinstance(model, _OSA_module):
            if hasattr(model, "conv_reduction"):
                first_layer_name, _ = list(model.conv_reduction.named_children())[0]
                base_name = first_layer_name.split("/")[0]
                parameters[base_name] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.conv_reduction[0], model.conv_reduction[1])
                parameters[base_name]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[base_name]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            for i, layers in enumerate(model.layers):
                first_layer_name = list(layers.named_children())[0][0]
                prefix = first_layer_name.split("/")[0]
                parameters[prefix] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(layers[0], layers[1])
                if "OSA2_1" in prefix:
                    parameters[prefix]["weight"] = conv_weight
                    parameters[prefix]["bias"] = conv_bias
                else:
                    parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters[prefix]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )

            first_layer_name, _ = list(model.concat.named_children())[0]
            base_name = first_layer_name.split("/")[0]
            parameters[base_name] = {}
            if "OSA2_1" in base_name:
                parameters[base_name]["weight"] = model.concat[0].weight
                parameters[base_name]["bias"] = model.concat[0].bias
            else:
                concat_weight, concat_bias = fold_batch_norm2d_into_conv2d(model.concat[0], model.concat[1])
                parameters[base_name]["weight"] = ttnn.from_torch(concat_weight, dtype=ttnn.bfloat16)
                parameters[base_name]["bias"] = ttnn.from_torch(
                    torch.reshape(concat_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            parameters["fc"] = {}
            parameters["fc"]["weight"] = ttnn.from_torch(model.ese.fc.weight, dtype=ttnn.bfloat16)
            parameters["fc"]["bias"] = ttnn.from_torch(
                torch.reshape(model.ese.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )
        return parameters

    return custom_preprocessor


def create_custom_preprocessor_petr_head(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, PETRHead):
            parameters["input_proj"] = {}
            parameters["input_proj"]["weight"] = ttnn.from_torch(model.input_proj.weight, dtype=ttnn.bfloat16)
            parameters["input_proj"]["bias"] = ttnn.from_torch(
                torch.reshape(model.input_proj.bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
            )

            parameters["cls_branches"] = {}
            for index, child in enumerate(model.cls_branches):
                parameters["cls_branches"][index] = {}
                for index1, child1 in enumerate(child):
                    parameters["cls_branches"][index][index1] = {}
                    if isinstance(child1, Linear):
                        parameters["cls_branches"][index][index1]["weight"] = preprocess_linear_weight(
                            child1.weight, dtype=ttnn.bfloat8_b
                        )
                        parameters["cls_branches"][index][index1]["bias"] = preprocess_linear_bias(
                            child1.bias, dtype=ttnn.bfloat8_b
                        )
                    elif isinstance(child1, nn.LayerNorm):
                        parameters["cls_branches"][index][index1]["weight"] = preprocess_layernorm_parameter(
                            child1.weight, dtype=ttnn.bfloat8_b
                        )
                        parameters["cls_branches"][index][index1]["bias"] = preprocess_layernorm_parameter(
                            child1.bias, dtype=ttnn.bfloat8_b
                        )

            parameters["reg_branches"] = {}
            for index, child in enumerate(model.reg_branches):
                parameters["reg_branches"][index] = {}
                for index1, child1 in enumerate(child):
                    parameters["reg_branches"][index][index1] = {}
                    if isinstance(child1, Linear):
                        parameters["reg_branches"][index][index1]["weight"] = preprocess_linear_weight(
                            child1.weight, dtype=ttnn.bfloat8_b
                        )
                        parameters["reg_branches"][index][index1]["bias"] = preprocess_linear_bias(
                            child1.bias, dtype=ttnn.bfloat8_b
                        )

            # Handle other components
            for comp_name in ["adapt_pos3d", "position_encoder"]:
                if hasattr(model, comp_name):
                    parameters[comp_name] = {}
                    component = getattr(model, comp_name)
                    for index, child in enumerate(component):
                        parameters[comp_name][index] = {}
                        if isinstance(child, Conv2d):
                            parameters[comp_name][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.bfloat16)
                            parameters[comp_name][index]["bias"] = ttnn.from_torch(
                                torch.reshape(child.bias, (1, 1, 1, -1)),
                                dtype=ttnn.bfloat16,
                            )

            if hasattr(model, "query_embedding"):
                parameters["query_embedding"] = {}
                for index, child in enumerate(model.query_embedding):
                    parameters["query_embedding"][index] = {}
                    if isinstance(child, Linear):
                        parameters["query_embedding"][index]["weight"] = preprocess_linear_weight(
                            child.weight, dtype=ttnn.bfloat8_b
                        )
                        parameters["query_embedding"][index]["bias"] = preprocess_linear_bias(
                            child.bias, dtype=ttnn.bfloat8_b
                        )

            if hasattr(model, "reference_points"):
                parameters["reference_points"] = {}
                parameters["reference_points"]["weight"] = ttnn.from_torch(model.reference_points.weight, device=device)

        return parameters

    return custom_preprocessor


def load_real_nuscenes_images(data_root="models/experimental/functional_petr/resources/nuscenes"):
    """Load real images directly from nuScenes samples"""
    import cv2
    from pathlib import Path

    cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    imgs_list = []
    for cam in cam_names:
        img_path = Path(data_root) / "samples" / cam

        img_files = list(img_path.glob("*.jpg"))[:1]  # Get first image

        if img_files:
            img = cv2.imread(str(img_files[0]))
            img = cv2.resize(img, (800, 320))
            print(f"Loaded: {cam}/{img_files[0].name}")
        else:
            img = np.zeros((320, 800, 3), dtype=np.uint8)
            print(f"Using dummy for {cam}")
        imgs_list.append(img)

    # Convert to tensor format
    imgs = np.stack(imgs_list).astype(np.float32)
    mean = np.array([103.530, 116.280, 123.675])  # BGR mean
    std = np.array([57.375, 57.120, 58.395])  # BGR std
    imgs = (imgs - mean) / std
    imgs = imgs.transpose(0, 3, 1, 2)
    imgs = torch.from_numpy(imgs).float().unsqueeze(0)
    print(f"   imgs: {imgs.shape}")
    # imgs = imgs.transpose(0, 3, 1, 2)  # to [6, C, H, W]
    # imgs = torch.from_numpy(imgs).unsqueeze(0)  # [1, 6, C, H, W]
    # imgs = imgs.squeeze(0)
    # Create data dict
    return {
        "imgs": imgs,
        "img_metas": [
            {
                "filename": cam_names,
                "ori_shape": (320, 800),
                "img_shape": (320, 800),
                "pad_shape": (320, 800),
                "cam2img": [np.eye(4, dtype=np.float32) for _ in range(6)],
                "lidar2img": [np.eye(4, dtype=np.float32) for _ in range(6)],
                "cam_intrinsic": [np.array([[1000, 0, 400], [0, 1000, 160], [0, 0, 1]]) for _ in range(6)],
            }
        ],
    }, imgs_list


# ============= Main Demo Function =============


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_demo(device, reset_seeds):
    """
    Main PETR demo function
    """
    print("=" * 60)
    print("PETR TT Demo - Standalone Version")
    print("=" * 60)

    # Load preprocessed data
    data = False
    # print(f"\n1. Loading preprocessed data from: {data_path}")
    # else:

    try:
        if data == True:
            data_path = "models/experimental/functional_petr/demo/data_passed_to_data_preprocessor.pt"
            input_data = torch.load(data_path, weights_only=False)
        else:
            print("Loading real nuScenes images...")
            input_data, camera_images = load_real_nuscenes_images(
                "models/experimental/functional_petr/resources/nuscenes"
            )
        print(f"   Data loaded successfully")
    except FileNotFoundError:
        print(f"   WARNING: Data file not found. Creating dummy data...")
        # Create dummy data for testing
        # input_data = {
        #     'imgs': torch.randn(6, 3, 320, 800),  # 6 cameras
        #     'img_metas': [{
        #         'filename': [f'cam_{i}.jpg' for i in range(6)],
        #         'ori_shape': [(900, 1600, 3)] * 6,
        #         'img_shape': [(320, 800, 3)] * 6,
        #         'box_type_3d': MockLiDARInstance3DBoxes,
        #     }]
        # }
    # print(f"   Input data: {input_data['imgs'].shape}")

    # Initialize data preprocessor (mock version)
    print("\n2. Initializing data preprocessor...")
    data_preprocessor = MockDet3DDataPreprocessor(
        mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], bgr_to_rgb=False, pad_size_divisor=32
    )
    # print(f"   Data preprocessor initialized", data_preprocessor.mean.shape, data_preprocessor.std.shape)
    # Process the input data
    print("3. Processing input data...")
    output_after_preprocess = data_preprocessor(input_data, False)
    # print(f"   Output after preprocess: {output_after_preprocess['inputs']['imgs'].shape}")
    batch_img_metas = [ds.metainfo for ds in output_after_preprocess["data_samples"]]

    print(f"   Images shape: {output_after_preprocess['inputs']['imgs'].shape}")
    print(f"   Batch size: {len(batch_img_metas)}")

    # Load model weights
    print("\n4. Loading PETR model weights...")
    weights_path = "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth"

    try:
        weights_state_dict = torch.load(weights_path, weights_only=False)["state_dict"]
        print(f"   Weights loaded successfully")
    except FileNotFoundError:
        print(f"   WARNING: Weight file not found at {weights_path}")
        print("   Model will use random initialization")
        weights_state_dict = None

    # Initialize PyTorch model
    print("\n5. Initializing PyTorch PETR model...")
    torch_model = PETR(use_grid_mask=True)

    if weights_state_dict:
        torch_model.load_state_dict(weights_state_dict)

    sample_weight = torch_model.img_backbone.stem[0].weight
    print(f"Sample weight stats: mean={sample_weight.mean():.4f}, std={sample_weight.std():.4f}")
    # sample_bias = torch_model.img_backbone.stem[0].bias
    # print(f"Sample bias stats: mean={sample_bias.mean():.4f}, std={sample_bias.std():.4f}")

    torch_model.eval()
    print("   Model initialized and set to eval mode")

    # Run PyTorch inference for comparison
    print("\n6. Running PyTorch inference...")
    with torch.no_grad():
        # img_feats = torch_model.extract_img_feat(
        #             output_after_preprocess["inputs"]["imgs"],
        #             batch_img_metas
        #     )
        # print(f"   img_feats: {img_feats}")
        # print(f"Feature stats: mean={img_feats[0].mean():.4f}, std={img_feats[0].std():.4f}")
        # print(f"   batch_img_metas: {batch_img_metas}")
        # print(f"   batch_img_metas[0]: {batch_img_metas[0].shape}")
        print(f"   output_after_preprocess['inputs']: {output_after_preprocess['inputs']['imgs'].shape}")
        # print(f"   output_after_preprocess['inputs'].shape: {output_after_preprocess['inputs'].shape}")
        torch_output = torch_model.predict(output_after_preprocess["inputs"], batch_img_metas)

    if torch_output and len(torch_output) > 0:
        print(f"   PyTorch output: {len(torch_output)} predictions")
        if "boxes_3d" in torch_output[0]:
            print(f"   Number of detected boxes: {len(torch_output[0]['boxes_3d'])}")

    # Convert to TTNN
    print("\n7. Converting to TTNN format...")
    ttnn_inputs = dict()
    imgs_tensor = output_after_preprocess["inputs"]["imgs"]
    if len(imgs_tensor.shape) == 4:  # [6, 3, 320, 800]
        imgs_tensor = imgs_tensor.unsqueeze(0)
    ttnn_inputs["imgs"] = ttnn.from_torch(imgs_tensor, device=device)

    ttnn_batch_img_metas = batch_img_metas.copy()
    # print(f"ttnn_inputs: {ttnn_inputs}")
    # print(f"ttnn_batch_img_metas: {ttnn_batch_img_metas}")

    # Preprocess parameters
    print("\n8. Preprocessing model parameters for TT...")

    # Head parameters
    parameters_petr_head = preprocess_model_parameters(
        initialize_model=lambda: torch_model.pts_bbox_head,
        custom_preprocessor=create_custom_preprocessor_petr_head(None),
        device=None,
    )
    parameters_petr_head = move_to_device(parameters_petr_head, device)

    # Transformer parameters
    child = torch_model.pts_bbox_head.transformer
    x = infer_ttnn_module_args(
        model=child,
        run_model=lambda model: model(
            torch.randn(1, 6, 256, 20, 50),
            torch.zeros((1, 6, 20, 50), dtype=torch.bool),
            torch.rand(900, 256),
            torch.rand(1, 6, 256, 20, 50),
        ),
        device=None,
    )
    if x is not None:
        for key in x.keys():
            x[key].module = getattr(child, key)
        parameters_petr_head["transformer"] = x

    # Neck and backbone parameters
    parameters_petr_cpfpn = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_neck,
        custom_preprocessor=create_custom_preprocessor_cpfpn(None),
        device=None,
    )

    parameters_petr_vovnetcp = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_backbone,
        custom_preprocessor=create_custom_preprocessor_vovnetcp(None),
        device=None,
    )

    # Combine parameters
    parameters = {
        "pts_bbox_head": parameters_petr_head,
        "img_neck": parameters_petr_cpfpn,
        "img_backbone": parameters_petr_vovnetcp,
        "stem_parameters": stem_parameters_preprocess(torch_model.img_backbone),
    }

    print("   Parameters preprocessed successfully")

    # Preprocess query embedding
    print("\n9. Preprocessing query embeddings...")
    query_embedding_input = torch_model.pts_bbox_head.reference_points.weight
    query_embedding_input = pos2posemb3d(query_embedding_input)
    query_embedding_input = ttnn.from_torch(query_embedding_input, layout=ttnn.TILE_LAYOUT, device=device)

    # Initialize TTNN model
    print("\n10. Initializing TTNN PETR model...")
    ttnn_model = ttnn_PETR(
        use_grid_mask=True,
        parameters=parameters,
        query_embedding_input=query_embedding_input,
        device=device,
    )

    # Run TTNN inference
    print("\n11. Running TTNN inference...")
    ttnn_output = ttnn_model.predict(ttnn_inputs, ttnn_batch_img_metas)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    if ttnn_output:
        print(f"TTNN Output: {ttnn_output}")
        if isinstance(ttnn_output, list) and len(ttnn_output) > 0:
            for i, pred in enumerate(ttnn_output):
                print(f"\nPrediction {i}:")
                if isinstance(pred, dict):
                    for key, value in pred.items():
                        if isinstance(value, (torch.Tensor, ttnn.Tensor)):
                            print(f"  {key}: shape={value.shape}")
                        else:
                            print(f"  {key}: {value}")
    else:
        print("No detections or empty output")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

    return ttnn_output, camera_images, ttnn_batch_img_metas


# ============= Standalone Execution =============

if __name__ == "__main__":
    print("Running PETR Demo in standalone mode...")
    print("Note: This requires a TT device to be available")

    # Create a mock device for CPU testing
    # In actual TT environment, this would be the real device
    # try:
    import ttnn

    device = ttnn.open_device(device_id=0, l1_small_size=32768)  # Open device 0
    # except:
    #     print("Warning: Could not open TT device, using CPU mode")
    #     device = None

    # Run the demo
    try:
        ttnn_output, camera_images, ttnn_batch_img_metas = test_demo(device, None)
        filtered_boxes = save_petr_visualizations(ttnn_output, camera_images, ttnn_batch_img_metas)
    finally:
        if device:
            ttnn.close_device(device)
