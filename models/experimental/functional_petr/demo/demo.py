# # SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# # SPDX-License-Identifier: Apache-2.0

# """
# Standalone PETR Demo that works without MMDetection3D dependencies
# This version uses mock/standalone implementations to avoid version conflicts
# """

# import torch
# import ttnn
# import pytest
# import numpy as np

# # Import the TT-specific modules
# from ttnn.model_preprocessing import (
#     preprocess_model_parameters,
#     preprocess_linear_weight,
#     preprocess_layernorm_parameter,
#     preprocess_linear_bias,
#     ParameterDict,
#     ParameterList,
#     fold_batch_norm2d_into_conv2d,
#     infer_ttnn_module_args,
# )

# from torch.nn import Conv2d, Linear
# from torch import nn
# from models.experimental.functional_petr.reference.utils import LiDARInstance3DBoxes
# from loguru import logger
# from models.experimental.functional_petr.reference.petr import PETR
# from models.experimental.functional_petr.reference.petr_head import PETRHead, pos2posemb3d
# from models.experimental.functional_petr.reference.cp_fpn import CPFPN
# from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP, eSEModule, _OSA_module, _OSA_stage
# from models.experimental.functional_petr.tt.ttnn_petr import ttnn_PETR

# import cv2

# # ============= Mock Classes to Replace MMDet3D Dependencies =============
# import matplotlib

# matplotlib.use("Agg")  # Use non-interactive backend for headless systems
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


# def transform_box(box, matrix):
#     """
#     Apply a 4x4 transformation matrix to a 3D bounding box.
#     Args:
#         box: numpy array of shape (7,) or (n,7) representing the 3D box parameters
#              (x, y, z, w, l, h, yaw)
#         matrix: numpy array of shape (4, 4) representing the transformation matrix
#     Returns:
#         box_transformed: numpy array of same shape as input box representing the transformed box
#     """
#     # Extract box parameters
#     print(f"box shape: {box.shape}")
#     print(f"box: {box}")
#     # if box.shape == (7,):
#     #     x, y, z, w, l, h, yaw = box
#     # else:
#     x, y, z, w, l, h, yaw = box[:7]

#     # Create a 4x4 box transformation matrix
#     box_to_lidar = np.eye(4)
#     box_to_lidar[:3, :3] = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
#     box_to_lidar[:3, 3] = [x, y, z]

#     # Multiply the box transformation matrix with the given matrix
#     lidar_to_cam = np.matmul(matrix, box_to_lidar)

#     # Extract transformed box parameters
#     x, y, z = lidar_to_cam[:3, 3]
#     yaw = np.arctan2(lidar_to_cam[1, 0], lidar_to_cam[0, 0])

#     if box.shape == (7,):
#         return np.array([x, y, z, w, l, h, yaw])
#     else:
#         return np.stack([x, y, z, w, l, h, yaw], axis=-1)


# def project_3d_to_2d(box_3d, intrinsic):
#     """
#     Project a 3D bounding box to 2D using the camera intrinsic matrix.
#     Args:
#         box_3d: numpy array of shape (7,) representing the 3D box parameters
#                 (x, y, z, w, l, h, yaw) in the camera frame
#         intrinsic: numpy array of shape (3, 3) representing the camera intrinsic matrix
#     Returns:
#         box_2d: tuple of (xmin, ymin, xmax, ymax) representing the 2D bounding box
#     """
#     # Extract box vertices in the camera frame
#     x, y, z, w, l, h, yaw = box_3d
#     vertices_3d = np.array(
#         [
#             [x - l / 2, y - w / 2, z - h / 2, 1],
#             [x - l / 2, y + w / 2, z - h / 2, 1],
#             [x + l / 2, y + w / 2, z - h / 2, 1],
#             [x + l / 2, y - w / 2, z - h / 2, 1],
#             [x - l / 2, y - w / 2, z + h / 2, 1],
#             [x - l / 2, y + w / 2, z + h / 2, 1],
#             [x + l / 2, y + w / 2, z + h / 2, 1],
#             [x + l / 2, y - w / 2, z + h / 2, 1],
#         ]
#     )

#     # Project vertices onto the image plane
#     vertices_2d = np.matmul(intrinsic, vertices_3d[:, :3].T)

#     # Check for zero values in the third row
#     mask = vertices_2d[2, :] != 0
#     vertices_2d = vertices_2d[:, mask]

#     # Perform division only for non-zero values
#     vertices_2d = vertices_2d[:2, :] / vertices_2d[2, :]
#     vertices_2d = vertices_2d.T

#     # Find the bounding box of the projected vertices
#     if vertices_2d.shape[0] > 0:
#         xmin, ymin = np.min(vertices_2d, axis=0)
#         xmax, ymax = np.max(vertices_2d, axis=0)
#         return xmin, ymin, xmax, ymax
#     else:
#         return None


# def draw_box_on_image(img, box_2d, color=(0, 255, 0), thickness=2):
#     """
#     Draw a 2D bounding box on an image.
#     Args:
#         img: numpy array representing the image
#         box_2d: tuple of (xmin, ymin, xmax, ymax) representing the 2D bounding box
#         color: tuple of (B, G, R) values for the box color
#         thickness: integer for the box line thickness
#     Returns:
#         None (the function modifies the input image in-place)
#     """
#     xmin, ymin, xmax, ymax = map(int, box_2d)
#     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)


# def save_petr_visualizations(ttnn_output, camera_images, img_metas, output_dir="./"):
#     output_dict = ttnn_output[0]["pts_bbox"]
#     boxes = output_dict["bboxes_3d"].tensor.cpu().numpy()
#     scores = output_dict["scores_3d"].to(torch.float32).cpu().numpy()
#     labels = output_dict["labels_3d"].to(torch.float32).cpu().numpy()

#     # 1. Bird's Eye View (BEV) visualization
#     fig, ax = plt.subplots(figsize=(10, 10))

#     # Filter by confidence
#     threshold = 0.14
#     mask = scores > threshold
#     filtered_boxes = boxes[mask][:50]  # Limit to 50 boxes
#     cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

#     for cam_id, cam_name in enumerate(cam_names):
#         img = camera_images[cam_id]

#         # Get intrinsic and extrinsic matrices for this camera
#         intrinsic = img_metas[0]["cam_intrinsic"][cam_id]
#         lidar2cam = img_metas[0]["lidar2cam"][cam_id]

#         for box in filtered_boxes:
#             # Transform box from LiDAR to camera frame
#             box_cam = transform_box(box, lidar2cam)

#             # Project 3D box vertices to 2D
#             box_2d = project_3d_to_2d(box_cam, intrinsic)
#             if box_2d is not None:
#                 draw_box_on_image(img, box_2d)

#             # Draw 2D box on image
#             # draw_box_on_image(img, box_2d)

#         # Save annotated image
#         cv2.imwrite(f"{output_dir}/petr_{cam_name}.jpg", img)
#     # Draw boxes
#     for i, box in enumerate(filtered_boxes):
#         x, y, z, w, l, h, yaw = box[:7]

#         # Create rectangle
#         rect = patches.Rectangle(
#             (x - l / 2, y - w / 2), l, w, angle=np.degrees(yaw), linewidth=2, edgecolor="red", facecolor="none"
#         )
#         ax.add_patch(rect)

#         # Add confidence text
#         ax.text(x, y, f"{scores[mask][i]:.2f}", fontsize=8)

#     ax.set_xlim(-50, 50)
#     ax.set_ylim(-50, 50)
#     ax.set_xlabel("X (meters)")
#     ax.set_ylabel("Y (meters)")
#     ax.set_title(f"PETR BEV Detection - {len(filtered_boxes)} objects")
#     ax.grid(True, alpha=0.3)
#     ax.set_aspect("equal")

#     plt.savefig(f"{output_dir}/petr_bev.jpg", dpi=100, bbox_inches="tight")
#     plt.close()

#     # 2. 3D scatter plot (side view)
#     fig = plt.figure(figsize=(12, 6))

#     ax1 = fig.add_subplot(121)
#     ax1.scatter(filtered_boxes[:, 0], filtered_boxes[:, 2], c="blue", s=20)
#     ax1.set_xlabel("X (meters)")
#     ax1.set_ylabel("Z (meters)")
#     ax1.set_title("Side View (X-Z)")
#     ax1.grid(True, alpha=0.3)

#     ax2 = fig.add_subplot(122)
#     ax2.scatter(filtered_boxes[:, 1], filtered_boxes[:, 2], c="green", s=20)
#     ax2.set_xlabel("Y (meters)")
#     ax2.set_ylabel("Z (meters)")
#     ax2.set_title("Side View (Y-Z)")
#     ax2.grid(True, alpha=0.3)

#     plt.savefig(f"{output_dir}/petr_3d_views.jpg", dpi=100, bbox_inches="tight")
#     plt.close()

#     print(f"Saved visualizations:")
#     print(f"  - petr_bev.jpg")
#     print(f"  - petr_3d_views.jpg")

#     return filtered_boxes


# class MockLiDARInstance3DBoxes:
#     """Mock 3D bounding box class"""

#     def __init__(self, tensor=None):
#         if tensor is None:
#             self.tensor = torch.zeros(1, 9)
#         else:
#             self.tensor = tensor


# class MockDet3DDataPreprocessor:
#     """
#     Standalone implementation of Det3DDataPreprocessor
#     Processes the saved data without requiring MMDetection3D
#     """

#     def __init__(
#         self, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], bgr_to_rgb=False, pad_size_divisor=32
#     ):
#         self.mean = torch.tensor(mean).view(1, 1, 3, 1, 1)
#         self.std = torch.tensor(std).view(1, 1, 3, 1, 1)
#         self.bgr_to_rgb = bgr_to_rgb
#         self.pad_size_divisor = pad_size_divisor

#     def __call__(self, data, training=False):
#         """Process the input data"""

#         # Handle different input formats
#         if isinstance(data, str):
#             data = torch.load(data)

#         # Extract images - handle different formats
#         if "imgs" in data:
#             imgs = data["imgs"]
#         elif "inputs" in data and "imgs" in data["inputs"]:
#             imgs = data["inputs"]["imgs"]
#         else:
#             # Create dummy images
#             print("Warning: No images found, creating dummy data")
#             imgs = torch.zeros(1, 6, 3, 320, 800)

#         # Ensure tensor and correct shape
#         if isinstance(imgs, np.ndarray):
#             imgs = torch.from_numpy(imgs).float()

#         # Add batch dimension if needed
#         if len(imgs.shape) == 4:  # [cameras, channels, height, width]
#             imgs = imgs.unsqueeze(0)

#         # Normalize if needed (check if already normalized)
#         if imgs.max() > 10.0:  # Assume unnormalized if large values
#             imgs = (imgs - self.mean) / self.std

#         # Pad to divisible size
#         b, c, ch, h, w = 1, 6, 3, 320, 800
#         pad_h = (self.pad_size_divisor - h % self.pad_size_divisor) % self.pad_size_divisor
#         pad_w = (self.pad_size_divisor - w % self.pad_size_divisor) % self.pad_size_divisor

#         if pad_h > 0 or pad_w > 0:
#             imgs = torch.nn.functional.pad(imgs, (0, pad_w, 0, pad_h))

#         # Create output structure
#         output = {"inputs": {"imgs": imgs}, "data_samples": []}

#         # Create data samples with metadata
#         batch_size = imgs.shape[0]
#         for i in range(batch_size):
#             sample = type("DataSample", (), {})()  # Create simple object

#             # Extract or create metadata
#             # print(f"data: {data}")
#             if "img_metas" in data:
#                 if isinstance(data["img_metas"], list):
#                     metainfo = data["img_metas"][i] if i < len(data["img_metas"]) else data["img_metas"][0]
#                 else:
#                     metainfo = data["img_metas"]
#                 if "cam2img" not in metainfo:
#                     metainfo["cam2img"] = [np.eye(4, dtype=np.float32) for _ in range(6)]
#                 if "lidar2img" not in metainfo:
#                     metainfo["lidar2img"] = [np.eye(4, dtype=np.float32) for _ in range(6)]
#                 if "lidar2cam" not in metainfo:
#                     metainfo["lidar2cam"] = [np.eye(4, dtype=np.float32) for _ in range(6)]
#                 if "cam2lidar" not in metainfo:
#                     metainfo["cam2lidar"] = [np.eye(4, dtype=np.float32) for _ in range(6)]
#                 if "ego2global" not in metainfo:
#                     metainfo["ego2global"] = np.eye(4, dtype=np.float32)
#                 if "cam_intrinsic" not in metainfo:
#                     metainfo["cam_intrinsic"] = [
#                         np.array([[1000, 0, 800], [0, 1000, 450], [0, 0, 1]]) for _ in range(6)
#                     ]
#                 if "img_shape" in metainfo:
#                     metainfo["img_shape"] = (h, w)
#                 if "pad_shape" in metainfo:
#                     metainfo["pad_shape"] = (h, w)
#                 if "ori_shape" in metainfo:
#                     metainfo["ori_shape"] = [(h, w)] * 6
#                 # if 'box_mode_3d' in metainfo:
#                 #     metainfo['box_mode_3d'] = 'LiDAR'
#                 # if 'box_type_3d' in metainfo:
#                 #     metainfo['box_type_3d'] = MockLiDARInstance3DBoxes
#                 # if 'sample_idx' in metainfo:
#                 #     metainfo['sample_idx'] = i
#             else:
#                 # Create default metadata
#                 metainfo = {
#                     "filename": [f"cam_{j}.jpg" for j in range(6)],
#                     "ori_shape": [(h, w)] * 6,
#                     "img_shape": (h, w),
#                     "pad_shape": (h + pad_h, w + pad_w),
#                     "box_mode_3d": "LiDAR",
#                     "box_type_3d": MockLiDARInstance3DBoxes,
#                     "sample_idx": i,
#                     "lidar2img": [np.eye(4) for _ in range(6)],
#                     "cam_intrinsic": [np.array([[1000, 0, 800], [0, 1000, 450], [0, 0, 1]]) for _ in range(6)],
#                     "cam2img": [np.eye(4, dtype=np.float32) for _ in range(6)],
#                     "lidar2cam": [np.eye(4, dtype=np.float32) for _ in range(6)],
#                     "cam2lidar": [np.eye(4, dtype=np.float32) for _ in range(6)],
#                     "ego2global": np.eye(4, dtype=np.float32),
#                 }

#             # Ensure box_type_3d is set correctly
#             metainfo["box_type_3d"] = LiDARInstance3DBoxes
#             # print(f"metainfo: {metainfo}")

#             sample.metainfo = metainfo
#             output["data_samples"].append(sample)

#         return output


# # ============= Import PETR modules (assuming these exist) =============
# try:
#     from models.experimental.functional_petr.reference.petr import PETR
#     from models.experimental.functional_petr.reference.petr_head import PETRHead, pos2posemb3d
#     from models.experimental.functional_petr.reference.cp_fpn import CPFPN
#     from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP, eSEModule, _OSA_module
#     from models.experimental.functional_petr.tt.ttnn_petr import ttnn_PETR
# except ImportError as e:
#     print(f"Warning: Could not import PETR modules: {e}")
#     print("Make sure the PETR reference implementation is available")


# # ============= Helper Functions (same as original) =============

# def move_to_device(object, device):
#     if isinstance(object, ParameterDict):
#         for name, value in list(object.items()):
#             if name in ["input_proj", "adapt_pos3d", "position_encoder"]:
#                 continue
#             object[name] = move_to_device(value, device)
#         return object
#     elif isinstance(object, ParameterList):
#         for index, element in enumerate(object):
#             object[index] = move_to_device(element, device)
#         return object
#     elif isinstance(object, ttnn.Tensor):
#         return ttnn.to_device(object, device)
#     else:
#         return object


# def stem_parameters_preprocess(model):
#     parameters = {}
#     if isinstance(model, VoVNetCP):
#         if hasattr(model, "stem"):
#             layers = list(model.stem.named_children())

#         for i, (name, layer) in enumerate(layers):
#             if "conv" in name:
#                 conv_name, conv_layer = layers[i]
#                 norm_name, norm_layer = layers[i + 1]
#                 prefix = conv_name.split("/")[0]

#                 if prefix not in parameters:
#                     parameters[prefix] = {}

#                 conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv_layer, norm_layer)

#                 logger.info(
#                     f"[PREPROCESS] {prefix}: weight shape={conv_weight.shape}, mean={conv_weight.mean():.6f}, std={conv_weight.std():.6f}"
#                 )

#                 # Convert to ttnn format (same as other Conv layers)
#                 parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
#                 parameters[prefix]["bias"] = ttnn.from_torch(
#                     torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
#                 )

#     return parameters


# def create_custom_preprocessor_cpfpn(device):
#     def custom_preprocessor(model, name, ttnn_module_args):
#         parameters = {}
#         if isinstance(model, CPFPN):
#             parameters["lateral_convs"] = {}
#             for i, child in enumerate(model.lateral_convs):
#                 parameters["lateral_convs"][i] = {}
#                 parameters["lateral_convs"][i]["conv"] = {}
#                 parameters["lateral_convs"][i]["conv"]["weight"] = ttnn.from_torch(
#                     child.conv.weight, dtype=ttnn.bfloat16
#                 )
#                 parameters["lateral_convs"][i]["conv"]["bias"] = ttnn.from_torch(
#                     torch.reshape(child.conv.bias, (1, 1, 1, -1)),
#                     dtype=ttnn.bfloat16,
#                 )
#             parameters["fpn_convs"] = {}
#             for i, child in enumerate(model.fpn_convs):
#                 parameters["fpn_convs"][i] = {}
#                 parameters["fpn_convs"][i]["conv"] = {}
#                 parameters["fpn_convs"][i]["conv"]["weight"] = ttnn.from_torch(child.conv.weight, dtype=ttnn.bfloat16)
#                 parameters["fpn_convs"][i]["conv"]["bias"] = ttnn.from_torch(
#                     torch.reshape(child.conv.bias, (1, 1, 1, -1)),
#                     dtype=ttnn.bfloat16,
#                 )
#         return parameters

#     return custom_preprocessor


# def create_custom_preprocessor_vovnetcp(device):
#     def custom_preprocessor(model, name, ttnn_module_args):
#         parameters = {}
#         if isinstance(model, eSEModule):
#             parameters["fc"] = {}
#             parameters["fc"]["weight"] = ttnn.from_torch(model.fc.weight, dtype=ttnn.bfloat16)
#             parameters["fc"]["bias"] = ttnn.from_torch(torch.reshape(model.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)
#         if isinstance(model, _OSA_module):
#             if hasattr(model, "conv_reduction"):
#                 first_layer_name, _ = list(model.conv_reduction.named_children())[0]
#                 base_name = first_layer_name.split("/")[0]
#                 parameters[base_name] = {}
#                 conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.conv_reduction[0], model.conv_reduction[1])
#                 parameters[base_name]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
#                 parameters[base_name]["bias"] = ttnn.from_torch(
#                     torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
#                 )

#             for i, layers in enumerate(model.layers):
#                 first_layer_name = list(layers.named_children())[0][0]
#                 prefix = first_layer_name.split("/")[0]
#                 parameters[prefix] = {}
#                 conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(layers[0], layers[1])
#                 # if "OSA2_1" in prefix:
#                 #     parameters[prefix]["weight"] = conv_weight
#                 #     parameters[prefix]["bias"] = conv_bias
#                 # else:
#                 parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
#                 parameters[prefix]["bias"] = ttnn.from_torch(
#                     torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
#                 )

#             first_layer_name, _ = list(model.concat.named_children())[0]
#             base_name = first_layer_name.split("/")[0]
#             parameters[base_name] = {}
#             # if "OSA2_1" in base_name:
#             #     parameters[base_name]["weight"] = model.concat[0].weight
#             #     parameters[base_name]["bias"] = model.concat[0].bias
#             # else:
#             concat_weight, concat_bias = fold_batch_norm2d_into_conv2d(model.concat[0], model.concat[1])
#             parameters[base_name]["weight"] = ttnn.from_torch(concat_weight, dtype=ttnn.bfloat16)
#             parameters[base_name]["bias"] = ttnn.from_torch(
#                 torch.reshape(concat_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
#             )

#             parameters["fc"] = {}
#             parameters["fc"]["weight"] = ttnn.from_torch(model.ese.fc.weight, dtype=ttnn.bfloat16)
#             parameters["fc"]["bias"] = ttnn.from_torch(
#                 torch.reshape(model.ese.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
#             )
#         if isinstance(model, _OSA_stage):
#             if isinstance(model, _OSA_module):
#                 if hasattr(model, "conv_reduction"):
#                     first_layer_name, _ = list(model.conv_reduction.named_children())[0]
#                     base_name = first_layer_name.split("/")[0]
#                     parameters[base_name] = {}
#                     conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
#                         model.conv_reduction[0], model.conv_reduction[1]
#                     )
#                     parameters[base_name]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
#                     parameters[base_name]["bias"] = ttnn.from_torch(
#                         torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
#                     )

#                 for i, layers in enumerate(model.layers):
#                     first_layer_name = list(layers.named_children())[0][0]
#                     prefix = first_layer_name.split("/")[0]
#                     parameters[prefix] = {}
#                     conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(layers[0], layers[1])
#                     parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
#                     parameters[prefix]["bias"] = ttnn.from_torch(
#                         torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
#                     )

#                 first_layer_name, _ = list(model.concat.named_children())[0]
#                 base_name = first_layer_name.split("/")[0]
#                 parameters[base_name] = {}
#                 parameters[base_name]["weight"] = model.concat[0].weight
#                 parameters[base_name]["bias"] = model.concat[0].bias

#                 parameters["fc"] = {}
#                 parameters["fc"]["weight"] = ttnn.from_torch(model.ese.fc.weight, dtype=ttnn.bfloat16)
#                 parameters["fc"]["bias"] = ttnn.from_torch(
#                     torch.reshape(model.ese.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
#                 )

#         return parameters

#     return custom_preprocessor


# def create_custom_preprocessor_petr_head(device):
#     def custom_preprocessor(model, name, ttnn_module_args):
#         parameters = {}
#         if isinstance(model, PETRHead):
#             parameters["input_proj"] = {}
#             parameters["input_proj"]["weight"] = ttnn.from_torch(model.input_proj.weight, dtype=ttnn.bfloat16)
#             parameters["input_proj"]["bias"] = ttnn.from_torch(
#                 torch.reshape(model.input_proj.bias, (1, 1, 1, -1)),
#                 dtype=ttnn.bfloat16,
#             )

#             parameters["cls_branches"] = {}
#             for index, child in enumerate(model.cls_branches):
#                 parameters["cls_branches"][index] = {}
#                 for index1, child1 in enumerate(child):
#                     parameters["cls_branches"][index][index1] = {}
#                     if isinstance(child1, Linear):
#                         parameters["cls_branches"][index][index1]["weight"] = preprocess_linear_weight(
#                             child1.weight, dtype=ttnn.bfloat16
#                         )
#                         parameters["cls_branches"][index][index1]["bias"] = preprocess_linear_bias(
#                             child1.bias, dtype=ttnn.bfloat16
#                         )
#                     elif isinstance(child1, nn.LayerNorm):
#                         parameters["cls_branches"][index][index1]["weight"] = preprocess_layernorm_parameter(
#                             child1.weight, dtype=ttnn.bfloat16
#                         )
#                         parameters["cls_branches"][index][index1]["bias"] = preprocess_layernorm_parameter(
#                             child1.bias, dtype=ttnn.bfloat16
#                         )

#             parameters["reg_branches"] = {}
#             for index, child in enumerate(model.reg_branches):
#                 parameters["reg_branches"][index] = {}
#                 for index1, child1 in enumerate(child):
#                     parameters["reg_branches"][index][index1] = {}
#                     if isinstance(child1, Linear):
#                         parameters["reg_branches"][index][index1]["weight"] = preprocess_linear_weight(
#                             child1.weight, dtype=ttnn.bfloat16
#                         )
#                         parameters["reg_branches"][index][index1]["bias"] = preprocess_linear_bias(
#                             child1.bias, dtype=ttnn.bfloat16
#                         )

#             parameters["adapt_pos3d"] = {}
#             for index, child in enumerate(model.adapt_pos3d):
#                 parameters["adapt_pos3d"][index] = {}
#                 if isinstance(child, Conv2d):
#                     parameters["adapt_pos3d"][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.bfloat16)
#                     parameters["adapt_pos3d"][index]["bias"] = ttnn.from_torch(
#                         torch.reshape(child.bias, (1, 1, 1, -1)),
#                         dtype=ttnn.bfloat16,
#                     )

#             parameters["position_encoder"] = {}
#             for index, child in enumerate(model.position_encoder):
#                 parameters["position_encoder"][index] = {}
#                 if isinstance(child, Conv2d):
#                     parameters["position_encoder"][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.bfloat16)
#                     parameters["position_encoder"][index]["bias"] = ttnn.from_torch(
#                         torch.reshape(child.bias, (1, 1, 1, -1)),
#                         dtype=ttnn.bfloat16,
#                     )

#             parameters["query_embedding"] = {}
#             for index, child in enumerate(model.query_embedding):
#                 parameters["query_embedding"][index] = {}
#                 if isinstance(child, Linear):
#                     parameters["query_embedding"][index]["weight"] = preprocess_linear_weight(
#                         child.weight, dtype=ttnn.bfloat16
#                     )
#                     parameters["query_embedding"][index]["bias"] = preprocess_linear_bias(
#                         child.bias, dtype=ttnn.bfloat16
#                     )
#             parameters["reference_points"] = {}
#             parameters["reference_points"]["weight"] = ttnn.from_torch(model.reference_points.weight, device=device)

#         return parameters

#     return custom_preprocessor


# def load_real_nuscenes_images(data_root="models/experimental/functional_petr/resources/nuscenes"):
#     """Load real images directly from nuScenes samples"""
#     import cv2
#     from pathlib import Path

#     cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

#     imgs_list = []
#     for cam in cam_names:
#         img_path = Path(data_root) / "samples" / cam

#         img_files = list(img_path.glob("*.jpg"))[:1]  # Get first image

#         if img_files:
#             img = cv2.imread(str(img_files[0]))
#             img = cv2.resize(img, (800, 320))
#             print(f"Loaded: {cam}/{img_files[0].name}")
#         else:
#             img = np.zeros((320, 800, 3), dtype=np.uint8)
#             print(f"Using dummy for {cam}")
#         imgs_list.append(img)

#     # Convert to tensor format
#     imgs = np.stack(imgs_list).astype(np.float32)
#     mean = np.array([103.530, 116.280, 123.675])  # BGR mean
#     std = np.array([57.375, 57.120, 58.395])  # BGR std
#     imgs = (imgs - mean) / std
#     imgs = imgs.transpose(0, 3, 1, 2)
#     imgs = torch.from_numpy(imgs).float().unsqueeze(0)
#     print(f"   imgs: {imgs.shape}")
#     # imgs = imgs.transpose(0, 3, 1, 2)  # to [6, C, H, W]
#     # imgs = torch.from_numpy(imgs).unsqueeze(0)  # [1, 6, C, H, W]
#     # imgs = imgs.squeeze(0)
#     # Create data dict
#     return {
#         "imgs": imgs,
#         "img_metas": [
#             {
#                 "filename": cam_names,
#                 "ori_shape": (320, 800),
#                 "img_shape": (320, 800),
#                 "pad_shape": (320, 800),
#                 "cam2img": [np.eye(4, dtype=np.float32) for _ in range(6)],
#                 "lidar2img": [np.eye(4, dtype=np.float32) for _ in range(6)],
#                 "cam_intrinsic": [np.array([[1000, 0, 400], [0, 1000, 160], [0, 0, 1]]) for _ in range(6)],
#             }
#         ],
#     }, imgs_list


# # ============= Main Demo Function =============


# @pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
# def test_demo(device, reset_seeds):
#     """
#     Main PETR demo function
#     """
#     print("=" * 60)
#     print("PETR TT Demo - Standalone Version")
#     print("=" * 60)

#     # Load preprocessed data
#     data = False
#     # print(f"\n1. Loading preprocessed data from: {data_path}")
#     # else:

#     try:
#         if data == True:
#             data_path = "models/experimental/functional_petr/demo/data_passed_to_data_preprocessor.pt"
#             input_data = torch.load(data_path, weights_only=False)
#         else:
#             print("Loading real nuScenes images...")
#             input_data, camera_images = load_real_nuscenes_images(
#                 "models/experimental/functional_petr/resources/nuscenes"
#             )
#         print(f"   Data loaded successfully")
#     except FileNotFoundError:
#         print(f"   WARNING: Data file not found. Creating dummy data...")
#         # Create dummy data for testing
#         # input_data = {
#         #     'imgs': torch.randn(6, 3, 320, 800),  # 6 cameras
#         #     'img_metas': [{
#         #         'filename': [f'cam_{i}.jpg' for i in range(6)],
#         #         'ori_shape': [(900, 1600, 3)] * 6,
#         #         'img_shape': [(320, 800, 3)] * 6,
#         #         'box_type_3d': MockLiDARInstance3DBoxes,
#         #     }]
#         # }
#     # print(f"   Input data: {input_data['imgs'].shape}")

#     # Initialize data preprocessor (mock version)
#     print("\n2. Initializing data preprocessor...")
#     data_preprocessor = MockDet3DDataPreprocessor(
#         mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], bgr_to_rgb=False, pad_size_divisor=32
#     )
#     # print(f"   Data preprocessor initialized", data_preprocessor.mean.shape, data_preprocessor.std.shape)
#     # Process the input data
#     print("3. Processing input data...")
#     output_after_preprocess = data_preprocessor(input_data, False)
#     # print(f"   Output after preprocess: {output_after_preprocess['inputs']['imgs'].shape}")
#     batch_img_metas = [ds.metainfo for ds in output_after_preprocess["data_samples"]]

#     print(f"   Images shape: {output_after_preprocess['inputs']['imgs'].shape}")
#     print(f"   Batch size: {len(batch_img_metas)}")

#     # Load model weights
#     print("\n4. Loading PETR model weights...")
#     weights_path = "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth"

#     try:
#         weights_state_dict = torch.load(weights_path, weights_only=False)["state_dict"]
#         print(f"   Weights loaded successfully")
#     except FileNotFoundError:
#         print(f"   WARNING: Weight file not found at {weights_path}")
#         print("   Model will use random initialization")
#         weights_state_dict = None

#     # Initialize PyTorch model
#     print("\n5. Initializing PyTorch PETR model...")
#     torch_model = PETR(use_grid_mask=True)

#     if weights_state_dict:
#         torch_model.load_state_dict(weights_state_dict)

#     sample_weight = torch_model.img_backbone.stem[0].weight
#     print(f"Sample weight stats: mean={sample_weight.mean():.4f}, std={sample_weight.std():.4f}")
#     # sample_bias = torch_model.img_backbone.stem[0].bias
#     # print(f"Sample bias stats: mean={sample_bias.mean():.4f}, std={sample_bias.std():.4f}")

#     torch_model.eval()
#     print("   Model initialized and set to eval mode")

#     # Run PyTorch inference for comparison
#     print("\n6. Running PyTorch inference...")
#     with torch.no_grad():
#         # img_feats = torch_model.extract_img_feat(
#         #             output_after_preprocess["inputs"]["imgs"],
#         #             batch_img_metas
#         #     )
#         # print(f"   img_feats: {img_feats}")
#         # print(f"Feature stats: mean={img_feats[0].mean():.4f}, std={img_feats[0].std():.4f}")
#         # print(f"   batch_img_metas: {batch_img_metas}")
#         # print(f"   batch_img_metas[0]: {batch_img_metas[0].shape}")
#         print(f"   output_after_preprocess['inputs']: {output_after_preprocess['inputs']['imgs'].shape}")
#         # print(f"   output_after_preprocess['inputs'].shape: {output_after_preprocess['inputs'].shape}")
#         torch_output = torch_model.predict(output_after_preprocess["inputs"], batch_img_metas)

#     if torch_output and len(torch_output) > 0:
#         print(f"   PyTorch output: {len(torch_output)} predictions")
#         if "boxes_3d" in torch_output[0]:
#             print(f"   Number of detected boxes: {len(torch_output[0]['boxes_3d'])}")

#     # Convert to TTNN
#     print("\n7. Converting to TTNN format...")
#     ttnn_inputs = dict()
#     imgs_tensor = output_after_preprocess["inputs"]["imgs"]
#     if len(imgs_tensor.shape) == 4:  # [6, 3, 320, 800]
#         imgs_tensor = imgs_tensor.unsqueeze(0)
#     ttnn_inputs["imgs"] = ttnn.from_torch(imgs_tensor, device=device)

#     ttnn_batch_img_metas = batch_img_metas.copy()
#     # print(f"ttnn_inputs: {ttnn_inputs}")
#     # print(f"ttnn_batch_img_metas: {ttnn_batch_img_metas}")

#     # Preprocess parameters
#     print("\n8. Preprocessing model parameters for TT...")

#     # Head parameters
#     parameters_petr_head = preprocess_model_parameters(
#         initialize_model=lambda: torch_model.pts_bbox_head,
#         custom_preprocessor=create_custom_preprocessor_petr_head(None),
#         device=None,
#     )
#     parameters_petr_head = move_to_device(parameters_petr_head, device)

#     # Transformer parameters
#     child = torch_model.pts_bbox_head.transformer
#     x = infer_ttnn_module_args(
#         model=child,
#         run_model=lambda model: model(
#             torch.randn(1, 6, 256, 20, 50),
#             torch.zeros((1, 6, 20, 50), dtype=torch.bool),
#             torch.rand(900, 256),
#             torch.rand(1, 6, 256, 20, 50),
#         ),
#         device=None,
#     )
#     if x is not None:
#         for key in x.keys():
#             x[key].module = getattr(child, key)
#         parameters_petr_head["transformer"] = x

#     # Neck and backbone parameters
#     parameters_petr_cpfpn = preprocess_model_parameters(
#         initialize_model=lambda: torch_model.img_neck,
#         custom_preprocessor=create_custom_preprocessor_cpfpn(None),
#         device=None,
#     )

#     parameters_petr_vovnetcp = preprocess_model_parameters(
#         initialize_model=lambda: torch_model.img_backbone,
#         custom_preprocessor=create_custom_preprocessor_vovnetcp(None),
#         device=None,
#     )

#     # Combine parameters
#     parameters = {
#         "pts_bbox_head": parameters_petr_head,
#         "img_neck": parameters_petr_cpfpn,
#         "img_backbone": parameters_petr_vovnetcp,
#         "stem_parameters": stem_parameters_preprocess(torch_model.img_backbone),
#     }

#     print("   Parameters preprocessed successfully")

#     # Preprocess query embedding
#     print("\n9. Preprocessing query embeddings...")
#     query_embedding_input = torch_model.pts_bbox_head.reference_points.weight
#     query_embedding_input = pos2posemb3d(query_embedding_input)
#     query_embedding_input = ttnn.from_torch(query_embedding_input, layout=ttnn.TILE_LAYOUT, device=device)

#     # Initialize TTNN model
#     print("\n10. Initializing TTNN PETR model...")
#     ttnn_model = ttnn_PETR(
#         use_grid_mask=True,
#         parameters=parameters,
#         query_embedding_input=query_embedding_input,
#         device=device,
#     )

#     # Run TTNN inference
#     print("\n11. Running TTNN inference...")
#     ttnn_output = ttnn_model.predict(ttnn_inputs, ttnn_batch_img_metas)

#     # Display results
#     print("\n" + "=" * 60)
#     print("RESULTS:")
#     print("=" * 60)

#     if ttnn_output:
#         print(f"TTNN Output: {ttnn_output}")
#         if isinstance(ttnn_output, list) and len(ttnn_output) > 0:
#             for i, pred in enumerate(ttnn_output):
#                 print(f"\nPrediction {i}:")
#                 if isinstance(pred, dict):
#                     for key, value in pred.items():
#                         if isinstance(value, (torch.Tensor, ttnn.Tensor)):
#                             print(f"  {key}: shape={value.shape}")
#                         else:
#                             print(f"  {key}: {value}")
#     else:
#         print("No detections or empty output")

#     print("\n" + "=" * 60)
#     print("Demo completed successfully!")
#     print("=" * 60)

#     return ttnn_output, camera_images, ttnn_batch_img_metas


# # ============= Standalone Execution =============

# if __name__ == "__main__":
#     print("Running PETR Demo in standalone mode...")
#     print("Note: This requires a TT device to be available")

#     # Create a mock device for CPU testing
#     # In actual TT environment, this would be the real device
#     # try:
#     import ttnn

#     device = ttnn.open_device(device_id=0, l1_small_size=32768)  # Open device 0
#     # except:
#     #     print("Warning: Could not open TT device, using CPU mode")
#     #     device = None

#     # Run the demo
#     try:
#         ttnn_output, camera_images, ttnn_batch_img_metas = test_demo(device, None)
#         filtered_boxes = save_petr_visualizations(ttnn_output, camera_images, ttnn_batch_img_metas)
#     finally:
#         if device:
#             ttnn.close_device(device)


# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone PETR Demo that works without MMDetection3D dependencies
Fixed version based on mmdetection3d/tools/test.py
"""

import torch
import ttnn
import pytest
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

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
from loguru import logger
from models.experimental.functional_petr.reference.petr import PETR
from models.experimental.functional_petr.reference.petr_head import PETRHead, pos2posemb3d
from models.experimental.functional_petr.reference.cp_fpn import CPFPN
from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP, eSEModule, _OSA_module, _OSA_stage
from models.experimental.functional_petr.tt.ttnn_petr import ttnn_PETR


def load_nuscenes_info(data_root="models/experimental/functional_petr/resources/nuscenes"):
    """
    Load proper nuScenes camera calibration information
    This should match the format from mmdetection3d
    """
    # Standard nuScenes camera intrinsics (approximate)
    cam_intrinsics = {
        "CAM_FRONT": np.array([[1252.8, 0.0, 826.6], [0.0, 1252.8, 469.98], [0.0, 0.0, 1.0]]),
        "CAM_FRONT_RIGHT": np.array([[1256.7, 0.0, 827.2], [0.0, 1256.7, 450.5], [0.0, 0.0, 1.0]]),
        "CAM_FRONT_LEFT": np.array([[1257.9, 0.0, 827.2], [0.0, 1257.9, 450.7], [0.0, 0.0, 1.0]]),
        "CAM_BACK": np.array([[809.2, 0.0, 829.2], [0.0, 809.2, 481.8], [0.0, 0.0, 1.0]]),
        "CAM_BACK_LEFT": np.array([[1256.7, 0.0, 826.4], [0.0, 1256.7, 479.8], [0.0, 0.0, 1.0]]),
        "CAM_BACK_RIGHT": np.array([[1259.5, 0.0, 807.3], [0.0, 1259.5, 501.2], [0.0, 0.0, 1.0]]),
    }

    # Approximate camera extrinsics (lidar to camera transformation)
    # These are example values and should be replaced with actual calibration
    lidar2cam = {
        "CAM_FRONT": np.array(
            [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0], [1.0, 0.0, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
        ),
        "CAM_FRONT_RIGHT": np.array(
            [[-0.7071, -0.7071, 0.0, 0.7], [0.0, 0.0, -1.0, -1.0], [0.7071, -0.7071, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
        ),
        "CAM_FRONT_LEFT": np.array(
            [[0.7071, -0.7071, 0.0, -0.7], [0.0, 0.0, -1.0, -1.0], [0.7071, 0.7071, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
        ),
        "CAM_BACK": np.array(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0], [-1.0, 0.0, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
        ),
        "CAM_BACK_LEFT": np.array(
            [[0.7071, 0.7071, 0.0, -0.7], [0.0, 0.0, -1.0, -1.0], [-0.7071, 0.7071, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
        ),
        "CAM_BACK_RIGHT": np.array(
            [[-0.7071, 0.7071, 0.0, 0.7], [0.0, 0.0, -1.0, -1.0], [-0.7071, -0.7071, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
        ),
    }

    return cam_intrinsics, lidar2cam


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point."""
    ndim = int(dims.shape[0])
    corners_norm = np.stack(np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(dims.dtype)

    if ndim == 2:
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]

    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([1, ndim]) * corners_norm.reshape([-1, ndim])
    return corners


def project_to_image(points_3d, proj_mat):
    """Project 3D points to 2D image plane."""
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.ones(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def draw_lidar_bbox3d_on_img(bboxes3d, img, lidar2img, img_shape, color=(0, 255, 0), thickness=2):
    """Draw 3D bounding boxes on image."""

    if isinstance(lidar2img, torch.Tensor):
        lidar2img = lidar2img.cpu().numpy()

    if len(bboxes3d) == 0:
        return img

    boxes_drawn = 0
    boxes_skipped_behind = 0
    boxes_skipped_outside = 0

    for bbox in bboxes3d:
        # Get 8 corners of the 3D box
        l, w, h = bbox[3:6]
        x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
        y_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
        z_corners = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
        corners = np.vstack([x_corners, y_corners, z_corners])  # 3x8

        # Rotate
        yaw = bbox[6]
        rot_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        corners = rot_mat @ corners  # 3x8

        # Translate
        corners[0, :] += bbox[0]
        corners[1, :] += bbox[1]
        corners[2, :] += bbox[2]

        # corners is now 3x8, transpose to 8x3
        corners = corners.T  # 8x3

        # Project to image - add homogeneous coordinate
        pts_4d = np.concatenate([corners, np.ones((8, 1))], axis=-1)  # 8x4
        pts_2d = pts_4d @ lidar2img.T  # 8x4

        # Check if points are in front of camera
        if (pts_2d[:, 2] <= 0.1).all():  # Changed threshold
            boxes_skipped_behind += 1
            continue  # All points behind camera

        # Normalize
        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e10)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]

        corners_2d = pts_2d[:, :2].astype(np.int32)

        # Check if box is visible in image - be more lenient
        x_min, x_max = corners_2d[:, 0].min(), corners_2d[:, 0].max()
        y_min, y_max = corners_2d[:, 1].min(), corners_2d[:, 1].max()

        # Allow boxes that are partially visible
        if x_max < -200 or x_min > img_shape[1] + 200 or y_max < -200 or y_min > img_shape[0] + 200:
            boxes_skipped_outside += 1
            continue  # Box not visible

        # Draw the 12 edges of the 3D box
        # Bottom face (indices 0,1,2,3)
        cv2.line(img, tuple(corners_2d[0]), tuple(corners_2d[1]), color, thickness)
        cv2.line(img, tuple(corners_2d[1]), tuple(corners_2d[2]), color, thickness)
        cv2.line(img, tuple(corners_2d[2]), tuple(corners_2d[3]), color, thickness)
        cv2.line(img, tuple(corners_2d[3]), tuple(corners_2d[0]), color, thickness)

        # Top face (indices 4,5,6,7)
        cv2.line(img, tuple(corners_2d[4]), tuple(corners_2d[5]), color, thickness)
        cv2.line(img, tuple(corners_2d[5]), tuple(corners_2d[6]), color, thickness)
        cv2.line(img, tuple(corners_2d[6]), tuple(corners_2d[7]), color, thickness)
        cv2.line(img, tuple(corners_2d[7]), tuple(corners_2d[4]), color, thickness)

        # Vertical edges
        cv2.line(img, tuple(corners_2d[0]), tuple(corners_2d[4]), color, thickness)
        cv2.line(img, tuple(corners_2d[1]), tuple(corners_2d[5]), color, thickness)
        cv2.line(img, tuple(corners_2d[2]), tuple(corners_2d[6]), color, thickness)
        cv2.line(img, tuple(corners_2d[3]), tuple(corners_2d[7]), color, thickness)

        boxes_drawn += 1

    print(
        f"    Boxes drawn: {boxes_drawn}, skipped (behind): {boxes_skipped_behind}, skipped (outside): {boxes_skipped_outside}"
    )
    return img


# def draw_lidar_bbox3d_on_img(bboxes3d, img, lidar2img, img_shape, color=(0, 255, 0), thickness=2):
#     """Draw 3D bounding boxes on image."""

#     if isinstance(lidar2img, torch.Tensor):
#         lidar2img = lidar2img.cpu().numpy()

#     if len(bboxes3d) == 0:
#         return img

#     for bbox in bboxes3d:
#         # Get 8 corners of the 3D box
#         l, w, h = bbox[3:6]
#         x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
#         y_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
#         z_corners = np.array([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])
#         corners = np.vstack([x_corners, y_corners, z_corners])  # 3x8

#         # Rotate
#         yaw = bbox[6]
#         rot_mat = np.array([
#             [np.cos(yaw), -np.sin(yaw), 0],
#             [np.sin(yaw), np.cos(yaw), 0],
#             [0, 0, 1]
#         ])
#         corners = rot_mat @ corners  # 3x8

#         # Translate
#         corners[0, :] += bbox[0]
#         corners[1, :] += bbox[1]
#         corners[2, :] += bbox[2]

#         # corners is now 3x8, transpose to 8x3
#         corners = corners.T  # 8x3

#         # Project to image - add homogeneous coordinate
#         pts_4d = np.concatenate([corners, np.ones((8, 1))], axis=-1)  # 8x4
#         pts_2d = pts_4d @ lidar2img.T  # 8x4

#         # Check if points are in front of camera
#         if (pts_2d[:, 2] <= 0).all():
#             continue  # All points behind camera

#         # Normalize
#         pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e10)
#         pts_2d[:, 0] /= pts_2d[:, 2]
#         pts_2d[:, 1] /= pts_2d[:, 2]

#         corners_2d = pts_2d[:, :2].astype(np.int32)

#         # Check if box is visible in image
#         x_min, x_max = corners_2d[:, 0].min(), corners_2d[:, 0].max()
#         y_min, y_max = corners_2d[:, 1].min(), corners_2d[:, 1].max()

#         if x_max < 0 or x_min > img_shape[1] or y_max < 0 or y_min > img_shape[0]:
#             continue  # Box not visible

#         # Draw the 12 edges of the 3D box
#         # Bottom face (indices 0,1,2,3)
#         cv2.line(img, tuple(corners_2d[0]), tuple(corners_2d[1]), color, thickness)
#         cv2.line(img, tuple(corners_2d[1]), tuple(corners_2d[2]), color, thickness)
#         cv2.line(img, tuple(corners_2d[2]), tuple(corners_2d[3]), color, thickness)
#         cv2.line(img, tuple(corners_2d[3]), tuple(corners_2d[0]), color, thickness)

#         # Top face (indices 4,5,6,7)
#         cv2.line(img, tuple(corners_2d[4]), tuple(corners_2d[5]), color, thickness)
#         cv2.line(img, tuple(corners_2d[5]), tuple(corners_2d[6]), color, thickness)
#         cv2.line(img, tuple(corners_2d[6]), tuple(corners_2d[7]), color, thickness)
#         cv2.line(img, tuple(corners_2d[7]), tuple(corners_2d[4]), color, thickness)

#         # Vertical edges
#         cv2.line(img, tuple(corners_2d[0]), tuple(corners_2d[4]), color, thickness)
#         cv2.line(img, tuple(corners_2d[1]), tuple(corners_2d[5]), color, thickness)
#         cv2.line(img, tuple(corners_2d[2]), tuple(corners_2d[6]), color, thickness)
#         cv2.line(img, tuple(corners_2d[3]), tuple(corners_2d[7]), color, thickness)

#     return img


class MockDet3DDataPreprocessor:
    """Standalone implementation of Det3DDataPreprocessor"""

    def __init__(
        self, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], bgr_to_rgb=False, pad_size_divisor=32
    ):
        self.mean = torch.tensor(mean).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 1, 3, 1, 1)
        self.bgr_to_rgb = bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor

    def __call__(self, data, training=False):
        """Process the input data"""
        if isinstance(data, str):
            data = torch.load(data)

        # Extract images
        if "imgs" in data:
            imgs = data["imgs"]
        elif "inputs" in data and "imgs" in data["inputs"]:
            imgs = data["inputs"]["imgs"]
        else:
            print("Warning: No images found, creating dummy data")
            imgs = torch.zeros(1, 6, 3, 320, 800)

        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs).float()

        if len(imgs.shape) == 4:
            imgs = imgs.unsqueeze(0)

        # Normalize if needed
        if imgs.max() > 10.0:
            imgs = (imgs - self.mean) / self.std

        # Pad to divisible size
        _, _, _, h, w = imgs.shape
        pad_h = (self.pad_size_divisor - h % self.pad_size_divisor) % self.pad_size_divisor
        pad_w = (self.pad_size_divisor - w % self.pad_size_divisor) % self.pad_size_divisor

        if pad_h > 0 or pad_w > 0:
            imgs = torch.nn.functional.pad(imgs, (0, pad_w, 0, pad_h))

        output = {"inputs": {"imgs": imgs}, "data_samples": []}

        batch_size = imgs.shape[0]
        for i in range(batch_size):
            sample = type("DataSample", (), {})()

            if "img_metas" in data:
                if isinstance(data["img_metas"], list):
                    metainfo = data["img_metas"][i] if i < len(data["img_metas"]) else data["img_metas"][0]
                else:
                    metainfo = data["img_metas"]
            else:
                metainfo = {}

            # Ensure all required fields exist
            metainfo.setdefault("img_shape", (h, w))
            metainfo.setdefault("pad_shape", (h + pad_h, w + pad_w))
            metainfo.setdefault("ori_shape", [(h, w)] * 6)
            # if "img_shape" not in metainfo:
            #     metainfo["img_shape"] = [(h, w)] * 6  # List of tuples
            # elif not isinstance(metainfo["img_shape"], list):
            #     # Convert single tuple to list
            #     metainfo["img_shape"] = [metainfo["img_shape"]] * 6

            # # Ensure ori_shape is also a list
            # if "ori_shape" not in metainfo:
            #     metainfo["ori_shape"] = [(h, w)] * 6
            # elif not isinstance(metainfo["ori_shape"], list):
            #     metainfo["ori_shape"] = [metainfo["ori_shape"]] * 6

            # # pad_shape can be a single tuple (it's not indexed by camera)
            # metainfo.setdefault("pad_shape", (h + pad_h, w + pad_w))

            metainfo["box_type_3d"] = LiDARInstance3DBoxes
            metainfo.setdefault("sample_idx", i)

            sample.metainfo = metainfo
            output["data_samples"].append(sample)

        return output


def save_petr_visualizations(ttnn_output, camera_images, img_metas, output_dir="./"):
    """Save BEV and camera view visualizations"""

    output_dict = ttnn_output[0]["pts_bbox"]
    boxes = output_dict["bboxes_3d"].tensor.cpu().numpy()
    scores = output_dict["scores_3d"].to(torch.float32).cpu().numpy()
    labels = output_dict["labels_3d"].to(torch.float32).cpu().numpy()

    # Filter by confidence - use lower threshold since scores are low
    threshold = 0.14  # Lowered from 0.14
    mask = scores > threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    print(f"\nFiltered {len(filtered_boxes)} boxes with score > {threshold}")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Box positions (first 5): {filtered_boxes[:5, :3]}")  # x, y, z positions

    # 1. Bird's Eye View
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define colors for different classes (nuScenes has 10 classes)
    class_colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta", "brown", "pink"]

    for i, box in enumerate(filtered_boxes[:100]):  # Limit to 100 boxes
        x, y, z, l, w, h, yaw = box[:7]
        label = int(filtered_labels[i])
        color = class_colors[label % len(class_colors)]

        # Create rectangle
        rect = patches.Rectangle(
            (x - l / 2, y - w / 2),
            l,
            w,
            angle=np.degrees(yaw),
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            label=f"Class {label}" if i == 0 else "",
        )
        ax.add_patch(rect)

        # Add confidence text
        ax.text(x, y, f"{filtered_scores[i]:.2f}", fontsize=6, ha="center")

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"PETR BEV Detection - {len(filtered_boxes)} objects")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend()

    plt.savefig(f"{output_dir}/petr_bev.jpg", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: petr_bev.jpg")

    # 2. Camera views with projected boxes
    cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    for cam_id, cam_name in enumerate(cam_names):
        img = camera_images[cam_id].copy()

        # Get lidar2img transformation
        if "lidar2img" in img_metas[0] and len(img_metas[0]["lidar2img"]) > cam_id:
            lidar2img = img_metas[0]["lidar2img"][cam_id]
            if isinstance(lidar2img, torch.Tensor):
                lidar2img = lidar2img.cpu().numpy()
            print(f"    lidar2img shape: {lidar2img.shape}")
            print(f"    lidar2img sample:\n{lidar2img[:2, :2]}")
            # Draw boxes
            img = draw_lidar_bbox3d_on_img(
                filtered_boxes[:50], img, lidar2img, img.shape[:2], color=(0, 255, 0), thickness=2  # Limit to 50 boxes
            )
        else:
            print(f"Warning: No lidar2img for {cam_name}, skipping")
            continue

        cv2.imwrite(f"{output_dir}/petr_{cam_name}.jpg", img)
        print(f"Saved: petr_{cam_name}.jpg")

    # 3. 3D scatter plots
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(filtered_boxes[:, 0], filtered_boxes[:, 2], c=filtered_labels, s=20, cmap="tab10")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Z (meters)")
    ax1.set_title("Side View (X-Z)")
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label="Class")

    ax2 = fig.add_subplot(132)
    scatter = ax2.scatter(filtered_boxes[:, 1], filtered_boxes[:, 2], c=filtered_labels, s=20, cmap="tab10")
    ax2.set_xlabel("Y (meters)")
    ax2.set_ylabel("Z (meters)")
    ax2.set_title("Side View (Y-Z)")
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label="Class")

    ax3 = fig.add_subplot(133)
    scatter = ax3.scatter(filtered_boxes[:, 0], filtered_boxes[:, 1], c=filtered_scores, s=20, cmap="viridis")
    ax3.set_xlabel("X (meters)")
    ax3.set_ylabel("Y (meters)")
    ax3.set_title("Top View (X-Y) - Colored by Score")
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label="Score")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/petr_3d_views.jpg", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: petr_3d_views.jpg")

    return filtered_boxes


def load_single_camera_image(
    data_root="models/experimental/functional_petr/resources/nuscenes", camera_name="CAM_FRONT"
):
    """Load just ONE camera for testing"""

    print("\n" + "=" * 60)
    print(f"Loading SINGLE camera: {camera_name}")
    print("=" * 60)

    # Load calibration for this camera only
    calibrations, sample_tokens = load_real_nuscenes_calibration(data_root)

    if camera_name not in calibrations:
        raise ValueError(f"Camera {camera_name} not found in calibrations")

    calib = calibrations[camera_name]

    # Load the exact image
    exact_filename = calib["filename"]
    img_path = Path(data_root) / exact_filename

    if img_path.exists():
        img = cv2.imread(str(img_path))
        print(f"✓ Loaded: {exact_filename}")
        print(f"  Original shape: {img.shape}")
    else:
        # Try samples folder
        alt_path = Path(data_root) / "samples" / camera_name / Path(exact_filename).name
        if alt_path.exists():
            img = cv2.imread(str(alt_path))
            print(f"✓ Loaded from: {alt_path}")
        else:
            raise FileNotFoundError(f"Could not find {exact_filename}")

    # Store original for visualization
    img_original = img.copy()

    # Resize and normalize for model
    img = cv2.resize(img, (800, 320))
    img_for_viz = img.copy()

    img = img.astype(np.float32)
    mean = np.array([103.530, 116.280, 123.675])
    std = np.array([57.375, 57.120, 58.395])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # CHW

    # Add batch and camera dimensions: [1, 1, 3, H, W]
    imgs = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    # Prepare calibration
    lidar2img = calib["lidar2img"].astype(np.float32)
    cam_intrinsic = calib["intrinsic"].astype(np.float32)
    lidar2cam = calib["lidar2cam"].astype(np.float32)

    cam2img = np.eye(4, dtype=np.float32)
    cam2img[:3, :3] = cam_intrinsic

    img_metas = [
        {
            "filename": [exact_filename],
            "ori_shape": [(900, 1600)],
            "img_shape": (320, 800),
            "pad_shape": (320, 800),
            "lidar2img": [lidar2img],
            "cam2img": [cam2img],
            "cam_intrinsic": [cam_intrinsic],
            "lidar2cam": [lidar2cam],
        }
    ]

    print(f"  Model input shape: {imgs.shape}")
    print(f"  lidar2img:\n{lidar2img}")
    print("=" * 60 + "\n")

    return {"imgs": imgs, "img_metas": img_metas}, [img_for_viz]


def save_single_camera_visualization(output, camera_image, img_metas, camera_name="CAM_FRONT", output_dir="./"):
    """Visualize detection on single camera"""

    output_dict = output[0]["pts_bbox"]
    boxes = output_dict["bboxes_3d"].tensor.cpu().numpy()
    scores = output_dict["scores_3d"].to(torch.float32).cpu().numpy()
    labels = output_dict["labels_3d"].to(torch.float32).cpu().numpy()

    threshold = 0.05  # Reasonable threshold
    mask = scores > threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    print(f"\nDetection Results for {camera_name}:")
    print(f"  Total detections: {len(boxes)}")
    print(f"  After threshold {threshold}: {len(filtered_boxes)}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Unique labels: {np.unique(labels)}")

    # Draw boxes on image
    img = camera_image[0].copy()
    lidar2img = img_metas[0]["lidar2img"][0]

    if isinstance(lidar2img, torch.Tensor):
        lidar2img = lidar2img.cpu().numpy()

    print(f"\nDrawing boxes on {camera_name}...")
    img = draw_lidar_bbox3d_on_img(filtered_boxes, img, lidar2img, img.shape[:2], color=(0, 255, 0), thickness=2)

    cv2.imwrite(f"{output_dir}/petr_{camera_name}_detection.jpg", img)
    print(f"✓ Saved: petr_{camera_name}_detection.jpg")

    # BEV view
    fig, ax = plt.subplots(figsize=(12, 12))

    class_colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta", "brown", "pink"]

    for i, box in enumerate(filtered_boxes):
        x, y, z, l, w, h, yaw = box[:7]
        label = int(filtered_labels[i])
        color = class_colors[label % len(class_colors)]

        rect = patches.Rectangle(
            (x - l / 2, y - w / 2),
            l,
            w,
            angle=np.degrees(yaw),
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            label=f"Class {label}" if i == 0 else "",
        )
        ax.add_patch(rect)
        ax.text(x, y, f"{filtered_scores[i]:.2f}", fontsize=8, ha="center")

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"BEV Detection from {camera_name} - {len(filtered_boxes)} objects")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    if len(filtered_boxes) > 0:
        ax.legend()

    plt.savefig(f"{output_dir}/petr_{camera_name}_bev.jpg", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: petr_{camera_name}_bev.jpg")

    return filtered_boxes


# def load_nuscenes_calibration(data_root, sample_token=None):
#     """
#     Load actual nuScenes calibration from dataset structure
#     Requires: data_root/v1.0-mini/sample_data.json, sensor.json, calibrated_sensor.json
#     """
#     from pathlib import Path
#     import json

#     meta_path = Path(data_root) / "v1.0-mini"

#     # Try to load nuScenes metadata
#     if not meta_path.exists():
#         print(f"No nuScenes metadata found at {meta_path}")
#         return None

#     try:
#         # Load necessary JSON files
#         with open(meta_path / "calibrated_sensor.json") as f:
#             calib_sensors = {x['token']: x for x in json.load(f)}

#         with open(meta_path / "ego_pose.json") as f:
#             ego_poses = {x['token']: x for x in json.load(f)}

#         with open(meta_path / "sample_data.json") as f:
#             sample_data = {x['token']: x for x in json.load(f)}

#         with open(meta_path / "sensor.json") as f:
#             sensors = {x['token']: x for x in json.load(f)}

#         # Get the first sample with all 6 cameras
#         cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
#                      "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

#         calibrations = {}

#         for cam_name in cam_names:
#             # Find sample data for this camera
#             cam_samples = [sd for sd in sample_data.values()
#                           if any(s['channel'] == cam_name for s in sensors.values()
#                                 if s['token'] == sd['sensor_token'])]

#             if cam_samples:
#                 sd = cam_samples[0]
#                 calib = calib_sensors[sd['calibrated_sensor_token']]
#                 ego = ego_poses[sd['ego_pose_token']]

#                 # Build transformation matrices
#                 cam_intrinsic = np.array(calib['camera_intrinsic'])

#                 # Camera to ego
#                 rotation = calib['rotation']  # quaternion [w, x, y, z]
#                 translation = calib['translation']
#                 # Convert quaternion to rotation matrix
#                 from scipy.spatial.transform import Rotation
#                 cam2ego_rot = Rotation.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]]).as_matrix()
#                 cam2ego = np.eye(4)
#                 cam2ego[:3, :3] = cam2ego_rot
#                 cam2ego[:3, 3] = translation

#                 # Ego to global (not needed for lidar2cam)
#                 # Lidar is at ego frame, so lidar2cam = inv(cam2ego)
#                 lidar2cam = np.linalg.inv(cam2ego)

#                 calibrations[cam_name] = {
#                     'intrinsic': cam_intrinsic,
#                     'lidar2cam': lidar2cam
#                 }

#         return calibrations

#     except Exception as e:
#         print(f"Error loading nuScenes calibration: {e}")
#         return None

# def load_real_nuscenes_images(data_root="models/experimental/functional_petr/resources/nuscenes"):
#     """Load real images with proper calibration"""

#     cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
#                  "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

#     real_calib = load_nuscenes_calibration(data_root)

#     if real_calib is None:
#         print("WARNING: Using approximate calibration - boxes may not align correctly!")
#         cam_intrinsics, lidar2cam_transforms = load_nuscenes_info(data_root)
#     else:
#         print("✓ Loaded real nuScenes calibration")
#         cam_intrinsics = {cam: real_calib[cam]['intrinsic'] for cam in cam_names}
#         lidar2cam_transforms = {cam: real_calib[cam]['lidar2cam'] for cam in cam_names}

#     # Load calibration
#     cam_intrinsics, lidar2cam_transforms = load_nuscenes_info(data_root)

#     imgs_list = []
#     camera_images = []
#     lidar2img_list = []
#     cam_intrinsic_list = []
#     cam2img_list = []

#     for cam in cam_names:
#         img_path = Path(data_root) / "samples" / cam
#         img_files = list(img_path.glob("*.jpg"))[:1]

#         if img_files:
#             img = cv2.imread(str(img_files[0]))
#             img = cv2.resize(img, (800, 320))
#             camera_images.append(img.copy())  # Store original for visualization
#             print(f"Loaded: {cam}/{img_files[0].name}")
#         else:
#             img = np.zeros((320, 800, 3), dtype=np.uint8)
#             camera_images.append(img.copy())
#             print(f"Using dummy for {cam}")

#         # Normalize
#         img = img.astype(np.float32)
#         mean = np.array([103.530, 116.280, 123.675])
#         std = np.array([57.375, 57.120, 58.395])
#         img = (img - mean) / std
#         img = img.transpose(2, 0, 1)  # HWC -> CHW
#         imgs_list.append(img)

#         # Build lidar2img matrix
#         intrinsic = cam_intrinsics[cam]
#         lidar2cam = lidar2cam_transforms[cam]

#         # Create 4x4 intrinsic matrix
#         intrinsic_4x4 = np.eye(4)
#         intrinsic_4x4[:3, :3] = intrinsic

#         # Combine: lidar2img = intrinsic @ lidar2cam
#         lidar2img = intrinsic_4x4 @ lidar2cam
#         lidar2img_list.append(lidar2img.astype(np.float32))
#         cam_intrinsic_list.append(intrinsic.astype(np.float32))


#         cam2img = np.eye(4, dtype=np.float32)
#         cam2img[:3, :3] = intrinsic
#         cam2img_list.append(cam2img)

#     # Stack and convert to tensor
#     imgs = np.stack(imgs_list).astype(np.float32)
#     imgs = torch.from_numpy(imgs).unsqueeze(0)  # [1, 6, 3, H, W]

#     # Create metadata
#     img_metas = [{
#         "filename": cam_names,
#         "ori_shape": [(320, 800)] * 6,
#         "img_shape": (320, 800),
#         "pad_shape": (320, 800),
#         "cam2img": cam2img_list,
#         "lidar2img": lidar2img_list,
#         "cam_intrinsic": cam_intrinsic_list,
#         "lidar2cam": [lidar2cam_transforms[cam] for cam in cam_names],
#     }]


#     return {"imgs": imgs, "img_metas": img_metas}, camera_images
def load_real_nuscenes_calibration(data_root):
    """
    Load actual nuScenes calibration from v1.0-mini dataset
    """
    import json
    from pathlib import Path

    meta_path = Path(data_root) / "v1.0-mini"

    print(f"Loading nuScenes calibration from {meta_path}")

    # Load all required JSON files
    with open(meta_path / "calibrated_sensor.json") as f:
        calib_sensors = {x["token"]: x for x in json.load(f)}

    with open(meta_path / "sample.json") as f:
        samples = json.load(f)

    with open(meta_path / "sample_data.json") as f:
        sample_data_list = json.load(f)
        sample_data = {x["token"]: x for x in sample_data_list}

    with open(meta_path / "sensor.json") as f:
        sensors_list = json.load(f)
        sensors = {x["token"]: x for x in sensors_list}

    # Debug: Check structures
    print(f"DEBUG: First sample_data keys: {sample_data_list[0].keys()}")
    print(f"DEBUG: First sample_data: {sample_data_list[0]}")
    print(f"DEBUG: First sensor keys: {sensors_list[0].keys()}")
    print(f"DEBUG: First sensor: {sensors_list[0]}")

    # Since sample doesn't have 'data' field, we need to link sample -> sample_data
    # In nuScenes, each sample_data has a 'sample_token' field

    first_sample = samples[0]
    sample_token = first_sample["token"]

    cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    calibrations = {}
    sample_tokens = {}

    # Build mapping: calibrated_sensor_token -> sensor info
    calib_to_sensor = {}
    for sensor in sensors_list:
        # Find calibrated sensors for this sensor
        for calib_token, calib in calib_sensors.items():
            if "sensor_token" in calib and calib["sensor_token"] == sensor["token"]:
                calib_to_sensor[calib_token] = sensor

    # Find sample_data entries that belong to our sample
    for cam_name in cam_names:
        # Find sample_data with matching sample_token and camera channel
        matching_sd = None
        for sd in sample_data_list:
            # Check if this sample_data belongs to our sample
            if "sample_token" in sd and sd["sample_token"] == sample_token:
                # Check if it's the right camera
                calib_token = sd["calibrated_sensor_token"]
                if calib_token in calib_to_sensor:
                    sensor = calib_to_sensor[calib_token]
                    if sensor["channel"] == cam_name:
                        matching_sd = sd
                        break

        if matching_sd is None:
            print(f"  WARNING: Could not find sample_data for {cam_name}")
            continue

        sample_tokens[cam_name] = matching_sd["token"]

        # Process calibration
        calib = calib_sensors[matching_sd["calibrated_sensor_token"]]

        cam_intrinsic = np.array(calib["camera_intrinsic"])
        rotation_quat = calib["rotation"]
        translation = np.array(calib["translation"])

        w, x, y, z = rotation_quat
        rot_matrix = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ]
        )

        cam2ego = np.eye(4)
        cam2ego[:3, :3] = rot_matrix
        cam2ego[:3, 3] = translation

        ego2cam = np.linalg.inv(cam2ego)

        intrinsic_4x4 = np.eye(4)
        intrinsic_4x4[:3, :3] = cam_intrinsic

        lidar2img = intrinsic_4x4 @ ego2cam

        calibrations[cam_name] = {
            "intrinsic": cam_intrinsic,
            "lidar2cam": ego2cam,
            "lidar2img": lidar2img,
            "filename": matching_sd["filename"],
        }

        print(f"  ✓ {cam_name}: {matching_sd['filename']}")

    if not calibrations:
        raise ValueError("Could not load any camera calibrations!")

    print(f"Loaded calibration for {len(calibrations)} cameras")
    return calibrations, sample_tokens


def load_real_nuscenes_images(data_root="models/experimental/functional_petr/resources/nuscenes"):
    """Load images that work + real calibration"""

    cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    # Load REAL calibration
    print("\n" + "=" * 60)
    print("Loading REAL nuScenes calibration...")
    print("=" * 60)
    calibrations, sample_tokens = load_real_nuscenes_calibration(data_root)

    imgs_list = []
    camera_images = []

    # Load images the OLD WAY (from samples folder, not from JSON filenames)
    for cam in cam_names:
        img_path = Path(data_root) / "samples" / cam
        img_files = list(img_path.glob("*.jpg"))[:1]  # First image

        if img_files:
            img = cv2.imread(str(img_files[0]))
            img = cv2.resize(img, (800, 320))
            camera_images.append(img.copy())
            print(f"✓ Loaded {cam}: {img_files[0].name}")
        else:
            img = np.zeros((320, 800, 3), dtype=np.uint8)
            camera_images.append(img.copy())
            print(f"✗ Using dummy for {cam}")

        # Normalize
        img = img.astype(np.float32)
        mean = np.array([103.530, 116.280, 123.675])
        std = np.array([57.375, 57.120, 58.395])
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        imgs_list.append(img)

    imgs = np.stack(imgs_list).astype(np.float32)
    imgs = torch.from_numpy(imgs).unsqueeze(0)

    # Use REAL calibration (NEW)
    lidar2img_list = [calibrations[cam]["lidar2img"].astype(np.float32) for cam in cam_names]
    cam_intrinsic_list = [calibrations[cam]["intrinsic"].astype(np.float32) for cam in cam_names]
    lidar2cam_list = [calibrations[cam]["lidar2cam"].astype(np.float32) for cam in cam_names]

    cam2img_list = []
    for cam in cam_names:
        cam2img = np.eye(4, dtype=np.float32)
        cam2img[:3, :3] = calibrations[cam]["intrinsic"]
        cam2img_list.append(cam2img)

    img_metas = [
        {
            "filename": cam_names,
            "ori_shape": [(320, 800)],
            "img_shape": (320, 800),
            "pad_shape": (320, 800),
            "lidar2img": lidar2img_list,
            "cam2img": cam2img_list,
            "cam_intrinsic": cam_intrinsic_list,
            "lidar2cam": lidar2cam_list,
        }
    ]

    print("=" * 60)
    print("✓ Using original images + REAL calibration!")
    print("=" * 60 + "\n")

    return {"imgs": imgs, "img_metas": img_metas}, camera_images


# [Keep all the preprocessing functions from the original code]
# move_to_device, stem_parameters_preprocess, create_custom_preprocessor_*
# (Copy these from the original file - they're unchanged)


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

                logger.info(
                    f"[PREPROCESS] {prefix}: weight shape={conv_weight.shape}, mean={conv_weight.mean():.6f}, std={conv_weight.std():.6f}"
                )

                # Convert to ttnn format (same as other Conv layers)
                parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[prefix]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

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
                # if "OSA2_1" in prefix:
                #     parameters[prefix]["weight"] = conv_weight
                #     parameters[prefix]["bias"] = conv_bias
                # else:
                parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[prefix]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            first_layer_name, _ = list(model.concat.named_children())[0]
            base_name = first_layer_name.split("/")[0]
            parameters[base_name] = {}
            # if "OSA2_1" in base_name:
            #     parameters[base_name]["weight"] = model.concat[0].weight
            #     parameters[base_name]["bias"] = model.concat[0].bias
            # else:
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
        if isinstance(model, _OSA_stage):
            if isinstance(model, _OSA_module):
                if hasattr(model, "conv_reduction"):
                    first_layer_name, _ = list(model.conv_reduction.named_children())[0]
                    base_name = first_layer_name.split("/")[0]
                    parameters[base_name] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                        model.conv_reduction[0], model.conv_reduction[1]
                    )
                    parameters[base_name]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters[base_name]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )

                for i, layers in enumerate(model.layers):
                    first_layer_name = list(layers.named_children())[0][0]
                    prefix = first_layer_name.split("/")[0]
                    parameters[prefix] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(layers[0], layers[1])
                    parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters[prefix]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )

                first_layer_name, _ = list(model.concat.named_children())[0]
                base_name = first_layer_name.split("/")[0]
                parameters[base_name] = {}
                parameters[base_name]["weight"] = model.concat[0].weight
                parameters[base_name]["bias"] = model.concat[0].bias

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
                            child1.weight, dtype=ttnn.bfloat16
                        )
                        parameters["cls_branches"][index][index1]["bias"] = preprocess_linear_bias(
                            child1.bias, dtype=ttnn.bfloat16
                        )
                    elif isinstance(child1, nn.LayerNorm):
                        parameters["cls_branches"][index][index1]["weight"] = preprocess_layernorm_parameter(
                            child1.weight, dtype=ttnn.bfloat16
                        )
                        parameters["cls_branches"][index][index1]["bias"] = preprocess_layernorm_parameter(
                            child1.bias, dtype=ttnn.bfloat16
                        )

            parameters["reg_branches"] = {}
            for index, child in enumerate(model.reg_branches):
                parameters["reg_branches"][index] = {}
                for index1, child1 in enumerate(child):
                    parameters["reg_branches"][index][index1] = {}
                    if isinstance(child1, Linear):
                        parameters["reg_branches"][index][index1]["weight"] = preprocess_linear_weight(
                            child1.weight, dtype=ttnn.bfloat16
                        )
                        parameters["reg_branches"][index][index1]["bias"] = preprocess_linear_bias(
                            child1.bias, dtype=ttnn.bfloat16
                        )

            parameters["adapt_pos3d"] = {}
            for index, child in enumerate(model.adapt_pos3d):
                parameters["adapt_pos3d"][index] = {}
                if isinstance(child, Conv2d):
                    parameters["adapt_pos3d"][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.bfloat16)
                    parameters["adapt_pos3d"][index]["bias"] = ttnn.from_torch(
                        torch.reshape(child.bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                    )

            parameters["position_encoder"] = {}
            for index, child in enumerate(model.position_encoder):
                parameters["position_encoder"][index] = {}
                if isinstance(child, Conv2d):
                    parameters["position_encoder"][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.bfloat16)
                    parameters["position_encoder"][index]["bias"] = ttnn.from_torch(
                        torch.reshape(child.bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                    )

            parameters["query_embedding"] = {}
            for index, child in enumerate(model.query_embedding):
                parameters["query_embedding"][index] = {}
                if isinstance(child, Linear):
                    parameters["query_embedding"][index]["weight"] = preprocess_linear_weight(
                        child.weight, dtype=ttnn.bfloat16
                    )
                    parameters["query_embedding"][index]["bias"] = preprocess_linear_bias(
                        child.bias, dtype=ttnn.bfloat16
                    )
            parameters["reference_points"] = {}
            parameters["reference_points"]["weight"] = ttnn.from_torch(model.reference_points.weight, device=device)

        return parameters

    return custom_preprocessor


# ... [Include all preprocessing functions here] ...


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_demo(device, reset_seeds):
    """Main PETR demo function"""
    print("=" * 60)
    print("PETR TT Demo - Fixed Version")
    print("=" * 60)

    # Load data
    print("\n1. Loading nuScenes images with calibration...")
    input_data, camera_images = load_real_nuscenes_images("models/experimental/functional_petr/resources/nuscenes")
    # print("\n1. Loading single camera image...")
    # input_data, camera_images = load_single_camera_image(
    #     "models/experimental/functional_petr/resources/nuscenes", camera_name="CAM_FRONT"
    # )
    #####################################################################################
    print(f"   Images shape: {input_data['imgs'].shape}")
    print("\n" + "=" * 60)
    print("DEBUG: Input Comparison")
    print("=" * 60)

    # Load golden test input for comparison
    golden_inputs = torch.load(
        "models/experimental/functional_petr/resources/golden_input_inputs_sample1.pt", weights_only=False
    )

    print(f"Real input shape: {input_data['imgs'].shape}")
    print(f"Golden input shape: {golden_inputs['imgs'].shape}")
    print(f"Real input range: [{input_data['imgs'].min():.4f}, {input_data['imgs'].max():.4f}]")
    print(f"Golden input range: [{golden_inputs['imgs'].min():.4f}, {golden_inputs['imgs'].max():.4f}]")
    print(f"Real input mean/std: {input_data['imgs'].mean():.4f} / {input_data['imgs'].std():.4f}")
    print(f"Golden input mean/std: {golden_inputs['imgs'].mean():.4f} / {golden_inputs['imgs'].std():.4f}")
    print("=" * 60 + "\n")

    # After showing input comparison, add this:
    print("\n" + "=" * 60)
    print("VERIFICATION: PyTorch Baseline Comparison")
    print("=" * 60)

    # Load golden inputs
    golden_inputs = torch.load(
        "models/experimental/functional_petr/resources/golden_input_inputs_sample1.pt", weights_only=False
    )
    golden_metas = torch.load(
        "models/experimental/functional_petr/resources/modified_input_batch_img_metas_sample1.pt", weights_only=False
    )

    ##############################################################

    # Initialize preprocessor
    print("\n2. Initializing data preprocessor...")
    data_preprocessor = MockDet3DDataPreprocessor()

    print("3. Processing input data...")
    output_after_preprocess = data_preprocessor(input_data, False)
    batch_img_metas = [ds.metainfo for ds in output_after_preprocess["data_samples"]]

    # Load model
    print("\n4. Loading PETR model...")
    weights_path = "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth"
    weights_state_dict = torch.load(weights_path, weights_only=False)["state_dict"]

    torch_model = PETR(use_grid_mask=True)
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    # Run PyTorch inference
    print("\n5. Running PyTorch inference...")
    with torch.no_grad():
        torch_output = torch_model.predict(
            output_after_preprocess["inputs"], batch_img_metas, skip_post_processing=True
        )

    ###############
    # Run PyTorch on golden input
    print("Running PyTorch on GOLDEN input...")
    with torch.no_grad():
        torch_output_golden = torch_model.predict(golden_inputs, golden_metas, skip_post_processing=True)

    # Run PyTorch on real input
    print("Running PyTorch on REAL input...")
    # with torch.no_grad():
    #     torch_output_real = torch_model.predict(
    #         output_after_preprocess["inputs"],
    #         batch_img_metas,
    #         skip_post_processing=True
    #     )

    # Compare PyTorch outputs (baseline)
    from tests.ttnn.utils_for_testing import check_with_pcc

    passed_cls, pcc_cls = check_with_pcc(
        torch_output_golden["all_cls_scores"], torch_output["all_cls_scores"], pcc=0.90
    )
    print(f"PyTorch: Golden vs Real cls_scores PCC: {float(pcc_cls):.6f}")

    passed_bbox, pcc_bbox = check_with_pcc(
        torch_output_golden["all_bbox_preds"], torch_output["all_bbox_preds"], pcc=0.90
    )
    print(f"PyTorch: Golden vs Real bbox_preds PCC: {float(pcc_bbox):.6f}")

    print("\nThis shows how much PyTorch outputs differ between scenes.")
    print("TTNN should match PyTorch's behavior on the SAME input.")
    print("=" * 60 + "\n")

    # Continue with normal TTNN inference on real input...
    # torch_output = torch_output_real
    ############

    # print(f"   PyTorch detections: {len(torch_output[0]['pts_bbox']['bboxes_3d'])}")
    # print(f"   PyTorch detections: {len(torch_output[0]['pts_bbox']['bboxes_3d'].tensor)}")

    # # [Rest of the TTNN conversion code - same as original]

    # if torch_output and len(torch_output) > 0:
    #     print(f"   PyTorch output: {len(torch_output)} predictions")
    #     if "boxes_3d" in torch_output[0]:
    #         print(f"   Number of detected boxes: {len(torch_output[0]['boxes_3d'])}")

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
    ttnn_output = ttnn_model.predict(ttnn_inputs, ttnn_batch_img_metas, skip_post_processing=True)

    print("torch_output::", torch_output)
    print("ttnn_output::", ttnn_output)

    from tests.ttnn.utils_for_testing import check_with_pcc, assert_with_pcc

    # Compare ttnn_output with torch_output using PCC check
    print("\n12. Verifying TTNN output with PCC check against torch model...")
    # torch_output = torch_model.predict(output_after_preprocess["inputs"], batch_img_metas, skip_post_processing=True)

    ttnn_output_for_pcc = {
        "all_cls_scores": ttnn.to_torch(ttnn_output["all_cls_scores"])
        if isinstance(ttnn_output["all_cls_scores"], ttnn.Tensor)
        else ttnn_output["all_cls_scores"],
        "all_bbox_preds": ttnn.to_torch(ttnn_output["all_bbox_preds"])
        if isinstance(ttnn_output["all_bbox_preds"], ttnn.Tensor)
        else ttnn_output["all_bbox_preds"],
    }

    # def check_with_pcc(a, b, pcc=0.97):
    #     from scipy.stats import pearsonr
    #     a = a.detach().cpu().float().flatten()
    #     b = b.detach().cpu().float().flatten()
    #     corr, _ = pearsonr(a.numpy(), b.numpy())
    #     return corr >= pcc, corr

    passed_cls, pcc_cls = check_with_pcc(
        torch_output["all_cls_scores"], ttnn_output_for_pcc["all_cls_scores"], pcc=0.97
    )
    print(f"PETR all_cls_scores PCC: {float(pcc_cls):.6f}")

    passed_bbox, pcc_bbox = check_with_pcc(
        torch_output["all_bbox_preds"], ttnn_output_for_pcc["all_bbox_preds"], pcc=0.97
    )
    print(f"PETR all_bbox_preds PCC: {float(pcc_bbox):.6f}")

    if not (passed_cls and passed_bbox):
        print("Warning: TTNN output did not pass PCC check with thresholds (0.97).")
    else:
        print("TTNN output passed PCC check!")

    assert_with_pcc(torch_output["all_cls_scores"], ttnn_output_for_pcc["all_cls_scores"], pcc=0.97)
    assert_with_pcc(torch_output["all_bbox_preds"], ttnn_output_for_pcc["all_bbox_preds"], pcc=0.97)

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

    # ...

    print("\n6. Saving visualizations...")
    # filtered_boxes = save_petr_visualizations(torch_output, camera_images, batch_img_metas)
    filtered_boxes = save_single_camera_visualization(
        torch_output, camera_images, batch_img_metas, camera_name="CAM_FRONT"
    )

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

    return torch_output, camera_images, batch_img_metas


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        output, images, metas = test_demo(device, None)
        save_petr_visualizations(output, images, metas)
        # save_single_camera_visualization(output, images, metas, camera_name="CAM_FRONT")
    finally:
        ttnn.close_device(device)
