#!/usr/bin/env python3
"""Debug: Compare anchors from PyTorch vs TTNN"""

import torch

import ttnn
from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
from models.demos.yolov11.tt.model_preprocessing_pose import create_yolov11_pose_model_parameters
from models.demos.yolov11.tt.ttnn_yolov11_pose_model import TtnnYoloV11Pose

# Load PyTorch model
torch_model = YoloV11Pose()
torch_model.load_state_dict(torch.load("models/demos/yolov11/reference/yolov11_pose_pretrained_correct.pth"))
torch_model.eval()

# Run PyTorch to see what anchors it generates
test_input = torch.randn(1, 3, 640, 640)

# Intercept the make_anchors call in PyTorch forward pass
# Actually, let's just check what TTNN's anchors look like

device = ttnn.open_device(0)
parameters = create_yolov11_pose_model_parameters(torch_model, test_input, device)
ttnn_model = TtnnYoloV11Pose(device, parameters)

# Get TTNN anchors
ttnn_anchors = ttnn.to_torch(ttnn_model.pose_head.anchors)
ttnn_strides = ttnn.to_torch(ttnn_model.pose_head.strides)

print("TTNN Anchors:")
print(f"  Shape: {ttnn_anchors.shape}")
print(f"  Range: [{ttnn_anchors.min():.2f}, {ttnn_anchors.max():.2f}]")
print(f"  First 10: {ttnn_anchors.squeeze()[0:2, 0:10]}")

print("\nTTNN Strides:")
print(f"  Shape: {ttnn_strides.shape}")
print(f"  Range: [{ttnn_strides.min():.2f}, {ttnn_strides.max():.2f}]")
print(f"  First 10: {ttnn_strides.squeeze()[0:10]}")
print(f"  Count of 8s: {(ttnn_strides.squeeze() == 8).sum()}")
print(f"  Count of 16s: {(ttnn_strides.squeeze() == 16).sum()}")
print(f"  Count of 32s: {(ttnn_strides.squeeze() == 32).sum()}")

ttnn.close_device(device)
