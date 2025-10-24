# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
CPU-based Postprocessing for YOLO11 Pose Estimation

This module provides CPU-based keypoint decoding for TTNN pose model outputs.
The TTNN model outputs raw keypoint values, which are then decoded on CPU
for efficiency.
"""

import torch


def decode_pose_keypoints_cpu(output, anchors, strides):
    """
    Decode keypoints on CPU after TTNN inference

    The TTNN model outputs:
    - Channels 0-3: Decoded bounding boxes (already processed)
    - Channel 4: Sigmoid confidence (already processed)
    - Channels 5-55: RAW keypoint values (need decoding)

    This function decodes the raw keypoint values to pixel coordinates.

    Args:
        output: Model output tensor [batch, 56, num_anchors]
                - Already has decoded bbox and conf
                - Has RAW keypoints in channels 5-55
        anchors: Anchor points [2, num_anchors] or [num_anchors, 2]
        strides: Stride values [1, num_anchors] or [num_anchors, 1]

    Returns:
        Decoded output tensor [batch, 56, num_anchors]
        - Channels 0-4: Unchanged (bbox + conf)
        - Channels 5-55: Decoded keypoints in pixel coordinates
    """

    # Extract components
    bbox = output[:, 0:4, :]  # Already decoded
    conf = output[:, 4:5, :]  # Already sigmoid
    keypoints = output[:, 5:56, :]  # RAW values - need decoding

    batch_size = output.shape[0]
    num_anchors = output.shape[2]

    # Reshape keypoints to [batch, 17, 3, num_anchors]
    keypoints = keypoints.reshape(batch_size, 17, 3, num_anchors)

    # Extract x, y, visibility
    kpt_x = keypoints[:, :, 0, :]  # [batch, 17, num_anchors]
    kpt_y = keypoints[:, :, 1, :]  # [batch, 17, num_anchors]
    kpt_v = keypoints[:, :, 2, :]  # [batch, 17, num_anchors]

    # Prepare anchors and strides for broadcasting
    # Handle different anchor formats
    if anchors.shape[0] == 2:
        # Format: [2, num_anchors]
        anchor_x = anchors[0, :].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
        anchor_y = anchors[1, :].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
    else:
        # Format: [num_anchors, 2]
        anchor_x = anchors[:, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
        anchor_y = anchors[:, 1].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]

    # Handle different stride formats
    if strides.shape[0] == 1:
        # Format: [1, num_anchors]
        stride_val = strides[0, :].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
    else:
        # Format: [num_anchors, 1]
        stride_val = strides[:, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]

    # Apply keypoint decoding formula (NO sigmoid on x,y!)
    # Formula: (x * 2 - 0.5 + anchor) * stride
    kpt_x_decoded = (kpt_x * 2.0 - 0.5 + anchor_x) * stride_val
    kpt_y_decoded = (kpt_y * 2.0 - 0.5 + anchor_y) * stride_val

    # Apply sigmoid only to visibility
    kpt_v_decoded = torch.sigmoid(kpt_v)

    # Stack back together [batch, 17, 3, num_anchors]
    keypoints_decoded = torch.stack([kpt_x_decoded, kpt_y_decoded, kpt_v_decoded], dim=2)

    # Reshape back to [batch, 51, num_anchors]
    keypoints_decoded = keypoints_decoded.reshape(batch_size, 51, num_anchors)

    # Concatenate with bbox and conf
    output_decoded = torch.cat([bbox, conf, keypoints_decoded], dim=1)

    return output_decoded


def decode_pose_output_from_ttnn(ttnn_output, anchors, strides, device=None):
    """
    Complete postprocessing pipeline for TTNN pose output

    1. Convert TTNN tensor to PyTorch
    2. Decode keypoints on CPU
    3. Return decoded output ready for NMS and visualization

    Args:
        ttnn_output: TTNN tensor output from pose model
        anchors: TTNN anchor tensor
        strides: TTNN stride tensor
        device: TTNN device (optional, for conversion)

    Returns:
        PyTorch tensor [batch, 56, num_anchors] with decoded keypoints
    """
    import ttnn

    # Convert TTNN tensors to PyTorch
    output_torch = ttnn.to_torch(ttnn_output)
    anchors_torch = ttnn.to_torch(anchors) if not isinstance(anchors, torch.Tensor) else anchors
    strides_torch = ttnn.to_torch(strides) if not isinstance(strides, torch.Tensor) else strides

    # Decode keypoints on CPU
    decoded_output = decode_pose_keypoints_cpu(output_torch, anchors_torch, strides_torch)

    return decoded_output
