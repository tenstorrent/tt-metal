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
    # Ensure we have correct 2D format without extra dimensions

    # Anchors should be [2, num_anchors]
    if anchors.dim() > 2:
        # Squeeze extra dimensions
        while anchors.dim() > 2:
            anchors = anchors.squeeze(0)

    if anchors.shape[0] == 2:
        # Format: [2, num_anchors] - correct!
        anchor_x = anchors[0, :].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
        anchor_y = anchors[1, :].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
    elif anchors.shape[1] == 2:
        # Format: [num_anchors, 2] - transpose needed
        anchors = anchors.transpose(0, 1)
        anchor_x = anchors[0, :].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
        anchor_y = anchors[1, :].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
    else:
        raise ValueError(f"Unexpected anchor shape: {anchors.shape}")

    # Strides should be [1, num_anchors]
    if strides.dim() > 2:
        while strides.dim() > 2:
            strides = strides.squeeze(0)

    if strides.dim() == 1:
        # Format: [num_anchors] - add batch dim
        stride_val = strides.unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
    elif strides.shape[0] == 1:
        # Format: [1, num_anchors] - correct!
        stride_val = strides.unsqueeze(0)  # [1, 1, num_anchors]
    elif strides.shape[1] == 1:
        # Format: [num_anchors, 1] - transpose and reshape
        stride_val = strides.transpose(0, 1).unsqueeze(0)  # [1, 1, num_anchors]
    else:
        raise ValueError(f"Unexpected stride shape: {strides.shape}")

    # Debug: Check intermediate values
    print(f"\n[DEBUG] Keypoint decoding:")
    print(f"  kpt_x range: [{kpt_x.min():.2f}, {kpt_x.max():.2f}]")
    print(f"  kpt_x shape: {kpt_x.shape}")
    print(f"  anchor_x shape: {anchor_x.shape}, range: [{anchor_x.min():.2f}, {anchor_x.max():.2f}]")
    print(f"  stride_val shape: {stride_val.shape}, range: [{stride_val.min():.2f}, {stride_val.max():.2f}]")
    print(f"  Sample anchor_x[0,0,0:10]: {anchor_x[0,0,0:10]}")
    print(f"  Sample stride_val[0,0,0:10]: {stride_val[0,0,0:10]}")

    # Apply keypoint decoding formula (NO sigmoid on x,y!)
    # Formula: (x * 2 - 0.5 + anchor) * stride
    term1 = kpt_x * 2.0
    term2 = term1 - 0.5
    term3 = term2 + anchor_x
    kpt_x_decoded = term3 * stride_val

    print(f"  After x*2: [{term1.min():.2f}, {term1.max():.2f}]")
    print(f"  After -0.5: [{term2.min():.2f}, {term2.max():.2f}]")
    print(f"  After +anchor: [{term3.min():.2f}, {term3.max():.2f}]")
    print(f"  After *stride: [{kpt_x_decoded.min():.2f}, {kpt_x_decoded.max():.2f}]")

    kpt_y_decoded = (kpt_y * 2.0 - 0.5 + anchor_y) * stride_val

    # Apply sigmoid only to visibility
    kpt_v_decoded = torch.sigmoid(kpt_v)

    # Squeeze any extra dimensions from broadcasting
    kpt_x_decoded = kpt_x_decoded.squeeze(1) if kpt_x_decoded.dim() > 3 else kpt_x_decoded
    kpt_y_decoded = kpt_y_decoded.squeeze(1) if kpt_y_decoded.dim() > 3 else kpt_y_decoded

    # Ensure all have shape [batch, 17, num_anchors]
    assert (
        kpt_x_decoded.shape == kpt_y_decoded.shape == kpt_v_decoded.shape
    ), f"Shape mismatch: x={kpt_x_decoded.shape}, y={kpt_y_decoded.shape}, v={kpt_v_decoded.shape}"

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

    # Remove extra dimensions from TTNN tensors if present
    # TTNN tensors might have shape [1, 2, 8400] or [1, 1, 8400]
    while anchors_torch.dim() > 2:
        anchors_torch = anchors_torch.squeeze(0)
    while strides_torch.dim() > 2:
        strides_torch = strides_torch.squeeze(0)

    # Ensure anchors is [2, N] format
    if anchors_torch.dim() == 2:
        if anchors_torch.shape[0] != 2 and anchors_torch.shape[1] == 2:
            anchors_torch = anchors_torch.transpose(0, 1)
        elif anchors_torch.shape[0] == 2 and anchors_torch.shape[1] != 2:
            pass  # Already [2, N]
        else:
            # Shape is [N, 2], transpose to [2, N]
            if anchors_torch.shape[1] == 2:
                anchors_torch = anchors_torch.transpose(0, 1)

    # Ensure strides is [1, N] format
    if strides_torch.dim() == 1:
        strides_torch = strides_torch.unsqueeze(0)
    elif strides_torch.dim() == 2:
        if strides_torch.shape[0] != 1:
            strides_torch = strides_torch.transpose(0, 1)

    # Decode keypoints on CPU
    decoded_output = decode_pose_keypoints_cpu(output_torch, anchors_torch, strides_torch)

    return decoded_output
