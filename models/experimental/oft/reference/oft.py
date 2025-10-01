# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..reference import utils

EPSILON = 1e-6


class OFT(nn.Module):
    def __init__(self, channels, cell_size, grid_height, scale, dtype):
        super().__init__()

        y_corners = torch.arange(0, grid_height, cell_size, dtype=dtype) - grid_height / 2.0
        y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
        self.register_buffer("y_corners", y_corners)

        # Create Linear layer with the specified dtype
        self.conv3d = nn.Linear((len(y_corners) - 1) * channels, channels, dtype=dtype)
        self.scale = scale
        self.dtype = dtype

        # Convert all parameters to the specified dtype using PyTorch's to() method
        self.to(dtype)

    def forward(self, features, calib, grid):
        # Ensure inputs are of the right dtype
        if features.dtype != self.dtype:
            features = features.to(self.dtype)
        if calib.dtype != self.dtype:
            calib = calib.to(self.dtype)
        if grid.dtype != self.dtype:
            grid = grid.to(self.dtype)

        # Expand the grid in the y dimension
        corners = grid.unsqueeze(1) + self.y_corners.view(-1, 1, 1, 3)

        # Project grid corners to image plane and normalize to [-1, 1]
        img_corners = utils.perspective(calib.view(-1, 1, 1, 1, 3, 4), corners, self.dtype)

        # Normalize to [-1, 1]
        img_height, img_width = features.size()[2:]
        img_size = torch.tensor([img_width, img_height], dtype=self.dtype, device=features.device) / self.scale
        norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)

        # Get top-left and bottom-right coordinates of voxel bounding boxes
        bbox_corners = torch.cat(
            [
                torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
                torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
            ],
            dim=-1,
        )
        batch, _, depth, width, _ = bbox_corners.size()
        bbox_corners = bbox_corners.flatten(2, 3)

        # Compute the area of each bounding box
        area = (
            (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * img_height * img_width * 0.25
            + torch.tensor(EPSILON, dtype=self.dtype, device=features.device)
        ).unsqueeze(1)
        visible = area > torch.tensor(EPSILON, dtype=self.dtype, device=features.device)

        # Sample integral image at bounding box locations
        integral_img = integral_image(features)
        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]], align_corners=False)
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]], align_corners=False)
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]], align_corners=False)
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]], align_corners=False)

        # Compute voxel features (ignore features which are not visible))
        vox_feats = top_left + btm_right - top_right - btm_left
        # vox_feats = vox_feats * features.shape[-1] * features.shape[-2] * 8
        vox_feats = vox_feats / area
        # Make sure visible mask is in the right dtype
        vox_feats = vox_feats * visible.to(dtype=self.dtype)
        # vox_feats = vox_feats.view(batch, -1, depth, width)
        vox_feats = vox_feats.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2)
        # Ensure vox_feats is in the right dtype before passing to the linear layer
        if vox_feats.dtype != self.dtype:
            vox_feats = vox_feats.to(self.dtype)

        # Flatten to orthographic feature map
        ortho_feats = self.conv3d(vox_feats).view(batch, depth, width, -1)
        ortho_feats = F.relu(ortho_feats.permute(0, 3, 1, 2), inplace=True)

        # Final check to ensure output is in the right dtype
        if ortho_feats.dtype != self.dtype:
            ortho_feats = ortho_feats.to(self.dtype)
        # ortho_feats = F.relu(self.conv3d(vox_feats))

        # Block gradients to pixels which are not visible in the image

        return (
            ortho_feats,
            integral_img,
            bbox_corners[..., [0, 1]],
            bbox_corners[..., [2, 3]],
            bbox_corners[..., [2, 1]],
            bbox_corners[..., [0, 3]],
        )


def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)
