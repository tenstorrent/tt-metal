# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

# from .. import utils
EPSILON = 1e-6
EPSILON2 = 1


def perspective(matrix, vector):
    """
    Applies perspective projection to a vector using projection matrix
    """
    vector = vector.unsqueeze(-1)
    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]


class OFT(nn.Module):
    def __init__(self, channels, cell_size, grid_height, scale=1):
        super().__init__()

        y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
        y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
        self.register_buffer("y_corners", y_corners)

        self.conv3d = nn.Linear((len(y_corners) - 1) * channels, channels)

        self.scale = scale

    def forward(self, features, calib, grid):
        # print(f"TORCH: features shape: {features.shape}, dtype: {features.dtype}")
        corners = grid.unsqueeze(1) + self.y_corners.view(-1, 1, 1, 3)
        # print(f"corners shape: {corners.shape}, dtype: {corners.dtype}")

        img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners)
        img_height, img_width = features.size()[2:]
        img_size = corners.new([img_width, img_height]) / self.scale
        print(f"img_size {img_size}, shape: {img_size.shape}, dtype: {img_size.dtype}")
        norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)

        bbox_corners = torch.cat(
            [
                torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
                torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
            ],
            dim=-1,
        )

        batch, _, depth, width, _ = bbox_corners.size()
        # print("TORCH: bbox corners shape: ", bbox_corners.shape)
        bbox_corners = bbox_corners.flatten(2, 3)

        area = (
            (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * img_height * img_width * 0.25 + EPSILON
        ).unsqueeze(1)

        visible = area > EPSILON

        integral_img = integral_image(features)  # .to(torch.bfloat16))

        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]])  # .to(torch.bfloat16))
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]])  # .to(torch.bfloat16))
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]])  # .to(torch.bfloat16))
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]])  # .to(torch.bfloat16))
        # return btm_left
        vox_feats = top_left + btm_right - top_right - btm_left
        # return vox_feats
        vox_feats = vox_feats / area

        vox_feats = vox_feats * visible.float()
        print(f"TORCH: vox_feats after visibility mask shape: {vox_feats.shape}, dtype: {vox_feats.dtype}")

        vox_feats = vox_feats.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2)
        print(f"TORCH: vox_feats shape: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        # return vox_feats
        # Flatten to orthographic feature map
        ortho_feats = self.conv3d(vox_feats).view(batch, depth, width, -1)
        ortho_feats = F.relu(ortho_feats.permute(0, 3, 1, 2), inplace=True)

        return ortho_feats


def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)
