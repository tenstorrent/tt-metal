# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
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
    # print(f"TORCH: matrix shape: {matrix.shape}, dtype: {matrix.dtype}")
    # print(f"TORCH: vector shape: {vector.shape}, dtype: {vector.dtype}")
    vector = vector.unsqueeze(-1)
    # print(f"TORCH: after unsqueeze vector shape: {vector.shape}, dtype: {vector.dtype}")
    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    # print(f"TORCH: homogenous shape: {homogenous.shape}, dtype: {homogenous.dtype}")
    homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]


class OFT(nn.Module):
    def __init__(self, channels, cell_size, grid_height, scale=1):
        super().__init__()

        y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
        # print(f"y_corners shape: {y_corners.shape}, dtype: {y_corners.dtype}")
        y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
        # print(f"padded y_corners shape: {y_corners.shape}, dtype: {y_corners.dtype}")
        self.register_buffer("y_corners", y_corners)

        # self.conv3d = nn.Conv2d((len(y_corners)-1) * channels, channels,1)

        self.conv3d = nn.Linear((len(y_corners) - 1) * channels, channels)

        # print(f"{(len(y_corners) - 1) * channels}, channels: {channels}")
        self.scale = scale

    def forward(self, features, calib, grid):
        # print(f"TORCH: features shape: {features.shape}, dtype: {features.dtype}")
        corners = grid.unsqueeze(1) + self.y_corners.view(-1, 1, 1, 3)
        # print(f"corners shape: {corners.shape}, dtype: {corners.dtype}")

        img_corners = perspective(
            calib.view(-1, 1, 1, 1, 3, 4), corners
        )  # utils.perspective(calib.view(-1, 1, 1, 1, 3, 4), corners)
        # print(f"TORCH: img_corners shape: {img_corners.shape}, dtype: {img_corners.dtype}")
        # Normalize to [-1, 1]
        img_height, img_width = features.size()[2:]
        # print(f"TORCH: img_height: {img_height}, img_width: {img_width}")
        img_size = corners.new([img_width, img_height]) / self.scale
        print(f"img_size {img_size}, shape: {img_size.shape}, dtype: {img_size.dtype}")
        norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)
        # print(f"TORCH:: norm_corners shape: {norm_corners.shape}, dtype: {norm_corners.dtype}")
        # print(f"img_size: {img_size}, shape: {img_size.shape}, dtype: {img_size.dtype}")
        # print(f"scale: {self.scale}")
        # Get top-left and bottom-right coordinates of voxel bounding boxes
        bbox_corners = torch.cat(
            [
                torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
                torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
            ],
            dim=-1,
        )
        # print(f"TORCH: bbox_corners shape: {bbox_corners.shape}, dtype: {bbox_corners.dtype}")

        batch, _, depth, width, _ = bbox_corners.size()
        # print("TORCH: bbox corners shape: ", bbox_corners.shape)
        bbox_corners = bbox_corners.flatten(2, 3)
        # print(f"TORCH: box corners min: {torch.min(bbox_corners)}, max: {torch.max(bbox_corners)}")
        # print(f"TORCH: bbox_corners after flatten shape: {bbox_corners.shape}, dtype: {bbox_corners.dtype}")
        # print(f"TORCH: bbox_corners[..., [0, 1]]: {bbox_corners[..., [0, 1]].shape}")
        # Compute the area of each bounding box
        area = (
            (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * img_height * img_width * 0.25 + EPSILON
        ).unsqueeze(1)
        print(f"TORCH: min area: {torch.min(area)}, max area: {torch.max(area)}")
        print(f"TORCH: area shape: {area.shape}, dtype: {area.dtype}")
        visible = area > EPSILON
        # print(f"TORCH: visible shape: {visible.shape}, dtype: {visible.dtype}")
        # Sample integral image at bounding box locations
        # print(f"TORCH: features shape: {features.shape}, dtype: {features.dtype}")
        integral_img = integral_image(features)  # .to(torch.bfloat16))
        print(f"TORCH::: integral_img shape: {integral_img.shape}, dtype: {integral_img.dtype}")
        # print(f"TORCH: integral_img shape: {integral_img.shape}, dtype: {integral_img.dtype}")
        # print(f"TORCH: bbox_corners[..., [0, 1]] shape: {bbox_corners[..., [0, 1]].shape}, dtype: {bbox_corners[..., [0, 1]].dtype}")
        # print(f"TORCH: {bbox_corners[..., [0, 1]]}")
        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]])  # .to(torch.bfloat16))
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]])  # .to(torch.bfloat16))
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]])  # .to(torch.bfloat16))
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]])  # .to(torch.bfloat16))
        # return btm_left
        vox_feats = top_left + btm_right - top_right - btm_left
        # return vox_feats
        vox_feats = vox_feats / area
        # vox_feats = (top_left + btm_right - btm_left )/area
        # return vox_feats
        # print(f"TORCH: visible shape: {visible.shape}, dtype: {visible.dtype}")
        # print(f"TORCH: area shape: {area.shape}, dtype: {area.dtype}")
        # print(f"TORCH: vox_feats before visibility mask shape: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        # return vox_feats

        # print(f"TORCH: vox_feats shape: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        vox_feats = vox_feats * visible.float()
        print(f"TORCH: vox_feats after visibility mask shape: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        # vox_feats = vox_feats.view(batch, -1, depth, width)
        # return vox_feats
        vox_feats = vox_feats.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2)
        print(f"TORCH: vox_feats shape: {vox_feats.shape}, dtype: {vox_feats.dtype}")
        # return vox_feats
        # Flatten to orthographic feature map
        ortho_feats = self.conv3d(vox_feats).view(batch, depth, width, -1)
        ortho_feats = F.relu(ortho_feats.permute(0, 3, 1, 2), inplace=True)

        # Block gradients to pixels which are not visible in the image
        # print(f"TORCH: ortho_feats shape: {ortho_feats.shape}, dtype: {ortho_feats.dtype}")

        return ortho_feats


def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)


@pytest.mark.parametrize(
    "batch,channels,cell_size,grid_height,depth,width",
    [
        (2, 128, 0.5, 6.0, 4, 4),
    ],
)
def test_oft_forward(batch, channels, cell_size, grid_height, depth, width):
    # Create dummy inputs
    # features = torch.randn(batch, channels, 16, 16)
    # calib = torch.randn(batch, 3, 4)
    # grid = torch.randn(batch, depth, width, 3)

    torch.manual_seed(0)
    features = torch.randn(1, 128, 48, 160)
    calib = torch.randn(1, 3, 4)
    grid = torch.randn(1, 160, 160, 3)
    oft = OFT(channels, cell_size, grid_height, scale=1 / 8.0)
    output = oft(features, calib, grid)

    # Output shape: (batch, channels, depth, width)
    print(f"output.shape:", output.shape)
    # assert output.shape[0] == batch
    # assert output.shape[1] == channels
    # assert output.shape[2] == depth-1
    # assert output.shape[3] == width
