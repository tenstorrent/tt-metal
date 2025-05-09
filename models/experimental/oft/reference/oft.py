import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6


def perspective(matrix, vector):
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
        corners = grid.unsqueeze(1) + self.y_corners.view(-1, 1, 1, 3)

        img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners)

        img_height, img_width = features.size()[2:]
        img_size = corners.new([img_width, img_height]) / self.scale
        norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)

        bbox_corners = torch.cat(
            [
                torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
                torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
            ],
            dim=-1,
        )
        batch, _, depth, width, _ = bbox_corners.size()
        bbox_corners = bbox_corners.flatten(2, 3)

        area = (
            (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * img_height * img_width * 0.25 + EPSILON
        ).unsqueeze(1)
        visible = area > EPSILON

        integral_img = integral_image(features)
        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]])
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]])
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]])
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]])

        vox_feats = (top_left + btm_right - top_right - btm_left) / area
        vox_feats = vox_feats * visible.float()
        vox_feats = vox_feats.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2)

        ortho_feats = self.conv3d(vox_feats).view(batch, depth, width, -1)
        ortho_feats = F.relu(ortho_feats.permute(0, 3, 1, 2), inplace=True)

        return ortho_feats


def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)
