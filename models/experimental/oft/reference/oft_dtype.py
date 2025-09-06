import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6


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
    """Orthographic Feature Transform (OFT) module.

    This module transforms perspective image features into orthographic (bird's eye view)
    features using camera calibration and a 3D grid.

    Args:
        channels (int): Number of input feature channels
        cell_size (float): Size of each grid cell
        grid_height (float): Height of the grid
        scale (float, optional): Scale factor for image coordinates. Defaults to 1.
        dtype (torch.dtype, optional): Data type for computations. Must be one of
            [torch.float16, torch.bfloat16, torch.float32]. Defaults to torch.float32.
    """

    def __init__(self, channels, cell_size, grid_height, scale=1, dtype=torch.float32):
        super().__init__()
        print(f"{cell_size=} {grid_height=} {scale=}")

        # Validate dtype parameter
        if dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Must be one of [torch.float16, torch.bfloat16, torch.float32]"
            )

        self.dtype = dtype

        y_corners = torch.arange(0, grid_height, cell_size, dtype=self.dtype) - grid_height / 2.0
        y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
        self.register_buffer("y_corners", y_corners)

        # self.conv3d = nn.Conv2d((len(y_corners)-1) * channels, channels,1)
        self.conv3d = nn.Linear((len(y_corners) - 1) * channels, channels)
        self.scale = scale
        self.cell_size = cell_size
        self.grid_height = grid_height

        # Ensure the linear layer uses the specified dtype
        self.conv3d = self.conv3d.to(dtype=self.dtype)

    def set_dtype(self, dtype):
        """Change the dtype of the module and all its parameters.

        Args:
            dtype (torch.dtype): New data type. Must be one of
                [torch.float16, torch.bfloat16, torch.float32].

        Returns:
            self: Returns self for method chaining.
        """
        if dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Must be one of [torch.float16, torch.bfloat16, torch.float32]"
            )

        self.dtype = dtype
        self.y_corners = self.y_corners.to(dtype=dtype)
        self.conv3d = self.conv3d.to(dtype=dtype)
        return self

    def get_dtype(self):
        """Get the current dtype of the module.

        Returns:
            torch.dtype: Current data type.
        """
        return self.dtype

    def forward(self, features, calib, grid):
        # Ensure inputs are in the correct dtype
        features = features.to(dtype=self.dtype)
        calib = calib.to(dtype=self.dtype)
        grid = grid.to(dtype=self.dtype)

        # Print calib tensor in red color
        import termcolor

        print(termcolor.colored(f"{self.cell_size=} {self.grid_height=} {self.scale=}", "blue"))
        print(termcolor.colored(f"{features.shape=} {features.dtype=}", "red"))
        # print(termcolor.colored(f"Calib tensor: {calib}", "red"))
        print(termcolor.colored(f"{grid.shape=} {grid.dtype=}", "yellow"))
        # print(termcolor.colored(f"Grid tensor: {grid}", "yellow"))
        # Expand the grid in the y dimension
        corners = grid.unsqueeze(1) + self.y_corners.view(-1, 1, 1, 3)

        # Project grid corners to image plane and normalize to [-1, 1]
        img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners)

        # Normalize to [-1, 1]
        img_height, img_width = features.size()[2:]
        img_size = torch.tensor([img_width, img_height], dtype=self.dtype, device=corners.device) / self.scale
        norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)

        # Get top-left and bottom-right coordinates of voxel bounding boxes
        bbox_corners = torch.cat(
            [
                torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
                torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
            ],
            dim=-1,
        )
        assert (
            bbox_corners.dtype == self.dtype
        ), f"Expected bbox_corners dtype to be {self.dtype}, but got {bbox_corners.dtype}"
        batch, _, depth, width, _ = bbox_corners.size()
        bbox_corners = bbox_corners.flatten(2, 3)

        # Compute the area of each bounding box
        epsilon_val = torch.tensor(EPSILON, dtype=self.dtype, device=bbox_corners.device)
        area = (
            (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * img_height * img_width * 0.25 + epsilon_val
        ).unsqueeze(1)
        visible = (area > epsilon_val).to(dtype=self.dtype)
        assert (
            epsilon_val.dtype == self.dtype
        ), f"Expected epsilon_val dtype to be {epsilon_val.dtype}, but got {epsilon_val.dtype}"
        assert area.dtype == self.dtype, f"Expected area dtype to be {self.dtype}, but got {area.dtype}"
        assert visible.dtype == self.dtype, f"Expected visible dtype to be {visible.dtype}, but got {area.dtype}"

        handle_file = False
        if handle_file:
            # Save bbox_corners to a file
            ref_file = (
                f"bbox_corners_{features.shape[0]}_{features.shape[1]}_{features.shape[2]}_{features.shape[3]}.pt"
            )
            import os

            if os.path.exists(ref_file):
                ref_bbox_corners = torch.load(ref_file)
            else:
                print(f"Saving bbox_corners to {ref_file}")
                torch.save(
                    bbox_corners,
                    f"bbox_corners_{features.shape[0]}_{features.shape[1]}_{features.shape[2]}_{features.shape[3]}.pt",
                )
                ref_bbox_corners = bbox_corners.clone()

            # Check if ref_bbox_corners and bbox_corners are the same
            if not torch.allclose(ref_bbox_corners, bbox_corners, atol=1e-6):
                print("Warning: ref_bbox_corners and bbox_corners are not the same.")

        # Sample integral image at bounding box locations
        integral_img = integral_image(features)
        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]], align_corners=False)
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]], align_corners=False)
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]], align_corners=False)
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]], align_corners=False)

        # Compute voxel features (ignore features which are not visible)
        vox_feats = top_left + btm_right - top_right - btm_left
        ref_vox_feats_ = vox_feats.clone()
        vox_feats = vox_feats / area
        ref_vox_feats_over_area = vox_feats.clone()
        vox_feats = vox_feats * visible.to(dtype=self.dtype)
        ref_vox_feats = vox_feats.clone()
        # vox_feats = vox_feats.view(batch, -1, depth, width)
        vox_feats = vox_feats.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2)

        # Flatten to orthographic feature map
        ortho_feats = self.conv3d(vox_feats).view(batch, depth, width, -1)
        ortho_feats = F.relu(ortho_feats.permute(0, 3, 1, 2), inplace=True)
        # ortho_feats = F.relu(self.conv3d(vox_feats))

        # Block gradients to pixels which are not visible in the image

        # return ortho_feats
        return ortho_feats, (
            top_left,
            btm_right,
            top_right,
            btm_left,
            ref_vox_feats_,
            ref_vox_feats_over_area,
            ref_vox_feats,
            1 / area,
            visible,
        )


def integral_image(features):
    """Compute integral image preserving the input dtype."""
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)
