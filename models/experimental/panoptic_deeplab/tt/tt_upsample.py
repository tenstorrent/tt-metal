# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import ttnn
from models.common.lightweightmodule import LightweightModule


class BilinearUpsampleTorch(LightweightModule):
    """
    Torch Bilinear upsampling using pure matrix multiplication.
    Pre-computes interpolation matrices for efficiency.
    """

    def __init__(self, input_height, input_width, scale=4, input_channels_first=False, output_channels_first=None):
        self.H = input_height
        self.W = input_width
        self.scale = scale
        self.input_channels_first = input_channels_first
        # Default output format matches input format if not specified
        self.output_channels_first = (
            output_channels_first if output_channels_first is not None else input_channels_first
        )
        self.H_out = self.H * self.scale
        self.W_out = self.W * self.scale

        # Pre-compute interpolation coordinates
        row_coords = torch.linspace(0, self.H - 1, self.H_out)
        col_coords = torch.linspace(0, self.W - 1, self.W_out)

        # Pre-compute interpolation matrices
        self.Mh = self._interp_matrix(row_coords, self.H)
        self.Mw = self._interp_matrix(col_coords, self.W)

    def _interp_matrix(self, coords, size):
        floor = torch.floor(coords).long()
        ceil = torch.clamp(floor + 1, 0, size - 1)
        w = coords - floor.float()
        M = torch.zeros((size, len(coords)))
        M[floor, torch.arange(len(coords))] = 1 - w
        M[ceil, torch.arange(len(coords))] += w
        return M

    def _check_input_shape(self, img):
        if self.input_channels_first:
            # Input format: (B, C, H, W)
            B, C, H, W = img.shape
            assert H == self.H and W == self.W, f"Input shape mismatch: expected ({self.H}, {self.W}), got ({H}, {W})"
        else:
            # Input format: (B, H, W, C)
            B, H, W, C = img.shape
            assert H == self.H and W == self.W, f"Input shape mismatch: expected ({self.H}, {self.W}), got ({H}, {W})"

    def forward(self, img):
        self._check_input_shape(img)

        # 4D tensor matrix multiplication constraint: all operations must use 4D tensors
        # Convert input to 4D: (B, C, H, W) format for all operations
        if self.input_channels_first:
            X_4d = img  # Already (B, C, H, W)
        else:
            X_4d = img.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # Apply 4D tensor matrix multiplication using torch.matmul broadcasting
        # Step 1: Width interpolation - apply Mw to last dimension
        # X_4d: (B, C, H, W) @ Mw: (W, W_out) -> (B, C, H, W_out)
        Y_width = torch.matmul(X_4d, self.Mw)
        logger.warning(f"{Y_width.shape=} = {X_4d.shape=} @ {self.Mw.shape=}")

        # Step 2: Height interpolation - apply Mh.T to second-to-last dimension
        # Mh.T: (H_out, H) @ Y_width: (B, C, H, W_out) -> (B, C, H_out, W_out)
        Y_final_4d = torch.matmul(self.Mh.T, Y_width)
        logger.warning(f"{Y_final_4d.shape=} = {self.Mh.T.shape=} @ {Y_width.shape=}")

        if self.output_channels_first:
            # Return in channels-first format: (B, C, H_out, W_out)
            return Y_final_4d
        else:
            # Return in channels-last format: (B, H_out, W_out, C)
            # Y_final_4d is (B, C, H_out, W_out), permute to (B, H_out, W_out, C)
            return Y_final_4d.permute(0, 2, 3, 1)


class BilinearUpsampleMatmulTTNN(LightweightModule):
    """
    TTNN Bilinear upsampling using pure matrix multiplication.
    Pre-computes interpolation matrices for efficiency.
    """

    def __init__(
        self,
        device,
        input_batch,
        input_channels,
        input_height,
        input_width,
        scale=4,
        input_channels_first=False,
        output_channels_first=None,
    ):
        self.device = device
        self.N = input_batch
        self.C = input_channels
        self.H = input_height
        self.W = input_width
        self.scale = scale
        self.input_channels_first = input_channels_first
        # Default output format matches input format if not specified
        self.output_channels_first = (
            output_channels_first if output_channels_first is not None else input_channels_first
        )
        self.H_out = self.H * self.scale
        self.W_out = self.W * self.scale

        # Pre-compute interpolation coordinates (align_corners=True behavior)
        row_coords = torch.linspace(0, self.H - 1, self.H_out)
        col_coords = torch.linspace(0, self.W - 1, self.W_out)

        # Pre-compute interpolation matrices
        Mh = self._interp_matrix(row_coords, self.H)
        Mw = self._interp_matrix(col_coords, self.W)

        # Convert to TTNN tensors
        self.Mh = ttnn.unsqueeze_to_4D(ttnn.from_torch(Mh, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16))
        # Broadcast Mh to (self.N, self.C, :, :)
        Mh_broadcasted = Mh.unsqueeze(0).unsqueeze(0).expand(self.N, self.C, Mh.shape[0], Mh.shape[1])
        self.MhT = ttnn.unsqueeze_to_4D(
            ttnn.from_torch(
                Mh_broadcasted.transpose(-2, -1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
        )
        self.Mw = ttnn.unsqueeze_to_4D(ttnn.from_torch(Mw, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16))

    def _interp_matrix(self, coords, size):
        floor = torch.floor(coords).long()
        ceil = torch.clamp(floor + 1, 0, size - 1)
        w = coords - floor.float()
        M = torch.zeros((size, len(coords)), dtype=torch.float32)
        M[floor, torch.arange(len(coords))] = 1 - w
        M[ceil, torch.arange(len(coords))] += w
        return M

    def _check_input_shape(self, img):
        if self.input_channels_first:
            # Input format: (B, C, H, W)
            B, C, H, W = img.shape
            assert H == self.H and W == self.W, f"Input shape mismatch: expected ({self.H}, {self.W}), got ({H}, {W})"
        else:
            # Input format: (B, H, W, C)
            B, H, W, C = img.shape
            assert H == self.H and W == self.W, f"Input shape mismatch: expected ({self.H}, {self.W}), got ({H}, {W})"

    def forward(self, img_ttnn):
        # Note: Permute is faster on row major layout
        if self.input_channels_first == False:
            img_ttnn = ttnn.permute(img_ttnn, (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)

        if img_ttnn.layout != ttnn.TILE_LAYOUT:
            img_ttnn = ttnn.to_layout(img_ttnn, ttnn.TILE_LAYOUT)

        # Step 1: Width interpolation
        Y_width = ttnn.matmul(img_ttnn, self.Mw)
        logger.warning(f"{Y_width.shape=} = {img_ttnn.shape=} @ {self.Mw.shape=}")

        # Step 2: Height interpolation
        Y_final = ttnn.matmul(
            self.MhT, Y_width
        )  # batch broadcast works only with input A; broadcasted manually at init
        # Y_final = ttnn.matmul(Y_width,self.Mh, transpose_a=True)
        logger.warning(f"{Y_final.shape=} = {Y_width.shape=}.T @ {self.Mh.shape=}")

        if self.output_channels_first:
            # Y_final = ttnn.permute(Y_final, (0, 1, 3, 2))
            logger.warning(f"{Y_final.shape=}")
            return Y_final  # (B, C, H_out, W_out)
        else:
            # Note Permute is 10x faster on row major layout
            Y_final = ttnn.to_layout(Y_final, ttnn.ROW_MAJOR_LAYOUT)
            Y_final = ttnn.permute(Y_final, (0, 2, 3, 1))  # (B, H_out, W_out, C)
            Y_final = ttnn.to_layout(Y_final, ttnn.TILE_LAYOUT)
            logger.warning(f"{Y_final.shape=}")
            return Y_final  # (B, H_out, W_out, C)
