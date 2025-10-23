# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch
from loguru import logger
import ttnn


class TtImageNetNormalization:
    """
    Efficient ImageNet normalization module for TTNN tensors.

    Pre-loads mean and std tensors to device during initialization to avoid
    repeated data movement on every forward pass.
    """

    def __init__(self, device: ttnn.Device, target_size: Tuple[int, int]):
        """
        Initialize with device and target image size.

        Args:
            device: TTNN device
            target_size: Target size as (height, width)
        """
        self.device = device
        self.height, self.width = target_size

        # ImageNet normalization constants
        # mean = [0.485, 0.456, 0.406] for RGB channels
        # std = [0.229, 0.224, 0.225] for RGB channels

        # Pre-create mean and std tensors on device with target size
        # Shape: [1, H, W, 3] for NHWC format
        mean_values = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3).expand(1, self.height, self.width, 3)
        self.mean_tensor = ttnn.from_torch(
            mean_values, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )

        std_values = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3).expand(1, self.height, self.width, 3)
        self.std_tensor = ttnn.from_torch(std_values, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

        logger.debug(f"TtImageNetNormalization initialized for size {target_size}")

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """
        Apply ImageNet normalization to input tensor.

        Args:
            input_tensor: Input tensor in NHWC format on device, values in [0,1]

        Returns:
            ImageNet normalized tensor on device
        """
        # Verify input tensor size matches initialized size
        batch_size, height, width, channels = input_tensor.shape
        if height != self.height or width != self.width:
            raise ValueError(
                f"Input tensor size ({height}, {width}) doesn't match initialized size ({self.height}, {self.width})"
            )

        # Apply normalization: (input - mean) / std
        normalized = ttnn.subtract(input_tensor, self.mean_tensor)
        normalized = ttnn.divide(normalized, self.std_tensor)

        return normalized
