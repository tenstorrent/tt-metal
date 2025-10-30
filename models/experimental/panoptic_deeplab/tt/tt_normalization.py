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
        Normalizes using device tensors, then converts to torch and back with sharded memory config.

        Args:
            input_tensor: Input tensor in NHWC format on device, values in [0,1]

        Returns:
            ImageNet normalized tensor on device with HEIGHT sharded memory config
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

        # Pull back to host and convert to torch
        torch_normalized = ttnn.to_torch(normalized)

        # Pad to match preprocess_nchw_input_tensor format (8 channels)
        SHARD_WIDTH = 8
        if torch_normalized.shape[-1] == 3:
            # Pad channels from 3 to 8 (NHWC format)
            torch_normalized = torch.nn.functional.pad(torch_normalized, (0, SHARD_WIDTH - 3), mode="constant", value=0)

        # Create sharded memory config matching preprocess_nchw_input_tensor
        HW = height * width
        core_range_set = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.device.core_grid.x - 1, self.device.core_grid.y - 1)
                )
            }
        )
        num_cores = self.device.core_grid.x * self.device.core_grid.y
        shard_height = (1 * HW + num_cores - 1) // num_cores

        sharded_memory_config = ttnn.create_sharded_memory_config_(
            shape=(shard_height, SHARD_WIDTH),
            core_grid=core_range_set,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Convert back to TTNN with sharded memory config
        return ttnn.from_torch(
            torch_normalized,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=sharded_memory_config,
        )
