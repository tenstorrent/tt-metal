# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN-accelerated visualization module for DeepLabV3+ semantic segmentation.

This module provides on-device acceleration for:
- Argmax operation to get class predictions
- Color mapping using embedding operation (or where operations as fallback)
- Image blending operations
"""

from typing import Optional, Tuple, Dict
import numpy as np
import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

# Cityscapes color map (19 classes)
CITYSCAPES_COLORS = [
    [128, 64, 128],  # road
    [244, 35, 232],  # sidewalk
    [70, 70, 70],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],  # traffic light
    [220, 220, 0],  # traffic sign
    [107, 142, 35],  # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],  # sky
    [220, 20, 60],  # person
    [255, 0, 0],  # rider
    [0, 0, 142],  # car
    [0, 0, 70],  # truck
    [0, 60, 100],  # bus
    [0, 80, 100],  # train
    [0, 0, 230],  # motorcycle
    [119, 11, 32],  # bicycle
]


class TtDeeplabV3PlusVisualization(LightweightModule):
    """
    TTNN-accelerated visualization for DeepLabV3+ semantic segmentation.

    Accelerates:
    - Argmax operation on device
    - Color mapping using embedding operation
    - Image blending on device
    """

    def __init__(
        self,
        device: ttnn.Device,
        num_classes: int = 19,
        alpha: float = 0.6,
        dtype: ttnn.DataType = ttnn.bfloat16,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        use_softmax_alternative: bool = False,
    ):
        """
        Initialize the visualization module.

        Args:
            device: TTNN device to run operations on
            num_classes: Number of semantic classes (default: 19 for Cityscapes)
            alpha: Blending factor for image overlay (default: 0.6)
            dtype: Data type for TTNN operations (default: bfloat16)
            memory_config: Memory configuration for tensors (default: DRAM)
        """
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.alpha = alpha
        self.dtype = dtype
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.use_softmax_alternative = use_softmax_alternative

        # Create color lookup table: normalize colors to [0, 1] and convert to TTNN tensor
        colors_array = np.array(CITYSCAPES_COLORS[:num_classes], dtype=np.float32) / 255.0
        colors_torch = torch.from_numpy(colors_array).unsqueeze(0).unsqueeze(0)  # [1, 1, num_classes, 3]
        self.color_lookup = ttnn.from_torch(
            colors_torch,
            device=device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # Embedding requires ROW_MAJOR for weights
            memory_config=self.memory_config,
        )

    def argmax_on_device(self, semantic_pred: ttnn.Tensor, use_softmax_alternative: bool = None) -> ttnn.Tensor:
        """
        Perform argmax on device to get class predictions.

        Args:
            semantic_pred: Semantic prediction tensor [1, H, W, num_classes] in NHWC format
            use_softmax_alternative: If True, use softmax+max+eq+multiply+sum instead of argmax.
                                    If None, uses self.use_softmax_alternative from __init__.

        Returns:
            Class ID tensor [1, H, W, 1] in NHWC format
        """
        if use_softmax_alternative is None:
            use_softmax_alternative = self.use_softmax_alternative
        if use_softmax_alternative:
            return self._argmax_via_softmax(semantic_pred)
        return ttnn.argmax(semantic_pred, dim=3, keepdim=True, memory_config=self.memory_config)

    def _argmax_via_softmax(self, semantic_pred: ttnn.Tensor) -> ttnn.Tensor:
        """
        Alternative argmax implementation using softmax + max + eq + multiply + sum.

        This method assumes that max, eq, multiply, and sum operations are more efficient
        than argmax. The approach:
        1. Apply softmax to get probabilities
        2. Find max probability along channel dimension
        3. Create one-hot encoding by comparing each probability to max
        4. Convert one-hot to class index by multiplying with class indices and summing

        Args:
            semantic_pred: Semantic prediction tensor [1, H, W, num_classes] in NHWC format

        Returns:
            Class ID tensor [1, H, W, 1] in NHWC format
        """
        h, w, num_classes = semantic_pred.shape[1], semantic_pred.shape[2], semantic_pred.shape[3]

        # Step 1: Apply softmax to get probabilities
        probs = ttnn.softmax(semantic_pred, dim=3, memory_config=self.memory_config)

        # Step 2: Find max probability along channel dimension (dim=3)
        # ttnn.max returns (values, indices), we only need values
        # Note: ttnn.max may not support keepdim, so we'll use unsqueeze if needed
        max_result = ttnn.max(probs, dim=3, memory_config=self.memory_config)
        max_probs = max_result[0] if isinstance(max_result, tuple) else max_result  # [1, H, W]
        # Add keepdim manually if needed
        if len(max_probs.shape) == 3:  # [1, H, W] -> [1, H, W, 1]
            max_probs = ttnn.unsqueeze(max_probs, dim=3)

        # Step 3: Create one-hot encoding by comparing each probability to max
        # Expand max_probs to match probs shape: [1, H, W, 1] -> [1, H, W, num_classes]
        max_probs_expanded = ttnn.repeat(max_probs, (1, 1, 1, num_classes), memory_config=self.memory_config)

        # Create one-hot mask: probs >= max_probs (use ge to handle floating point precision)
        # For softmax, the max should be unique, but floating point precision might cause issues with eq
        # Using ge ensures we catch the max value even with small precision errors
        one_hot = ttnn.ge(probs, max_probs_expanded, memory_config=self.memory_config)  # [1, H, W, num_classes]

        # However, ge might match multiple classes if they're all >= max (shouldn't happen but be safe)
        # We need to ensure only one class matches. Since softmax ensures unique max,
        # we can use a trick: multiply by a mask that ensures only the first match counts
        # Actually, if multiple match, the sum will be wrong. Let's use a different approach:
        # Find the argmax directly from the max operation if it returns indices

        # Step 4: Convert one-hot to class index by multiplying with class indices and summing
        # Create class indices tensor: [0, 1, 2, ..., num_classes-1]
        # Shape: [1, 1, 1, num_classes] to broadcast to [1, H, W, num_classes]
        class_indices_torch = torch.arange(0, num_classes, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        class_indices = ttnn.from_torch(
            class_indices_torch,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )

        # Broadcast class_indices to match spatial dimensions: [1, 1, 1, num_classes] -> [1, H, W, num_classes]
        class_indices_broadcast = ttnn.repeat(class_indices, (1, h, w, 1), memory_config=self.memory_config)

        # Multiply one-hot mask with class indices: only the max class will contribute
        weighted_indices = ttnn.multiply(
            ttnn.typecast(one_hot, dtype=self.dtype, memory_config=self.memory_config),
            class_indices_broadcast,
            memory_config=self.memory_config,
        )

        # Sum along channel dimension to get class index: [1, H, W, num_classes] -> [1, H, W, 1]
        class_ids = ttnn.sum(weighted_indices, dim=3, keepdim=True, memory_config=self.memory_config)

        # Clamp class_ids to valid range [0, num_classes-1] to handle floating point precision issues
        # If multiple classes matched or no classes matched, the sum might be out of range
        zero_tensor = ttnn.zeros_like(class_ids, memory_config=self.memory_config)
        max_class_tensor = ttnn.full_like(
            class_ids, fill_value=float(num_classes - 1), memory_config=self.memory_config
        )
        class_ids = ttnn.maximum(class_ids, zero_tensor, memory_config=self.memory_config)
        class_ids = ttnn.minimum(class_ids, max_class_tensor, memory_config=self.memory_config)
        ttnn.deallocate(zero_tensor)
        ttnn.deallocate(max_class_tensor)

        # Clean up intermediate tensors
        ttnn.deallocate(probs)
        ttnn.deallocate(max_probs)
        ttnn.deallocate(max_probs_expanded)
        ttnn.deallocate(one_hot)
        ttnn.deallocate(class_indices)
        ttnn.deallocate(class_indices_broadcast)
        ttnn.deallocate(weighted_indices)

        return class_ids

    def color_map_on_device(self, class_ids: ttnn.Tensor, use_embedding: bool = True) -> ttnn.Tensor:
        """
        Map class IDs to colors using embedding or where operations.

        Args:
            class_ids: Class ID tensor [1, H, W, 1] in NHWC format, values in [0, num_classes-1]
            use_embedding: If True, use embedding operation (more efficient). If False, use where operations (fallback).

        Returns:
            Colorized image tensor [1, H, W, 3] in NHWC format with RGB colors
        """
        return self._color_map_embedding(class_ids) if use_embedding else self._color_map_where(class_ids)

    def _color_map_embedding(self, class_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        Map class IDs to colors using embedding operation (more efficient).

        Args:
            class_ids: Class ID tensor [1, H, W, 1] in NHWC format, values in [0, num_classes-1]

        Returns:
            Colorized image tensor [1, H, W, 3] in NHWC format with RGB colors
        """
        h, w = class_ids.shape[1], class_ids.shape[2]

        # Ensure TILE layout for typecast
        if class_ids.layout != ttnn.TILE_LAYOUT:
            class_ids = ttnn.to_layout(class_ids, ttnn.TILE_LAYOUT, memory_config=self.memory_config)

        # Convert to UINT32 and flatten: [1, H, W, 1] -> [1, 1, 1, H*W]
        class_ids_uint32 = ttnn.typecast(class_ids, dtype=ttnn.uint32, memory_config=self.memory_config)
        class_ids_flat = ttnn.reshape(class_ids_uint32, ttnn.Shape([1, 1, 1, h * w]))

        # Embedding lookup: [1, 1, 1, H*W] -> [1, 1, H*W, 3]
        colors_flat = ttnn.embedding(
            class_ids_flat,
            self.color_lookup,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )

        # Reshape back: [1, 1, H*W, 3] -> [1, H, W, 3]
        return ttnn.reshape(colors_flat, ttnn.Shape([1, h, w, 3]))

    def _color_map_where(self, class_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        Map class IDs to colors using where operations (fallback method).

        Args:
            class_ids: Class ID tensor [1, H, W, 1] in NHWC format, values in [0, num_classes-1]

        Returns:
            Colorized image tensor [1, H, W, 3] in NHWC format with RGB colors
        """
        h, w = class_ids.shape[1], class_ids.shape[2]

        # Initialize output with zeros
        vis_image = ttnn.zeros(
            shape=ttnn.Shape([1, h, w, 3]),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self.memory_config,
        )

        # For each class, create a mask and apply the color (fallback method)
        for class_id in range(self.num_classes):
            # Get color for this class and broadcast to spatial dimensions
            color_tensor = ttnn.slice(
                self.color_lookup,
                start=[0, 0, class_id, 0],
                end=[1, 1, class_id + 1, 3],
            )
            color_broadcast = ttnn.repeat(color_tensor, (1, h, w, 1), memory_config=self.memory_config)

            # Create mask: class_ids == class_id
            class_id_tensor = ttnn.full(
                shape=class_ids.shape,
                fill_value=float(class_id),
                dtype=class_ids.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=self.memory_config,
            )
            mask = ttnn.eq(class_ids, class_id_tensor, memory_config=self.memory_config)
            mask_expanded = ttnn.repeat(mask, (1, 1, 1, 3), memory_config=self.memory_config)

            # Apply color where mask is true
            vis_image = ttnn.where(mask_expanded, color_broadcast, vis_image, memory_config=self.memory_config)

            # Deallocate temporary tensors
            ttnn.deallocate(class_id_tensor)
            ttnn.deallocate(mask)
            ttnn.deallocate(mask_expanded)
            ttnn.deallocate(color_broadcast)

        return vis_image

    def blend_images_on_device(
        self,
        vis_image: ttnn.Tensor,
        original_image: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Blend visualization image with original image on device.

        Args:
            vis_image: Colorized visualization [1, H, W, 3] in NHWC format
            original_image: Original image [1, H, W, 3] in NHWC format, normalized to [0, 1]

        Returns:
            Blended image [1, H, W, 3] in NHWC format
        """
        # Blend: (1-alpha) * original + alpha * vis_image
        vis_weighted = ttnn.mul(vis_image, self.alpha, memory_config=self.memory_config)
        orig_weighted = ttnn.mul(original_image, 1.0 - self.alpha, memory_config=self.memory_config)
        return ttnn.add(orig_weighted, vis_weighted, memory_config=self.memory_config)

    def forward(
        self,
        semantic_pred: ttnn.Tensor,
        original_image: Optional[ttnn.Tensor] = None,
        num_channels_to_use: Optional[int] = None,
    ) -> Tuple[ttnn.Tensor, Dict]:
        """
        Perform full visualization pipeline on device.

        Args:
            semantic_pred: Semantic prediction tensor [1, H, W, num_classes] in NHWC format (TTNN tensor on device)
            original_image: Optional original image [1, H, W, 3] in NHWC format, normalized to [0, 1]
            num_channels_to_use: Optional number of channels to use (for slicing padded channels).
                                If None, uses all channels.

        Returns:
            Tuple of:
            - vis_image: Visualization tensor [1, H, W, 3] in NHWC format
            - panoptic_info: Dictionary with visualization metadata
        """
        # Slice channels if needed (handle padding)
        if num_channels_to_use is not None:
            semantic_pred = ttnn.slice(
                semantic_pred,
                start=[0, 0, 0, 0],
                end=[1, semantic_pred.shape[1], semantic_pred.shape[2], num_channels_to_use],
                memory_config=self.memory_config,
            )

        # Argmax to get class IDs, then map to colors
        class_ids = self.argmax_on_device(semantic_pred, use_softmax_alternative=True)
        vis_image = self.color_map_on_device(class_ids)

        # Blend with original image if provided
        if original_image is not None:
            vis_image = self.blend_images_on_device(vis_image, original_image)

        # Create panoptic info from class_ids
        class_ids_torch = ttnn.to_torch(class_ids).float().squeeze().cpu().numpy()
        unique_classes = np.unique(class_ids_torch)

        panoptic_info = {
            "mode": "DEEPLAB_V3_PLUS",
            "num_classes": len(unique_classes),
            "class_distribution": {int(cls): int(np.sum(class_ids_torch == cls)) for cls in unique_classes},
        }

        return vis_image, panoptic_info
