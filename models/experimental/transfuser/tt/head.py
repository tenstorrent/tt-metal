# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Tuple, List


class TTLidarCenterNetHead:
    """
    TTNN implementation of LidarCenterNetHead forward pass.

    This implements the seven detection heads (heatmap, wh, offset, yaw_class,
    yaw_res, velocity, brake) using TTNN operations.

    Note: Post-processing (get_bboxes, NMS) should remain on CPU.
    """

    def __init__(
        self,
        device,
        parameters,
        in_channel: int,
        feat_channel: int,
        num_classes: int,
        num_dir_bins: int,
    ):
        self.device = device
        self.in_channel = in_channel
        self.feat_channel = feat_channel
        self.num_classes = num_classes
        self.num_dir_bins = num_dir_bins

        # Store preprocessed parameters for each head
        self.heatmap_head_params = parameters["heatmap_head"]
        self.wh_head_params = parameters["wh_head"]
        self.offset_head_params = parameters["offset_head"]
        self.yaw_class_head_params = parameters["yaw_class_head"]
        self.yaw_res_head_params = parameters["yaw_res_head"]
        self.velocity_head_params = parameters["velocity_head"]
        self.brake_head_params = parameters["brake_head"]

    def _apply_head(
        self,
        feat: ttnn.Tensor,
        conv1_weight: ttnn.Tensor,
        conv1_bias: ttnn.Tensor,
        conv2_weight: ttnn.Tensor,
        conv2_bias: ttnn.Tensor,
        out_channels: int,
        batch_size: int,
        height: int,
        width: int,
    ) -> ttnn.Tensor:
        """
        Apply a single detection head (Conv3x3 -> ReLU -> Conv1x1).

        Args:
            feat: Input feature map in NHWC format
            conv1_weight: First conv layer weights (3x3)
            conv1_bias: First conv layer bias
            conv2_weight: Second conv layer weights (1x1)
            conv2_bias: Second conv layer bias
            out_channels: Number of output channels for this head
            batch_size, height, width: Input dimensions

        Returns:
            Output tensor after applying the head
        """

        # First conv: 3x3, padding=1
        x = ttnn.conv2d(
            input_tensor=feat,
            weight_tensor=conv1_weight,
            bias_tensor=conv1_bias,
            in_channels=self.in_channel,
            out_channels=self.feat_channel,
            device=self.device,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=batch_size,
            input_height=height,
            input_width=width,
        )

        # ReLU activation
        x = ttnn.relu(x)

        # Second conv: 1x1, no padding
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=conv2_weight,
            bias_tensor=conv2_bias,
            in_channels=self.feat_channel,
            out_channels=out_channels,
            device=self.device,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=batch_size,
            input_height=height,
            input_width=width,
        )

        return x

    def forward_single(
        self,
        feat: ttnn.Tensor,
        batch_size: int,
        height: int,
        width: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Forward pass for a single feature level.

        Args:
            feat: Input feature map (B, H, W, C) in TTNN NHWC format
            batch_size, height, width: Input dimensions

        Returns:
            Tuple of (center_heatmap, wh, offset, yaw_class, yaw_res, velocity, brake)
            All outputs are in NHWC format
        """
        # Apply heatmap head (outputs num_classes channels)
        center_heatmap_pred = self._apply_head(
            feat,
            self.heatmap_head_params["conv1_weight"],
            self.heatmap_head_params["conv1_bias"],
            self.heatmap_head_params["conv2_weight"],
            self.heatmap_head_params["conv2_bias"],
            self.num_classes,
            batch_size,
            height,
            width,
        )
        # center_heatmap_pred = ttnn.sigmoid(center_heatmap_pred)

        # Apply wh head (outputs 2 channels)
        wh_pred = self._apply_head(
            feat,
            self.wh_head_params["conv1_weight"],
            self.wh_head_params["conv1_bias"],
            self.wh_head_params["conv2_weight"],
            self.wh_head_params["conv2_bias"],
            2,
            batch_size,
            height,
            width,
        )

        # Apply offset head (outputs 2 channels)
        offset_pred = self._apply_head(
            feat,
            self.offset_head_params["conv1_weight"],
            self.offset_head_params["conv1_bias"],
            self.offset_head_params["conv2_weight"],
            self.offset_head_params["conv2_bias"],
            2,
            batch_size,
            height,
            width,
        )

        # Apply yaw class head (outputs num_dir_bins channels)
        yaw_class_pred = self._apply_head(
            feat,
            self.yaw_class_head_params["conv1_weight"],
            self.yaw_class_head_params["conv1_bias"],
            self.yaw_class_head_params["conv2_weight"],
            self.yaw_class_head_params["conv2_bias"],
            self.num_dir_bins,
            batch_size,
            height,
            width,
        )

        # Apply yaw residual head (outputs 1 channel)
        yaw_res_pred = self._apply_head(
            feat,
            self.yaw_res_head_params["conv1_weight"],
            self.yaw_res_head_params["conv1_bias"],
            self.yaw_res_head_params["conv2_weight"],
            self.yaw_res_head_params["conv2_bias"],
            1,
            batch_size,
            height,
            width,
        )

        # Apply velocity head (outputs 1 channel)
        velocity_pred = self._apply_head(
            feat,
            self.velocity_head_params["conv1_weight"],
            self.velocity_head_params["conv1_bias"],
            self.velocity_head_params["conv2_weight"],
            self.velocity_head_params["conv2_bias"],
            1,
            batch_size,
            height,
            width,
        )

        # Apply brake head (outputs 2 channels)
        brake_pred = self._apply_head(
            feat,
            self.brake_head_params["conv1_weight"],
            self.brake_head_params["conv1_bias"],
            self.brake_head_params["conv2_weight"],
            self.brake_head_params["conv2_bias"],
            2,
            batch_size,
            height,
            width,
        )

        return (
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            yaw_class_pred,
            yaw_res_pred,
            velocity_pred,
            brake_pred,
        )

    def forward(
        self,
        feats: List[ttnn.Tensor],
        batch_size: int,
        height: int,
        width: int,
    ) -> Tuple[
        List[ttnn.Tensor],
        List[ttnn.Tensor],
        List[ttnn.Tensor],
        List[ttnn.Tensor],
        List[ttnn.Tensor],
        List[ttnn.Tensor],
        List[ttnn.Tensor],
    ]:
        """
        Forward pass for multiple feature levels.

        This mimics the reference implementation's forward() method which uses
        multi_apply() to process a list of feature maps.

        Args:
            feats: List of feature tensors in NHWC format
            batch_size, height, width: Dimensions for each feature level

        Returns:
            Tuple of lists, one list per output type:
            ([heatmaps], [whs], [offsets], [yaw_classes], [yaw_res], [velocities], [brakes])
        """
        # Apply forward_single to each feature level
        results = []
        for feat in feats:
            result = self.forward_single(feat, batch_size, height, width)
            results.append(result)

        # Transpose results: from list of tuples to tuple of lists
        # results = [(h1,w1,o1,...), (h2,w2,o2,...), ...]
        # output = ([h1,h2,...], [w1,w2,...], [o1,o2,...], ...)
        if len(results) == 0:
            return tuple([[] for _ in range(7)])

        return tuple(map(list, zip(*results)))
