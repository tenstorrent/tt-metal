# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn.functional as F
from loguru import logger
import math

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

from models.experimental.oft.reference.oft import EPSILON
from models.experimental.oft.reference.utils import perspective

from models.experimental.oft.tt.common import (
    Linear,
)

GRID_SAMPLE_NHW = 159 * 159
PAD_AMOUNT = 63
PAD_VALUE = 0.0
NUM_SLICES = 18


def split_tensor(tensor, num_slices, dim=1):
    # tensor shape: [1, total_height, 1, channels] (default split on dim=1)
    total_size = tensor.shape[dim]
    slice_size = total_size // num_slices

    splits = [slice_size] * (num_slices - 1)
    splits.append(total_size - slice_size * (num_slices - 1))
    return torch.split(tensor, splits, dim=dim)


def calculate_initialization_parameters(
    device,
    channels,
    cell_size,
    grid_height,
    feature_shape_hw,
    calib,
    grid,
    scale,
    use_precomputed_grid,
    num_slices=NUM_SLICES,
):
    y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
    y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
    # Expand the grid in the y dimension
    corners = grid.unsqueeze(1) + y_corners.view(-1, 1, 1, 3)

    # Project grid corners to image plane and normalize to [-1, 1]
    img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners, dtype=torch.float32)
    feature_height, feature_width = feature_shape_hw
    # Normalize to [-1, 1]
    img_size = corners.new([feature_width, feature_height]) / scale
    norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)
    # Get top-left and bottom-right coordinates of voxel bounding boxes
    bbox_corners = torch.cat(
        [
            torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
            torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
        ],
        dim=-1,
    )
    batch, height, depth, width, _ = bbox_corners.size()
    bbox_corners = bbox_corners.flatten(2, 3).permute(0, 2, 1, 3)
    area = (
        (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * feature_height * feature_width * 0.25 + EPSILON
    ).unsqueeze(1)
    visible = area > EPSILON

    area = 1 / area
    area_nhwc = (
        area.permute(0, 2, 3, 1)
        .repeat(1, 1, 1, channels)
        .reshape(1, depth * width, 1, channels * height)
        .permute(0, 2, 1, 3)
    )  # Convert to N1HW*C format
    visible_nhwc = (
        visible.permute(0, 2, 3, 1)
        .repeat(1, 1, 1, channels)
        .reshape(1, depth * width, 1, channels * height)
        .permute(0, 2, 1, 3)
    )  # Convert to N1HW*C format

    top_left_bc = bbox_corners[..., [0, 1]]
    top_left_bc = torch.nn.functional.pad(top_left_bc, ((0, 0, 0, 0, 0, PAD_AMOUNT, 0, 0)), value=PAD_VALUE)
    btm_right_bc = bbox_corners[..., [2, 3]]
    btm_right_bc = torch.nn.functional.pad(btm_right_bc, ((0, 0, 0, 0, 0, PAD_AMOUNT, 0, 0)), value=PAD_VALUE)
    top_right_bc = bbox_corners[..., [2, 1]]
    top_right_bc = torch.nn.functional.pad(top_right_bc, ((0, 0, 0, 0, 0, PAD_AMOUNT, 0, 0)), value=PAD_VALUE)
    btm_left_bc = bbox_corners[..., [0, 3]]
    btm_left_bc = torch.nn.functional.pad(btm_left_bc, ((0, 0, 0, 0, 0, PAD_AMOUNT, 0, 0)), value=PAD_VALUE)

    batch_size, grid_h, grid_w, _ = top_left_bc.shape
    input_shape_nhwc = [batch_size, feature_height, feature_width, channels]

    # Split each tensor into num_slices lists
    top_left_bc_slices = split_tensor(top_left_bc, num_slices)
    btm_right_bc_slices = split_tensor(btm_right_bc, num_slices)
    top_right_bc_slices = split_tensor(top_right_bc, num_slices)
    btm_left_bc_slices = split_tensor(btm_left_bc, num_slices)

    # Convert each slice to TT tensor
    if use_precomputed_grid:
        prepare_grid_lambda = lambda torch_grid_bf16, input_shape_nhwc: ttnn.to_device(
            ttnn.prepare_grid_sample_grid(
                ttnn.from_torch(torch_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32),
                input_shape_nhwc,
                padding_mode="zeros",
                output_dtype=ttnn.bfloat16,
            ),
            device,
        )

        top_left_bc_tt = [prepare_grid_lambda(t, input_shape_nhwc) for t in top_left_bc_slices]
        btm_right_bc_tt = [prepare_grid_lambda(t, input_shape_nhwc) for t in btm_right_bc_slices]
        top_right_bc_tt = [prepare_grid_lambda(t, input_shape_nhwc) for t in top_right_bc_slices]
        btm_left_bc_tt = [prepare_grid_lambda(t, input_shape_nhwc) for t in btm_left_bc_slices]
    else:
        prepare_corner_lambda = lambda t: ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

        top_left_bc_tt = [prepare_corner_lambda(t) for t in top_left_bc_slices]
        btm_right_bc_tt = [prepare_corner_lambda(t) for t in btm_right_bc_slices]
        top_right_bc_tt = [prepare_corner_lambda(t) for t in top_right_bc_slices]
        btm_left_bc_tt = [prepare_corner_lambda(t) for t in btm_left_bc_slices]

    reshape_corners_lambda = lambda t: ttnn.reshape(t, [batch_size, 1, grid_h // num_slices, grid_w * t.shape[-1]])
    top_left_bc_tt = [reshape_corners_lambda(t) for t in top_left_bc_tt]
    btm_right_bc_tt = [reshape_corners_lambda(t) for t in btm_right_bc_tt]
    top_right_bc_tt = [reshape_corners_lambda(t) for t in top_right_bc_tt]
    btm_left_bc_tt = [reshape_corners_lambda(t) for t in btm_left_bc_tt]

    visible_tt = ttnn.from_torch(visible_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    area = torch.nn.functional.pad(area_nhwc * visible_nhwc, ((0, 0, 0, PAD_AMOUNT, 0, 0, 0, 0)), value=PAD_VALUE)
    area = split_tensor(area, num_slices, dim=2)
    area_tt = [ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for a in area]
    return (
        [top_left_bc_tt, btm_right_bc_tt, top_right_bc_tt, btm_left_bc_tt],
        visible_tt,
        area_tt,
        [batch, height, depth, width],
    )


class OFT:
    def __init__(
        self,
        device,
        parameters,
        channels,
        cell_size,
        grid_height,
        features_shape_hw,
        calib,
        grid,
        scale,
        use_precomputed_grid,
        num_slices=NUM_SLICES,
    ):
        # params for conv3d
        self.linear_weight = parameters.conv3d.weight
        self.linear_bias = parameters.conv3d.bias
        self.scale = scale
        self.use_precomputed_grid = use_precomputed_grid
        self.features_shape_hw = features_shape_hw
        self.device = device

        self.input_dtype = ttnn.bfloat16

        self.num_slices = num_slices

        self.bbox_corners, self.visible, self.area, self.shape = calculate_initialization_parameters(
            device,
            channels,
            cell_size,
            grid_height,
            features_shape_hw,
            calib,
            grid,
            self.scale,
            use_precomputed_grid,
            num_slices=num_slices,
        )

        self.in_channels = self.linear_weight.shape[0]
        self.linear_weight = ttnn.reshape(
            self.linear_weight, [self.in_channels // self.shape[1], self.shape[1], channels]
        )
        self.linear_weight = ttnn.permute(self.linear_weight, (1, 0, 2))
        self.linear_weight = ttnn.reshape(self.linear_weight, [self.in_channels, channels])

        # integral_image_quantization_strategy
        # None - no quantization
        # "to_uint32" - quantize to uint32 before integral image, dequantize after
        # "to_float32" - quantize to float32 before integral image, dequantize after
        self.integral_image_quantization_strategy = "to_uint32"
        logger.info(f"Integral image quantization strategy: {self.integral_image_quantization_strategy}")
        if self.integral_image_quantization_strategy == "to_uint32":
            self.prescaler = ttnn.from_torch(torch.tensor(1024 * 1024), device=device, dtype=ttnn.bfloat16)
            self.postscaler = ttnn.from_torch(torch.tensor(1 / 1024 / 1024), device=device, dtype=ttnn.bfloat16)

        # Initialize sharding and linear layer configurations
        linear_pt = {
            "in_channels": self.in_channels,
            "out_channels": self.linear_weight.shape[1],
            "nhw": (GRID_SAMPLE_NHW + PAD_AMOUNT) // self.num_slices,
            "height_sharding": True,
        }

        self._setup_sharding_configs()
        self.linear_layer = Linear(self.linear_weight, self.linear_bias, linear_pt)

    def _setup_sharding_configs(self):
        """Setup sharding configurations for slicing operations"""
        compute_grid = self.device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
        self.sharding_strategy = "height"

        # Sharding parameters for slicing
        self.slice_memory_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            if self.sharding_strategy == "height"
            else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.shard_orientation = ttnn.ShardOrientation.ROW_MAJOR

    def create_sharded_tensor(
        self,
        input_tensor,
        target_shard_height=None,
        target_shard_width=None,
        sharding_strategy=None,
        memory_layout=None,
        use_all_cores=True,
    ):
        """
        Helper function to create a sharded tensor using available cores.

        Args:
            input_tensor: Input tensor to be sharded
            target_shard_height: Target height for each shard (optional, calculated if None)
            target_shard_width: Target width for each shard (optional, calculated if None)
            sharding_strategy: "height", "block", or None (uses self.sharding_strategy)
            memory_layout: Memory layout or None (uses self.slice_memory_layout)
            use_all_cores: Whether to use all available cores (default: True)

        Returns:
            Sharded tensor using all available cores

        Example usage:
            # Basic usage with automatic shard size calculation
            sharded_tensor = self.create_sharded_tensor(my_tensor)

            # Use specific shard dimensions
            sharded_tensor = self.create_sharded_tensor(
                my_tensor,
                target_shard_height=1024,
                target_shard_width=512
            )

            # Use height sharding strategy with all cores
            sharded_tensor = self.create_sharded_tensor(
                my_tensor,
                sharding_strategy="height",
                use_all_cores=True
            )
        """
        # Use class defaults if not specified
        strategy = sharding_strategy if sharding_strategy is not None else self.sharding_strategy
        mem_layout = memory_layout if memory_layout is not None else self.slice_memory_layout

        # Get tensor dimensions
        tensor_shape = input_tensor.shape
        n, h, w, channels = tensor_shape

        # Calculate core grid - use all available cores if requested
        if use_all_cores:
            grid_size = self.device.compute_with_storage_grid_size()
            core_grid = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)
        else:
            core_grid = self.core_grid

        # Calculate shard dimensions
        if target_shard_height is None or target_shard_width is None:
            if strategy == "height":
                # Height sharding: distribute height across cores
                total_elements = n * h * w
                per_core_elements = (total_elements + (core_grid.y * core_grid.x) - 1) // (core_grid.y * core_grid.x)
                shard_height = ((per_core_elements + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                shard_width = (channels + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
            else:  # block sharding
                # Block sharding: distribute both height and width
                per_core_h = (h + core_grid.y - 1) // core_grid.y
                per_core_w = (w + core_grid.x - 1) // core_grid.x
                shard_height = ((per_core_h + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                shard_width = ((per_core_w * channels + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        else:
            shard_height = target_shard_height
            shard_width = target_shard_width

        # Create shard strategy
        shard_strategy = ttnn.ShardStrategy.HEIGHT if strategy == "height" else ttnn.ShardStrategy.BLOCK

        # Create sharded memory configuration
        sharded_mem_config = ttnn.create_sharded_memory_config(
            (shard_height, shard_width),
            core_grid,
            shard_strategy,
            self.shard_orientation,
            use_height_and_width_as_shard_shape=True,
        )

        # Convert to sharded tensor
        if input_tensor.is_sharded():
            # If already sharded, convert to interleaved first
            interleaved_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
            sharded_tensor = ttnn.to_memory_config(interleaved_tensor, sharded_mem_config)
        else:
            # Convert directly to sharded
            sharded_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

        logger.debug(
            f"Created sharded tensor with shape {sharded_tensor.shape}, "
            f"shard_height: {shard_height}, shard_width: {shard_width}, "
            f"core_grid: {core_grid}, strategy: {strategy}"
        )

        return sharded_tensor

    def forward(self, device, features, calib, grid):
        if use_signpost:
            signpost(header="OFT block started")

        features = ttnn.reshape(features, [1, self.features_shape_hw[0], self.features_shape_hw[1], -1])
        if features.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
            features = ttnn.to_layout(features, ttnn.TILE_LAYOUT)

        if self.integral_image_quantization_strategy is None:
            integral_image = ttnn_integral_image_channel_last(features)
        elif self.integral_image_quantization_strategy == "to_uint32":
            features = ttnn.mul(features, self.prescaler, dtype=ttnn.bfloat16)
            features = ttnn.typecast(features, ttnn.uint32)
            integral_image = ttnn_integral_image_channel_last(features)
            integral_image = ttnn.typecast(integral_image, ttnn.bfloat16)
            integral_image = ttnn.mul(integral_image, self.postscaler, dtype=ttnn.bfloat16)
        elif self.integral_image_quantization_strategy == "to_float32":
            features = ttnn.typecast(features, ttnn.float32)
            integral_image = ttnn_integral_image_channel_last(features)
            integral_image = ttnn.typecast(integral_image, ttnn.bfloat16)

        if integral_image.get_layout() == ttnn.TILE_LAYOUT:
            integral_image = ttnn.to_layout(integral_image, ttnn.ROW_MAJOR_LAYOUT)

        integral_image = ttnn.to_memory_config(integral_image, ttnn.L1_MEMORY_CONFIG)

        grid_size = self.device.compute_with_storage_grid_size()
        core_grid = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)

        n, h, w, in_ch = [1, 1, GRID_SAMPLE_NHW + PAD_AMOUNT, self.in_channels]  # features.shape

        # Get output channels from linear weight
        out_ch = self.linear_weight.shape[1]

        # Calculate dynamic configurations based on tensor dimensions
        grid_sample_shard_height = (
            math.ceil(n * h * w // (self.num_slices * ttnn.TILE_SIZE) / (core_grid.y * core_grid.x)) * ttnn.TILE_SIZE
        )
        grid_sample_shard_width = math.ceil(in_ch // ttnn.TILE_SIZE) * ttnn.TILE_SIZE

        logger.debug(
            f"Grid sample shard dimensions - height: {grid_sample_shard_height}, width: {grid_sample_shard_width}, "
            f"core_grid: {core_grid}"
        )

        grid_sample_memory_config = ttnn.create_sharded_memory_config(
            (grid_sample_shard_height, grid_sample_shard_width),
            core_grid,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        out_initial = torch.randn([n, h, w, out_ch], dtype=torch.float32)
        ortho_feats = ttnn.from_torch(
            out_initial, self.input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        for i in range(self.num_slices):
            # Top left corner slice
            vox_feats_slice = ttnn.grid_sample(
                integral_image,
                self.bbox_corners[0][i],
                use_precomputed_grid=self.use_precomputed_grid,
                batch_output_channels=True,
                memory_config=grid_sample_memory_config,
            )
            vox_feats_slice = ttnn.to_layout(vox_feats_slice, ttnn.TILE_LAYOUT)
            logger.debug(f"Grid sample output slice {i} shape: {self.bbox_corners[0][i].shape}")
            logger.debug(f"Voxel features slice {i} initial shape: {vox_feats_slice.shape}")

            # Top right corner slice
            top_right_slice = ttnn.grid_sample(
                integral_image,
                self.bbox_corners[2][i],
                use_precomputed_grid=self.use_precomputed_grid,
                batch_output_channels=True,
                memory_config=grid_sample_memory_config,
            )

            top_right_slice = ttnn.to_layout(top_right_slice, ttnn.TILE_LAYOUT)
            ttnn.sub_(vox_feats_slice, top_right_slice)
            ttnn.deallocate(top_right_slice)

            # Bottom right corner slice
            btm_right_slice = ttnn.grid_sample(
                integral_image,
                self.bbox_corners[1][i],
                use_precomputed_grid=self.use_precomputed_grid,
                batch_output_channels=True,
                memory_config=grid_sample_memory_config,
            )
            btm_right_slice = ttnn.to_layout(btm_right_slice, ttnn.TILE_LAYOUT)

            ttnn.add_(vox_feats_slice, btm_right_slice)
            ttnn.deallocate(btm_right_slice)

            # Bottom left corner slice
            btm_left_slice = ttnn.grid_sample(
                integral_image,
                self.bbox_corners[3][i],
                use_precomputed_grid=self.use_precomputed_grid,
                batch_output_channels=True,
                memory_config=grid_sample_memory_config,
            )
            btm_left_slice = ttnn.to_layout(btm_left_slice, ttnn.TILE_LAYOUT)
            ttnn.sub_(vox_feats_slice, btm_left_slice)
            ttnn.deallocate(btm_left_slice)

            area_slice = self.create_sharded_tensor(self.area[i])
            ttnn.mul_(vox_feats_slice, area_slice)
            ttnn.deallocate(area_slice)

            vox_feats_slice = ttnn.move(vox_feats_slice)
            vox_feats_slice = self.linear_layer(vox_feats_slice, device)

            ttnn.sharded_to_interleaved_partial(
                vox_feats_slice, ortho_feats, self.num_slices, i, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        ortho_feats = ortho_feats[:, :, : w - PAD_AMOUNT, :]

        # this is used as intermediate tensor for tracking pcc over model
        # if removing intermediate tensors, remove this but call ttnn.deallocate(integral_image)
        integral_image = ttnn.to_torch(integral_image).permute(0, 3, 1, 2)

        if use_signpost:
            signpost(header="OFT block ended")
        # return ortho_feats
        return (
            ortho_feats,
            integral_image,
            self.bbox_corners[0],
            self.bbox_corners[1],
            self.bbox_corners[2],
            self.bbox_corners[3],
        )


def ttnn_integral_image_channel_last(features_nhwc):
    assert len(features_nhwc.shape) == 4, "Input tensor must be 4D"
    assert features_nhwc.shape[0] == 1, "Batch size must be 1"
    tmp = ttnn.cumsum(features_nhwc, dim=1, dtype=features_nhwc.dtype)
    # ttnn.deallocate(features_nhwc) remove if needed, for now it work without move
    # tmp = ttnn.move(tmp)
    return ttnn.cumsum(tmp, dim=2, dtype=features_nhwc.dtype)
