# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import ttnn
from typing import List
from typing import Optional
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class RetinaNetHeadOptimizer:
    """Optimization configuration for RetinaNet head conv blocks"""

    fpn0_conv_blocks: dict
    fpn1_conv_blocks: dict
    fpn2_conv_blocks: dict
    fpn3_conv_blocks: dict
    fpn4_conv_blocks: dict

    # cls_logits configs (one per FPN level)
    fpn0_final_conv: dict
    fpn1_final_conv: dict
    fpn2_final_conv: dict
    fpn3_final_conv: dict
    fpn4_final_conv: dict


retinanet_head_optimizations = {
    "optimized": RetinaNetHeadOptimizer(
        # Conv block 0 - First convolution in the sequence
        # FPN Level 0: Largest spatial (64x64)
        fpn0_conv_blocks={
            "act_block_h_override": 1024,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn0_final_conv={
            "act_block_h_override": 1024,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        # FPN Level 1: Medium-large spatial (32x32)
        fpn1_conv_blocks={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn1_final_conv={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        # FPN Level 2: Medium spatial (16x16)
        fpn2_conv_blocks={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn2_final_conv={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        # FPN Level 3: Small spatial (8x8)
        fpn3_conv_blocks={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn3_final_conv={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        # FPN Level 4: Smallest spatial (4x4)
        fpn4_conv_blocks={
            "act_block_h_override": 32,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn4_final_conv={
            "act_block_h_override": 32,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
    ),
}


class Conv2dNormActivation:
    """
    TTNN implementation of Conv2d + GroupNorm + ReLU block.

    Encapsulates the pattern used in RetinaNet regression head:
    - Conv2d with DRAM slicing
    - GroupNorm with tile alignment padding
    - ReLU activation
    """

    def __init__(
        self,
        parameters: dict,
        device: ttnn.Device,
        in_channels: int = 256,
        out_channels: int = 256,
        kernel_size: tuple = (3, 3),
        stride: tuple = (1, 1),
        padding: tuple = (1, 1),
        num_groups: int = 32,
        grid_size: Optional[ttnn.CoreGrid] = None,
        input_mask: Optional[ttnn.Tensor] = None,
        model_config: dict = None,
        compute_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
        conv_config: Optional[ttnn.Conv2dConfig] = None,
    ):
        """
        Args:
            parameters: Dict with keys 'weight', 'norm_weight', 'norm_bias'
            device: TTNN device
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            num_groups: Number of groups for GroupNorm
            grid_size: CoreGrid for GroupNorm (defaults to 8x8)
            input_mask: Pre-created input mask for GroupNorm
        """
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        self.model_config = model_config
        self.compute_config = compute_config
        self.conv_config = conv_config
        # Store parameters
        self.conv_weight = parameters["weight"]
        self.conv_bias = parameters["bias"]
        self.norm_weight = parameters["norm_weight"]
        self.norm_bias = parameters["norm_bias"]

        # Grid size for GroupNorm
        self.grid_size = grid_size if grid_size is not None else ttnn.CoreGrid(y=8, x=8)

        # Input mask for GroupNorm
        self.input_mask = input_mask

        # DRAM slicing config for conv2d
        self.slice_config = ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceHeight,
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        batch_size: int,
        input_height: int,
        input_width: int,
        fpn_level: int = None,
        conv_block_idx: int = None,
    ) -> ttnn.Tensor:
        """
        Forward pass: Conv2d -> GroupNorm -> ReLU

        Args:
            x: Input tensor in NHWC format
            batch_size: Batch size
            input_height: Input height
            input_width: Input width

        Returns:
            Output tensor after Conv2d + GroupNorm + ReLU
        """
        from loguru import logger

        # Create hierarchical log prefix
        prefix = (
            f"[FPN{fpn_level}][Conv{conv_block_idx}]"
            if fpn_level is not None and conv_block_idx is not None
            else "[Conv]"
        )

        logger.info(f"{prefix} Starting forward pass")
        logger.info(f"{prefix}   Input: {x.shape}, batch={batch_size}, H={input_height}, W={input_width}")

        # Weight/bias debug with prefix
        logger.info(f"{prefix} [WEIGHT] Before conv2d:")
        logger.info(
            f"{prefix}   Shape: {self.conv_weight.shape}, Layout: {self.conv_weight.layout}, Storage: {self.conv_weight.storage_type()}"
        )

        logger.info(f"{prefix} [BIAS] Before conv2d:")
        if self.conv_bias is not None:
            logger.info(
                f"{prefix}   Shape: {self.conv_bias.shape}, Layout: {self.conv_bias.layout}, Storage: {self.conv_bias.storage_type()}"
            )

        # Conv2d operation
        logger.info(f"Conv2dNormActivation: Calling ttnn.conv2d...")
        logger.info(f"  kernel_size={list(self.kernel_size)}, stride={list(self.stride)}, padding={list(self.padding)}")

        x, [H_out, W_out], [prepared_weight, prepared_bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv_weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.conv_bias,
            kernel_size=list(self.kernel_size),
            stride=list(self.stride),
            padding=list(self.padding),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            slice_config=self.slice_config,
            compute_config=self.compute_config,
            conv_config=self.conv_config,
            return_output_dim=True,
            return_weights_and_bias=True,  # ADD THIS
        )

        # After conv2d
        logger.info(f"{prefix} [WEIGHT] After conv2d:")
        logger.info(
            f"{prefix}   Shape: {prepared_weight.shape}, Layout: {prepared_weight.layout}, Storage: {prepared_weight.storage_type()}"
        )

        logger.info(f"{prefix} [CACHE] Updated weights/bias")
        logger.info(f"{prefix} Output: {x.shape}")
        # Get output shape after conv
        N, H_out, W_out, C = x.shape
        logger.info(f"  After conv2d: shape={x.shape}, H_out={H_out}, W_out={W_out}")

        # Calculate padding needed for tile alignment
        # GroupNorm requires H_out * W_out divisible by (grid_size.y * 32)
        spatial_size = H_out * W_out
        required_size = ((spatial_size + self.grid_size.y * 32 - 1) // (self.grid_size.y * 32)) * (
            self.grid_size.y * 32
        )

        if spatial_size != required_size:
            # Pad spatial dimension to required size
            pad_amount = required_size - spatial_size

            # Reshape to (N, 1, H*W, C) for padding
            x_flat = ttnn.reshape(x, (N, 1, spatial_size, C))

            # Pad along spatial dimension
            x_padded = ttnn.pad(x_flat, padding=((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)
        else:
            # Reshape to (N, 1, H*W, C) without padding
            x_padded = ttnn.reshape(x, (N, 1, spatial_size, C))

        # Apply GroupNorm
        x_normalized = ttnn.group_norm(
            x_padded,
            num_groups=self.num_groups,
            input_mask=self.input_mask,
            weight=self.norm_weight,
            bias=self.norm_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=self.grid_size,
            inplace=False,
            compute_kernel_config=self.compute_config,
        )

        # Unpad
        if spatial_size != required_size:
            # Slice back to original spatial size
            x_normalized = x_normalized[:, :, :spatial_size, :]

        # Reshape back using PRESERVED dimensions
        x = ttnn.reshape(x_normalized, (N, input_height, input_width, C))
        logger.info(f"  After reshape back: shape={x.shape}")
        H_out = input_height
        W_out = input_width
        # ReLU activation
        x = ttnn.relu(x)

        return x


# Usage in ttnn_retinanet_regression_head
def ttnn_retinanet_regression_head(
    feature_maps: List[ttnn.Tensor],
    parameters: dict,
    device: ttnn.Device,
    in_channels: int = 256,
    num_anchors: int = 9,
    batch_size: int = 1,
    input_shapes: List[tuple] = None,
    model_config: dict = None,
    optimization_profile: str = "optimized",
) -> ttnn.Tensor:
    # Get optimization config
    opt_config = retinanet_head_optimizations[optimization_profile]

    # Map FPN level index to config attributes
    fpn_conv_configs = [
        opt_config.fpn0_conv_blocks,
        opt_config.fpn1_conv_blocks,
        opt_config.fpn2_conv_blocks,
        opt_config.fpn3_conv_blocks,
        opt_config.fpn4_conv_blocks,
    ]

    fpn_final_configs = [
        opt_config.fpn0_final_conv,
        opt_config.fpn1_final_conv,
        opt_config.fpn2_final_conv,
        opt_config.fpn3_final_conv,
        opt_config.fpn4_final_conv,
    ]

    all_bbox_regression = []

    # Setup shared resources
    grid_size = ttnn.CoreGrid(y=8, x=8)
    input_mask_tensor = ttnn.create_group_norm_input_mask(in_channels, 32, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=model_config["ACTIVATIONS_DTYPE"],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
        math_approx_mode=model_config.get("MATH_APPROX_MODE", False),
        fp32_dest_acc_en=model_config.get("FP32_DEST_ACC_EN", True),
        packer_l1_acc=model_config.get("PACKER_L1_ACC", False),
    )

    # Process each FPN level
    for fpn_idx, feature_map in enumerate(feature_maps):
        N, H, W, C = feature_map.shape
        logger.info(f"Processing FPN Level {fpn_idx}: H={H}, W={W}")

        # Get configs for THIS FPN level
        conv_blocks_config = fpn_conv_configs[fpn_idx]
        final_conv_config = fpn_final_configs[fpn_idx]

        # Create Conv2dConfig for this FPN's conv blocks
        if conv_blocks_config is not None:
            conv_config = ttnn.Conv2dConfig(**conv_blocks_config)
            logger.info(
                f"[FPN{fpn_idx}] ✓ Conv blocks config: act_block_h={conv_blocks_config.get('act_block_h_override', 'default')}, shard={conv_blocks_config.get('shard_layout', 'auto')}"
            )
        else:
            conv_config = None
            logger.info(f"[FPN{fpn_idx}] ⚠ No conv blocks config (using defaults)")

        # Create 4 Conv2dNormActivation blocks for THIS FPN level
        conv_blocks = []
        for conv_idx in range(4):
            conv_block = Conv2dNormActivation(
                parameters=parameters["conv"][conv_idx],
                device=device,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                num_groups=32,
                grid_size=grid_size,
                input_mask=input_mask_tensor,
                model_config=model_config,
                compute_config=compute_config,
                conv_config=conv_config,  # Use THIS FPN's config
            )
            conv_blocks.append(conv_block)

        # Apply 4 conv blocks to this FPN's feature map
        x = feature_map
        for conv_idx, conv_block in enumerate(conv_blocks):
            x = conv_block(
                x, batch_size=batch_size, input_height=H, input_width=W, fpn_level=fpn_idx, conv_block_idx=conv_idx
            )

        # Final bbox_reg conv layer with THIS FPN's config
        bbox_reg_slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=4)
        if final_conv_config is not None:
            bbox_reg_config = ttnn.Conv2dConfig(**final_conv_config)
            logger.info(
                f"[FPN{fpn_idx}][BBoxReg] ✓ Config: act_block_h={final_conv_config.get('act_block_h_override', 'default')}, shard={final_conv_config.get('shard_layout', 'auto')}"
            )
        else:
            bbox_reg_config = None
            logger.info(f"[FPN{fpn_idx}][BBoxReg] ⚠ No config (using defaults)")

        bbox_regression = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=parameters["bbox_reg"]["weight"],
            in_channels=in_channels,
            out_channels=num_anchors * 4,
            device=device,
            bias_tensor=parameters["bbox_reg"]["bias"],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=batch_size,
            input_height=H,
            input_width=W,
            slice_config=bbox_reg_slice_config,
            compute_config=compute_config,
            conv_config=bbox_reg_config,
        )

        # Reshape to (N, H*W*num_anchors, 4)
        N, H_final, W_final, C_final = bbox_regression.shape
        bbox_regression = ttnn.reshape(bbox_regression, (N, H_final, W_final, num_anchors, 4))
        bbox_regression = ttnn.reshape(bbox_regression, (N, H_final * W_final * num_anchors, 4))

        all_bbox_regression.append(bbox_regression)

    # Concatenate all FPN levels
    output = ttnn.concat(all_bbox_regression, dim=1)
    return output
