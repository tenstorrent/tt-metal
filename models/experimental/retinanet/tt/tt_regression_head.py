# models/experimental/retinanet/tt/regressionhead.py
import ttnn
from typing import List
from typing import Optional
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class RetinaNetHeadOptimizer:
    """Optimization configuration for RetinaNet head conv blocks"""

    conv_blocks: dict
    final_conv: dict
    groupnorm_config: dict  # NEW: Add GroupNorm-specific config


retinanet_head_optimizations = {
    "optimized": RetinaNetHeadOptimizer(
        conv_blocks={
            # "enable_act_double_buffer": True,
            # "enable_weights_double_buffer": True,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "act_block_h_override": 256,
            # "act_block_w_div": 1,
        },
        final_conv={
            # "enable_act_double_buffer": True,
            # "enable_weights_double_buffer": True,
        },
        groupnorm_config={
            "use_sharded_memory": True,
            "adaptive_grid_size": True,
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
    optimization_profile: str = "optimized",  # ADD THIS
) -> ttnn.Tensor:
    # Get optimization config
    opt_config = retinanet_head_optimizations[optimization_profile]
    # Create Conv2dConfig objects using the optimizer settings
    conv_blocks_config = ttnn.Conv2dConfig(**opt_config.conv_blocks)
    final_conv_config = ttnn.Conv2dConfig(**opt_config.final_conv)

    """
    TTNN implementation of RetinaNet regression head with all 4 conv layers + GroupNorm + ReLU.

    Args:
        feature_maps: List of FPN feature tensors in NHWC format
        parameters: Dict containing 'conv' (list of 4 Conv2dNormActivation params) and 'bbox_reg'
        device: TTNN device
        in_channels: Number of input channels (256 for RetinaNet)
        num_anchors: Number of anchors per location (9 for RetinaNet)
        batch_size: Batch size
        input_shapes: List of (H, W) tuples for each FPN level

    Returns:
        Concatenated bbox regressions in shape (N, total_anchors, 4)
    """
    all_bbox_regression = []

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
    # Initialize 4 Conv2dNormActivation blocks
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
            conv_config=conv_blocks_config,
        )
        conv_blocks.append(conv_block)

    for level_idx, x in enumerate(feature_maps):
        H, W = input_shapes[level_idx]
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing FPN Level {level_idx}: H={H}, W={W}")
        logger.info(f"{'='*60}")
        # Apply 4 conv blocks
        for conv_idx, conv_block in enumerate(conv_blocks):
            x = conv_block(
                x, batch_size=batch_size, input_height=H, input_width=W, fpn_level=level_idx, conv_block_idx=conv_idx
            )
        # Final bbox_reg conv layer
        bbox_reg_slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=4)

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
            conv_config=final_conv_config,
        )

        # Reshape to (N, H*W*num_anchors, 4)
        N, H_final, W_final, C_final = bbox_regression.shape
        bbox_regression = ttnn.reshape(bbox_regression, (N, H_final, W_final, num_anchors, 4))
        bbox_regression = ttnn.reshape(bbox_regression, (N, H_final * W_final * num_anchors, 4))

        all_bbox_regression.append(bbox_regression)

    # Concatenate all FPN levels
    output = ttnn.concat(all_bbox_regression, dim=1)
    return output
