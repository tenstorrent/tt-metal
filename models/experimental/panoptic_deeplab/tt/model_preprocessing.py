# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters
from loguru import logger

from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_conv2d_wrapper import Conv2d
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    UpsampleConfiguration,
)


def fuse_conv_bn_weights_ttnn(
    conv_weight_ttnn, conv_bias_ttnn, bn_weight_ttnn, bn_bias_ttnn, bn_running_mean_ttnn, bn_running_var_ttnn, eps=1e-5
):
    """
    Fuse convolution and batch normalization weights for TTNN tensors.

    This function implements the weight fusion described in:
    https://medium.com/@sim30217/fusing-convolution-with-batch-normalization-f9fe13b3c111

    The mathematical formulation is:
    - new_weight = conv_weight * (bn_weight / sqrt(bn_running_var + eps))
    - new_bias = (conv_bias - bn_running_mean) * (bn_weight / sqrt(bn_running_var + eps)) + bn_bias

    Args:
        conv_weight_ttnn (ttnn.Tensor): Convolution weight tensor
        conv_bias_ttnn (ttnn.Tensor or None): Convolution bias tensor or None
        bn_weight_ttnn (ttnn.Tensor): BatchNorm weight (gamma) tensor
        bn_bias_ttnn (ttnn.Tensor): BatchNorm bias (beta) tensor
        bn_running_mean_ttnn (ttnn.Tensor): BatchNorm running mean tensor
        bn_running_var_ttnn (ttnn.Tensor): BatchNorm running variance tensor
        eps (float): BatchNorm epsilon value for numerical stability

    Returns:
        tuple: (fused_weight_ttnn, fused_bias_ttnn) where both are ttnn.Tensors
    """
    # Convert TTNN tensors to PyTorch tensors for computation
    conv_weight = ttnn.to_torch(conv_weight_ttnn)
    conv_bias = ttnn.to_torch(conv_bias_ttnn) if conv_bias_ttnn is not None else None
    bn_weight = ttnn.to_torch(bn_weight_ttnn).squeeze()  # Remove batch/spatial dims if present
    bn_bias = ttnn.to_torch(bn_bias_ttnn).squeeze()
    bn_running_mean = ttnn.to_torch(bn_running_mean_ttnn).squeeze()
    bn_running_var = ttnn.to_torch(bn_running_var_ttnn).squeeze()

    # Ensure all tensors are on the same device and dtype
    device = conv_weight.device
    dtype = conv_weight.dtype

    # Move all tensors to the same device if needed
    bn_weight = bn_weight.to(device=device, dtype=dtype)
    bn_bias = bn_bias.to(device=device, dtype=dtype)
    bn_running_mean = bn_running_mean.to(device=device, dtype=dtype)
    bn_running_var = bn_running_var.to(device=device, dtype=dtype)

    # Handle case where conv has no bias
    if conv_bias is None:
        conv_bias = torch.zeros(conv_weight.shape[0], device=device, dtype=dtype)
    else:
        conv_bias = conv_bias.to(device=device, dtype=dtype)

    # Compute the scaling factor: bn_weight / sqrt(bn_running_var + eps)
    scale = bn_weight / torch.sqrt(bn_running_var + eps)

    # Reshape scale to match the dimensions of the convolutional weights [out_channels, 1, 1, 1]
    scale_reshaped = scale.view(-1, 1, 1, 1)

    # Scale the convolutional weights
    fused_weight = conv_weight * scale_reshaped

    # Compute the fused bias: (conv_bias - bn_running_mean) * scale + bn_bias
    fused_bias = (conv_bias - bn_running_mean) * scale + bn_bias

    # Convert back to TTNN tensors
    # Weights stay on host for now
    fused_weight_ttnn = ttnn.from_torch(fused_weight, dtype=ttnn.bfloat16)

    # Handle bias tensor shape for TTNN (reshape to [1, 1, 1, -1] if needed)
    if len(fused_bias.shape) == 1:
        fused_bias = fused_bias.reshape((1, 1, 1, -1))
    fused_bias_ttnn = ttnn.from_torch(fused_bias, device=conv_weight_ttnn.device(), dtype=ttnn.bfloat16)

    return fused_weight_ttnn, fused_bias_ttnn


def fuse_conv_bn_parameters(parameters, eps=1e-5):
    """
    Fuse Conv+BatchNorm patterns in preprocessed parameters.

    This function takes the parameters object returned by create_panoptic_deeplab_parameters()
    and performs Conv+BN fusion, returning a new parameters object with only the fused
    convolution weights.

    Args:
        parameters: Parameter dict returned by create_panoptic_deeplab_parameters()
        eps (float): BatchNorm epsilon value for numerical stability

    Returns:
        dict: New parameter dict with fused Conv weights and biases
    """

    def process_module(module_params, path=""):
        """Recursively process parameters and fuse Conv+BN patterns."""
        fused_params = {}

        for key, value in module_params.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, dict):
                # Check if this is a Conv+BN pattern (has both 'weight' and 'norm' keys)
                if "weight" in value and "norm" in value:
                    logger.debug(f"Fusing Conv+BN parameters at: {current_path}")

                    # Extract conv parameters (TTNN tensors)
                    conv_weight = value["weight"]
                    conv_bias = value.get("bias", None)

                    # Extract norm parameters (TTNN tensors)
                    norm_params = value["norm"]
                    bn_weight = norm_params["weight"]
                    bn_bias = norm_params["bias"]
                    bn_running_mean = norm_params["running_mean"]
                    bn_running_var = norm_params["running_var"]

                    # Perform fusion
                    fused_weight, fused_bias = fuse_conv_bn_weights_ttnn(
                        conv_weight_ttnn=conv_weight,
                        conv_bias_ttnn=conv_bias,
                        bn_weight_ttnn=bn_weight,
                        bn_bias_ttnn=bn_bias,
                        bn_running_mean_ttnn=bn_running_mean,
                        bn_running_var_ttnn=bn_running_var,
                        eps=eps,
                    )

                    # Store fused conv parameters (now TTNN tensors)
                    fused_params[key] = {"weight": fused_weight, "bias": fused_bias}

                    # Update Conv2dConfiguration with fused weights if present (for TT CNN Builder API)
                    if "conv_config" in value:
                        from dataclasses import replace

                        # Update the Conv2dConfiguration to use the fused TTNN weights and bias
                        fused_config = replace(value["conv_config"], weight=fused_weight, bias=fused_bias)
                        fused_params[key]["conv_config"] = fused_config
                        logger.debug(f"Updated conv_config with fused weights for {current_path}")

                    # Preserve other configuration objects if present
                    if "pool_config" in value:
                        fused_params[key]["pool_config"] = value["pool_config"]
                    if "upsample_config" in value:
                        fused_params[key]["upsample_config"] = value["upsample_config"]
                else:
                    # Recursively process nested dictionaries
                    fused_params[key] = process_module(value, current_path)

            elif isinstance(value, list):
                # Handle lists (like ParameterList structures)
                fused_list = []
                for i, item in enumerate(value):
                    item_path = f"{current_path}[{i}]"
                    if isinstance(item, dict) and "weight" in item and "norm" in item:
                        logger.debug(f"Fusing Conv+BN parameters at: {item_path}")

                        # Extract conv parameters (TTNN tensors)
                        conv_weight = item["weight"]
                        conv_bias = item.get("bias", None)

                        # Extract norm parameters (TTNN tensors)
                        norm_params = item["norm"]
                        bn_weight = norm_params["weight"]
                        bn_bias = norm_params["bias"]
                        bn_running_mean = norm_params["running_mean"]
                        bn_running_var = norm_params["running_var"]

                        # Perform fusion
                        fused_weight, fused_bias = fuse_conv_bn_weights_ttnn(
                            conv_weight_ttnn=conv_weight,
                            conv_bias_ttnn=conv_bias,
                            bn_weight_ttnn=bn_weight,
                            bn_bias_ttnn=bn_bias,
                            bn_running_mean_ttnn=bn_running_mean,
                            bn_running_var_ttnn=bn_running_var,
                            eps=eps,
                        )

                        # Store fused conv parameters
                        fused_item = {"weight": fused_weight, "bias": fused_bias}

                        # Update Conv2dConfiguration with fused weights if present (for TT CNN Builder API)
                        if "conv_config" in item:
                            from dataclasses import replace

                            # Update the Conv2dConfiguration to use the fused TTNN weights and bias
                            fused_config = replace(item["conv_config"], weight=fused_weight, bias=fused_bias)
                            fused_item["conv_config"] = fused_config
                            logger.debug(f"Updated conv_config with fused weights for {item_path}")

                        # Preserve other configuration objects if present
                        if "pool_config" in item:
                            fused_item["pool_config"] = item["pool_config"]
                        if "upsample_config" in item:
                            fused_item["upsample_config"] = item["upsample_config"]
                        fused_list.append(fused_item)
                    elif isinstance(item, dict):
                        # Recursively process nested dict items
                        fused_list.append(process_module(item, item_path))
                    else:
                        # Keep non-dict items as-is
                        fused_list.append(item)

                fused_params[key] = fused_list

            else:
                fused_params[key] = value

        return fused_params

    return process_module(parameters)


def compute_panoptic_deeplab_input_shapes(input_height: int = 512, input_width: int = 1024, batch_size: int = 1):
    """
    Compute input shapes for all layers in Panoptic-DeepLab model.

    This function hardcodes the layer input shapes based on the model architecture.
    The shapes are deterministic given the input image dimensions.

    Architecture flow:
    1. Stem: 3 convs (first with stride 2) + maxpool (stride 2) -> reduces spatial dims by 4x
    2. ResNet backbone: res2 (256x512), res3 (128x256), res4 (64x128), res5 (64x128)
    3. ASPP: processes res5 output
    4. Decoder: upsamples progressively from res5 -> res4 -> res3
    5. Heads: produce final segmentation outputs at 1/4 resolution

    Args:
        input_height: Input image height (default: 512)
        input_width: Input image width (default: 1024)
        batch_size: Batch size (default: 1)

    Returns:
        dict: Mapping from layer name to (batch_size, height, width, channels)
    """
    shapes = {}

    # === STEM LAYERS ===
    # Input: HxWx3
    # conv1: stride 2, 3x3 -> H/2 x W/2 x 64
    # conv2: stride 1, 3x3 -> H/2 x W/2 x 64
    # conv3: stride 1, 3x3 -> H/2 x W/2 x 128
    # maxpool: stride 2, 3x3 -> H/4 x W/4 x 128

    shapes["backbone.stem.conv1"] = (batch_size, input_height, input_width, 3)
    shapes["backbone.stem.conv2"] = (batch_size, input_height // 2, input_width // 2, 64)
    shapes["backbone.stem.conv3"] = (batch_size, input_height // 2, input_width // 2, 64)
    shapes["backbone.stem.maxpool"] = (batch_size, input_height // 2, input_width // 2, 128)

    # === RES2 STAGE (stride 1, no spatial reduction) ===
    # Input: H/4 x W/4 x 128 -> Output: H/4 x W/4 x 256
    res2_h, res2_w = input_height // 4, input_width // 4

    # res2.0 - first bottleneck with shortcut
    shapes["backbone.res2.0.conv1"] = (batch_size, res2_h, res2_w, 128)  # 1x1: 128 -> 64
    shapes["backbone.res2.0.conv2"] = (batch_size, res2_h, res2_w, 64)  # 3x3: 64 -> 64
    shapes["backbone.res2.0.conv3"] = (batch_size, res2_h, res2_w, 64)  # 1x1: 64 -> 256
    shapes["backbone.res2.0.shortcut"] = (batch_size, res2_h, res2_w, 128)  # 1x1: 128 -> 256

    # res2.1 - standard bottleneck
    shapes["backbone.res2.1.conv1"] = (batch_size, res2_h, res2_w, 256)
    shapes["backbone.res2.1.conv2"] = (batch_size, res2_h, res2_w, 64)
    shapes["backbone.res2.1.conv3"] = (batch_size, res2_h, res2_w, 64)

    # res2.2 - standard bottleneck
    shapes["backbone.res2.2.conv1"] = (batch_size, res2_h, res2_w, 256)
    shapes["backbone.res2.2.conv2"] = (batch_size, res2_h, res2_w, 64)
    shapes["backbone.res2.2.conv3"] = (batch_size, res2_h, res2_w, 64)

    # === RES3 STAGE (stride 2, spatial reduction by 2x) ===
    # Input: H/4 x W/4 x 256 -> Output: H/8 x W/8 x 512
    res3_h, res3_w = input_height // 8, input_width // 8

    # res3.0 - first bottleneck with stride 2 and shortcut
    shapes["backbone.res3.0.conv1"] = (batch_size, res2_h, res2_w, 256)  # 1x1: 256 -> 128 (input at res2)
    shapes["backbone.res3.0.conv2"] = (
        batch_size,
        res2_h,
        res2_w,
        128,
    )  # 3x3 s2: 128 -> 128 (input at res2, stride2 -> res3)
    shapes["backbone.res3.0.conv3"] = (batch_size, res3_h, res3_w, 128)  # 1x1: 128 -> 512 (input at res3)
    shapes["backbone.res3.0.shortcut"] = (
        batch_size,
        res2_h,
        res2_w,
        256,
    )  # 1x1 s2: 256 -> 512 (input at res2, stride2 -> res3)

    # res3.1-3 - standard bottlenecks
    for i in range(1, 4):
        shapes[f"backbone.res3.{i}.conv1"] = (batch_size, res3_h, res3_w, 512)
        shapes[f"backbone.res3.{i}.conv2"] = (batch_size, res3_h, res3_w, 128)
        shapes[f"backbone.res3.{i}.conv3"] = (batch_size, res3_h, res3_w, 128)

    # === RES4 STAGE (stride 2, spatial reduction by 2x) ===
    # Input: H/8 x W/8 x 512 -> Output: H/16 x W/16 x 1024
    res4_h, res4_w = input_height // 16, input_width // 16

    # res4.0 - first bottleneck with stride 2 and shortcut
    shapes["backbone.res4.0.conv1"] = (batch_size, res3_h, res3_w, 512)  # 1x1: 512 -> 256 (input at res3)
    shapes["backbone.res4.0.conv2"] = (
        batch_size,
        res3_h,
        res3_w,
        256,
    )  # 3x3 s2: 256 -> 256 (input at res3, stride2 -> res4)
    shapes["backbone.res4.0.conv3"] = (batch_size, res4_h, res4_w, 256)  # 1x1: 256 -> 1024 (input at res4)
    shapes["backbone.res4.0.shortcut"] = (
        batch_size,
        res3_h,
        res3_w,
        512,
    )  # 1x1 s2: 512 -> 1024 (input at res3, stride2 -> res4)

    # res4.1-5 - standard bottlenecks (ResNet50 has 6 blocks in res4)
    for i in range(1, 6):
        shapes[f"backbone.res4.{i}.conv1"] = (batch_size, res4_h, res4_w, 1024)
        shapes[f"backbone.res4.{i}.conv2"] = (batch_size, res4_h, res4_w, 256)
        shapes[f"backbone.res4.{i}.conv3"] = (batch_size, res4_h, res4_w, 256)

    # === RES5 STAGE (stride 1 with dilation 2, no spatial reduction) ===
    # Input: H/16 x W/16 x 1024 -> Output: H/16 x W/16 x 2048
    res5_h, res5_w = input_height // 16, input_width // 16

    # res5.0 - first bottleneck with shortcut
    shapes["backbone.res5.0.conv1"] = (batch_size, res4_h, res4_w, 1024)
    shapes["backbone.res5.0.conv2"] = (batch_size, res5_h, res5_w, 512)
    shapes["backbone.res5.0.conv3"] = (batch_size, res5_h, res5_w, 512)
    shapes["backbone.res5.0.shortcut"] = (batch_size, res4_h, res4_w, 1024)

    # res5.1-2 - standard bottlenecks
    for i in range(1, 3):
        shapes[f"backbone.res5.{i}.conv1"] = (batch_size, res5_h, res5_w, 2048)
        shapes[f"backbone.res5.{i}.conv2"] = (batch_size, res5_h, res5_w, 512)
        shapes[f"backbone.res5.{i}.conv3"] = (batch_size, res5_h, res5_w, 512)

    # === ASPP (processes res5 output) ===
    # Input: H/16 x W/16 x 2048
    aspp_h, aspp_w = res5_h, res5_w
    aspp_in_channels = 2048

    # 5 parallel branches (4 convs + 1 global pooling)
    for i in range(5):
        shapes[f"aspp.convs.{i}"] = (batch_size, aspp_h, aspp_w, aspp_in_channels)

    # Project layer (concatenates all branches: 5 * 256 = 1280)
    shapes["aspp.project"] = (batch_size, aspp_h, aspp_w, 1280)

    # === DECODER ===
    # Decoder progressively upsamples from res5 -> res3 -> res2
    # Note: Decoder layers are duplicated for semantic_head and instance_head

    for head_prefix in ["semantic_head", "instance_head"]:
        # res5 uses ASPP, not simple project_conv
        # ASPP conv paths are handled separately above

        # Decoder res3 (processes res3 features)
        shapes[f"{head_prefix}.decoder.res3.project_conv"] = (batch_size, res3_h, res3_w, 512)
        shapes[f"{head_prefix}.decoder.res3.fuse_conv.0"] = (
            batch_size,
            res3_h,
            res3_w,
            256 + 32,
        )  # ASPP output + projected
        shapes[f"{head_prefix}.decoder.res3.fuse_conv.1"] = (batch_size, res3_h, res3_w, 128)

        # Decoder res2 (processes res2 features, upsampled to match res2 size)
        shapes[f"{head_prefix}.decoder.res2.project_conv"] = (batch_size, res2_h, res2_w, 256)
        shapes[f"{head_prefix}.decoder.res2.fuse_conv.0"] = (
            batch_size,
            res2_h,
            res2_w,
            128 + 32,
        )  # decoder output + projected
        shapes[f"{head_prefix}.decoder.res2.fuse_conv.1"] = (batch_size, res2_h, res2_w, 128)

        # res5 ASPP convs - each head has its own ASPP
        for i in range(5):
            if i < 4:
                shapes[f"{head_prefix}.decoder.res5.project_conv.convs.{i}"] = (batch_size, res5_h, res5_w, 2048)
            else:
                # convs.4 is pooling branch with conv
                shapes[f"{head_prefix}.decoder.res5.project_conv.convs.4.1"] = (batch_size, 1, 1, 2048)
        shapes[f"{head_prefix}.decoder.res5.project_conv.project"] = (batch_size, res5_h, res5_w, 1280)

    # === SEMANTIC SEGMENTATION HEAD ===
    # Input: decoder output at H/4 x W/4 (res2 resolution)
    # Final output: H x W (upsampled 4x)
    sem_h, sem_w = res2_h, res2_w  # Fixed: decoder outputs at res2, not res3!
    shapes["semantic_head.head.0"] = (batch_size, sem_h, sem_w, 256)
    shapes["semantic_head.head.1"] = (batch_size, sem_h, sem_w, 256)
    shapes["semantic_head.predictor"] = (batch_size, sem_h, sem_w, 256)

    # === INSTANCE EMBEDDING HEAD ===
    # Center head - predicts instance centers
    shapes["instance_head.center_head.0"] = (batch_size, sem_h, sem_w, 256)
    shapes["instance_head.center_head.1"] = (batch_size, sem_h, sem_w, 32)
    shapes["instance_head.center_predictor"] = (batch_size, sem_h, sem_w, 32)

    # Offset head - predicts instance offsets
    shapes["instance_head.offset_head.0"] = (batch_size, sem_h, sem_w, 256)
    shapes["instance_head.offset_head.1"] = (batch_size, sem_h, sem_w, 32)
    shapes["instance_head.offset_predictor"] = (batch_size, sem_h, sem_w, 32)

    logger.debug(f"Computed {len(shapes)} layer input shapes for input size {input_height}x{input_width}")

    return shapes


def custom_preprocessor(model, name, input_shapes=None):
    """
    Custom preprocessor for all Panoptic-DeepLab model components using TT CNN Builder API.

    This function handles weight preprocessing for:
    - Conv2d wrapper layers (with optional normalization)
    - Pure BatchNorm/SyncBatchNorm layers
    - Standard PyTorch Conv2d layers (for ResNet backbone)
    - MaxPool2d layers
    - Upsample layers

    For convolutional layers, it extracts Conv2dConfiguration using the TT CNN Builder's from_torch method.

    Args:
        model: PyTorch model/layer to preprocess
        name: Name of the layer in the model hierarchy
        input_shapes: Optional dict mapping layer names to input shapes (batch_size, height, width, channels)
                     Required for Conv2d, MaxPool2d, and Upsample layers

    Returns:
        Dict containing preprocessed parameters and configurations
    """
    parameters = {}

    # Handle Conv2d wrapper layers (used in heads)
    if isinstance(model, Conv2d):
        # Extract Conv2dConfiguration using TT CNN Builder API
        if input_shapes and name in input_shapes:
            batch_size, height, width, channels = input_shapes[name]

            # Create Conv2dConfiguration from torch layer
            conv_config = Conv2dConfiguration.from_torch(
                torch_layer=model,
                input_height=height,
                input_width=width,
                batch_size=batch_size,
            )

            # Pad predictor layers to tile-aligned output channels (32)
            # TTNN conv2d requires output channels to be multiples of 32
            original_out_channels = conv_config.out_channels
            is_predictor = "predictor" in name.split(".")[-1]  # Check if last component contains "predictor"

            if is_predictor:
                logger.debug(
                    f"[PADDING DEBUG] {name}: is_predictor=True, original_out_channels={original_out_channels}"
                )

            if is_predictor and original_out_channels < 32:
                import torch
                from dataclasses import replace

                # Calculate padding needed to reach 32 channels
                padded_out_channels = 32
                padding_needed = padded_out_channels - original_out_channels

                logger.info(
                    f"Padding predictor layer '{name}' from {original_out_channels} to {padded_out_channels} output channels"
                )

                # Pad weights: [out_channels, in_channels, kernel_h, kernel_w]
                # Convert TTNN tensor back to torch, pad, then convert back
                # Use ROW_MAJOR layout for weights (conv2d requires ROW_MAJOR for host weights)
                weight_torch = ttnn.to_torch(conv_config.weight)
                weight_padded = torch.nn.functional.pad(weight_torch, (0, 0, 0, 0, 0, 0, 0, padding_needed), value=0.0)
                weight_padded_ttnn = ttnn.from_torch(weight_padded, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

                # Pad bias if present: [1, 1, 1, out_channels]
                bias_padded_ttnn = None
                if conv_config.bias is not None:
                    bias_torch = ttnn.to_torch(conv_config.bias)
                    bias_padded = torch.nn.functional.pad(bias_torch, (0, padding_needed), value=0.0)
                    bias_padded_ttnn = ttnn.from_torch(bias_padded, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

                # Create new config with padded weights (Conv2dConfiguration is frozen)
                conv_config = replace(
                    conv_config,
                    out_channels=padded_out_channels,
                    weight=weight_padded_ttnn,
                    bias=bias_padded_ttnn if bias_padded_ttnn is not None else conv_config.bias,
                )

                # Store original output channels for slicing after forward pass
                parameters["original_out_channels"] = original_out_channels

            parameters["conv_config"] = conv_config

            # Extract weight and bias from Conv2dConfiguration for fusion
            # Conv2dConfiguration stores them as TTNN tensors
            parameters["weight"] = conv_config.weight
            parameters["bias"] = conv_config.bias

            logger.debug(
                f"Extracted Conv2dConfiguration for {name}: in={conv_config.in_channels}, out={conv_config.out_channels}, kernel={conv_config.kernel_size}"
            )
        else:
            # Fallback: Store weights and bias as before if shape information is not available
            parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
            if model.bias is not None:
                bias = model.bias.reshape((1, 1, 1, -1))
                parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
            logger.warning(f"No input shape provided for Conv2d layer '{name}', using fallback weight extraction")

        # Handle normalization if present
        if hasattr(model, "norm") and model.norm is not None:
            norm_params = {}
            if isinstance(model.norm, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                # Reshape 1D BatchNorm parameters to [1, C, 1, 1] for TTNN batch_norm
                norm_params["weight"] = ttnn.from_torch(
                    model.norm.weight.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["bias"] = ttnn.from_torch(
                    model.norm.bias.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["running_mean"] = ttnn.from_torch(
                    model.norm.running_mean.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["running_var"] = ttnn.from_torch(
                    model.norm.running_var.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
            elif isinstance(model.norm, nn.LayerNorm):
                norm_params["weight"] = ttnn.from_torch(model.norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                norm_params["bias"] = ttnn.from_torch(model.norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            parameters["norm"] = norm_params

    # Handle pure normalization layers
    elif isinstance(model, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        # Reshape 1D BatchNorm parameters to [1, C, 1, 1] for TTNN batch_norm
        parameters["weight"] = ttnn.from_torch(
            model.weight.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        parameters["bias"] = ttnn.from_torch(
            model.bias.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        parameters["running_mean"] = ttnn.from_torch(
            model.running_mean.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        parameters["running_var"] = ttnn.from_torch(
            model.running_var.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    # Handle standard PyTorch Conv2d layers (used in ResNet backbone)
    elif isinstance(model, nn.Conv2d):
        if ".predictor" in name:
            logger.debug(f"[PATH DEBUG nn.Conv2d] {name}: Processing in nn.Conv2d path")
        # Extract Conv2dConfiguration using TT CNN Builder API
        if input_shapes and name in input_shapes:
            batch_size, height, width, channels = input_shapes[name]

            # Create Conv2dConfiguration from torch layer
            conv_config = Conv2dConfiguration.from_torch(
                torch_layer=model,
                input_height=height,
                input_width=width,
                batch_size=batch_size,
            )

            # Pad predictor layers to tile-aligned output channels (32)
            # TTNN conv2d requires output channels to be multiples of 32
            original_out_channels = conv_config.out_channels
            is_predictor = "predictor" in name.split(".")[-1]  # Check if last component contains "predictor"

            if is_predictor:
                logger.debug(
                    f"[PADDING DEBUG] {name}: is_predictor=True, original_out_channels={original_out_channels}"
                )

            if is_predictor and original_out_channels < 32:
                import torch
                from dataclasses import replace

                # Calculate padding needed to reach 32 channels
                padded_out_channels = 32
                padding_needed = padded_out_channels - original_out_channels

                logger.info(
                    f"Padding predictor layer '{name}' from {original_out_channels} to {padded_out_channels} output channels"
                )

                # Pad weights: [out_channels, in_channels, kernel_h, kernel_w]
                # Convert TTNN tensor back to torch, pad, then convert back
                # Use ROW_MAJOR layout for weights (conv2d requires ROW_MAJOR for host weights)
                weight_torch = ttnn.to_torch(conv_config.weight)
                weight_padded = torch.nn.functional.pad(weight_torch, (0, 0, 0, 0, 0, 0, 0, padding_needed), value=0.0)
                weight_padded_ttnn = ttnn.from_torch(weight_padded, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

                # Pad bias if present: [1, 1, 1, out_channels]
                bias_padded_ttnn = None
                if conv_config.bias is not None:
                    bias_torch = ttnn.to_torch(conv_config.bias)
                    bias_padded = torch.nn.functional.pad(bias_torch, (0, padding_needed), value=0.0)
                    bias_padded_ttnn = ttnn.from_torch(bias_padded, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

                # Create new config with padded weights (Conv2dConfiguration is frozen)
                conv_config = replace(
                    conv_config,
                    out_channels=padded_out_channels,
                    weight=weight_padded_ttnn,
                    bias=bias_padded_ttnn if bias_padded_ttnn is not None else conv_config.bias,
                )

                # Store original output channels for slicing after forward pass
                parameters["original_out_channels"] = original_out_channels

            parameters["conv_config"] = conv_config

            # Extract weight and bias from Conv2dConfiguration for fusion
            # Conv2dConfiguration stores them as TTNN tensors
            parameters["weight"] = conv_config.weight
            parameters["bias"] = conv_config.bias

            logger.debug(
                f"Extracted Conv2dConfiguration for {name}: in={conv_config.in_channels}, out={conv_config.out_channels}, kernel={conv_config.kernel_size}"
            )
        else:
            # Fallback: Store weights and bias as before if shape information is not available
            parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
            if model.bias is not None:
                bias = model.bias.reshape((1, 1, 1, -1))
                parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
            logger.warning(f"No input shape provided for Conv2d layer '{name}', using fallback weight extraction")

        # Handle attached normalization (ResNet style: conv.norm)
        if hasattr(model, "norm") and model.norm is not None:
            norm_params = {}
            if isinstance(model.norm, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                # Reshape 1D BatchNorm parameters to [1, C, 1, 1] for TTNN batch_norm
                norm_params["weight"] = ttnn.from_torch(
                    model.norm.weight.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["bias"] = ttnn.from_torch(
                    model.norm.bias.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["running_mean"] = ttnn.from_torch(
                    model.norm.running_mean.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["running_var"] = ttnn.from_torch(
                    model.norm.running_var.reshape(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
            parameters["norm"] = norm_params

    # Handle MaxPool2d layers
    elif isinstance(model, nn.MaxPool2d):
        if input_shapes and name in input_shapes:
            batch_size, height, width, channels = input_shapes[name]

            # Create MaxPool2dConfiguration from torch layer
            pool_config = MaxPool2dConfiguration.from_torch(
                torch_layer=model,
                input_height=height,
                input_width=width,
                batch_size=batch_size,
                channels=channels,
            )
            parameters["pool_config"] = pool_config
            logger.debug(
                f"Extracted MaxPool2dConfiguration for {name}: channels={pool_config.channels}, kernel={pool_config.kernel_size}"
            )
        else:
            logger.warning(f"No input shape provided for MaxPool2d layer '{name}', skipping configuration extraction")

    # Handle Upsample layers
    elif isinstance(model, nn.Upsample):
        if input_shapes and name in input_shapes:
            batch_size, height, width, channels = input_shapes[name]

            # Create UpsampleConfiguration from torch layer
            upsample_config = UpsampleConfiguration.from_torch(
                torch_layer=model,
                input_height=height,
                input_width=width,
                batch_size=batch_size,
                channels=channels,
            )
            parameters["upsample_config"] = upsample_config
            logger.debug(
                f"Extracted UpsampleConfiguration for {name}: channels={upsample_config.channels}, scale={upsample_config.scale_factor}"
            )
        else:
            logger.warning(f"No input shape provided for Upsample layer '{name}', skipping configuration extraction")

    return parameters


def create_panoptic_deeplab_parameters(
    model: PytorchPanopticDeepLab,
    device,
    input_shapes=None,
    input_height: int = 512,
    input_width: int = 1024,
    batch_size: int = 1,
):
    """
    Create preprocessed parameters for the complete Panoptic-DeepLab model.

    This function uses ttnn.preprocess_model_parameters to automatically traverse
    the entire PyTorch model (including ResNet backbone) and convert all weights
    to TTNN tensors using the custom preprocessor. It also extracts Conv2dConfiguration
    objects using the TT CNN Builder API for each convolutional layer.

    Args:
        model: PyTorch Panoptic-DeepLab model with loaded weights
        device: TTNN device for tensor placement
        input_shapes: Optional dict mapping layer names to input shapes (batch_size, height, width, channels).
                     If not provided, will be computed automatically using compute_panoptic_deeplab_input_shapes()
                     based on input_height, input_width, and batch_size.
        input_height: Input image height (default: 512). Used to compute input_shapes if not provided.
        input_width: Input image width (default: 1024). Used to compute input_shapes if not provided.
        batch_size: Batch size (default: 1). Used to compute input_shapes if not provided.

    Returns:
        Preprocessed parameters suitable for TTNN model initialization, including Conv2dConfiguration objects
    """
    from loguru import logger

    logger.info("Starting unified weight preprocessing for complete Panoptic-DeepLab model")

    # Compute input shapes if not provided
    if input_shapes is None:
        logger.info(
            f"Computing input shapes automatically for image size {input_height}x{input_width}, batch_size={batch_size}"
        )
        input_shapes = compute_panoptic_deeplab_input_shapes(
            input_height=input_height, input_width=input_width, batch_size=batch_size
        )
    else:
        logger.info(f"Using provided input shapes dictionary with {len(input_shapes)} entries")

    def model_initializer():
        return model

    # Create a wrapper for custom_preprocessor that passes input_shapes
    def preprocessor_with_shapes(model, name):
        return custom_preprocessor(model, name, input_shapes=input_shapes)

    # Use preprocess_model_parameters to handle the entire model uniformly
    parameters = preprocess_model_parameters(
        initialize_model=model_initializer,
        custom_preprocessor=preprocessor_with_shapes,
        device=None,
    )

    logger.info("Unified weight preprocessing completed successfully")
    logger.debug(f"Generated parameter structure: {list(parameters.keys())}")

    return parameters
