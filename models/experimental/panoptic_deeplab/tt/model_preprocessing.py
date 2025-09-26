# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters
from loguru import logger

from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_conv2d_wrapper import Conv2d


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

                    # Store only the fused conv parameters (now TTNN tensors)
                    fused_params[key] = {"weight": fused_weight, "bias": fused_bias}
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

                        # Store only the fused conv parameters
                        fused_list.append({"weight": fused_weight, "bias": fused_bias})
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


def custom_preprocessor(model, name):
    """
    Custom preprocessor for all Panoptic-DeepLab model components including ResNet backbone.

    This function handles weight preprocessing for:
    - Conv2d wrapper layers (with optional normalization)
    - Pure BatchNorm/SyncBatchNorm layers
    - Standard PyTorch Conv2d layers (for ResNet backbone)
    - BottleneckBlock and StemBlock (ResNet components)
    """
    parameters = {}

    # Handle Conv2d wrapper layers (used in heads)
    if isinstance(model, Conv2d):
        # Convolutional part
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)

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
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)

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

    return parameters


def create_panoptic_deeplab_parameters(model: PytorchPanopticDeepLab, device):
    """
    Create preprocessed parameters for the complete Panoptic-DeepLab model.

    This function uses ttnn.preprocess_model_parameters to automatically traverse
    the entire PyTorch model (including ResNet backbone) and convert all weights
    to TTNN tensors using the custom preprocessor.

    Args:
        model: PyTorch Panoptic-DeepLab model with loaded weights
        device: TTNN device for tensor placement

    Returns:
        Preprocessed parameters suitable for TTNN model initialization
    """
    from loguru import logger

    logger.info("Starting unified weight preprocessing for complete Panoptic-DeepLab model")

    def model_initializer():
        return model

    # Use preprocess_model_parameters to handle the entire model uniformly
    parameters = preprocess_model_parameters(
        initialize_model=model_initializer,
        custom_preprocessor=custom_preprocessor,
        device=None,
    )

    logger.info("Unified weight preprocessing completed successfully")
    logger.debug(f"Generated parameter structure: {list(parameters.keys())}")

    return parameters
