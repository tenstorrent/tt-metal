# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for MonoDiffusion model
Following vanilla_unet pattern for consistency
"""

import os
import torch
import ttnn
from typing import Dict

# Performance constants
MONODIFFUSION_PCC_TARGET = 0.99


def load_reference_model(model_location_generator=None):
    """
    Load PyTorch reference model for validation
    Following vanilla_unet pattern
    """
    if model_location_generator is None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = "models/demos/monodiffusion/weights/monodiffusion_kitti.pth"
        if not os.path.exists(weights_path):
            os.system("bash models/demos/monodiffusion/weights_download.sh")
    else:
        weights_path = (
            model_location_generator("vision-models/monodiffusion", model_subdir="", download_if_ci_v2=True)
            / "monodiffusion_kitti.pth"
        )

    from models.demos.monodiffusion.reference.pytorch_model import create_reference_model

    reference_model = create_reference_model(num_inference_steps=20)

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        reference_model.load_state_dict(state_dict, strict=False)

    reference_model.eval()
    return reference_model


def create_monodiffusion_preprocessor(device, mesh_mapper=None):
    """
    Create preprocessor for converting PyTorch weights to TTNN tensors
    Following vanilla_unet pattern
    """
    assert (
        device.get_num_devices() == 1 or mesh_mapper is not None
    ), "Expected a mesh mapper for weight tensors if using multiple devices"

    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}

        # Encoder parameters
        parameters["encoder"] = {}
        for i, layer_name in enumerate(["conv1", "conv2", "conv3", "conv4"]):
            parameters["encoder"][layer_name] = {}

            # Get conv layer from model
            conv_layer = getattr(model.encoder, f"layer{i+1}", None)
            if conv_layer is not None:
                # Fold batch norm into conv if present
                if hasattr(conv_layer, "bn"):
                    from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                        conv_layer.conv, conv_layer.bn
                    )
                else:
                    conv_weight = conv_layer.weight
                    conv_bias = conv_layer.bias if hasattr(conv_layer, "bias") else None

                parameters["encoder"][layer_name]["weight"] = ttnn.from_torch(
                    conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )
                if conv_bias is not None:
                    parameters["encoder"][layer_name]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                        mesh_mapper=mesh_mapper
                    )

        # Diffusion U-Net parameters
        parameters["unet"] = {}
        unet_layers = ["down1_conv1", "down1_conv2", "mid_conv1", "mid_conv2", "up1_conv1", "up1_conv2"]
        for layer_name in unet_layers:
            parameters["unet"][layer_name] = {}
            # Placeholder - will be populated from actual model
            # parameters["unet"][layer_name]["weight"] = ...
            # parameters["unet"][layer_name]["bias"] = ...

        # Decoder parameters
        parameters["decoder"] = {}
        for i, layer_name in enumerate(["conv1", "conv2", "conv3", "conv4", "final"]):
            parameters["decoder"][layer_name] = {}
            # Placeholder - will be populated from actual model

        # Uncertainty head parameters
        parameters["uncertainty"] = {}
        for layer_name in ["conv1", "conv2"]:
            parameters["uncertainty"][layer_name] = {}
            # Placeholder - will be populated from actual model

        return parameters

    return custom_preprocessor


def concatenate_skip_connection(
    upsampled: ttnn.Tensor,
    skip: ttnn.Tensor,
    use_row_major_layout: bool = True
) -> ttnn.Tensor:
    """
    Concatenate upsampled tensor with skip connection along channel dimension
    Directly from vanilla_unet implementation
    """
    assert upsampled.shape[0:3] == skip.shape[0:3], \
        f"Spatial dimensions must match: {upsampled.shape} vs {skip.shape}"

    # Reshard skip connection to match upsampled tensor's memory config
    if not skip.is_sharded():
        input_core_grid = upsampled.memory_config().shard_spec.grid
        input_shard_shape = upsampled.memory_config().shard_spec.shape
        input_shard_spec = ttnn.ShardSpec(
            input_core_grid,
            input_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR
        )
        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            input_shard_spec
        )
        skip = ttnn.to_memory_config(skip, input_memory_config)

    # Calculate output shard shape (doubled channels)
    output_core_grid = upsampled.memory_config().shard_spec.grid
    output_shard_shape = (
        upsampled.memory_config().shard_spec.shape[0],
        upsampled.memory_config().shard_spec.shape[1] * 2,
    )
    output_shard_spec = ttnn.ShardSpec(
        output_core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR
    )
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec
    )

    if use_row_major_layout:
        upsampled_rm = ttnn.to_layout(upsampled, ttnn.ROW_MAJOR_LAYOUT)
        skip_rm = ttnn.to_layout(skip, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(upsampled)
        ttnn.deallocate(skip)

        concatenated = ttnn.concat(
            [upsampled_rm, skip_rm],
            dim=3,
            memory_config=output_memory_config
        )
        ttnn.deallocate(upsampled_rm)
        ttnn.deallocate(skip_rm)

        concat_tiled = ttnn.to_layout(concatenated, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(concatenated)
        return concat_tiled
    else:
        concatenated = ttnn.concat(
            [upsampled, skip],
            dim=3,
            memory_config=output_memory_config
        )
        ttnn.deallocate(upsampled)
        ttnn.deallocate(skip)
        return concatenated


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute Pearson Correlation Coefficient between two tensors
    Used for accuracy validation against PyTorch reference
    """
    tensor1_flat = tensor1.flatten().float()
    tensor2_flat = tensor2.flatten().float()

    mean1 = tensor1_flat.mean()
    mean2 = tensor2_flat.mean()

    numerator = ((tensor1_flat - mean1) * (tensor2_flat - mean2)).sum()
    denominator = torch.sqrt(
        ((tensor1_flat - mean1) ** 2).sum() * ((tensor2_flat - mean2) ** 2).sum()
    )

    pcc = numerator / (denominator + 1e-8)
    return pcc.item()
