# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

from models.experimental.retinanet.tt.utils import group_norm_channel_chunks


def conv_bn_to_params(conv, bn, mesh_mapper):
    if bn is None:
        weight = conv.weight.detach().clone().contiguous()
        bias = conv.bias.detach().clone().contiguous() if conv.bias is not None else torch.zeros(conv.out_channels)
    else:
        weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)

    return {
        "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
    }


def create_custom_mesh_preprocessor(mesh_mapper):
    """Return a custom preprocessor closure with mesh_mapper captured."""

    def custom_preprocessor(model, name, *, ttnn_module_args=None, convert_to_ttnn=True):
        parameters = {}

        if isinstance(model, torch.nn.Conv2d) or isinstance(model, torch.nn.BatchNorm2d):
            return {}

        elif isinstance(model, torch.nn.Module):
            children = list(model.named_children())
            i = 0
            while i < len(children):
                child_name, child = children[i]

                # Detect Conv + BN pair
                if isinstance(child, torch.nn.Conv2d):
                    next_bn = None
                    if i + 1 < len(children):
                        next_name, next_child = children[i + 1]
                        if isinstance(next_child, torch.nn.BatchNorm2d):
                            next_bn = next_child
                            i += 1  # skip BN

                    params = conv_bn_to_params(child, next_bn, mesh_mapper)
                    parameters[child_name] = params

                else:
                    # Recurse
                    subparams = custom_preprocessor(
                        child,
                        f"{name}.{child_name}" if name else child_name,
                        ttnn_module_args=ttnn_module_args,
                        convert_to_ttnn=convert_to_ttnn,
                    )
                    if subparams:
                        parameters[child_name] = subparams

                i += 1

        return parameters

    return custom_preprocessor


def head_conv_blocks(torch_head, device, mesh_mapper, model_config):
    """The parameters of a head's conv blocks: each conv's host weights, which conv2d prepares for
    each pyramid level on its first run there (conv2d_with_level_weights), and its GroupNorm weight
    and bias, laid out for the GroupNorm grid of ``device`` (group_norm_channel_chunks)."""
    channel_chunks = group_norm_channel_chunks(device, num_channels=256, num_groups=32)
    layout = (
        ttnn.TILE_LAYOUT if model_config["WEIGHTS_DTYPE"] in [ttnn.bfloat8_b, ttnn.bfloat4_b] else ttnn.ROW_MAJOR_LAYOUT
    )
    blocks = []
    for conv, norm, _ in torch_head.conv:
        params = conv_bn_to_params(conv, None, mesh_mapper)
        for name, value in (("norm_weight", norm.weight), ("norm_bias", norm.bias)):
            formatted = ttnn.create_group_norm_weight_bias_rm(
                value.detach(), num_channels=256, num_cores_x=channel_chunks
            )
            params[name] = ttnn.from_torch(
                formatted,
                dtype=model_config["WEIGHTS_DTYPE"],
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
        blocks.append(params)
    return blocks


def preprocess_regression_head_parameters(torch_head, device, mesh_mapper, model_config):
    """The regression head's parameters for ``device``; its GroupNorm weights are laid out for the
    device's grid, so run the head on the same device (head_conv_blocks)."""
    return {
        "conv": head_conv_blocks(torch_head, device, mesh_mapper, model_config),
        "bbox_reg": conv_bn_to_params(torch_head.bbox_reg, None, mesh_mapper),
    }


def preprocess_classification_head_parameters(torch_head, device, mesh_mapper, model_config):
    """The classification head's parameters for ``device``; its GroupNorm weights are laid out for
    the device's grid, so run the head on the same device (head_conv_blocks)."""
    return {
        "conv": head_conv_blocks(torch_head, device, mesh_mapper, model_config),
        "cls_logits": conv_bn_to_params(torch_head.cls_logits, None, mesh_mapper),
    }
