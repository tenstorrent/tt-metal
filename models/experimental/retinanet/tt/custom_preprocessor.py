# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d


def conv_bn_gn_to_params(conv, bn, gn, mesh_mapper):
    if bn is None and gn is None:
        weight = conv.weight.detach().clone().contiguous()
        bias = conv.bias.detach().clone().contiguous() if conv.bias is not None else torch.zeros(conv.out_channels)

        return {
            "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
            "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        }
    elif bn is not None:
        weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)
        return {
            "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
            "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        }

    elif gn is not None:
        weight = conv.weight.detach().clone().contiguous()
        bias = conv.bias.detach().clone().contiguous() if conv.bias is not None else torch.zeros(conv.out_channels)
        norm_weight = gn.weight.detach()
        norm_bias = gn.bias.detach()
        grid_size = ttnn.CoreGrid(y=8, x=8)
        formatted_norm_weight = ttnn.create_group_norm_weight_bias_rm(
            norm_weight, num_channels=256, num_cores_x=grid_size.y
        )
        formatted_norm_bias = ttnn.create_group_norm_weight_bias_rm(
            norm_bias, num_channels=256, num_cores_x=grid_size.y
        )

        return {
            "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
            "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
            "norm_weight": ttnn.from_torch(
                formatted_norm_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "norm_bias": ttnn.from_torch(
                formatted_norm_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
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
                    next_gn = None
                    if i + 1 < len(children):
                        next_name, next_child = children[i + 1]
                        if isinstance(next_child, torch.nn.BatchNorm2d):
                            next_bn = next_child
                            i += 1  # skip BN
                        if isinstance(next_child, torch.nn.GroupNorm):
                            next_gn = next_child

                    params = conv_bn_gn_to_params(child, next_bn, next_gn, mesh_mapper)
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
