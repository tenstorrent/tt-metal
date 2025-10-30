import torch
import ttnn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d


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
            # Skip here — handled in parent scope
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


def preprocess_regression_head_parameters(torch_head, device, mesh_mapper, model_config):
    """Convert PyTorch regression head weights to TTNN format."""
    parameters = {}

    # Grid size for GroupNorm
    grid_size = ttnn.CoreGrid(y=8, x=8)
    layout = (
        ttnn.TILE_LAYOUT if model_config["WEIGHTS_DTYPE"] in [ttnn.bfloat8_b, ttnn.bfloat4_b] else ttnn.ROW_MAJOR_LAYOUT
    )
    # Convert 4 conv layers (Conv2d + GroupNorm weights)
    parameters["conv"] = []
    for i in range(4):
        # Conv2d weights
        conv_weight = torch_head.conv[i][0].weight.detach().to(torch.bfloat16)  # Was: torch.bfloat16

        # GroupNorm weights - MUST use create_group_norm_weight_bias_rm()
        norm_weight = torch_head.conv[i][1].weight.detach()
        norm_bias = torch_head.conv[i][1].bias.detach()

        # Format GroupNorm parameters using helper function
        formatted_norm_weight = ttnn.create_group_norm_weight_bias_rm(
            norm_weight, num_channels=256, num_cores_x=grid_size.y
        )
        formatted_norm_bias = ttnn.create_group_norm_weight_bias_rm(
            norm_bias, num_channels=256, num_cores_x=grid_size.y
        )

        conv_params = {
            "weight": ttnn.from_torch(
                conv_weight, model_config["WEIGHTS_DTYPE"], device=device, mesh_mapper=mesh_mapper
            ),  # bfloat16
            "norm_weight": ttnn.from_torch(
                formatted_norm_weight,
                dtype=model_config["WEIGHTS_DTYPE"],
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            ),
            "norm_bias": ttnn.from_torch(
                formatted_norm_bias,
                dtype=model_config["WEIGHTS_DTYPE"],
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            ),
        }

        parameters["conv"].append(conv_params)

    # Convert bbox_reg layer
    bbox_weight = torch_head.bbox_reg.weight.detach().to(torch.bfloat16)
    bbox_bias = torch_head.bbox_reg.bias.detach().to(torch.bfloat16)

    parameters["bbox_reg"] = {
        "weight": ttnn.from_torch(
            bbox_weight, dtype=model_config["WEIGHTS_DTYPE"], device=device, mesh_mapper=mesh_mapper
        ),
        "bias": ttnn.from_torch(
            bbox_bias.reshape(1, 1, 1, -1), dtype=model_config["WEIGHTS_DTYPE"], device=device, mesh_mapper=mesh_mapper
        ),
    }

    return parameters


def preprocess_classification_head_parameters(torch_head, device, mesh_mapper, model_config):
    """Convert PyTorch classification head weights to TTNN format."""
    parameters = {}

    # Grid size for GroupNorm
    grid_size = ttnn.CoreGrid(y=8, x=8)
    layout = (
        ttnn.TILE_LAYOUT if model_config["WEIGHTS_DTYPE"] in [ttnn.bfloat8_b, ttnn.bfloat4_b] else ttnn.ROW_MAJOR_LAYOUT
    )
    # layout=ttnn.ROW_MAJOR_LAYOUT
    # Convert 4 conv layers (Conv2d + GroupNorm weights)
    parameters["conv"] = []
    for i in range(4):
        # Conv2d weights
        conv_weight = torch_head.conv[i][0].weight.detach().to(torch.bfloat16)

        # GroupNorm weights - format using helper function
        norm_weight = torch_head.conv[i][1].weight.detach()
        norm_bias = torch_head.conv[i][1].bias.detach()

        # Format GroupNorm parameters using helper function
        formatted_norm_weight = ttnn.create_group_norm_weight_bias_rm(
            norm_weight, num_channels=256, num_cores_x=grid_size.y
        )
        formatted_norm_bias = ttnn.create_group_norm_weight_bias_rm(
            norm_bias, num_channels=256, num_cores_x=grid_size.y
        )

        conv_params = {
            "weight": ttnn.from_torch(
                conv_weight, dtype=model_config["WEIGHTS_DTYPE"], device=device, mesh_mapper=mesh_mapper
            ),
            "norm_weight": ttnn.from_torch(
                formatted_norm_weight,
                dtype=model_config["WEIGHTS_DTYPE"],
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            ),
            "norm_bias": ttnn.from_torch(
                formatted_norm_bias,
                dtype=model_config["WEIGHTS_DTYPE"],
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            ),
        }

        parameters["conv"].append(conv_params)

    # Convert cls_logits layer
    cls_logits_weight = torch_head.cls_logits.weight.detach().to(torch.bfloat16)
    cls_logits_bias = torch_head.cls_logits.bias.detach().to(torch.bfloat16)

    # Reshape bias to (1, 1, 1, 819)
    cls_logits_bias = cls_logits_bias.reshape(1, 1, 1, -1)

    parameters["cls_logits"] = {
        "weight": ttnn.from_torch(
            cls_logits_weight, dtype=model_config["WEIGHTS_DTYPE"], device=device, mesh_mapper=mesh_mapper
        ),
        "bias": ttnn.from_torch(
            cls_logits_bias, dtype=model_config["WEIGHTS_DTYPE"], device=device, mesh_mapper=mesh_mapper
        ),
    }

    return parameters
