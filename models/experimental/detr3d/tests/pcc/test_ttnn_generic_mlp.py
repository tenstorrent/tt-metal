# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.experimental.detr3d.ttnn.ttnn_generic_mlp import TttnnGenericMLP
from models.experimental.detr3d.reference.detr3d_model import GenericMLP, Conv1d
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import convert_torch_model_to_ttnn_model


def fold_batch_norm1d_into_conv1d(conv, bn):
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm1d must have track_running_stats=True to be folded into Conv1d")

    weight = conv.weight  # Shape: [out_channels, in_channels, kernel_size]
    bias = conv.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    scale = bn.weight
    shift = bn.bias

    # For 1D: scale factor applied per output channel
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None]

    if bias is not None:
        bias = (bias - running_mean) * (scale / torch.sqrt(running_var + eps)) + shift
    else:
        bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))

    # For 1D convolutions, bias shape should be [1, 1, -1] instead of [1, 1, 1, -1]
    bias = bias.reshape(1, 1, 1, -1)

    return weight, bias


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    parameters = {}
    if isinstance(model, Conv1d):
        if model.norm is not None:
            weight, bias = fold_batch_norm1d_into_conv1d(model, model.norm)
            parameters["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
            parameters["bias"] = ttnn.from_torch(bias, mesh_mapper=mesh_mapper)
        else:
            weight = model.weight
            if model.bias is not None:
                bias = model.bias
                if bias.dim() < 4:
                    bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                parameters["bias"] = ttnn.from_torch(bias, mesh_mapper=mesh_mapper)
            parameters["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
    # if isinstance(model, torch.nn.Conv1d):
    #     weight = model.weight
    #     if model.bias is not None:
    #         bias = model.bias
    #         if bias.dim() < 4:
    #             bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    #         parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.float32)
    #     parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.float32)
    else:
        # Let the sub-modules handle their own preprocessing
        for child_name, child in model.named_children():
            parameters[child_name] = convert_torch_model_to_ttnn_model(
                child,
                name=f"{name}.{child_name}",
                custom_preprocessor=custom_preprocessor_func,
                convert_to_ttnn=convert_to_ttnn,
                ttnn_module_args=ttnn_module_args,
            )

    return parameters


def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
    return custom_preprocessor(model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, None)


@pytest.mark.parametrize(
    "input_dim,hidden_dims,output_dim,norm_fn_name,activation,use_conv,"
    "hidden_use_bias,output_use_bias,output_use_activation,output_use_norm,"
    "weight_init_name,dropout,x_shape",
    [
        (256, [256], 256, "bn1d", "relu", True, False, False, True, True, None, None, (1, 256, 1024)),
        (256, [256], 256, None, "relu", True, True, True, True, False, None, None, (1, 256, 128)),
        (256, [256, 256], 12, "bn1d", "relu", True, False, True, False, False, None, None, (8, 256, 128)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_generic_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    norm_fn_name,
    activation,
    use_conv,
    hidden_use_bias,
    output_use_bias,
    output_use_activation,
    output_use_norm,
    weight_init_name,
    dropout,
    x_shape,
    device,
):
    torch_model = GenericMLP(
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name,
        activation,
        use_conv,
        dropout,
        hidden_use_bias,
        output_use_bias,
        output_use_activation,
        output_use_norm,
        weight_init_name,
    ).to(torch.bfloat16)
    torch_model.eval()
    x = torch.randn(x_shape, dtype=torch.bfloat16)
    torch_out = torch_model(x)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_mesh_preprocessor,
        device=device,
    )
    print("param are", parameters)
    ttnn_model = TttnnGenericMLP(torch_model, parameters, device)
    ttnn_x = ttnn.from_torch(
        x.permute(0, 2, 1).unsqueeze(dim=0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    p(ttnn_x, "ttnn input")
    ttnn_out = ttnn_model(ttnn_x)
    print("outputs are", ttnn_out.shape, torch_out.shape)
    ttnn_out = ttnn.to_torch(ttnn_out)
    if ttnn_out.shape[-1] == 12:
        ttnn_out = ttnn_out.reshape(1, 8, 128, 12).permute(0, 1, 3, 2).squeeze(0)
    else:
        ttnn_out = ttnn_out.squeeze(dim=0).permute(0, 2, 1)
    assert_with_pcc(torch_out, ttnn_out, 0.999)
