import ttnn
import torch
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args

# from models.experimental.oft.reference.oftnet import OftNet


def preprocess_linear_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=layout)
    return weight


def preprocess_linear_bias(bias, *, dtype, layout=ttnn.TILE_LAYOUT):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=layout)
    return bias


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
    if isinstance(model, torch.nn.Linear):
        parameters[f"weight"] = preprocess_linear_weight(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            parameters[f"bias"] = preprocess_linear_bias(model.bias, dtype=ttnn.bfloat16)
    if isinstance(model, nn.GroupNorm):
        parameters["weight"] = model.weight
        if model.bias is not None:
            parameters["bias"] = model.bias

    return parameters


# def create_OFT_model_parameters_resnet(model: OftNet, input_tensor: torch.Tensor, device):
def create_OFT_model_parameters_resnet(model, input_tensor: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=None,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    parameters["model_args"] = model

    return parameters


# def create_OFT_model_parameters_oft(
#     model: OftNet, input_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device
# ):
def create_OFT_model_parameters_oft(model, input_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    input1, input2, input3 = input_tensors
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(input1, input2, input3),
        device=None,
    )

    parameters["model_args"] = model

    return parameters


# def create_OFT_model_parameters(model: OftNet, input_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device):
#     parameters = preprocess_model_parameters(
#         initialize_model=lambda: model,
#         custom_preprocessor=custom_preprocessor,
#         device=None,
#     )
#     parameters.oft8.conv3d.weight = ttnn.to_device(parameters.oft8.conv3d.weight, device=device)
#     parameters.oft8.conv3d.bias = ttnn.to_device(parameters.oft8.conv3d.bias, device=device)
#     parameters.oft16.conv3d.weight = ttnn.to_device(parameters.oft16.conv3d.weight, device=device)
#     parameters.oft16.conv3d.bias = ttnn.to_device(parameters.oft16.conv3d.bias, device=device)
#     parameters.oft32.conv3d.weight = ttnn.to_device(parameters.oft32.conv3d.weight, device=device)
#     parameters.oft32.conv3d.bias = ttnn.to_device(parameters.oft32.conv3d.bias, device=device)

#     input1, input2, input3 = input_tensors
#     parameters.conv_args = {}
#     parameters.conv_args = infer_ttnn_module_args(
#         model=model,
#         run_model=lambda model: model(input1, input2, input3),
#         device=None,
#     )

#     parameters["model_args"] = model

#     return parameters
