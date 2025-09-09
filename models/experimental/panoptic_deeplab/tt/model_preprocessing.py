import ttnn
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_conv2d_wrapper import Conv2d

# def preprocess_linear_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT):
#     weight = weight.T.contiguous()
#     weight = ttnn.from_torch(weight, dtype=dtype, layout=layout)
#     return weight


def custom_preprocessor(model, name):
    parameters = {}
    # Ako je model vaš Conv2d wrapper
    if isinstance(model, Conv2d):
        # Konvolucioni dio
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
        if hasattr(model, "norm") and model.norm is not None:
            norm_params = {}
            # Provjeravamo koji je tip normalizacije
            if isinstance(model.norm, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                norm_params["weight"] = ttnn.from_torch(model.norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                norm_params["bias"] = ttnn.from_torch(model.norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                norm_params["running_mean"] = ttnn.from_torch(
                    model.norm.running_mean, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                norm_params["running_var"] = ttnn.from_torch(
                    model.norm.running_var, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
            elif isinstance(model.norm, (nn.LayerNorm, LayerNorm)):
                norm_params["weight"] = ttnn.from_torch(model.norm.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                norm_params["bias"] = ttnn.from_torch(model.norm.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            # Dodajemo 'norm' ključ u parametre za ovaj Conv2d sloj
            parameters["norm"] = norm_params
    elif isinstance(model, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = ttnn.from_torch(model.bias, dtype=ttnn.bfloat16)
        parameters["running_mean"] = ttnn.from_torch(model.running_mean, dtype=ttnn.bfloat16)
        parameters["running_var"] = ttnn.from_torch(model.running_var, dtype=ttnn.bfloat16)

    return parameters


def create_panoptic_deeplab_parameters(model: PytorchPanopticDeepLab, device):
    model_initializer = lambda: model

    parameters = preprocess_model_parameters(
        initialize_model=model_initializer,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    print(parameters)
    return parameters
