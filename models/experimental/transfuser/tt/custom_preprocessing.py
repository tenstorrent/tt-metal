import ttnn
import torch

from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

from models.experimental.transfuser.reference.transfuser_backbone import TransfuserBackbone
from models.experimental.transfuser.reference.common import Conv2d


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    parameters = {}
    if isinstance(model, Conv2d):
        if model.norm is not None:
            weight, bias = fold_batch_norm2d_into_conv2d(model, model.norm)
        else:
            weight = model.weight.clone().detach().contiguous()
            bias = (
                model.bias.clone().detach().contiguous() if model.bias is not None else torch.zeros(model.out_channels)
            )
        parameters["weight"] = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
        parameters["bias"] = ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper)
    # elif isinstance(
    #     model,
    #     (TransfuserBackbone,),
    # ):
    #     # Let the sub-modules handle their own preprocessing
    #     for child_name, child in model.named_children():
    #         parameters[child_name] = convert_torch_model_to_ttnn_model(
    #             child,
    #             name=f"{name}.{child_name}",
    #             custom_preprocessor=custom_preprocessor_func,
    #             convert_to_ttnn=convert_to_ttnn,
    #             ttnn_module_args=ttnn_module_args,
    #         )
    elif isinstance(model, TransfuserBackbone):
        # Image encoder conv1
        if hasattr(model, "image_encoder") and hasattr(model.image_encoder, "features"):
            weight, bias = fold_batch_norm2d_into_conv2d(
                model.image_encoder.features.conv1, model.image_encoder.features.bn1
            )
            parameters["image_encoder"] = {}
            parameters["image_encoder"]["features"] = {}
            parameters["image_encoder"]["features"]["conv1"] = {}
            parameters["image_encoder"]["features"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper
            )
            # Lidar encoder
            if hasattr(model, "lidar_encoder") and hasattr(model.lidar_encoder, "_model"):
                lidar_weight, lidar_bias = fold_batch_norm2d_into_conv2d(
                    model.lidar_encoder._model.conv1, model.lidar_encoder._model.bn1
                )
                parameters["lidar_encoder"] = {}
                parameters["lidar_encoder"]["_model"] = {}
                parameters["lidar_encoder"]["_model"]["conv1"] = {}
                parameters["lidar_encoder"]["_model"]["conv1"]["weight"] = ttnn.from_torch(
                    lidar_weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper
                )
                lidar_bias = lidar_bias.reshape((1, 1, 1, -1))
                parameters["lidar_encoder"]["_model"]["conv1"]["bias"] = ttnn.from_torch(
                    lidar_bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper
                )
    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
