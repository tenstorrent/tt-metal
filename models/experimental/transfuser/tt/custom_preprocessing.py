import ttnn
import torch

from ttnn.model_preprocessing import convert_torch_model_to_ttnn_model, fold_batch_norm2d_into_conv2d

# from models.experimental.panoptic_deeplab.reference.decoder import DecoderModel
# from models.experimental.panoptic_deeplab.reference.aspp import ASPPModel
# from models.experimental.panoptic_deeplab.reference.head import HeadModel
# from models.experimental.panoptic_deeplab.reference.res_block import ResModel
# from models.experimental.panoptic_deeplab.reference.resnet52_stem import DeepLabStem
# from models.experimental.panoptic_deeplab.reference.resnet52_bottleneck import Bottleneck
# from models.experimental.panoptic_deeplab.reference.resnet52_backbone import ResNet52BackBone as TorchBackbone
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
    elif isinstance(
        model,
        (TransfuserBackbone,),
    ):
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


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
