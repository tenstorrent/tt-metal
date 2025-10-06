import ttnn
import torch

from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

from models.experimental.transfuser.reference.transfuser_backbone import TransfuserBackbone
from models.experimental.transfuser.reference.bottleneck import Bottleneck
from models.experimental.transfuser.reference.stage import Stage
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
        print(f"{model=}")
        # Image encoder conv1
        if hasattr(model, "image_encoder") and hasattr(model.image_encoder, "features"):
            weight, bias = fold_batch_norm2d_into_conv2d(
                model.image_encoder.features.conv1, model.image_encoder.features.bn1
            )
            parameters["image_encoder"] = {}
            parameters["image_encoder"]["features"] = {}
            parameters["image_encoder"]["features"]["conv1"] = {}
            parameters["image_encoder"]["features"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            # Lidar encoder conv1
            if hasattr(model, "lidar_encoder") and hasattr(model.lidar_encoder, "_model"):
                lidar_weight, lidar_bias = fold_batch_norm2d_into_conv2d(
                    model.lidar_encoder._model.conv1, model.lidar_encoder._model.bn1
                )
                parameters["lidar_encoder"] = {}
                parameters["lidar_encoder"]["_model"] = {}
                parameters["lidar_encoder"]["_model"]["conv1"] = {}
                parameters["lidar_encoder"]["_model"]["conv1"]["weight"] = ttnn.from_torch(
                    lidar_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )
                lidar_bias = lidar_bias.reshape((1, 1, 1, -1))
                parameters["lidar_encoder"]["_model"]["conv1"]["bias"] = ttnn.from_torch(
                    lidar_bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )

        # layer1 preprocessing for image encoder
        if hasattr(model.image_encoder.features, "layer1"):
            parameters["image_encoder"]["features"]["layer1"] = {}

            # 1st bottleneck
            b1_block = model.image_encoder.features.layer1.b1
            parameters["image_encoder"]["features"]["layer1"]["b1"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv1.conv, b1_block.conv1.bn)
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv1"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv2.conv, b1_block.conv2.bn)
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv2"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["image_encoder"]["features"]["layer1"]["b1"]["se"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b1"]["se"]["fc1"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b1"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b1_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["image_encoder"]["features"]["layer1"]["b1"]["se"]["fc2"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b1"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b1_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv3.conv, b1_block.conv3.bn)
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv3"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer1"]["b1"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # Downsample
            if hasattr(b1_block, "downsample") and b1_block.downsample is not None:
                weight, bias = fold_batch_norm2d_into_conv2d(b1_block.downsample.conv, b1_block.downsample.bn)
                parameters["image_encoder"]["features"]["layer1"]["b1"]["downsample"] = {}
                parameters["image_encoder"]["features"]["layer1"]["b1"]["downsample"]["weight"] = ttnn.from_torch(
                    weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )
                bias = bias.reshape((1, 1, 1, -1))
                parameters["image_encoder"]["features"]["layer1"]["b1"]["downsample"]["bias"] = ttnn.from_torch(
                    bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )

            # 2nd bottleneck (no downsample)
            b2_block = model.image_encoder.features.layer1.b2
            parameters["image_encoder"]["features"]["layer1"]["b2"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv1.conv, b2_block.conv1.bn)
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv1"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv2.conv, b2_block.conv2.bn)
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv2"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv3.conv, b2_block.conv3.bn)
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv3"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["image_encoder"]["features"]["layer1"]["b2"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["image_encoder"]["features"]["layer1"]["b2"]["se"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b2"]["se"]["fc1"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b2"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b2_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["image_encoder"]["features"]["layer1"]["b2"]["se"]["fc2"] = {}
            parameters["image_encoder"]["features"]["layer1"]["b2"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b2_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
        # if isinstance(model, TransfuserBackbone):
        # layer1 preprocessing for lidar encoder
        if hasattr(model.lidar_encoder._model, "layer1"):
            parameters["lidar_encoder"]["_model"]["layer1"] = {}

            # 1st bottleneck
            b1_block = model.lidar_encoder._model.layer1.b1
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv1.conv, b1_block.conv1.bn)
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv1"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv2.conv, b1_block.conv2.bn)
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv2"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b1_block.conv3.conv, b1_block.conv3.bn)
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv3"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["se"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["se"]["fc1"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b1_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["se"]["fc2"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b1_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # Downsample
            if hasattr(b1_block, "downsample") and b1_block.downsample is not None:
                if not isinstance(b1_block.downsample, torch.nn.Identity):
                    weight, bias = fold_batch_norm2d_into_conv2d(b1_block.downsample.conv, b1_block.downsample.bn)
                    parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["downsample"] = {}
                    parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["downsample"]["weight"] = ttnn.from_torch(
                        weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                    )
                    bias = bias.reshape((1, 1, 1, -1))
                    parameters["lidar_encoder"]["_model"]["layer1"]["b1"]["downsample"]["bias"] = ttnn.from_torch(
                        bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                    )

            # 2nd bottleneck for lidar
            b2_block = model.lidar_encoder._model.layer1.b2
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"] = {}

            # conv1 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv1.conv, b2_block.conv1.bn)
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv1"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv1"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv1"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv2 (3x3 grouped convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv2.conv, b2_block.conv2.bn)
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv2"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv2"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv2"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # conv3 (1x1 convolution)
            weight, bias = fold_batch_norm2d_into_conv2d(b2_block.conv3.conv, b2_block.conv3.bn)
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv3"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv3"]["weight"] = ttnn.from_torch(
                weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            bias = bias.reshape((1, 1, 1, -1))
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["conv3"]["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            # SE module
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["se"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["se"]["fc1"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["se"]["fc1"]["weight"] = ttnn.from_torch(
                b2_block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["se"]["fc2"] = {}
            parameters["lidar_encoder"]["_model"]["layer1"]["b2"]["se"]["fc2"]["weight"] = ttnn.from_torch(
                b2_block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
    elif isinstance(model, Bottleneck):
        # Handle standalone Bottleneck model
        # conv1 (1x1 convolution)
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1.conv, model.conv1.bn)
        parameters["conv1"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

        # conv2 (3x3 grouped convolution)
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv2.conv, model.conv2.bn)
        parameters["conv2"] = {}
        parameters["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

        # conv3 (1x1 convolution)
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv3.conv, model.conv3.bn)
        parameters["conv3"] = {}
        parameters["conv3"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv3"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

        # SE module
        parameters["se"] = {}
        parameters["se"]["fc1"] = {}
        parameters["se"]["fc1"]["weight"] = ttnn.from_torch(
            model.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["se"]["fc2"] = {}
        parameters["se"]["fc2"]["weight"] = ttnn.from_torch(
            model.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )

        # Downsample
        if (
            hasattr(model, "downsample")
            and model.downsample is not None
            and model.downsample.__class__.__name__ != "Identity"
        ):
            weight, bias = fold_batch_norm2d_into_conv2d(model.downsample[0], model.downsample[1])
            parameters["downsample"] = {}
            parameters["downsample"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
            bias = bias.reshape((1, 1, 1, -1))
            parameters["downsample"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
    elif isinstance(model, Stage):
        # Extract the stage layer (e.g., layer1, layer2, etc.)
        stage_layer = getattr(model.image_encoder.features, model.stage_name)

        parameters[model.stage_name] = {}

        # Process each bottleneck in the stage
        for block_idx, block_name in enumerate(["b1", "b2"]):  # Adjust based on your stage structure
            if hasattr(stage_layer, block_name):
                block = getattr(stage_layer, block_name)
                parameters[model.stage_name][block_name] = {}

                # conv1 (1x1 convolution)
                weight, bias = fold_batch_norm2d_into_conv2d(block.conv1.conv, block.conv1.bn)
                parameters[model.stage_name][block_name]["conv1"] = {
                    "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                    "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                }

                # conv2 (3x3 grouped convolution)
                weight, bias = fold_batch_norm2d_into_conv2d(block.conv2.conv, block.conv2.bn)
                parameters[model.stage_name][block_name]["conv2"] = {
                    "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                    "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                }

                # conv3 (1x1 convolution)
                weight, bias = fold_batch_norm2d_into_conv2d(block.conv3.conv, block.conv3.bn)
                parameters[model.stage_name][block_name]["conv3"] = {
                    "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                    "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                }

                # SE module (no bias as you confirmed)
                parameters[model.stage_name][block_name]["se"] = {
                    "fc1": {
                        "weight": ttnn.from_torch(block.se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
                    },
                    "fc2": {
                        "weight": ttnn.from_torch(block.se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
                    },
                }

                # Downsample (if exists)
                if (
                    hasattr(block, "downsample")
                    and block.downsample is not None
                    and not isinstance(block.downsample, torch.nn.Identity)
                ):
                    weight, bias = fold_batch_norm2d_into_conv2d(block.downsample.conv, block.downsample.bn)
                    parameters[model.stage_name][block_name]["downsample"] = {
                        "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                        "bias": ttnn.from_torch(
                            bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                        ),
                    }
    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
