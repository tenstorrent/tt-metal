# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import pickle
import os

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.retinanet.TTNN.tt_bottleneck import TTBottleneck, get_bottleneck_optimisation
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d


pickle_map_torch = {
    "backbone_layer1_0": "models/experimental/retinanet/resources/pickle/retinanet_stem_output_torch.pkl",
    "backbone_layer1_1": "models/experimental/retinanet/resources/pickle/backbone_layer1_0_torch.pkl",
    "backbone_layer1_2": "models/experimental/retinanet/resources/pickle/backbone_layer1_1_torch.pkl",
    "backbone_layer2_0": "models/experimental/retinanet/resources/pickle/backbone_layer1_2_torch.pkl",
    "backbone_layer2_1": "models/experimental/retinanet/resources/pickle/backbone_layer2_0_torch.pkl",
    "backbone_layer2_2": "models/experimental/retinanet/resources/pickle/backbone_layer2_1_torch.pkl",
    "backbone_layer2_3": "models/experimental/retinanet/resources/pickle/backbone_layer2_2_torch.pkl",
    "backbone_layer3_0": "models/experimental/retinanet/resources/pickle/backbone_layer2_3_torch.pkl",
    "backbone_layer3_1": "models/experimental/retinanet/resources/pickle/backbone_layer3_0_torch.pkl",
    "backbone_layer3_2": "models/experimental/retinanet/resources/pickle/backbone_layer3_1_torch.pkl",
    "backbone_layer3_3": "models/experimental/retinanet/resources/pickle/backbone_layer3_2_torch.pkl",
    "backbone_layer3_4": "models/experimental/retinanet/resources/pickle/backbone_layer3_3_torch.pkl",
    "backbone_layer3_5": "models/experimental/retinanet/resources/pickle/backbone_layer3_4_torch.pkl",
    "backbone_layer4_0": "models/experimental/retinanet/resources/pickle/backbone_layer3_5_torch.pkl",
    "backbone_layer4_1": "models/experimental/retinanet/resources/pickle/backbone_layer4_0_torch.pkl",
    "backbone_layer4_2": "models/experimental/retinanet/resources/pickle/backbone_layer4_1_torch.pkl",
}

pickle_map_tt = {
    "backbone_layer1_0": "models/experimental/retinanet/resources/pickle/retinanet_stem_output_tt.pkl",
    "backbone_layer1_1": "models/experimental/retinanet/resources/pickle/backbone_layer1_0_tt.pkl",
    "backbone_layer1_2": "models/experimental/retinanet/resources/pickle/backbone_layer1_1_tt.pkl",
    "backbone_layer2_0": "models/experimental/retinanet/resources/pickle/backbone_layer1_2_tt.pkl",
    "backbone_layer2_1": "models/experimental/retinanet/resources/pickle/backbone_layer2_0_tt.pkl",
    "backbone_layer2_2": "models/experimental/retinanet/resources/pickle/backbone_layer2_1_tt.pkl",
    "backbone_layer2_3": "models/experimental/retinanet/resources/pickle/backbone_layer2_2_tt.pkl",
    "backbone_layer3_0": "models/experimental/retinanet/resources/pickle/backbone_layer2_3_tt.pkl",
    "backbone_layer3_1": "models/experimental/retinanet/resources/pickle/backbone_layer3_0_tt.pkl",
    "backbone_layer3_2": "models/experimental/retinanet/resources/pickle/backbone_layer3_1_tt.pkl",
    "backbone_layer3_3": "models/experimental/retinanet/resources/pickle/backbone_layer3_2_tt.pkl",
    "backbone_layer3_4": "models/experimental/retinanet/resources/pickle/backbone_layer3_3_tt.pkl",
    "backbone_layer3_5": "models/experimental/retinanet/resources/pickle/backbone_layer3_4_tt.pkl",
    "backbone_layer4_0": "models/experimental/retinanet/resources/pickle/backbone_layer3_5_tt.pkl",
    "backbone_layer4_1": "models/experimental/retinanet/resources/pickle/backbone_layer4_0_tt.pkl",
    "backbone_layer4_2": "models/experimental/retinanet/resources/pickle/backbone_layer4_1_tt.pkl",
}


def conv_bn_to_params(conv, bn, mesh_mapper):
    """Fold BN into Conv and return TTNN weight/bias."""
    if bn is None:
        weight = conv.weight
        bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
    else:
        weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)

    return {
        "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        "bias": ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
    }


def bottleneck_to_params(block, mesh_mapper):
    """Extract and fold conv+bn from a ResNet Bottleneck block (no activations)."""
    layers = {}

    # conv1 + bn1
    layers["conv1"] = conv_bn_to_params(block.conv1, block.bn1, mesh_mapper)

    # conv2 + bn2
    layers["conv2"] = conv_bn_to_params(block.conv2, block.bn2, mesh_mapper)

    # conv3 + bn3
    layers["conv3"] = conv_bn_to_params(block.conv3, block.bn3, mesh_mapper)

    # downsample (if present)
    if block.downsample is not None:
        layers["downsample"] = {}
        ds_conv = block.downsample[0]
        ds_bn = block.downsample[1] if len(block.downsample) > 1 else None
        layers["downsample"]["0"] = conv_bn_to_params(ds_conv, ds_bn, mesh_mapper)

    return layers


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        block = model
        return bottleneck_to_params(block, mesh_mapper)

    return custom_mesh_preprocessor


class BottleneckTestInfra:
    def __init__(self, device, batch_size, channels, height, width, stride, dilation, downsample, name, model_config):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)  # Seed once for determinism
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.pcc_passed = False
        self.pcc_message = "call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size
        self.name = name
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Torch model
        # Load RetinaNet model
        retinanet = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        backbone = retinanet.backbone.body
        layer = getattr(backbone, f"layer{int(name[-3])}")
        torch_model = layer[int(name[-1])]
        torch_model.eval()

        # Torch input + golden output
        # input_shape = (batch_size * self.num_devices, channels, height, width)
        # self.torch_input_tensor = torch.randn(input_shape, dtype=torch.float)

        with open(pickle_map_torch[self.name], "rb") as f:
            self.torch_input_tensor = pickle.load(f)
        with open(pickle_map_tt[self.name], "rb") as f:
            tt_host_tensor = pickle.load(f)
            tt_host_tensor = ttnn.from_torch(
                tt_host_tensor,
                dtype=ttnn.bfloat16,
                mesh_mapper=self.inputs_mesh_mapper,
            )
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

        # Preprocess model params
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Convert input to TTNN host tensor
        def to_ttnn_host(tensor):
            return ttnn.from_torch(
                tensor.permute(0, 2, 3, 1),
                dtype=ttnn.bfloat16,
                mesh_mapper=self.inputs_mesh_mapper,
            )

        tt_host_tensor = to_ttnn_host(self.torch_input_tensor)

        # TTNN model
        self.ttnn_model = TTBottleneck(
            parameters=parameters,
            downsample=downsample,
            stride=stride,
            dilation=dilation,
            name=name,
            model_config=model_config,
            layer_optimisations=get_bottleneck_optimisation(name),
        )

        # Move input to device
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        # self.input_tensor = ttnn.to_memory_config(self.input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        # Run + validate
        self.run()
        self.validate()

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),  # inputs
                None,  # weights
                ttnn.ConcatMeshToTensor(device, dim=0),  # outputs
            )
        return None, None, None

    def run(self):
        self.output_tensor, _ = self.ttnn_model(self.input_tensor, self.device, self.input_tensor.shape)
        return self.output_tensor

    def validate(self, output_tensor=None):
        base_path = "models/experimental/retinanet/resources/pickle"
        layer_path_torch = self.name + "_torch.pkl"
        layer_path_ttnn = self.name + "_tt.pkl"
        save_to_path_torch = os.path.join(base_path, layer_path_torch)
        save_to_path_ttnn = os.path.join(base_path, layer_path_ttnn)

        with open(save_to_path_ttnn, "wb") as f:
            pickle.dump(ttnn.to_torch(self.output_tensor, device=self.device), f)
        with open(save_to_path_torch, "wb") as f:
            pickle.dump(self.torch_output_tensor, f)

        tt_output_tensor = self.output_tensor if output_tensor is None else output_tensor
        tt_output_tensor_torch = ttnn.to_torch(
            tt_output_tensor, device=self.device, mesh_composer=self.output_mesh_composer
        )

        # Free device memory
        ttnn.deallocate(tt_output_tensor)

        expected_shape = self.torch_output_tensor.shape
        tt_output_tensor_torch = torch.reshape(
            tt_output_tensor_torch, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        tt_output_tensor_torch = torch.permute(tt_output_tensor_torch, (0, 3, 1, 2))

        valid_pcc = 0.99
        self.pcc_passed, self.pcc_message = check_with_pcc(
            self.torch_output_tensor, tt_output_tensor_torch, pcc=valid_pcc
        )

        assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
        logger.info(
            f"Bottleneck `{self.name}` passed: "
            f"batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, "
            f"PCC={self.pcc_message}"
        )

        return self.pcc_passed, self.pcc_message


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, channels, height, width, stride, dilation, downsample, name",
    [
        # layer1
        (1, 64, 128, 128, 1, 1, True, "backbone_layer1_0"),
        (1, 256, 128, 128, 1, 1, False, "backbone_layer1_1"),
        (1, 256, 128, 128, 1, 1, False, "backbone_layer1_2"),
        # layer2
        (1, 256, 128, 128, 2, 1, True, "backbone_layer2_0"),
        (1, 512, 64, 64, 1, 1, False, "backbone_layer2_1"),
        (1, 512, 64, 64, 1, 1, False, "backbone_layer2_2"),
        (1, 512, 64, 64, 1, 1, False, "backbone_layer2_3"),
        # layer3
        (1, 512, 64, 64, 2, 1, True, "backbone_layer3_0"),
        (1, 1024, 32, 32, 1, 1, False, "backbone_layer3_1"),
        (1, 1024, 32, 32, 1, 1, False, "backbone_layer3_2"),
        (1, 1024, 32, 32, 1, 1, False, "backbone_layer3_3"),
        (1, 1024, 32, 32, 1, 1, False, "backbone_layer3_4"),
        (1, 1024, 32, 32, 1, 1, False, "backbone_layer3_5"),
        # layer4
        (1, 1024, 32, 32, 2, 1, True, "backbone_layer4_0"),
        (1, 2048, 16, 16, 1, 1, False, "backbone_layer4_1"),
        (1, 2048, 16, 16, 1, 1, False, "backbone_layer4_2"),
    ],
)
def test_bottleneck(device, batch_size, channels, height, width, stride, dilation, downsample, name):
    BottleneckTestInfra(device, batch_size, channels, height, width, stride, dilation, downsample, name, model_config)
