# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from loguru import logger
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc
import pickle

from models.experimental.retinanet.tt.tt_stem import resnet50Stem, neck_optimisations
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d


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


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        conv = model.conv1
        bn = model.bn1
        relu = model.relu
        pool = model.maxpool

        stem_params = conv_bn_to_params(conv, bn, mesh_mapper)
        stem_params["activation"] = "relu"
        stem_params["pool"] = {"kernel_size": pool.kernel_size, "stride": pool.stride}

        return {"conv1": stem_params}

    return custom_mesh_preprocessor


class Resnet50StemTestInfra:
    def __init__(self, device, batch_size, inplanes, planes, height, width, stride, model_config, name):
        super().__init__()
        self._init_seeds()
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size * self.num_devices
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        self.name = name

        # Build reference torch model
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        preprocess = transforms.Compose(
            [
                transforms.CenterCrop((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        img = Image.open("models/experimental/retinanet/resources/dog.jpg").convert("RGB")

        retinanet = retinanet_resnet50_fpn_v2(weights=weights).eval()
        backbone = retinanet.backbone.body
        torch_model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv1", backbone.conv1),
                    ("bn1", backbone.bn1),
                    ("relu", backbone.relu),
                    ("maxpool", backbone.maxpool),
                ]
            )
        )

        # Preprocess parameters for TTNN
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Prepare golden inputs/outputs
        # input_shape = (self.batch_size, inplanes, height, width)
        # self.torch_input_tensor = torch.randn(input_shape, dtype=torch.float)

        self.torch_input_tensor = preprocess(img).unsqueeze(0)
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

        # Convert input to TTNN format
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)

        # Build TTNN model
        self.ttnn_model = resnet50Stem(
            parameters=parameters,
            stride=stride,
            model_config=model_config,
            layer_optimisations=neck_optimisations,
        )

        # Run + validate
        self.run()
        self.validate(model_config)

    def _init_seeds(self):
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            self._model_initialized = True

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),
                None,
                ttnn.ConcatMeshToTensor(device, dim=0),
            )
        return None, None, None

    def run(self):
        self.output_tensor = self.ttnn_model(self.input_tensor, self.device)
        with open("models/experimental/retinanet/resources/pickle/retinanet_stem_output_tt.pkl", "wb") as f:
            pickle.dump(ttnn.to_torch(self.output_tensor, device=self.device), f)
        return self.output_tensor

    def validate(self, model_config, output_tensor=None):
        tt_output_tensor = self.output_tensor if output_tensor is None else output_tensor
        tt_output_tensor_torch = ttnn.to_torch(
            tt_output_tensor,
            device=self.device,
            mesh_composer=self.output_mesh_composer,
        )

        # Deallocate output tensor
        ttnn.deallocate(tt_output_tensor)

        # Reshape + permute back to NCHW
        expected_shape = self.torch_output_tensor.shape
        tt_output_tensor_torch = torch.reshape(
            tt_output_tensor_torch,
            (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]),
        )
        tt_output_tensor_torch = torch.permute(tt_output_tensor_torch, (0, 3, 1, 2))

        with open("models/experimental/retinanet/resources/pickle/retinanet_stem_output_torch.pkl", "wb") as f:
            pickle.dump(self.torch_output_tensor, f)

        # PCC validation
        pcc_passed, pcc_message = check_with_pcc(self.torch_output_tensor, tt_output_tensor_torch, pcc=0.99)
        assert pcc_passed, logger.error(f"PCC check failed: {pcc_message}")

        logger.info(
            f"ResNet50 Stem Block [{self.name}] - "
            f"batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, "
            f"PCC={pcc_message}"
        )
        return pcc_passed, pcc_message


# Default model config
model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, inplanes, planes, height, width, stride, name",
    [
        (1, 3, 128, 512, 512, 1, "backbone.stem"),
    ],
)
def test_stem(device, batch_size, inplanes, planes, height, width, stride, name):
    Resnet50StemTestInfra(
        device,
        batch_size,
        inplanes,
        planes,
        height,
        width,
        stride,
        model_config,
        name,
    )
