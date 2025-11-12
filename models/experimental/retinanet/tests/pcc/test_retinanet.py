# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc

from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from PIL import Image
from torchvision import transforms

from models.experimental.retinanet.tt.tt_backbone import TTBackbone
from models.experimental.retinanet.tt.custom_preprocessor import (
    create_custom_mesh_preprocessor,
    preprocess_regression_head_parameters,
    preprocess_classification_head_parameters,
)

from models.experimental.retinanet.tt.tt_regression_head import ttnn_retinanet_regression_head
from models.experimental.retinanet.tt.tt_classification_head import ttnn_retinanet_classification_head


class BackboneTestInfra:
    def __init__(self, device, batch_size, in_channels, height, width, model_config, name):
        super().__init__()
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.pcc_passed = False
        self.pcc_message = "call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.batch_size = batch_size * self.num_devices
        self.name = name
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Load FULL RetinaNet model (not just backbone)
        retinanet = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        retinanet.eval()

        # Store both backbone + regression head +classification head
        self.torch_backbone = retinanet.backbone
        self.torch_regression_head = retinanet.head.regression_head
        self.torch_classification_head = retinanet.head.classification_head

        # Torch input preprocessing
        preprocess = transforms.Compose(
            [
                transforms.CenterCrop((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        img = Image.open("models/experimental/retinanet/resources/dog_800x800.jpg").convert("RGB")
        self.torch_input_tensor = preprocess(img).unsqueeze(0)

        # Get backbone features
        backbone_features = self.torch_backbone(self.torch_input_tensor)

        # Run PyTorch regression head to get golden output
        torch_regression_output = self.torch_regression_head(list(backbone_features.values()))

        # Run PyTorch classifiction head to get golden output
        torch_classification_output = self.torch_classification_head(list(backbone_features.values()))

        # Store all outputs (backbone + regression head)
        self.torch_output_tensor = {
            **backbone_features,  # FPN levels: "0", "1", "2", "p6", "p7"
            "regression": torch_regression_output,
            "classification": torch_classification_output,
        }

        # Preprocess parameters for FULL RetinaNet model (backbone + regression head)
        parameters = preprocess_model_parameters(
            initialize_model=lambda: retinanet,  # Full model, not just backbone
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        # Preprocess regression head parameters using specialized preprocessor
        regression_parameters = preprocess_regression_head_parameters(
            torch_head=retinanet.head.regression_head,
            device=device,
            mesh_mapper=self.weights_mesh_mapper,
            model_config=model_config,
        )
        # preprocess classification head parameters
        classification_parameters = preprocess_classification_head_parameters(
            torch_head=retinanet.head.classification_head,
            device=device,
            mesh_mapper=self.weights_mesh_mapper,
            model_config=model_config,
        )
        # Extract parameters for backbone and regression head
        self.backbone_parameters = parameters.get("backbone", parameters)  # Fallback to full params if no backbone key
        self.regression_parameters = regression_parameters
        self.classification_parameters = classification_parameters

        # Convert input to TTNN host tensor
        def to_ttnn_host(tensor):
            return ttnn.from_torch(
                tensor.permute(0, 2, 3, 1),
                dtype=ttnn.bfloat16,
                mesh_mapper=self.inputs_mesh_mapper,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        tt_host_tensor = to_ttnn_host(self.torch_input_tensor)

        # TTNN backbone model
        self.ttnn_model = TTBackbone(parameters=self.backbone_parameters, model_config=model_config)

        # Move input to device
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)

        # Run + validate
        self.run()
        self.validate()

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = None
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def run(self):
        # Run backbone to get FPN features
        backbone_output = self.ttnn_model(self.input_tensor, self.device)

        # Convert backbone output dict to list for regression head
        fpn_features = [backbone_output[key] for key in ["0", "1", "2", "p6", "p7"]]

        # Determine input shapes based on actual feature map sizes
        input_shapes = [
            (self.torch_output_tensor["0"].shape[2], self.torch_output_tensor["0"].shape[3]),
            (self.torch_output_tensor["1"].shape[2], self.torch_output_tensor["1"].shape[3]),
            (self.torch_output_tensor["2"].shape[2], self.torch_output_tensor["2"].shape[3]),
            (self.torch_output_tensor["p6"].shape[2], self.torch_output_tensor["p6"].shape[3]),
            (self.torch_output_tensor["p7"].shape[2], self.torch_output_tensor["p7"].shape[3]),
        ]

        # Run regression head
        regression_output = ttnn_retinanet_regression_head(
            feature_maps=fpn_features,
            parameters=self.regression_parameters,
            device=self.device,
            in_channels=256,
            num_anchors=9,
            batch_size=self.batch_size,
            input_shapes=input_shapes,
            model_config=model_config,
            optimization_profile="optimized",
        )

        logger.debug("✅✅✅ REGRESSION HEAD Complete ✅✅✅")
        # Run classification head
        classification_output = ttnn_retinanet_classification_head(
            feature_maps=fpn_features,
            parameters=self.classification_parameters,
            device=self.device,
            in_channels=256,
            num_anchors=9,
            batch_size=self.batch_size,
            input_shapes=input_shapes,
            model_config=model_config,
            optimization_profile="optimized",
        )
        logger.debug("✅✅✅ CLASSIFICATION HEAD Complete ✅✅✅")

        # Combine all outputs
        self.output_tensor = {
            **backbone_output,
            "regression": regression_output,
            "classification": classification_output,
        }

        return self.output_tensor

    def validate(self, output_tensor=None):
        tt_output = self.output_tensor if output_tensor is None else output_tensor

        # Add regression head PCC threshold
        valid_pcc = {
            "0": 0.99,
            "1": 0.99,
            "2": 0.99,
            "p6": 0.99,
            "p7": 0.99,
            "regression": 0.99,
            "classification": 0.99,
        }

        self.pcc_passed_all = []
        self.pcc_message_all = []
        pcc_results = {}
        for key in tt_output:
            tt_output_tensor = tt_output[key]
            torch_output_tensor = self.torch_output_tensor[key]

            # Convert TTNN output to torch
            if key == "regression" or key == "classification":
                tt_output_tensor_torch = ttnn.to_torch(tt_output_tensor)
            else:
                # Backbone FPN outputs need permutation
                tt_output_tensor_torch = ttnn.to_torch(
                    tt_output_tensor,
                    dtype=torch_output_tensor.dtype,
                    device=self.device,
                    mesh_composer=self.output_mesh_composer,
                )

                expected_shape = torch_output_tensor.shape
                tt_output_tensor_torch = torch.reshape(
                    tt_output_tensor_torch, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
                )
                tt_output_tensor_torch = torch.permute(tt_output_tensor_torch, (0, 3, 1, 2))

            # Free device memory
            ttnn.deallocate(tt_output_tensor)

            pcc_passed, pcc_message = check_with_pcc(torch_output_tensor, tt_output_tensor_torch, pcc=valid_pcc[key])
            self.pcc_passed_all.append(pcc_passed)
            self.pcc_message_all.append(pcc_message)
            pcc_results[key] = pcc_message

        assert all(self.pcc_passed_all), logger.error(f"PCC check failed: {self.pcc_message_all}")
        pcc_summary = ", ".join([f"{k}={v}" for k, v in pcc_results.items()])
        logger.info(
            f"ResNet52 Backbone + Regression Head - batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, "
            f"PCC={pcc_summary}"
        )

        return self.pcc_passed_all, self.pcc_message_all


model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, height, width, name",
    [
        (1, 3, 512, 512, "backbone"),
    ],
)
def test_backbone(device, batch_size, in_channels, height, width, name):
    BackboneTestInfra(device, batch_size, in_channels, height, width, model_config, name)
