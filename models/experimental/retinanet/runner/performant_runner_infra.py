# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import ttnn
import os
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

from models.experimental.retinanet.TTNN.tt_backbone import TTBackbone
from models.experimental.retinanet.TTNN.custom_preprocessor import (
    create_custom_mesh_preprocessor,
    preprocess_regression_head_parameters,
    preprocess_classification_head_parameters,
)
from models.experimental.retinanet.TTNN.regression_head import ttnn_retinanet_regression_head
from models.experimental.retinanet.TTNN.classification_head import ttnn_retinanet_classification_head
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc
from models.common.utility_functions import divup, is_wormhole_b0


class RetinaNetPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_config=None,
        model_location_generator=None,
        resolution=(512, 512),
        torch_input_tensor=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
        input_path=None,
    ):
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.resolution = resolution
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.model_config = model_config
        self.model_location_generator = model_location_generator
        self.torch_input_tensor = torch_input_tensor

        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer
        self.real_input_path = input_path

        # Load full RetinaNet model
        self.torch_model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        self.torch_model.eval()

        # Store components
        self.torch_backbone = self.torch_model.backbone
        self.torch_regression_head = self.torch_model.head.regression_head
        self.torch_classification_head = self.torch_model.head.classification_head

        # Create input tensor
        if self.real_input_path and os.path.exists(self.real_input_path):
            img = Image.open(self.real_input_path).convert("RGB")
            preprocess = transforms.Compose(
                [
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            input_tensor = preprocess(img)
            self.torch_input_tensor = input_tensor.unsqueeze(0)

            expected_shape = (batch_size * self.num_devices, 3, self.resolution[0], self.resolution[1])
            if self.torch_input_tensor.shape != expected_shape:
                logger.warning(
                    f"Input shape mismatch. Expected: {expected_shape}, Got: {self.torch_input_tensor.shape}"
                )
        else:
            self.torch_input_tensor = torch.randn(
                (self.batch_size * self.num_devices, 3, self.resolution[0], self.resolution[1]), dtype=torch.float32
            )

        # Get PyTorch reference outputs
        with torch.no_grad():
            backbone_features = self.torch_backbone(self.torch_input_tensor)
            self.torch_regression_output = self.torch_regression_head(list(backbone_features.values()))
            self.torch_classification_output = self.torch_classification_head(list(backbone_features.values()))
            self.torch_backbone_outputs = backbone_features

        # Preprocess model parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        self.model_config = (
            model_config
            if model_config is not None
            else {
                "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
                "WEIGHTS_DTYPE": ttnn.bfloat16,
                "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            }
        )
        # Preprocess head parameters
        self.regression_parameters = preprocess_regression_head_parameters(
            torch_head=self.torch_model.head.regression_head,
            device=device,
            mesh_mapper=self.weights_mesh_mapper,
            model_config=model_config,
        )

        self.classification_parameters = preprocess_classification_head_parameters(
            torch_head=self.torch_model.head.classification_head,
            device=device,
            mesh_mapper=self.weights_mesh_mapper,
            model_config=model_config,
        )

        # Extract backbone parameters
        self.backbone_parameters = parameters.get("backbone", parameters)

        # Create TTNN model
        self.ttnn_backbone = TTBackbone(parameters=self.backbone_parameters, model_config=model_config)

        # Store input shapes for heads
        self.input_shapes = [
            (self.torch_backbone_outputs["0"].shape[2], self.torch_backbone_outputs["0"].shape[3]),
            (self.torch_backbone_outputs["1"].shape[2], self.torch_backbone_outputs["1"].shape[3]),
            (self.torch_backbone_outputs["2"].shape[2], self.torch_backbone_outputs["2"].shape[3]),
            (self.torch_backbone_outputs["p6"].shape[2], self.torch_backbone_outputs["p6"].shape[3]),
            (self.torch_backbone_outputs["p7"].shape[2], self.torch_backbone_outputs["p7"].shape[3]),
        ]

    def setup_l1_sharded_input(self, device, torch_input_tensor=None, min_channels=16):
        """Setup L1 sharded input following the pattern from other models"""
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")

        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        n, c, h, w = torch_input_tensor.shape

        if c < min_channels:
            c = min_channels
        elif c % min_channels != 0:
            c = ((c // min_channels) + 1) * min_channels

        n = n // self.num_devices if n // self.num_devices != 0 else n

        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )

        assert torch_input_tensor.ndim == 4, "Expected input tensor to have shape (BS, C, H, W)"

        # Convert to NHWC format for TTNN
        torch_input_tensor_nhwc = torch_input_tensor.permute(0, 2, 3, 1)
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor_nhwc, dtype=ttnn.bfloat16, mesh_mapper=self.inputs_mesh_mapper
        )

        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None):
        """Setup DRAM sharded input for RetinaNet"""
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(device, torch_input_tensor)
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], (dram_grid_size.x * dram_grid_size.y)),
                tt_inputs_host.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def setup_dram_interleaved_input(self, torch_input_tensor=None, mesh_mapper=None):
        """Setup DRAM interleaved input for RetinaNet (supports DRAM sliced Conv2d)"""
        mesh_mapper = self.inputs_mesh_mapper if mesh_mapper is None else mesh_mapper
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        # Convert to NHWC format
        if torch_input_tensor.shape[-1] != 3:
            torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)

        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, mesh_mapper=self.inputs_mesh_mapper)
        return tt_inputs_host, ttnn.DRAM_MEMORY_CONFIG

    def run(self):
        """Run full RetinaNet pipeline: backbone → regression head → classification head"""
        # Run backbone
        backbone_output = self.ttnn_backbone(self.input_tensor, self.device)

        # Convert to list for heads
        fpn_features = [backbone_output[key] for key in ["0", "1", "2", "p6", "p7"]]

        # Run regression head
        self.regression_output = ttnn_retinanet_regression_head(
            feature_maps=fpn_features,
            parameters=self.regression_parameters,
            device=self.device,
            in_channels=256,
            num_anchors=9,
            batch_size=self.batch_size * self.num_devices,
            input_shapes=self.input_shapes,
            model_config=self.model_config,
            optimization_profile="optimized",
        )

        # Run classification head
        self.classification_output = ttnn_retinanet_classification_head(
            feature_maps=fpn_features,
            parameters=self.classification_parameters,
            device=self.device,
            in_channels=256,
            num_anchors=9,
            batch_size=self.batch_size * self.num_devices,
            input_shapes=self.input_shapes,
            model_config=self.model_config,
            optimization_profile="optimized",
        )

        # Store outputs with consistent naming
        self.output_tensor = self.regression_output
        self.regression_output_tensor = self.regression_output
        self.classification_output_tensor = self.classification_output

    def validate(self, regression_output=None, classification_output=None):
        """Validate outputs against PyTorch reference"""
        regression_output = self.regression_output if regression_output is None else regression_output
        classification_output = self.classification_output if classification_output is None else classification_output

        # Convert to torch
        regression_torch = ttnn.to_torch(regression_output, mesh_composer=self.outputs_mesh_composer)
        classification_torch = ttnn.to_torch(classification_output, mesh_composer=self.outputs_mesh_composer)

        # Validate regression head
        regression_pcc = 0.91
        passed_reg, msg_reg = check_with_pcc(self.torch_regression_output, regression_torch, pcc=regression_pcc)
        logger.info(f"RetinaNet Regression Head: batch_size={self.batch_size}, PCC={msg_reg}")

        # Validate classification head
        classification_pcc = 0.91
        passed_cls, msg_cls = check_with_pcc(
            self.torch_classification_output, classification_torch, pcc=classification_pcc
        )
        logger.info(f"RetinaNet Classification Head: batch_size={self.batch_size}, PCC={msg_cls}")

        self.pcc_passed = passed_reg and passed_cls
        self.pcc_message = f"Regression: {msg_reg}, Classification: {msg_cls}"

        assert self.pcc_passed, logger.error(f"RetinaNet PCC check failed: {self.pcc_message}")

    def dealloc_output(self):
        """Deallocate output tensors"""
        if hasattr(self, "regression_output") and self.regression_output is not None:
            ttnn.deallocate(self.regression_output)
        if hasattr(self, "classification_output") and self.classification_output is not None:
            ttnn.deallocate(self.classification_output)
