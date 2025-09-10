# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.panoptic_deeplab.reference.panoptic_deeplab import TorchPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.panoptic_deeplab import TTPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


class PanopticDeepLabTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        in_channels,
        height,
        width,
        model_config,
    ):
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
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Initialize torch model
        torch_model = TorchPanopticDeepLab().eval()

        # Create input tensor
        input_shape = (batch_size * self.num_devices, in_channels, height, width)
        self.torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)

        # Preprocess model parameters
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        parameters.conv_args = {}
        sample_x = torch.randn(1, 2048, 32, 64)
        sample_res3 = torch.randn(1, 512, 64, 128)
        sample_res2 = torch.randn(1, 256, 128, 256)

        # For semantic decoder
        if hasattr(parameters, "semantic_decoder"):
            # ASPP
            aspp_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.aspp, run_model=lambda model: model(sample_x), device=None
            )
            if hasattr(parameters.semantic_decoder, "aspp"):
                parameters.semantic_decoder.aspp.conv_args = aspp_args

            # Res3
            aspp_out = torch_model.semantic_decoder.aspp(sample_x)
            res3_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.res3,
                run_model=lambda model: model(aspp_out, sample_res3),
                device=None,
            )
            if hasattr(parameters.semantic_decoder, "res3"):
                parameters.semantic_decoder.res3.conv_args = res3_args

            # Res2
            res3_out = torch_model.semantic_decoder.res3(aspp_out, sample_res3)
            res2_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.res2,
                run_model=lambda model: model(res3_out, sample_res2),
                device=None,
            )
            if hasattr(parameters.semantic_decoder, "res2"):
                parameters.semantic_decoder.res2.conv_args = res2_args

            # Head
            res2_out = torch_model.semantic_decoder.res2(res3_out, sample_res2)
            head_args = infer_ttnn_module_args(
                model=torch_model.semantic_decoder.head_1, run_model=lambda model: model(res2_out), device=None
            )
            if hasattr(parameters.semantic_decoder, "head_1"):
                parameters.semantic_decoder.head_1.conv_args = head_args

        # For instance decoder
        if hasattr(parameters, "instance_decoder"):
            # ASPP
            aspp_args = infer_ttnn_module_args(
                model=torch_model.instance_decoder.aspp, run_model=lambda model: model(sample_x), device=None
            )
            if hasattr(parameters.instance_decoder, "aspp"):
                parameters.instance_decoder.aspp.conv_args = aspp_args

            # Res3
            aspp_out = torch_model.instance_decoder.aspp(sample_x)
            res3_args = infer_ttnn_module_args(
                model=torch_model.instance_decoder.res3,
                run_model=lambda model: model(aspp_out, sample_res3),
                device=None,
            )
            if hasattr(parameters.instance_decoder, "res3"):
                parameters.instance_decoder.res3.conv_args = res3_args

            # Res2
            res3_out = torch_model.instance_decoder.res3(aspp_out, sample_res3)
            res2_args = infer_ttnn_module_args(
                model=torch_model.instance_decoder.res2,
                run_model=lambda model: model(res3_out, sample_res2),
                device=None,
            )
            if hasattr(parameters.instance_decoder, "res2"):
                parameters.instance_decoder.res2.conv_args = res2_args

            # Head
            res2_out = torch_model.instance_decoder.res2(res3_out, sample_res2)
            head_args_1 = infer_ttnn_module_args(
                model=torch_model.instance_decoder.head_1, run_model=lambda model: model(res2_out), device=None
            )
            head_args_2 = infer_ttnn_module_args(
                model=torch_model.instance_decoder.head_2, run_model=lambda model: model(res2_out), device=None
            )
            if hasattr(parameters.instance_decoder, "head_1"):
                parameters.instance_decoder.head_1.conv_args = head_args_1
            if hasattr(parameters.instance_decoder, "head_2"):
                parameters.instance_decoder.head_2.conv_args = head_args_2

        # Run torch model with bfloat16
        logger.info("Running PyTorch model...")
        self.torch_output_tensor, self.torch_output_tensor_2, self.torch_output_tensor_3 = torch_model(
            self.torch_input_tensor
        )
        torch_model.to(torch.bfloat16)
        self.torch_input_tensor = self.torch_input_tensor.to(torch.bfloat16)

        # Convert input to TTNN format (NHWC)
        logger.info("Converting input to TTNN format...")
        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        # Initialize TTNN model
        logger.info("Initializing TTNN model...")
        print("Initializing TTNN model...")
        self.ttnn_model = TTPanopticDeepLab(
            parameters=parameters,
            model_config=model_config,
        )

        logger.info("Running first TTNN model pass (JIT configuration)...")
        # first run configures convs JIT
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.run()
        self.validate()

        logger.info("Running optimized TTNN model pass...")
        # Optimized run
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
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
        self.output_tensor, self.output_tensor_2, self.output_tensor_3 = self.ttnn_model(self.input_tensor, self.device)
        return self.output_tensor, self.output_tensor_2, self.output_tensor_3

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        batch_size = output_tensor.shape[0]

        valid_pcc = 0.97
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)
        assert self.pcc_passed, logger.error(f"Semantic Segmentation Head PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Semantic Segmentation Head: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}, shape={self.output_tensor.shape}"
        )

        # Validate instance segmentation head outputs
        output_tensor = self.output_tensor_2
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor_2.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        batch_size = output_tensor.shape[0]

        valid_pcc = 0.97
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor_2, output_tensor, pcc=valid_pcc)
        assert self.pcc_passed, logger.error(f"Instance Segmentation Head PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Instance Segmentation Offset Head: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}, shape={self.output_tensor_2.shape}"
        )

        output_tensor = self.output_tensor_3
        output_tensor = ttnn.to_torch(output_tensor, device=self.device, mesh_composer=self.output_mesh_composer)
        expected_shape = self.torch_output_tensor_3.shape
        output_tensor = torch.reshape(
            output_tensor, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

        batch_size = output_tensor.shape[0]

        valid_pcc = 0.97
        self.pcc_passed, self.pcc_message = check_with_pcc(self.torch_output_tensor_3, output_tensor, pcc=valid_pcc)
        assert self.pcc_passed, logger.error(f"Instance Segmentation Head 2 PCC check failed: {self.pcc_message}")
        logger.info(
            f"Panoptic DeepLab - Instance Segmentation Center Head: batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}, shape={self.output_tensor_3.shape}"
        )

        return self.pcc_passed, self.pcc_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (1, 3, 512, 1024),
    ],
)
def test_panoptic_deeplab(
    device,
    batch_size,
    in_channels,
    height,
    width,
):
    PanopticDeepLabTestInfra(
        device,
        batch_size,
        in_channels,
        height,
        width,
        model_config,
    )
