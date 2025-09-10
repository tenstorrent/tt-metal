# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.panoptic_deeplab.tt.decoder import (
    TTDecoder,
    decoder_layer_optimisations,
)
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.panoptic_deeplab.reference.decoder import (
    DecoderModel,
)

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


class HeadTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_config,
        in_channels,
        res3_intermediate_channels,
        res2_intermediate_channels,
        out_channels,
        upsample_channels,
        height,
        width,
        name,
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
        self.res3_intermediate_channels = res3_intermediate_channels
        self.res2_intermediate_channels = res2_intermediate_channels
        self.out_channels = out_channels
        self.upsample_channels = upsample_channels
        self.height = height
        self.width = width
        self.name = name
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Create input tensors
        self.torch_input_tensor = torch.randn(
            (self.batch_size, self.in_channels, self.height, self.width), dtype=torch.float32
        )

        # Create res3 and res2 feature maps with appropriate dimensions
        self.torch_res3_tensor = torch.randn(
            (self.batch_size, 512, self.height * 2, self.width * 2), dtype=torch.float32
        )

        self.torch_res2_tensor = torch.randn(
            (self.batch_size, upsample_channels, self.height * 4, self.width * 4), dtype=torch.float32
        )

        # torch model
        torch_model = DecoderModel(
            self.in_channels,
            self.res3_intermediate_channels,
            self.res2_intermediate_channels,
            self.out_channels,
            self.name,
        ).eval()

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        parameters.conv_args = {}
        # For ASPP
        aspp_args = infer_ttnn_module_args(
            model=torch_model.aspp, run_model=lambda model: model(self.torch_input_tensor), device=None
        )
        if hasattr(parameters, "aspp"):
            parameters.aspp.conv_args = aspp_args

        # For res3
        res3_output = torch_model.aspp(self.torch_input_tensor)
        res3_args = infer_ttnn_module_args(
            model=torch_model.res3, run_model=lambda model: model(res3_output, self.torch_res3_tensor), device=None
        )
        if hasattr(parameters, "res3"):
            parameters.res3.conv_args = res3_args

        # For res2
        res2_input = torch_model.res3(res3_output, self.torch_res3_tensor)
        res2_args = infer_ttnn_module_args(
            model=torch_model.res2, run_model=lambda model: model(res2_input, self.torch_res2_tensor), device=None
        )
        if hasattr(parameters, "res2"):
            parameters.res2.conv_args = res2_args

        # For head
        head_input = torch_model.res2(res2_input, self.torch_res2_tensor)
        head_1_args = infer_ttnn_module_args(
            model=torch_model.head_1, run_model=lambda model: model(head_input), device=None
        )
        if hasattr(parameters, "head_1"):
            parameters.head_1.conv_args = head_1_args

        if "instance" in name:
            head_2_args = infer_ttnn_module_args(
                model=torch_model.head_2, run_model=lambda model: model(head_input), device=None
            )
            if hasattr(parameters, "head_2"):
                parameters.head_2.conv_args = head_2_args

        # Convert to bfloat16
        torch_model.to(torch.bfloat16)
        self.torch_input_tensor = self.torch_input_tensor.to(torch.bfloat16)
        self.torch_res3_tensor = self.torch_res3_tensor.to(torch.bfloat16)
        self.torch_res2_tensor = self.torch_res2_tensor.to(torch.bfloat16)

        # Get torch output with all three inputs
        self.torch_output_tensor, self.torch_output_tensor_2 = torch_model(
            self.torch_input_tensor, self.torch_res3_tensor, self.torch_res2_tensor
        )

        # Convert torch tensors to TTNN host tensors
        def to_ttnn_host(tensor):
            return ttnn.from_torch(
                tensor.permute(0, 2, 3, 1),
                dtype=ttnn.bfloat8_b,
                device=device,
                mesh_mapper=self.inputs_mesh_mapper,
            )

        tt_host_tensor = to_ttnn_host(self.torch_input_tensor)
        tt_res3_tensor = to_ttnn_host(self.torch_res3_tensor)
        tt_res2_tensor = to_ttnn_host(self.torch_res2_tensor)

        # Move TTNN host tensors to device
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.res3_tensor = ttnn.to_device(tt_res3_tensor, device)
        self.res2_tensor = ttnn.to_device(tt_res2_tensor, device)

        # ttnn model
        logger.info("Initializing TTNN model...")
        self.ttnn_model = TTDecoder(
            parameters, model_config, layer_optimisations=decoder_layer_optimisations[self.name], name=self.name
        )

        # run and validate
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

    # Compute golden output for the selected block using a helper function
    def run(self):
        self.output_tensor, self.output_tensor_2 = self.ttnn_model(
            self.input_tensor,
            self.res3_tensor,
            self.res2_tensor,
            self.upsample_channels,
            self.device,
        )

        return self.output_tensor, self.output_tensor_2

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
        assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
        logger.info(
            f"{self.name},  batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        if "instance" in self.name:
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
            assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
            logger.info(
                f"{self.name},  batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
            )

        return self.pcc_passed, self.pcc_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, res3_intermediate_channels, res2_intermediate_channels, out_channels, upsample_channels, height, width, name",
    [
        (1, 2048, 320, 288, (19,), 256, 32, 64, "Semantics_head"),  # semantic head
        (1, 2048, 320, 160, (2, 1), 256, 32, 64, "instance_head"),  # instance offset head
    ],
)
def test_decoder(
    device,
    batch_size,
    in_channels,
    res3_intermediate_channels,
    res2_intermediate_channels,
    out_channels,
    upsample_channels,
    height,
    width,
    name,
):
    HeadTestInfra(
        device,
        batch_size,
        model_config,
        in_channels,
        res3_intermediate_channels,
        res2_intermediate_channels,
        out_channels,
        upsample_channels,
        height,
        width,
        name,
    )
