# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import tracy
from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.panoptic_deeplab.reference.resnet52_backbone import ResNet52BackBone as TorchBackbone
from models.experimental.panoptic_deeplab.tt.backbone import TTBackbone
from models.experimental.panoptic_deeplab.tt.custom_preprocessing import create_custom_mesh_preprocessor

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


class BackboneTestInfra:
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
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "call validate()?"
        self.device = device
        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # torch model
        torch_model = TorchBackbone().eval()

        input_shape = (batch_size * self.num_devices, in_channels, height, width)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # golden
        torch_model.to(torch.bfloat16)
        try:
            self.torch_input_tensor = torch.load(f"backbone_{input_shape}_input_tensor.pt")
            self.torch_output_tensor = torch.load(f"backbone_{input_shape}_output_tensor.pt")
        except:
            self.torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
            self.torch_input_tensor = self.torch_input_tensor.to(torch.bfloat16)
            self.torch_output_tensor = torch_model(self.torch_input_tensor)
            torch.save(self.torch_input_tensor, f"backbone_{input_shape}_input_tensor.pt")
            torch.save(self.torch_output_tensor, f"backbone_{input_shape}_output_tensor.pt")

        tt_host_tensor = ttnn.from_torch(
            self.torch_input_tensor.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        self.ttnn_model = TTBackbone(
            parameters=parameters,
            model_config=model_config,
            small_tensor=width < 2048,
        )

        # First run configures convs JIT
        tracy.signpost(f"Backbone_{input_shape}_compile")
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.run()
        self.validate()

        # Optimized run
        tracy.signpost(f"Backbone_{input_shape}_perf")
        self.input_tensor = ttnn.to_device(tt_host_tensor, device)
        self.run()
        self.validate()

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = None  # ttnn.ReplicateTensorToMesh(device) causes unnecessary replication/takes more time on the first pass
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def run(self):
        self.output_tensor = self.ttnn_model(
            self.input_tensor,
            self.device,
        )
        return self.output_tensor

    def validate(self, output_tensor=None):
        tt_output_tensor = self.output_tensor["res_5"] if output_tensor is None else output_tensor
        tt_output_tensor_torch = ttnn.to_torch(
            tt_output_tensor, device=self.device, mesh_composer=self.output_mesh_composer
        )
        ttnn.deallocate(tt_output_tensor)
        expected_shape = self.torch_output_tensor.shape
        tt_output_tensor_torch = torch.reshape(
            tt_output_tensor_torch, (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
        )
        tt_output_tensor_torch = torch.permute(tt_output_tensor_torch, (0, 3, 1, 2))

        batch_size = tt_output_tensor_torch.shape[0]

        valid_pcc = 0.97
        self.pcc_passed, self.pcc_message = check_with_pcc(
            self.torch_output_tensor, tt_output_tensor_torch, pcc=valid_pcc
        )

        assert self.pcc_passed, logger.error(f"PCC check failed: {self.pcc_message}")
        logger.info(
            f"ResNet52 Backbone batch_size={batch_size}, act_dtype={model_config['ACTIVATIONS_DTYPE']}, weight_dtype={model_config['WEIGHTS_DTYPE']}, math_fidelity={model_config['MATH_FIDELITY']}, PCC={self.pcc_message}"
        )

        return self.pcc_passed, self.pcc_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (1, 3, 512, 1024),
    ],
)
def test_backbone(
    device,
    batch_size,
    in_channels,
    height,
    width,
):
    BackboneTestInfra(
        device,
        batch_size,
        in_channels,
        height,
        width,
        model_config,
    )
