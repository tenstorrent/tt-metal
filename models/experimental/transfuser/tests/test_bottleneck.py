import torch
import pytest
import ttnn
from models.experimental.transfuser.reference.bottleneck import Bottleneck as PyTorchBottleneck
from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.transfuser.tt.stages import optimization_dict
from models.experimental.transfuser.tt.bottleneck import TTRegNetBottleneck
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from tests.ttnn.utils_for_testing import check_with_pcc
from loguru import logger


class TransfuserBottleneckInfra:
    def __init__(
        self,
        device,
        in_chs,
        out_chs,
        stride,
        input_size,
        stage_name,
        model_config,
    ):
        super().__init__()
        self.device = device
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.stride = stride
        self.input_size = input_size
        self.stage_name = stage_name
        self.model_config = model_config
        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # Build reference torch model
        torch_model = PyTorchBottleneck(in_chs=in_chs, out_chs=out_chs, stride=stride, group_size=24)
        torch_model.eval()

        # Create test input
        self.torch_input = torch.randn(self.input_size)
        with torch.no_grad():
            self.torch_output = torch_model(self.torch_input)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        downsample = True
        if in_chs == out_chs and stride == 1:
            downsample = False
        bottle_ratio = 1.0
        group_size = 24
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        layer_config = optimization_dict[stage_name]

        self.ttnn_model = TTRegNetBottleneck(
            parameters=parameters,
            model_config=self.model_config,
            stride=self.stride,
            downsample=downsample,
            groups=groups,
            layer_config=layer_config,
        )
        self.tt_input = ttnn.from_torch(
            self.torch_input,
            device=self.device,
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.tt_input = ttnn.permute(self.tt_input, (0, 2, 3, 1))
        # Run + validate
        self.run()
        self.validate(self.model_config)

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),
                None,
                ttnn.ConcatMeshToTensor(device, dim=0),
            )
        return None, None, None

    def run(self):
        self.tt_output, _ = self.ttnn_model(self.tt_input, self.device)
        return self.tt_output

    def validate(self, model_config):
        self.tt_torch_output = ttnn.to_torch(
            self.tt_output,
            device=self.device,
            mesh_composer=self.output_mesh_composer,
        )
        expected_image_shape = self.torch_output.shape
        self.tt_torch_output = torch.reshape(
            self.tt_torch_output,
            (expected_image_shape[0], expected_image_shape[2], expected_image_shape[3], expected_image_shape[1]),
        )
        self.tt_torch_output = torch.permute(self.tt_torch_output, (0, 3, 1, 2))
        pcc_passed, pcc_message = check_with_pcc(self.torch_output, self.tt_torch_output, pcc=0.99)

        logger.info(f"Image Output PCC: {pcc_message}")
        assert pcc_passed, logger.error(f"PCC check failed - pcc_message: {pcc_message}")

        print("RegNet bottleneck TTNN implementation matches PyTorch with PCC > 0.99")

        return pcc_passed, f"Bottleneck: {pcc_message}"


# High accuracy model config
model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    "fp32_dest_acc_en": True,
    "packer_l1_acc": True,
    "math_approx_mode": False,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "in_chs, out_chs, stride, input_size, stage_name",
    [
        (32, 72, 2, (1, 32, 80, 352), "layer1"),  # stage 1 DS
    ],
)
def test_transfuser_bottleneck(
    device,
    in_chs,
    out_chs,
    stride,
    input_size,
    stage_name,
):
    TransfuserBottleneckInfra(
        device,
        in_chs,
        out_chs,
        stride,
        input_size,
        stage_name,
        model_config,
    )
