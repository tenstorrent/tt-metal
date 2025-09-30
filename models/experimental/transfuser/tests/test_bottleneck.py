import torch
import pytest
import ttnn
from models.experimental.transfuser.reference.bottleneck import Bottleneck as PyTorchBottleneck
from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.transfuser.tt.bottleneck import TTRegNetBottleneck
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from tests.ttnn.utils_for_testing import check_with_pcc
from loguru import logger


# from models.experimental.transfuser.tt.ttn_bottleneck import TTNNBottleneck


def get_mesh_mappers(device):
    if device.get_num_devices() != 1:
        return (
            ttnn.ShardTensorToMesh(device, dim=0),
            None,
            ttnn.ConcatMeshToTensor(device, dim=0),
        )
    return None, None, None


def comp_pcc(golden, actual, pcc=0.99):
    """Compare tensors using PCC similar to codebase patterns."""
    golden_flat = golden.flatten()
    actual_flat = actual.flatten()

    correlation_matrix = torch.corrcoef(torch.stack([golden_flat, actual_flat]))
    pcc_value = correlation_matrix[0, 1].item()

    return pcc_value >= pcc, pcc_value


def preprocess_parameters_for_ttnn(torch_model, device):
    """Convert PyTorch parameters to TTNN tensors."""
    parameters = {}

    # Extract and convert all weights/biases to TTNN format
    conv1_weight = ttnn.from_torch(torch_model.conv1.conv.weight, device=device)
    conv1_bias = (
        ttnn.from_torch(torch_model.conv1.bn.bias, device=device) if torch_model.conv1.bn.bias is not None else None
    )

    conv2_weight = ttnn.from_torch(torch_model.conv2.conv.weight, device=device)
    conv2_bias = (
        ttnn.from_torch(torch_model.conv2.bn.bias, device=device) if torch_model.conv2.bn.bias is not None else None
    )

    conv3_weight = ttnn.from_torch(torch_model.conv3.conv.weight, device=device)
    conv3_bias = (
        ttnn.from_torch(torch_model.conv3.bn.bias, device=device) if torch_model.conv3.bn.bias is not None else None
    )

    # SE parameters (if exists)
    se_fc1_weight = se_fc1_bias = se_fc2_weight = se_fc2_bias = None
    if hasattr(torch_model.se, "fc1"):
        se_fc1_weight = ttnn.from_torch(torch_model.se.fc1.weight, device=device)
        se_fc1_bias = (
            ttnn.from_torch(torch_model.se.fc1.bias, device=device) if torch_model.se.fc1.bias is not None else None
        )
        se_fc2_weight = ttnn.from_torch(torch_model.se.fc2.weight, device=device)
        se_fc2_bias = (
            ttnn.from_torch(torch_model.se.fc2.bias, device=device) if torch_model.se.fc2.bias is not None else None
        )

    # Downsample parameters (if exists)
    downsample_weight = downsample_bias = None
    if torch_model.downsample is not None:
        downsample_weight = ttnn.from_torch(torch_model.downsample[0].weight, device=device)
        downsample_bias = (
            ttnn.from_torch(torch_model.downsample[1].bias, device=device)
            if torch_model.downsample[1].bias is not None
            else None
        )

    return {
        "conv1_weight": conv1_weight,
        "conv1_bias": conv1_bias,
        "conv2_weight": conv2_weight,
        "conv2_bias": conv2_bias,
        "conv3_weight": conv3_weight,
        "conv3_bias": conv3_bias,
        "se_fc1_weight": se_fc1_weight,
        "se_fc1_bias": se_fc1_bias,
        "se_fc2_weight": se_fc2_weight,
        "se_fc2_bias": se_fc2_bias,
        "downsample_weight": downsample_weight,
        "downsample_bias": downsample_bias,
    }


@pytest.mark.parametrize(
    "in_chs, out_chs, stride, input_size",
    [
        (32, 72, 2, (1, 32, 80, 352)),  # stage 1 DS
        # (72, 72, 1, (1, 72, 40, 176)),  # stage 1 NDS
    ],
)
def test_regnet_bottleneck_pcc(in_chs, out_chs, stride, input_size):
    """Test RegNet bottleneck with PCC assertion."""
    device = ttnn.open_device(device_id=0, l1_small_size=16384)

    try:
        # Create PyTorch model for reference
        torch_model = PyTorchBottleneck(in_chs=in_chs, out_chs=out_chs, stride=stride, group_size=24)
        torch_model.eval()

        # Create test input
        torch_input = torch.randn(input_size)

        # PyTorch forward pass
        with torch.no_grad():
            torch_output = torch_model(torch_input)

        inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
            device=None,
        )

        model_config = {
            "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
            "WEIGHTS_DTYPE": ttnn.bfloat8_b,
            "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
        }
        downsample = False
        if in_chs == out_chs and stride == 1:
            downsample = True

        bottle_ratio = 1.0
        group_size = 24
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        ttnn_model = TTRegNetBottleneck(
            parameters=parameters, model_config=model_config, stride=stride, downsample=downsample, groups=groups
        )
        tt_input = ttnn.from_torch(
            torch_input,
            # self.torch_image_input.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=inputs_mesh_mapper,
        )
        tt_input = ttnn.to_device(tt_input, device)
        tt_input = ttnn.permute(tt_input, (0, 2, 3, 1))
        tt_output = ttnn_model(tt_input, device)
        tt_torch_output = ttnn.to_torch(
            tt_output,
            device=device,
            mesh_composer=output_mesh_composer,
        )
        expected_image_shape = torch_output.shape
        tt_torch_output = torch.reshape(
            tt_torch_output,
            (expected_image_shape[0], expected_image_shape[2], expected_image_shape[3], expected_image_shape[1]),
        )
        tt_torch_output = torch.permute(tt_torch_output, (0, 3, 1, 2))
        pcc_passed, pcc_message = check_with_pcc(torch_output, tt_torch_output, pcc=0.99)

        logger.info(f"Image Output PCC: {pcc_message}")
        assert pcc_passed, logger.error(f"PCC check failed - pcc_message: {pcc_message}")

        print("✓ RegNet bottleneck TTNN implementation matches PyTorch with PCC > 0.99")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_regnet_bottleneck_pcc()
