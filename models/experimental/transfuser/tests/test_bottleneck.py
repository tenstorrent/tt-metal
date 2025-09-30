import torch
import ttnn
from models.experimental.transfuser.reference.bottleneck import Bottleneck as PyTorchBottleneck
from models.experimental.transfuser.tt.ttn_bottleneck import TTNNBottleneck


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


def test_regnet_bottleneck_pcc():
    """Test RegNet bottleneck with PCC assertion."""
    device = ttnn.open_device(device_id=0)

    try:
        # Create PyTorch model for reference
        torch_model = PyTorchBottleneck(in_chs=64, out_chs=256, stride=2, group_size=8)
        torch_model.eval()

        # Create test input
        torch_input = torch.randn(1, 64, 56, 56)

        # PyTorch forward pass
        with torch.no_grad():
            torch_output = torch_model(torch_input)

        # Preprocess parameters for TTNN
        params = preprocess_parameters_for_ttnn(torch_model, device)

        # Create TTNN model
        ttnn_model = TTNNBottleneck(device=device, **params, in_chs=64, out_chs=256, stride=2, group_size=8)

        # Convert input to TTNN format
        ttnn_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

        # TTNN forward pass
        ttnn_output = ttnn_model(ttnn_input)

        # Convert back to torch for comparison
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        # PCC assertion
        passes_pcc, pcc_value = comp_pcc(torch_output, ttnn_output_torch, pcc=0.99)

        print(f"PyTorch output shape: {torch_output.shape}")
        print(f"TTNN output shape: {ttnn_output_torch.shape}")
        print(f"PCC value: {pcc_value:.6f}")
        print(f"PCC test {'PASSED' if passes_pcc else 'FAILED'}")

        assert passes_pcc, f"PCC test failed: {pcc_value:.6f} < 0.99"
        assert torch_output.shape == ttnn_output_torch.shape, "Output shapes don't match"

        print("✓ RegNet bottleneck TTNN implementation matches PyTorch with PCC > 0.99")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_regnet_bottleneck_pcc()
