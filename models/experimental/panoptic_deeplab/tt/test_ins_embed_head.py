import torch
import ttnn
import pytest

from .tt_ins_embed_head import TtPanopticDeepLabInsEmbedHead
from .tt_pytorch_semSeg import ShapeSpec
from ..reference.pytorch_ins_embed_head import PanopticDeepLabInsEmbedHead
from tests.ttnn.utils_for_testing import assert_with_pcc


def print_comparison_stats(name: str, pytorch_tensor: torch.Tensor, ttnn_tensor: torch.Tensor):
    """Print detailed comparison statistics between PyTorch and TTNN tensors."""
    print(f"\n=== {name} Comparison Stats ===")
    print(
        f"PyTorch - Shape: {pytorch_tensor.shape}, Mean: {pytorch_tensor.mean():.6f}, Std: {pytorch_tensor.std():.6f}"
    )
    print(f"TTNN    - Shape: {ttnn_tensor.shape}, Mean: {ttnn_tensor.mean():.6f}, Std: {ttnn_tensor.std():.6f}")

    # Calculate absolute and relative differences
    abs_diff = torch.abs(pytorch_tensor - ttnn_tensor)
    rel_diff = abs_diff / (torch.abs(pytorch_tensor) + 1e-8)

    print(f"Abs Diff - Mean: {abs_diff.mean():.6f}, Max: {abs_diff.max():.6f}")
    print(f"Rel Diff - Mean: {rel_diff.mean():.6f}, Max: {rel_diff.max():.6f}")


def create_input_shape_dict():
    """Create input shape dictionary for testing."""
    res2_shape = ShapeSpec()
    res2_shape.channels = 256
    res2_shape.stride = 4

    res3_shape = ShapeSpec()
    res3_shape.channels = 512
    res3_shape.stride = 8

    res5_shape = ShapeSpec()
    res5_shape.channels = 2048
    res5_shape.stride = 32

    return {
        "res2": res2_shape,
        "res3": res3_shape,
        "res5": res5_shape,
    }


def create_test_weights():
    """Create test weights for the instance embedding head."""
    weights = {}

    # ASPP shared weights
    weights["shared_weight_tensor_kernel1"] = torch.randn(256, 2048, 1, 1)
    weights["shared_weight_tensor_kernel3"] = torch.randn(256, 2048, 3, 3)
    weights["shared_weight_tensor_kernel1_output5"] = torch.randn(256, 1280, 1, 1)

    # Project conv weights
    weights["project_conv_weights"] = {
        "res2": torch.randn(32, 256, 1, 1),
        "res3": torch.randn(64, 512, 1, 1),
    }

    # Fuse conv weights
    weights["fuse_conv_0_weights"] = {
        "res2": torch.randn(128, 160, 3, 3),  # 32 + 128 = 160
        "res3": torch.randn(128, 320, 3, 3),  # 64 + 256 = 320
    }

    weights["fuse_conv_1_weights"] = {
        "res2": torch.randn(128, 128, 3, 3),
        "res3": torch.randn(128, 128, 3, 3),
    }

    # Instance embedding head specific weights
    weights["center_head_0_weight"] = torch.randn(128, 128, 3, 3)
    weights["center_head_1_weight"] = torch.randn(32, 128, 3, 3)
    weights["center_predictor_weight"] = torch.randn(1, 32, 1, 1)

    weights["offset_head_0_weight"] = torch.randn(128, 128, 3, 3)
    weights["offset_head_1_weight"] = torch.randn(32, 128, 3, 3)
    weights["offset_predictor_weight"] = torch.randn(2, 32, 1, 1)

    return weights


def create_test_features_pytorch(batch_size=1, h_res2=128, w_res2=256):
    """Create test features for PyTorch model."""
    return {
        "res2": torch.randn(batch_size, 256, h_res2, w_res2),
        "res3": torch.randn(batch_size, 512, h_res2 // 2, w_res2 // 2),
        "res5": torch.randn(batch_size, 2048, h_res2 // 8, w_res2 // 8),
    }


def create_test_features_ttnn(device, batch_size=1, h_res2=128, w_res2=256):
    """Create test features for TTNN model."""
    features_torch = create_test_features_pytorch(batch_size, h_res2, w_res2)
    features_ttnn = {}

    for key, tensor in features_torch.items():
        # Convert to NHWC format for TTNN
        tensor_nhwc = tensor.permute(0, 2, 3, 1).contiguous()
        features_ttnn[key] = ttnn.from_torch(tensor_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return features_ttnn


def test_pytorch_ins_embed_head():
    """Test PyTorch instance embedding head implementation."""
    input_shape = create_input_shape_dict()
    weights = create_test_weights()

    # Create PyTorch model
    model = PanopticDeepLabInsEmbedHead(
        input_shape=input_shape,
        head_channels=32,
        center_loss_weight=1.0,
        offset_loss_weight=1.0,
        project_channels=[32, 64],
        aspp_dilations=[6, 12, 18],
        aspp_dropout=0.1,
        decoder_channels=[128, 128, 256],
        common_stride=4,
        norm="",
        train_size=(512, 1024),
        **weights,
    )

    # Create test features
    features = create_test_features_pytorch()

    # Test forward pass
    model.eval()
    with torch.no_grad():
        (center_pred, offset_pred), _ = model(features)

    # Check output shapes
    expected_h, expected_w = 512, 1024  # After upsampling by common_stride=4
    assert center_pred.shape == (1, 1, expected_h, expected_w)
    assert offset_pred.shape == (1, 2, expected_h, expected_w)

    print("PyTorch Instance Embedding Head test passed!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_ins_embed_head(device):
    """Test TTNN instance embedding head implementation."""
    input_shape = create_input_shape_dict()
    weights = create_test_weights()

    # Create TTNN model
    model = TtPanopticDeepLabInsEmbedHead(
        input_shape=input_shape,
        device=device,
        head_channels=32,
        norm="",
        project_channels=[32, 64],
        aspp_dilations=[6, 12, 18],
        aspp_dropout=0.1,
        decoder_channels=[128, 128, 256],
        common_stride=4,
        train_size=(512, 1024),
        **weights,
    )

    # Create test features
    features = create_test_features_ttnn(device)

    # Test forward pass
    (center_pred, offset_pred), _ = model(features)

    # Convert back to torch for shape checking
    center_pred_torch = ttnn.to_torch(center_pred).squeeze(0).permute(2, 0, 1)  # CHW
    offset_pred_torch = ttnn.to_torch(offset_pred).squeeze(0).permute(2, 0, 1)  # CHW

    # Check output shapes (accounting for potential padding in TTNN)
    assert center_pred_torch.shape[0] == 1  # 1 channel for center
    assert offset_pred_torch.shape[0] == 2  # 2 channels for offset

    # Clean up
    for tensor in features.values():
        ttnn.deallocate(tensor)
    ttnn.deallocate(center_pred)
    ttnn.deallocate(offset_pred)

    print("TTNN Instance Embedding Head test passed!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_compare_pytorch_ttnn_outputs(device):
    """Compare outputs between PyTorch and TTNN implementations."""
    input_shape = create_input_shape_dict()
    weights = create_test_weights()

    # Create models
    pytorch_model = PanopticDeepLabInsEmbedHead(
        input_shape=input_shape,
        head_channels=32,
        center_loss_weight=1.0,
        offset_loss_weight=1.0,
        project_channels=[32, 64],
        aspp_dilations=[6, 12, 18],
        aspp_dropout=0.1,
        decoder_channels=[128, 128, 256],
        common_stride=4,
        norm="",
        train_size=(512, 1024),
        **weights,
    )

    ttnn_model = TtPanopticDeepLabInsEmbedHead(
        input_shape=input_shape,
        device=device,
        head_channels=32,
        norm="",
        project_channels=[32, 64],
        aspp_dilations=[6, 12, 18],
        aspp_dropout=0.1,
        decoder_channels=[128, 128, 256],
        common_stride=4,
        train_size=(512, 1024),
        **weights,
    )

    # Create test features
    features_pytorch = create_test_features_pytorch(batch_size=1, h_res2=128, w_res2=256)
    features_ttnn = create_test_features_ttnn(device, batch_size=1, h_res2=128, w_res2=256)

    # Get PyTorch outputs
    pytorch_model.eval()
    with torch.no_grad():
        (pytorch_center, pytorch_offset), _ = pytorch_model(features_pytorch)

    # Get TTNN outputs
    (ttnn_center, ttnn_offset), _ = ttnn_model(features_ttnn)

    # Convert TTNN outputs to PyTorch format for comparison
    ttnn_center_torch = ttnn.to_torch(ttnn_center).squeeze(0).permute(2, 0, 1).unsqueeze(0)  # NHWC -> NCHW
    ttnn_offset_torch = ttnn.to_torch(ttnn_offset).squeeze(0).permute(2, 0, 1).unsqueeze(0)  # NHWC -> NCHW

    # Compare shapes
    print(f"PyTorch center shape: {pytorch_center.shape}")
    print(f"TTNN center shape: {ttnn_center_torch.shape}")
    print(f"PyTorch offset shape: {pytorch_offset.shape}")
    print(f"TTNN offset shape: {ttnn_offset_torch.shape}")

    # Ensure shapes match exactly
    assert (
        pytorch_center.shape == ttnn_center_torch.shape
    ), f"Center shape mismatch: PyTorch {pytorch_center.shape} vs TTNN {ttnn_center_torch.shape}"
    assert (
        pytorch_offset.shape == ttnn_offset_torch.shape
    ), f"Offset shape mismatch: PyTorch {pytorch_offset.shape} vs TTNN {ttnn_offset_torch.shape}"

    # Print detailed comparison statistics
    print_comparison_stats("Center Prediction", pytorch_center, ttnn_center_torch)
    print_comparison_stats("Offset Prediction", pytorch_offset, ttnn_offset_torch)

    # PCC comparison for center prediction
    center_pcc_passed, center_pcc_message = assert_with_pcc(pytorch_center, ttnn_center_torch, pcc=0.95)
    print(f"\nCenter Prediction PCC: {center_pcc_message}")

    # PCC comparison for offset prediction
    offset_pcc_passed, offset_pcc_message = assert_with_pcc(pytorch_offset, ttnn_offset_torch, pcc=0.95)
    print(f"Offset Prediction PCC: {offset_pcc_message}")

    # Assert both PCC tests pass
    assert center_pcc_passed, f"Center prediction PCC check failed: {center_pcc_message}"
    assert offset_pcc_passed, f"Offset prediction PCC check failed: {offset_pcc_message}"

    # Clean up TTNN tensors
    for tensor in features_ttnn.values():
        ttnn.deallocate(tensor)
    ttnn.deallocate(ttnn_center)
    ttnn.deallocate(ttnn_offset)

    print("PyTorch vs TTNN comparison test passed with PCC validation!")


# Test runner for manual verification
if __name__ == "__main__":
    print("Running Instance Embedding Head tests...")
    print("Note: This script requires pytest for device setup. Use 'pytest test_ins_embed_head.py' instead.")
    print("Available tests:")
    print("- test_pytorch_ins_embed_head: Tests PyTorch implementation")
    print("- test_ttnn_ins_embed_head: Tests TTNN implementation")
    print("- test_compare_pytorch_ttnn_outputs: Compares PyTorch vs TTNN with PCC validation")
