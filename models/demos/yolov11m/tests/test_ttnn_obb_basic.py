# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.yolov11m.common import YOLOV11_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov11m.reference import yolov11
from models.demos.yolov11m.tt import ttnn_yolov11
from models.demos.yolov11m.tt.model_preprocessing import create_yolov11_input_tensors, create_yolov11_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "resolution",
    [
        ([1, 3, 640, 640]),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weights",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
def test_yolov11_obb(device, reset_seeds, resolution, use_pretrained_weights, model_location_generator, min_channels=8):
    """
    Test TTNN YOLOv11 OBB implementation against PyTorch reference.
    
    This test verifies that:
    1. The TTNN OBB model loads correctly with OBB weights
    2. Output shape matches expected OBB format: [1, 20, 8400] 
       where 20 = 4(box coords) + 15(classes) + 1(angle)
    3. Output values have reasonable correlation with PyTorch reference
    """
    print("🚀 Testing TTNN YOLOv11 OBB Implementation...")
    
    # Load PyTorch OBB model
    torch_model = yolov11.YoloV11()
    torch_model.eval()
    
    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)
        print("✅ Loaded OBB pretrained weights")
    
    # Create input tensors
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=resolution[0],
        input_channels=resolution[1],
        input_height=resolution[2],
        input_width=resolution[3],
        is_sub_module=False,
    )
    
    # Setup TTNN input with proper sharding
    n, c, h, w = ttnn_input.shape
    if c == 3:  # for sharding config of padded input
        c = min_channels
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input = ttnn_input.to(device, input_mem_config)
    
    print(f"📥 Input shape: {torch_input.shape}")
    
    # Run PyTorch model
    print("🟦 Running PyTorch OBB model...")
    torch_output = torch_model(torch_input)
    print(f"📤 PyTorch output shape: {torch_output.shape}")
    
    # Verify PyTorch output has expected OBB shape
    expected_shape = (1, 20, 8400)  # 20 = 4(box) + 15(classes) + 1(angle)
    assert torch_output.shape == expected_shape, f"Expected PyTorch output shape {expected_shape}, got {torch_output.shape}"
    
    # Create TTNN model parameters
    print("⚙️  Creating TTNN model parameters...")
    parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
    
    # Initialize TTNN OBB model
    print("🔧 Initializing TTNN OBB model...")
    ttnn_model = ttnn_yolov11.TtnnYoloV11(device, parameters)
    
    # Run TTNN model
    print("🟩 Running TTNN OBB model...")
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    print(f"📤 TTNN output shape: {ttnn_output.shape}")
    
    # Verify TTNN output has expected OBB shape
    assert ttnn_output.shape == expected_shape, f"Expected TTNN output shape {expected_shape}, got {ttnn_output.shape}"
    
    # Analyze output components
    print("🔍 Analyzing OBB output components...")
    
    # PyTorch analysis
    torch_box_coords = torch_output[:, :4, :]      # Box coordinates
    torch_class_preds = torch_output[:, 4:19, :]   # Class predictions (15 classes)
    torch_angle_preds = torch_output[:, 19:20, :]  # Angle predictions
    
    print(f"   PyTorch - Box coords range: [{torch_box_coords.min():.3f}, {torch_box_coords.max():.3f}]")
    print(f"   PyTorch - Class preds range: [{torch_class_preds.min():.3f}, {torch_class_preds.max():.3f}]")
    print(f"   PyTorch - Angle preds range: [{torch_angle_preds.min():.3f}, {torch_angle_preds.max():.3f}]")
    
    # TTNN analysis
    ttnn_box_coords = ttnn_output[:, :4, :]      # Box coordinates
    ttnn_class_preds = ttnn_output[:, 4:19, :]   # Class predictions (15 classes)
    ttnn_angle_preds = ttnn_output[:, 19:20, :]  # Angle predictions
    
    print(f"   TTNN - Box coords range: [{ttnn_box_coords.min():.3f}, {ttnn_box_coords.max():.3f}]")
    print(f"   TTNN - Class preds range: [{ttnn_class_preds.min():.3f}, {ttnn_class_preds.max():.3f}]")
    print(f"   TTNN - Angle preds range: [{ttnn_angle_preds.min():.3f}, {ttnn_angle_preds.max():.3f}]")
    
    # Check for reasonable values
    assert torch.isfinite(ttnn_output).all(), "TTNN output contains non-finite values"
    assert not torch.isnan(ttnn_output).any(), "TTNN output contains NaN values"
    
    # Check that class predictions are probabilities (0-1 range)
    assert (ttnn_class_preds >= 0).all() and (ttnn_class_preds <= 1).all(), "TTNN class predictions should be between 0 and 1"
    
    # Compare outputs with PCC (Pearson Correlation Coefficient)
    print("📊 Comparing PyTorch vs TTNN outputs...")
    
    # Use a lower PCC threshold for initial testing since we're dealing with a complex OBB model
    min_pcc = 0.95  # Can be adjusted based on testing results
    
    try:
        assert_with_pcc(torch_output, ttnn_output, min_pcc)
        print(f"✅ TTNN OBB model passed PCC test with threshold {min_pcc}")
    except AssertionError as e:
        print(f"⚠️  PCC test failed with threshold {min_pcc}: {e}")
        print("   This might be expected for initial OBB implementation - checking component-wise...")
        
        # Try component-wise comparison for better debugging
        try:
            assert_with_pcc(torch_box_coords, ttnn_box_coords, 0.90)
            print("✅ Box coordinates passed PCC test")
        except AssertionError as e:
            print(f"❌ Box coordinates failed PCC: {e}")
            
        try:
            assert_with_pcc(torch_class_preds, ttnn_class_preds, 0.90)
            print("✅ Class predictions passed PCC test")
        except AssertionError as e:
            print(f"❌ Class predictions failed PCC: {e}")
            
        try:
            assert_with_pcc(torch_angle_preds, ttnn_angle_preds, 0.90)
            print("✅ Angle predictions passed PCC test")
        except AssertionError as e:
            print(f"❌ Angle predictions failed PCC: {e}")
        
        # Re-raise the original assertion for proper test failure
        raise
    
    print("🎉 TTNN YOLOv11 OBB test completed successfully!")


if __name__ == "__main__":
    print("🧪 Running standalone TTNN OBB test...")
    print("Note: This standalone test requires proper device setup.")
    print("Run with pytest for full test infrastructure.")
