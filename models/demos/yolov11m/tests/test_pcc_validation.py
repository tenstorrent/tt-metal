"""
Comprehensive PCC (Pearson Correlation Coefficient) Validation for YOLOv11m

This test runs PyTorch and TTNN models in parallel and validates PCC at each major step
to ensure model accuracy throughout the pipeline.
"""

import os
import sys
import torch
import ttnn
import pytest
from loguru import logger

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import YOLOv11 models and utilities
from models.demos.yolov11m.reference.yolov11 import YoloV11
from models.demos.yolov11m.tt.ttnn_yolov11 import TtnnYoloV11
from models.demos.yolov11m.tt.ttnn_yolov11_obb import TtnnOBB
from models.demos.yolov11m.tt.model_preprocessing import create_yolov11_input_tensors, create_yolov11_model_parameters
from models.demos.yolov11m.tt.pcc_validation import PCCValidator, validate_pcc, get_pcc_summary, reset_pcc_validation


class YOLOv11PCCValidator:
    """Complete PCC validation system for YOLOv11m model."""
    
    def __init__(self, device, model_path: str = "yolov11m.pt", image_size: int = 320):
        """
        Initialize PCC validator with PyTorch and TTNN models.
        
        Args:
            device: TTNN device
            model_path: Path to PyTorch model
            image_size: Input image size
        """
        self.device = device
        self.image_size = image_size
        self.pcc_validator = PCCValidator(min_pcc_threshold=0.95)
        
        # Load PyTorch model
        print("🔄 Loading PyTorch reference model...")
        self.pytorch_model = YoloV11(model_path)
        self.pytorch_model.eval()
        
        # Create TTNN model
        print("🔄 Creating TTNN model...")
        # Create dummy input for parameter extraction
        dummy_input = torch.randn(1, 3, image_size, image_size)
        
        # Create TTNN model parameters
        model_parameters = create_yolov11_model_parameters(self.pytorch_model, dummy_input, device)
        
        # Initialize TTNN models
        self.ttnn_backbone = TtnnYoloV11(device, model_parameters)
        self.ttnn_obb = TtnnOBB(device, model_parameters)
        
        print("✅ Models initialized successfully!")
        
    def validate_preprocessing(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Validate preprocessing step.
        
        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            
        Returns:
            torch.Tensor: TTNN input tensor converted back to PyTorch
        """
        print("\n📊 === PREPROCESSING VALIDATION ===")
        
        # Create TTNN input tensors
        torch_input_tensor, ttnn_input_tensor = create_yolov11_input_tensors(
            input_tensor, self.device
        )
        
        # Validate preprocessing
        ttnn_torch = ttnn.to_torch(ttnn_input_tensor)
        passed, pcc = validate_pcc(
            torch_input_tensor, ttnn_torch, 
            "Preprocessing", show_stats=True
        )
        
        return ttnn_input_tensor
    
    def validate_backbone_step_by_step(self, ttnn_input: ttnn.Tensor, pytorch_input: torch.Tensor):
        """
        Validate backbone step by step.
        
        Args:
            ttnn_input: TTNN input tensor
            pytorch_input: PyTorch input tensor
        """
        print("\n📊 === BACKBONE STEP-BY-STEP VALIDATION ===")
        
        # We'll need to modify the backbone to expose intermediate outputs
        # For now, we'll just validate the final output
        
        # Run PyTorch backbone (extract backbone layers)
        print("🔄 Running PyTorch backbone...")
        with torch.no_grad():
            # Access PyTorch model backbone layers
            pytorch_backbone = self.pytorch_model.model[:10]  # First 10 layers are typically backbone
            pytorch_output = pytorch_backbone(pytorch_input)
        
        # Run TTNN backbone
        print("🔄 Running TTNN backbone...")
        ttnn_output = self.ttnn_backbone(ttnn_input)
        
        # Validate backbone output
        ttnn_output_torch = ttnn.to_torch(ttnn_output)
        passed, pcc = validate_pcc(
            pytorch_output, ttnn_output_torch, 
            "Backbone_Final", show_stats=True
        )
        
        return ttnn_output, pytorch_output
    
    def validate_detection_head(self, ttnn_backbone_output: ttnn.Tensor, pytorch_backbone_output: torch.Tensor):
        """
        Validate detection head.
        
        Args:
            ttnn_backbone_output: Output from TTNN backbone
            pytorch_backbone_output: Output from PyTorch backbone
        """
        print("\n📊 === DETECTION HEAD VALIDATION ===")
        
        # Run PyTorch detection head (OBB)
        print("🔄 Running PyTorch detection head...")
        with torch.no_grad():
            # Access PyTorch OBB head - this might need adjustment based on model structure
            pytorch_obb_output = self.pytorch_model.model[10:](pytorch_backbone_output)  # Remaining layers
        
        # Run TTNN detection head
        print("🔄 Running TTNN detection head...")
        ttnn_obb_output = self.ttnn_obb(ttnn_backbone_output)
        
        # Validate detection head output
        ttnn_obb_torch = ttnn.to_torch(ttnn_obb_output)
        passed, pcc = validate_pcc(
            pytorch_obb_output, ttnn_obb_torch, 
            "Detection_Head", show_stats=True
        )
        
        return ttnn_obb_output, pytorch_obb_output
    
    def validate_full_pipeline(self, input_tensor: torch.Tensor):
        """
        Validate the complete model pipeline.
        
        Args:
            input_tensor: Input image tensor [1, 3, H, W]
        """
        print("\n🚀 === FULL PIPELINE PCC VALIDATION ===")
        reset_pcc_validation()
        
        # Step 1: Validate preprocessing
        ttnn_input = self.validate_preprocessing(input_tensor)
        
        # Step 2: Validate backbone
        ttnn_backbone_out, pytorch_backbone_out = self.validate_backbone_step_by_step(
            ttnn_input, input_tensor
        )
        
        # Step 3: Validate detection head
        ttnn_final_out, pytorch_final_out = self.validate_detection_head(
            ttnn_backbone_out, pytorch_backbone_out
        )
        
        # Step 4: Final end-to-end validation
        print("\n📊 === END-TO-END VALIDATION ===")
        
        # Run complete PyTorch model
        with torch.no_grad():
            pytorch_e2e_output = self.pytorch_model(input_tensor)
        
        # Validate final outputs
        ttnn_final_torch = ttnn.to_torch(ttnn_final_out)
        passed, pcc = validate_pcc(
            pytorch_e2e_output, ttnn_final_torch, 
            "End_to_End", show_stats=True
        )
        
        # Print comprehensive summary
        print("\n" + "="*60)
        print(get_pcc_summary())
        print("="*60)
        
        return pytorch_e2e_output, ttnn_final_torch


def test_pcc_validation_with_random_input():
    """Test PCC validation with a random input."""
    # Initialize device (you may need to adjust this based on your setup)
    device = ttnn.open_device(device_id=0)
    
    try:
        # Create validator
        validator = YOLOv11PCCValidator(device, image_size=320)
        
        # Create random input
        input_tensor = torch.randn(1, 3, 320, 320)
        
        # Run full pipeline validation
        pytorch_output, ttnn_output = validator.validate_full_pipeline(input_tensor)
        
        print("✅ PCC validation completed successfully!")
        
    finally:
        ttnn.close_device(device)


def test_pcc_validation_with_real_image():
    """Test PCC validation with a real image."""
    import torchvision.transforms as transforms
    from PIL import Image
    
    # Initialize device
    device = ttnn.open_device(device_id=0)
    
    try:
        # Create validator
        validator = YOLOv11PCCValidator(device, image_size=320)
        
        # Create a simple test image (you can replace with real image loading)
        # For demo purposes, create a synthetic image
        test_image = torch.zeros(1, 3, 320, 320)
        test_image[:, :, 100:220, 100:220] = 1.0  # White square
        test_image[:, 0, 150:170, 150:170] = 0.5   # Red patch
        
        # Run full pipeline validation
        pytorch_output, ttnn_output = validator.validate_full_pipeline(test_image)
        
        print("✅ Real image PCC validation completed!")
        
        # Additional analysis
        print(f"\n📈 Final Output Analysis:")
        print(f"   PyTorch shape: {pytorch_output.shape}")
        print(f"   TTNN shape: {ttnn_output.shape}")
        print(f"   Max difference: {torch.max(torch.abs(pytorch_output - ttnn_output)).item():.6f}")
        
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    """
    Run PCC validation tests.
    
    Usage:
        python test_pcc_validation.py
    """
    print("🎯 YOLOv11m PCC Validation Test Suite")
    print("=====================================")
    
    try:
        # Test with random input
        print("\n🎲 Testing with random input...")
        test_pcc_validation_with_random_input()
        
        # Test with synthetic image
        print("\n🖼️  Testing with synthetic image...")
        test_pcc_validation_with_real_image()
        
        print("\n🎉 All PCC validation tests completed!")
        
    except Exception as e:
        logger.error(f"PCC validation test failed: {e}")
        import traceback
        traceback.print_exc()


# Additional utility functions
def quick_pcc_check(pytorch_tensor: torch.Tensor, ttnn_tensor: ttnn.Tensor, name: str = "Tensor"):
    """Quick PCC check utility function."""
    passed, pcc = validate_pcc(pytorch_tensor, ttnn_tensor, name, show_stats=False)
    return pcc


def interpret_pcc_score(pcc: float) -> str:
    """Interpret PCC score and provide recommendations."""
    if pcc >= 0.999:
        return "🟢 EXCELLENT - Nearly identical outputs"
    elif pcc >= 0.99:
        return "🟢 VERY GOOD - Minimal differences, acceptable for production"
    elif pcc >= 0.95:
        return "🟡 GOOD - Some differences, investigate if critical"
    elif pcc >= 0.90:
        return "🟠 ACCEPTABLE - Noticeable differences, review implementation"
    elif pcc >= 0.80:
        return "🔴 POOR - Significant differences, needs debugging"
    else:
        return "🔴 CRITICAL - Major implementation issues, requires immediate attention"
