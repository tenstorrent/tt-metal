#!/usr/bin/env python3
"""
Experimental test to compare PyTorch vs TTNN permute operations
and measure diversity loss. Integrated with existing test infrastructure.
"""

import torch
import ttnn
import pytest

@pytest.mark.parametrize(
    "device",
    [{"l1_small_size": 24576}],
    indirect=["device"],
)
def test_permute_diversity_loss(device):
    """Test to compare PyTorch vs TTNN permute operations"""
    print("\n🔬 [PERMUTE EXPERIMENT] Testing diversity loss in PyTorch vs TTNN operations")
    print("=" * 80)

    def create_test_tensor():
        """Create a tensor with high diversity similar to our YOLOv11 input"""
        torch.manual_seed(42)  # Reproducible results
        tensor = torch.randn(1, 16, 320, 320, dtype=torch.float32) * 4.5
        
        unique_vals = torch.unique(tensor.flatten())
        print(f"🔍 [TEST SETUP] Created tensor with {len(unique_vals)} unique values")
        print(f"    Shape: {tensor.shape}, Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        return tensor

    def test_pytorch_permute(tensor):
        """Test PyTorch permute - should preserve all values"""
        print(f"\n🧪 [PYTORCH TEST] Testing torch.permute...")
        
        before_unique = torch.unique(tensor.flatten())
        print(f"    Before: {len(before_unique)} unique values")
        
        result = tensor.permute(0, 2, 3, 1)  # NCHW → NHWC
        
        after_unique = torch.unique(result.flatten())
        loss_pct = 100 * (len(before_unique) - len(after_unique)) / len(before_unique)
        
        print(f"    After:  {len(after_unique)} unique values")
        print(f"    Shape change: {tensor.shape} → {result.shape}")
        print(f"    Diversity loss: {loss_pct:.2f}%")
        
        return result, loss_pct

    def test_ttnn_permute(tensor, device):
        """Test TTNN permute - expected to lose values"""
        print(f"\n🧪 [TTNN TEST] Testing ttnn.permute...")
        
        try:
            # Convert to TTNN tensor with float32
            ttnn_tensor = ttnn.from_torch(
                tensor, 
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device
            )
            
            before_debug = ttnn.to_torch(ttnn_tensor)
            before_unique = torch.unique(before_debug.flatten())
            print(f"    Before: {len(before_unique)} unique values")
            
            # Apply TTNN permute with memory config
            memory_config = ttnn_tensor.memory_config()
            result = ttnn.permute(ttnn_tensor, (0, 2, 3, 1), memory_config=memory_config)
            
            after_debug = ttnn.to_torch(result)
            after_unique = torch.unique(after_debug.flatten())
            loss_pct = 100 * (len(before_unique) - len(after_unique)) / len(before_unique)
            
            print(f"    After:  {len(after_unique)} unique values")
            print(f"    Shape change: {ttnn_tensor.shape} → {result.shape}")
            print(f"    Diversity loss: {loss_pct:.2f}%")
            
            return result, loss_pct
            
        except Exception as e:
            print(f"    ❌ TTNN permute failed: {e}")
            return None, 100.0

    def test_preprocessing_comparison(tensor, device):
        """Test the difference between float32 vs bfloat16 preprocessing"""
        print(f"\n🧪 [PREPROCESSING COMPARISON] Testing dtype effects...")
        
        try:
            # Test 1: float32 preprocessing
            ttnn_float32 = ttnn.from_torch(
                tensor, 
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device
            )
            float32_debug = ttnn.to_torch(ttnn_float32)
            float32_unique = torch.unique(float32_debug.flatten())
            
            # Test 2: bfloat16 preprocessing  
            ttnn_bfloat16 = ttnn.from_torch(
                tensor, 
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device
            )
            bfloat16_debug = ttnn.to_torch(ttnn_bfloat16)
            bfloat16_unique = torch.unique(bfloat16_debug.flatten())
            
            original_unique = torch.unique(tensor.flatten())
            float32_loss = 100 * (len(original_unique) - len(float32_unique)) / len(original_unique)
            bfloat16_loss = 100 * (len(original_unique) - len(bfloat16_unique)) / len(original_unique)
            
            print(f"    Original: {len(original_unique)} unique values")
            print(f"    float32:  {len(float32_unique)} unique values ({float32_loss:.2f}% loss)")
            print(f"    bfloat16: {len(bfloat16_unique)} unique values ({bfloat16_loss:.2f}% loss)")
            
            return float32_loss, bfloat16_loss
            
        except Exception as e:
            print(f"    ❌ Preprocessing test failed: {e}")
            return 100.0, 100.0

    # Run the experiments
    test_tensor = create_test_tensor()
    
    # Test PyTorch permute
    pytorch_result, pytorch_loss = test_pytorch_permute(test_tensor)
    
    # Test TTNN permute
    ttnn_result, ttnn_loss = test_ttnn_permute(test_tensor, device)
    
    # Test preprocessing effects
    float32_loss, bfloat16_loss = test_preprocessing_comparison(test_tensor, device)
    
    # Summary
    print(f"\n📊 [EXPERIMENT RESULTS]")
    print(f"    PyTorch permute diversity loss:     {pytorch_loss:.2f}%")
    if ttnn_result is not None:
        print(f"    TTNN permute diversity loss:        {ttnn_loss:.2f}%")
    print(f"    Float32 preprocessing loss:         {float32_loss:.2f}%")
    print(f"    Bfloat16 preprocessing loss:        {bfloat16_loss:.2f}%")
    
    # Validate hypotheses
    print(f"\n🔍 [HYPOTHESIS VALIDATION]")
    if pytorch_loss < 1.0:
        print(f"    ✅ PyTorch permute preserves values (loss < 1%)")
    else:
        print(f"    ❌ PyTorch permute loses values ({pytorch_loss:.2f}%)")
    
    if ttnn_result is not None and ttnn_loss > 90.0:
        print(f"    ✅ TTNN permute causes major loss (>90%)")
    elif ttnn_result is not None:
        print(f"    ❓ TTNN permute loss lower than expected ({ttnn_loss:.2f}%)")
    
    if float32_loss < 1.0 and bfloat16_loss > 90.0:
        print(f"    ✅ Float32 preprocessing preserves values, bfloat16 destroys them")
    else:
        print(f"    ❓ Preprocessing results unexpected")
    
    print(f"\n🎯 [CONCLUSION]")
    if pytorch_loss < 1.0 and ttnn_loss > 90.0:
        print(f"    HYPOTHESIS CONFIRMED: PyTorch preserves values, TTNN destroys them")
        print(f"    RECOMMENDATION: Use PyTorch permute in preprocessing")
    else:
        print(f"    HYPOTHESIS NEEDS REVISION")

if __name__ == "__main__":
    # For standalone running
    device = ttnn.open_device(device_id=0)
    try:
        test_permute_diversity_loss(device)
    finally:
        ttnn.close_device(device)
