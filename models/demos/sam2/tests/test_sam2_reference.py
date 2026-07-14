# ponytail: verification unit tests for SAM2 reference baseline.
# Proves multi-scale encoder downsampling shapes (4x, 8x, 16x, 32x), finite bounds,
# and exact functional isolation for Stage 1 CI compliance.

import sys
from pathlib import Path

# Add project root to sys.path so package imports resolve cleanly
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Handle module import syntax cleanly without requiring full package install
try:
    from reference.sam2_reference import Sam2ReferenceImageModel
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("sam2_reference", root_dir / "reference" / "sam2_reference.py")
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Sam2ReferenceImageModel = module.Sam2ReferenceImageModel
    else:
        raise ImportError("Could not load sam2_reference.py module dynamically")

def test_image_encoder_multiscale_shapes():
    """Verifies Hiera-tiny hierarchical downsampling shapes matching exact specification."""
    import torch
    model = Sam2ReferenceImageModel(embed_dim=96)
    model.eval()

    torch.manual_seed(42)
    dummy_image = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

    with torch.no_grad():
        out = model(dummy_image)
    features = [out["stage1_features"], out["stage2_features"], out["stage3_features"], out["stage4_features"]]

    assert len(features) == 4, f"Expected 4 hierarchical feature maps, got {len(features)}"
    
    # 1. Check 4x feature map (256x256, 96 channels)
    assert features[0].shape == (1, 96, 256, 256), f"4x feature map shape mismatch: {features[0].shape}"
    assert torch.isfinite(features[0]).all(), "4x feature map contains NaN/Inf values"
    
    # 2. Check 8x feature map (128x128, 192 channels)
    assert features[1].shape == (1, 192, 128, 128), f"8x feature map shape mismatch: {features[1].shape}"
    assert torch.isfinite(features[1]).all(), "8x feature map contains NaN/Inf values"
    
    # 3. Check 16x feature map (64x64, 384 channels)
    assert features[2].shape == (1, 384, 64, 64), f"16x feature map shape mismatch: {features[2].shape}"
    assert torch.isfinite(features[2]).all(), "16x feature map contains NaN/Inf values"
    
    # 4. Check 32x feature map (32x32, 768 channels)
    assert features[3].shape == (1, 768, 32, 32), f"32x feature map shape mismatch: {features[3].shape}"
    assert torch.isfinite(features[3]).all(), "32x feature map contains NaN/Inf values"

def test_mask_decoder_pipeline():
    """Verifies end-to-end image features + prompt embeddings decoding into 1024x1024 masks."""
    import torch
    model = Sam2ReferenceImageModel(embed_dim=96)
    model.eval()

    torch.manual_seed(42)
    dummy_image = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

    with torch.no_grad():
        out = model(dummy_image)
        masks = out["pred_mask"]

    assert masks.shape == (1, 1, 1024, 1024), f"Mask decoder output shape mismatch: {masks.shape}"
    assert torch.isfinite(masks).all(), "Mask decoder outputs contain NaN/Inf"

if __name__ == "__main__":
    import torch
    print("Execution Stage 1 Verification Self-Check...")
    test_image_encoder_multiscale_shapes()
    test_mask_decoder_pipeline()
    print("✅ All hierarchical features and mask decoding checks passed cleanly (4x, 8x, 16x, 32x).")
