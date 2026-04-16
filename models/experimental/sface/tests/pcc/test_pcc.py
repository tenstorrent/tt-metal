# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for SFace model."""

import pytest
import torch
import numpy as np
import ttnn

from models.experimental.sface.common import SFACE_L1_SMALL_SIZE, get_sface_onnx_path
from models.experimental.sface.reference.sface_model import load_sface_from_onnx

# Import TTNN model only when needed (skipped tests)
# from models.experimental.sface.tt.ttnn_sface import create_sface_model


def compute_pcc(output, golden):
    """Compute Pearson Correlation Coefficient."""
    output_flat = output.flatten()
    golden_flat = golden.flatten()

    if np.std(output_flat) == 0 or np.std(golden_flat) == 0:
        return 1.0 if np.allclose(output_flat, golden_flat) else 0.0

    return np.corrcoef(output_flat, golden_flat)[0, 1]


def test_sface_reference_model():
    """Test SFace PyTorch reference model (no device needed)."""
    # Load PyTorch reference model
    onnx_path = get_sface_onnx_path()
    torch_model = load_sface_from_onnx(onnx_path)
    torch_model.eval()

    # Create test input (0-255 range to match real images)
    torch.manual_seed(42)
    torch_input = torch.randint(0, 256, (1, 3, 112, 112), dtype=torch.float32)

    # PyTorch inference
    with torch.no_grad():
        torch_output = torch_model(torch_input)

    print(f"\nInput shape: {torch_input.shape}")
    print(f"Output shape: {torch_output.shape}")
    print(f"Output (first 10): {torch_output[0, :10].numpy()}")
    print(f"Output L2 norm: {torch.norm(torch_output, dim=1).item():.4f}")

    # Verify output properties
    assert torch_output.shape == (1, 128), f"Expected shape (1, 128), got {torch_output.shape}"
    assert abs(torch.norm(torch_output, dim=1).item() - 1.0) < 0.01, "Output should be L2 normalized"


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SFACE_L1_SMALL_SIZE}],
    indirect=True,
)
def test_sface_pcc(device):
    """Test SFace model PCC against PyTorch reference.

    Uses DRAM slicing for early layers (112x112, 56x56) to avoid L1 overflow.
    """
    from models.experimental.sface.tt.ttnn_sface import create_sface_model

    # Load PyTorch reference model
    onnx_path = get_sface_onnx_path()
    torch_model = load_sface_from_onnx(onnx_path)
    torch_model.eval()

    # Create TTNN model
    ttnn_model = create_sface_model(device, torch_model)

    # Create test input [B, C, H, W] for PyTorch
    # Use 0-255 range to match real image data (model has built-in preprocessing)
    batch_size = 1
    input_h, input_w = 112, 112
    torch.manual_seed(42)
    torch_input_nchw = torch.randint(0, 256, (batch_size, 3, input_h, input_w), dtype=torch.float32)

    # PyTorch inference
    with torch.no_grad():
        torch_output = torch_model(torch_input_nchw)

    # Convert to NHWC for TTNN
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).contiguous()

    # Convert to TTNN tensor
    ttnn_input = ttnn.from_torch(
        torch_input_nhwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # TTNN inference (returns TTNN tensor, fully on-device)
    ttnn_output = ttnn_model(ttnn_input)

    # Convert outputs to numpy
    torch_output_np = torch_output.detach().numpy()
    # Convert TTNN tensor to torch, then numpy
    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    ttnn_output_np = ttnn_output_torch.float().numpy()

    # Compute PCC
    pcc = compute_pcc(ttnn_output_np, torch_output_np)
    print(f"\nSFace PCC: {pcc:.6f}")
    print(f"PyTorch output (first 10): {torch_output_np[0, :10]}")
    print(f"TTNN output (first 10): {ttnn_output_np[0, :10]}")

    # Check PCC threshold
    # Note: With eps=0.001 fix for BatchNorm, we achieve ~0.95 PCC
    # Real face images typically achieve ~0.98 PCC
    # The remaining error is cumulative bfloat16 precision loss through 14 deep blocks
    assert pcc > 0.95, f"PCC {pcc:.6f} is below threshold 0.95"


def test_sface_embedding_similarity_reference():
    """Test that SFace produces similar embeddings for same face (PyTorch only)."""
    # Load PyTorch reference model
    onnx_path = get_sface_onnx_path()
    torch_model = load_sface_from_onnx(onnx_path)
    torch_model.eval()

    # Create two similar inputs (same face with small noise)
    torch.manual_seed(42)
    face1 = torch.randn(1, 3, 112, 112, dtype=torch.float32)
    face2 = face1 + torch.randn_like(face1) * 0.01  # Small noise

    # PyTorch inference
    with torch.no_grad():
        emb1 = torch_model(face1)
        emb2 = torch_model(face2)

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    print(f"\nFace1 vs Face1+noise cosine similarity: {cos_sim:.6f}")

    # Similar faces should have high similarity
    assert cos_sim > 0.9, f"Similarity {cos_sim:.4f} too low for same face with small noise"

    # Test with completely different face
    face3 = torch.randn(1, 3, 112, 112, dtype=torch.float32)
    with torch.no_grad():
        emb3 = torch_model(face3)
    cos_sim_diff = torch.nn.functional.cosine_similarity(emb1, emb3).item()
    print(f"Face1 vs different face cosine similarity: {cos_sim_diff:.6f}")

    # Different faces should have lower similarity
    assert cos_sim > cos_sim_diff, "Same face should have higher similarity than different faces"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
