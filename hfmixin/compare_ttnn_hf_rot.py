"""Compare HuggingFace-style rotary embeddings cos/sin matrices with tt_transformers RotarySetup2.
The comparison is done at the cos/sin matrix level, not the rotated tensors."""

import torch

import ttnn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig

from models.tt_transformers.tt.rope import RotarySetup2
from models.tt_transformers.tt.common import RopeScaling, RopeScalingType, rope_scaling_model_factory


def comp_pcc(torch_tensor, tt_tensor, pcc_threshold=0.9997):
    """Compute Pearson Correlation Coefficient"""
    torch_flat = torch_tensor.flatten().float()
    tt_flat = tt_tensor.flatten().float()

    # Compute PCC
    mean_torch = torch_flat.mean()
    mean_tt = tt_flat.mean()

    centered_torch = torch_flat - mean_torch
    centered_tt = tt_flat - mean_tt

    numerator = (centered_torch * centered_tt).sum()
    denominator = torch.sqrt((centered_torch**2).sum() * (centered_tt**2).sum())

    if denominator == 0:
        return False, 0.0

    pcc_value = numerator / denominator
    return pcc_value.item() >= pcc_threshold, pcc_value.item()


def get_hf_cos_sin(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    rope_scaling: dict | None = None,
    position_ids: torch.Tensor | None = None,
):
    """Generate cos/sin matrices using HuggingFace's LlamaRotaryEmbedding.

    Args:
        head_dim: Head dimension
        max_seq_len: Maximum sequence length
        rope_theta: RoPE theta parameter
        rope_scaling: Optional rope scaling parameters dict
        position_ids: Optional position IDs to extract specific positions

    Returns:
        cos, sin tensors in HF format: [batch, seq_len, head_dim]
    """
    # Create HF config
    hf_config = LlamaConfig(
        hidden_size=head_dim,  # Not used for RoPE, but required
        num_attention_heads=1,  # Not used for RoPE, but required
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        max_position_embeddings=max_seq_len * 2,
    )

    # Initialize HF rotary embedding
    hf_rotary_emb = LlamaRotaryEmbedding(hf_config)

    if position_ids is not None:
        # Get cos/sin for specific positions
        # position_ids shape: [batch] or [batch, seq_len]
        if len(position_ids.shape) == 1:
            position_ids = position_ids.unsqueeze(1)  # [batch, 1]

        batch_size = position_ids.shape[0]
        seq_len = position_ids.shape[1]

        # Create dummy input with correct shape
        dummy_input = torch.zeros(batch_size, seq_len, head_dim)

        # HF's forward computes cos/sin on-the-fly based on position_ids
        cos, sin = hf_rotary_emb(dummy_input, position_ids)
        # cos, sin shape: [batch, seq_len, head_dim]

        return cos, sin
    else:
        # Get cos/sin for all positions up to max_seq_len
        dummy_input = torch.zeros(1, max_seq_len, head_dim)
        dummy_position_ids = torch.arange(max_seq_len).unsqueeze(0)  # [1, max_seq_len]

        cos, sin = hf_rotary_emb(dummy_input, dummy_position_ids)
        # cos, sin shape: [1, max_seq_len, head_dim]

        return cos, sin


def get_tt_transformers_cos_sin(
    device: ttnn.Device,
    batch_size: int,
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    rope_scaling: RopeScaling = None,
    position_ids: torch.Tensor = None,
):
    """Generate cos/sin matrices using tt_transformers' RotarySetup2.

    Args:
        device: TTNN device
        batch_size: Batch size
        head_dim: Head dimension
        max_seq_len: Maximum sequence length
        rope_theta: RoPE theta parameter
        rope_scaling: Optional RopeScaling object
        position_ids: Optional position IDs to extract specific positions

    Returns:
        cos, sin tensors in torch format: [batch, seq_len, head_dim]
    """
    # Initialize RotarySetup2
    rope_setup = RotarySetup2(
        device=device,
        batch_size=batch_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        datatype=ttnn.bfloat16,
    )

    if position_ids is not None:
        # Get cos/sin for specific positions using get_rot_mats
        # position_ids should be [batch] for decode mode
        rot_mats = rope_setup.get_rot_mats(position_ids)
        cos_tt, sin_tt = rot_mats

        # Convert from TTNN tensors to torch
        # These are sharded, so we need to gather them properly
        # For multi-device, we might need to use mesh composer
        try:
            if isinstance(device, ttnn._ttnn.multi_device.MeshDevice):
                # Multi-device: use mesh composer to gather
                cos_torch = ttnn.to_torch(
                    cos_tt,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(1, 3), mesh_shape=list(device.shape)),
                )
                sin_torch = ttnn.to_torch(
                    sin_tt,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(1, 3), mesh_shape=list(device.shape)),
                )
            else:
                cos_torch = ttnn.to_torch(cos_tt)
                sin_torch = ttnn.to_torch(sin_tt)
        except Exception:
            # Fallback: try without mesh composer
            cos_torch = ttnn.to_torch(cos_tt)
            sin_torch = ttnn.to_torch(sin_tt)

        # Handle sharded format: typically [1, 1, batch, head_dim] or [1, batch, 1, head_dim]
        # Reshape to [batch, seq_len, head_dim]
        # Remove leading 1s
        while len(cos_torch.shape) > 2 and cos_torch.shape[0] == 1:
            cos_torch = cos_torch.squeeze(0)
            sin_torch = sin_torch.squeeze(0)

        # Now should be [1, batch, head_dim] or [batch, 1, head_dim] or [batch, head_dim]
        if len(cos_torch.shape) == 3:
            if cos_torch.shape[0] == 1:
                # [1, batch, head_dim] -> [batch, 1, head_dim]
                cos_torch = cos_torch.transpose(0, 1)
                sin_torch = sin_torch.transpose(0, 1)
            # Now should be [batch, 1, head_dim] for decode
            if cos_torch.shape[1] != 1:
                # If not, might need to add seq_len dimension
                cos_torch = cos_torch.unsqueeze(1)
                sin_torch = sin_torch.unsqueeze(1)
        elif len(cos_torch.shape) == 2:
            # [batch, head_dim] -> [batch, 1, head_dim]
            cos_torch = cos_torch.unsqueeze(1)
            sin_torch = sin_torch.unsqueeze(1)

        return cos_torch, sin_torch
    else:
        # Get the full cos/sin matrices from the setup
        # These are stored in cos_matrix and sin_matrix
        cos_matrix = rope_setup.cos_matrix
        sin_matrix = rope_setup.sin_matrix

        # Convert to torch
        try:
            if isinstance(device, ttnn._ttnn.multi_device.MeshDevice):
                cos_torch = ttnn.to_torch(
                    cos_matrix,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(1, 3), mesh_shape=list(device.shape)),
                )
                sin_torch = ttnn.to_torch(
                    sin_matrix,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(1, 3), mesh_shape=list(device.shape)),
                )
            else:
                cos_torch = ttnn.to_torch(cos_matrix)
                sin_torch = ttnn.to_torch(sin_matrix)
        except Exception:
            # Fallback: try without mesh composer
            cos_torch = ttnn.to_torch(cos_matrix)
            sin_torch = ttnn.to_torch(sin_matrix)

        # Shape should be [1, 1, max_seq_len, head_dim]
        # Remove leading 1s and get [max_seq_len, head_dim]
        while len(cos_torch.shape) > 2 and cos_torch.shape[0] == 1:
            cos_torch = cos_torch.squeeze(0)
            sin_torch = sin_torch.squeeze(0)

        # Add batch dimension for consistency: [1, max_seq_len, head_dim]
        if len(cos_torch.shape) == 2:
            cos_torch = cos_torch.unsqueeze(0)
            sin_torch = sin_torch.unsqueeze(0)

        return cos_torch, sin_torch


def compare_full_matrices(
    device: ttnn.Device,
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    rope_scaling: dict | None = None,
    pcc_threshold: float = 0.9997,
):
    """Compare full cos/sin matrices between HF and tt_transformers."""
    print(f"\n{'='*60}")
    print(f"Comparing FULL cos/sin matrices")
    print(f"{'='*60}")
    print(f"head_dim={head_dim}, max_seq_len={max_seq_len}, rope_theta={rope_theta}")
    if rope_scaling:
        print(f"rope_scaling={rope_scaling}")

    # Get HF matrices
    hf_cos, hf_sin = get_hf_cos_sin(
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )
    print(f"\nHF cos shape: {hf_cos.shape}")
    print(f"HF sin shape: {hf_sin.shape}")

    # Convert rope_scaling dict to RopeScaling object if needed
    rope_scaling_obj = None
    if rope_scaling:
        # Use factory function to create the appropriate RopeScaling object
        rope_scaling_obj = rope_scaling_model_factory(
            rope_scaling,
            original_max_context_len=rope_scaling.get("original_max_position_embeddings"),
        )

    # Get tt_transformers matrices
    tt_cos, tt_sin = get_tt_transformers_cos_sin(
        device=device,
        batch_size=1,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling_obj,
    )
    print(f"TT cos shape: {tt_cos.shape}")
    print(f"TT sin shape: {tt_sin.shape}")

    # Reshape to match if needed
    # HF: [1, max_seq_len, head_dim]
    # TT: might be [1, max_seq_len, head_dim] or [max_seq_len, head_dim]
    if len(tt_cos.shape) == 2:
        tt_cos = tt_cos.unsqueeze(0)
        tt_sin = tt_sin.unsqueeze(0)

    # Take first batch item and compare
    hf_cos_flat = hf_cos[0]  # [max_seq_len, head_dim]
    hf_sin_flat = hf_sin[0]  # [max_seq_len, head_dim]
    tt_cos_flat = tt_cos[0] if len(tt_cos.shape) == 3 else tt_cos  # [max_seq_len, head_dim]
    tt_sin_flat = tt_sin[0] if len(tt_sin.shape) == 3 else tt_sin  # [max_seq_len, head_dim]

    # Compare
    cos_pass, cos_pcc = comp_pcc(hf_cos_flat, tt_cos_flat, pcc_threshold)
    sin_pass, sin_pcc = comp_pcc(hf_sin_flat, tt_sin_flat, pcc_threshold)

    print(f"\nFull Matrix Comparison:")
    print(f"  COS PCC: {cos_pcc:.6f}, Pass: {cos_pass}")
    print(f"  SIN PCC: {sin_pcc:.6f}, Pass: {sin_pass}")

    # Also check max absolute difference
    cos_max_diff = (hf_cos_flat - tt_cos_flat).abs().max().item()
    sin_max_diff = (hf_sin_flat - tt_sin_flat).abs().max().item()
    print(f"  COS Max Diff: {cos_max_diff:.6e}")
    print(f"  SIN Max Diff: {sin_max_diff:.6e}")

    return cos_pass and sin_pass, cos_pcc, sin_pcc


def compare_specific_positions(
    device: ttnn.Device,
    batch_size: int,
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    rope_scaling: dict = None,
    position_ids: torch.Tensor = None,
    pcc_threshold: float = 0.9997,
):
    """Compare cos/sin matrices for specific positions (decode mode)."""
    print(f"\n{'='*60}")
    print(f"Comparing cos/sin for SPECIFIC POSITIONS (decode mode)")
    print(f"{'='*60}")
    print(f"batch_size={batch_size}, head_dim={head_dim}, max_seq_len={max_seq_len}")
    print(f"rope_theta={rope_theta}")
    if position_ids is not None:
        print(f"position_ids={position_ids.tolist()}")

    # Get HF matrices for specific positions
    hf_cos, hf_sin = get_hf_cos_sin(
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        position_ids=position_ids,
    )
    print(f"\nHF cos shape: {hf_cos.shape}")
    print(f"HF sin shape: {hf_sin.shape}")

    # Convert rope_scaling dict to RopeScaling object if needed
    rope_scaling_obj = None
    if rope_scaling:
        # Use factory function to create the appropriate RopeScaling object
        rope_scaling_obj = rope_scaling_model_factory(
            rope_scaling,
            original_max_context_len=rope_scaling.get("original_max_position_embeddings"),
        )

    # Get tt_transformers matrices for specific positions
    tt_cos, tt_sin = get_tt_transformers_cos_sin(
        device=device,
        batch_size=batch_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling_obj,
        position_ids=position_ids,
    )
    print(f"TT cos shape: {tt_cos.shape}")
    print(f"TT sin shape: {tt_sin.shape}")

    # Reshape to match
    # Both should be [batch, seq_len, head_dim] or [batch, 1, head_dim]
    # Flatten for comparison
    hf_cos_flat = hf_cos.flatten()
    hf_sin_flat = hf_sin.flatten()
    tt_cos_flat = tt_cos.flatten()
    tt_sin_flat = tt_sin.flatten()

    # Compare
    cos_pass, cos_pcc = comp_pcc(hf_cos_flat, tt_cos_flat, pcc_threshold)
    sin_pass, sin_pcc = comp_pcc(hf_sin_flat, tt_sin_flat, pcc_threshold)

    print(f"\nPosition-Specific Comparison:")
    print(f"  COS PCC: {cos_pcc:.6f}, Pass: {cos_pass}")
    print(f"  SIN PCC: {sin_pcc:.6f}, Pass: {sin_pass}")

    # Also check max absolute difference
    cos_max_diff = (hf_cos_flat - tt_cos_flat).abs().max().item()
    sin_max_diff = (hf_sin_flat - tt_sin_flat).abs().max().item()
    print(f"  COS Max Diff: {cos_max_diff:.6e}")
    print(f"  SIN Max Diff: {sin_max_diff:.6e}")

    return cos_pass and sin_pass, cos_pcc, sin_pcc


if __name__ == "__main__":
    # Test parameters
    head_dim = 128
    max_seq_len = 512
    rope_theta = 500000.0
    batch_size = 4

    # Initialize device
    device_id = 1
    device = ttnn.open_device(device_id=device_id)

    try:
        # Test 1: Compare full matrices (no rope scaling)
        print("\n" + "=" * 60)
        print("TEST 1: Full matrices comparison (no rope scaling)")
        print("=" * 60)
        pass1, cos_pcc1, sin_pcc1 = compare_full_matrices(
            device=device,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rope_scaling=None,
        )

        assert pass1, f"Full matrix comparison failed: COS PCC={cos_pcc1:.6f}, SIN PCC={sin_pcc1:.6f}"
        print("✓ TEST 1 PASSED")

        # Test 2: Compare specific positions (decode mode)
        print("\n" + "=" * 60)
        print("TEST 2: Specific positions comparison (decode mode)")
        print("=" * 60)
        position_ids = torch.tensor([0, 1, 2, 3])  # Different positions for each batch item
        pass2, cos_pcc2, sin_pcc2 = compare_specific_positions(
            device=device,
            batch_size=batch_size,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rope_scaling=None,
            position_ids=position_ids,
        )

        assert pass2, f"Position-specific comparison failed: COS PCC={cos_pcc2:.6f}, SIN PCC={sin_pcc2:.6f}"
        print("✓ TEST 2 PASSED")

        # Test 3: Compare with rope scaling (if applicable)
        print("\n" + "=" * 60)
        print("TEST 3: Full matrices with rope scaling")
        print("=" * 60)
        rope_scaling = {
            "type": "llama3",
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        }
        pass3, cos_pcc3, sin_pcc3 = compare_full_matrices(
            device=device,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )

        assert pass3, f"Rope scaling comparison failed: COS PCC={cos_pcc3:.6f}, SIN PCC={sin_pcc3:.6f}"
        print("✓ TEST 3 PASSED")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!!")
        print("=" * 60)

    finally:
        ttnn.close_device(device)
