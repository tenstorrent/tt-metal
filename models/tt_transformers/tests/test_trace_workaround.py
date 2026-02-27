# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for trace-compatible HF-style RoPE workaround.

This test validates a workaround for using ttnn.experimental.rotary_embedding in traced
models. The operation normally requires an integer token_idx parameter, but extracting
this from a TT tensor requires device reads which are forbidden during tracing.

The workaround: Pre-slice the cos/sin matrices to the correct position externally (on host),
placing the values for position N at index 0 of the cache. Then always pass token_idx=0
to ttnn.experimental.rotary_embedding. This allows tracing because:
1. No device reads are needed to get position information
2. token_idx=0 is a compile-time constant

Constraint from ttnn.experimental.rotary_embedding:
- When token_idx is provided, input_tensor.padded_shape()[0] (batch dim) must be 1
- So we process each batch element separately
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.common import precompute_freqs


def rotate_half(x):
    """Rotates half the hidden dims of the input (HuggingFace implementation)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding (HuggingFace implementation).

    Args:
        q: Query tensor [batch_size, n_heads, seq_len, head_dim]
        k: Key tensor [batch_size, n_heads, seq_len, head_dim]
        cos: Cosine part of rotary embedding [batch_size, seq_len, head_dim]
        sin: Sine part of rotary embedding [batch_size, seq_len, head_dim]
        position_ids: Deprecated and unused
        unsqueeze_dim: Dimension to unsqueeze cos/sin (default 1 for standard attention)

    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def generate_hf_cos_sin_cache(head_dim, max_seq_len, theta=10000.0):
    """Generate HF-format cos/sin cache.

    Returns cos/sin in HF format: [c0, c1, ..., c_{d/2-1}, c0, c1, ..., c_{d/2-1}]
    Shape: [1, 1, max_seq_len, head_dim]
    """
    cos_freqs, sin_freqs = precompute_freqs(
        head_dim, max_seq_len * 2, theta, scale_factor=None, orig_context_len=None, rope_type="llama3"
    )

    # HF format: concat freqs with itself [c0, c1, ..., c_{d/2-1}, c0, c1, ..., c_{d/2-1}]
    cos_hf = torch.cat([cos_freqs[:max_seq_len], cos_freqs[:max_seq_len]], dim=-1)  # [max_seq_len, head_dim]
    sin_hf = torch.cat([sin_freqs[:max_seq_len], sin_freqs[:max_seq_len]], dim=-1)  # [max_seq_len, head_dim]

    # Add batch dimensions: [1, 1, max_seq_len, head_dim]
    cos_hf = cos_hf.unsqueeze(0).unsqueeze(0)
    sin_hf = sin_hf.unsqueeze(0).unsqueeze(0)

    return cos_hf, sin_hf


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1, 4, 8], ids=["batch_1", "batch_4", "batch_8"])
@pytest.mark.parametrize("head_dim", [64, 128], ids=["head_64", "head_128"])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("position_idx", [0, 100, 500, 1024], ids=["pos_0", "pos_100", "pos_500", "pos_1024"])
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_trace_workaround(batch_size, head_dim, n_heads, position_idx, mesh_device, reset_seeds):
    """Test trace-compatible HF-style RoPE using pre-shifted cos/sin matrices.

    This test validates that ttnn.experimental.rotary_embedding produces correct results
    when cos/sin matrices are pre-sliced to the target position and token_idx=0 is used.

    The workaround:
    1. Pre-slice cos/sin cache at position_idx (on host): cache[:, :, position_idx:position_idx+1, :]
    2. Pass the sliced cache (with position N values at index 0) to the device
    3. Call ttnn.experimental.rotary_embedding with token_idx=0

    This enables tracing because token_idx=0 is a compile-time constant.

    Note: ttnn.experimental.rotary_embedding requires batch_dim=1 when token_idx is provided,
    so we process each batch element separately.
    """
    seq_len = 1  # Decode mode
    max_seq_len = 2048  # Cache size
    theta = 10000.0
    pcc_threshold = 0.99

    logger.info(f"Testing trace workaround: batch_size={batch_size}, head_dim={head_dim}, position_idx={position_idx}")

    # Generate full HF-format cos/sin cache
    # Shape: [1, 1, max_seq_len, head_dim]
    cos_cache, sin_cache = generate_hf_cos_sin_cache(head_dim, max_seq_len, theta)

    # Create random Q and K tensors
    # Shape: [batch_size, n_heads, seq_len, head_dim]
    q_torch = torch.randn(batch_size, n_heads, seq_len, head_dim).bfloat16().float()
    k_torch = torch.randn(batch_size, n_heads, seq_len, head_dim).bfloat16().float()

    # WORKAROUND: Pre-slice cos/sin at position_idx, placing values at index 0
    # Full cache shape: [1, 1, max_seq_len, head_dim]
    # Sliced cache shape: [1, 1, 1, head_dim] - values for position_idx are now at index 0
    cos_sliced = cos_cache[:, :, position_idx : position_idx + 1, :]
    sin_sliced = sin_cache[:, :, position_idx : position_idx + 1, :]

    # Reference: Apply HF RoPE using sliced cos/sin
    # Reference expects cos/sin shape: [batch, seq_len, head_dim]
    cos_ref = cos_sliced[0, 0, :, :].unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 1, head_dim]
    sin_ref = sin_sliced[0, 0, :, :].unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 1, head_dim]

    q_ref, k_ref = apply_rotary_pos_emb(q_torch, k_torch, cos_ref, sin_ref)

    # Convert pre-sliced cos/sin to TT tensors
    # Shape: [1, 1, 1, head_dim] - position_idx values are at index 0
    cos_tt = ttnn.from_torch(
        cos_sliced,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    sin_tt = ttnn.from_torch(
        sin_sliced,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Process each batch element separately (required when token_idx is provided)
    # ttnn.experimental.rotary_embedding requires input_tensor.padded_shape()[0] == 1
    q_tt_outputs = []
    k_tt_outputs = []

    for i in range(batch_size):
        # Extract single batch element: [1, n_heads, seq_len, head_dim]
        q_batch_i = q_torch[i : i + 1, :, :, :]
        k_batch_i = k_torch[i : i + 1, :, :, :]

        # Convert to TT tensors
        q_tt_i = ttnn.from_torch(
            q_batch_i,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        k_tt_i = ttnn.from_torch(
            k_batch_i,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Apply TTNN RoPE with token_idx=0 (TRACE-COMPATIBLE!)
        # Since we pre-sliced the cache, index 0 contains the values for position_idx
        q_tt_out_i = ttnn.experimental.rotary_embedding(
            q_tt_i,
            cos_tt,
            sin_tt,
            0,  # Always 0 - trace compatible! Pre-sliced cache has position_idx values at index 0
        )

        k_tt_out_i = ttnn.experimental.rotary_embedding(
            k_tt_i,
            cos_tt,
            sin_tt,
            0,  # Always 0 - trace compatible! Pre-sliced cache has position_idx values at index 0
        )

        # Convert back to torch
        q_tt_torch_i = to_torch_auto_compose(q_tt_out_i, mesh_device)
        k_tt_torch_i = to_torch_auto_compose(k_tt_out_i, mesh_device)

        q_tt_outputs.append(q_tt_torch_i)
        k_tt_outputs.append(k_tt_torch_i)

    # Concatenate batch outputs
    q_tt_torch = torch.cat(q_tt_outputs, dim=0)
    k_tt_torch = torch.cat(k_tt_outputs, dim=0)

    # Handle potential padding in output (rotary_embedding may pad outputs)
    q_tt_torch = q_tt_torch[..., :seq_len, :]
    k_tt_torch = k_tt_torch[..., :seq_len, :]

    # Compare Q results
    q_passing, q_pcc_message = comp_pcc(q_ref, q_tt_torch, pcc_threshold)
    logger.info(f"Q PCC: {q_pcc_message}")

    # Compare K results
    k_passing, k_pcc_message = comp_pcc(k_ref, k_tt_torch, pcc_threshold)
    logger.info(f"K PCC: {k_pcc_message}")

    if q_passing and k_passing:
        logger.info(
            f"✓ Trace workaround PASSED for batch_size={batch_size}, head_dim={head_dim}, position_idx={position_idx}"
        )
    else:
        logger.warning(
            f"✗ Trace workaround FAILED for batch_size={batch_size}, head_dim={head_dim}, position_idx={position_idx}"
        )

    assert q_passing, f"Q tensor PCC {q_pcc_message} is below threshold {pcc_threshold}"
    assert k_passing, f"K tensor PCC {k_pcc_message} is below threshold {pcc_threshold}"


if __name__ == "__main__":
    print("Running quick local test (CPU only)...")
    print("Note: This validates the logic but requires TTNN device for full test.")

    batch_size = 4
    head_dim = 64
    n_heads = 8
    seq_len = 1
    max_seq_len = 2048
    theta = 10000.0
    position_idx = 100

    print(f"Test parameters: batch_size={batch_size}, head_dim={head_dim}, position_idx={position_idx}")

    # Generate full cos/sin cache
    cos_cache, sin_cache = generate_hf_cos_sin_cache(head_dim, max_seq_len, theta)

    # Create random Q and K
    q_torch = torch.randn(batch_size, n_heads, seq_len, head_dim).bfloat16().float()
    k_torch = torch.randn(batch_size, n_heads, seq_len, head_dim).bfloat16().float()

    # WORKAROUND: Pre-slice cos/sin at position_idx
    cos_sliced = cos_cache[:, :, position_idx : position_idx + 1, :]
    sin_sliced = sin_cache[:, :, position_idx : position_idx + 1, :]

    # Reference computation
    cos_ref = cos_sliced[0, 0, :, :].unsqueeze(0).expand(batch_size, -1, -1)
    sin_ref = sin_sliced[0, 0, :, :].unsqueeze(0).expand(batch_size, -1, -1)
    q_ref, k_ref = apply_rotary_pos_emb(q_torch, k_torch, cos_ref, sin_ref)

    print(f"cos_cache shape: {cos_cache.shape}")
    print(f"cos_sliced shape: {cos_sliced.shape}")
    print(f"cos_ref shape: {cos_ref.shape}")
    print(f"q_torch shape: {q_torch.shape}")
    print(f"q_ref shape: {q_ref.shape}")

    print("\n✓ Local test structure validated. Run with pytest for full TTNN test.")
    print("\nWorkaround summary:")
    print("  1. Pre-slice cos/sin cache at position_idx (on host)")
    print("  2. Pass sliced cache to device (position_idx values now at index 0)")
    print("  3. Call ttnn.experimental.rotary_embedding with token_idx=0")
    print("  4. This is trace-compatible because token_idx=0 is a compile-time constant!")
