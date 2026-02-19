# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for HfRotarySetupNew with per-batch position rotation support.

This test validates that HfRotarySetupNew correctly extracts per-batch cos/sin
values and works with ttnn.experimental.rotary_embedding_hf for decode mode.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.common import precompute_freqs
from models.tt_transformers.tt.rope import HfRotarySetupNew


def rotate_half(x):
    """Rotates half the hidden dims of the input (HuggingFace implementation)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_hf(x, cos, sin):
    """Golden function for HF-style rotary embedding."""
    return (x * cos) + (rotate_half(x) * sin)


def generate_hf_cos_sin_cache(head_dim, max_seq_len, theta=10000.0):
    """Generate HF-format cos/sin cache.

    Returns cos/sin in HF format: [c0, c1, ..., c_{d/2-1}, c0, c1, ..., c_{d/2-1}]
    Shape: [1, 1, max_seq_len, head_dim]
    """
    # Use precompute_freqs from common.py
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


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_hf_rotary_setup_new_per_batch_positions(mesh_device, device_params):
    """Test that HfRotarySetupNew correctly extracts per-batch cos/sin values."""
    head_dim = 64
    max_seq_len = 128
    batch_size = 8
    theta = 10000.0

    # Create HfRotarySetupNew instance
    rope_setup = HfRotarySetupNew(
        device=mesh_device,
        batch_size=batch_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=theta,
        rope_scaling=None,
        use_qk_fused=False,
        datatype=ttnn.bfloat16,
    )

    # Create position indices - different position for each batch element
    position_indices = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35], dtype=torch.int32)

    # Get rotation matrices for decode mode
    rot_mats = rope_setup.get_rot_mats(position_indices)

    cos_tt = rot_mats[0]
    sin_tt = rot_mats[1]

    # Convert to torch for comparison
    cos_torch = ttnn.to_torch(cos_tt)
    sin_torch = ttnn.to_torch(sin_tt)

    # Expected shapes: [1, batch, 1, head_dim]
    assert cos_torch.shape == (
        1,
        batch_size,
        1,
        head_dim,
    ), f"Expected cos shape (1, {batch_size}, 1, {head_dim}), got {cos_torch.shape}"
    assert sin_torch.shape == (
        1,
        batch_size,
        1,
        head_dim,
    ), f"Expected sin shape (1, {batch_size}, 1, {head_dim}), got {sin_torch.shape}"

    # Generate reference cos/sin cache
    cos_cache_ref, sin_cache_ref = generate_hf_cos_sin_cache(head_dim, max_seq_len, theta)

    # Verify that each batch element has the correct cos/sin values
    for i, pos_idx in enumerate(position_indices):
        # Reference: extract cos/sin at position pos_idx
        cos_ref = cos_cache_ref[:, :, pos_idx : pos_idx + 1, :]  # [1, 1, 1, head_dim]
        sin_ref = sin_cache_ref[:, :, pos_idx : pos_idx + 1, :]  # [1, 1, 1, head_dim]

        # Our implementation: batch element i
        cos_actual = cos_torch[:, i : i + 1, :, :]  # [1, 1, 1, head_dim]
        sin_actual = sin_torch[:, i : i + 1, :, :]  # [1, 1, 1, head_dim]

        # Compare values
        cos_diff = torch.abs(cos_ref - cos_actual)
        sin_diff = torch.abs(sin_ref - sin_actual)

        max_cos_diff = torch.max(cos_diff).item()
        max_sin_diff = torch.max(sin_diff).item()

        assert (
            max_cos_diff < 1e-3
        ), f"Batch element {i} cos values don't match at position {pos_idx}, max diff: {max_cos_diff}"
        assert (
            max_sin_diff < 1e-3
        ), f"Batch element {i} sin values don't match at position {pos_idx}, max diff: {max_sin_diff}"

    logger.info(f"✓ HfRotarySetupNew correctly extracts per-batch cos/sin values for {batch_size} batch elements")


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_hf_rotary_setup_new_with_rotary_embedding_hf(mesh_device, device_params):
    """Test that HfRotarySetupNew works correctly with rotary_embedding_hf."""
    head_dim = 64
    max_seq_len = 128
    batch_size = 4
    num_heads = 8
    theta = 10000.0

    # Create HfRotarySetupNew instance
    rope_setup = HfRotarySetupNew(
        device=mesh_device,
        batch_size=batch_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=theta,
        rope_scaling=None,
        use_qk_fused=False,
        datatype=ttnn.bfloat16,
    )

    # Create position indices - different position for each batch element
    position_indices = torch.tensor([10, 20, 30, 40], dtype=torch.int32)

    # Get rotation matrices for decode mode
    rot_mats = rope_setup.get_rot_mats(position_indices)

    # Create input tensors [1, batch, num_heads, head_dim]
    torch_input_q = torch.randn(1, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
    torch_input_k = torch.randn(1, batch_size, num_heads, head_dim, dtype=torch.bfloat16)

    # Convert to TTNN tensors with HEIGHT_SHARDED memory config
    shard_height = num_heads * 32  # TILE_HEIGHT
    shard_width = head_dim
    core_grid = mesh_device.compute_with_storage_grid_size()
    max_cores = core_grid.x * core_grid.y
    num_cores = min(batch_size, max_cores)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, core_grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    input_q_tt = ttnn.from_torch(
        torch_input_q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=sharded_mem_config,
    )
    input_k_tt = ttnn.from_torch(
        torch_input_k,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=sharded_mem_config,
    )

    print("rot_mats.shape", rot_mats[0].shape)
    # Apply rotary embedding using rotary_embedding_hf
    output_q_tt = ttnn.experimental.rotary_embedding_hf(input_q_tt, rot_mats[0], rot_mats[1], is_decode=True)
    output_k_tt = ttnn.experimental.rotary_embedding_hf(input_k_tt, rot_mats[0], rot_mats[1], is_decode=True)

    # Convert back to torch
    output_q_torch = ttnn.to_torch(output_q_tt)
    output_k_torch = ttnn.to_torch(output_k_tt)

    # Generate reference output
    cos_cache_ref, sin_cache_ref = generate_hf_cos_sin_cache(head_dim, max_seq_len, theta)
    output_q_ref = torch.zeros_like(torch_input_q)
    output_k_ref = torch.zeros_like(torch_input_k)

    for i, pos_idx in enumerate(position_indices):
        cos_ref = cos_cache_ref[:, :, pos_idx : pos_idx + 1, :]  # [1, 1, 1, head_dim]
        sin_ref = sin_cache_ref[:, :, pos_idx : pos_idx + 1, :]  # [1, 1, 1, head_dim]
        # Expand for num_heads: [1, 1, num_heads, head_dim]
        cos_expanded = cos_ref.expand(-1, -1, num_heads, -1)
        sin_expanded = sin_ref.expand(-1, -1, num_heads, -1)

        q_batch = torch_input_q[:, i : i + 1, :, :]  # [1, 1, num_heads, head_dim]
        k_batch = torch_input_k[:, i : i + 1, :, :]  # [1, 1, num_heads, head_dim]

        output_q_ref[:, i : i + 1, :, :] = apply_rotary_pos_emb_hf(q_batch, cos_expanded, sin_expanded)
        output_k_ref[:, i : i + 1, :, :] = apply_rotary_pos_emb_hf(k_batch, cos_expanded, sin_expanded)

    # Compare outputs
    q_pcc = comp_pcc(output_q_torch, output_q_ref)
    k_pcc = comp_pcc(output_k_torch, output_k_ref)

    logger.info(f"Query PCC: {q_pcc:.6f}, Key PCC: {k_pcc:.6f}")

    assert q_pcc > 0.99, f"Query output PCC {q_pcc} is too low (expected > 0.99)"
    assert k_pcc > 0.99, f"Key output PCC {k_pcc} is too low (expected > 0.99)"

    logger.info("✓ HfRotarySetupNew works correctly with rotary_embedding_hf for per-batch positions")
