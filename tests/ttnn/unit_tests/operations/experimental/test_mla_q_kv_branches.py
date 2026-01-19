# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for individual MLA (Multi-head Latent Attention) Q/KV branch operations
from the DeepSeek V3 model.

This file contains separate tests for each operation in the MLA layer:
- Q RMS Norm
- KV RMS Norm
- Q RoPE (Rotary Position Embedding)
- KV RoPE (Rotary Position Embedding)
- Paged Update Cache
- WKV_B Linear (wkv_b1 and wkv_b2 projections)
- WQ_B Linear

Each test runs on a single device (not mesh device).
"""

import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.reference.modeling_deepseek import apply_rotary_pos_emb
from models.demos.deepseek_v3.tt.rms_norm.rms_norm import RMSNorm
from models.demos.deepseek_v3.tt.rope import get_cos_sin_matrix, get_rot_transformation_mat
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict

# Use single device fixture
pytestmark = pytest.mark.use_module_device


# =============================================================================
# Test configuration
# =============================================================================

# Default batch size for single device tests (reduced from USERS_PER_ROW for faster tests)
BATCH_SIZE = 32
EXPECTED_PCC = 0.999
EXPECTED_ATOL = 0.2
EXPECTED_RTOL = 0.2


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def model_path():
    """Path to the DeepSeek model."""
    return Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "models/demos/deepseek_v3/reference"))


@pytest.fixture(scope="session")
def hf_config(model_path):
    """Load DeepSeek config for testing."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return config


@pytest.fixture(scope="session")
def hf_config_short(hf_config):
    """Shortened config with 1 layer and reduced sequence length."""
    hf_config_out = deepcopy(hf_config)
    hf_config_out.num_hidden_layers = 1
    hf_config_out.max_seq_len = 4096
    return hf_config_out


# =============================================================================
# Reference implementations (PyTorch)
# =============================================================================


def _rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMS normalization."""
    x_dtype = x.dtype
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    weight = weight.view((1,) * (y.dim() - 1) + (-1,))
    return (y * weight).to(x_dtype)


def _compare_with_reference(
    tt_output: torch.Tensor, ref_output: torch.Tensor, expected_pcc: float, atol: float, rtol: float
):
    """Compare TTNN output with reference."""
    passing, pcc = comp_pcc(ref_output, tt_output, expected_pcc)
    logger.info(f"PCC: {pcc}")
    assert passing, f"PCC {pcc} is below required {expected_pcc}"
    torch.testing.assert_close(tt_output, ref_output, rtol=rtol, atol=atol)


# =============================================================================
# Helper functions for single-device tensor creation
# =============================================================================


def to_ttnn_single_device(
    torch_tensor: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """Convert a torch tensor to a TTNN tensor on a single device (no mesh mapper)."""
    return ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def create_rope_tensors_single_device(
    hf_config,
    batch_size: int,
    seq_len: int,
    position_ids: torch.Tensor,
    device: ttnn.Device,
) -> dict:
    """
    Create rotation matrices for rotary position embeddings on a single device.
    Adapted from get_rope_tensors but without mesh device.
    """
    # Get cos/sin matrices
    cos, sin = get_cos_sin_matrix(hf_config)
    cos = cos.squeeze(0).squeeze(0).to(torch.bfloat16)
    sin = sin.squeeze(0).squeeze(0).to(torch.bfloat16)

    # Get transformation matrix
    trans_mat = get_rot_transformation_mat()

    # For decode mode, gather cos/sin for specific positions
    # position_ids shape: [batch_size]
    gathered_cos = cos[position_ids]  # [batch_size, head_dim]
    gathered_sin = sin[position_ids]  # [batch_size, head_dim]

    # Reshape for rotary_embedding_llama: [1, batch, 1, head_dim]
    gathered_cos = gathered_cos.unsqueeze(0).unsqueeze(2)
    gathered_sin = gathered_sin.unsqueeze(0).unsqueeze(2)

    # Create sharded memory config for transformation matrix
    head_dim = hf_config.qk_rope_head_dim
    core_grid = device.compute_with_storage_grid_size()
    num_cores = min(batch_size, core_grid.x * core_grid.y)
    batch_grid = ttnn.num_cores_to_corerangeset(num_cores, core_grid, row_wise=True)

    trans_mat_repeated = trans_mat.repeat(1, 1, batch_size, 1)
    trans_mat_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tt_cos = to_ttnn_single_device(gathered_cos, device)
    tt_sin = to_ttnn_single_device(gathered_sin, device)
    tt_trans = ttnn.from_torch(
        trans_mat_repeated,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=trans_mat_mem_config,
    )

    return {
        "cos_matrix": tt_cos,
        "sin_matrix": tt_sin,
        "trans_matrix": tt_trans,
        "torch_cos": cos,
        "torch_sin": sin,
    }


# =============================================================================
# Individual operation tests
# =============================================================================


@pytest.mark.parametrize("batch_size", [BATCH_SIZE])
def test_q_rms_norm(device, hf_config_short, batch_size):
    """
    Test Q RMS normalization operation.

    This tests the RMSNorm applied to the Q input in the MLA layer.
    """
    torch.manual_seed(42)

    q_lora_rank = hf_config_short.q_lora_rank
    rms_norm_eps = hf_config_short.rms_norm_eps

    # Create input tensor: shape [1, 1, batch, q_lora_rank]
    torch_input = torch.randn(1, 1, batch_size, q_lora_rank, dtype=torch.bfloat16)

    # Create weight tensor (random for testing)
    torch_weight = torch.randn(q_lora_rank, dtype=torch.bfloat16)

    # Reference computation
    ref_output = _rms_norm_reference(torch_input, torch_weight, rms_norm_eps)

    # TTNN computation
    tt_input = to_ttnn_single_device(torch_input, device)
    tt_weight = to_ttnn_single_device(
        torch_weight.reshape(1, 1, 1, q_lora_rank),
        device,
    )

    tt_output = ttnn.rms_norm(tt_input, epsilon=rms_norm_eps, weight=tt_weight)
    tt_output_torch = ttnn.to_torch(tt_output)

    # Compare
    _compare_with_reference(tt_output_torch, ref_output, EXPECTED_PCC, EXPECTED_ATOL, EXPECTED_RTOL)


@pytest.mark.parametrize("batch_size", [BATCH_SIZE])
def test_kv_rms_norm(device, hf_config_short, batch_size):
    """
    Test KV RMS normalization operation.

    This tests the RMSNorm applied to the KV nope input in the MLA layer.
    """
    torch.manual_seed(42)

    kv_lora_rank = hf_config_short.kv_lora_rank
    rms_norm_eps = hf_config_short.rms_norm_eps

    # Create input tensor: shape [1, 1, batch, kv_lora_rank]
    torch_input = torch.randn(1, 1, batch_size, kv_lora_rank, dtype=torch.bfloat16)

    # Create weight tensor
    torch_weight = torch.randn(kv_lora_rank, dtype=torch.bfloat16)

    # Reference computation
    ref_output = _rms_norm_reference(torch_input, torch_weight, rms_norm_eps)

    # TTNN computation
    tt_input = to_ttnn_single_device(torch_input, device)
    tt_weight = to_ttnn_single_device(
        torch_weight.reshape(1, 1, 1, kv_lora_rank),
        device,
    )

    tt_output = ttnn.rms_norm(tt_input, epsilon=rms_norm_eps, weight=tt_weight)
    tt_output_torch = ttnn.to_torch(tt_output)

    # Compare
    _compare_with_reference(tt_output_torch, ref_output, EXPECTED_PCC, EXPECTED_ATOL, EXPECTED_RTOL)


@pytest.mark.parametrize("batch_size", [BATCH_SIZE])
def test_kv_rope(device, hf_config_short, batch_size):
    """
    Test KV rotary position embedding operation.

    This tests the rotary embedding applied to kv_rope in the MLA layer.
    """
    torch.manual_seed(42)

    qk_rope_head_dim = hf_config_short.qk_rope_head_dim
    max_seq_len = hf_config_short.max_seq_len

    # Position IDs - random positions within the sequence
    position_ids = torch.randint(0, max_seq_len - 1, (batch_size,))

    # Create rope tensors
    rope_tensors = create_rope_tensors_single_device(hf_config_short, batch_size, 1, position_ids, device)

    # KV rope input: [1, 1, batch, head_dim]
    torch_input = torch.randn(1, 1, batch_size, qk_rope_head_dim, dtype=torch.bfloat16)

    # Reference computation using DeepSeek's apply_rotary_pos_emb
    # Permute for reference: [batch, seq, heads, dim]
    torch_input_for_ref = torch_input.permute(2, 0, 1, 3)  # [batch, 1, 1, head_dim]
    cos = rope_tensors["torch_cos"]
    sin = rope_tensors["torch_sin"]

    ref_output, _ = apply_rotary_pos_emb(
        torch_input_for_ref.permute(1, 0, 2, 3),  # [1, batch, 1, head_dim]
        torch_input_for_ref.permute(1, 0, 2, 3),
        cos.unsqueeze(0).unsqueeze(0),  # Add batch and head dims
        sin.unsqueeze(0).unsqueeze(0),
        position_ids.unsqueeze(1),
        unsqueeze_dim=1,
        meta_style=True,
    )
    ref_output = ref_output.permute(1, 0, 2, 3)  # Back to [batch, 1, 1, head_dim]
    ref_output = ref_output.permute(1, 2, 0, 3)  # [1, 1, batch, head_dim]

    # TTNN computation
    tt_input = to_ttnn_single_device(torch_input, device)

    tt_output = ttnn.experimental.rotary_embedding_llama(
        tt_input,
        rope_tensors["cos_matrix"],
        rope_tensors["sin_matrix"],
        rope_tensors["trans_matrix"],
        is_decode_mode=True,
    )
    tt_output_torch = ttnn.to_torch(tt_output)

    # Compare - rotary embedding can have slightly lower PCC
    _compare_with_reference(tt_output_torch, ref_output, 0.99, EXPECTED_ATOL, EXPECTED_RTOL)


@pytest.mark.parametrize("batch_size", [BATCH_SIZE])
def test_wq_b_linear(device, hf_config_short, batch_size):
    """
    Test W_q^B linear projection operation.

    This tests the linear projection that expands the compressed Q representation
    to the full Q dimension (num_heads * qk_head_dim).
    """
    torch.manual_seed(42)

    q_lora_rank = hf_config_short.q_lora_rank
    num_heads = hf_config_short.num_attention_heads
    qk_nope_head_dim = hf_config_short.qk_nope_head_dim
    qk_rope_head_dim = hf_config_short.qk_rope_head_dim
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    output_dim = num_heads * qk_head_dim

    # Input: compressed Q representation [1, 1, batch, q_lora_rank]
    torch_input = torch.randn(1, 1, batch_size, q_lora_rank, dtype=torch.bfloat16)

    # Weight: [output_dim, q_lora_rank] (random for testing)
    torch_weight = torch.randn(output_dim, q_lora_rank, dtype=torch.bfloat16) * 0.02

    # Reference computation
    ref_output = torch.nn.functional.linear(torch_input, torch_weight)

    # TTNN computation
    tt_input = to_ttnn_single_device(torch_input, device)
    tt_weight = to_ttnn_single_device(torch_weight.T, device)  # Transpose for TTNN

    tt_output = ttnn.linear(tt_input, tt_weight)
    tt_output_torch = ttnn.to_torch(tt_output)

    # Compare
    _compare_with_reference(tt_output_torch, ref_output, 0.99, EXPECTED_ATOL, EXPECTED_RTOL)


@pytest.mark.parametrize("batch_size", [BATCH_SIZE])
def test_wkv_b_linear(device, hf_config_short, batch_size):
    """
    Test W_kv^B linear projection operation.

    This tests the combined KV projection weight (wkv_b) that projects
    the compressed KV representation.
    """
    torch.manual_seed(42)

    kv_lora_rank = hf_config_short.kv_lora_rank
    num_heads = hf_config_short.num_attention_heads
    qk_nope_head_dim = hf_config_short.qk_nope_head_dim
    v_head_dim = hf_config_short.v_head_dim
    output_dim = num_heads * (qk_nope_head_dim + v_head_dim)

    # Input: compressed KV representation [1, 1, batch, kv_lora_rank]
    torch_input = torch.randn(1, 1, batch_size, kv_lora_rank, dtype=torch.bfloat16)

    # Weight: [output_dim, kv_lora_rank]
    torch_weight = torch.randn(output_dim, kv_lora_rank, dtype=torch.bfloat16) * 0.02

    # Reference computation
    ref_output = torch.nn.functional.linear(torch_input, torch_weight)

    # TTNN computation
    tt_input = to_ttnn_single_device(torch_input, device)
    tt_weight = to_ttnn_single_device(torch_weight.T, device)

    tt_output = ttnn.linear(tt_input, tt_weight)
    tt_output_torch = ttnn.to_torch(tt_output)

    # Compare
    _compare_with_reference(tt_output_torch, ref_output, 0.99, EXPECTED_ATOL, EXPECTED_RTOL)


@pytest.mark.parametrize("batch_size", [BATCH_SIZE])
def test_paged_update_cache(device, hf_config_short, batch_size):
    """
    Test paged update cache operation.

    This tests updating the KV cache with new key-value pairs at specific positions.
    """
    torch.manual_seed(42)

    kv_lora_rank = hf_config_short.kv_lora_rank
    qk_rope_head_dim = hf_config_short.qk_rope_head_dim
    kvpe_dim = kv_lora_rank + qk_rope_head_dim
    max_seq_len = min(hf_config_short.max_seq_len, 2048)  # Limit for testing

    # Create initial cache (zeros): [batch, 1, max_seq_len, kvpe_dim]
    torch_cache = torch.zeros(batch_size, 1, max_seq_len, kvpe_dim, dtype=torch.bfloat16)

    # Update tensor: [1, 1, batch, kvpe_dim]
    torch_update = torch.randn(1, 1, batch_size, kvpe_dim, dtype=torch.bfloat16)

    # Random position indices
    position_idxs = torch.randint(0, max_seq_len - 1, (batch_size,), dtype=torch.int32)

    # Reference computation
    ref_cache = torch_cache.clone()
    for user_idx, update_idx in enumerate(position_idxs.tolist()):
        if update_idx >= 0:
            ref_cache[user_idx, 0, update_idx : update_idx + 1, :] = torch_update[0, 0, user_idx : user_idx + 1, :]

    # TTNN computation
    tt_cache = to_ttnn_single_device(torch_cache, device)

    # Pad and permute update for paged_update_cache
    # Expected shape: [1, batch, tile_height, kvpe_dim]
    torch_update_padded = torch.nn.functional.pad(
        torch_update,
        (0, 0, 0, ttnn.TILE_SIZE - 1, 0, 0, 0, 0),
        value=0,
    )  # [1, 32, batch, kvpe_dim]
    torch_update_permuted = torch_update_padded.permute(0, 2, 1, 3)  # [1, batch, 32, kvpe_dim]

    tt_update = to_ttnn_single_device(torch_update_permuted, device)
    tt_position_idxs = ttnn.from_torch(position_idxs, device=device, dtype=ttnn.int32)

    # Run paged_update_cache
    ttnn.experimental.paged_update_cache(
        tt_cache,
        tt_update,
        update_idxs_tensor=tt_position_idxs,
    )

    tt_cache_torch = ttnn.to_torch(tt_cache)

    # Compare only the updated positions (cache update can have some numerical differences)
    for user_idx, pos_idx in enumerate(position_idxs.tolist()):
        if pos_idx >= 0:
            expected = ref_cache[user_idx, 0, pos_idx]
            actual = tt_cache_torch[user_idx, 0, pos_idx]
            passing, pcc = comp_pcc(expected.unsqueeze(0), actual.unsqueeze(0), 0.99)
            assert passing, f"PCC {pcc} at position {pos_idx} for user {user_idx} is below 0.99"


# =============================================================================
# Combined operation tests (pipelines)
# =============================================================================


@pytest.mark.parametrize("batch_size", [BATCH_SIZE])
def test_q_norm_then_linear(device, hf_config_short, batch_size):
    """
    Test Q path: RMS norm followed by W_q^B linear.

    This tests the sequential application of normalization and linear projection
    in the Q branch of the MLA layer.
    """
    torch.manual_seed(42)

    q_lora_rank = hf_config_short.q_lora_rank
    num_heads = hf_config_short.num_attention_heads
    qk_nope_head_dim = hf_config_short.qk_nope_head_dim
    qk_rope_head_dim = hf_config_short.qk_rope_head_dim
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    output_dim = num_heads * qk_head_dim
    rms_norm_eps = hf_config_short.rms_norm_eps

    # Inputs
    torch_input = torch.randn(1, 1, batch_size, q_lora_rank, dtype=torch.bfloat16)
    torch_norm_weight = torch.randn(q_lora_rank, dtype=torch.bfloat16)
    torch_linear_weight = torch.randn(output_dim, q_lora_rank, dtype=torch.bfloat16) * 0.02

    # Reference: norm -> linear
    ref_normed = _rms_norm_reference(torch_input, torch_norm_weight, rms_norm_eps)
    ref_output = torch.nn.functional.linear(ref_normed, torch_linear_weight)

    # TTNN computation
    tt_input = to_ttnn_single_device(torch_input, device)
    tt_norm_weight = to_ttnn_single_device(torch_norm_weight.reshape(1, 1, 1, q_lora_rank), device)
    tt_linear_weight = to_ttnn_single_device(torch_linear_weight.T, device)

    tt_normed = ttnn.rms_norm(tt_input, epsilon=rms_norm_eps, weight=tt_norm_weight)
    tt_output = ttnn.linear(tt_normed, tt_linear_weight)
    tt_output_torch = ttnn.to_torch(tt_output)

    _compare_with_reference(tt_output_torch, ref_output, 0.99, EXPECTED_ATOL, EXPECTED_RTOL)


@pytest.mark.parametrize("batch_size", [BATCH_SIZE])
def test_kv_norm_and_concat(device, hf_config_short, batch_size):
    """
    Test KV path: norm the kv_nope, then concatenate with kv_rope.

    This tests part of the KV processing pipeline where the non-positional
    KV component is normalized and concatenated with the positional component.
    """
    torch.manual_seed(42)

    kv_lora_rank = hf_config_short.kv_lora_rank
    qk_rope_head_dim = hf_config_short.qk_rope_head_dim
    rms_norm_eps = hf_config_short.rms_norm_eps

    # Inputs
    torch_kv_nope = torch.randn(1, 1, batch_size, kv_lora_rank, dtype=torch.bfloat16)
    torch_kv_rope = torch.randn(1, 1, batch_size, qk_rope_head_dim, dtype=torch.bfloat16)
    torch_norm_weight = torch.randn(kv_lora_rank, dtype=torch.bfloat16)

    # Reference: norm the nope, concat with rope
    ref_normed = _rms_norm_reference(torch_kv_nope, torch_norm_weight, rms_norm_eps)
    ref_output = torch.cat([ref_normed, torch_kv_rope], dim=-1)

    # TTNN computation
    tt_kv_nope = to_ttnn_single_device(torch_kv_nope, device)
    tt_kv_rope = to_ttnn_single_device(torch_kv_rope, device)
    tt_norm_weight = to_ttnn_single_device(torch_norm_weight.reshape(1, 1, 1, kv_lora_rank), device)

    tt_normed = ttnn.rms_norm(tt_kv_nope, epsilon=rms_norm_eps, weight=tt_norm_weight)
    tt_output = ttnn.concat([tt_normed, tt_kv_rope], dim=-1)
    tt_output_torch = ttnn.to_torch(tt_output)

    _compare_with_reference(tt_output_torch, ref_output, EXPECTED_PCC, EXPECTED_ATOL, EXPECTED_RTOL)
