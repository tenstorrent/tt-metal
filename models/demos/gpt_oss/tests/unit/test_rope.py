# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for GPT-OSS RoPE (Rotary Position Embedding) implementation.

This module compares our rope setup (sin/cos embedding matrices) with the
HuggingFace transformers reference implementation.
"""

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.tt.model import create_rope_setup
from models.tt_transformers.tt.common import rope_scaling_model_factory
from models.tt_transformers.tt.rope import rotary_embedding_factory

from ..test_factory import parametrize_mesh_with_fabric


def get_rope_cos_sin_from_tt_setup(rope_setup, mesh_device, seq_len):
    """
    Extract cos/sin matrices from our TT rope setup and convert to torch tensors.

    Args:
        rope_setup: RotarySetup instance
        mesh_device: TTNN mesh device
        seq_len: Sequence length to extract

    Returns:
        Tuple of (cos_torch, sin_torch) tensors in shape [1, 1, seq_len, head_dim]
    """
    # Get the cos/sin matrices from rope_setup (stored in Meta format)
    cos_tt = rope_setup.cos_matrix[:, :, :seq_len, :]
    sin_tt = rope_setup.sin_matrix[:, :, :seq_len, :]

    # Convert to torch tensors
    cos_torch = ttnn.to_torch(ttnn.get_device_tensors(cos_tt)[0])
    sin_torch = ttnn.to_torch(ttnn.get_device_tensors(sin_tt)[0])

    return cos_torch, sin_torch


def get_rope_cos_sin_from_hf_reference(hf_config, seq_len, device=None):
    """
    Get cos/sin matrices from HuggingFace transformers reference implementation.

    This uses the official HuggingFace modeling_gpt_oss.GptOssRotaryEmbedding
    class as the reference.

    Args:
        hf_config: HuggingFace config
        seq_len: Sequence length
        device: PyTorch device (default: None for CPU)

    Returns:
        Tuple of (cos_ref, sin_ref) tensors in HF format
    """
    try:
        # Try to use the HuggingFace transformers implementation
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

        rope_emb = GptOssRotaryEmbedding(hf_config)

        # Create dummy input to get cos/sin embeddings
        # HF expects input shape [batch, seq_len, hidden_size]
        dummy_input = torch.randn(1, seq_len, hf_config.hidden_size)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # Get cos/sin from HF reference
        cos_ref, sin_ref = rope_emb(dummy_input, position_ids)

        return cos_ref, sin_ref

    except ImportError:
        pytest.skip("HuggingFace transformers gpt_oss model not available")


def get_rope_cos_sin_from_our_pytorch_impl(hf_config, seq_len):
    """
    Get cos/sin matrices from our PyTorch reference implementation (rotary_embedding_factory).

    This uses the same PyTorch implementation that our TTNN code uses internally.

    Args:
        hf_config: HuggingFace config
        seq_len: Sequence length

    Returns:
        Tuple of (cos, sin) tensors in Meta format [1, 1, seq_len, head_dim]
    """
    rope_scaling = rope_scaling_model_factory(hf_config.rope_scaling)

    # Create rotary embedding using our factory (same code path as RotarySetup)
    rotary_emb = rotary_embedding_factory(
        dim=hf_config.head_dim,
        max_position_embeddings=seq_len,
        base=getattr(hf_config, "rope_theta", 150000.0),
        rope_scaling=rope_scaling,
    )

    # cos_cached and sin_cached are already in Meta format [1, 1, seq_len, head_dim]
    return rotary_emb.cos_cached, rotary_emb.sin_cached


def extract_unique_from_meta_format(cos_meta, sin_meta):
    """
    Extract unique cos/sin values from Meta format.

    Meta format: [1, 1, seq_len, head_dim] with interleaved duplicates (cos0, cos0, cos1, cos1, ...)
    Returns: [seq_len, head_dim // 2] with unique values (cos0, cos1, cos2, ...)

    Args:
        cos_meta: Cosine tensor in Meta format [1, 1, seq_len, head_dim]
        sin_meta: Sine tensor in Meta format [1, 1, seq_len, head_dim]

    Returns:
        Tuple of (cos_unique, sin_unique) tensors with shape [seq_len, head_dim // 2]
    """
    # Meta format has interleaved duplicates: [cos0, cos0, cos1, cos1, ...]
    cos_meta = cos_meta.squeeze(0).squeeze(0)  # [seq_len, head_dim]
    sin_meta = sin_meta.squeeze(0).squeeze(0)  # [seq_len, head_dim]

    # Get unique values (every other element since they're duplicated)
    cos_unique = cos_meta[..., 0::2]  # [seq_len, head_dim/2]
    sin_unique = sin_meta[..., 0::2]  # [seq_len, head_dim/2]

    return cos_unique, sin_unique


def extract_unique_from_hf_format(cos_hf, sin_hf):
    """
    Extract unique cos/sin values from HuggingFace format.

    The HF GptOssRotaryEmbedding returns cos/sin with shape [batch, seq_len, head_dim]
    where head_dim is already only the unique values (head_dim // 2 of the full dimension).

    Args:
        cos_hf: Cosine tensor in HF format [batch, seq_len, head_dim//2]
        sin_hf: Sine tensor in HF format [batch, seq_len, head_dim//2]

    Returns:
        Tuple of (cos_unique, sin_unique) tensors with shape [seq_len, head_dim//2]
    """
    # HF GptOssRotaryEmbedding returns [batch, seq_len, head_dim//2]
    # Squeeze batch dimension
    cos_unique = cos_hf.squeeze(0)  # [seq_len, head_dim//2]
    sin_unique = sin_hf.squeeze(0)  # [seq_len, head_dim//2]

    return cos_unique, sin_unique


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize(
    "seq_len",
    [128, 1024, 4096, 8192, 16384, 32768, 65536, 131072],
    ids=["seq_128", "seq_1024", "seq_4096", "seq_8192", "seq_16384", "seq_32768", "seq_65536", "seq_131072"],
)
@pytest.mark.parametrize(
    "max_local_batch_size",
    [1, 32],
    ids=["batch_1", "batch_32"],
)
def test_rope_vs_hf_reference(mesh_device, device_params, seq_len, max_local_batch_size, reset_seeds):
    """
    Test that our rope setup produces the same cos/sin matrices as the HuggingFace reference.

    This test:
    1. Creates a RotarySetup using create_rope_setup (same as Model.__init__)
    2. Gets the cos/sin matrices from HuggingFace transformers reference
    3. Compares them by extracting unique values from both formats

    The formats are different but should contain the same underlying values:
    - Our TT/Meta format: [1, 1, seq_len, head_dim] with interleaved duplicates [cos0, cos0, cos1, cos1, ...]
    - HF format: [batch, seq_len, head_dim//2] with unique values [cos0, cos1, cos2, ...]

    Args:
        mesh_device: TTNN mesh device fixture
        device_params: Device parameters fixture
        seq_len: Sequence length to test
        max_local_batch_size: Local batch size per device (1 or 32)
        reset_seeds: Reset random seeds fixture
    """
    import os

    model_path = os.getenv("HF_MODEL")
    if model_path is None:
        pytest.skip("HF_MODEL environment variable not set")

    # Load config from HF model
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    logger.info(f"Testing rope setup with seq_len={seq_len}, max_local_batch_size={max_local_batch_size}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Rope scaling config: {hf_config.rope_scaling}")
    logger.info(f"Rope theta: {getattr(hf_config, 'rope_theta', 'not set')}")
    logger.info(f"Head dim: {hf_config.head_dim}")

    # Create our rope setup using the refactored function
    rope_setup = create_rope_setup(
        mesh_device=mesh_device,
        hf_config=hf_config,
        max_local_batch_size=max_local_batch_size,
        users_row_sharded=False,
    )

    # Get cos/sin from our TT implementation
    cos_tt, sin_tt = get_rope_cos_sin_from_tt_setup(rope_setup, mesh_device, seq_len)

    # Get cos/sin from HuggingFace reference
    cos_hf, sin_hf = get_rope_cos_sin_from_hf_reference(hf_config, seq_len)

    # Extract unique values from both formats for fair comparison
    cos_tt_unique, sin_tt_unique = extract_unique_from_meta_format(cos_tt, sin_tt)
    cos_hf_unique, sin_hf_unique = extract_unique_from_hf_format(cos_hf, sin_hf)

    logger.info(f"TT cos unique shape: {cos_tt_unique.shape}, HF cos unique shape: {cos_hf_unique.shape}")
    logger.info(f"TT sin unique shape: {sin_tt_unique.shape}, HF sin unique shape: {sin_hf_unique.shape}")

    # Compare cos matrices
    cos_passing, cos_pcc = comp_pcc(cos_hf_unique.float(), cos_tt_unique.float(), 0.99)
    logger.info(f"Cosine matrix PCC: {cos_pcc}")

    # Compare sin matrices
    sin_passing, sin_pcc = comp_pcc(sin_hf_unique.float(), sin_tt_unique.float(), 0.99)
    logger.info(f"Sine matrix PCC: {sin_pcc}")

    # Log some sample values for debugging
    logger.info(f"TT cos unique[0, :8]: {cos_tt_unique[0, :8]}")
    logger.info(f"HF cos unique[0, :8]: {cos_hf_unique[0, :8]}")
    logger.info(f"TT sin unique[0, :8]: {sin_tt_unique[0, :8]}")
    logger.info(f"HF sin unique[0, :8]: {sin_hf_unique[0, :8]}")

    # Also check at a later position to verify scaling is applied correctly
    mid_pos = min(seq_len // 2, 512)
    logger.info(f"TT cos unique[{mid_pos}, :8]: {cos_tt_unique[mid_pos, :8]}")
    logger.info(f"HF cos unique[{mid_pos}, :8]: {cos_hf_unique[mid_pos, :8]}")

    # Assert both pass
    if not cos_passing:
        # Calculate max absolute error for debugging
        cos_error = torch.abs(cos_hf_unique.float() - cos_tt_unique.float())
        logger.error(f"Cosine max abs error: {cos_error.max()}")
        logger.error(f"Cosine mean abs error: {cos_error.mean()}")
        # Find the position with maximum error
        max_idx = torch.unravel_index(cos_error.argmax(), cos_error.shape)
        logger.error(f"Max error at position: {max_idx}")
        logger.error(f"TT value: {cos_tt_unique[max_idx]}, HF value: {cos_hf_unique[max_idx]}")

    if not sin_passing:
        sin_error = torch.abs(sin_hf_unique.float() - sin_tt_unique.float())
        logger.error(f"Sine max abs error: {sin_error.max()}")
        logger.error(f"Sine mean abs error: {sin_error.mean()}")

    assert cos_passing, f"Cosine matrix mismatch. PCC: {cos_pcc}"
    assert sin_passing, f"Sine matrix mismatch. PCC: {sin_pcc}"

    logger.info("✓ Rope setup matches HuggingFace reference!")


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize(
    "seq_len",
    [128, 1024, 4096, 8192, 16384, 32768, 65536, 131072],
    ids=["seq_128", "seq_1024", "seq_4096", "seq_8192", "seq_16384", "seq_32768", "seq_65536", "seq_131072"],
)
def test_rope_pytorch_vs_hf_reference(mesh_device, device_params, seq_len, reset_seeds):
    """
    Test that our PyTorch rope implementation produces the same cos/sin matrices as HuggingFace.

    This is a pure PyTorch test (no TTNN) to isolate potential bugs in our PyTorch
    rope implementation (rotary_embedding_factory) vs the HuggingFace reference.

    Args:
        mesh_device: TTNN mesh device fixture (needed for parametrization but not used)
        device_params: Device parameters fixture (needed for parametrization but not used)
        seq_len: Sequence length to test
        reset_seeds: Reset random seeds fixture
    """
    import os

    model_path = os.getenv("HF_MODEL")
    if model_path is None:
        pytest.skip("HF_MODEL environment variable not set")

    # Load config from HF model
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    logger.info(f"Testing PyTorch rope implementation with seq_len={seq_len}")
    logger.info(f"Rope scaling config: {hf_config.rope_scaling}")

    # Get cos/sin from our PyTorch implementation (in Meta format)
    cos_ours, sin_ours = get_rope_cos_sin_from_our_pytorch_impl(hf_config, seq_len)

    # Get cos/sin from HuggingFace reference
    cos_hf, sin_hf = get_rope_cos_sin_from_hf_reference(hf_config, seq_len)

    # Extract unique values from both formats for fair comparison
    cos_ours_unique, sin_ours_unique = extract_unique_from_meta_format(cos_ours, sin_ours)
    cos_hf_unique, sin_hf_unique = extract_unique_from_hf_format(cos_hf, sin_hf)

    logger.info(f"Our cos unique shape: {cos_ours_unique.shape}, HF cos unique shape: {cos_hf_unique.shape}")

    # Compare cos matrices
    cos_passing, cos_pcc = comp_pcc(cos_hf_unique.float(), cos_ours_unique.float(), 0.99)
    logger.info(f"Cosine matrix PCC: {cos_pcc}")

    # Compare sin matrices
    sin_passing, sin_pcc = comp_pcc(sin_hf_unique.float(), sin_ours_unique.float(), 0.99)
    logger.info(f"Sine matrix PCC: {sin_pcc}")

    # Log sample values for debugging
    logger.info(f"Our cos unique[0, :8]: {cos_ours_unique[0, :8]}")
    logger.info(f"HF cos unique[0, :8]: {cos_hf_unique[0, :8]}")

    # Check at a later position to verify scaling
    mid_pos = min(seq_len // 2, 512)
    logger.info(f"Our cos unique[{mid_pos}, :8]: {cos_ours_unique[mid_pos, :8]}")
    logger.info(f"HF cos unique[{mid_pos}, :8]: {cos_hf_unique[mid_pos, :8]}")

    if not cos_passing:
        cos_error = torch.abs(cos_hf_unique.float() - cos_ours_unique.float())
        logger.error(f"Cosine max abs error: {cos_error.max()}")
        logger.error(f"Cosine mean abs error: {cos_error.mean()}")
        # Show where the biggest differences are
        max_idx = torch.unravel_index(cos_error.argmax(), cos_error.shape)
        logger.error(f"Max error at position: {max_idx}")
        logger.error(f"Our value: {cos_ours_unique[max_idx]}, HF value: {cos_hf_unique[max_idx]}")

    if not sin_passing:
        sin_error = torch.abs(sin_hf_unique.float() - sin_ours_unique.float())
        logger.error(f"Sine max abs error: {sin_error.max()}")
        logger.error(f"Sine mean abs error: {sin_error.mean()}")

    assert cos_passing, f"Cosine matrix mismatch. PCC: {cos_pcc}"
    assert sin_passing, f"Sine matrix mismatch. PCC: {sin_pcc}"

    logger.info("✓ PyTorch rope implementation matches HuggingFace reference!")


@parametrize_mesh_with_fabric()
def test_rope_scaling_parameters(mesh_device, device_params, reset_seeds):
    """
    Test that rope scaling parameters are correctly parsed and applied.

    This test verifies:
    1. The rope_scaling_model_factory correctly parses the HF config
    2. The scaling parameters match what we expect for Yarn scaling

    Args:
        mesh_device: TTNN mesh device fixture
        device_params: Device parameters fixture
        reset_seeds: Reset random seeds fixture
    """
    import os

    model_path = os.getenv("HF_MODEL")
    if model_path is None:
        pytest.skip("HF_MODEL environment variable not set")

    # Load config from HF model
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    logger.info(f"Rope scaling config from HF: {hf_config.rope_scaling}")

    # Parse the rope scaling config
    rope_scaling = rope_scaling_model_factory(hf_config.rope_scaling)

    if rope_scaling is None:
        logger.info("No rope scaling configured")
        return

    logger.info(f"Parsed rope scaling type: {rope_scaling.rope_type}")
    logger.info(f"Parsed rope scaling parameters: {rope_scaling.model_dump()}")

    # Verify the scaling type is what we expect
    if hasattr(hf_config.rope_scaling, "get"):
        expected_type = hf_config.rope_scaling.get("rope_type") or hf_config.rope_scaling.get("type")
    else:
        expected_type = getattr(hf_config.rope_scaling, "rope_type", None) or getattr(
            hf_config.rope_scaling, "type", None
        )

    logger.info(f"Expected rope type: {expected_type}")
    logger.info(f"Actual rope type: {rope_scaling.rope_type.value}")

    assert (
        rope_scaling.rope_type.value == expected_type
    ), f"Rope type mismatch: expected {expected_type}, got {rope_scaling.rope_type.value}"

    # For Yarn scaling, verify specific parameters
    if rope_scaling.rope_type.value == "yarn":
        from models.tt_transformers.tt.common import RopeScalingYarn

        assert isinstance(rope_scaling, RopeScalingYarn), "Expected RopeScalingYarn for yarn type"

        logger.info(f"Yarn scaling factor: {rope_scaling.factor}")
        logger.info(f"Yarn original_max_position_embeddings: {rope_scaling.original_max_position_embeddings}")
        logger.info(f"Yarn beta_fast: {rope_scaling.beta_fast}")
        logger.info(f"Yarn beta_slow: {rope_scaling.beta_slow}")
        logger.info(f"Yarn mscale: {rope_scaling.mscale}")
        logger.info(f"Yarn mscale_all_dim: {rope_scaling.mscale_all_dim}")

    logger.info("✓ Rope scaling parameters parsed correctly!")


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize(
    "batch_size",
    [1, 32],
    ids=["batch_1", "batch_32"],
)
def test_rope_embedding_lookup_multi_user(mesh_device, device_params, batch_size, reset_seeds):
    """
    Test that the rope embedding lookup (get_rot_mats) returns correct cos/sin values
    for multiple users at different positions.

    This test verifies the full lookup mechanism:
    1. Creates a RotarySetup with specified batch_size
    2. Creates position indices for batch_size users at different positions
    3. Calls get_rot_mats to lookup cos/sin for those positions
    4. Compares returned values against HuggingFace reference for those same positions

    This is important because the main test only verifies the precomputed lookup tables,
    but doesn't verify the embedding lookup works correctly for multiple users.

    Args:
        mesh_device: TTNN mesh device fixture
        device_params: Device parameters fixture
        batch_size: Number of users (1 or 32)
        reset_seeds: Reset random seeds fixture
    """
    import os

    model_path = os.getenv("HF_MODEL")
    if model_path is None:
        pytest.skip("HF_MODEL environment variable not set")

    # Load config from HF model
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Use a reasonable max sequence length for the test
    max_seq_len = 8192

    logger.info(f"Testing rope embedding lookup with batch_size={batch_size}")
    logger.info(f"Rope scaling config: {hf_config.rope_scaling}")

    # Create our rope setup
    rope_setup = create_rope_setup(
        mesh_device=mesh_device,
        hf_config=hf_config,
        max_local_batch_size=batch_size,
        users_row_sharded=False,
    )

    # Create position indices for each user - spread them across the sequence
    # User 0 at position 0, user 1 at position 100, user 2 at position 200, etc.
    position_step = max_seq_len // (batch_size + 1)
    position_idxs = torch.tensor([i * position_step for i in range(batch_size)], dtype=torch.int64)

    logger.info(f"Testing positions: {position_idxs.tolist()}")

    # Get cos/sin from our TT implementation via embedding lookup
    cos_sin_tt = rope_setup.get_rot_mats(position_idxs)
    cos_tt = cos_sin_tt[0]  # [1, batch, 1, head_dim] - replicated across mesh, sharded across cores
    sin_tt = cos_sin_tt[1]  # [1, batch, 1, head_dim] - replicated across mesh, sharded across cores

    # The tensor is replicated across devices but sharded across cores on each device.
    # We only need to check one device since they all have the same data.
    # The actual batch dimension after sharding may be padded to tile size.
    cos_tt_torch = ttnn.to_torch(ttnn.get_device_tensors(cos_tt)[0])
    sin_tt_torch = ttnn.to_torch(ttnn.get_device_tensors(sin_tt)[0])

    logger.info(f"TT cos shape after lookup (from device 0): {cos_tt_torch.shape}")
    logger.info(f"TT sin shape after lookup (from device 0): {sin_tt_torch.shape}")

    # Get HF reference cos/sin for comparison
    cos_hf_full, sin_hf_full = get_rope_cos_sin_from_hf_reference(hf_config, max_seq_len)

    # Extract unique values from HF format
    cos_hf_unique, sin_hf_unique = extract_unique_from_hf_format(cos_hf_full, sin_hf_full)

    # The tensor is padded to tile size (32). Only the first batch_size users have valid data.
    # However, the actual number of valid users depends on how many fit in the tensor.
    actual_batch_in_tensor = cos_tt_torch.shape[1]
    users_to_compare = min(batch_size, actual_batch_in_tensor)

    logger.info(f"Comparing {users_to_compare} users (requested={batch_size}, in_tensor={actual_batch_in_tensor})")

    # For each user, compare the cos/sin values at their position
    for user_idx in range(users_to_compare):
        pos = position_idxs[user_idx].item()

        # Get TT values for this user - shape is [1, batch, 1, head_dim] in Meta format
        # Extract unique values (every other element since they're duplicated)
        cos_tt_user = cos_tt_torch[0, user_idx, 0, ::2]  # [head_dim/2]
        sin_tt_user = sin_tt_torch[0, user_idx, 0, ::2]  # [head_dim/2]

        # Get HF reference values for this position
        cos_hf_pos = cos_hf_unique[pos, :]  # [head_dim/2]
        sin_hf_pos = sin_hf_unique[pos, :]  # [head_dim/2]

        # Compare
        cos_passing, cos_pcc = comp_pcc(cos_hf_pos.float(), cos_tt_user.float(), 0.99)
        sin_passing, sin_pcc = comp_pcc(sin_hf_pos.float(), sin_tt_user.float(), 0.99)

        if user_idx < 4 or user_idx == users_to_compare - 1:  # Log first few and last user
            logger.info(f"User {user_idx} at position {pos}: cos PCC={cos_pcc:.6f}, sin PCC={sin_pcc:.6f}")

        if not cos_passing:
            cos_error = torch.abs(cos_hf_pos.float() - cos_tt_user.float())
            logger.error(f"User {user_idx} at position {pos}: Cosine max error: {cos_error.max()}")
            logger.error(f"TT cos[0:4]: {cos_tt_user[:4]}")
            logger.error(f"HF cos[0:4]: {cos_hf_pos[:4]}")

        if not sin_passing:
            sin_error = torch.abs(sin_hf_pos.float() - sin_tt_user.float())
            logger.error(f"User {user_idx} at position {pos}: Sine max error: {sin_error.max()}")

        assert cos_passing, f"User {user_idx} at position {pos}: Cosine mismatch. PCC: {cos_pcc}"
        assert sin_passing, f"User {user_idx} at position {pos}: Sine mismatch. PCC: {sin_pcc}"

    logger.info(f"✓ All {users_to_compare} users' rope embedding lookups match HuggingFace reference!")
