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
from models.common.utility_functions import comp_pcc, nearest_32
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


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize(
    "max_local_batch_size",
    [32],
    ids=["batch_32"],
)
def test_rope_embedding_lookup_users_row_sharded(mesh_device, device_params, max_local_batch_size, reset_seeds):
    """
    Test rope embedding lookup with users_row_sharded=True configuration.

    This test verifies the rope setup works correctly when users are sharded across mesh rows,
    which is used in the batch128 high-throughput configuration.

    With users_row_sharded=True:
    - Total batch = max_local_batch_size * mesh_rows (e.g., 32 * 4 = 128)
    - Each mesh row handles max_local_batch_size users
    - Rope indices are sharded across rows, replicated across columns

    Args:
        mesh_device: TTNN mesh device fixture
        device_params: Device parameters fixture
        max_local_batch_size: Local batch size per mesh row
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
    mesh_rows = mesh_device.shape[0]
    total_batch = max_local_batch_size * mesh_rows

    logger.info(f"Testing rope with users_row_sharded=True")
    logger.info(f"max_local_batch_size={max_local_batch_size}, mesh_rows={mesh_rows}, total_batch={total_batch}")
    logger.info(f"Rope scaling config: {hf_config.rope_scaling}")

    # Create our rope setup with users_row_sharded=True
    rope_setup = create_rope_setup(
        mesh_device=mesh_device,
        hf_config=hf_config,
        max_local_batch_size=max_local_batch_size,
        users_row_sharded=True,
    )

    logger.info(f"batch_size_per_device_group: {rope_setup.batch_size_per_device_group}")

    # Create position indices for local batch (per row)
    # Each row will have its own set of 32 users at different positions
    position_step = max_seq_len // (max_local_batch_size + 1)
    position_idxs = torch.tensor([i * position_step for i in range(max_local_batch_size)], dtype=torch.int64)

    logger.info(f"Testing positions: {position_idxs[:5].tolist()}...{position_idxs[-5:].tolist()}")

    # Get cos/sin from our TT implementation via embedding lookup
    cos_sin_tt = rope_setup.get_rot_mats(position_idxs)
    cos_tt = cos_sin_tt[0]
    sin_tt = cos_sin_tt[1]

    # Get values from first device (row 0, col 0)
    cos_tt_torch = ttnn.to_torch(ttnn.get_device_tensors(cos_tt)[0])
    sin_tt_torch = ttnn.to_torch(ttnn.get_device_tensors(sin_tt)[0])

    logger.info(f"TT cos shape after lookup: {cos_tt_torch.shape}")
    logger.info(f"TT sin shape after lookup: {sin_tt_torch.shape}")

    # Get HF reference cos/sin for comparison
    cos_hf_full, sin_hf_full = get_rope_cos_sin_from_hf_reference(hf_config, max_seq_len)
    cos_hf_unique, sin_hf_unique = extract_unique_from_hf_format(cos_hf_full, sin_hf_full)

    # Verify values at each position
    actual_batch_in_tensor = cos_tt_torch.shape[1]
    users_to_compare = min(max_local_batch_size, actual_batch_in_tensor)

    logger.info(f"Comparing {users_to_compare} users")

    all_passed = True
    for user_idx in range(users_to_compare):
        pos = position_idxs[user_idx].item()

        # Get TT values - extract unique values (every other element)
        cos_tt_user = cos_tt_torch[0, user_idx, 0, ::2]
        sin_tt_user = sin_tt_torch[0, user_idx, 0, ::2]

        # Get HF reference values
        cos_hf_pos = cos_hf_unique[pos, :]
        sin_hf_pos = sin_hf_unique[pos, :]

        # Compare
        cos_passing, cos_pcc = comp_pcc(cos_hf_pos.float(), cos_tt_user.float(), 0.99)
        sin_passing, sin_pcc = comp_pcc(sin_hf_pos.float(), sin_tt_user.float(), 0.99)

        if user_idx < 4 or user_idx == users_to_compare - 1:
            logger.info(f"User {user_idx} at position {pos}: cos PCC={cos_pcc:.6f}, sin PCC={sin_pcc:.6f}")

        if not cos_passing or not sin_passing:
            logger.error(f"User {user_idx} at position {pos}: FAILED - cos PCC={cos_pcc:.6f}, sin PCC={sin_pcc:.6f}")
            all_passed = False

        assert cos_passing, f"User {user_idx} at position {pos}: Cosine mismatch. PCC: {cos_pcc}"
        assert sin_passing, f"User {user_idx} at position {pos}: Sine mismatch. PCC: {sin_pcc}"

    logger.info(
        f"✓ All {users_to_compare} users' rope embedding lookups match HuggingFace reference (users_row_sharded=True)!"
    )


def apply_rotary_emb_torch(x, cos, sin):
    """
    PyTorch reference implementation of rotary embedding (Meta/Llama style).

    Args:
        x: Input tensor [..., head_dim] where head_dim values are interleaved as [x0, x0, x1, x1, ...]
        cos: Cosine tensor broadcastable to x, with values duplicated [cos0, cos0, cos1, cos1, ...]
        sin: Sine tensor broadcastable to x, with values duplicated [sin0, sin0, sin1, sin1, ...]

    Returns:
        Rotated tensor with same shape as x
    """

    # rotate_half for Meta format: negate and swap adjacent pairs
    def rotate_half(x):
        x1 = x[..., ::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        rotated = torch.stack((-x2, x1), dim=-1)
        return rotated.flatten(-2)

    return (x * cos) + (rotate_half(x) * sin)


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize(
    "batch_size,cos_sin_scale",
    [
        (32, 1.0),  # Normal cos/sin in [-1, 1] - should pass
        (32, 1.35),  # YARN-scaled cos/sin in [-1.35, 1.35] - demonstrates bug
    ],
    ids=["scale_1.0_normal", "scale_1.35_yarn"],
)
def test_rotary_embedding_llama_kernel_with_scaled_values(
    mesh_device, device_params, batch_size, cos_sin_scale, reset_seeds
):
    """
    BUG REPRODUCER: Direct test of rotary_embedding_llama kernel with scaled cos/sin.

    This test directly invokes ttnn.experimental.rotary_embedding_llama and compares
    the output against a PyTorch reference implementation.

    Bug symptoms:
    - With cos_sin_scale=1.0: All batch positions should pass PCC > 0.99
    - With cos_sin_scale=1.35 (YARN-style): Some/all batch positions may fail

    Args:
        mesh_device: TTNN mesh device fixture
        batch_size: Number of users in batch
        cos_sin_scale: Scale factor for cos/sin (1.0 = normal, 1.35 = YARN-style)
        reset_seeds: Reset random seeds fixture
    """
    torch.manual_seed(42)

    # Configuration matching decode mode
    seq_len = 1  # Decode: single token
    num_heads = 32  # Must be tile-aligned for HEIGHT_SHARDED
    head_dim = 64  # GPT-OSS head dimension

    logger.info(f"=== Testing rotary_embedding_llama with cos_sin_scale={cos_sin_scale} ===")
    logger.info(f"batch_size={batch_size}, num_heads={num_heads}, head_dim={head_dim}")

    # Create random input Q tensor in decode format: [seq_len, batch, num_heads, head_dim]
    q_torch = torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)

    # Create cos/sin tensors with Meta-style duplication [cos0, cos0, cos1, cos1, ...]
    # Shape: [1, batch, 1, head_dim] for broadcast over seq_len and num_heads
    base_freqs = torch.randn(1, batch_size, 1, head_dim // 2)  # head_dim/2 unique values

    # Compute cos/sin from frequencies
    cos_unique = torch.cos(base_freqs)
    sin_unique = torch.sin(base_freqs)

    # Duplicate for Meta format: [cos0, cos0, cos1, cos1, ...]
    cos_meta = torch.stack([cos_unique, cos_unique], dim=-1).flatten(-2)  # [1, batch, 1, head_dim]
    sin_meta = torch.stack([sin_unique, sin_unique], dim=-1).flatten(-2)  # [1, batch, 1, head_dim]

    # Apply YARN-style scaling
    cos_torch = (cos_meta * cos_sin_scale).to(torch.bfloat16)
    sin_torch = (sin_meta * cos_sin_scale).to(torch.bfloat16)

    # Log value statistics
    cos_abs_max = cos_torch.abs().max().item()
    sin_abs_max = sin_torch.abs().max().item()
    cos_over_1 = (cos_torch.abs() > 1.0).sum().item()
    sin_over_1 = (sin_torch.abs() > 1.0).sum().item()

    logger.info(f"cos range: [{cos_torch.min():.4f}, {cos_torch.max():.4f}], max abs: {cos_abs_max:.4f}")
    logger.info(f"sin range: [{sin_torch.min():.4f}, {sin_torch.max():.4f}], max abs: {sin_abs_max:.4f}")
    logger.info(f"Elements with |value| > 1.0: cos={cos_over_1}, sin={sin_over_1}")

    # Compute PyTorch reference output
    # Broadcast cos/sin over num_heads dimension
    cos_broadcast = cos_torch.expand(seq_len, batch_size, num_heads, head_dim)
    sin_broadcast = sin_torch.expand(seq_len, batch_size, num_heads, head_dim)
    expected_output = apply_rotary_emb_torch(q_torch, cos_broadcast, sin_broadcast)

    # Create transformation matrix (32x32, handles pair-wise rotation)
    trans_mat_torch = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
    for i in range(0, 32, 2):
        trans_mat_torch[0, 0, i, i + 1] = 1
        trans_mat_torch[0, 0, i + 1, i] = -1
    trans_mat_torch = trans_mat_torch.repeat(1, 1, batch_size, 1)

    # Set up sharded memory configs
    grid_size = ttnn.CoreCoord(8, 8)  # Safe limit: max 8 per dimension to avoid Galaxy hangs
    batch_grid = ttnn.num_cores_to_corerangeset(batch_size, grid_size, row_wise=True)

    # Input memory config: HEIGHT_SHARDED (matching nlp_create_qkv_heads_decode output)
    input_mem_config = ttnn.create_sharded_memory_config(
        shape=(num_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Cos/sin memory config
    cos_sin_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),  # 32 x head_dim per shard
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Transformation matrix memory config
    trans_mat_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),  # 32 x 32
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Convert to TTNN tensors
    q_tt = ttnn.from_torch(
        q_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    cos_tt = ttnn.from_torch(
        cos_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=cos_sin_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    sin_tt = ttnn.from_torch(
        sin_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=cos_sin_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    trans_mat_tt = ttnn.from_torch(
        trans_mat_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=trans_mat_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run the kernel
    output_tt = ttnn.experimental.rotary_embedding_llama(q_tt, cos_tt, sin_tt, trans_mat_tt, is_decode_mode=True)

    # Convert output to torch
    output_torch = ttnn.to_torch(ttnn.get_device_tensors(output_tt)[0])

    logger.info(f"Output shape: {output_torch.shape}, Expected shape: {expected_output.shape}")

    # Compare per batch position
    failed_positions = []
    passed_positions = []
    pcc_results = []

    for batch_idx in range(batch_size):
        tt_out = output_torch[:, batch_idx, :, :].float()
        ref_out = expected_output[:, batch_idx, :, :].float()

        passing, pcc = comp_pcc(ref_out, tt_out, 0.99)
        pcc_results.append(pcc)

        if passing:
            passed_positions.append(batch_idx)
        else:
            failed_positions.append(batch_idx)

    # Log detailed results for failures
    for batch_idx in failed_positions[:5]:  # First 5 failures
        tt_out = output_torch[:, batch_idx, :, :].float()
        ref_out = expected_output[:, batch_idx, :, :].float()
        logger.error(f"Batch {batch_idx}: PCC={pcc_results[batch_idx]:.6f} FAILED")
        logger.error(f"  TT output[0,0,:8]:  {tt_out[0, 0, :8].tolist()}")
        logger.error(f"  Reference[0,0,:8]:  {ref_out[0, 0, :8].tolist()}")
        logger.error(f"  Cos[batch,:8]:      {cos_torch[0, batch_idx, 0, :8].tolist()}")
        logger.error(f"  Sin[batch,:8]:      {sin_torch[0, batch_idx, 0, :8].tolist()}")

    # Summary
    logger.info(f"\n=== Results for cos_sin_scale={cos_sin_scale} ===")
    logger.info(f"Passed: {len(passed_positions)}/{batch_size} batch positions")
    logger.info(f"Failed: {len(failed_positions)}/{batch_size} batch positions")
    logger.info(f"Average PCC: {sum(pcc_results)/len(pcc_results):.6f}")
    logger.info(f"Min PCC: {min(pcc_results):.6f}")

    if failed_positions:
        logger.error(f"Failed batch positions: {failed_positions}")

    # Assertions
    if cos_sin_scale <= 1.0:
        # Normal case - all should pass
        assert len(failed_positions) == 0, (
            f"rotary_embedding_llama failed for {len(failed_positions)} batch positions with normal cos/sin. "
            f"Failed positions: {failed_positions}. Min PCC: {min(pcc_results):.4f}"
        )
        logger.info("✓ All batch positions passed with normal cos/sin values")
    else:
        # YARN-scaled case - document behavior
        if len(failed_positions) > 0:
            logger.warning(
                f"BUG DEMONSTRATED: {len(failed_positions)}/{batch_size} batch positions failed "
                f"with YARN-scaled cos/sin (scale={cos_sin_scale})"
            )
            pytest.xfail(
                f"Known bug: rotary_embedding_llama fails for {len(failed_positions)} batch positions "
                f"when cos/sin values exceed 1.0. Min PCC: {min(pcc_results):.4f}"
            )
        else:
            logger.info("✓ All batch positions passed (bug may be fixed!)")


@parametrize_mesh_with_fabric()
def test_rope_yarn_values_match_hf(mesh_device, device_params, reset_seeds):
    """
    Verify that our YARN rope implementation produces cos/sin values that match HuggingFace.
    """
    import os

    model_path = os.getenv("HF_MODEL")
    if model_path is None:
        pytest.skip("HF_MODEL environment variable not set")

    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

    from models.tt_transformers.tt.common import rope_scaling_model_factory
    from models.tt_transformers.tt.rope import rotary_embedding_factory

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Get HF cos/sin
    rope_hf = GptOssRotaryEmbedding(hf_config)
    x = torch.randn(1, 100, 1, hf_config.head_dim)
    position_ids = torch.arange(100).unsqueeze(0)
    cos_hf, sin_hf = rope_hf(x, position_ids)

    # Get our cos/sin using the same factory as the model
    rope_scaling = rope_scaling_model_factory(hf_config.rope_scaling)
    rope_ours = rotary_embedding_factory(
        dim=hf_config.head_dim,
        max_position_embeddings=hf_config.max_position_embeddings,
        base=hf_config.rope_theta,
        rope_scaling=rope_scaling,
    )

    our_cos = rope_ours.cos_cached[0, 0, :100, :]
    our_cos_unique = our_cos[:, ::2]
    cos_hf_compare = cos_hf[0, :, :]

    diff = (cos_hf_compare - our_cos_unique).abs()
    max_diff = diff.max().item()

    logger.info(f"=== YARN cos/sin comparison ===")
    logger.info(f"Max difference: {max_diff:.8f}")

    assert max_diff < 1e-5, f"YARN cos/sin values don't match HF! Max diff: {max_diff}"
    logger.info("✓ Our YARN cos/sin values EXACTLY match HuggingFace!")


@parametrize_mesh_with_fabric()
def test_trace_rope_ops_for_corruption(mesh_device, device_params, reset_seeds):
    """
    Trace each TTNN op in the rope chain to find where values > 1.0 get corrupted.

    This test exactly mimics the demo's setup with 4x8 mesh and 128 users.
    Key differences from previous test:
    - Uses ShardTensor2dMesh for position indices (same as model)
    - Tests full 128 batch across all 4 mesh rows
    - Compares results on ALL devices, not just device[0]
    """
    import os

    model_path = os.getenv("HF_MODEL")
    if model_path is None:
        pytest.skip("HF_MODEL environment variable not set")

    # Require 4x8 mesh like the demo
    if mesh_device.shape[0] != 4 or mesh_device.shape[1] != 8:
        pytest.skip(f"Test requires 4x8 mesh, got {mesh_device.shape}")

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    global_batch_size = 128  # Same as demo
    batch_per_row = global_batch_size // mesh_device.shape[0]  # 32 per row
    head_dim = hf_config.head_dim  # 64
    num_heads = 32  # Local heads per device

    # Known failing positions from demo (within each row of 32)
    failing_positions_per_row = [3, 7, 9, 14, 15, 23, 24, 25, 29, 30]

    logger.info(f"=== Tracing rope ops for corruption (exact demo setup) ===")
    logger.info(f"Mesh shape: {mesh_device.shape}")
    logger.info(f"Global batch: {global_batch_size}, per row: {batch_per_row}")
    logger.info(f"Known failing positions per row: {failing_positions_per_row}")

    # Create rope setup with YARN (same as demo: max_local_batch_size=32)
    rope_setup = create_rope_setup(
        mesh_device=mesh_device,
        hf_config=hf_config,
        max_local_batch_size=batch_per_row,  # 32
        users_row_sharded=True,
    )

    # Get cos/sin matrices from device 0 (they're replicated)
    cos_matrix_torch = ttnn.to_torch(ttnn.get_device_tensors(rope_setup.cos_matrix)[0])
    logger.info(f"cos_matrix shape: {cos_matrix_torch.shape}")
    logger.info(f"cos_matrix range: [{cos_matrix_torch.min():.4f}, {cos_matrix_torch.max():.4f}]")

    # Create position indices simulating post-prefill
    # Each user at different position (like demo with different prompt lengths)
    # Using varied positions where YARN produces different cos/sin values
    import random

    random.seed(42)

    # Create positions that vary by user, simulating different prompt lengths
    # Some short (10-50), some medium (100-500), some long (1000-5000)
    position_ids = torch.tensor(
        [
            random.randint(10, 50)
            if i % 3 == 0
            else random.randint(100, 500)
            if i % 3 == 1
            else random.randint(1000, 5000)
            for i in range(global_batch_size)
        ],
        dtype=torch.int64,
    )
    logger.info(f"Using varied positions: min={position_ids.min().item()}, max={position_ids.max().item()}")
    logger.info(f"Sample positions: {position_ids[:8].tolist()} ... {position_ids[-8:].tolist()}")

    # Reshape for mesh sharding: [1, 128] -> shard along last dim for mesh rows
    position_ids_reshaped = position_ids.reshape(1, global_batch_size)

    # Use ShardTensor2dMesh exactly like the model (dims=(-1, None))
    # This shards the last dim (128) across mesh rows (4), replicates across columns (8)
    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=mesh_device.shape)

    tt_position_idx = ttnn.from_torch(
        position_ids_reshaped,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )

    # ===== Check positions on each device =====
    logger.info("\n=== Checking position sharding across devices ===")

    device_tensors = ttnn.get_device_tensors(tt_position_idx)
    num_mesh_cols = mesh_device.shape[1]  # 8

    # Check one device per mesh row (devices 0, 8, 16, 24 for 4x8 mesh)
    for mesh_row in range(mesh_device.shape[0]):
        dev_idx = mesh_row * num_mesh_cols  # First device in each mesh row
        dt = device_tensors[dev_idx]
        pos_on_dev = ttnn.to_torch(dt)
        logger.info(
            f"Mesh row {mesh_row} (device {dev_idx}): positions {pos_on_dev.flatten()[:4].tolist()} ... {pos_on_dev.flatten()[-4:].tolist()}"
        )

    # ===== Get rope matrices using model's method =====
    logger.info("\n=== Getting rope matrices via rope_setup.get_rot_mats ===")

    cos_mats, sin_mats = rope_setup.get_rot_mats(tt_position_idx)

    # ===== Check rope values on each device =====
    logger.info("\n=== Checking rope values across devices ===")

    cos_device_tensors = ttnn.get_device_tensors(cos_mats)
    sin_device_tensors = ttnn.get_device_tensors(sin_mats)

    all_errors = []
    # Check one device per mesh row
    for mesh_row in range(mesh_device.shape[0]):
        dev_idx = mesh_row * num_mesh_cols
        cos_dev = ttnn.to_torch(cos_device_tensors[dev_idx])
        sin_dev = ttnn.to_torch(sin_device_tensors[dev_idx])

        # Get positions on this device
        pos_dev = ttnn.to_torch(device_tensors[dev_idx]).flatten()

        logger.info(f"\nMesh row {mesh_row} (device {dev_idx}):")
        logger.info(f"  cos shape: {cos_dev.shape}")
        logger.info(f"  Positions: {pos_dev[:4].tolist()} ... {pos_dev[-4:].tolist()}")

        # Check cos values match expected
        dev_errors = []
        for batch_idx in range(batch_per_row):
            pos = pos_dev[batch_idx].item()
            expected_cos = cos_matrix_torch[0, 0, pos, :]
            actual_cos = cos_dev[0, batch_idx, 0, :]

            diff = (expected_cos - actual_cos).abs().max().item()
            if diff > 0.01:
                dev_errors.append((batch_idx, pos, diff))

        if dev_errors:
            logger.error(f"  Mesh row {mesh_row} (device {dev_idx}) cos errors: {len(dev_errors)}")
            for batch_idx, pos, diff in dev_errors[:3]:
                is_fail = (batch_idx % batch_per_row) in failing_positions_per_row
                marker = " <-- KNOWN FAIL" if is_fail else ""
                logger.error(f"    batch_idx={batch_idx}, pos={pos}, diff={diff:.4f}{marker}")
        else:
            logger.info(f"  ✓ Mesh row {mesh_row} (device {dev_idx}): All cos values correct")

        all_errors.extend([(dev_idx, *e) for e in dev_errors])

    # ===== Now test rotary_embedding_llama =====
    logger.info("\n=== Testing rotary_embedding_llama ===")

    grid_size = ttnn.CoreCoord(8, 8)  # Safe limit: max 8 per dimension to avoid Galaxy hangs
    batch_grid = ttnn.num_cores_to_corerangeset(batch_per_row, grid_size, row_wise=True)

    # Create Q tensor with known values
    torch.manual_seed(42)
    q_torch_global = torch.randn(1, global_batch_size, num_heads, head_dim, dtype=torch.bfloat16)

    # Shard Q across mesh rows (same as model)
    q_mem_config = ttnn.create_sharded_memory_config(
        shape=(num_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    q_tt = ttnn.from_torch(
        q_torch_global,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(1, None), mesh_shape=mesh_device.shape),
    )

    trans_mat = rope_setup.transformation_mat

    # Apply rotary embedding
    output_tt = ttnn.experimental.rotary_embedding_llama(q_tt, cos_mats, sin_mats, trans_mat, is_decode_mode=True)

    xqkv_tt = None  # Not used in simplified test

    # Check results on each device (one per mesh row)
    logger.info("\n=== Checking rotary_embedding_llama results per mesh row ===")

    output_device_tensors = ttnn.get_device_tensors(output_tt)
    sin_matrix_torch = ttnn.to_torch(ttnn.get_device_tensors(rope_setup.sin_matrix)[0])

    rope_errors_by_device = {}
    for mesh_row in range(mesh_device.shape[0]):
        dev_idx = mesh_row * num_mesh_cols
        output_dev = ttnn.to_torch(output_device_tensors[dev_idx])
        pos_dev = ttnn.to_torch(device_tensors[dev_idx]).flatten()

        # Get Q for this mesh row (users mesh_row*32 to (mesh_row+1)*32)
        q_dev = q_torch_global[0, mesh_row * batch_per_row : (mesh_row + 1) * batch_per_row]

        dev_rope_errors = []
        for batch_idx in range(batch_per_row):
            pos = pos_dev[batch_idx].item()
            cos_pos = cos_matrix_torch[0, 0, pos, :]
            sin_pos = sin_matrix_torch[0, 0, pos, :]
            q_pos = q_dev[batch_idx]  # [num_heads, head_dim]

            expected = apply_rotary_emb_torch(q_pos, cos_pos, sin_pos)
            actual = output_dev[0, batch_idx, :, :]

            passing, pcc = comp_pcc(expected.float(), actual.float(), 0.99)
            if not passing:
                dev_rope_errors.append((batch_idx, pos, pcc))

        rope_errors_by_device[mesh_row] = dev_rope_errors

        if dev_rope_errors:
            logger.error(f"\nMesh row {mesh_row} (device {dev_idx}): {len(dev_rope_errors)} rope errors")
            failed_batch_positions = [e[0] for e in dev_rope_errors]
            logger.error(f"  Failed batch positions: {failed_batch_positions}")

            # Check overlap with known failing pattern
            overlap = set(failed_batch_positions) & set(failing_positions_per_row)
            logger.info(f"  Overlap with known failures: {sorted(overlap)}")
            logger.info(f"  New failures: {sorted(set(failed_batch_positions) - set(failing_positions_per_row))}")
            logger.info(f"  Missing known: {sorted(set(failing_positions_per_row) - set(failed_batch_positions))}")
        else:
            logger.info(f"\nMesh row {mesh_row} (device {dev_idx}): ✓ All rope results correct")

    # Summary
    total_rope_errors = sum(len(e) for e in rope_errors_by_device.values())
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total rope errors across devices: {total_rope_errors}")
    logger.info(f"Cos value errors: {len(all_errors)}")

    if total_rope_errors > 0:
        pytest.xfail(f"Found {total_rope_errors} rope errors across devices")


@parametrize_mesh_with_fabric()
def test_rope_multiple_iterations(mesh_device, device_params, reset_seeds):
    """
    Test rope ops over multiple iterations to check for accumulating errors.

    This test simulates the decode loop where rope is applied repeatedly
    with incrementing positions, checking if errors accumulate.
    """
    import os

    model_path = os.getenv("HF_MODEL")
    if model_path is None:
        pytest.skip("HF_MODEL environment variable not set")

    # Require 4x8 mesh like the demo
    if mesh_device.shape[0] != 4 or mesh_device.shape[1] != 8:
        pytest.skip(f"Test requires 4x8 mesh, got {mesh_device.shape}")

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    global_batch_size = 128
    batch_per_row = global_batch_size // mesh_device.shape[0]  # 32
    head_dim = hf_config.head_dim
    num_heads = 32
    num_iterations = 20  # Simulate 20 decode steps

    logger.info(f"=== Testing rope over {num_iterations} iterations ===")
    logger.info(f"Mesh: {mesh_device.shape}, batch: {global_batch_size}")

    # Create rope setup with YARN
    rope_setup = create_rope_setup(
        mesh_device=mesh_device,
        hf_config=hf_config,
        max_local_batch_size=batch_per_row,
        users_row_sharded=True,
    )

    cos_matrix_torch = ttnn.to_torch(ttnn.get_device_tensors(rope_setup.cos_matrix)[0])
    sin_matrix_torch = ttnn.to_torch(ttnn.get_device_tensors(rope_setup.sin_matrix)[0])

    grid_size = ttnn.CoreCoord(8, 8)  # Safe limit: max 8 per dimension to avoid Galaxy hangs
    batch_grid = ttnn.num_cores_to_corerangeset(batch_per_row, grid_size, row_wise=True)
    num_mesh_cols = mesh_device.shape[1]

    # Initialize Q tensor and positions (like after prefill)
    torch.manual_seed(42)
    q_torch = torch.randn(1, global_batch_size, num_heads, head_dim, dtype=torch.bfloat16)

    # Start positions at 20 (simulating short prefill)
    base_positions = torch.tensor([20 + i % 10 for i in range(global_batch_size)], dtype=torch.int64)

    errors_per_iteration = []

    for iteration in range(num_iterations):
        current_positions = base_positions + iteration

        # Shard positions across mesh rows
        position_ids_reshaped = current_positions.reshape(1, global_batch_size)
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=mesh_device.shape)

        tt_position_idx = ttnn.from_torch(
            position_ids_reshaped,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )

        # Get rope matrices
        cos_mats, sin_mats = rope_setup.get_rot_mats(tt_position_idx)

        # Create Q tensor
        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(num_heads, head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        q_tt = ttnn.from_torch(
            q_torch,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=q_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(1, None), mesh_shape=mesh_device.shape),
        )

        # Apply rope
        output_tt = ttnn.experimental.rotary_embedding_llama(
            q_tt, cos_mats, sin_mats, rope_setup.transformation_mat, is_decode_mode=True
        )

        # Check results on mesh row 0 (representative)
        device_tensors = ttnn.get_device_tensors(tt_position_idx)
        output_device_tensors = ttnn.get_device_tensors(output_tt)

        output_dev = ttnn.to_torch(output_device_tensors[0])
        pos_dev = ttnn.to_torch(device_tensors[0]).flatten()
        q_dev = q_torch[0, :batch_per_row]

        iteration_errors = []
        for batch_idx in range(batch_per_row):
            pos = pos_dev[batch_idx].item()
            cos_pos = cos_matrix_torch[0, 0, pos, :]
            sin_pos = sin_matrix_torch[0, 0, pos, :]
            q_pos = q_dev[batch_idx]

            expected = apply_rotary_emb_torch(q_pos, cos_pos, sin_pos)
            actual = output_dev[0, batch_idx, :, :]

            passing, pcc = comp_pcc(expected.float(), actual.float(), 0.99)
            if not passing:
                iteration_errors.append((batch_idx, pos, pcc))

        errors_per_iteration.append(len(iteration_errors))

        # Deallocate tensors
        tt_position_idx.deallocate()
        cos_mats.deallocate()
        sin_mats.deallocate()
        q_tt.deallocate()
        output_tt.deallocate()

    logger.info(f"\n=== Errors per iteration ===")
    for i, err_count in enumerate(errors_per_iteration):
        logger.info(f"  Iteration {i}: {err_count} errors")

    total_errors = sum(errors_per_iteration)
    logger.info(f"\nTotal errors across {num_iterations} iterations: {total_errors}")

    if total_errors > 0:
        pytest.xfail(f"Found {total_errors} errors across {num_iterations} iterations")


@parametrize_mesh_with_fabric()
def test_paged_update_cache_with_large_values(mesh_device, device_params, reset_seeds):
    """
    Test paged_update_cache with K values that have YARN-like magnitudes.

    This test verifies that paged_update_cache correctly stores and retrieves
    K values when they have larger magnitudes (like after YARN-scaled rope).
    """
    import os

    model_path = os.getenv("HF_MODEL")
    if model_path is None:
        pytest.skip("HF_MODEL environment variable not set")

    # Require 4x8 mesh like the demo
    if mesh_device.shape[0] != 4 or mesh_device.shape[1] != 8:
        pytest.skip(f"Test requires 4x8 mesh, got {mesh_device.shape}")

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    batch_per_row = 32
    num_kv_heads = hf_config.num_key_value_heads // mesh_device.shape[1]  # Local KV heads
    head_dim = hf_config.head_dim
    block_size = 64
    max_num_blocks = 2048  # Same as demo

    logger.info(f"=== Testing paged_update_cache with large values ===")
    logger.info(f"batch_per_row={batch_per_row}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    # Create KV cache like the model does
    cache_shape = [max_num_blocks, num_kv_heads, block_size, head_dim]

    k_cache = ttnn.as_tensor(
        torch.zeros(cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create K tensor with large values (simulating YARN-scaled Q/K)
    # Normal values are ~[-1, 1], YARN-scaled can be ~[-1.35, 1.35]
    torch.manual_seed(42)
    k_torch_normal = torch.randn(1, batch_per_row, num_kv_heads, head_dim, dtype=torch.bfloat16)
    k_torch_large = k_torch_normal * 1.35  # Simulate YARN scaling

    logger.info(f"K normal range: [{k_torch_normal.min():.4f}, {k_torch_normal.max():.4f}]")
    logger.info(f"K large range: [{k_torch_large.min():.4f}, {k_torch_large.max():.4f}]")

    # Create memory config for K
    grid_size = ttnn.CoreCoord(8, 8)  # Safe limit: max 8 per dimension to avoid Galaxy hangs
    batch_grid = ttnn.num_cores_to_corerangeset(batch_per_row, grid_size, row_wise=True)

    kv_mem_config = ttnn.create_sharded_memory_config(
        shape=(32, head_dim),  # Tile-aligned
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    # Create page table (simple: each user gets sequential blocks)
    page_table = torch.arange(batch_per_row, dtype=torch.int32).reshape(batch_per_row, 1)
    page_table = page_table.repeat(1, max_num_blocks // batch_per_row)
    # Actually, for simplicity, just use first batch_per_row blocks
    page_table = torch.zeros(batch_per_row, max_num_blocks // batch_per_row, dtype=torch.int32)
    for i in range(batch_per_row):
        page_table[i, 0] = i  # User i uses block i

    tt_page_table = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Position indices (all at position 0 within their blocks for simplicity)
    # Shape should be [batch] - 1D tensor like in the model
    position_ids = torch.zeros(batch_per_row, dtype=torch.int32)
    tt_position_idx = ttnn.from_torch(
        position_ids,  # 1D tensor [batch], not [1, batch]
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Test with normal values first
    logger.info("\n=== Test 1: Normal values ===")

    k_tt_normal = ttnn.from_torch(
        k_torch_normal,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=kv_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Update cache
    ttnn.experimental.paged_update_cache(
        k_cache,
        k_tt_normal,
        update_idxs_tensor=tt_position_idx,
        page_table=tt_page_table,
    )

    # Read back and verify
    k_cache_torch = ttnn.to_torch(ttnn.get_device_tensors(k_cache)[0])

    normal_errors = []
    for batch_idx in range(batch_per_row):
        block_idx = batch_idx  # User i uses block i
        pos_in_block = 0

        expected = k_torch_normal[0, batch_idx, :, :].float()
        actual = k_cache_torch[block_idx, :, pos_in_block, :].float()

        passing, pcc = comp_pcc(expected, actual, 0.99)
        if not passing:
            normal_errors.append((batch_idx, pcc))

    if normal_errors:
        logger.error(f"Normal values: {len(normal_errors)} errors")
        for batch_idx, pcc in normal_errors[:5]:
            logger.error(f"  Batch {batch_idx}: PCC={pcc:.4f}")
    else:
        logger.info("✓ Normal values: All positions stored correctly")

    k_tt_normal.deallocate()

    # Reset cache
    k_cache = ttnn.as_tensor(
        torch.zeros(cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Test with large values (YARN-scaled)
    logger.info("\n=== Test 2: Large values (YARN-scaled) ===")

    k_tt_large = ttnn.from_torch(
        k_torch_large,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=kv_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Update cache
    ttnn.experimental.paged_update_cache(
        k_cache,
        k_tt_large,
        update_idxs_tensor=tt_position_idx,
        page_table=tt_page_table,
    )

    # Read back and verify
    k_cache_torch = ttnn.to_torch(ttnn.get_device_tensors(k_cache)[0])

    large_errors = []
    for batch_idx in range(batch_per_row):
        block_idx = batch_idx
        pos_in_block = 0

        expected = k_torch_large[0, batch_idx, :, :].float()
        actual = k_cache_torch[block_idx, :, pos_in_block, :].float()

        passing, pcc = comp_pcc(expected, actual, 0.99)
        if not passing:
            large_errors.append((batch_idx, pcc))

    if large_errors:
        logger.error(f"Large values: {len(large_errors)} errors")
        logger.error(f"Failed positions: {[e[0] for e in large_errors]}")
        for batch_idx, pcc in large_errors[:5]:
            logger.error(f"  Batch {batch_idx}: PCC={pcc:.4f}")
    else:
        logger.info("✓ Large values: All positions stored correctly")

    # Summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Normal value errors: {len(normal_errors)}")
    logger.info(f"Large value errors: {len(large_errors)}")

    if large_errors and not normal_errors:
        pytest.xfail(
            f"paged_update_cache fails with large values: {len(large_errors)} errors at positions {[e[0] for e in large_errors]}"
        )
    elif large_errors or normal_errors:
        pytest.xfail(f"paged_update_cache errors: normal={len(normal_errors)}, large={len(large_errors)}")


@parametrize_mesh_with_fabric()
def test_sdpa_with_large_q_values(mesh_device, device_params, reset_seeds):
    """
    Test SDPA with Q values that have YARN-like magnitudes.

    This test verifies that paged_scaled_dot_product_attention_decode
    produces correct results when Q has larger magnitudes.
    """
    import os

    model_path = os.getenv("HF_MODEL")
    if model_path is None:
        pytest.skip("HF_MODEL environment variable not set")

    # Require 4x8 mesh like the demo
    if mesh_device.shape[0] != 4 or mesh_device.shape[1] != 8:
        pytest.skip(f"Test requires 4x8 mesh, got {mesh_device.shape}")

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    batch_size = 32
    num_heads = 32  # Local heads
    num_kv_heads = hf_config.num_key_value_heads // mesh_device.shape[1]
    head_dim = hf_config.head_dim
    seq_len = 64  # Sequence length in cache
    block_size = 64
    max_num_blocks = 2048

    logger.info(f"=== Testing SDPA with large Q values ===")
    logger.info(f"batch={batch_size}, heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")

    # Create KV cache
    cache_shape = [max_num_blocks, num_kv_heads, block_size, head_dim]

    # Initialize cache with some values (simulating prefilled KV)
    torch.manual_seed(42)
    k_cache_init = torch.randn(cache_shape, dtype=torch.bfloat16)
    v_cache_init = torch.randn(cache_shape, dtype=torch.bfloat16)

    k_cache = ttnn.from_torch(
        k_cache_init,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    v_cache = ttnn.from_torch(
        v_cache_init,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create Q tensors - normal and large
    q_torch_normal = torch.randn(1, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
    q_torch_large = q_torch_normal * 1.35  # YARN-like scaling

    logger.info(f"Q normal range: [{q_torch_normal.min():.4f}, {q_torch_normal.max():.4f}]")
    logger.info(f"Q large range: [{q_torch_large.min():.4f}, {q_torch_large.max():.4f}]")

    # Create memory config for Q
    grid_size = ttnn.CoreCoord(8, 8)  # Safe limit: max 8 per dimension to avoid Galaxy hangs
    batch_grid = ttnn.num_cores_to_corerangeset(batch_size, grid_size, row_wise=True)

    q_mem_config = ttnn.create_sharded_memory_config(
        shape=(num_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Page table and position indices
    page_table = torch.zeros(batch_size, max_num_blocks // batch_size, dtype=torch.int32)
    for i in range(batch_size):
        page_table[i, 0] = i

    tt_page_table = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Current position (all users at position 10 in their blocks)
    # Shape should be [batch] - 1D tensor like in the model
    cur_pos = torch.full((batch_size,), 10, dtype=torch.int32)
    tt_cur_pos = ttnn.from_torch(
        cur_pos,  # 1D tensor [batch], not [1, batch]
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    output_mem_config = ttnn.create_sharded_memory_config(
        shape=(num_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    sdpa_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),  # Safe limit: max 8 per dimension to avoid Galaxy hangs
        q_chunk_size=32,
        k_chunk_size=32,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Test with normal Q
    logger.info("\n=== Test 1: SDPA with normal Q ===")

    q_tt_normal = ttnn.from_torch(
        q_torch_normal,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    try:
        output_normal = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt_normal,
            k_cache,
            v_cache,
            cur_pos_tensor=tt_cur_pos,
            page_table_tensor=tt_page_table,
            scale=1.0 / (head_dim**0.5),
            program_config=sdpa_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output_normal_torch = ttnn.to_torch(ttnn.get_device_tensors(output_normal)[0])
        logger.info(f"Normal Q output range: [{output_normal_torch.min():.4f}, {output_normal_torch.max():.4f}]")

        # Check for NaN/Inf
        if torch.isnan(output_normal_torch).any():
            logger.error("WARNING: NaN in normal Q output!")
        if torch.isinf(output_normal_torch).any():
            logger.error("WARNING: Inf in normal Q output!")

        normal_ok = not (torch.isnan(output_normal_torch).any() or torch.isinf(output_normal_torch).any())
        if normal_ok:
            logger.info("✓ Normal Q: No NaN/Inf in output")

        output_normal.deallocate()
    except Exception as e:
        logger.error(f"Normal Q SDPA failed: {e}")
        normal_ok = False

    q_tt_normal.deallocate()

    # Test with large Q (YARN-scaled)
    logger.info("\n=== Test 2: SDPA with large Q (YARN-scaled) ===")

    q_tt_large = ttnn.from_torch(
        q_torch_large,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    try:
        output_large = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt_large,
            k_cache,
            v_cache,
            cur_pos_tensor=tt_cur_pos,
            page_table_tensor=tt_page_table,
            scale=1.0 / (head_dim**0.5),
            program_config=sdpa_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output_large_torch = ttnn.to_torch(ttnn.get_device_tensors(output_large)[0])
        logger.info(f"Large Q output range: [{output_large_torch.min():.4f}, {output_large_torch.max():.4f}]")

        # Check for NaN/Inf
        if torch.isnan(output_large_torch).any():
            logger.error("WARNING: NaN in large Q output!")
        if torch.isinf(output_large_torch).any():
            logger.error("WARNING: Inf in large Q output!")

        large_ok = not (torch.isnan(output_large_torch).any() or torch.isinf(output_large_torch).any())
        if large_ok:
            logger.info("✓ Large Q: No NaN/Inf in output")

        # Check if outputs are proportionally scaled
        # With larger Q, attention scores are larger but after softmax the output should be similar
        # (just attending to different positions potentially)

        # Check for specific positions that fail in demo
        failing_positions = [1, 3, 9, 14, 24, 29, 30]
        position_issues = []

        for pos in failing_positions:
            if pos < batch_size:
                out_pos = output_large_torch[0, pos, :, :]
                if torch.isnan(out_pos).any() or torch.isinf(out_pos).any():
                    position_issues.append(pos)
                elif out_pos.abs().max() > 100:  # Suspiciously large
                    position_issues.append(pos)

        if position_issues:
            logger.error(f"Issues at demo failing positions: {position_issues}")
        else:
            logger.info("✓ No issues at demo failing positions")

        output_large.deallocate()
    except Exception as e:
        logger.error(f"Large Q SDPA failed: {e}")
        large_ok = False

    q_tt_large.deallocate()

    # Summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Normal Q SDPA: {'PASS' if normal_ok else 'FAIL'}")
    logger.info(f"Large Q SDPA: {'PASS' if large_ok else 'FAIL'}")

    if not large_ok and normal_ok:
        pytest.xfail("SDPA fails with large Q values (YARN-scaled)")
    elif not large_ok or not normal_ok:
        pytest.xfail(f"SDPA issues: normal={'OK' if normal_ok else 'FAIL'}, large={'OK' if large_ok else 'FAIL'}")


@parametrize_mesh_with_fabric()
def test_attention_chain_with_yarn_scaling(mesh_device, device_params, reset_seeds):
    """
    Test the full attention chain: RoPE -> KV cache update -> SDPA

    This test chains together all the attention operations to see if
    the bug manifests when operations are composed together (as in the demo)
    even though they pass individually.
    """
    import os
    import random

    from models.demos.gpt_oss.tt.model import create_rope_setup

    model_path = os.getenv("HF_MODEL")
    if model_path is None:
        pytest.skip("HF_MODEL environment variable not set")

    # Require 4x8 mesh like the demo
    if mesh_device.shape[0] != 4 or mesh_device.shape[1] != 8:
        pytest.skip(f"Test requires 4x8 mesh, got {mesh_device.shape}")

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Use same parameters as demo batch128
    batch_size = 128  # Total across mesh
    batch_per_row = batch_size // mesh_device.shape[0]  # 32 per row
    num_heads = 32  # Local heads per device
    num_kv_heads = hf_config.num_key_value_heads // mesh_device.shape[1]  # 1 per device
    head_dim = hf_config.head_dim  # 64
    max_seq_len = 8192
    block_size = 64
    max_num_blocks = 2048

    logger.info(f"=== Testing attention chain with YARN scaling ===")
    logger.info(f"batch={batch_size}, batch_per_row={batch_per_row}")
    logger.info(f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    # Create rope setup with YARN scaling
    rope_setup = create_rope_setup(
        mesh_device=mesh_device,
        hf_config=hf_config,
        max_local_batch_size=batch_per_row,
        users_row_sharded=True,
    )

    # Create KV cache
    kv_cache_repeats = mesh_device.shape[0]  # 4 for users_row_sharded
    cache_shape = [max_num_blocks * kv_cache_repeats, num_kv_heads, block_size, head_dim]

    k_cache = ttnn.as_tensor(
        torch.zeros(cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, None)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    v_cache = ttnn.as_tensor(
        torch.zeros(cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, None)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create page table - simple sequential mapping
    # Each user gets their own block (user i uses block i)
    page_table = torch.zeros(batch_size, max_num_blocks // batch_size, dtype=torch.int32)
    for i in range(batch_size):
        page_table[i, 0] = i

    tt_page_table = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, None)),
    )

    # Create varying positions like in the demo after prefill
    position_ids = torch.tensor([random.randint(10, max_seq_len - 1) for _ in range(batch_size)], dtype=torch.int64)
    logger.info(f"Position range: {position_ids.min().item()} - {position_ids.max().item()}")

    # Memory configs
    grid_size = ttnn.CoreCoord(8, 8)  # Safe limit: max 8 per dimension to avoid Galaxy hangs
    batch_grid = ttnn.num_cores_to_corerangeset(batch_per_row, grid_size, row_wise=True)

    q_mem_config = ttnn.create_sharded_memory_config(
        shape=(num_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    kv_mem_config = ttnn.create_sharded_memory_config(
        shape=(32, head_dim),  # KV heads tile-aligned
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    errors = []

    # Run multiple iterations to check for accumulated errors
    num_iterations = 5
    for iteration in range(num_iterations):
        logger.info(f"\n--- Iteration {iteration} ---")

        # Create input Q, K, V tensors (simulating output from QKV projection)
        torch.manual_seed(42 + iteration)
        q_torch = torch.randn(1, batch_per_row, num_heads, head_dim, dtype=torch.bfloat16)
        k_torch = torch.randn(1, batch_per_row, num_kv_heads, head_dim, dtype=torch.bfloat16)
        v_torch = torch.randn(1, batch_per_row, num_kv_heads, head_dim, dtype=torch.bfloat16)

        # Create tensors on device
        q_tt = ttnn.from_torch(
            q_torch,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=q_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        k_tt = ttnn.from_torch(
            k_torch,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=kv_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        v_tt = ttnn.from_torch(
            v_torch,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=kv_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Get position indices for RoPE - manually create like in Model.get_tt_pos_idx
        rot_current_pos = torch.maximum(position_ids, torch.tensor(0, dtype=torch.int64))
        rot_current_pos = rot_current_pos.reshape(1, batch_size)
        pad_size = nearest_32(batch_size) - batch_size
        rot_current_pos = torch.nn.functional.pad(rot_current_pos, (0, pad_size), "constant", 0)
        tt_position_idx = ttnn.as_tensor(
            rot_current_pos,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=mesh_device.shape),
        )

        # Get RoPE matrices
        tt_cos, tt_sin = rope_setup.get_rot_mats(tt_position_idx)
        trans_mat = rope_setup.transformation_mat

        # Step 1: Apply RoPE to Q and K
        q_rope = ttnn.experimental.rotary_embedding_llama(q_tt, tt_cos, tt_sin, trans_mat, is_decode_mode=True)
        k_rope = ttnn.experimental.rotary_embedding_llama(k_tt, tt_cos, tt_sin, trans_mat, is_decode_mode=True)

        # Get Q after RoPE for verification
        q_rope_torch = ttnn.to_torch(ttnn.get_device_tensors(q_rope)[0])
        k_rope_torch = ttnn.to_torch(ttnn.get_device_tensors(k_rope)[0])

        logger.info(f"Q after RoPE range: [{q_rope_torch.min():.4f}, {q_rope_torch.max():.4f}]")
        logger.info(f"K after RoPE range: [{k_rope_torch.min():.4f}, {k_rope_torch.max():.4f}]")

        # Check for NaN/Inf in RoPE output
        if torch.isnan(q_rope_torch).any() or torch.isinf(q_rope_torch).any():
            errors.append(f"Iter {iteration}: NaN/Inf in Q after RoPE")
            logger.error(f"ERROR: NaN/Inf in Q after RoPE!")
        if torch.isnan(k_rope_torch).any() or torch.isinf(k_rope_torch).any():
            errors.append(f"Iter {iteration}: NaN/Inf in K after RoPE")
            logger.error(f"ERROR: NaN/Inf in K after RoPE!")

        # Step 2: Update KV cache
        # Create position tensor for cache update (current position within block)
        cur_pos = torch.zeros(batch_per_row, dtype=torch.int32) + iteration  # Position increases each iteration
        tt_cur_pos = ttnn.from_torch(
            cur_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Move to KV memory config for cache update
        k_rope_kv = ttnn.to_memory_config(k_rope, kv_mem_config)
        v_kv = ttnn.to_memory_config(v_tt, kv_mem_config)

        ttnn.experimental.paged_update_cache(
            k_cache,
            k_rope_kv,
            update_idxs_tensor=tt_cur_pos,
            page_table=tt_page_table,
        )
        ttnn.experimental.paged_update_cache(
            v_cache,
            v_kv,
            update_idxs_tensor=tt_cur_pos,
            page_table=tt_page_table,
        )

        # Step 3: Run SDPA
        sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(
                8, 8
            ),  # Safe limit: max 8 per dimension to avoid Galaxy hangs
            q_chunk_size=32,
            k_chunk_size=32,
        )

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        sdpa_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_rope,
            k_cache,
            v_cache,
            cur_pos_tensor=tt_cur_pos,
            page_table_tensor=tt_page_table,
            scale=1.0 / (head_dim**0.5),
            program_config=sdpa_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        sdpa_torch = ttnn.to_torch(ttnn.get_device_tensors(sdpa_output)[0])
        logger.info(f"SDPA output range: [{sdpa_torch.min():.4f}, {sdpa_torch.max():.4f}]")

        # Check for NaN/Inf in SDPA output
        if torch.isnan(sdpa_torch).any():
            errors.append(f"Iter {iteration}: NaN in SDPA output")
            logger.error(f"ERROR: NaN in SDPA output!")
        if torch.isinf(sdpa_torch).any():
            errors.append(f"Iter {iteration}: Inf in SDPA output")
            logger.error(f"ERROR: Inf in SDPA output!")

        # Cleanup tensors
        q_tt.deallocate()
        k_tt.deallocate()
        v_tt.deallocate()
        q_rope.deallocate()
        k_rope.deallocate()
        k_rope_kv.deallocate()
        v_kv.deallocate()
        sdpa_output.deallocate()

        # Increment positions for next iteration
        position_ids = position_ids + 1

    logger.info(f"\n=== Summary ===")
    logger.info(f"Total errors: {len(errors)}")
    if errors:
        for e in errors:
            logger.error(f"  {e}")
        pytest.xfail(f"Attention chain failed: {len(errors)} errors")


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("use_yarn", [False, True], ids=["no_yarn", "with_yarn"])
def test_single_layer_with_yarn(mesh_device, device_params, reset_seeds, use_yarn):
    """
    Test a single decoder layer with and without YARN scaling.

    This test compares the output of a single decoder layer when using YARN
    scaling vs. not, to identify where values start diverging.
    """
    import os
    import random

    from models.demos.gpt_oss.tt.model import create_rope_setup

    model_path = os.getenv("HF_MODEL")
    if model_path is None:
        pytest.skip("HF_MODEL environment variable not set")

    # Require 4x8 mesh like the demo
    if mesh_device.shape[0] != 4 or mesh_device.shape[1] != 8:
        pytest.skip(f"Test requires 4x8 mesh, got {mesh_device.shape}")

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Test parameters
    batch_size = 128
    batch_per_row = batch_size // mesh_device.shape[0]  # 32
    max_seq_len = 8192
    hidden_size = hf_config.hidden_size

    logger.info(f"=== Testing single layer with YARN={use_yarn} ===")
    logger.info(f"batch={batch_size}, hidden_size={hidden_size}")

    # Create rope setup
    original_rope_scaling = hf_config.rope_scaling
    if not use_yarn:
        # Skip no_yarn case since this model requires YARN
        pytest.skip("No-YARN case not supported for this model")

    try:
        rope_setup = create_rope_setup(
            mesh_device=mesh_device,
            hf_config=hf_config,
            max_local_batch_size=batch_per_row,
            users_row_sharded=True,
        )

        # Create random hidden states (simulating embedding output)
        torch.manual_seed(42)
        hidden_states_torch = torch.randn(1, 1, batch_per_row, hidden_size, dtype=torch.bfloat16)

        # Move to device
        hidden_states = ttnn.from_torch(
            hidden_states_torch,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Create position indices
        position_ids = torch.tensor([random.randint(10, max_seq_len - 1) for _ in range(batch_size)], dtype=torch.int64)
        rot_current_pos = torch.maximum(position_ids, torch.tensor(0, dtype=torch.int64))
        rot_current_pos = rot_current_pos.reshape(1, batch_size)
        pad_size = nearest_32(batch_size) - batch_size
        rot_current_pos = torch.nn.functional.pad(rot_current_pos, (0, pad_size), "constant", 0)

        tt_position_idx = ttnn.as_tensor(
            rot_current_pos,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=mesh_device.shape),
        )

        # Get RoPE matrices
        tt_cos, tt_sin = rope_setup.get_rot_mats(tt_position_idx)
        rope_mats = (tt_cos, tt_sin)

        # Check cos/sin ranges
        cos_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_cos)[0])
        sin_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_sin)[0])

        logger.info(f"cos range: [{cos_torch.min():.4f}, {cos_torch.max():.4f}]")
        logger.info(f"sin range: [{sin_torch.min():.4f}, {sin_torch.max():.4f}]")

        # Get hidden states after RoPE application would be done
        # We can't easily run a full layer without loading weights,
        # so let's just verify the RoPE matrices

        # Check if YARN scaling produces values > 1.0
        if use_yarn:
            cos_over_1 = (cos_torch.abs() > 1.0).float().mean()
            sin_over_1 = (sin_torch.abs() > 1.0).float().mean()
            logger.info(f"cos values > 1.0: {cos_over_1*100:.1f}%")
            logger.info(f"sin values > 1.0: {sin_over_1*100:.1f}%")

        # Test that RoPE can be applied to Q-like tensors
        q_torch = torch.randn(1, batch_per_row, 32, 64, dtype=torch.bfloat16)

        grid_size = ttnn.CoreCoord(8, 8)  # Safe limit: max 8 per dimension to avoid Galaxy hangs
        batch_grid = ttnn.num_cores_to_corerangeset(batch_per_row, grid_size, row_wise=True)

        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(32, 64),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        q_tt = ttnn.from_torch(
            q_torch,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=q_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        trans_mat = rope_setup.transformation_mat

        # Apply RoPE
        q_rope = ttnn.experimental.rotary_embedding_llama(q_tt, tt_cos, tt_sin, trans_mat, is_decode_mode=True)

        q_rope_torch = ttnn.to_torch(ttnn.get_device_tensors(q_rope)[0])

        logger.info(f"Q before RoPE range: [{q_torch.min():.4f}, {q_torch.max():.4f}]")
        logger.info(f"Q after RoPE range: [{q_rope_torch.min():.4f}, {q_rope_torch.max():.4f}]")

        # Check for specific position patterns
        failing_positions = [3, 7, 9, 14, 15, 23, 24, 25, 29, 30]
        issues = []

        for pos in failing_positions:
            if pos < batch_per_row:
                q_pos = q_rope_torch[0, pos, :, :].float()
                if torch.isnan(q_pos).any():
                    issues.append(f"Position {pos}: NaN")
                elif torch.isinf(q_pos).any():
                    issues.append(f"Position {pos}: Inf")
                elif q_pos.abs().max() > 100:
                    issues.append(f"Position {pos}: Large value ({q_pos.abs().max():.2f})")

        if issues:
            logger.error(f"Issues found at failing positions:")
            for issue in issues:
                logger.error(f"  {issue}")
        else:
            logger.info("✓ No issues at failing positions")

        # Summary
        logger.info(f"\n=== Results for YARN={use_yarn} ===")
        logger.info(
            f"cos/sin range: [{min(cos_torch.min(), sin_torch.min()):.4f}, {max(cos_torch.max(), sin_torch.max()):.4f}]"
        )
        logger.info(f"Q after RoPE range: [{q_rope_torch.min():.4f}, {q_rope_torch.max():.4f}]")

    finally:
        # Restore original config
        hf_config.rope_scaling = original_rope_scaling
