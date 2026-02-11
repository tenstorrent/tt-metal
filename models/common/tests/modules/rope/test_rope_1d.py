# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the RotarySetup1D module (1D mesh topology: N150, N300, T3K).

This test suite verifies:
1. Unit tests for config dataclass and RotaryEmbedding reference implementations
2. RotarySetup1D init + API methods (get_both_trans_mats, get_rot_idxs, get_rot_mats)
3. from_model_args backward compatibility
"""


import pytest
import torch
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.rope.rope_1d import (
    RotarySetup1D,
    RotarySetup1DConfig,
    _compute_cos_sin_matrices,
    _get_rot_transformation_mat,
)
from models.common.utility_functions import comp_pcc

# RopeScaling helpers (import from common.py for test setup)
from models.tt_transformers.tt.common import RopeScalingLlama3, RopeScalingType

# ============================================================================
# Constants from rope_1d_init_test_cases.csv
# ============================================================================


def _llama3_scaling():
    """Llama-3.x rope scaling config (factor=8, low=1, high=4)."""
    return RopeScalingLlama3(
        rope_type=RopeScalingType.LLAMA3,
        factor=8.0,
        original_max_position_embeddings=8192,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
    )


_slow = pytest.mark.slow


# ============================================================================
# Unit Tests - No device required
# ============================================================================


def test_rope_1d_config_creation():
    """Test that RotarySetup1DConfig can be created with all required fields."""
    from unittest.mock import MagicMock

    config = RotarySetup1DConfig(
        device=MagicMock(),
        batch_size=32,
        head_dim=128,
        max_seq_len=8192,
        rope_theta=500000.0,
    )
    assert config.batch_size == 32
    assert config.head_dim == 128
    assert config.max_seq_len == 8192
    assert config.rope_theta == 500000.0
    assert config.use_qk_fused is False
    assert config.datatype == ttnn.bfloat16
    assert config.rope_scaling is None


def test_rope_1d_config_with_scaling():
    """Test config with rope_scaling."""
    from unittest.mock import MagicMock

    scaling = _llama3_scaling()
    config = RotarySetup1DConfig(
        device=MagicMock(),
        batch_size=1,
        head_dim=128,
        max_seq_len=8192,
        rope_theta=500000.0,
        rope_scaling=scaling,
        use_qk_fused=True,
    )
    assert config.rope_scaling is scaling
    assert config.use_qk_fused is True


def test_compute_cos_sin_matrices_no_scaling():
    """Test cos/sin computation without scaling (Mistral/Qwen-style)."""
    cos, sin = _compute_cos_sin_matrices(head_dim=128, max_seq_len=8192, rope_theta=1000000.0)
    assert cos.shape == (1, 1, 8192, 128)
    assert sin.shape == (1, 1, 8192, 128)
    # cos/sin values should be in [-1, 1]
    assert cos.abs().max() <= 1.0 + 1e-6
    assert sin.abs().max() <= 1.0 + 1e-6


def test_compute_cos_sin_matrices_llama3_scaling():
    """Test cos/sin computation with Llama-3.x scaling."""
    scaling = _llama3_scaling()
    cos, sin = _compute_cos_sin_matrices(head_dim=128, max_seq_len=8192, rope_theta=500000.0, rope_scaling=scaling)
    assert cos.shape == (1, 1, 8192, 128)
    assert sin.shape == (1, 1, 8192, 128)


def test_compute_cos_sin_head_dim_64():
    """Test cos/sin computation with head_dim=64 (Llama-3.2-1B)."""
    cos, sin = _compute_cos_sin_matrices(
        head_dim=64, max_seq_len=8192, rope_theta=500000.0, rope_scaling=_llama3_scaling()
    )
    assert cos.shape == (1, 1, 8192, 64)
    assert sin.shape == (1, 1, 8192, 64)


# ============================================================================
# Integration Tests - Require device
# ============================================================================


# Collected from rope_1d_init_test_cases.csv (deduplicated)
# Format: (device_shape, batch_size, head_dim, max_seq_len, rope_theta, rope_scaling_str, use_qk_fused)
def _list_init_test_cases() -> list[pytest.param]:
    # fmt: off
    return [
        # === Fast tests (one per unique model family) ===
        # Llama-3.2-1B: head_dim=64, llama3 scaling, theta=500000
        pytest.param((1, 1), 1, 64, 8192, 500000.0, "llama3", True, id="1x1-b1-hd64-llama3-fused"),
        # Llama-3.1-8B: head_dim=128, llama3 scaling, theta=500000
        pytest.param((1, 2), 1, 128, 8192, 500000.0, "llama3", True, id="1x2-b1-hd128-llama3-fused"),
        # Mistral-7B: head_dim=128, no scaling, theta=1000000
        pytest.param((1, 1), 1, 128, 8192, 1000000.0, "none", True, id="1x1-b1-hd128-none-fused"),
        # Llama-3.2-11B: use_qk_fused=False
        pytest.param((1, 2), 1, 128, 8192, 500000.0, "llama3", False, id="1x2-b1-hd128-llama3-nofused"),
        # T3K Llama-3.3-70B
        pytest.param((1, 8), 1, 128, 8192, 500000.0, "llama3", True, id="1x8-b1-hd128-llama3-fused"),
        # T3K Qwen2.5-72B: no scaling, theta=1000000
        pytest.param((1, 8), 1, 128, 8192, 1000000.0, "none", True, id="1x8-b1-hd128-none-fused"),

        # === Slow tests (remaining from CSV) ===
        # (1,1) batch=32
        pytest.param((1, 1), 32, 64, 2048, 500000.0, "llama3", True, id="1x1-b32-hd64-llama3-fused", marks=_slow),
        pytest.param((1, 1), 32, 128, 2048, 500000.0, "llama3", True, id="1x1-b32-hd128-llama3-fused-8B", marks=_slow),
        pytest.param((1, 1), 32, 128, 2048, 1000000.0, "none", True, id="1x1-b32-hd128-none-fused-Mistral", marks=_slow),
        # (1,1) hd=128, llama3, batch=1 (missing from original — Llama-3.2-3B, 3.1-8B on N150)
        pytest.param((1, 1), 1, 128, 8192, 500000.0, "llama3", True, id="1x1-b1-hd128-llama3-fused-8B", marks=_slow),
        pytest.param((1, 1), 1, 128, 1024, 500000.0, "llama3", True, id="1x1-b1-hd128-llama3-fused-seqlen1024", marks=_slow),
        pytest.param((1, 1), 32, 128, 1024, 500000.0, "llama3", True, id="1x1-b32-hd128-llama3-fused-seqlen1024", marks=_slow),
        pytest.param((1, 1), 1, 128, 32768, 500000.0, "llama3", True, id="1x1-b1-hd128-llama3-fused-seqlen32k", marks=_slow),
        # (1,1) hd=128, none (Mistral), additional seq_len
        pytest.param((1, 1), 1, 128, 1024, 1000000.0, "none", True, id="1x1-b1-hd128-none-fused-seqlen1024", marks=_slow),
        pytest.param((1, 1), 32, 128, 1024, 1000000.0, "none", True, id="1x1-b32-hd128-none-fused-seqlen1024", marks=_slow),
        pytest.param((1, 1), 1, 128, 32768, 1000000.0, "none", True, id="1x1-b1-hd128-none-fused-seqlen32k", marks=_slow),
        # (1,1) hd=64 additional seq_len
        pytest.param((1, 1), 1, 64, 1024, 500000.0, "llama3", True, id="1x1-b1-hd64-llama3-seqlen1024", marks=_slow),
        pytest.param((1, 1), 32, 64, 1024, 500000.0, "llama3", True, id="1x1-b32-hd64-llama3-seqlen1024", marks=_slow),
        pytest.param((1, 1), 1, 64, 32768, 500000.0, "llama3", True, id="1x1-b1-hd64-llama3-seqlen32k", marks=_slow),
        # (1,2) variants
        pytest.param((1, 2), 32, 64, 2048, 500000.0, "llama3", True, id="1x2-b32-hd64-llama3-fused", marks=_slow),
        pytest.param((1, 2), 32, 128, 2048, 500000.0, "llama3", True, id="1x2-b32-hd128-llama3-fused", marks=_slow),
        pytest.param((1, 2), 32, 128, 2048, 1000000.0, "none", True, id="1x2-b32-hd128-none-fused-Mistral", marks=_slow),
        pytest.param((1, 2), 32, 128, 2048, 500000.0, "llama3", False, id="1x2-b32-hd128-llama3-nofused-11B", marks=_slow),
        pytest.param((1, 2), 1, 128, 1024, 500000.0, "llama3", True, id="1x2-b1-hd128-llama3-fused-seqlen1024", marks=_slow),
        pytest.param((1, 2), 32, 128, 1024, 500000.0, "llama3", True, id="1x2-b32-hd128-llama3-fused-seqlen1024", marks=_slow),
        pytest.param((1, 2), 1, 128, 32768, 500000.0, "llama3", True, id="1x2-b1-hd128-llama3-fused-seqlen32k", marks=_slow),
        pytest.param((1, 2), 1, 128, 8192, 1000000.0, "none", True, id="1x2-b1-hd128-none-fused-Qwen2", marks=_slow),
        pytest.param((1, 2), 1, 64, 8192, 500000.0, "llama3", True, id="1x2-b1-hd64-llama3-fused", marks=_slow),
        pytest.param((1, 2), 1, 64, 1024, 500000.0, "llama3", True, id="1x2-b1-hd64-llama3-fused-seqlen1024", marks=_slow),
        pytest.param((1, 2), 32, 64, 1024, 500000.0, "llama3", True, id="1x2-b32-hd64-llama3-fused-seqlen1024", marks=_slow),
        pytest.param((1, 2), 1, 64, 32768, 500000.0, "llama3", True, id="1x2-b1-hd64-llama3-fused-seqlen32k", marks=_slow),
        pytest.param((1, 2), 1, 128, 1024, 500000.0, "llama3", False, id="1x2-b1-hd128-llama3-nofused-11B-seqlen1024", marks=_slow),
        pytest.param((1, 2), 32, 128, 1024, 500000.0, "llama3", False, id="1x2-b32-hd128-llama3-nofused-11B-seqlen1024", marks=_slow),
        pytest.param((1, 2), 1, 128, 32768, 500000.0, "llama3", False, id="1x2-b1-hd128-llama3-nofused-11B-seqlen32k", marks=_slow),
        pytest.param((1, 2), 1, 128, 1024, 1000000.0, "none", True, id="1x2-b1-hd128-none-fused-seqlen1024", marks=_slow),
        pytest.param((1, 2), 32, 128, 1024, 1000000.0, "none", True, id="1x2-b32-hd128-none-fused-seqlen1024", marks=_slow),
        pytest.param((1, 2), 1, 128, 32768, 1000000.0, "none", True, id="1x2-b1-hd128-none-fused-seqlen32k", marks=_slow),
        # (1,2) batch=4, 16 from Llama-3.2-11B (nofused)
        pytest.param((1, 2), 16, 128, 512, 500000.0, "llama3", False, id="1x2-b16-hd128-llama3-nofused-11B", marks=_slow),
        pytest.param((1, 2), 4, 128, 512, 500000.0, "llama3", False, id="1x2-b4-hd128-llama3-nofused-11B", marks=_slow),
        # (1,8) variants — hd=64
        pytest.param((1, 8), 1, 64, 8192, 500000.0, "llama3", True, id="1x8-b1-hd64-llama3-fused", marks=_slow),
        pytest.param((1, 8), 32, 64, 2048, 500000.0, "llama3", True, id="1x8-b32-hd64-llama3-fused", marks=_slow),
        pytest.param((1, 8), 1, 64, 1024, 500000.0, "llama3", True, id="1x8-b1-hd64-llama3-fused-seqlen1024", marks=_slow),
        pytest.param((1, 8), 32, 64, 1024, 500000.0, "llama3", True, id="1x8-b32-hd64-llama3-fused-seqlen1024", marks=_slow),
        pytest.param((1, 8), 1, 64, 32768, 500000.0, "llama3", True, id="1x8-b1-hd64-llama3-fused-seqlen32k", marks=_slow),
        # (1,8) variants — hd=128, llama3, fused
        pytest.param((1, 8), 32, 128, 2048, 500000.0, "llama3", True, id="1x8-b32-hd128-llama3-fused-8B", marks=_slow),
        pytest.param((1, 8), 1, 128, 1024, 500000.0, "llama3", True, id="1x8-b1-hd128-llama3-fused-seqlen1024", marks=_slow),
        pytest.param((1, 8), 32, 128, 1024, 500000.0, "llama3", True, id="1x8-b32-hd128-llama3-fused-seqlen1024", marks=_slow),
        pytest.param((1, 8), 1, 128, 32768, 500000.0, "llama3", True, id="1x8-b1-hd128-llama3-fused-seqlen32k", marks=_slow),
        # (1,8) variants — hd=128, llama3, nofused (11B)
        pytest.param((1, 8), 1, 128, 8192, 500000.0, "llama3", False, id="1x8-b1-hd128-llama3-nofused-11B", marks=_slow),
        pytest.param((1, 8), 32, 128, 2048, 500000.0, "llama3", False, id="1x8-b32-hd128-llama3-nofused-11B", marks=_slow),
        pytest.param((1, 8), 1, 128, 1024, 500000.0, "llama3", False, id="1x8-b1-hd128-llama3-nofused-11B-seqlen1024", marks=_slow),
        pytest.param((1, 8), 32, 128, 1024, 500000.0, "llama3", False, id="1x8-b32-hd128-llama3-nofused-11B-seqlen1024", marks=_slow),
        pytest.param((1, 8), 1, 128, 32768, 500000.0, "llama3", False, id="1x8-b1-hd128-llama3-nofused-11B-seqlen32k", marks=_slow),
        # (1,8) variants — hd=128, none (Qwen/Mistral/Mixtral)
        pytest.param((1, 8), 32, 128, 2048, 1000000.0, "none", True, id="1x8-b32-hd128-none-fused-Qwen72B", marks=_slow),
        pytest.param((1, 8), 1, 128, 1024, 1000000.0, "none", True, id="1x8-b1-hd128-none-fused-seqlen1024", marks=_slow),
        pytest.param((1, 8), 32, 128, 1024, 1000000.0, "none", True, id="1x8-b32-hd128-none-fused-seqlen1024", marks=_slow),
        pytest.param((1, 8), 1, 128, 32768, 1000000.0, "none", True, id="1x8-b1-hd128-none-fused-seqlen32k", marks=_slow),
        # (1,8) batch=4 from Llama-3.2-11B (nofused)
        pytest.param((1, 8), 4, 128, 512, 500000.0, "llama3", False, id="1x8-b4-hd128-llama3-nofused-11B", marks=_slow),
        pytest.param((1, 8), 32, 128, 512, 500000.0, "llama3", False, id="1x8-b32-hd128-llama3-nofused-11B-seqlen512", marks=_slow),
        # Llama-3.2-90B (from api CSV)
        pytest.param((1, 8), 1, 128, 512, 500000.0, "llama3", False, id="1x8-b1-hd128-llama3-nofused-90B", marks=_slow),
    ]
    # fmt: on


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape,batch_size,head_dim,max_seq_len,rope_theta,rope_scaling_str,use_qk_fused",
    _list_init_test_cases(),
)
def test_rope_1d_init_and_api(
    ttnn_mesh_device: ttnn.MeshDevice,
    mesh_shape,
    batch_size,
    head_dim,
    max_seq_len,
    rope_theta,
    rope_scaling_str,
    use_qk_fused,
):
    """
    Test RotarySetup1D initialization and all 3 API methods.

    Verifies:
    1. Construction succeeds with given parameters
    2. get_both_trans_mats() returns valid tensors
    3. get_rot_idxs() produces correct shape
    4. get_rot_mats() produces cos/sin with correct shapes
    """
    rope_scaling = _llama3_scaling() if rope_scaling_str == "llama3" else None

    rope = RotarySetup1D(
        device=ttnn_mesh_device,
        batch_size=batch_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        use_qk_fused=use_qk_fused,
    )

    trans_mats = rope.get_both_trans_mats()
    assert "decode" in trans_mats
    assert "prefill" in trans_mats

    # Prefill trans mat: _get_rot_transformation_mat(head_dim) → 32x32 (dhead forced to TILE_SIZE)
    prefill_ref = _get_rot_transformation_mat(dhead=head_dim)  # [1, 1, 32, 32]
    prefill_tt = to_torch_auto_compose(trans_mats["prefill"])
    # Trim TT output to match ref shape (replicated, same on all devices)
    prefill_tt_trimmed = prefill_tt[:1, :1, : prefill_ref.shape[2], : prefill_ref.shape[3]]
    pcc_prefill, msg_prefill = comp_pcc(prefill_ref.to(torch.bfloat16), prefill_tt_trimmed.to(torch.bfloat16), 0.9999)
    assert pcc_prefill, f"prefill trans_mat PCC failed: {msg_prefill}"

    # Decode trans mat: same base matrix repeated batch_size_per_device_group times on dim 2
    effective_batch = batch_size * 2 if use_qk_fused else batch_size
    decode_ref = _get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(1, 1, effective_batch, 1)
    decode_tt = to_torch_auto_compose(trans_mats["decode"])
    decode_tt_trimmed = decode_tt[:1, :1, : decode_ref.shape[2], : decode_ref.shape[3]]
    pcc_decode, msg_decode = comp_pcc(decode_ref.to(torch.bfloat16), decode_tt_trimmed.to(torch.bfloat16), 0.9999)
    assert pcc_decode, f"decode trans_mat PCC failed: {msg_decode}"

    # Test get_rot_idxs
    position_idxs = torch.arange(batch_size)
    rot_idxs = rope.get_rot_idxs(position_idxs, on_host=True)
    assert isinstance(rot_idxs, ttnn.Tensor)

    # Test get_rot_mats
    pos_batch = batch_size
    if use_qk_fused:
        # For fused ops, get_rot_idxs doubles internally; get_rot_mats with torch input calls get_rot_idxs
        pass
    rot_mats = rope.get_rot_mats(position_idxs)
    assert len(rot_mats) == 2
    cos, sin = rot_mats
    assert isinstance(cos, ttnn.Tensor)
    assert isinstance(sin, ttnn.Tensor)

    # Test get_rot_mats with return_rot_idxs
    result = rope.get_rot_mats(position_idxs, return_rot_idxs=True)
    assert len(result) == 2
    rot_mats_2, rot_idxs_2 = result
    assert len(rot_mats_2) == 2
    assert isinstance(rot_idxs_2, ttnn.Tensor)

    # PCC check: use non-zero positions to avoid sin(0)=0 (PCC undefined for all-zero tensors)
    pcc_position_idxs = torch.arange(42, 42 + batch_size)  # positions [42..42+batch-1], always non-zero sin

    # Test production calling pattern: get_rot_idxs(on_host=True) → get_rot_mats(ttnn_input)
    # This is how the API is used in real models (see rope_1d_api_test_cases.csv)
    pcc_rot_idxs_host = rope.get_rot_idxs(pcc_position_idxs, on_host=True)
    pcc_rot_mats_via_ttnn = rope.get_rot_mats(pcc_rot_idxs_host)
    assert len(pcc_rot_mats_via_ttnn) == 2

    # Also test the torch→internal path for comparison
    pcc_rot_mats = rope.get_rot_mats(pcc_position_idxs)

    cos_torch_ref, sin_torch_ref = _compute_cos_sin_matrices(
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )

    cos_tt = to_torch_auto_compose(pcc_rot_mats[0])
    sin_tt = to_torch_auto_compose(pcc_rot_mats[1])

    # Build expected cos/sin from torch reference for the chosen positions
    effective_batch = batch_size * 2 if use_qk_fused else batch_size
    # For fused: positions are [42..42+batch-1] repeated 2x → [42,..,42+b-1, 42,..,42+b-1]
    if use_qk_fused:
        ref_positions = list(range(42, 42 + batch_size)) * 2
    else:
        ref_positions = list(range(42, 42 + batch_size))
    expected_cos = cos_torch_ref[:, :, ref_positions, :]  # [1, 1, effective_batch, head_dim]
    expected_sin = sin_torch_ref[:, :, ref_positions, :]

    # Reshape TT output: [1, batch, TILE_SIZE, head_dim] → [1, 1, batch*TILE_SIZE, head_dim]
    cos_tt_flat = cos_tt.reshape(1, 1, -1, head_dim)
    sin_tt_flat = sin_tt.reshape(1, 1, -1, head_dim)

    # Compare first effective_batch rows (rest is padding from tile alignment)
    cos_tt_trimmed = cos_tt_flat[:, :, :effective_batch, :]
    sin_tt_trimmed = sin_tt_flat[:, :, :effective_batch, :]

    pcc_cos, msg_cos = comp_pcc(expected_cos.to(torch.bfloat16), cos_tt_trimmed.to(torch.bfloat16), 0.999)
    pcc_sin, msg_sin = comp_pcc(expected_sin.to(torch.bfloat16), sin_tt_trimmed.to(torch.bfloat16), 0.999)

    logger.info(f"cos PCC: {msg_cos}")
    logger.info(f"sin PCC: {msg_sin}")

    assert pcc_cos, f"cos PCC failed: {msg_cos}"
    assert pcc_sin, f"sin PCC failed: {msg_sin}"

    # PCC check: ttnn-input path (production pattern) must match torch-input path
    cos_tt_via_ttnn = to_torch_auto_compose(pcc_rot_mats_via_ttnn[0])
    sin_tt_via_ttnn = to_torch_auto_compose(pcc_rot_mats_via_ttnn[1])
    pcc_cos_path, msg_cos_path = comp_pcc(cos_tt, cos_tt_via_ttnn, 0.9999)
    pcc_sin_path, msg_sin_path = comp_pcc(sin_tt, sin_tt_via_ttnn, 0.9999)
    logger.info(f"torch-path vs ttnn-path cos PCC: {msg_cos_path}")
    logger.info(f"torch-path vs ttnn-path sin PCC: {msg_sin_path}")
    assert pcc_cos_path, f"ttnn-input path cos mismatch vs torch-input path: {msg_cos_path}"
    assert pcc_sin_path, f"ttnn-input path sin mismatch vs torch-input path: {msg_sin_path}"

    logger.info(
        f"RotarySetup1D: PASSED for mesh={mesh_shape}, batch={batch_size}, "
        f"head_dim={head_dim}, max_seq_len={max_seq_len}, rope_theta={rope_theta}, "
        f"scaling={rope_scaling_str}, fused={use_qk_fused}"
    )


# ============================================================================
# Numerical correctness: compare cos/sin against TTTv1 RotarySetup
# ============================================================================


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size,head_dim,max_seq_len,rope_theta,rope_scaling_str,use_qk_fused",
    [
        pytest.param(1, 128, 8192, 500000.0, "llama3", True, id="b1-hd128-llama3-fused"),
        pytest.param(32, 128, 2048, 500000.0, "llama3", True, id="b32-hd128-llama3-fused"),
        pytest.param(1, 64, 8192, 500000.0, "llama3", True, id="b1-hd64-llama3-fused"),
        pytest.param(1, 128, 8192, 1000000.0, "none", True, id="b1-hd128-none-fused"),
        pytest.param(1, 128, 8192, 500000.0, "llama3", False, id="b1-hd128-llama3-nofused"),
    ],
)
def test_rope_1d_cos_sin_vs_reference(
    ttnn_mesh_device: ttnn.MeshDevice,
    batch_size,
    head_dim,
    max_seq_len,
    rope_theta,
    rope_scaling_str,
    use_qk_fused,
):
    """
    Compare RotarySetup1D get_rot_mats output against TTTv1 RotarySetup.

    Uses identical parameters and position indices, comparing cos/sin outputs
    via PCC to ensure numerical equivalence.
    """
    from models.tt_transformers.tt.rope import RotarySetup as TTTv1RotarySetup

    rope_scaling = _llama3_scaling() if rope_scaling_str == "llama3" else None

    # Build TTTv2
    rope_v2 = RotarySetup1D(
        device=ttnn_mesh_device,
        batch_size=batch_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        use_qk_fused=use_qk_fused,
    )

    # Build TTTv1 (for reference)
    rope_v1 = TTTv1RotarySetup(
        device=ttnn_mesh_device,
        batch_size=batch_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        use_qk_fused=use_qk_fused,
        datatype=ttnn.bfloat16,
    )

    # Compare get_rot_mats with non-zero position indices (avoid sin(0)=0 → PCC undefined)
    position_idxs = torch.arange(42, 42 + batch_size)
    v2_cos_sin = rope_v2.get_rot_mats(position_idxs)
    v1_cos_sin = rope_v1.get_rot_mats(position_idxs)

    # Replicated tensors: use auto_compose to get single-device view
    v2_cos_torch = to_torch_auto_compose(v2_cos_sin[0])
    v1_cos_torch = to_torch_auto_compose(v1_cos_sin[0])
    v2_sin_torch = to_torch_auto_compose(v2_cos_sin[1])
    v1_sin_torch = to_torch_auto_compose(v1_cos_sin[1])

    pcc_cos, msg_cos = comp_pcc(v1_cos_torch, v2_cos_torch, 0.9999)
    pcc_sin, msg_sin = comp_pcc(v1_sin_torch, v2_sin_torch, 0.9999)

    logger.info(f"cos PCC: {msg_cos}")
    logger.info(f"sin PCC: {msg_sin}")

    assert pcc_cos, f"cos mismatch: {msg_cos}"
    assert pcc_sin, f"sin mismatch: {msg_sin}"

    # Compare transformation matrices
    v2_trans = rope_v2.get_both_trans_mats()
    v1_trans = rope_v1.get_both_trans_mats()

    for key in ["decode", "prefill"]:
        v2_t = to_torch_auto_compose(v2_trans[key])
        v1_t = to_torch_auto_compose(v1_trans[key])
        pcc_ok, msg = comp_pcc(v1_t, v2_t, 0.9999)
        logger.info(f"trans_mat[{key}] PCC: {msg}")
        assert pcc_ok, f"trans_mat[{key}] mismatch: {msg}"

    logger.info(f"RotarySetup1D vs TTTv1: PASSED")


# ============================================================================
# from_model_args backward compatibility test
# ============================================================================


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
def test_rope_1d_vs_reference_from_model_args(ttnn_mesh_device: ttnn.MeshDevice):
    """
    Test that RotarySetup1D.from_model_args produces numerically identical
    rotation matrices compared to TTTv1 RotarySetup built with the same args.
    """
    from models.tt_transformers.tt.model_config import ModelArgs
    from models.tt_transformers.tt.rope import RotarySetup as TTTv1RotarySetup

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=1, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    if model_args.is_galaxy:
        pytest.skip("RotarySetup1D test only runs on non-TG devices")

    # Build TTTv2 via from_model_args
    rope_v2 = RotarySetup1D.from_model_args(
        device=ttnn_mesh_device,
        args=model_args,
        model_name=model_args.model_name,
    )

    # Build TTTv1 reference with same params
    from models.tt_transformers.tt.common import rope_scaling_model_factory

    rope_scaling = None
    if hasattr(model_args, "rope_scaling_params") and model_args.rope_scaling_params is not None:
        rope_scaling = rope_scaling_model_factory(
            model_args.rope_scaling_params, getattr(model_args, "original_max_context_len", None)
        )

    rope_v1 = TTTv1RotarySetup(
        device=ttnn_mesh_device,
        batch_size=model_args.max_batch_size,
        head_dim=model_args.head_dim,
        max_seq_len=model_args.max_seq_len,
        rope_theta=model_args.rope_theta,
        rope_scaling=rope_scaling,
        use_qk_fused=getattr(model_args, "use_qk_fused", False),
        datatype=ttnn.bfloat16,
    )

    # PCC check: get_rot_mats (use non-zero positions to avoid sin(0)=0 → PCC undefined)
    position_idxs = torch.arange(42, 42 + model_args.max_batch_size)
    v2_cos_sin = rope_v2.get_rot_mats(position_idxs)
    v1_cos_sin = rope_v1.get_rot_mats(position_idxs)

    v2_cos_torch = to_torch_auto_compose(v2_cos_sin[0])
    v1_cos_torch = to_torch_auto_compose(v1_cos_sin[0])
    v2_sin_torch = to_torch_auto_compose(v2_cos_sin[1])
    v1_sin_torch = to_torch_auto_compose(v1_cos_sin[1])

    pcc_cos, msg_cos = comp_pcc(v1_cos_torch, v2_cos_torch, 0.9999)
    pcc_sin, msg_sin = comp_pcc(v1_sin_torch, v2_sin_torch, 0.9999)

    logger.info(f"from_model_args cos PCC: {msg_cos}")
    logger.info(f"from_model_args sin PCC: {msg_sin}")

    assert pcc_cos, f"from_model_args cos mismatch: {msg_cos}"
    assert pcc_sin, f"from_model_args sin mismatch: {msg_sin}"

    # PCC check: transformation matrices
    v2_trans = rope_v2.get_both_trans_mats()
    v1_trans = rope_v1.get_both_trans_mats()

    for key in ["decode", "prefill"]:
        v2_t = to_torch_auto_compose(v2_trans[key])
        v1_t = to_torch_auto_compose(v1_trans[key])
        pcc_ok, msg = comp_pcc(v1_t, v2_t, 0.9999)
        logger.info(f"from_model_args trans_mat[{key}] PCC: {msg}")
        assert pcc_ok, f"from_model_args trans_mat[{key}] mismatch: {msg}"

    logger.info(f"RotarySetup1D.from_model_args vs TTTv1: PASSED for {model_args.model_name}")
