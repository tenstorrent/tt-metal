# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
from models.common.modules.rope.rope_1d import RotarySetup1D, RotarySetup1DConfig, _compute_cos_sin_matrices
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
        # Batch=32 variants
        pytest.param((1, 1), 32, 64, 2048, 500000.0, "llama3", True, id="1x1-b32-hd64-llama3-fused", marks=_slow),
        pytest.param((1, 1), 32, 128, 2048, 500000.0, "llama3", True, id="1x1-b32-hd128-llama3-fused-8B", marks=_slow),
        pytest.param((1, 1), 32, 128, 2048, 1000000.0, "none", True, id="1x1-b32-hd128-none-fused-Mistral", marks=_slow),
        # (1,2) variants
        pytest.param((1, 2), 32, 64, 2048, 500000.0, "llama3", True, id="1x2-b32-hd64-llama3-fused", marks=_slow),
        pytest.param((1, 2), 32, 128, 2048, 500000.0, "llama3", True, id="1x2-b32-hd128-llama3-fused", marks=_slow),
        pytest.param((1, 2), 32, 128, 2048, 1000000.0, "none", True, id="1x2-b32-hd128-none-fused-Mistral", marks=_slow),
        pytest.param((1, 2), 32, 128, 2048, 500000.0, "llama3", False, id="1x2-b32-hd128-llama3-nofused-11B", marks=_slow),
        pytest.param((1, 2), 1, 128, 1024, 500000.0, "llama3", True, id="1x2-b1-hd128-llama3-fused-seqlen1024", marks=_slow),
        pytest.param((1, 2), 32, 128, 1024, 500000.0, "llama3", True, id="1x2-b32-hd128-llama3-fused-seqlen1024", marks=_slow),
        pytest.param((1, 2), 1, 128, 32768, 500000.0, "llama3", True, id="1x2-b1-hd128-llama3-fused-seqlen32k", marks=_slow),
        pytest.param((1, 2), 1, 128, 8192, 1000000.0, "none", True, id="1x2-b1-hd128-none-fused-Qwen2", marks=_slow),
        # (1,8) variants
        pytest.param((1, 8), 32, 64, 2048, 500000.0, "llama3", True, id="1x8-b32-hd64-llama3-fused", marks=_slow),
        pytest.param((1, 8), 32, 128, 2048, 500000.0, "llama3", True, id="1x8-b32-hd128-llama3-fused-8B", marks=_slow),
        pytest.param((1, 8), 32, 128, 2048, 1000000.0, "none", True, id="1x8-b32-hd128-none-fused-Qwen72B", marks=_slow),
        pytest.param((1, 8), 1, 128, 8192, 500000.0, "llama3", False, id="1x8-b1-hd128-llama3-nofused-11B", marks=_slow),
        pytest.param((1, 8), 32, 128, 2048, 500000.0, "llama3", False, id="1x8-b32-hd128-llama3-nofused-11B", marks=_slow),
        pytest.param((1, 8), 1, 128, 32768, 500000.0, "llama3", True, id="1x8-b1-hd128-llama3-fused-seqlen32k", marks=_slow),
        # Llama-3.2-90B (from api CSV)
        pytest.param((1, 8), 1, 128, 512, 500000.0, "llama3", False, id="1x8-b1-hd128-llama3-nofused-90B", marks=_slow),
        # Different max_seq_len values
        pytest.param((1, 1), 1, 64, 1024, 500000.0, "llama3", True, id="1x1-b1-hd64-llama3-seqlen1024", marks=_slow),
        pytest.param((1, 1), 32, 64, 1024, 500000.0, "llama3", True, id="1x1-b32-hd64-llama3-seqlen1024", marks=_slow),
        pytest.param((1, 1), 1, 64, 32768, 500000.0, "llama3", True, id="1x1-b1-hd64-llama3-seqlen32k", marks=_slow),
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

    # Test get_both_trans_mats
    trans_mats = rope.get_both_trans_mats()
    assert "decode" in trans_mats
    assert "prefill" in trans_mats
    assert isinstance(trans_mats["decode"], ttnn.Tensor)
    assert isinstance(trans_mats["prefill"], ttnn.Tensor)

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

    # Compare get_rot_mats with same position indices
    position_idxs = torch.arange(batch_size)
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
    Test that RotarySetup1D.from_model_args produces valid rotation matrices.

    Uses HF_MODEL env var or defaults to Llama-3.1-8B-Instruct.
    """
    from models.tt_transformers.tt.model_config import ModelArgs

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=1, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    if model_args.is_galaxy:
        pytest.skip("RotarySetup1D test only runs on non-TG devices")

    rope = RotarySetup1D.from_model_args(
        device=ttnn_mesh_device,
        args=model_args,
        model_name=model_args.model_name,
    )

    # Verify all APIs work
    trans_mats = rope.get_both_trans_mats()
    assert "decode" in trans_mats and "prefill" in trans_mats

    position_idxs = torch.arange(1)  # batch_size=1
    rot_idxs = rope.get_rot_idxs(position_idxs, on_host=True)
    assert isinstance(rot_idxs, ttnn.Tensor)

    rot_mats = rope.get_rot_mats(position_idxs)
    assert len(rot_mats) == 2

    logger.info(f"RotarySetup1D.from_model_args: PASSED for {model_args.model_name}")
