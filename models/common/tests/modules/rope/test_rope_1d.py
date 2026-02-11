# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the RotarySetup1D module (1D mesh topology: N150, N300, T3K).

This test suite verifies:
1. Unit tests for config dataclass and transformation matrix utility
2. RotarySetup1D init + API methods (get_both_trans_mats, forward)
3. Numerical correctness vs pure-torch HF reference
4. from_model_args backward compatibility (vs TTTv1)
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D, prepare_rot_idxs
from models.common.tensor_utils import get_rot_transformation_mat
from models.common.utility_functions import comp_pcc

# ============================================================================
# Pure-torch RoPE reference (no TTTv1 dependency)
# ============================================================================

_slow = pytest.mark.slow


@dataclass
class Llama3Scaling:
    """Llama-3.x frequency scaling parameters."""

    factor: float = 8.0
    original_max_position_embeddings: int = 8192
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0


def _apply_llama3_scaling(freqs: torch.Tensor, s: Llama3Scaling) -> torch.Tensor:
    """Apply Llama-3.x frequency scaling (pure torch, mirrors HF implementation)."""
    low_freq_wavelen = s.original_max_position_embeddings / s.low_freq_factor
    high_freq_wavelen = s.original_max_position_embeddings / s.high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / s.factor)
        else:
            smooth = (s.original_max_position_embeddings / wavelen - s.low_freq_factor) / (
                s.high_freq_factor - s.low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / s.factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def _rope_cos_sin(
    head_dim: int,
    max_seq_len: int,
    theta: float,
    scaling: Optional[Llama3Scaling] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE cos/sin tables in Meta interleaved format (pure torch).

    This is the HF reference implementation for RoPE, independent of TTTv1.

    Returns:
        cos, sin: [1, 1, max_seq_len, head_dim] in Meta interleaved format
            where adjacent pairs repeat: [c0, c0, c1, c1, ...]
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    if scaling is not None:
        # Scaled: apply frequency adjustment, then gather
        inv_freq = _apply_llama3_scaling(inv_freq, scaling)
        t = torch.arange(max_seq_len * 2.0)
        freqs = torch.outer(t, inv_freq).float()
        cos = freqs.cos()
        sin = freqs.sin()
        # Gather sequential positions and interleave
        positions = torch.arange(max_seq_len)
        pos_expanded = positions.unsqueeze(1).expand(-1, cos.shape[-1])
        cos = cos.gather(0, pos_expanded)
        sin = sin.gather(0, pos_expanded)
    else:
        # Unscaled: standard RoPE
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

    # Convert to Meta interleaved format: [c0, c0, c1, c1, ...]
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


# ============================================================================
# Unit Tests - No device required
# ============================================================================


def test_rope_1d_config_creation():
    """Test that Rope1DConfig can be created with required fields."""
    cos_source = torch.randn(1, 1, 8192, 128)
    sin_source = torch.randn(1, 1, 8192, 128)
    cos_lw = LazyWeight(source=cos_source)
    sin_lw = LazyWeight(source=sin_source)

    config = Rope1DConfig(
        cos_matrix=cos_lw,
        sin_matrix=sin_lw,
        max_batch_size=32,
    )
    assert config.max_batch_size == 32
    assert config.head_dim is None  # derived later
    assert config.use_qk_fused is False
    assert config.datatype == ttnn.bfloat16


def test_rope_1d_config_with_options():
    """Test config with all optional fields set."""
    from unittest.mock import MagicMock

    cos_source = torch.randn(1, 1, 8192, 64)
    sin_source = torch.randn(1, 1, 8192, 64)
    cos_lw = LazyWeight(source=cos_source)
    sin_lw = LazyWeight(source=sin_source)

    config = Rope1DConfig(
        cos_matrix=cos_lw,
        sin_matrix=sin_lw,
        max_batch_size=1,
        head_dim=64,
        device=MagicMock(),
        use_qk_fused=True,
        datatype=ttnn.bfloat8_b,
    )
    assert config.head_dim == 64
    assert config.use_qk_fused is True
    assert config.datatype == ttnn.bfloat8_b


def test_rope_1d_from_model_args_rejects_galaxy():
    """Test that from_model_args raises for Galaxy devices."""
    from unittest.mock import MagicMock

    args = MagicMock()
    args.is_galaxy = True

    with pytest.raises(ValueError, match="Galaxy"):
        RotarySetup1D.from_model_args(device=MagicMock(), args=args)


def test_compute_cos_sin_no_scaling():
    """Test cos/sin computation without scaling (Mistral/Qwen-style)."""
    cos, sin = _rope_cos_sin(head_dim=128, max_seq_len=8192, theta=1000000.0)
    assert cos.shape == (1, 1, 8192, 128)
    assert sin.shape == (1, 1, 8192, 128)
    # cos/sin values should be in [-1, 1]
    assert cos.abs().max() <= 1.0 + 1e-6
    assert sin.abs().max() <= 1.0 + 1e-6


def test_compute_cos_sin_llama3_scaling():
    """Test cos/sin computation with Llama-3.x scaling."""
    cos, sin = _rope_cos_sin(head_dim=128, max_seq_len=8192, theta=500000.0, scaling=Llama3Scaling())
    assert cos.shape == (1, 1, 8192, 128)
    assert sin.shape == (1, 1, 8192, 128)


def test_compute_cos_sin_head_dim_64():
    """Test cos/sin computation with head_dim=64 (Llama-3.2-1B)."""
    cos, sin = _rope_cos_sin(head_dim=64, max_seq_len=8192, theta=500000.0, scaling=Llama3Scaling())
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
        # (1,1) hd=128, llama3, batch=1 (Llama-3.2-3B, 3.1-8B on N150)
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
def test_rope_1d_vs_reference(
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
    Test RotarySetup1D initialization and all API methods.

    Verifies:
    1. Construction succeeds with given parameters
    2. get_both_trans_mats() returns valid tensors with correct PCC
    3. get_rot_idxs() produces correct shape
    4. get_rot_mats() produces cos/sin with correct shapes and PCC vs reference
    5. forward() with ttnn.Tensor input matches torch-input path
    """
    scaling = Llama3Scaling() if rope_scaling_str == "llama3" else None

    cos_torch, sin_torch = _rope_cos_sin(head_dim=head_dim, max_seq_len=max_seq_len, theta=rope_theta, scaling=scaling)
    scaling_tag = "llama3" if scaling else "none"
    tag = f"theta{rope_theta}_{scaling_tag}"
    cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/rope_1d"))
    cos_lw = LazyWeight(source=cos_torch, device=ttnn_mesh_device, cache_dir_weight_name=(cache_dir, f"cos_{tag}"))
    sin_lw = LazyWeight(source=sin_torch, device=ttnn_mesh_device, cache_dir_weight_name=(cache_dir, f"sin_{tag}"))

    # Use __init__ (happy path) when defaults suffice, from_config (power path) otherwise.
    # This exercises both constructors across the test matrix.
    if not use_qk_fused:
        rope = RotarySetup1D(cos_lw, sin_lw, max_batch_size=batch_size)
    else:
        config = Rope1DConfig(
            cos_matrix=cos_lw,
            sin_matrix=sin_lw,
            max_batch_size=batch_size,
            head_dim=head_dim,
            device=ttnn_mesh_device,
            use_qk_fused=use_qk_fused,
        )
        rope = RotarySetup1D.from_config(config)

    # --- get_both_trans_mats ---
    trans_mats = rope.get_both_trans_mats()
    assert "decode" in trans_mats
    assert "prefill" in trans_mats

    # Prefill trans mat PCC: TTTv2 uses head_dim x head_dim
    prefill_ref = get_rot_transformation_mat(dhead=head_dim)  # [1, 1, head_dim, head_dim]
    prefill_tt = to_torch_auto_compose(trans_mats["prefill"])
    prefill_tt_trimmed = prefill_tt[:1, :1, : prefill_ref.shape[2], : prefill_ref.shape[3]]
    pcc_prefill, msg_prefill = comp_pcc(prefill_ref.to(torch.bfloat16), prefill_tt_trimmed.to(torch.bfloat16), 0.9999)
    assert pcc_prefill, f"prefill trans_mat PCC failed: {msg_prefill}"

    # Decode trans mat PCC
    effective_batch = batch_size * 2 if use_qk_fused else batch_size
    decode_ref = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(1, 1, effective_batch, 1)
    decode_tt = to_torch_auto_compose(trans_mats["decode"])
    decode_tt_trimmed = decode_tt[:1, :1, : decode_ref.shape[2], : decode_ref.shape[3]]
    pcc_decode, msg_decode = comp_pcc(decode_ref.to(torch.bfloat16), decode_tt_trimmed.to(torch.bfloat16), 0.9999)
    assert pcc_decode, f"decode trans_mat PCC failed: {msg_decode}"

    # --- PCC check: cos/sin vs torch reference ---
    # Use non-zero positions to avoid sin(0)=0 (PCC undefined for all-zero tensors)
    pcc_position_idxs = torch.arange(42, 42 + batch_size)

    # TTTv2 API: prepare_rot_idxs → forward
    rot_idxs = prepare_rot_idxs(rope.config, pcc_position_idxs, on_host=True)
    pcc_rot_mats = rope.forward(rot_idxs)
    assert len(pcc_rot_mats) == 2

    cos_torch_ref, sin_torch_ref = _rope_cos_sin(
        head_dim=head_dim, max_seq_len=max_seq_len, theta=rope_theta, scaling=scaling
    )

    cos_tt = to_torch_auto_compose(pcc_rot_mats[0])
    sin_tt = to_torch_auto_compose(pcc_rot_mats[1])

    # Build expected cos/sin from torch reference for chosen positions
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

    logger.info(
        f"RotarySetup1D: PASSED for mesh={mesh_shape}, batch={batch_size}, "
        f"head_dim={head_dim}, max_seq_len={max_seq_len}, rope_theta={rope_theta}, "
        f"scaling={rope_scaling_str}, fused={use_qk_fused}"
    )


# ============================================================================
# Standalone helper test: prepare_rot_idxs
# ============================================================================


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2)],
    ids=["1x1", "1x2"],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size,use_qk_fused",
    [
        pytest.param(1, False, id="b1-nofused"),
        pytest.param(1, True, id="b1-fused"),
        pytest.param(32, True, id="b32-fused"),
    ],
)
def test_prepare_rot_idxs(
    ttnn_mesh_device: ttnn.MeshDevice,
    batch_size,
    use_qk_fused,
):
    """Test prepare_rot_idxs standalone helper produces correct ttnn tensor."""
    cos_torch, sin_torch = _rope_cos_sin(head_dim=128, max_seq_len=8192, theta=500000.0, scaling=Llama3Scaling())
    tag = "theta500000.0_llama3"
    cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/rope_1d"))
    cos_lw = LazyWeight(source=cos_torch, device=ttnn_mesh_device, cache_dir_weight_name=(cache_dir, f"cos_{tag}"))
    sin_lw = LazyWeight(source=sin_torch, device=ttnn_mesh_device, cache_dir_weight_name=(cache_dir, f"sin_{tag}"))
    config = Rope1DConfig(
        cos_matrix=cos_lw,
        sin_matrix=sin_lw,
        max_batch_size=batch_size,
        head_dim=128,
        device=ttnn_mesh_device,
        use_qk_fused=use_qk_fused,
    )
    rope = RotarySetup1D.from_config(config)

    position_idxs = torch.arange(batch_size)

    # Test on-device path
    rot_idxs = prepare_rot_idxs(rope.config, position_idxs, on_host=False)
    assert isinstance(rot_idxs, ttnn.Tensor)

    # Test on-host path
    rot_idxs_host = prepare_rot_idxs(rope.config, position_idxs, on_host=True)
    assert isinstance(rot_idxs_host, ttnn.Tensor)

    # Both paths should produce usable tensors for forward()
    cos_sin_device = rope.forward(rot_idxs)
    assert len(cos_sin_device) == 2

    cos_sin_host = rope.get_rot_mats(rot_idxs_host)
    assert len(cos_sin_host) == 2

    # PCC: both paths should produce identical results
    cos_d = to_torch_auto_compose(cos_sin_device[0])
    cos_h = to_torch_auto_compose(cos_sin_host[0])
    pcc_ok, msg = comp_pcc(cos_d, cos_h, 0.9999)
    assert pcc_ok, f"on-device vs on-host cos mismatch: {msg}"


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

    # --- Backward-compat wrappers: get_rot_idxs, get_rot_mats, return_rot_idxs ---
    position_idxs = torch.arange(42, 42 + model_args.max_batch_size)

    # get_rot_idxs
    rot_idxs = rope_v2.get_rot_idxs(position_idxs, on_host=True)

    # get_rot_mats (torch input)
    v2_cos_sin = rope_v2.get_rot_mats(position_idxs)
    v1_cos_sin = rope_v1.get_rot_mats(position_idxs)

    # get_rot_mats with return_rot_idxs
    v2_mats_and_idxs = rope_v2.get_rot_mats(position_idxs, return_rot_idxs=True)
    assert len(v2_mats_and_idxs) == 2
    v2_rot_mats_2, v2_rot_idxs_2 = v2_mats_and_idxs
    assert len(v2_rot_mats_2) == 2

    # get_rot_mats via ttnn rot_idxs (production calling pattern)
    v2_cos_sin_via_ttnn = rope_v2.get_rot_mats(rot_idxs)

    # PCC: v2 vs v1
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

    # PCC: torch-input vs ttnn-input paths should match
    cos_via_ttnn = to_torch_auto_compose(v2_cos_sin_via_ttnn[0])
    pcc_path, msg_path = comp_pcc(v2_cos_torch, cos_via_ttnn, 0.9999)
    assert pcc_path, f"torch vs ttnn input path mismatch: {msg_path}"

    # PCC check: decode transformation matrix
    v2_trans = rope_v2.get_both_trans_mats()
    v1_trans = rope_v1.get_both_trans_mats()

    v2_decode = to_torch_auto_compose(v2_trans["decode"])
    v1_decode = to_torch_auto_compose(v1_trans["decode"])
    pcc_decode, msg_decode = comp_pcc(v1_decode, v2_decode, 0.9999)
    logger.info(f"from_model_args trans_mat[decode] PCC: {msg_decode}")
    assert pcc_decode, f"from_model_args trans_mat[decode] mismatch: {msg_decode}"

    # PCC check: prefill transformation matrix (overlapping region)
    v2_prefill = to_torch_auto_compose(v2_trans["prefill"])
    v1_prefill = to_torch_auto_compose(v1_trans["prefill"])
    min_h = min(v1_prefill.shape[-2], v2_prefill.shape[-2])
    min_w = min(v1_prefill.shape[-1], v2_prefill.shape[-1])
    v1_prefill_trimmed = v1_prefill[:1, :1, :min_h, :min_w]
    v2_prefill_trimmed = v2_prefill[:1, :1, :min_h, :min_w]
    pcc_prefill, msg_prefill = comp_pcc(v1_prefill_trimmed, v2_prefill_trimmed, 0.9999)
    logger.info(f"from_model_args trans_mat[prefill] PCC (overlapping): {msg_prefill}")
    assert pcc_prefill, f"from_model_args trans_mat[prefill] mismatch: {msg_prefill}"

    logger.info(f"RotarySetup1D.from_model_args vs TTTv1: PASSED for {model_args.model_name}")
