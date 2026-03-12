# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for KimiLazyStateDict and dequantize_i32_packed.

Run in the tt-metal dev container (has PyTorch + safetensors)::

    pytest models/demos/kimi_k25/tests/test_weight_loader.py -v

Integration test against real weights (requires KIMI_HF_MODEL env var)::

    KIMI_HF_MODEL=/workspace/extra/Kimi-K2.5 \
        pytest models/demos/kimi_k25/tests/test_weight_loader.py -v -k real

Test structure
--------------
Unit tests (no real weights needed):
  - test_dequantize_i32_packed_roundtrip: direct function test, PCC > 0.9999
  - test_dequantize_i32_shape_dtype: shape and dtype assertions
  - test_dequantize_i32_wrong_dtype_raises: TypeError for uint8 input
  - test_dequantize_i32_shape_mismatch_raises: ValueError on bad weight_shape
  - test_kimi_state_dict_expert_weight_shape_dtype: KimiLazyStateDict returns BF16
  - test_kimi_state_dict_expert_weight_pcc: round-trip PCC > 0.9999
  - test_kimi_state_dict_passthrough_bf16: attention weights unchanged
  - test_kimi_state_dict_scale_accessible: weight_scale key readable
  - test_kimi_state_dict_missing_key_raises: KeyError for missing key
  - test_kimi_state_dict_cache_hit: second access returns same tensor

Integration tests (require KIMI_HF_MODEL):
  - test_real_gate_proj_shape_dtype: real gate_proj expert weight
  - test_real_weight_not_nan_not_zero: sanity checks on real values
  - test_real_down_proj_shape: down_proj has shape (7168, 2048)
  - test_real_attention_passthrough: BF16 attention weight untouched
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import pytest
import torch
from safetensors.torch import save_file

from models.demos.kimi_k25.utils.weight_loader import (
    KimiLazyStateDict,
    KIMI_MODEL_PREFIX,
    dequantize_i32_packed,
    is_int4_packed_key,
    packed_key_to_scale_key,
    packed_key_to_shape_key,
)


# ── Test helpers ─────────────────────────────────────────────────────────────


def _pack_int4_to_i32(weights_i8: torch.Tensor) -> torch.Tensor:
    """Reference packer: int8 weights in [-8, 7] → I32 (8 nibbles per int32).

    Packing layout (matching compressed_tensors / Kimi K2.5 format):
      - 2 nibbles per byte; low nibble = even element, high nibble = odd element
      - 4 bytes per I32, byte 0 first (little-endian)
      ⟹ Element index 2k   → low  nibble of byte k//2
         Element index 2k+1 → high nibble of byte k//2
    """
    out_f, in_f = weights_i8.shape
    assert in_f % 8 == 0, f"in_features ({in_f}) must be divisible by 8"

    u4 = (weights_i8 + 8).to(torch.uint8)  # unsigned [0, 15]
    n_bytes = in_f // 2
    u8 = torch.zeros(out_f, n_bytes, dtype=torch.uint8)
    for j in range(n_bytes):
        u8[:, j] = u4[:, 2 * j] | (u4[:, 2 * j + 1] << 4)

    # Every 4 consecutive bytes → one int32 (shape: out_f, in_f // 8)
    return u8.view(torch.int32)


def _make_checkpoint(
    tmp_path: Path,
    out_f: int = 16,
    in_f: int = 64,
    group_size: int = 32,
    include_weight_shape: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a minimal Kimi-like safetensors checkpoint in *tmp_path*.

    Returns:
        (weights_i8, scales, attn_weight) — original tensors for reference.
    """
    assert in_f % group_size == 0
    n_groups = in_f // group_size

    torch.manual_seed(42)
    weights_i8 = torch.randint(-8, 8, (out_f, in_f), dtype=torch.int8)
    scales = (torch.randn(out_f, n_groups, dtype=torch.float32).abs() + 0.01).to(
        torch.bfloat16
    )
    attn_weight = torch.randn(out_f, in_f, dtype=torch.bfloat16)

    packed_i32 = _pack_int4_to_i32(weights_i8)  # (out_f, in_f // 8)

    prefix = KIMI_MODEL_PREFIX  # "language_model.model."
    tensors: dict[str, torch.Tensor] = {
        f"{prefix}layers.1.mlp.experts.0.gate_proj.weight_packed": packed_i32,
        f"{prefix}layers.1.mlp.experts.0.gate_proj.weight_scale": scales,
        f"{prefix}layers.1.self_attn.q_proj.weight": attn_weight,
    }
    if include_weight_shape:
        shape_tensor = torch.tensor([out_f, in_f], dtype=torch.int32)
        tensors[f"{prefix}layers.1.mlp.experts.0.gate_proj.weight_shape"] = shape_tensor

    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, tmp_path / shard)

    index = {"weight_map": {k: shard for k in tensors}}
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    return weights_i8, scales, attn_weight


def _reference_dequant(
    weights_i8: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """Compute reference dequantization in float32 for PCC comparison."""
    out_f, in_f = weights_i8.shape
    n_groups = in_f // group_size
    w = weights_i8.float().reshape(out_f, n_groups, group_size)
    s = scales.float().unsqueeze(-1)  # (out, n_groups, 1)
    return (w * s).reshape(out_f, in_f)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ── Unit tests: dequantize_i32_packed ────────────────────────────────────────


class TestDequantizeI32Packed:
    """Direct unit tests for the :func:`dequantize_i32_packed` helper."""

    @pytest.fixture
    def weight_data(self):
        torch.manual_seed(7)
        out_f, in_f = 8, 64
        weights_i8 = torch.randint(-8, 8, (out_f, in_f), dtype=torch.int8)
        scales = (torch.randn(out_f, 2, dtype=torch.float32).abs() + 0.01).to(
            torch.bfloat16
        )
        packed_i32 = _pack_int4_to_i32(weights_i8)
        return weights_i8, scales, packed_i32

    def test_shape_dtype(self, weight_data):
        _, scales, packed_i32 = weight_data
        result = dequantize_i32_packed(packed_i32, scales)
        assert result.shape == (8, 64)
        assert result.dtype == torch.bfloat16

    def test_pcc_roundtrip(self, weight_data):
        weights_i8, scales, packed_i32 = weight_data
        result = dequantize_i32_packed(packed_i32, scales)
        ref = _reference_dequant(weights_i8, scales)
        pcc = _pcc(result, ref)
        assert pcc > 0.9999, f"PCC={pcc:.6f} < 0.9999"

    def test_not_nan_not_inf(self, weight_data):
        _, scales, packed_i32 = weight_data
        result = dequantize_i32_packed(packed_i32, scales)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_wrong_dtype_raises(self, weight_data):
        _, scales, packed_i32 = weight_data
        packed_u8 = packed_i32.view(torch.uint8)
        with pytest.raises(TypeError, match="torch.int32"):
            dequantize_i32_packed(packed_u8, scales)

    def test_weight_shape_validation_pass(self, weight_data):
        _, scales, packed_i32 = weight_data
        shape_tensor = torch.tensor([8, 64], dtype=torch.int32)
        # Should not raise
        result = dequantize_i32_packed(packed_i32, scales, weight_shape=shape_tensor)
        assert result.shape == (8, 64)

    def test_weight_shape_validation_fail(self, weight_data):
        _, scales, packed_i32 = weight_data
        wrong_shape = torch.tensor([8, 128], dtype=torch.int32)  # wrong in_features
        with pytest.raises(ValueError, match="weight_shape"):
            dequantize_i32_packed(packed_i32, scales, weight_shape=wrong_shape)


# ── Unit tests: key helpers ───────────────────────────────────────────────────


def test_is_int4_packed_key():
    assert is_int4_packed_key("layers.1.mlp.experts.0.gate_proj.weight_packed")
    assert not is_int4_packed_key("layers.1.mlp.experts.0.gate_proj.weight_scale")
    assert not is_int4_packed_key("layers.0.self_attn.q_proj.weight")


def test_packed_to_scale_key():
    k = "layers.1.mlp.experts.0.gate_proj.weight_packed"
    assert packed_key_to_scale_key(k) == "layers.1.mlp.experts.0.gate_proj.weight_scale"


def test_packed_to_shape_key():
    k = "layers.1.mlp.experts.0.gate_proj.weight_packed"
    assert packed_key_to_shape_key(k) == "layers.1.mlp.experts.0.gate_proj.weight_shape"


# ── Unit tests: KimiLazyStateDict ────────────────────────────────────────────


@pytest.fixture
def ckpt(tmp_path):
    weights_i8, scales, attn_weight = _make_checkpoint(tmp_path)
    return {
        "path": tmp_path,
        "weights_i8": weights_i8,
        "scales": scales,
        "attn_weight": attn_weight,
        "out_f": 16,
        "in_f": 64,
    }


class TestKimiLazyStateDict:
    def test_expert_weight_dtype(self, ckpt):
        state = KimiLazyStateDict(ckpt["path"])
        w = state["layers.1.mlp.experts.0.gate_proj.weight_packed"]
        assert w.dtype == torch.bfloat16

    def test_expert_weight_shape(self, ckpt):
        state = KimiLazyStateDict(ckpt["path"])
        w = state["layers.1.mlp.experts.0.gate_proj.weight_packed"]
        assert w.shape == (ckpt["out_f"], ckpt["in_f"])

    def test_expert_weight_pcc(self, ckpt):
        state = KimiLazyStateDict(ckpt["path"])
        w = state["layers.1.mlp.experts.0.gate_proj.weight_packed"]
        ref = _reference_dequant(ckpt["weights_i8"], ckpt["scales"])
        pcc = _pcc(w, ref)
        assert pcc > 0.9999, f"PCC={pcc:.6f} < 0.9999"

    def test_passthrough_bf16_weight(self, ckpt):
        state = KimiLazyStateDict(ckpt["path"])
        q = state["layers.1.self_attn.q_proj.weight"]
        assert q.dtype == torch.bfloat16
        assert torch.allclose(q, ckpt["attn_weight"])

    def test_scale_key_accessible(self, ckpt):
        state = KimiLazyStateDict(ckpt["path"])
        scales = state["layers.1.mlp.experts.0.gate_proj.weight_scale"]
        assert scales.dtype == torch.bfloat16
        assert scales.shape == (ckpt["out_f"], ckpt["in_f"] // 32)

    def test_missing_key_raises(self, ckpt):
        state = KimiLazyStateDict(ckpt["path"])
        with pytest.raises(KeyError):
            _ = state["layers.99.mlp.experts.0.gate_proj.weight_packed"]

    def test_cache_hit_returns_same_object(self, ckpt):
        """Second access should return the cached tensor (same identity)."""
        state = KimiLazyStateDict(ckpt["path"])
        w1 = state["layers.1.mlp.experts.0.gate_proj.weight_packed"]
        w2 = state["layers.1.mlp.experts.0.gate_proj.weight_packed"]
        assert w1 is w2

    def test_len_matches_visible_keys(self, ckpt):
        """Len should reflect only language_model.model.* keys."""
        state = KimiLazyStateDict(ckpt["path"])
        # Our fixture: weight_packed, weight_scale, weight_shape, attn weight = 4
        assert len(state) == 4

    def test_expert_weight_not_nan(self, ckpt):
        state = KimiLazyStateDict(ckpt["path"])
        w = state["layers.1.mlp.experts.0.gate_proj.weight_packed"]
        assert not torch.isnan(w).any()
        assert not torch.isinf(w).any()

    def test_no_weight_shape_still_works(self, tmp_path):
        """KimiLazyStateDict dequantizes correctly even without weight_shape."""
        weights_i8, scales, _ = _make_checkpoint(
            tmp_path, include_weight_shape=False
        )
        state = KimiLazyStateDict(tmp_path)
        w = state["layers.1.mlp.experts.0.gate_proj.weight_packed"]
        assert w.shape == (16, 64)
        ref = _reference_dequant(weights_i8, scales)
        assert _pcc(w, ref) > 0.9999


# ── Integration tests (require real Kimi K2.5 weights) ───────────────────────

KIMI_HF_MODEL = os.environ.get("KIMI_HF_MODEL", "/workspace/extra/Kimi-K2.5")
_REAL_WEIGHTS = pytest.mark.skipif(
    not Path(KIMI_HF_MODEL).is_dir(),
    reason=f"Real Kimi weights not found at {KIMI_HF_MODEL}. "
    "Set KIMI_HF_MODEL env var or copy weights to default path.",
)


@_REAL_WEIGHTS
def test_real_gate_proj_shape_dtype():
    """gate_proj for expert 0 of layer 1 → (2048, 7168) BF16."""
    state = KimiLazyStateDict(KIMI_HF_MODEL)
    w = state["layers.1.mlp.experts.0.gate_proj.weight_packed"]
    assert w.dtype == torch.bfloat16, f"Expected BF16, got {w.dtype}"
    assert w.shape == (2048, 7168), f"Expected (2048, 7168), got {w.shape}"


@_REAL_WEIGHTS
def test_real_weight_not_nan_not_zero():
    """Dequantized real weight is finite and non-trivial."""
    state = KimiLazyStateDict(KIMI_HF_MODEL)
    w = state["layers.1.mlp.experts.0.gate_proj.weight_packed"]
    assert not torch.isnan(w).any(), "NaN in real dequantized weight"
    assert not torch.isinf(w).any(), "Inf in real dequantized weight"
    assert w.abs().max() > 1e-6, "All-zero dequantized weight (suspicious)"
    print(
        f"\n[INFO] gate_proj expert 0 layer 1: "
        f"shape={w.shape}, max={w.abs().max():.5f}, mean={w.abs().mean():.6f}"
    )


@_REAL_WEIGHTS
def test_real_down_proj_shape():
    """down_proj for expert 0 of layer 1 → (7168, 2048) BF16."""
    state = KimiLazyStateDict(KIMI_HF_MODEL)
    w = state["layers.1.mlp.experts.0.down_proj.weight_packed"]
    assert w.shape == (7168, 2048), f"Expected (7168, 2048), got {w.shape}"
    assert w.dtype == torch.bfloat16


@_REAL_WEIGHTS
def test_real_attention_passthrough():
    """Layer-0 self-attention weight is BF16 and not dequantized."""
    state = KimiLazyStateDict(KIMI_HF_MODEL)
    # Layer 0 is dense (not MoE) — all weights are native BF16
    # Access via weight_scale key would fail (no INT4 for attn)
    # Just verify a self_attn key exists and returns BF16
    # Key: layers.0.self_attn.q_proj.weight (no weight_packed suffix)
    q = state["layers.0.self_attn.q_proj.weight"]
    assert q.dtype == torch.bfloat16, f"Expected BF16, got {q.dtype}"
    assert q.ndim == 2
    print(f"\n[INFO] q_proj layer 0: shape={q.shape}")
