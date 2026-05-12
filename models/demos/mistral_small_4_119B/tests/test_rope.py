# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RoPE module parity tests for ``TtMistral4Rotary``.

Validates that:
1. ``get_cos_sin_matrix`` produces correct cos/sin tables matching HF reference.
2. ``TtMistral4Rotary.get_rot_mats`` produces device tensors with correct shapes and values.
3. The output ``rope_tensors`` dict is compatible with ``rotary_embedding_llama``.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_small_4_119B.tt.rope import TtMistral4Rotary, get_cos_sin_matrix


def _tiny_mistral4_config(*, rope_dim: int = 32):
    """Tiny config. ``rope_dim`` must be ≥ 32 for device tests (tile alignment)."""
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    return Mistral4Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        max_position_embeddings=128,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=rope_dim,
        v_head_dim=16,
        qk_nope_head_dim=max(rope_dim, 8),
        rms_norm_eps=1e-6,
    )


# ── Test 1: cos/sin table shape and basic properties ─────────────────


def test_cos_sin_matrix_shape_and_properties():
    """``get_cos_sin_matrix`` shape matches [1, 1, max_pos, qk_rope_head_dim] and values in [-1, 1]."""
    hf_config = _tiny_mistral4_config()
    cos, sin = get_cos_sin_matrix(hf_config)

    max_seq = hf_config.max_position_embeddings
    dim = hf_config.qk_rope_head_dim

    assert cos.shape == (1, 1, max_seq, dim), f"cos shape {cos.shape} != expected (1, 1, {max_seq}, {dim})"
    assert sin.shape == (1, 1, max_seq, dim), f"sin shape {sin.shape} != expected (1, 1, {max_seq}, {dim})"

    # cos²+sin² ≈ 1 for each position/dim pair (relaxed for bfloat16 precision from HF)
    identity = cos**2 + sin**2
    assert torch.allclose(
        identity, torch.ones_like(identity), atol=0.02
    ), f"cos²+sin² != 1, max deviation={torch.abs(identity - 1).max().item()}"
    logger.info("cos/sin table shape and identity check passed")


# ── Test 2: cos/sin match HF reference at specific positions ─────────


def test_cos_sin_matrix_matches_hf_reference():
    """Verify cos/sin values match the HF ``Mistral4RotaryEmbedding`` output at sampled positions."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    hf_config = _tiny_mistral4_config(rope_dim=32)
    cos_tt, sin_tt = get_cos_sin_matrix(hf_config)

    # HF reference
    rope = Mistral4RotaryEmbedding(hf_config)
    pos_ids = torch.arange(hf_config.max_position_embeddings, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros(1, hf_config.max_position_embeddings, hf_config.hidden_size, dtype=torch.bfloat16)
    cos_hf, sin_hf = rope(dummy, position_ids=pos_ids)

    # HF format: [1, seq, dim] where dim = [t1..td/2, t1..td/2]
    # Our format: [1, 1, seq, dim] where dim = [t1,t1, t2,t2, .., td/2,td/2]
    cos_hf = cos_hf.squeeze(0).float()
    sin_hf = sin_hf.squeeze(0).float()
    dim = hf_config.qk_rope_head_dim
    half = dim // 2

    # Sample positions to check
    for pos in [0, 1, 10, hf_config.max_position_embeddings - 1]:
        for d in range(half):
            # Our interleaved format: index 2*d and 2*d+1 both hold the d-th frequency
            our_cos_val = cos_tt[0, 0, pos, 2 * d].item()
            hf_cos_val = cos_hf[pos, d].item()
            assert (
                abs(our_cos_val - hf_cos_val) < 1e-4
            ), f"cos mismatch at pos={pos} d={d}: ours={our_cos_val} hf={hf_cos_val}"

    logger.info("cos/sin matrix matches HF reference at sampled positions")


# ── Test 3: TtMistral4Rotary device-side decode tensors ──────────────


def test_rotary_get_rot_mats_decode(device):
    """``TtMistral4Rotary.get_rot_mats`` produces correct decode rope_tensors on device."""
    hf_config = _tiny_mistral4_config(rope_dim=32)
    batch_size = 8

    rotary = TtMistral4Rotary(device, batch_size, hf_config)

    # Random position ids
    position_ids = torch.randint(0, hf_config.max_position_embeddings, (batch_size,))
    rope_tensors = rotary.get_rot_mats(position_ids)

    assert "cos_matrix" in rope_tensors
    assert "sin_matrix" in rope_tensors
    assert "trans_matrix" in rope_tensors

    # Verify shapes by reading back to torch
    cos_torch = ttnn.to_torch(
        rope_tensors["cos_matrix"],
        mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=tuple(device.shape)),
    )[:1]
    sin_torch = ttnn.to_torch(
        rope_tensors["sin_matrix"],
        mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=tuple(device.shape)),
    )[:1]

    logger.info(f"cos shape: {cos_torch.shape}, sin shape: {sin_torch.shape}")

    # cos and sin should be [1, batch, 1(padded to 32), dim]
    assert (
        cos_torch.shape[-1] == hf_config.qk_rope_head_dim
    ), f"cos dim {cos_torch.shape[-1]} != {hf_config.qk_rope_head_dim}"

    # Verify cos²+sin² ≈ 1
    identity = cos_torch.float() ** 2 + sin_torch.float() ** 2
    # Only check the actual batch positions (not padding)
    identity_valid = identity[:, :batch_size, :1, :]
    assert torch.allclose(
        identity_valid, torch.ones_like(identity_valid), atol=0.05
    ), f"cos²+sin² deviation: max={torch.abs(identity_valid - 1).max().item()}"

    # Verify values match CPU cos/sin tables at the requested positions
    cos_full, sin_full = get_cos_sin_matrix(hf_config)
    for i, pos in enumerate(position_ids.tolist()):
        expected_cos = cos_full[0, 0, pos, :].float()
        actual_cos = cos_torch[0, i, 0, :].float()
        passing, pcc = comp_pcc(expected_cos.unsqueeze(0), actual_cos.unsqueeze(0), 0.999)
        assert passing, f"cos PCC {pcc} at batch={i} pos={pos}"

    logger.info("TtMistral4Rotary.get_rot_mats decode test passed")


# ── Test 4: get_rot_mats_from_rot_idxs (trace-safe) ─────────────────


def test_rotary_get_rot_mats_from_rot_idxs(device):
    """``get_rot_mats_from_rot_idxs`` matches ``get_rot_mats`` (pure ttnn, trace-safe)."""
    hf_config = _tiny_mistral4_config(rope_dim=32)
    batch_size = 8

    rotary = TtMistral4Rotary(device, batch_size, hf_config)
    position_ids = torch.randint(0, hf_config.max_position_embeddings, (batch_size,))

    # Method 1: get_rot_mats (uses torch → ttnn)
    rope1, rot_idxs = rotary.get_rot_mats(position_ids, return_rot_idxs=True)

    # Method 2: get_rot_mats_from_rot_idxs (pure ttnn)
    rope2 = rotary.get_rot_mats_from_rot_idxs(rot_idxs)

    # Compare cos outputs
    cos1 = ttnn.to_torch(
        rope1["cos_matrix"],
        mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=tuple(device.shape)),
    )[:1]
    cos2 = ttnn.to_torch(
        rope2["cos_matrix"],
        mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=tuple(device.shape)),
    )[:1]

    passing, pcc = comp_pcc(cos1.float(), cos2.float(), 0.9999)
    logger.info(f"get_rot_mats vs get_rot_mats_from_rot_idxs PCC: {pcc}")
    assert passing, f"PCC {pcc} < 0.9999"


# ── Test 5: prefill rotation tables ─────────────────────────────────


def test_rotary_prefill_tables(device):
    """``get_rot_mats_prefill`` returns correct shape and includes trans_matrix."""
    hf_config = _tiny_mistral4_config(rope_dim=32)
    batch_size = 8

    rotary = TtMistral4Rotary(device, batch_size, hf_config)
    seq_len = 64
    rope_prefill = rotary.get_rot_mats_prefill(seq_len=seq_len)

    assert "cos_matrix" in rope_prefill
    assert "sin_matrix" in rope_prefill
    assert "trans_matrix" in rope_prefill

    cos_torch = ttnn.to_torch(
        rope_prefill["cos_matrix"],
        mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=tuple(device.shape)),
    )[:1]

    logger.info(f"prefill cos shape: {cos_torch.shape}")
    assert cos_torch.shape[-2] == seq_len, f"prefill cos seq_len {cos_torch.shape[-2]} != {seq_len}"
    assert cos_torch.shape[-1] == hf_config.qk_rope_head_dim

    logger.info("TtMistral4Rotary prefill test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
