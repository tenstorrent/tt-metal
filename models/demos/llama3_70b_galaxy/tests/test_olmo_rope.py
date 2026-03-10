# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Test OLMo YaRN RoPE TTNN implementation against PyTorch reference.

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_rope.py -v
"""

import pytest
import torch
import ttnn

from models.demos.llama3_70b_galaxy.reference.yarn_rope import (
    YaRNConfig,
    precompute_yarn_freqs,
    apply_rotary_emb_yarn,
)
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    precompute_freqs_yarn,
    gather_cos_sin,
)
from models.common.utility_functions import comp_pcc


class TestYaRNRoPEReference:
    """Test YaRN RoPE reference implementation."""

    def test_yarn_config(self):
        """Test YaRN config creation."""
        config = YaRNConfig.from_olmo()
        assert config.dim == 128
        assert config.base == 500000.0
        assert config.scaling_factor == 8.0
        assert config.attention_factor == 1.2079441541679836

    def test_yarn_freqs_shape(self):
        """Test YaRN frequency tensor shapes."""
        config = YaRNConfig.from_olmo()
        cos, sin, mscale = precompute_yarn_freqs(config, seq_len=128)

        assert cos.shape == (128, 64), f"cos shape: {cos.shape}"
        assert sin.shape == (128, 64), f"sin shape: {sin.shape}"
        assert mscale == config.attention_factor

    def test_yarn_freqs_no_nan(self):
        """Test YaRN frequencies have no NaN/Inf."""
        config = YaRNConfig.from_olmo()
        cos, sin, mscale = precompute_yarn_freqs(config, seq_len=1024)

        assert not torch.isnan(cos).any(), "cos contains NaN"
        assert not torch.isnan(sin).any(), "sin contains NaN"
        assert not torch.isinf(cos).any(), "cos contains Inf"
        assert not torch.isinf(sin).any(), "sin contains Inf"

    def test_rotary_emb_application(self):
        """Test rotary embedding application preserves shapes."""
        config = YaRNConfig.from_olmo()
        batch, seq_len, n_heads, n_kv_heads, head_dim = 1, 128, 40, 8, 128

        cos, sin, _ = precompute_yarn_freqs(config, seq_len=seq_len)

        q = torch.randn(batch, seq_len, n_heads, head_dim)
        k = torch.randn(batch, seq_len, n_kv_heads, head_dim)

        q_rot, k_rot = apply_rotary_emb_yarn(q, k, cos, sin)

        assert q_rot.shape == q.shape, f"q_rot shape: {q_rot.shape}"
        assert k_rot.shape == k.shape, f"k_rot shape: {k_rot.shape}"


class TestYaRNRoPETTNN:
    """Test TTNN YaRN RoPE frequency computation matches reference."""

    def test_yarn_freqs_pcc(self):
        """Test TTNN YaRN frequency computation PCC against reference."""
        # OLMo YaRN config
        dim = 128  # head_dim
        seq_len = 1024
        theta = 500000.0
        scaling_factor = 8.0
        original_max_position_embeddings = 8192
        beta_fast = 32.0
        beta_slow = 1.0
        attention_factor = 1.2079441541679836

        # Reference (from yarn_rope.py)
        config = YaRNConfig(
            dim=dim,
            base=theta,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            attention_factor=attention_factor,
        )
        ref_cos, ref_sin, ref_mscale = precompute_yarn_freqs(config, seq_len=seq_len)

        # TTNN implementation (from llama_common.py)
        ttnn_cos, ttnn_sin, ttnn_mscale = precompute_freqs_yarn(
            dim=dim,
            end=seq_len,
            theta=theta,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            attention_factor=attention_factor,
        )

        # Compare mscale
        assert ref_mscale == ttnn_mscale, f"mscale mismatch: {ref_mscale} vs {ttnn_mscale}"

        # Compare cos/sin PCC
        pcc_required = 0.9999
        passing_cos, pcc_cos = comp_pcc(ref_cos, ttnn_cos, pcc_required)
        passing_sin, pcc_sin = comp_pcc(ref_sin, ttnn_sin, pcc_required)

        print(f"YaRN cos PCC: {pcc_cos}")
        print(f"YaRN sin PCC: {pcc_sin}")

        assert passing_cos, f"cos PCC {pcc_cos} < {pcc_required}"
        assert passing_sin, f"sin PCC {pcc_sin} < {pcc_required}"


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],  # Galaxy TG mesh shape
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestYaRNRoPETTNNDevice:
    """Test YaRN RoPE on TTNN device."""

    def test_yarn_rope_prefill_pcc(self, mesh_device):
        """Test YaRN RoPE prefill on device vs reference."""

        # OLMo config
        head_dim = 128
        seq_len = 128
        max_seq_len = 4096
        scale_factor = 8.0

        # Reference YaRN cos/sin
        config = YaRNConfig.from_olmo()
        ref_cos, ref_sin, mscale = precompute_yarn_freqs(config, seq_len=max_seq_len * 2)

        # Gather for positions 0 to seq_len
        position_ids = torch.arange(seq_len)
        ref_cos_gathered, ref_sin_gathered = gather_cos_sin(position_ids, ref_cos, ref_sin)

        # TTNN rot mats (using existing infrastructure)
        # Note: get_prefill_rot_mat uses precompute_freqs which uses Llama scaling
        # For OLMo, we need to use precompute_freqs_yarn
        ttnn_cos, ttnn_sin, _ = precompute_freqs_yarn(
            dim=head_dim,
            end=max_seq_len * 2,
            theta=500000.0,
            scaling_factor=scale_factor,
            original_max_position_embeddings=8192,
            beta_fast=32.0,
            beta_slow=1.0,
            attention_factor=1.2079,
        )
        ttnn_cos_gathered, ttnn_sin_gathered = gather_cos_sin(position_ids, ttnn_cos, ttnn_sin)

        # Convert to TTNN tensors
        cos_tt = ttnn.from_torch(
            ttnn_cos_gathered,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        sin_tt = ttnn.from_torch(
            ttnn_sin_gathered,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Convert back to torch and compare
        cos_back = ttnn.to_torch(
            cos_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=(8, 4)),
        )
        sin_back = ttnn.to_torch(
            sin_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=(8, 4)),
        )

        # Take first replica
        cos_back = cos_back[0:1, :, :, :head_dim]
        sin_back = sin_back[0:1, :, :, :head_dim]

        # Compare PCC
        pcc_required = 0.99
        passing_cos, pcc_cos = comp_pcc(ref_cos_gathered.float(), cos_back.float(), pcc_required)
        passing_sin, pcc_sin = comp_pcc(ref_sin_gathered.float(), sin_back.float(), pcc_required)

        print(f"YaRN RoPE cos PCC: {pcc_cos}")
        print(f"YaRN RoPE sin PCC: {pcc_sin}")

        assert passing_cos, f"cos PCC {pcc_cos} < {pcc_required}"
        assert passing_sin, f"sin PCC {pcc_sin} < {pcc_required}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
