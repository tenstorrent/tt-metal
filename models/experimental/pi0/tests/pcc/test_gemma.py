# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for Gemma components.

Tests:
    - GemmaAttention: Torch vs TTNN attention with RoPE
    - GemmaMLP: Torch vs TTNN GeGLU MLP
    - GemmaBlock: Full transformer block comparison
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.pi0.common.configs import GemmaConfig
from models.experimental.pi0.reference.torch_gemma import (
    GemmaAttention as GemmaAttentionTorch,
    GemmaMLP as GemmaMLPTorch,
    GemmaBlock as GemmaBlockTorch,
    precompute_freqs_cis,
)
from models.experimental.pi0.tt.ttnn_gemma import (
    TtGemmaAttention,
    TtGemmaMLP,
    TtGemmaBlock,
)


@pytest.fixture
def gemma_config():
    """Default Gemma config for testing."""
    return GemmaConfig(
        width=512,
        depth=2,
        mlp_dim=2048,
        num_heads=8,
        num_kv_heads=1,
        head_dim=64,
    )


def create_gemma_attention_weights(config: GemmaConfig):
    """Create random weights for Gemma attention."""
    return {
        "self_attn.q_proj.weight": torch.randn(config.num_heads * config.head_dim, config.width),
        "self_attn.k_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
        "self_attn.v_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
        "self_attn.o_proj.weight": torch.randn(config.width, config.num_heads * config.head_dim),
    }


def create_gemma_mlp_weights(config: GemmaConfig):
    """Create random weights for Gemma MLP."""
    return {
        "mlp.gate_proj.weight": torch.randn(config.mlp_dim, config.width),
        "mlp.up_proj.weight": torch.randn(config.mlp_dim, config.width),
        "mlp.down_proj.weight": torch.randn(config.width, config.mlp_dim),
    }


def create_gemma_block_weights(config: GemmaConfig, layer_idx: int = 0):
    """Create random weights for full Gemma block."""
    weights = {}
    weights.update(create_gemma_attention_weights(config))
    weights.update(create_gemma_mlp_weights(config))
    weights["input_layernorm.weight"] = torch.randn(config.width)
    weights["post_attention_layernorm.weight"] = torch.randn(config.width)
    return weights


def convert_weights_to_ttnn(weights: dict, device: ttnn.Device):
    """Convert PyTorch weights to TTNN format."""
    ttnn_weights = {}
    for key, value in weights.items():
        # Transpose weight matrices for TTNN linear
        if "weight" in key and "layernorm" not in key and "norm" not in key:
            value = value.T.contiguous()
            layout = ttnn.TILE_LAYOUT
        else:
            if value.dim() == 1:
                value = value.unsqueeze(0)
            layout = ttnn.TILE_LAYOUT
        
        ttnn_weights[key] = ttnn.from_torch(
            value,
            dtype=ttnn.bfloat16,
            layout=layout,
            device=device,
        )
    return ttnn_weights


class TestGemmaAttention:
    """PCC tests for Gemma Attention."""
    
    @pytest.mark.parametrize("seq_len", [32, 64, 128])
    def test_gemma_attention_pcc(self, device, gemma_config, seq_len):
        """Test Gemma attention Torch vs TTNN."""
        config = gemma_config
        weights = create_gemma_attention_weights(config)
        weights_ttnn = convert_weights_to_ttnn(weights, device)
        
        # Create input
        hidden = torch.randn(1, 1, seq_len, config.width)
        cos, sin = precompute_freqs_cis(config.head_dim, seq_len)
        
        # Torch forward
        attn_torch = GemmaAttentionTorch(config, weights, layer_idx=0)
        out_torch, _ = attn_torch.forward(hidden, cos, sin)
        
        # TTNN forward
        attn_ttnn = TtGemmaAttention(config, weights_ttnn, layer_idx=0, device=device)
        hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cos_ttnn = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        sin_ttnn = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn, _ = attn_ttnn.forward(hidden_ttnn, cos_ttnn, sin_ttnn)
        out_ttnn_torch = ttnn.to_torch(out_ttnn)
        
        assert_with_pcc(out_torch, out_ttnn_torch, pcc=0.90)


class TestGemmaMLP:
    """PCC tests for Gemma MLP (GeGLU)."""
    
    @pytest.mark.parametrize("seq_len", [32, 64])
    def test_gemma_mlp_pcc(self, device, gemma_config, seq_len):
        """Test Gemma MLP Torch vs TTNN."""
        config = gemma_config
        weights = create_gemma_mlp_weights(config)
        weights_ttnn = convert_weights_to_ttnn(weights, device)
        
        # Create input
        hidden = torch.randn(1, 1, seq_len, config.width)
        
        # Torch forward
        mlp_torch = GemmaMLPTorch(config, weights)
        out_torch = mlp_torch.forward(hidden)
        
        # TTNN forward
        mlp_ttnn = TtGemmaMLP(config, weights_ttnn, device)
        hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = mlp_ttnn.forward(hidden_ttnn)
        out_ttnn_torch = ttnn.to_torch(out_ttnn)
        
        assert_with_pcc(out_torch, out_ttnn_torch, pcc=0.95)


class TestGemmaBlock:
    """PCC tests for full Gemma Block."""
    
    @pytest.mark.parametrize("seq_len", [32, 64])
    def test_gemma_block_pcc(self, device, gemma_config, seq_len):
        """Test full Gemma block Torch vs TTNN."""
        config = gemma_config
        weights = create_gemma_block_weights(config)
        weights_ttnn = convert_weights_to_ttnn(weights, device)
        
        # Create input
        hidden = torch.randn(1, 1, seq_len, config.width)
        cos, sin = precompute_freqs_cis(config.head_dim, seq_len)
        
        # Torch forward
        block_torch = GemmaBlockTorch(config, weights, layer_idx=0)
        out_torch, _ = block_torch.forward(hidden, cos, sin)
        
        # TTNN forward
        block_ttnn = TtGemmaBlock(config, weights_ttnn, layer_idx=0, device=device)
        hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cos_ttnn = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        sin_ttnn = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn, _ = block_ttnn.forward(hidden_ttnn, cos_ttnn, sin_ttnn)
        out_ttnn_torch = ttnn.to_torch(out_ttnn)
        
        assert_with_pcc(out_torch, out_ttnn_torch, pcc=0.85)
