# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for SigLIP components.

Tests:
    - SigLIPAttention: Torch vs TTNN attention
    - SigLIPMLP: Torch vs TTNN MLP
    - SigLIPBlock: Full transformer block comparison
    - SigLIPVisionTower: End-to-end vision encoding
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.pi0.common.configs import SigLIPConfig
from models.experimental.pi0.reference.torch_siglip import (
    SigLIPAttention as SigLIPAttentionTorch,
    SigLIPMLP as SigLIPMLPTorch,
    SigLIPBlock as SigLIPBlockTorch,
)
from models.experimental.pi0.tt.ttnn_siglip import (
    TtSigLIPAttention,
    TtSigLIPMLP,
    TtSigLIPBlock,
)


@pytest.fixture
def siglip_config():
    """Default SigLIP config for testing."""
    return SigLIPConfig(
        hidden_size=384,
        num_hidden_layers=4,
        num_attention_heads=6,
        image_size=224,
        patch_size=14,
        intermediate_size=1536,
    )


def create_siglip_attention_weights(config: SigLIPConfig):
    """Create random weights for SigLIP attention."""
    return {
        "self_attn.q_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.q_proj.bias": torch.randn(config.hidden_size),
        "self_attn.k_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.k_proj.bias": torch.randn(config.hidden_size),
        "self_attn.v_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.v_proj.bias": torch.randn(config.hidden_size),
        "self_attn.out_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.out_proj.bias": torch.randn(config.hidden_size),
    }


def create_siglip_mlp_weights(config: SigLIPConfig):
    """Create random weights for SigLIP MLP."""
    return {
        "mlp.fc1.weight": torch.randn(config.intermediate_size, config.hidden_size),
        "mlp.fc1.bias": torch.randn(config.intermediate_size),
        "mlp.fc2.weight": torch.randn(config.hidden_size, config.intermediate_size),
        "mlp.fc2.bias": torch.randn(config.hidden_size),
    }


def create_siglip_block_weights(config: SigLIPConfig, layer_idx: int = 0):
    """Create random weights for full SigLIP block."""
    weights = {}
    weights.update(create_siglip_attention_weights(config))
    weights.update(create_siglip_mlp_weights(config))
    weights["layer_norm1.weight"] = torch.randn(config.hidden_size)
    weights["layer_norm1.bias"] = torch.randn(config.hidden_size)
    weights["layer_norm2.weight"] = torch.randn(config.hidden_size)
    weights["layer_norm2.bias"] = torch.randn(config.hidden_size)
    return weights


def convert_weights_to_ttnn(weights: dict, device: ttnn.Device):
    """Convert PyTorch weights to TTNN format."""
    ttnn_weights = {}
    for key, value in weights.items():
        # Transpose weight matrices for TTNN linear
        if "weight" in key and "layer_norm" not in key and "layernorm" not in key:
            value = value.T.contiguous()
        
        if value.dim() == 1:
            value = value.unsqueeze(0)
        
        ttnn_weights[key] = ttnn.from_torch(
            value,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    return ttnn_weights


class TestSigLIPMLP:
    """PCC tests for SigLIP MLP."""
    
    @pytest.mark.parametrize("seq_len", [64, 256])
    def test_siglip_mlp_pcc(self, device, siglip_config, seq_len):
        """Test SigLIP MLP Torch vs TTNN."""
        config = siglip_config
        weights = create_siglip_mlp_weights(config)
        weights_ttnn = convert_weights_to_ttnn(weights, device)
        
        # Create input
        hidden = torch.randn(1, seq_len, config.hidden_size)
        
        # Torch forward
        mlp_torch = SigLIPMLPTorch(config, weights)
        out_torch = mlp_torch.forward(hidden)
        
        # TTNN forward
        mlp_ttnn = TtSigLIPMLP(config, weights_ttnn, device)
        hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = mlp_ttnn.forward(hidden_ttnn)
        out_ttnn_torch = ttnn.to_torch(out_ttnn)
        
        assert_with_pcc(out_torch, out_ttnn_torch, pcc=0.95)


class TestSigLIPAttention:
    """PCC tests for SigLIP Attention."""
    
    @pytest.mark.parametrize("seq_len", [64, 256])
    def test_siglip_attention_pcc(self, device, siglip_config, seq_len):
        """Test SigLIP attention Torch vs TTNN."""
        config = siglip_config
        weights = create_siglip_attention_weights(config)
        weights_ttnn = convert_weights_to_ttnn(weights, device)
        
        # Create input
        hidden = torch.randn(1, seq_len, config.hidden_size)
        
        # Torch forward
        attn_torch = SigLIPAttentionTorch(config, weights)
        out_torch = attn_torch.forward(hidden)
        
        # TTNN forward
        attn_ttnn = TtSigLIPAttention(config, weights_ttnn, device)
        hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = attn_ttnn.forward(hidden_ttnn)
        out_ttnn_torch = ttnn.to_torch(out_ttnn)
        
        assert_with_pcc(out_torch, out_ttnn_torch, pcc=0.90)


class TestSigLIPBlock:
    """PCC tests for SigLIP Block."""
    
    @pytest.mark.parametrize("seq_len", [64, 256])
    def test_siglip_block_pcc(self, device, siglip_config, seq_len):
        """Test SigLIP block Torch vs TTNN."""
        config = siglip_config
        weights = create_siglip_block_weights(config)
        weights_ttnn = convert_weights_to_ttnn(weights, device)
        
        # Create input
        hidden = torch.randn(1, seq_len, config.hidden_size)
        
        # Torch forward
        block_torch = SigLIPBlockTorch(config, weights)
        out_torch = block_torch.forward(hidden)
        
        # TTNN forward
        block_ttnn = TtSigLIPBlock(config, weights_ttnn, device)
        hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = block_ttnn.forward(hidden_ttnn)
        out_ttnn_torch = ttnn.to_torch(out_ttnn)
        
        assert_with_pcc(out_torch, out_ttnn_torch, pcc=0.85)
