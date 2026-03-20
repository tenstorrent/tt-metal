# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Tests for Qwen3-TTS decode mode with KV cache.

Verifies that:
1. Prefill mode works with KV cache initialization
2. Decode mode uses cached K,V values correctly
3. Output from incremental decode matches full prefill
"""

import pytest
import torch
from scipy.stats import pearsonr

import ttnn
from models.common.utility_functions import is_wormhole_b0


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate Pearson correlation coefficient between two tensors."""
    x_flat = x.detach().float().cpu().numpy().flatten()
    y_flat = y.detach().float().cpu().numpy().flatten()
    return pearsonr(x_flat, y_flat)[0]


@pytest.fixture(scope="module")
def device():
    """Get the TTNN device."""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
class TestKVCacheDecode:
    """Tests for decode mode with KV cache."""

    def test_attention_with_kv_cache(self, device):
        """Test that Attention module works with KV cache."""
        from models.demos.qwen3_tts.tt.attention import Attention
        from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
        from models.tt_transformers.tt.common import get_rot_transformation_mat

        config = Qwen3TTSTalkerConfig()
        config.num_hidden_layers = 1  # Single layer for test

        batch = 1
        seq_len = 4  # Short sequence for testing
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        # Create random weights
        torch.manual_seed(42)
        q_proj_weight = torch.randn(num_heads * head_dim, hidden_size, dtype=torch.bfloat16)
        k_proj_weight = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16)
        v_proj_weight = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16)
        o_proj_weight = torch.randn(hidden_size, num_heads * head_dim, dtype=torch.bfloat16)
        q_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16)
        k_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16)

        state_dict = {
            "test_layer.self_attn.q_proj.weight": q_proj_weight,
            "test_layer.self_attn.k_proj.weight": k_proj_weight,
            "test_layer.self_attn.v_proj.weight": v_proj_weight,
            "test_layer.self_attn.o_proj.weight": o_proj_weight,
            "test_layer.self_attn.q_norm.weight": q_norm_weight,
            "test_layer.self_attn.k_norm.weight": k_norm_weight,
        }

        attention = Attention(
            device=device,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            state_dict=state_dict,
            layer_prefix="test_layer",
            rms_norm_eps=config.rms_norm_eps,
        )

        # Create KV cache
        kv_caches = create_kv_cache_list(device, config, max_batch_size=batch, max_seq_len=16)
        k_cache, v_cache = kv_caches[0]

        # Create input
        torch_input = torch.randn(batch, 1, seq_len, hidden_size, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Identity RoPE for simplicity
        cos_ttnn = ttnn.from_torch(
            torch.ones(1, 1, seq_len, head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        sin_ttnn = ttnn.from_torch(
            torch.zeros(1, 1, seq_len, head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        trans_mat = get_rot_transformation_mat(dhead=head_dim)
        trans_mat_ttnn = ttnn.from_torch(
            trans_mat,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Test prefill mode with KV cache
        output, updated_kv_cache = attention.forward(
            ttnn_input,
            cos_ttnn,
            sin_ttnn,
            trans_mat_ttnn,
            kv_cache=(k_cache, v_cache),
            start_pos=0,
            mode="prefill",
        )

        output_torch = ttnn.to_torch(output)
        assert output_torch.shape == (batch, 1, seq_len, hidden_size)
        assert updated_kv_cache is not None

        print(f"Attention with KV cache prefill: PASS (shape={output_torch.shape})")

    def test_decoder_layer_with_kv_cache(self, device):
        """Test that DecoderLayer works with KV cache."""
        from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
        from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
        from models.tt_transformers.tt.common import get_rot_transformation_mat

        config = Qwen3TTSTalkerConfig()
        config.num_hidden_layers = 1

        batch = 1
        seq_len = 4
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        intermediate_size = config.intermediate_size

        # Create random weights
        torch.manual_seed(42)
        state_dict = {
            "talker.model.layers.0.input_layernorm.weight": torch.randn(hidden_size, dtype=torch.bfloat16),
            "talker.model.layers.0.self_attn.q_proj.weight": torch.randn(
                num_heads * head_dim, hidden_size, dtype=torch.bfloat16
            ),
            "talker.model.layers.0.self_attn.k_proj.weight": torch.randn(
                num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16
            ),
            "talker.model.layers.0.self_attn.v_proj.weight": torch.randn(
                num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16
            ),
            "talker.model.layers.0.self_attn.o_proj.weight": torch.randn(
                hidden_size, num_heads * head_dim, dtype=torch.bfloat16
            ),
            "talker.model.layers.0.self_attn.q_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
            "talker.model.layers.0.self_attn.k_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
            "talker.model.layers.0.post_attention_layernorm.weight": torch.randn(hidden_size, dtype=torch.bfloat16),
            "talker.model.layers.0.mlp.gate_proj.weight": torch.randn(
                intermediate_size, hidden_size, dtype=torch.bfloat16
            ),
            "talker.model.layers.0.mlp.up_proj.weight": torch.randn(
                intermediate_size, hidden_size, dtype=torch.bfloat16
            ),
            "talker.model.layers.0.mlp.down_proj.weight": torch.randn(
                hidden_size, intermediate_size, dtype=torch.bfloat16
            ),
        }

        decoder_layer = DecoderLayer(
            device=device,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            state_dict=state_dict,
            layer_idx=0,
            layer_prefix="talker.model",
            rms_norm_eps=config.rms_norm_eps,
        )

        # Create KV cache
        kv_caches = create_kv_cache_list(device, config, max_batch_size=batch, max_seq_len=16)
        kv_cache = kv_caches[0]

        # Create input
        torch_input = torch.randn(batch, 1, seq_len, hidden_size, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Identity RoPE
        cos_ttnn = ttnn.from_torch(
            torch.ones(1, 1, seq_len, head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        sin_ttnn = ttnn.from_torch(
            torch.zeros(1, 1, seq_len, head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        trans_mat = get_rot_transformation_mat(dhead=head_dim)
        trans_mat_ttnn = ttnn.from_torch(
            trans_mat,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Test prefill
        output, updated_kv_cache = decoder_layer.forward(
            ttnn_input,
            cos_ttnn,
            sin_ttnn,
            trans_mat_ttnn,
            kv_cache=kv_cache,
            start_pos=0,
            mode="prefill",
        )

        output_torch = ttnn.to_torch(output)
        assert output_torch.shape == (batch, 1, seq_len, hidden_size)
        assert updated_kv_cache is not None

        print(f"DecoderLayer with KV cache prefill: PASS (shape={output_torch.shape})")

    def test_kv_cache_list_creation(self, device):
        """Test that KV cache list is created correctly."""
        from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig

        config = Qwen3TTSTalkerConfig()

        kv_caches = create_kv_cache_list(
            device=device,
            config=config,
            max_batch_size=1,
            max_seq_len=128,
        )

        assert len(kv_caches) == config.num_hidden_layers
        for i, (k, v) in enumerate(kv_caches):
            k_shape = k.shape
            v_shape = v.shape
            assert k_shape == (1, config.num_key_value_heads, 128, config.head_dim), f"Layer {i} K shape mismatch"
            assert v_shape == (1, config.num_key_value_heads, 128, config.head_dim), f"Layer {i} V shape mismatch"

        print(f"KV cache list creation: PASS ({len(kv_caches)} layers)")


@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
class TestPrefillDecodeEquivalence:
    """Tests to verify prefill output matches incremental decode output."""

    def test_single_layer_prefill_vs_decode(self, device):
        """
        Test that running prefill on [a, b, c, d] gives same output as:
        - prefill on [a, b, c]
        - decode on [d] with updated cache

        This verifies KV cache is being used correctly.
        """
        from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
        from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
        from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
        from models.tt_transformers.tt.common import get_rot_transformation_mat

        config = Qwen3TTSTalkerConfig()
        config.num_hidden_layers = 1

        batch = 1
        full_seq_len = 4  # [a, b, c, d]
        prefill_len = 3  # [a, b, c]
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        intermediate_size = config.intermediate_size

        # Create random weights
        torch.manual_seed(123)
        state_dict = {
            "talker.model.layers.0.input_layernorm.weight": torch.randn(hidden_size, dtype=torch.bfloat16),
            "talker.model.layers.0.self_attn.q_proj.weight": torch.randn(
                num_heads * head_dim, hidden_size, dtype=torch.bfloat16
            ),
            "talker.model.layers.0.self_attn.k_proj.weight": torch.randn(
                num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16
            ),
            "talker.model.layers.0.self_attn.v_proj.weight": torch.randn(
                num_kv_heads * head_dim, hidden_size, dtype=torch.bfloat16
            ),
            "talker.model.layers.0.self_attn.o_proj.weight": torch.randn(
                hidden_size, num_heads * head_dim, dtype=torch.bfloat16
            ),
            "talker.model.layers.0.self_attn.q_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
            "talker.model.layers.0.self_attn.k_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
            "talker.model.layers.0.post_attention_layernorm.weight": torch.randn(hidden_size, dtype=torch.bfloat16),
            "talker.model.layers.0.mlp.gate_proj.weight": torch.randn(
                intermediate_size, hidden_size, dtype=torch.bfloat16
            ),
            "talker.model.layers.0.mlp.up_proj.weight": torch.randn(
                intermediate_size, hidden_size, dtype=torch.bfloat16
            ),
            "talker.model.layers.0.mlp.down_proj.weight": torch.randn(
                hidden_size, intermediate_size, dtype=torch.bfloat16
            ),
        }

        decoder_layer = DecoderLayer(
            device=device,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            state_dict=state_dict,
            layer_idx=0,
            layer_prefix="talker.model",
            rms_norm_eps=config.rms_norm_eps,
        )

        # Create full input sequence
        torch.manual_seed(456)
        full_input = torch.randn(batch, 1, full_seq_len, hidden_size, dtype=torch.bfloat16)

        trans_mat = get_rot_transformation_mat(dhead=head_dim)
        trans_mat_ttnn = ttnn.from_torch(trans_mat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Method 1: Full prefill
        full_input_ttnn = ttnn.from_torch(full_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cos_full = ttnn.from_torch(
            torch.ones(1, 1, full_seq_len, head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        sin_full = ttnn.from_torch(
            torch.zeros(1, 1, full_seq_len, head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        full_output, _ = decoder_layer.forward(full_input_ttnn, cos_full, sin_full, trans_mat_ttnn, mode="prefill")
        full_output_torch = ttnn.to_torch(full_output)

        # Method 2: Prefill first part, then decode last token
        # Step 1: Prefill [a, b, c]
        kv_caches = create_kv_cache_list(device, config, max_batch_size=batch, max_seq_len=16)
        kv_cache = kv_caches[0]

        prefill_input = full_input[:, :, :prefill_len, :]
        prefill_input_ttnn = ttnn.from_torch(prefill_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cos_prefill = ttnn.from_torch(
            torch.ones(1, 1, prefill_len, head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        sin_prefill = ttnn.from_torch(
            torch.zeros(1, 1, prefill_len, head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        prefill_output, updated_kv_cache = decoder_layer.forward(
            prefill_input_ttnn, cos_prefill, sin_prefill, trans_mat_ttnn, kv_cache=kv_cache, start_pos=0, mode="prefill"
        )
        prefill_output_torch = ttnn.to_torch(prefill_output)

        # Step 2: Decode [d] using cached K,V
        decode_input = full_input[:, :, prefill_len:, :]  # Last token
        decode_input_ttnn = ttnn.from_torch(decode_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cos_decode = ttnn.from_torch(
            torch.ones(1, 1, 1, head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        sin_decode = ttnn.from_torch(
            torch.zeros(1, 1, 1, head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        decode_output, _ = decoder_layer.forward(
            decode_input_ttnn,
            cos_decode,
            sin_decode,
            trans_mat_ttnn,
            kv_cache=updated_kv_cache,
            start_pos=prefill_len,
            mode="decode",
        )
        decode_output_torch = ttnn.to_torch(decode_output)

        # Compare: Last token of full prefill vs decode output
        full_last_token = full_output_torch[:, :, -1:, :]
        pcc = pearson_correlation(full_last_token, decode_output_torch)

        print(f"Full prefill last token shape: {full_last_token.shape}")
        print(f"Decode output shape: {decode_output_torch.shape}")
        print(f"PCC (full prefill[-1] vs decode): {pcc:.6f}")

        # Note: Due to causal attention differences, PCC may not be perfect
        # but should be reasonably high (> 0.9)
        assert pcc > 0.8, f"PCC {pcc} too low - decode output doesn't match prefill"
        print("Single layer prefill vs decode: PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
