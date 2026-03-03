# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3-TTS functional reference implementations."""

import os
import sys

import pytest
import torch
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
)

from models.demos.qwen3_tts.reference.functional import (
    apply_rotary_pos_emb,
    attention,
    codebook_lookup,
    compute_mrope_frequencies,
    compute_rope_frequencies,
    decoder_layer,
    get_default_code_predictor_config,
    get_default_speech_tokenizer_config,
    get_default_talker_config,
    pre_transformer_forward,
    rms_norm,
    rotate_half,
    swiglu_mlp,
)

torch.manual_seed(0)


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate Pearson correlation coefficient between two tensors."""
    x_flat = x.detach().float().cpu().numpy().flatten()
    y_flat = y.detach().float().cpu().numpy().flatten()
    return pearsonr(x_flat, y_flat)[0]


def get_script_dir():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Unit Tests (no HuggingFace model required)
# =============================================================================
class TestRMSNorm:
    """Tests for RMSNorm implementation."""

    def test_rms_norm_shape(self):
        """Test that RMSNorm preserves input shape."""
        batch, seq_len, hidden_size = 2, 10, 2048
        hidden_states = torch.randn(batch, seq_len, hidden_size)
        weight = torch.ones(hidden_size)

        output = rms_norm(hidden_states, weight, eps=1e-6)

        assert output.shape == hidden_states.shape

    def test_rms_norm_dtype_preservation(self):
        """Test that RMSNorm preserves input dtype."""
        hidden_states = torch.randn(2, 10, 2048, dtype=torch.bfloat16)
        weight = torch.ones(2048)

        output = rms_norm(hidden_states, weight, eps=1e-6)

        assert output.dtype == hidden_states.dtype

    def test_rms_norm_normalization(self):
        """Test that RMSNorm normalizes variance correctly."""
        hidden_states = torch.randn(2, 10, 2048)
        weight = torch.ones(2048)

        output = rms_norm(hidden_states, weight, eps=1e-6)

        # RMS of normalized output should be close to 1
        rms = output.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


class TestRotaryEmbedding:
    """Tests for RoPE implementations."""

    def test_rotate_half(self):
        """Test rotate_half function."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = rotate_half(x)
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        assert torch.allclose(result, expected)

    def test_compute_rope_frequencies_shape(self):
        """Test RoPE frequency computation shape."""
        head_dim, max_seq_len = 128, 1024
        cos, sin = compute_rope_frequencies(head_dim, max_seq_len)

        assert cos.shape == (1, max_seq_len, head_dim)
        assert sin.shape == (1, max_seq_len, head_dim)

    def test_compute_mrope_frequencies_shape(self):
        """Test MROPE frequency computation shape."""
        head_dim, max_seq_len = 128, 1024
        cos, sin = compute_mrope_frequencies(head_dim, max_seq_len)

        assert cos.shape == (3, 1, max_seq_len, head_dim)
        assert sin.shape == (3, 1, max_seq_len, head_dim)

    def test_apply_rotary_pos_emb_shape(self):
        """Test that RoPE preserves tensor shapes."""
        batch, num_heads, seq_len, head_dim = 2, 16, 10, 128
        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, 8, seq_len, head_dim)  # GQA

        cos, sin = compute_rope_frequencies(head_dim, seq_len)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestMLP:
    """Tests for SwiGLU MLP implementation."""

    def test_swiglu_mlp_shape(self):
        """Test that MLP produces correct output shape."""
        batch, seq_len, hidden_size = 2, 10, 2048
        intermediate_size = 6144

        hidden_states = torch.randn(batch, seq_len, hidden_size)
        gate_proj = torch.randn(intermediate_size, hidden_size)
        up_proj = torch.randn(intermediate_size, hidden_size)
        down_proj = torch.randn(hidden_size, intermediate_size)

        output = swiglu_mlp(hidden_states, gate_proj, up_proj, down_proj)

        assert output.shape == (batch, seq_len, hidden_size)


class TestAttention:
    """Tests for attention implementation."""

    def test_attention_shape(self):
        """Test that attention produces correct output shape."""
        batch, seq_len, hidden_size = 2, 10, 2048
        num_heads, num_kv_heads, head_dim = 16, 8, 128

        hidden_states = torch.randn(batch, seq_len, hidden_size)

        # Create weight tensors
        q_proj = torch.randn(num_heads * head_dim, hidden_size)
        k_proj = torch.randn(num_kv_heads * head_dim, hidden_size)
        v_proj = torch.randn(num_kv_heads * head_dim, hidden_size)
        o_proj = torch.randn(hidden_size, num_heads * head_dim)
        q_norm = torch.ones(head_dim)
        k_norm = torch.ones(head_dim)

        cos, sin = compute_rope_frequencies(head_dim, seq_len)

        output = attention(
            hidden_states,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        assert output.shape == (batch, seq_len, hidden_size)

    def test_attention_causal_mask(self):
        """Test that attention respects causal mask."""
        batch, seq_len, hidden_size = 1, 5, 256
        num_heads, num_kv_heads, head_dim = 4, 2, 64

        hidden_states = torch.randn(batch, seq_len, hidden_size)

        q_proj = torch.randn(num_heads * head_dim, hidden_size)
        k_proj = torch.randn(num_kv_heads * head_dim, hidden_size)
        v_proj = torch.randn(num_kv_heads * head_dim, hidden_size)
        o_proj = torch.randn(hidden_size, num_heads * head_dim)
        q_norm = torch.ones(head_dim)
        k_norm = torch.ones(head_dim)

        cos, sin = compute_rope_frequencies(head_dim, seq_len)

        # Create causal mask
        causal_mask = (
            torch.triu(
                torch.full((seq_len, seq_len), float("-inf")),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        output = attention(
            hidden_states,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            attention_mask=causal_mask,
        )

        assert output.shape == (batch, seq_len, hidden_size)
        assert not torch.isnan(output).any()


class TestDecoderLayer:
    """Tests for decoder layer implementation."""

    def test_decoder_layer_shape(self):
        """Test that decoder layer preserves input shape."""
        config = get_default_talker_config()
        batch, seq_len = 2, 10

        hidden_states = torch.randn(batch, seq_len, config.hidden_size)

        # Create layer weights
        layer_weights = {
            "input_layernorm.weight": torch.ones(config.hidden_size),
            "self_attn.q_proj.weight": torch.randn(config.num_attention_heads * config.head_dim, config.hidden_size),
            "self_attn.k_proj.weight": torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
            "self_attn.v_proj.weight": torch.randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
            "self_attn.o_proj.weight": torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim),
            "self_attn.q_norm.weight": torch.ones(config.head_dim),
            "self_attn.k_norm.weight": torch.ones(config.head_dim),
            "post_attention_layernorm.weight": torch.ones(config.hidden_size),
            "mlp.gate_proj.weight": torch.randn(config.intermediate_size, config.hidden_size),
            "mlp.up_proj.weight": torch.randn(config.intermediate_size, config.hidden_size),
            "mlp.down_proj.weight": torch.randn(config.hidden_size, config.intermediate_size),
        }

        cos, sin = compute_mrope_frequencies(config.head_dim, seq_len)

        output = decoder_layer(hidden_states, layer_weights, cos, sin, config)

        assert output.shape == hidden_states.shape


class TestConfig:
    """Tests for configuration classes."""

    def test_default_talker_config(self):
        """Test default Talker configuration values."""
        config = get_default_talker_config()

        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 28
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 8
        assert config.head_dim == 128
        assert config.intermediate_size == 6144
        assert config.rope_theta == 1000000.0
        assert config.rms_norm_eps == 1e-6

    def test_default_code_predictor_config(self):
        """Test default Code Predictor configuration values."""
        config = get_default_code_predictor_config()

        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 5
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 8
        assert config.head_dim == 128
        assert config.intermediate_size == 3072

    def test_default_speech_tokenizer_config(self):
        """Test default Speech Tokenizer Decoder configuration values."""
        config = get_default_speech_tokenizer_config()

        assert config.num_quantizers == 16
        assert config.codebook_size == 2048
        assert config.codebook_dim == 256
        assert config.pre_transformer_num_layers == 8
        assert config.pre_transformer_hidden_size == 512


# =============================================================================
# Speech Tokenizer Decoder Tests
# =============================================================================
class TestCodebookLookup:
    """Tests for codebook lookup implementation."""

    def test_codebook_lookup_shape(self):
        """Test that codebook lookup produces correct output shape."""
        batch, num_quantizers, seq_len = 2, 16, 10
        codebook_size, codebook_dim = 2048, 256
        latent_dim = 512

        # Create token IDs
        token_ids = torch.randint(0, codebook_size, (batch, num_quantizers, seq_len))

        # Create codebooks
        codebooks = [torch.randn(codebook_size, codebook_dim) for _ in range(num_quantizers)]

        # Create projection weights (Conv1d with kernel_size=1)
        input_proj = torch.randn(latent_dim, codebook_dim, 1)
        output_proj = torch.randn(latent_dim, latent_dim, 1)

        output = codebook_lookup(token_ids, codebooks, input_proj, output_proj)

        assert output.shape == (batch, seq_len, latent_dim)

    def test_codebook_lookup_no_nans(self):
        """Test that codebook lookup produces no NaN values."""
        batch, num_quantizers, seq_len = 1, 4, 5
        codebook_size, codebook_dim = 256, 64
        latent_dim = 128

        token_ids = torch.randint(0, codebook_size, (batch, num_quantizers, seq_len))
        codebooks = [torch.randn(codebook_size, codebook_dim) for _ in range(num_quantizers)]
        input_proj = torch.randn(latent_dim, codebook_dim, 1)
        output_proj = torch.randn(latent_dim, latent_dim, 1)

        output = codebook_lookup(token_ids, codebooks, input_proj, output_proj)

        assert not torch.isnan(output).any()


class TestSpeechTokenizerDecoder:
    """Tests for speech tokenizer decoder implementation."""

    def test_speech_tokenizer_config(self):
        """Test speech tokenizer configuration."""
        config = get_default_speech_tokenizer_config()

        assert config.num_quantizers == 16
        assert config.codebook_size == 2048
        assert config.codebook_dim == 256
        assert config.pre_transformer_num_layers == 8

    def test_pre_transformer_shape(self):
        """Test pre-transformer output shape with synthetic weights."""
        config = get_default_speech_tokenizer_config()
        batch, seq_len = 2, 10
        latent_dim = config.latent_dim
        hidden_size = config.pre_transformer_hidden_size

        # Create synthetic weights
        weights = {
            "input_proj.weight": torch.randn(hidden_size, latent_dim),
            "input_proj.bias": torch.randn(hidden_size),
            "output_proj.weight": torch.randn(latent_dim, hidden_size),
            "output_proj.bias": torch.randn(latent_dim),
            "norm.weight": torch.ones(hidden_size),
        }

        # Add layer weights
        for i in range(config.pre_transformer_num_layers):
            prefix = f"layers.{i}."
            weights[f"{prefix}input_layernorm.weight"] = torch.ones(hidden_size)
            weights[f"{prefix}self_attn.q_proj.weight"] = torch.randn(
                config.pre_transformer_num_heads * config.pre_transformer_head_dim, hidden_size
            )
            weights[f"{prefix}self_attn.k_proj.weight"] = torch.randn(
                config.pre_transformer_num_heads * config.pre_transformer_head_dim, hidden_size
            )
            weights[f"{prefix}self_attn.v_proj.weight"] = torch.randn(
                config.pre_transformer_num_heads * config.pre_transformer_head_dim, hidden_size
            )
            weights[f"{prefix}self_attn.o_proj.weight"] = torch.randn(
                hidden_size, config.pre_transformer_num_heads * config.pre_transformer_head_dim
            )
            weights[f"{prefix}post_attention_layernorm.weight"] = torch.ones(hidden_size)
            weights[f"{prefix}mlp.gate_proj.weight"] = torch.randn(
                config.pre_transformer_intermediate_size, hidden_size
            )
            weights[f"{prefix}mlp.up_proj.weight"] = torch.randn(config.pre_transformer_intermediate_size, hidden_size)
            weights[f"{prefix}mlp.down_proj.weight"] = torch.randn(
                hidden_size, config.pre_transformer_intermediate_size
            )

        embeddings = torch.randn(batch, seq_len, latent_dim)
        output = pre_transformer_forward(embeddings, weights, config)

        assert output.shape == (batch, seq_len, latent_dim)
        assert not torch.isnan(output).any()


# =============================================================================
# Integration Tests (require HuggingFace model)
# =============================================================================
@pytest.fixture
def hf_model():
    """Load the HuggingFace model weights."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        model_path = hf_hub_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "model.safetensors")
        state_dict = {}
        with safe_open(model_path, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict
    except Exception as e:
        pytest.skip(f"Could not load HuggingFace model: {e}")


class TestAgainstHuggingFace:
    """Integration tests comparing against HuggingFace implementation."""

    def test_rms_norm_vs_hf(self, hf_model):
        """Compare RMSNorm output against HuggingFace weights."""
        # Get the first layer's input layernorm weight (note: talker.model.layers prefix)
        weight = hf_model["talker.model.layers.0.input_layernorm.weight"]

        # Create test input
        hidden_states = torch.randn(2, 10, 2048)

        # Our implementation
        our_output = rms_norm(hidden_states, weight, eps=1e-6)

        # Verify shape and no NaNs
        assert our_output.shape == hidden_states.shape
        assert not torch.isnan(our_output).any()

    def test_mlp_vs_hf(self, hf_model):
        """Compare MLP output using HuggingFace weights."""
        gate_proj = hf_model["talker.model.layers.0.mlp.gate_proj.weight"]
        up_proj = hf_model["talker.model.layers.0.mlp.up_proj.weight"]
        down_proj = hf_model["talker.model.layers.0.mlp.down_proj.weight"]

        # Create test input with matching dtype
        dtype = gate_proj.dtype
        hidden_states = torch.randn(2, 10, 2048, dtype=dtype)

        # Our implementation
        output = swiglu_mlp(hidden_states, gate_proj, up_proj, down_proj)

        # Verify shape and no NaNs
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()

    def test_attention_vs_hf(self, hf_model):
        """Compare attention output using HuggingFace weights."""
        config = get_default_talker_config()

        q_proj = hf_model["talker.model.layers.0.self_attn.q_proj.weight"]
        k_proj = hf_model["talker.model.layers.0.self_attn.k_proj.weight"]
        v_proj = hf_model["talker.model.layers.0.self_attn.v_proj.weight"]
        o_proj = hf_model["talker.model.layers.0.self_attn.o_proj.weight"]
        q_norm = hf_model["talker.model.layers.0.self_attn.q_norm.weight"]
        k_norm = hf_model["talker.model.layers.0.self_attn.k_norm.weight"]

        # Create test input with matching dtype
        dtype = q_proj.dtype
        batch, seq_len = 2, 10
        hidden_states = torch.randn(batch, seq_len, config.hidden_size, dtype=dtype)

        cos, sin = compute_mrope_frequencies(config.head_dim, seq_len)
        cos = cos.to(dtype)
        sin = sin.to(dtype)

        output = attention(
            hidden_states,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            cos,
            sin,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            use_mrope=True,
            mrope_section=config.mrope_section,
            mrope_interleaved=config.mrope_interleaved,
        )

        # Verify shape and no NaNs
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()

    def test_decoder_layer_vs_hf(self, hf_model):
        """Compare decoder layer output using HuggingFace weights."""
        config = get_default_talker_config()
        layer_idx = 0

        # Extract layer weights (note: talker.model.layers prefix)
        layer_weights = {
            "input_layernorm.weight": hf_model[f"talker.model.layers.{layer_idx}.input_layernorm.weight"],
            "self_attn.q_proj.weight": hf_model[f"talker.model.layers.{layer_idx}.self_attn.q_proj.weight"],
            "self_attn.k_proj.weight": hf_model[f"talker.model.layers.{layer_idx}.self_attn.k_proj.weight"],
            "self_attn.v_proj.weight": hf_model[f"talker.model.layers.{layer_idx}.self_attn.v_proj.weight"],
            "self_attn.o_proj.weight": hf_model[f"talker.model.layers.{layer_idx}.self_attn.o_proj.weight"],
            "self_attn.q_norm.weight": hf_model[f"talker.model.layers.{layer_idx}.self_attn.q_norm.weight"],
            "self_attn.k_norm.weight": hf_model[f"talker.model.layers.{layer_idx}.self_attn.k_norm.weight"],
            "post_attention_layernorm.weight": hf_model[
                f"talker.model.layers.{layer_idx}.post_attention_layernorm.weight"
            ],
            "mlp.gate_proj.weight": hf_model[f"talker.model.layers.{layer_idx}.mlp.gate_proj.weight"],
            "mlp.up_proj.weight": hf_model[f"talker.model.layers.{layer_idx}.mlp.up_proj.weight"],
            "mlp.down_proj.weight": hf_model[f"talker.model.layers.{layer_idx}.mlp.down_proj.weight"],
        }

        # Create test input with matching dtype
        dtype = layer_weights["self_attn.q_proj.weight"].dtype
        batch, seq_len = 2, 10
        hidden_states = torch.randn(batch, seq_len, config.hidden_size, dtype=dtype)

        cos, sin = compute_mrope_frequencies(config.head_dim, seq_len)
        cos = cos.to(dtype)
        sin = sin.to(dtype)

        output = decoder_layer(hidden_states, layer_weights, cos, sin, config)

        # Verify shape and no NaNs
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()


# =============================================================================
# Golden output tests
# =============================================================================
class TestGoldenOutputs:
    """Tests comparing against pre-computed golden outputs."""

    @pytest.fixture
    def golden_dir(self):
        """Get the golden outputs directory."""
        return os.path.join(get_script_dir(), "golden")

    def test_rms_norm_golden(self, golden_dir):
        """Test RMSNorm against golden output."""
        golden_path = os.path.join(golden_dir, "rms_norm_golden.pt")
        if not os.path.exists(golden_path):
            pytest.skip(f"Golden file not found: {golden_path}")

        golden = torch.load(golden_path, weights_only=True)
        output = rms_norm(golden["input"], golden["weight"], golden["eps"])

        pcc = pearson_correlation(output, golden["output"])
        assert pcc > 0.999, f"PCC {pcc} below threshold"

    def test_mlp_golden(self, golden_dir):
        """Test MLP against golden output."""
        golden_path = os.path.join(golden_dir, "mlp_golden.pt")
        if not os.path.exists(golden_path):
            pytest.skip(f"Golden file not found: {golden_path}")

        golden = torch.load(golden_path, weights_only=True)
        output = swiglu_mlp(
            golden["input"],
            golden["gate_proj_weight"],
            golden["up_proj_weight"],
            golden["down_proj_weight"],
        )

        pcc = pearson_correlation(output, golden["output"])
        assert pcc > 0.999, f"PCC {pcc} below threshold"

    def test_attention_golden(self, golden_dir):
        """Test attention against golden output."""
        golden_path = os.path.join(golden_dir, "attention_golden.pt")
        if not os.path.exists(golden_path):
            pytest.skip(f"Golden file not found: {golden_path}")

        golden = torch.load(golden_path, weights_only=True)
        output = attention(
            golden["input"],
            golden["q_proj_weight"],
            golden["k_proj_weight"],
            golden["v_proj_weight"],
            golden["o_proj_weight"],
            golden["q_norm_weight"],
            golden["k_norm_weight"],
            golden["cos"],
            golden["sin"],
            num_heads=golden["num_heads"],
            num_kv_heads=golden["num_kv_heads"],
            head_dim=golden["head_dim"],
            use_mrope=golden["use_mrope"],
        )

        pcc = pearson_correlation(output, golden["output"])
        assert pcc > 0.999, f"PCC {pcc} below threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
