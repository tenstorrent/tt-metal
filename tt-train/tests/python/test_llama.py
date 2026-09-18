# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import numpy as np
import pytest

import ttnn
import ttml
from ttml.common.utils import build_causal_mask
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig, LlamaRopeScalingConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tiny_config():
    """Create a tiny Llama config for testing.

    Note: All dimensions must be tile-aligned (multiples of 32) for TTNN operations.
    """
    return LlamaConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        max_position_embeddings=64,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        runner_type=RunnerType.Default,
        weight_tying=WeightTyingType.Disabled,
    )


def create_causal_mask(seq_len: int) -> ttml.autograd.Tensor:
    """Create a causal attention mask as a tensor using common utility."""
    mask_np = build_causal_mask(seq_len)
    return ttml.autograd.Tensor.from_numpy(mask_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)


# =============================================================================
# LlamaConfig.from_hf
# =============================================================================

# Llama-3.2-1B-Instruct's config.json, minus fields the model has no use for.
HF_LLAMA_3_2_1B = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "attention_bias": False,
    "head_dim": 64,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "num_attention_heads": 32,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": True,
    "vocab_size": 128256,
}


class TestLlamaConfigFromHf:
    def test_reads_the_architecture(self):
        config = LlamaConfig.from_hf(HF_LLAMA_3_2_1B, max_position_embeddings=2048)
        assert (config.hidden_size, config.intermediate_size, config.num_hidden_layers) == (2048, 8192, 16)
        assert (config.num_attention_heads, config.num_key_value_heads, config.vocab_size) == (32, 8, 128256)
        assert config.rope_theta == 500000.0
        assert config.rope_scaling == LlamaRopeScalingConfig(32.0, 4.0, 1.0, 8192)
        assert config.weight_tying == WeightTyingType.Enabled
        assert config.attention_bias is False

    def test_sequence_length_is_the_callers_not_the_trained_context(self):
        config = LlamaConfig.from_hf(HF_LLAMA_3_2_1B, max_position_embeddings=2048)
        assert config.max_position_embeddings == 2048

    def test_overrides_win(self):
        config = LlamaConfig.from_hf(
            HF_LLAMA_3_2_1B, max_position_embeddings=256, runner_type=RunnerType.MemoryEfficient, mlp_dropout=0.1
        )
        assert config.runner_type == RunnerType.MemoryEfficient
        assert config.mlp_dropout == 0.1

    def test_reads_config_json_from_a_checkpoint_directory(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(HF_LLAMA_3_2_1B))
        from_dir = LlamaConfig.from_hf(tmp_path, max_position_embeddings=256)
        assert from_dir == LlamaConfig.from_hf(HF_LLAMA_3_2_1B, max_position_embeddings=256)

    def test_key_value_heads_default_to_the_attention_heads(self):
        hf = {k: v for k, v in HF_LLAMA_3_2_1B.items() if k != "num_key_value_heads"}
        assert LlamaConfig.from_hf(hf, max_position_embeddings=256).num_key_value_heads == 32

    def test_no_rope_scaling_means_none(self):
        hf = {k: v for k, v in HF_LLAMA_3_2_1B.items() if k != "rope_scaling"}
        assert LlamaConfig.from_hf(hf, max_position_embeddings=256).rope_scaling == LlamaRopeScalingConfig()

    @pytest.mark.parametrize(
        "patch,message",
        [
            ({"model_type": "qwen3"}, "not a Llama checkpoint"),
            ({"mlp_bias": True}, "mlp_bias"),
            ({"head_dim": 128}, "head_dim"),
            ({"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, "rope_scaling type 'yarn'"),
        ],
        ids=["model_type", "mlp_bias", "head_dim", "rope_type"],
    )
    def test_rejects_what_the_model_cannot_express(self, patch, message, expect_error):
        with expect_error(ValueError, message):
            LlamaConfig.from_hf({**HF_LLAMA_3_2_1B, **patch}, max_position_embeddings=256)


# =============================================================================
# Llama Model Tests
# =============================================================================


class TestLlama:
    """Tests for Llama model."""

    def test_model_creation(self, tiny_config):
        """Test that Llama can be created."""
        model = Llama(tiny_config)

        assert model is not None
        assert isinstance(model, Llama)
        assert model.config == tiny_config

    def test_model_forward_shape(self, tiny_config):
        """Test that Llama forward pass produces correct output shape."""
        model = Llama(tiny_config)
        model.eval()

        batch_size = 2
        seq_len = tiny_config.max_position_embeddings

        # Create input tokens
        tokens = np.random.randint(0, tiny_config.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        # Create mask
        mask = create_causal_mask(seq_len)

        # Forward pass
        logits = model(input_tensor, mask)
        logits_shape = logits.shape()

        # Output should be [batch_size, 1, seq_len, vocab_size]
        assert len(logits_shape) == 4
        assert logits_shape[0] == batch_size
        assert logits_shape[2] == seq_len
        assert logits_shape[3] == tiny_config.vocab_size

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_model_backward(self, tiny_config):
        """Test that Llama backward pass computes gradients."""
        model = Llama(tiny_config)
        model.train()

        batch_size = 2
        seq_len = tiny_config.max_position_embeddings

        # Create input
        tokens = np.random.randint(0, tiny_config.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        mask = create_causal_mask(seq_len)

        # Forward and backward
        logits = model(input_tensor, mask)
        loss = ttml.ops.unary.mean(logits)
        loss.backward(False)

        # Check that some gradients are computed
        params = model.parameters()
        grads_initialized = sum(1 for p in params.values() if p.is_grad_initialized())

        assert grads_initialized > 0, "At least some parameters should have gradients"

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_model_callable(self, tiny_config):
        """Test that model is callable via __call__."""
        model = Llama(tiny_config)
        model.eval()

        batch_size = 2
        seq_len = tiny_config.max_position_embeddings

        tokens = np.random.randint(0, tiny_config.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        mask = create_causal_mask(seq_len)

        # Call model directly (not model.forward())
        output = model(input_tensor, mask)

        assert output is not None
        assert len(output.shape()) == 4

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_weight_tying(self, tiny_config):
        """Test that weight tying shares embedding and output weights."""
        cfg = replace(tiny_config, weight_tying=WeightTyingType.Enabled)
        model = Llama(cfg)

        # With weight tying, tok_emb and fc should share the same Parameter
        # (so the backing tensor is identical too).
        assert model.tok_emb.weight is model.fc.weight, "Weight tying should share embedding and output weights"

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_rope_scaling_config(self, tiny_config):
        """Test that RoPE scaling config is accepted and produces finite outputs."""
        rope_cfg = LlamaRopeScalingConfig(
            scaling_factor=8.0,
            high_freq_factor=4.0,
            low_freq_factor=1.0,
            original_context_length=32,
        )
        cfg = replace(tiny_config, rope_scaling=rope_cfg)
        model = Llama(cfg)
        model.eval()

        batch_size = 2
        seq_len = cfg.max_position_embeddings

        tokens = np.random.randint(0, cfg.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        mask = create_causal_mask(seq_len)

        logits = model(input_tensor, mask)
        logits_np = logits.to_numpy(ttnn.DataType.FLOAT32)
        assert np.all(np.isfinite(logits_np)), "Logits should be finite with RoPE scaling"

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_explicit_intermediate_size(self, tiny_config):
        """Test that explicit intermediate_size overrides the default formula."""
        cfg = replace(tiny_config, intermediate_size=128)
        model = Llama(cfg)
        model.eval()

        batch_size = 2
        seq_len = cfg.max_position_embeddings

        tokens = np.random.randint(0, cfg.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        mask = create_causal_mask(seq_len)

        logits = model(input_tensor, mask)
        logits_shape = logits.shape()

        # Output shape should be unchanged
        assert logits_shape[0] == batch_size
        assert logits_shape[2] == seq_len
        assert logits_shape[3] == cfg.vocab_size

        ttml.autograd.AutoContext.get_instance().reset_graph()


# =============================================================================
# Integration Tests
# =============================================================================


class TestLlamaIntegration:
    """Integration tests for Llama."""

    @pytest.mark.parametrize(
        "runner_type",
        [ttml.models.RunnerType.Default, ttml.models.RunnerType.MemoryEfficient],
    )
    def test_training_step(self, tiny_config, runner_type):
        """Test a single training step with optimizer."""
        cfg = replace(tiny_config, runner_type=runner_type)
        model = Llama(cfg)
        model.train()

        batch_size = 2
        seq_len = cfg.max_position_embeddings

        # Create optimizer
        params = model.parameters()
        opt_cfg = ttml.optimizers.SGDConfig.make(0.01, 0.0, 0.0, 0.0, False)
        optimizer = ttml.optimizers.SGD(params, opt_cfg)

        # Create input and targets
        tokens = np.random.randint(0, cfg.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        targets = np.random.randint(0, cfg.vocab_size, size=(batch_size, seq_len)).astype(np.uint32)
        target_tensor = ttml.autograd.Tensor.from_numpy(
            targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        mask = create_causal_mask(seq_len)

        # Training step
        optimizer.zero_grad()
        logits = model(input_tensor, mask)
        loss = ttml.ops.loss.cross_entropy_loss(logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN)
        loss.backward(False)
        optimizer.step()

        # Verify loss is valid
        loss_np = loss.to_numpy(ttnn.DataType.FLOAT32)
        assert np.isfinite(loss_np).all(), "Loss should be finite"

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_output_values_finite(self, tiny_config):
        """Test that model outputs are finite (no NaN/Inf)."""
        model = Llama(tiny_config)
        model.eval()

        batch_size = 2
        seq_len = tiny_config.max_position_embeddings

        tokens = np.random.randint(0, tiny_config.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        mask = create_causal_mask(seq_len)

        logits = model(input_tensor, mask)
        logits_np = logits.to_numpy(ttnn.DataType.FLOAT32)

        assert np.all(np.isfinite(logits_np)), "Logits should be finite"

        ttml.autograd.AutoContext.get_instance().reset_graph()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
