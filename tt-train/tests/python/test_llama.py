# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

import ttnn
import ttml
from ttml.common.data import build_causal_mask
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
    return ttml.autograd.Tensor.from_numpy(
        mask_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )


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
        tokens = np.random.randint(
            0, tiny_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
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
        tokens = np.random.randint(
            0, tiny_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
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

        tokens = np.random.randint(
            0, tiny_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
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

        # With weight tying, tok_emb.weight should be the same as fc weight
        emb_weight = model.tok_emb.weight
        fc_weight = model.fc.get_weight()
        assert (
            emb_weight is fc_weight
        ), "Weight tying should share embedding and output weights"

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

        tokens = np.random.randint(
            0, cfg.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        mask = create_causal_mask(seq_len)

        logits = model(input_tensor, mask)
        logits_np = logits.to_numpy(ttnn.DataType.FLOAT32)
        assert np.all(
            np.isfinite(logits_np)
        ), "Logits should be finite with RoPE scaling"

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_explicit_intermediate_size(self, tiny_config):
        """Test that explicit intermediate_size overrides the default formula."""
        cfg = replace(tiny_config, intermediate_size=128)
        model = Llama(cfg)
        model.eval()

        batch_size = 2
        seq_len = cfg.max_position_embeddings

        tokens = np.random.randint(
            0, cfg.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
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
        tokens = np.random.randint(
            0, cfg.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        targets = np.random.randint(
            0, cfg.vocab_size, size=(batch_size, seq_len)
        ).astype(np.uint32)
        target_tensor = ttml.autograd.Tensor.from_numpy(
            targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        mask = create_causal_mask(seq_len)

        # Training step
        optimizer.zero_grad()
        logits = model(input_tensor, mask)
        loss = ttml.ops.loss.cross_entropy_loss(
            logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN
        )
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

        tokens = np.random.randint(
            0, tiny_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
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
