# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for NanoGPT Python implementation.

This test suite verifies that the Python NanoGPT modules work correctly:
- Embedding layer
- GPTMLP (feed-forward layer)
- MultiHeadAttention
- GPTBlock (transformer block)
- NanoGPT (full model)
"""

import numpy as np
import pytest
import ml_dtypes

import ttnn
import ttml
from ttml.common.data import build_causal_mask
from ttml.models.nanogpt import (
    NanoGPT,
    NanoGPTConfig,
    create_nanogpt,
    Embedding,
    GPTBlock,
)
from ttml.models.nanogpt.gpt_mlp import GPTMLP
from ttml.models.nanogpt.multi_head_attention import MultiHeadAttention
from ttml.modules import Parameter, RunMode


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_config():
    """Create a small NanoGPT config for testing.

    Note: All dimensions must be tile-aligned (multiples of 32) for TTNN operations.
    """
    return NanoGPTConfig(
        vocab_size=64,  # Small vocab for fast tests (tile-aligned)
        block_size=64,  # Sequence length (tile-aligned)
        n_embd=64,  # Embedding dim (tile-aligned, divisible by n_head)
        n_layer=2,  # Few layers
        n_head=2,  # Few heads (64/2=32 head_dim, tile-aligned)
        dropout=0.0,  # Disable dropout for deterministic tests
        bias=True,
    )


@pytest.fixture
def tiny_config():
    """Create a tiny NanoGPT config for unit tests.

    Note: All dimensions must be tile-aligned (multiples of 32) for TTNN operations.
    """
    return NanoGPTConfig(
        vocab_size=64,  # Tile-aligned
        block_size=64,  # Tile-aligned
        n_embd=64,  # Tile-aligned
        n_layer=1,
        n_head=2,  # 64/2=32 head_dim, tile-aligned
        dropout=0.0,
        bias=True,
    )


def create_causal_mask(seq_len: int) -> ttml.autograd.Tensor:
    """Create a causal attention mask as a tensor using common utility."""
    mask_np = build_causal_mask(seq_len)
    return ttml.autograd.Tensor.from_numpy(
        mask_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )


# =============================================================================
# Embedding Tests
# =============================================================================


class TestEmbedding:
    """Tests for Embedding layer."""

    def test_embedding_creation(self):
        """Test that Embedding layer can be created."""
        num_embeddings = 100
        embedding_dim = 64

        embedding = Embedding(num_embeddings, embedding_dim)

        assert embedding is not None
        assert hasattr(embedding, "weight")
        assert isinstance(embedding.weight, Parameter)

    def test_embedding_weight_shape(self):
        """Test that embedding weight has correct shape."""
        num_embeddings = 100
        embedding_dim = 64

        embedding = Embedding(num_embeddings, embedding_dim)
        weight_np = embedding.weight.tensor.to_numpy(ttnn.DataType.FLOAT32)

        assert weight_np.shape == (1, 1, num_embeddings, embedding_dim)

    def test_embedding_forward_shape(self):
        """Test that embedding forward pass produces correct shape."""
        num_embeddings = 128  # Tile-aligned
        embedding_dim = 64  # Tile-aligned
        batch_size = 4
        seq_len = 32  # Tile-aligned

        embedding = Embedding(num_embeddings, embedding_dim)

        # Create input token indices
        indices = np.random.randint(
            0, num_embeddings, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            indices, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        # Forward pass
        output = embedding(input_tensor)
        output_shape = output.shape()

        # Output should be [batch_size, 1, seq_len, embedding_dim] (4D)
        assert len(output_shape) == 4
        assert output_shape[0] == batch_size
        assert output_shape[1] == 1
        assert output_shape[2] == seq_len
        assert output_shape[3] == embedding_dim

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_embedding_backward(self):
        """Test that embedding backward pass computes gradients."""
        num_embeddings = 64  # Tile-aligned
        embedding_dim = 64  # Tile-aligned
        batch_size = 2
        seq_len = 32  # Tile-aligned (must be divisible by tile width)

        embedding = Embedding(num_embeddings, embedding_dim)

        # Create input
        indices = np.random.randint(
            0, num_embeddings, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            indices, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        # Forward and backward
        output = embedding(input_tensor)
        loss = ttml.ops.unary.mean(output)
        loss.backward(False)

        # Check gradient exists on weight
        assert (
            embedding.weight.tensor.is_grad_initialized()
        ), "Embedding weight should have gradient"

        ttml.autograd.AutoContext.get_instance().reset_graph()


# =============================================================================
# GPTMLP Tests
# =============================================================================


class TestGPTMLP:
    """Tests for GPTMLP (feed-forward) layer."""

    def test_mlp_creation(self):
        """Test that GPTMLP can be created."""
        embedding_dim = 64

        mlp = GPTMLP(embedding_dim, dropout=0.0)

        assert mlp is not None
        assert hasattr(mlp, "fc1")
        assert hasattr(mlp, "fc2")
        assert isinstance(mlp.fc1, Parameter)
        assert isinstance(mlp.fc2, Parameter)

    def test_mlp_weight_shapes(self):
        """Test that MLP weights have correct shapes."""
        embedding_dim = 64

        mlp = GPTMLP(embedding_dim, dropout=0.0)

        fc1_np = mlp.fc1.tensor.to_numpy(ttnn.DataType.FLOAT32)
        fc2_np = mlp.fc2.tensor.to_numpy(ttnn.DataType.FLOAT32)

        # fc1: embedding_dim -> embedding_dim * 4
        assert fc1_np.shape == (1, 1, embedding_dim * 4, embedding_dim)
        # fc2: embedding_dim * 4 -> embedding_dim
        assert fc2_np.shape == (1, 1, embedding_dim, embedding_dim * 4)

    def test_mlp_forward_shape(self):
        """Test that MLP forward pass preserves shape."""
        embedding_dim = 64  # Tile-aligned
        batch_size = 4
        seq_len = 32  # Tile-aligned

        mlp = GPTMLP(embedding_dim, dropout=0.0)
        mlp.eval()  # Disable dropout

        # Create input: [batch_size, 1, seq_len, embedding_dim] (4D matching embedding output)
        input_data = np.random.randn(batch_size, 1, seq_len, embedding_dim).astype(
            ml_dtypes.bfloat16
        )
        input_tensor = ttml.autograd.Tensor.from_numpy(
            input_data, layout=ttnn.Layout.TILE
        )

        # Forward pass
        output = mlp(input_tensor)
        output_shape = output.shape()

        # Output should preserve input shape (4D)
        assert output_shape == [batch_size, 1, seq_len, embedding_dim]

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_mlp_backward(self):
        """Test that MLP backward pass computes gradients."""
        embedding_dim = 64  # Tile-aligned
        batch_size = 2
        seq_len = 32  # Tile-aligned

        mlp = GPTMLP(embedding_dim, dropout=0.0)
        mlp.eval()

        # Create input (4D matching embedding output)
        input_data = np.random.randn(batch_size, 1, seq_len, embedding_dim).astype(
            ml_dtypes.bfloat16
        )
        input_tensor = ttml.autograd.Tensor.from_numpy(
            input_data, layout=ttnn.Layout.TILE
        )

        # Forward and backward
        output = mlp(input_tensor)
        loss = ttml.ops.unary.mean(output)
        loss.backward(False)

        # Check gradients exist
        assert mlp.fc1.tensor.is_grad_initialized(), "fc1 should have gradient"
        assert mlp.fc2.tensor.is_grad_initialized(), "fc2 should have gradient"

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_mlp_train_eval_mode(self):
        """Test that MLP train/eval mode switching works."""
        mlp = GPTMLP(64, dropout=0.1)

        # Default is train mode
        assert mlp.get_run_mode() == RunMode.TRAIN

        mlp.eval()
        assert mlp.get_run_mode() == RunMode.EVAL

        mlp.train()
        assert mlp.get_run_mode() == RunMode.TRAIN


# =============================================================================
# MultiHeadAttention Tests
# =============================================================================


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention layer."""

    def test_attention_creation(self):
        """Test that MultiHeadAttention can be created."""
        embedding_dim = 64
        num_heads = 4

        attention = MultiHeadAttention(embedding_dim, num_heads, dropout=0.0)

        assert attention is not None
        assert hasattr(attention, "qkv")
        assert hasattr(attention, "out_proj")
        assert attention.num_heads == num_heads
        assert attention.head_dim == embedding_dim // num_heads

    def test_attention_weight_shapes(self):
        """Test that attention weights have correct shapes."""
        embedding_dim = 64
        num_heads = 4

        attention = MultiHeadAttention(embedding_dim, num_heads, dropout=0.0)

        qkv_np = attention.qkv.get_weight_numpy()
        out_np = attention.out_proj.get_weight_numpy()

        # QKV projection: embedding_dim -> embedding_dim * 3
        assert qkv_np.shape == (1, 1, embedding_dim * 3, embedding_dim)
        # Output projection: embedding_dim -> embedding_dim
        assert out_np.shape == (1, 1, embedding_dim, embedding_dim)

    def test_attention_forward_shape(self):
        """Test that attention forward pass produces correct shape."""
        embedding_dim = 64  # Tile-aligned
        num_heads = 2  # 64/2=32 head_dim, tile-aligned
        batch_size = 4
        seq_len = 32  # Tile-aligned

        attention = MultiHeadAttention(embedding_dim, num_heads, dropout=0.0)
        attention.eval()

        # Create input: [batch_size, 1, seq_len, embedding_dim] (4D matching embedding output)
        input_data = np.random.randn(batch_size, 1, seq_len, embedding_dim).astype(
            ml_dtypes.bfloat16
        )
        input_tensor = ttml.autograd.Tensor.from_numpy(
            input_data, layout=ttnn.Layout.TILE
        )

        # Create causal mask
        mask = create_causal_mask(seq_len)

        # Forward pass
        output = attention(input_tensor, mask)
        output_shape = output.shape()

        # Output should preserve input shape (4D)
        assert output_shape == [batch_size, 1, seq_len, embedding_dim]

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_attention_backward(self):
        """Test that attention backward pass computes gradients."""
        embedding_dim = 64  # Tile-aligned
        num_heads = 2  # 64/2=32 head_dim, tile-aligned
        batch_size = 2
        seq_len = 32  # Tile-aligned

        attention = MultiHeadAttention(embedding_dim, num_heads, dropout=0.0)
        attention.eval()

        # Create input (4D matching embedding output)
        input_data = np.random.randn(batch_size, 1, seq_len, embedding_dim).astype(
            ml_dtypes.bfloat16
        )
        input_tensor = ttml.autograd.Tensor.from_numpy(
            input_data, layout=ttnn.Layout.TILE
        )
        mask = create_causal_mask(seq_len)

        # Forward and backward
        output = attention(input_tensor, mask)
        loss = ttml.ops.unary.mean(output)
        loss.backward(False)

        # Check gradients exist
        assert (
            attention.qkv.get_weight().is_grad_initialized()
        ), "qkv should have gradient"
        assert (
            attention.out_proj.get_weight().is_grad_initialized()
        ), "out_proj should have gradient"

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_attention_head_dim_validation(self):
        """Test that attention validates head dimension."""
        with pytest.raises(AssertionError):
            # embedding_dim (65) not divisible by num_heads (4)
            MultiHeadAttention(65, 4, dropout=0.0)


# =============================================================================
# GPTBlock Tests
# =============================================================================


class TestGPTBlock:
    """Tests for GPTBlock (transformer block)."""

    def test_block_creation(self):
        """Test that GPTBlock can be created."""
        embedding_dim = 64
        num_heads = 4

        block = GPTBlock(embedding_dim, num_heads, dropout=0.0, bias=True)

        assert block is not None
        assert hasattr(block, "attention")
        assert hasattr(block, "mlp")
        assert hasattr(block, "ln1_gamma")
        assert hasattr(block, "ln2_gamma")

    def test_block_forward_shape(self):
        """Test that GPTBlock forward pass preserves shape."""
        embedding_dim = 64  # Tile-aligned
        num_heads = 2  # 64/2=32 head_dim, tile-aligned
        batch_size = 4
        seq_len = 32  # Tile-aligned

        block = GPTBlock(embedding_dim, num_heads, dropout=0.0, bias=True)
        block.eval()

        # Create input (4D matching embedding output)
        input_data = np.random.randn(batch_size, 1, seq_len, embedding_dim).astype(
            ml_dtypes.bfloat16
        )
        input_tensor = ttml.autograd.Tensor.from_numpy(
            input_data, layout=ttnn.Layout.TILE
        )
        mask = create_causal_mask(seq_len)

        # Forward pass
        output = block(input_tensor, mask)
        output_shape = output.shape()

        # Output should preserve input shape (4D)
        assert output_shape == [batch_size, 1, seq_len, embedding_dim]

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_block_backward(self):
        """Test that GPTBlock backward pass computes gradients."""
        embedding_dim = 64  # Tile-aligned
        num_heads = 2  # 64/2=32 head_dim, tile-aligned
        batch_size = 2
        seq_len = 32  # Tile-aligned

        block = GPTBlock(embedding_dim, num_heads, dropout=0.0, bias=True)
        block.eval()

        # Create input (4D matching embedding output)
        input_data = np.random.randn(batch_size, 1, seq_len, embedding_dim).astype(
            ml_dtypes.bfloat16
        )
        input_tensor = ttml.autograd.Tensor.from_numpy(
            input_data, layout=ttnn.Layout.TILE
        )
        mask = create_causal_mask(seq_len)

        # Forward and backward
        output = block(input_tensor, mask)
        loss = ttml.ops.unary.mean(output)
        loss.backward(False)

        # Check gradients exist on layer norms
        assert (
            block.ln1_gamma.tensor.is_grad_initialized()
        ), "ln1_gamma should have gradient"
        assert (
            block.ln2_gamma.tensor.is_grad_initialized()
        ), "ln2_gamma should have gradient"

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_block_train_eval_propagation(self):
        """Test that train/eval mode propagates to submodules."""
        block = GPTBlock(64, 4, dropout=0.1, bias=True)

        # Set to eval
        block.eval()
        assert block.get_run_mode() == RunMode.EVAL
        assert block.attention.get_run_mode() == RunMode.EVAL
        assert block.mlp.get_run_mode() == RunMode.EVAL

        # Set to train
        block.train()
        assert block.get_run_mode() == RunMode.TRAIN
        assert block.attention.get_run_mode() == RunMode.TRAIN
        assert block.mlp.get_run_mode() == RunMode.TRAIN


# =============================================================================
# NanoGPT Model Tests
# =============================================================================


class TestNanoGPT:
    """Tests for NanoGPT model."""

    def test_model_creation(self, tiny_config):
        """Test that NanoGPT can be created."""
        model = create_nanogpt(tiny_config)

        assert model is not None
        assert isinstance(model, NanoGPT)
        assert model.config == tiny_config

    def test_model_has_components(self, tiny_config):
        """Test that NanoGPT has all expected components."""
        model = create_nanogpt(tiny_config)

        # Embeddings
        assert hasattr(model, "wte")
        assert hasattr(model, "wpe")
        assert isinstance(model.wte, Embedding)
        assert isinstance(model.wpe, Embedding)

        # Blocks
        assert hasattr(model, "blocks")
        assert len(model.blocks) == tiny_config.n_layer
        for i, block in enumerate(model.blocks):
            assert isinstance(block, GPTBlock)
            assert hasattr(model, f"block_{i}")

        # Final layer norm
        assert hasattr(model, "ln_f_gamma")
        assert isinstance(model.ln_f_gamma, Parameter)

        # LM head (weight tied with wte)
        assert hasattr(model, "lm_head_weight")
        assert model.lm_head_weight is model.wte.weight

    def test_weight_tying(self, tiny_config):
        """Test that LM head weight is tied to token embedding."""
        model = create_nanogpt(tiny_config)

        # Same Parameter object
        assert model.lm_head_weight is model.wte.weight

        # Modifying one should affect the other
        weight_before = model.wte.weight.tensor.to_numpy(ttnn.DataType.FLOAT32).copy()
        lm_head_before = model.lm_head_weight.tensor.to_numpy(
            ttnn.DataType.FLOAT32
        ).copy()

        np.testing.assert_array_equal(weight_before, lm_head_before)

    def test_model_forward_shape(self, tiny_config):
        """Test that NanoGPT forward pass produces correct output shape."""
        model = create_nanogpt(tiny_config)
        model.eval()

        batch_size = 2
        seq_len = 32  # Tile-aligned

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

    def test_model_forward_without_mask(self, tiny_config):
        """Test that NanoGPT forward pass works without mask."""
        model = create_nanogpt(tiny_config)
        model.eval()

        batch_size = 2
        seq_len = 32  # Tile-aligned

        tokens = np.random.randint(
            0, tiny_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        # Forward pass without mask
        logits = model(input_tensor, mask=None)
        logits_shape = logits.shape()

        assert len(logits_shape) == 4
        assert logits_shape[0] == batch_size

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_model_parameters(self, tiny_config):
        """Test that model parameters are accessible."""
        model = create_nanogpt(tiny_config)

        params = model.parameters()

        assert isinstance(params, dict)
        assert len(params) > 0

        # Check some expected parameter names
        param_names = list(params.keys())

        # Should have token embedding weight
        wte_params = [k for k in param_names if "wte" in k.lower() or "weight" in k]
        assert len(wte_params) > 0

    def test_model_train_eval_propagation(self, tiny_config):
        """Test that train/eval mode propagates to all submodules."""
        model = create_nanogpt(tiny_config)

        # Set to eval
        model.eval()
        assert model.get_run_mode() == RunMode.EVAL
        assert model.wte.get_run_mode() == RunMode.EVAL
        assert model.wpe.get_run_mode() == RunMode.EVAL
        for block in model.blocks:
            assert block.get_run_mode() == RunMode.EVAL

        # Set to train
        model.train()
        assert model.get_run_mode() == RunMode.TRAIN
        assert model.wte.get_run_mode() == RunMode.TRAIN
        assert model.wpe.get_run_mode() == RunMode.TRAIN
        for block in model.blocks:
            assert block.get_run_mode() == RunMode.TRAIN

    def test_model_backward(self, tiny_config):
        """Test that NanoGPT backward pass computes gradients."""
        model = create_nanogpt(tiny_config)
        model.eval()  # Disable dropout for deterministic test

        batch_size = 2
        seq_len = 32  # Tile-aligned

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
        model = create_nanogpt(tiny_config)
        model.eval()

        batch_size = 2
        seq_len = 32  # Tile-aligned

        tokens = np.random.randint(
            0, tiny_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        # Call model directly (not model.forward())
        output = model(input_tensor)

        assert output is not None
        assert len(output.shape()) == 4

        ttml.autograd.AutoContext.get_instance().reset_graph()


# =============================================================================
# NanoGPTConfig Tests
# =============================================================================


class TestNanoGPTConfig:
    """Tests for NanoGPTConfig."""

    def test_default_config(self):
        """Test that default config has expected values."""
        config = NanoGPTConfig()

        assert config.vocab_size == 50304
        assert config.block_size == 1024
        assert config.n_embd == 768
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.dropout == 0.2
        assert config.bias is True

    def test_custom_config(self):
        """Test that custom config values are preserved."""
        config = NanoGPTConfig(
            vocab_size=1000,
            block_size=256,
            n_embd=128,
            n_layer=4,
            n_head=4,
            dropout=0.1,
            bias=False,
        )

        assert config.vocab_size == 1000
        assert config.block_size == 256
        assert config.n_embd == 128
        assert config.n_layer == 4
        assert config.n_head == 4
        assert config.dropout == 0.1
        assert config.bias is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestNanoGPTIntegration:
    """Integration tests for NanoGPT."""

    def test_training_step(self, tiny_config):
        """Test a single training step with optimizer."""
        model = create_nanogpt(tiny_config)
        model.train()

        batch_size = 2
        seq_len = 32  # Tile-aligned

        # Create optimizer
        params = model.parameters()
        opt_cfg = ttml.optimizers.SGDConfig.make(0.01, 0.0, 0.0, 0.0, False)
        optimizer = ttml.optimizers.SGD(params, opt_cfg)

        # Create input and targets
        tokens = np.random.randint(
            0, tiny_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        targets = np.random.randint(
            0, tiny_config.vocab_size, size=(batch_size, seq_len)
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
        model = create_nanogpt(tiny_config)
        model.eval()

        batch_size = 2
        seq_len = 32  # Tile-aligned

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


@pytest.mark.parametrize("n_layer", [1, 2])
@pytest.mark.parametrize("n_head", [2, 4])
def test_model_various_configs(n_layer, n_head):
    """Test model with various configuration combinations.

    Note: All dimensions must be tile-aligned (multiples of 32).
    head_dim = n_embd / n_head must also be tile-aligned.
    """
    # Ensure n_embd and head_dim are tile-aligned
    # With n_head=2 or 4, we need n_embd such that n_embd/n_head >= 32
    n_embd = n_head * 32  # head_dim = 32 (tile-aligned)

    config = NanoGPTConfig(
        vocab_size=64,  # Tile-aligned
        block_size=64,  # Tile-aligned
        n_embd=n_embd,  # 64 or 128 depending on n_head
        n_layer=n_layer,
        n_head=n_head,
        dropout=0.0,
        bias=True,
    )

    model = create_nanogpt(config)
    model.eval()

    batch_size = 2
    seq_len = 32  # Tile-aligned

    tokens = np.random.randint(
        0, config.vocab_size, size=(batch_size, 1, 1, seq_len)
    ).astype(np.uint32)
    input_tensor = ttml.autograd.Tensor.from_numpy(
        tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )

    output = model(input_tensor)

    assert output is not None
    assert output.shape()[0] == batch_size
    assert output.shape()[3] == config.vocab_size

    ttml.autograd.AutoContext.get_instance().reset_graph()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
