# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LoRA (Low-Rank Adaptation) module.

Covers:
- LoraConfig dataclass (all fields, defaults, custom values)
- Weight freezing (base model frozen, LoRA adapters trainable)
- LoraLinear injection into target modules
- LoRA adapter shapes (lora_A, lora_B)
- Scaling (standard and rsLoRA)
- Bias trainability
- trainable_modules for selective unfreezing
- Dropout configuration
- Trainable parameter count vs total (parameter efficiency)
- Optimizer state size proportional to LoRA adapters
- Forward pass validity
- Full training step (forward + backward + optimizer)
- Frozen params unchanged after training
- Train/eval mode propagation
- Per-block adapters on fused layers (LoraBlocks)

Tests exercise both NanoGPT (nanogpt config) and Llama (nanollama config).
"""

import math
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

import ttnn
import ttml
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig
from ttml.models.nanogpt import NanoGPTConfig, create_nanogpt
from ttml.modules import LinearLayer, LoraBlocks, LoraConfig, LoraLinear, LoraModel, RunMode


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def toy_llama_config():
    """Toy Llama config for fast tests (tile-aligned dims)."""
    return LlamaConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        max_position_embeddings=64,
        rope_theta=10000.0,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        runner_type=RunnerType.Default,
        weight_tying=WeightTyingType.Disabled,
    )


@pytest.fixture
def toy_gpt_config():
    """Toy NanoGPT config for fast tests (tile-aligned dims)."""
    return NanoGPTConfig(
        vocab_size=64,
        block_size=64,
        n_embd=64,
        n_layer=2,
        n_head=2,
        dropout=0.0,
        bias=True,
    )


def _count_trainable_params(model) -> int:
    total = 0
    for _, param in model.parameters().items():
        if param.get_requires_grad():
            n = 1
            for d in param.shape():
                n *= d
            total += n
    return total


def _count_total_params(model) -> int:
    total = 0
    for _, param in model.parameters().items():
        n = 1
        for d in param.shape():
            n *= d
        total += n
    return total


def _count_trainable_tensors(model) -> int:
    return sum(1 for p in model.parameters().values() if p.get_requires_grad())


# =============================================================================
# LoRA Scaling Tests
# =============================================================================


class TestLoraScaling:
    @pytest.mark.parametrize("rank,alpha", [(4, 8.0), (8, 16.0), (16, 32.0), (32, 1.0)])
    def test_standard_scaling(self, rank, alpha):
        config = LoraConfig(rank=rank, alpha=alpha, use_rslora=False)
        linear = LinearLayer(64, 64)
        lora_linear = LoraLinear(linear, config)
        assert lora_linear.scaling == pytest.approx(alpha / rank)

    @pytest.mark.parametrize("rank,alpha", [(4, 8.0), (8, 16.0), (16, 32.0)])
    def test_rslora_scaling(self, rank, alpha):
        config = LoraConfig(rank=rank, alpha=alpha, use_rslora=True)
        linear = LinearLayer(64, 64)
        lora_linear = LoraLinear(linear, config)
        assert lora_linear.scaling == pytest.approx(alpha / math.sqrt(rank))


# =============================================================================
# LoRA Dropout Tests
# =============================================================================


class TestLoraDropout:
    """Test LoRA dropout is stored correctly."""

    @pytest.mark.parametrize("dropout", [0.0, 0.05, 0.1, 0.5])
    def test_dropout_stored(self, dropout):
        config = LoraConfig(rank=4, alpha=8.0, lora_dropout=dropout)
        linear = LinearLayer(64, 64)
        lora_linear = LoraLinear(linear, config)
        assert lora_linear.dropout_prob == dropout


# =============================================================================
# Weight Freezing Tests
# =============================================================================


class TestLoraWeightFreezing:
    """Test that LoRA correctly freezes base model weights and keeps adapters trainable."""

    def test_all_base_params_frozen_llama(self, toy_llama_config):
        model = Llama(toy_llama_config)

        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear", "out_linear"])
        lora_model = LoraModel(model, lora_config)

        for name, param in lora_model.parameters().items():
            if "lora_A" in name or "lora_B" in name:
                assert param.get_requires_grad(), f"LoRA param {name} should be trainable"
            else:
                assert not param.get_requires_grad(), f"Base param {name} should be frozen"

    def test_all_base_params_frozen_nanogpt(self, toy_gpt_config):
        model = create_nanogpt(toy_gpt_config)

        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear", "out_linear"])
        lora_model = LoraModel(model, lora_config)

        for name, param in lora_model.parameters().items():
            if "lora_A" in name or "lora_B" in name:
                assert param.get_requires_grad(), f"LoRA param {name} should be trainable"
            else:
                assert not param.get_requires_grad(), f"Base param {name} should be frozen"


# =============================================================================
# LoRA Injection Tests
# =============================================================================


class TestLoraInjection:
    """Test that LoRA correctly replaces target LinearLayers with LoraLinear."""

    def test_target_modules_replaced_llama(self, toy_llama_config):
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(
            rank=4,
            alpha=8.0,
            target_modules=["qkv_linear", "out_linear"],
        )
        lora_model = LoraModel(model, lora_config)

        for block in lora_model.model.blocks:
            assert isinstance(block.attention.qkv_linear, LoraLinear)
            assert isinstance(block.attention.out_linear, LoraLinear)

    def test_non_target_modules_unchanged_llama(self, toy_llama_config):
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear"])
        lora_model = LoraModel(model, lora_config)

        for block in lora_model.model.blocks:
            assert isinstance(block.attention.qkv_linear, LoraLinear)
            assert isinstance(block.attention.out_linear, LinearLayer)
            assert isinstance(block.mlp.w_gate_up, LinearLayer)
            assert isinstance(block.mlp.w2, LinearLayer)

    def test_target_modules_replaced_nanogpt(self, toy_gpt_config):
        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(
            rank=4,
            alpha=8.0,
            target_modules=["qkv_linear", "out_linear", "fc1", "fc2"],
        )
        lora_model = LoraModel(model, lora_config)

        for block in lora_model.model.blocks:
            assert isinstance(block.attention.qkv_linear, LoraLinear)
            assert isinstance(block.attention.out_linear, LoraLinear)
            assert isinstance(block.mlp.fc1, LoraLinear)
            assert isinstance(block.mlp.fc2, LoraLinear)

    def test_non_target_modules_unchanged_nanogpt(self, toy_gpt_config):
        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear"])
        lora_model = LoraModel(model, lora_config)

        for block in lora_model.model.blocks:
            assert isinstance(block.attention.qkv_linear, LoraLinear)
            assert isinstance(block.attention.out_linear, LinearLayer)
            assert isinstance(block.mlp.fc1, LinearLayer)
            assert isinstance(block.mlp.fc2, LinearLayer)

    def test_empty_target_modules_no_injection(self, toy_llama_config):
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=[])
        lora_model = LoraModel(model, lora_config)

        for name, _ in lora_model.parameters().items():
            assert "lora_A" not in name and "lora_B" not in name

    def test_lora_adapter_shapes_llama(self, toy_llama_config):
        rank = 4
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(rank=rank, alpha=8.0, target_modules=["qkv_linear"])
        lora_model = LoraModel(model, lora_config)

        for block in lora_model.model.blocks:
            ll = block.attention.qkv_linear
            assert isinstance(ll, LoraLinear)
            assert tuple(ll.lora_A.tensor.shape()) == (
                1,
                1,
                rank,
                ll.in_features,
            )
            assert tuple(ll.lora_B.tensor.shape()) == (
                1,
                1,
                ll.out_features,
                rank,
            )

    def test_lora_adapter_shapes_nanogpt(self, toy_gpt_config):
        rank = 4
        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(rank=rank, alpha=8.0, target_modules=["qkv_linear"])
        lora_model = LoraModel(model, lora_config)

        for block in lora_model.model.blocks:
            ll = block.attention.qkv_linear
            assert isinstance(ll, LoraLinear)
            assert tuple(ll.lora_A.tensor.shape()) == (
                1,
                1,
                rank,
                ll.in_features,
            )
            assert tuple(ll.lora_B.tensor.shape()) == (
                1,
                1,
                ll.out_features,
                rank,
            )


# =============================================================================
# Bias Trainability Tests
# =============================================================================


class TestLoraBiasTrainability:
    """Test LoRA bias trainability configuration."""

    def test_bias_frozen_by_default(self, toy_gpt_config):
        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(
            rank=4,
            alpha=8.0,
            target_modules=["qkv_linear", "out_linear"],
            is_bias_trainable=False,
        )
        lora_model = LoraModel(model, lora_config)

        for name, param in lora_model.parameters().items():
            if "bias" in name:
                assert not param.get_requires_grad(), f"Bias {name} should be frozen"

    def test_bias_trainable_when_enabled(self, toy_gpt_config):
        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(
            rank=4,
            alpha=8.0,
            target_modules=["qkv_linear", "out_linear"],
            is_bias_trainable=True,
        )
        lora_model = LoraModel(model, lora_config)

        for block in lora_model.model.blocks:
            qkv = block.attention.qkv_linear
            out = block.attention.out_linear
            if qkv.bias is not None:
                assert qkv.bias.tensor.get_requires_grad()
            if out.bias is not None:
                assert out.bias.tensor.get_requires_grad()


# =============================================================================
# Trainable Modules Tests
# =============================================================================


class TestLoraTrainableModules:
    """Test selective unfreezing via trainable_modules."""

    def test_trainable_modules_unfreezes_params(self, toy_llama_config):
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(
            rank=4,
            alpha=8.0,
            target_modules=["qkv_linear"],
            trainable_modules=["tok_emb"],
        )
        lora_model = LoraModel(model, lora_config)

        tok_emb_found = False
        for name, param in lora_model.parameters().items():
            if "tok_emb" in name:
                assert param.get_requires_grad(), f"tok_emb param {name} should be trainable"
                tok_emb_found = True
        assert tok_emb_found, "Should find tok_emb parameters"

    def test_trainable_modules_unfreezes_params_nanogpt(self, toy_gpt_config):
        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(
            rank=4,
            alpha=8.0,
            target_modules=["qkv_linear"],
            trainable_modules=["tok_emb"],
        )
        lora_model = LoraModel(model, lora_config)

        tok_emb_found = False
        for name, param in lora_model.parameters().items():
            if "tok_emb" in name:
                assert param.get_requires_grad(), f"tok_emb param {name} should be trainable"
                tok_emb_found = True
        assert tok_emb_found


# =============================================================================
# Trainable Parameter Count Tests
# =============================================================================


class TestLoraTrainableParameterCount:
    """Verify trainable parameter count is small and matches LoRA math."""

    def test_exact_trainable_count_llama(self, toy_llama_config):
        """Verify exact trainable parameter count matches LoRA math for Llama."""
        rank = 4
        hidden = toy_llama_config.hidden_size
        num_kv_heads = toy_llama_config.num_key_value_heads
        num_heads = toy_llama_config.num_attention_heads
        head_dim = hidden // num_heads
        kv_dim = num_kv_heads * head_dim * 2

        model = Llama(toy_llama_config)
        lora_config = LoraConfig(
            rank=rank,
            alpha=8.0,
            target_modules=["qkv_linear", "out_linear"],
        )
        lora_model = LoraModel(model, lora_config)

        # Per layer: lora_A + lora_B for each target
        #   qkv_linear: rank*hidden + qkv_dim*rank
        #   out_linear: rank*hidden + hidden*rank
        qkv_dim = hidden + kv_dim
        per_layer = rank * hidden + qkv_dim * rank + rank * hidden + hidden * rank
        expected = per_layer * toy_llama_config.num_hidden_layers
        actual = _count_trainable_params(lora_model)
        assert actual == expected, f"Expected {expected} trainable params, got {actual}"

        total = _count_total_params(lora_model)
        assert total == 133952

    def test_exact_trainable_count_nanogpt(self, toy_gpt_config):
        """Verify exact trainable parameter count for NanoGPT."""
        rank = 4
        embd = toy_gpt_config.n_embd

        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(
            rank=rank,
            alpha=8.0,
            target_modules=["qkv_linear", "out_linear"],
        )
        lora_model = LoraModel(model, lora_config)

        # qkv_linear: in=embd, out=embd*3 -> lora_A: rank*embd, lora_B: (embd*3)*rank
        # out_linear: in=embd, out=embd    -> lora_A: rank*embd, lora_B: embd*rank
        per_layer = rank * embd + (embd * 3) * rank + rank * embd + embd * rank
        expected = per_layer * toy_gpt_config.n_layer
        actual = _count_trainable_params(lora_model)
        assert actual == expected, f"Expected {expected} trainable params, got {actual}"

        total = _count_total_params(lora_model)
        assert total == 115456

    def test_rank_affects_param_count(self, toy_llama_config):
        """Doubling rank should double trainable parameters."""
        model_r4 = Llama(toy_llama_config)
        lora_r4 = LoraModel(
            model_r4,
            LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear"]),
        )
        count_r4 = _count_trainable_params(lora_r4)

        model_r8 = Llama(toy_llama_config)
        lora_r8 = LoraModel(
            model_r8,
            LoraConfig(rank=8, alpha=16.0, target_modules=["qkv_linear"]),
        )
        count_r8 = _count_trainable_params(lora_r8)

        assert count_r8 == count_r4 * 2

    def test_more_targets_more_params(self, toy_llama_config):
        """Targeting more modules should increase trainable parameter count."""
        model1 = Llama(toy_llama_config)
        lora1 = LoraModel(
            model1,
            LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear"]),
        )
        count1 = _count_trainable_params(lora1)

        model3 = Llama(toy_llama_config)
        lora3 = LoraModel(
            model3,
            LoraConfig(
                rank=4,
                alpha=8.0,
                target_modules=["qkv_linear", "out_linear"],
            ),
        )
        count3 = _count_trainable_params(lora3)

        assert count3 > count1


# =============================================================================
# Optimizer State Size Tests
# =============================================================================


class TestLoraOptimizerState:
    """Verify that optimizer state is proportional to LoRA adapters, not full model."""

    def test_optimizer_state_matches_trainable_llama(self, toy_llama_config):
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear", "out_linear"])
        lora_model = LoraModel(model, lora_config)

        trainable_count = _count_trainable_tensors(lora_model)
        total_count = len(lora_model.parameters())
        assert trainable_count < total_count
        assert trainable_count > 0

        adamw_cfg = ttml.optimizers.AdamWConfig.make(0.001, 0.9, 0.999, 1e-8, 0.01)
        optimizer = ttml.optimizers.AdamW(lora_model.parameters(), adamw_cfg)

        state = optimizer.get_state_dict()
        for key, tensors in state.items():
            if isinstance(tensors, list) and len(tensors) > 0:
                assert len(tensors) == trainable_count, (
                    f"State '{key}' has {len(tensors)} entries, "
                    f"expected {trainable_count} (trainable param tensors)"
                )

    def test_optimizer_state_matches_trainable_nanogpt(self, toy_gpt_config):
        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear", "out_linear"])
        lora_model = LoraModel(model, lora_config)

        trainable_count = _count_trainable_tensors(lora_model)
        total_count = len(lora_model.parameters())
        assert trainable_count < total_count

        adamw_cfg = ttml.optimizers.AdamWConfig.make(0.001, 0.9, 0.999, 1e-8, 0.01)
        optimizer = ttml.optimizers.AdamW(lora_model.parameters(), adamw_cfg)

        state = optimizer.get_state_dict()
        for key, tensors in state.items():
            if isinstance(tensors, list) and len(tensors) > 0:
                assert len(tensors) == trainable_count, (
                    f"State '{key}' has {len(tensors)} entries, "
                    f"expected {trainable_count} (trainable param tensors)"
                )


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestLoraForwardPass:
    """Test LoRA-wrapped models produce valid forward pass outputs."""

    def test_forward_llama(self, toy_llama_config):
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(
            rank=4,
            alpha=8.0,
            target_modules=["qkv_linear", "out_linear"],
        )
        lora_model = LoraModel(model, lora_config)
        lora_model.eval()

        batch_size = 2
        seq_len = toy_llama_config.max_position_embeddings

        tokens = np.random.randint(0, toy_llama_config.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        logits = lora_model(input_tensor, None)
        logits_shape = logits.shape()

        assert len(logits_shape) == 4
        assert logits_shape[0] == batch_size
        assert logits_shape[2] == seq_len
        assert logits_shape[3] == toy_llama_config.vocab_size

        logits_np = logits.to_numpy(ttnn.DataType.FLOAT32)
        assert np.all(np.isfinite(logits_np)), "Logits should be finite"

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_forward_nanogpt(self, toy_gpt_config):
        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(
            rank=4,
            alpha=8.0,
            target_modules=["qkv_linear", "out_linear", "fc1", "fc2"],
        )
        lora_model = LoraModel(model, lora_config)
        lora_model.eval()

        batch_size = 2
        seq_len = toy_gpt_config.block_size

        tokens = np.random.randint(0, toy_gpt_config.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        logits = lora_model(input_tensor, None)
        logits_shape = logits.shape()

        assert len(logits_shape) == 4
        assert logits_shape[0] == batch_size
        assert logits_shape[2] == seq_len
        assert logits_shape[3] == toy_gpt_config.vocab_size

        logits_np = logits.to_numpy(ttnn.DataType.FLOAT32)
        assert np.all(np.isfinite(logits_np)), "Logits should be finite"

        ttml.autograd.AutoContext.get_instance().reset_graph()

    def test_forward_with_all_mlp_targets_llama(self, toy_llama_config):
        """LoRA on all linear layers including MLP."""
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(
            rank=4,
            alpha=8.0,
            target_modules=["qkv_linear", "out_linear", "w_gate_up", "w2"],
        )
        lora_model = LoraModel(model, lora_config)
        lora_model.eval()

        batch_size = 2
        seq_len = toy_llama_config.max_position_embeddings

        tokens = np.random.randint(0, toy_llama_config.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        logits = lora_model(input_tensor, None)
        logits_np = logits.to_numpy(ttnn.DataType.FLOAT32)
        assert np.all(np.isfinite(logits_np))

        ttml.autograd.AutoContext.get_instance().reset_graph()


# =============================================================================
# Training Step Tests
# =============================================================================


class TestLoraTrainingStep:
    """Test full training steps (forward + backward + optimizer) with LoRA models."""

    def test_training_step_llama(self, toy_llama_config):
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(
            rank=4,
            alpha=8.0,
            target_modules=["qkv_linear", "out_linear"],
        )
        lora_model = LoraModel(model, lora_config)
        lora_model.train()

        batch_size = 2
        seq_len = toy_llama_config.max_position_embeddings

        adamw_cfg = ttml.optimizers.AdamWConfig.make(0.001, 0.9, 0.999, 1e-8, 0.01)
        optimizer = ttml.optimizers.AdamW(lora_model.parameters(), adamw_cfg)

        tokens = np.random.randint(0, toy_llama_config.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        targets = np.random.randint(0, toy_llama_config.vocab_size, size=(batch_size, seq_len)).astype(np.uint32)
        target_tensor = ttml.autograd.Tensor.from_numpy(
            targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        optimizer.zero_grad()
        logits = lora_model(input_tensor, None)
        loss = ttml.ops.loss.cross_entropy_loss(logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN)
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()

        loss_np = loss.to_numpy(ttnn.DataType.FLOAT32)
        assert np.isfinite(loss_np).all(), "Loss should be finite"

    def test_training_step_nanogpt(self, toy_gpt_config):
        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear", "out_linear"])
        lora_model = LoraModel(model, lora_config)
        lora_model.train()

        batch_size = 2
        seq_len = toy_gpt_config.block_size

        adamw_cfg = ttml.optimizers.AdamWConfig.make(0.001, 0.9, 0.999, 1e-8, 0.01)
        optimizer = ttml.optimizers.AdamW(lora_model.parameters(), adamw_cfg)

        tokens = np.random.randint(0, toy_gpt_config.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        targets = np.random.randint(0, toy_gpt_config.vocab_size, size=(batch_size, seq_len)).astype(np.uint32)
        target_tensor = ttml.autograd.Tensor.from_numpy(
            targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        optimizer.zero_grad()
        logits = lora_model(input_tensor, None)
        loss = ttml.ops.loss.cross_entropy_loss(logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN)
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()

        loss_np = loss.to_numpy(ttnn.DataType.FLOAT32)
        assert np.isfinite(loss_np).all(), "Loss should be finite"

    def test_only_lora_params_updated_llama(self, toy_llama_config):
        """Frozen params must not change; at least some LoRA params must change."""
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear"])
        lora_model = LoraModel(model, lora_config)
        lora_model.train()

        frozen_before = {}
        lora_before = {}
        for name, param in lora_model.parameters().items():
            arr = param.to_numpy(ttnn.DataType.FLOAT32).copy()
            if param.get_requires_grad():
                lora_before[name] = arr
            else:
                frozen_before[name] = arr

        batch_size = 2
        seq_len = toy_llama_config.max_position_embeddings

        adamw_cfg = ttml.optimizers.AdamWConfig.make(0.01, 0.9, 0.999, 1e-8, 0.01)
        optimizer = ttml.optimizers.AdamW(lora_model.parameters(), adamw_cfg)

        tokens = np.random.randint(0, toy_llama_config.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        targets = np.random.randint(0, toy_llama_config.vocab_size, size=(batch_size, seq_len)).astype(np.uint32)
        target_tensor = ttml.autograd.Tensor.from_numpy(
            targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        optimizer.zero_grad()
        logits = lora_model(input_tensor, None)
        loss = ttml.ops.loss.cross_entropy_loss(logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN)
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()

        for name, before in frozen_before.items():
            after = lora_model.parameters()[name].to_numpy(ttnn.DataType.FLOAT32)
            assert np.allclose(before, after, atol=1e-6), f"Frozen param {name} should not change during training"

        any_changed = False
        for name, before in lora_before.items():
            after = lora_model.parameters()[name].to_numpy(ttnn.DataType.FLOAT32)
            if not np.allclose(before, after, atol=1e-6):
                any_changed = True
                break
        assert any_changed, "At least some LoRA params should change after training"

    def test_only_lora_params_updated_nanogpt(self, toy_gpt_config):
        """Frozen params must not change; at least some LoRA params must change."""
        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear"])
        lora_model = LoraModel(model, lora_config)
        lora_model.train()

        frozen_before = {}
        lora_before = {}
        for name, param in lora_model.parameters().items():
            arr = param.to_numpy(ttnn.DataType.FLOAT32).copy()
            if param.get_requires_grad():
                lora_before[name] = arr
            else:
                frozen_before[name] = arr

        batch_size = 2
        seq_len = toy_gpt_config.block_size

        adamw_cfg = ttml.optimizers.AdamWConfig.make(0.01, 0.9, 0.999, 1e-8, 0.01)
        optimizer = ttml.optimizers.AdamW(lora_model.parameters(), adamw_cfg)

        tokens = np.random.randint(0, toy_gpt_config.vocab_size, size=(batch_size, 1, 1, seq_len)).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        targets = np.random.randint(0, toy_gpt_config.vocab_size, size=(batch_size, seq_len)).astype(np.uint32)
        target_tensor = ttml.autograd.Tensor.from_numpy(
            targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        optimizer.zero_grad()
        logits = lora_model(input_tensor, None)
        loss = ttml.ops.loss.cross_entropy_loss(logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN)
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()

        for name, before in frozen_before.items():
            after = lora_model.parameters()[name].to_numpy(ttnn.DataType.FLOAT32)
            assert np.allclose(before, after, atol=1e-6), f"Frozen param {name} should not change during training"

        any_changed = False
        for name, before in lora_before.items():
            after = lora_model.parameters()[name].to_numpy(ttnn.DataType.FLOAT32)
            if not np.allclose(before, after, atol=1e-6):
                any_changed = True
                break
        assert any_changed, "At least some LoRA params should change after training"


# =============================================================================
# Per-block LoRA on fused layers (LoraBlocks)
# =============================================================================

# The q, k and v rows of toy_llama_config's qkv_linear: 2 query heads and 1 kv head, of 32.
QKV_ROWS = (64, 32, 32)
Q_AND_V = LoraBlocks(rows=QKV_ROWS, adapted=(True, False, True))


def _block_rows(blocks: LoraBlocks) -> list[slice]:
    """The rows of each block of a fused layer."""
    ends = np.cumsum(blocks.rows)
    return [slice(end - rows, end) for end, rows in zip(ends, blocks.rows)]


def _read_2d(tensor) -> np.ndarray:
    values = tensor.to_numpy(ttnn.DataType.FLOAT32)
    return values.reshape(values.shape[-2:])


def _bf16_values(rng, shape) -> np.ndarray:
    """Random float32 values that bf16 holds exactly, so the device sees the same inputs as numpy."""
    return rng.standard_normal(shape).astype(ml_dtypes.bfloat16).astype(np.float32)


def _q_and_v_llama(config) -> LoraModel:
    lora_config = LoraConfig(
        rank=4, alpha=8.0, target_modules=["qkv_linear", "out_linear"], target_blocks={"qkv_linear": Q_AND_V}
    )
    return LoraModel(Llama(config), lora_config)


class TestLoraBlocks:
    def test_needs_a_flag_per_block(self, expect_error):
        with expect_error(ValueError, "3 blocks of rows, 2 adapted flags"):
            LoraBlocks(rows=QKV_ROWS, adapted=(True, False))

    def test_needs_an_adapted_block(self, expect_error):
        with expect_error(ValueError, "no block is adapted"):
            LoraBlocks(rows=QKV_ROWS, adapted=(False, False, False))

    def test_the_blocks_must_cover_the_layer(self, expect_error):
        with expect_error(ValueError, "LoraBlocks of 128 rows for a layer of 96"):
            LoraLinear(LinearLayer(64, 96), LoraConfig(rank=4), Q_AND_V)

    def test_each_adapted_block_gets_an_adapter_of_its_own(self):
        rank = 4
        layer = LoraLinear(LinearLayer(64, sum(QKV_ROWS)), LoraConfig(rank=rank), Q_AND_V)

        assert tuple(layer.lora_A.tensor.shape()) == (1, 1, 2 * rank, 64), "one A per adapted block, stacked"
        assert tuple(layer.lora_B.tensor.shape()) == (1, 1, sum(QKV_ROWS), rank)
        np.testing.assert_array_equal(_read_2d(layer.lora_expand.tensor), np.tile(np.eye(rank), (2, 1)))
        q, _, v = _block_rows(Q_AND_V)
        mask = np.zeros((sum(QKV_ROWS), 2 * rank))
        mask[q, :rank] = 1.0
        mask[v, rank:] = 1.0
        np.testing.assert_array_equal(_read_2d(layer.lora_mask.tensor), mask)

    def test_the_layout_constants_are_not_parameters(self):
        """They'd otherwise be trained, and saved with the adapters."""
        layer = LoraLinear(LinearLayer(64, sum(QKV_ROWS)), LoraConfig(rank=4), Q_AND_V)
        names = {name.rsplit("/", 1)[-1] for name in layer.parameters()}
        assert names == {"weight", "bias", "lora_A", "lora_B"}
        assert not layer.lora_expand.tensor.get_requires_grad()
        assert not layer.lora_mask.tensor.get_requires_grad()

    def test_each_block_sees_only_its_own_adapter(self):
        """Block i's rows get ``B_i @ A_i @ x``, and the rows of a block that isn't adapted get no update, whatever
        lora_B holds there."""
        rank, in_features, out_features, tokens = 4, 64, sum(QKV_ROWS), 32
        layer = LoraLinear(LinearLayer(in_features, out_features, has_bias=False), LoraConfig(rank=rank), Q_AND_V)
        rng = np.random.default_rng(0)
        a = _bf16_values(rng, (2 * rank, in_features))
        # The k rows are not zero, as no training would leave them, so masking them out shows.
        b = _bf16_values(rng, (out_features, rank))
        x = _bf16_values(rng, (1, 1, tokens, in_features))
        # A zero base weight leaves the output to the update alone.
        ttml.autograd.assign_numpy(layer.weight.tensor, np.zeros((out_features, in_features), np.float32))
        ttml.autograd.assign_numpy(layer.lora_A.tensor, a)
        ttml.autograd.assign_numpy(layer.lora_B.tensor, b)
        layer.eval()

        output = layer(ttml.autograd.Tensor.from_numpy(x, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16))
        got = _read_2d(output)
        ttml.autograd.AutoContext.get_instance().reset_graph()

        q, k, v = _block_rows(Q_AND_V)
        h = x.reshape(tokens, in_features) @ a.T
        expected = np.zeros((tokens, out_features), np.float32)
        expected[:, q] = layer.scaling * h[:, :rank] @ b[q].T
        expected[:, v] = layer.scaling * h[:, rank:] @ b[v].T
        assert not got[:, k].any(), "the rows of k, which has no adapter, were updated"
        np.testing.assert_allclose(got, expected, rtol=0.05, atol=0.02 * np.abs(expected).max())

    def test_target_blocks_pick_the_layers_they_apply_to(self, toy_llama_config):
        lora_model = _q_and_v_llama(toy_llama_config)
        for block in lora_model.model.blocks:
            qkv, out = block.attention.qkv_linear, block.attention.out_linear
            assert qkv.blocks == Q_AND_V and tuple(qkv.lora_A.tensor.shape()) == (1, 1, 8, qkv.in_features)
            assert out.blocks is None and tuple(out.lora_A.tensor.shape()) == (1, 1, 4, out.in_features)

    def test_the_layers_of_a_model_share_the_layout_constants(self, toy_llama_config):
        first, second = (block.attention.qkv_linear for block in _q_and_v_llama(toy_llama_config).model.blocks)
        assert first.lora_mask.tensor is second.lora_mask.tensor
        assert first.lora_expand.tensor is second.lora_expand.tensor

    def test_rejects_target_blocks_that_match_no_adapted_layer(self, toy_llama_config, expect_error):
        lora_config = LoraConfig(rank=4, target_modules=["qkv_linear"], target_blocks={"w_gate_up": Q_AND_V})
        with expect_error(ValueError, r"target_blocks pattern\(s\) matched no layer .*w_gate_up"):
            LoraModel(Llama(toy_llama_config), lora_config)

    def test_the_lora_b_rows_of_a_block_that_is_not_adapted_stay_zero(self, toy_llama_config):
        """k gets no gradient, so its rows of lora_B keep their initial zeros through training, while q's and v's
        train."""
        lora_model = _q_and_v_llama(toy_llama_config)
        lora_model.train()
        optimizer = ttml.optimizers.AdamW(
            lora_model.parameters(), ttml.optimizers.AdamWConfig.make(0.01, 0.9, 0.999, 1e-8, 0.01)
        )
        batch_size, seq_len = 2, toy_llama_config.max_position_embeddings
        rng = np.random.default_rng(0)
        tokens = rng.integers(0, toy_llama_config.vocab_size, (batch_size, 1, 1, seq_len)).astype(np.uint32)
        targets = rng.integers(0, toy_llama_config.vocab_size, (batch_size, seq_len)).astype(np.uint32)
        q, k, v = _block_rows(Q_AND_V)

        for _ in range(2):
            optimizer.zero_grad()
            logits = lora_model(
                ttml.autograd.Tensor.from_numpy(tokens, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32), None
            )
            target_tensor = ttml.autograd.Tensor.from_numpy(targets, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
            loss = ttml.ops.loss.cross_entropy_loss(logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN)
            loss.backward(False)
            for i, block in enumerate(lora_model.model.blocks):
                grad = _read_2d(block.attention.qkv_linear.lora_B.tensor.get_grad_tensor())
                assert not grad[k].any(), f"block {i}: the k rows of lora_B got a gradient"
                assert grad[q].any() and grad[v].any(), f"block {i}: the q or v rows of lora_B got no gradient"
            ttml.autograd.AutoContext.get_instance().reset_graph()
            optimizer.step()

        for i, block in enumerate(lora_model.model.blocks):
            lora_b = _read_2d(block.attention.qkv_linear.lora_B.tensor)
            assert not lora_b[k].any(), f"block {i}: the k rows of lora_B moved off zero"
            assert lora_b[q].any() and lora_b[v].any(), f"block {i}: the q or v rows of lora_B did not train"


# =============================================================================
# Train/Eval Mode Propagation Tests
# =============================================================================


class TestLoraTrainEvalMode:
    """Test train/eval mode propagation through LoRA-wrapped models."""

    def test_mode_propagation_llama(self, toy_llama_config):
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear"])
        lora_model = LoraModel(model, lora_config)

        lora_model.eval()
        assert lora_model.get_run_mode() == RunMode.EVAL

        lora_model.train()
        assert lora_model.get_run_mode() == RunMode.TRAIN

    def test_mode_propagation_nanogpt(self, toy_gpt_config):
        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear"])
        lora_model = LoraModel(model, lora_config)

        lora_model.eval()
        assert lora_model.get_run_mode() == RunMode.EVAL

        lora_model.train()
        assert lora_model.get_run_mode() == RunMode.TRAIN

    def test_lora_linear_inherits_mode(self, toy_llama_config):
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear"])
        lora_model = LoraModel(model, lora_config)

        lora_model.eval()
        for block in lora_model.model.blocks:
            assert block.attention.qkv_linear.get_run_mode() == RunMode.EVAL

        lora_model.train()
        for block in lora_model.model.blocks:
            assert block.attention.qkv_linear.get_run_mode() == RunMode.TRAIN


# =============================================================================
# The LoRA example's --resume
# =============================================================================


def _lora_llama_example():
    """sources/examples/lora_llama/train_lora_llama.py, which imports its sibling llama_config.py."""
    example_dir = str(Path(__file__).resolve().parents[2] / "sources" / "examples" / "lora_llama")
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)
    import train_lora_llama

    return train_lora_llama


class TestLoraLlamaExampleResume:
    @staticmethod
    def lora_llama(config):
        return LoraModel(Llama(config), LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear", "out_linear"]))

    def test_resumes_every_adapter(self, toy_llama_config, tmp_path):
        example = _lora_llama_example()
        saved = self.lora_llama(toy_llama_config)
        example.save_lora_checkpoint(saved, str(tmp_path), step=1)
        resumed = self.lora_llama(toy_llama_config)

        example.load_lora_checkpoint(resumed, str(tmp_path / "lora_step_1.safetensors"))

        resumed_params = resumed.parameters()
        for name, param in saved.parameters().items():
            if not param.get_requires_grad():
                continue
            np.testing.assert_array_equal(
                resumed_params[name].to_numpy(ttnn.DataType.FLOAT32), param.to_numpy(ttnn.DataType.FLOAT32), name
            )

    def test_refuses_adapters_saved_before_the_qkv_fusion(self, toy_llama_config, tmp_path, expect_error):
        from safetensors.numpy import load_file, save_file

        example = _lora_llama_example()
        model = self.lora_llama(toy_llama_config)
        example.save_lora_checkpoint(model, str(tmp_path), step=1)
        path = str(tmp_path / "lora_step_1.safetensors")
        save_file({name.replace("qkv_linear", "q_linear"): array for name, array in load_file(path).items()}, path)

        with expect_error(RuntimeError, "not in the checkpoint: .*qkv_linear"):
            example.load_lora_checkpoint(model, path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
