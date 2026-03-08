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

Tests exercise both NanoGPT (nanogpt config) and Llama (nanollama config).
"""

import math

import numpy as np
import pytest

import ttnn
import ttml
from ttml.common.data import build_causal_mask
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig
from ttml.models.nanogpt import NanoGPT, NanoGPTConfig, create_nanogpt
from ttml.modules import LinearLayer, LoraConfig, LoraLinear, LoraModel, RunMode


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

        lora_config = LoraConfig(
            rank=4, alpha=8.0, target_modules=["q_linear", "out_linear"]
        )
        lora_model = LoraModel(model, lora_config)

        for name, param in lora_model.parameters().items():
            if "lora_A" in name or "lora_B" in name:
                assert (
                    param.get_requires_grad()
                ), f"LoRA param {name} should be trainable"
            else:
                assert (
                    not param.get_requires_grad()
                ), f"Base param {name} should be frozen"

    def test_all_base_params_frozen_nanogpt(self, toy_gpt_config):
        model = create_nanogpt(toy_gpt_config)

        lora_config = LoraConfig(
            rank=4, alpha=8.0, target_modules=["qkv_linear", "out_linear"]
        )
        lora_model = LoraModel(model, lora_config)

        for name, param in lora_model.parameters().items():
            if "lora_A" in name or "lora_B" in name:
                assert (
                    param.get_requires_grad()
                ), f"LoRA param {name} should be trainable"
            else:
                assert (
                    not param.get_requires_grad()
                ), f"Base param {name} should be frozen"


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
            target_modules=["q_linear", "kv_linear", "out_linear"],
        )
        lora_model = LoraModel(model, lora_config)

        for block in lora_model.model.blocks:
            assert isinstance(block.attention.q_linear, LoraLinear)
            assert isinstance(block.attention.kv_linear, LoraLinear)
            assert isinstance(block.attention.out_linear, LoraLinear)

    def test_non_target_modules_unchanged_llama(self, toy_llama_config):
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["q_linear"])
        lora_model = LoraModel(model, lora_config)

        for block in lora_model.model.blocks:
            assert isinstance(block.attention.q_linear, LoraLinear)
            assert isinstance(block.attention.kv_linear, LinearLayer)
            assert isinstance(block.attention.out_linear, LinearLayer)
            assert isinstance(block.mlp.w1, LinearLayer)
            assert isinstance(block.mlp.w2, LinearLayer)
            assert isinstance(block.mlp.w3, LinearLayer)

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
        lora_config = LoraConfig(rank=rank, alpha=8.0, target_modules=["q_linear"])
        lora_model = LoraModel(model, lora_config)

        for block in lora_model.model.blocks:
            ll = block.attention.q_linear
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
            target_modules=["q_linear"],
            trainable_modules=["tok_emb"],
        )
        lora_model = LoraModel(model, lora_config)

        tok_emb_found = False
        for name, param in lora_model.parameters().items():
            if "tok_emb" in name:
                assert (
                    param.get_requires_grad()
                ), f"tok_emb param {name} should be trainable"
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
                assert (
                    param.get_requires_grad()
                ), f"tok_emb param {name} should be trainable"
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
            target_modules=["q_linear", "kv_linear", "out_linear"],
        )
        lora_model = LoraModel(model, lora_config)

        # Per layer: lora_A + lora_B for each target
        #   q_linear:   rank*hidden + hidden*rank
        #   kv_linear:  rank*hidden + kv_dim*rank
        #   out_linear: rank*hidden + hidden*rank
        per_layer = (
            rank * hidden
            + hidden * rank
            + rank * hidden
            + kv_dim * rank
            + rank * hidden
            + hidden * rank
        )
        expected = per_layer * toy_llama_config.num_hidden_layers
        actual = _count_trainable_params(lora_model)
        assert actual == expected, f"Expected {expected} trainable params, got {actual}"

        total = _count_total_params(lora_model)
        assert total == 134464

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
            LoraConfig(rank=4, alpha=8.0, target_modules=["q_linear"]),
        )
        count_r4 = _count_trainable_params(lora_r4)

        model_r8 = Llama(toy_llama_config)
        lora_r8 = LoraModel(
            model_r8,
            LoraConfig(rank=8, alpha=16.0, target_modules=["q_linear"]),
        )
        count_r8 = _count_trainable_params(lora_r8)

        assert count_r8 == count_r4 * 2

    def test_more_targets_more_params(self, toy_llama_config):
        """Targeting more modules should increase trainable parameter count."""
        model1 = Llama(toy_llama_config)
        lora1 = LoraModel(
            model1,
            LoraConfig(rank=4, alpha=8.0, target_modules=["q_linear"]),
        )
        count1 = _count_trainable_params(lora1)

        model3 = Llama(toy_llama_config)
        lora3 = LoraModel(
            model3,
            LoraConfig(
                rank=4,
                alpha=8.0,
                target_modules=["q_linear", "kv_linear", "out_linear"],
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
        lora_config = LoraConfig(
            rank=4, alpha=8.0, target_modules=["q_linear", "out_linear"]
        )
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
        lora_config = LoraConfig(
            rank=4, alpha=8.0, target_modules=["qkv_linear", "out_linear"]
        )
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
            target_modules=["q_linear", "kv_linear", "out_linear"],
        )
        lora_model = LoraModel(model, lora_config)
        lora_model.eval()

        batch_size = 2
        seq_len = toy_llama_config.max_position_embeddings

        tokens = np.random.randint(
            0, toy_llama_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
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

        tokens = np.random.randint(
            0, toy_gpt_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
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
            target_modules=["q_linear", "kv_linear", "out_linear", "w1", "w2", "w3"],
        )
        lora_model = LoraModel(model, lora_config)
        lora_model.eval()

        batch_size = 2
        seq_len = toy_llama_config.max_position_embeddings

        tokens = np.random.randint(
            0, toy_llama_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
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
            target_modules=["q_linear", "kv_linear", "out_linear"],
        )
        lora_model = LoraModel(model, lora_config)
        lora_model.train()

        batch_size = 2
        seq_len = toy_llama_config.max_position_embeddings

        adamw_cfg = ttml.optimizers.AdamWConfig.make(0.001, 0.9, 0.999, 1e-8, 0.01)
        optimizer = ttml.optimizers.AdamW(lora_model.parameters(), adamw_cfg)

        tokens = np.random.randint(
            0, toy_llama_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        targets = np.random.randint(
            0, toy_llama_config.vocab_size, size=(batch_size, seq_len)
        ).astype(np.uint32)
        target_tensor = ttml.autograd.Tensor.from_numpy(
            targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        optimizer.zero_grad()
        logits = lora_model(input_tensor, None)
        loss = ttml.ops.loss.cross_entropy_loss(
            logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN
        )
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()

        loss_np = loss.to_numpy(ttnn.DataType.FLOAT32)
        assert np.isfinite(loss_np).all(), "Loss should be finite"

    def test_training_step_nanogpt(self, toy_gpt_config):
        model = create_nanogpt(toy_gpt_config)
        lora_config = LoraConfig(
            rank=4, alpha=8.0, target_modules=["qkv_linear", "out_linear"]
        )
        lora_model = LoraModel(model, lora_config)
        lora_model.train()

        batch_size = 2
        seq_len = toy_gpt_config.block_size

        adamw_cfg = ttml.optimizers.AdamWConfig.make(0.001, 0.9, 0.999, 1e-8, 0.01)
        optimizer = ttml.optimizers.AdamW(lora_model.parameters(), adamw_cfg)

        tokens = np.random.randint(
            0, toy_gpt_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        targets = np.random.randint(
            0, toy_gpt_config.vocab_size, size=(batch_size, seq_len)
        ).astype(np.uint32)
        target_tensor = ttml.autograd.Tensor.from_numpy(
            targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        optimizer.zero_grad()
        logits = lora_model(input_tensor, None)
        loss = ttml.ops.loss.cross_entropy_loss(
            logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN
        )
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()

        loss_np = loss.to_numpy(ttnn.DataType.FLOAT32)
        assert np.isfinite(loss_np).all(), "Loss should be finite"

    def test_only_lora_params_updated_llama(self, toy_llama_config):
        """Frozen params must not change; at least some LoRA params must change."""
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["q_linear"])
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

        tokens = np.random.randint(
            0, toy_llama_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        targets = np.random.randint(
            0, toy_llama_config.vocab_size, size=(batch_size, seq_len)
        ).astype(np.uint32)
        target_tensor = ttml.autograd.Tensor.from_numpy(
            targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        optimizer.zero_grad()
        logits = lora_model(input_tensor, None)
        loss = ttml.ops.loss.cross_entropy_loss(
            logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN
        )
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()

        for name, before in frozen_before.items():
            after = lora_model.parameters()[name].to_numpy(ttnn.DataType.FLOAT32)
            assert np.allclose(
                before, after, atol=1e-6
            ), f"Frozen param {name} should not change during training"

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

        tokens = np.random.randint(
            0, toy_gpt_config.vocab_size, size=(batch_size, 1, 1, seq_len)
        ).astype(np.uint32)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        targets = np.random.randint(
            0, toy_gpt_config.vocab_size, size=(batch_size, seq_len)
        ).astype(np.uint32)
        target_tensor = ttml.autograd.Tensor.from_numpy(
            targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        optimizer.zero_grad()
        logits = lora_model(input_tensor, None)
        loss = ttml.ops.loss.cross_entropy_loss(
            logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN
        )
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()

        for name, before in frozen_before.items():
            after = lora_model.parameters()[name].to_numpy(ttnn.DataType.FLOAT32)
            assert np.allclose(
                before, after, atol=1e-6
            ), f"Frozen param {name} should not change during training"

        any_changed = False
        for name, before in lora_before.items():
            after = lora_model.parameters()[name].to_numpy(ttnn.DataType.FLOAT32)
            if not np.allclose(before, after, atol=1e-6):
                any_changed = True
                break
        assert any_changed, "At least some LoRA params should change after training"


# =============================================================================
# Train/Eval Mode Propagation Tests
# =============================================================================


class TestLoraTrainEvalMode:
    """Test train/eval mode propagation through LoRA-wrapped models."""

    def test_mode_propagation_llama(self, toy_llama_config):
        model = Llama(toy_llama_config)
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["q_linear"])
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
        lora_config = LoraConfig(rank=4, alpha=8.0, target_modules=["q_linear"])
        lora_model = LoraModel(model, lora_config)

        lora_model.eval()
        for block in lora_model.model.blocks:
            assert block.attention.q_linear.get_run_mode() == RunMode.EVAL

        lora_model.train()
        for block in lora_model.model.blocks:
            assert block.attention.q_linear.get_run_mode() == RunMode.TRAIN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
