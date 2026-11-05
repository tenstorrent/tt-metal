# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SFTTrainer applies the YAML's ``precision`` block before it builds the optimizer from a dict or the default."""

import numpy as np
import pytest

import ttnn
import ttml
from ttml.datasets.hf_dataloader import InMemoryDataloader, sft_collate_fn
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig
from ttml.modules import LoraConfig
from ttml.trainers import SFTConfig, SFTTrainer, TrainerCallback

FP32_PRECISION = {"master_dtype": "float32", "reduce_dtype": "float32"}
VOCAB_SIZE = 64
SEQ_LEN = 32


@pytest.fixture(autouse=True)
def single_device_loss(monkeypatch):
    """The trainer picks its loss from the named device mesh, which the lazily opened single device of this test
    session doesn't have. On one device without a TP axis that pick is the plain cross entropy."""
    monkeypatch.setattr(SFTTrainer, "_build_loss_fn", lambda self: ttml.ops.loss.cross_entropy_loss)


def _toy_llama():
    return Llama(
        LlamaConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=SEQ_LEN,
            rope_theta=10000.0,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            runner_type=RunnerType.Default,
            weight_tying=WeightTyingType.Disabled,
        )
    )


def _config(**overrides):
    return SFTConfig(
        **{"max_steps": 1, "eval_interval": 0, "save_interval": 0, "disable_progress_bar": True, **overrides}
    )


def _stored_dtype(parameter):
    return parameter.get_value(ttml.autograd.PreferredPrecision.NATIVE).dtype


def _split(model):
    parameters = model.parameters()
    trainable = {name: p for name, p in parameters.items() if p.get_requires_grad()}
    frozen = {name: p for name, p in parameters.items() if not p.get_requires_grad()}
    return trainable, frozen


@pytest.mark.requires_device
def test_applies_precision_before_building_the_optimizer(gradient_reduce_default):
    """The LoRA adapters, the trainable parameters, get fp32 master weights and fp32 gradient accumulation; the frozen
    base stays bf16; the reduce dtype becomes the gradient syncs' default."""
    trainer = SFTTrainer(
        _toy_llama(),
        [],
        None,
        _config(),
        optimizer={"type": "AdamW", "lr": 1e-3},
        peft_config=LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear"]),
        precision=FP32_PRECISION,
    )

    trainable, frozen = _split(trainer.model)
    assert trainable and frozen, "premise: LoRA leaves a frozen base and trainable adapters"
    assert {_stored_dtype(p) for p in trainable.values()} == {ttnn.DataType.FLOAT32}
    assert {p.get_grad_accumulation_dtype() for p in trainable.values()} == {ttnn.DataType.FLOAT32}
    assert {_stored_dtype(p) for p in frozen.values()} == {ttnn.DataType.BFLOAT16}
    assert {p.get_grad_accumulation_dtype() for p in frozen.values()} == {None}
    assert ttml.autograd.AutoContext.get_instance().get_gradient_reduce_dtype() == ttnn.DataType.FLOAT32


@pytest.mark.requires_device
def test_without_precision_nothing_changes(gradient_reduce_default):
    """precision=None applies nothing: the parameters keep the model's dtype and a reduce default set before stays."""
    gradient_reduce_default(ttnn.DataType.FLOAT32)

    trainer = SFTTrainer(_toy_llama(), [], None, _config(), optimizer={"type": "AdamW", "lr": 1e-3})

    parameters = trainer.model.parameters()
    assert {_stored_dtype(p) for p in parameters.values()} == {ttnn.DataType.BFLOAT16}
    assert {p.get_grad_accumulation_dtype() for p in parameters.values()} == {None}
    assert ttml.autograd.AutoContext.get_instance().get_gradient_reduce_dtype() == ttnn.DataType.FLOAT32


@pytest.mark.requires_device
def test_rejects_precision_with_a_ready_optimizer(expect_error):
    """A ready optimizer's state is already sized for the parameters' current dtype, so precision can't follow it."""
    model = _toy_llama()
    optimizer = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": 1e-3}, model.parameters())

    with expect_error(ValueError, "this optimizer already exists"):
        SFTTrainer(model, [], None, _config(), optimizer=optimizer, precision=FP32_PRECISION)


class _StepLosses(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_step_end(self, trainer, step, loss, lr):
        self.losses.append(loss)


@pytest.mark.requires_device
def test_trains_fp32_master_weights_with_fp32_gradient_accumulation(gradient_reduce_default):
    """Two steps of two micro-batches each with fused AdamW: the gradients add up in fp32 from the second micro-batch,
    AdamW reads them as they are, and the fp32 master weights move."""
    rng = np.random.default_rng(0)
    examples = []
    for _ in range(8):
        tokens = rng.integers(1, VOCAB_SIZE, SEQ_LEN).tolist()
        examples.append({"input_ids": tokens, "labels": [-100] * 4 + tokens[4:]})
    loader = InMemoryDataloader(examples, lambda batch: sft_collate_fn(batch, SEQ_LEN, 0), 2)
    model = _toy_llama()
    before = {name: p.to_numpy(ttnn.DataType.FLOAT32) for name, p in model.parameters().items()}
    losses = _StepLosses()

    SFTTrainer(
        model,
        loader,
        None,
        _config(max_steps=2, gradient_accumulation_steps=2),
        optimizer={"type": "AdamW", "lr": 1e-3},
        callbacks=[losses],
        precision=FP32_PRECISION,
    ).train()

    assert len(losses.losses) == 2 and all(np.isfinite(losses.losses)), losses.losses
    parameters = model.parameters()
    assert {_stored_dtype(p) for p in parameters.values()} == {ttnn.DataType.FLOAT32}
    assert {p.get_grad().dtype for p in parameters.values()} == {ttnn.DataType.FLOAT32}
    moved = [
        name for name, p in parameters.items() if not np.array_equal(p.to_numpy(ttnn.DataType.FLOAT32), before[name])
    ]
    assert len(moved) == len(parameters), f"unchanged: {sorted(set(parameters) - set(moved))}"
    ttml.autograd.AutoContext.get_instance().reset_graph()
