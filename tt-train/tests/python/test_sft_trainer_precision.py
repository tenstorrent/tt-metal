# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SFTTrainer applies the YAML's ``precision`` block before it builds the optimizer from a dict or the default."""

import logging

import ml_dtypes
import numpy as np
import pytest

import ttnn
import ttml
from ttml.common.utils import apply_precision
from ttml.datasets.hf_dataloader import InMemoryDataloader, sft_collate_fn
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig
from ttml.modules import LoraConfig
from ttml.trainers import SFTConfig, SFTTrainer, TrainerCallback

FP32_PRECISION = {"master_dtype": "float32", "reduce_dtype": "float32"}
LORA = LoraConfig(rank=4, alpha=8.0, target_modules=["qkv_linear"])
VOCAB_SIZE = 64
SEQ_LEN = 32
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


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


def _loader():
    rng = np.random.default_rng(0)
    examples = []
    for _ in range(8):
        tokens = rng.integers(1, VOCAB_SIZE, SEQ_LEN).tolist()
        examples.append({"input_ids": tokens, "labels": [-100] * 4 + tokens[4:]})
    return InMemoryDataloader(examples, lambda batch: sft_collate_fn(batch, SEQ_LEN, 0), 2)


def _dtypes(parameters) -> set:
    return {p.get_value(NATIVE).dtype for p in parameters.values()}


def _split(model):
    parameters = model.parameters()
    trainable = {name: p for name, p in parameters.items() if p.get_requires_grad()}
    frozen = {name: p for name, p in parameters.items() if not p.get_requires_grad()}
    return trainable, frozen


def _reduce_default():
    return ttml.autograd.AutoContext.get_instance().get_gradient_reduce_dtype()


@pytest.mark.requires_device
def test_applies_precision_before_building_the_optimizer(gradient_reduce_default, caplog):
    """With LoRA the trainable adapters get fp32 master weights and the frozen base stays bf16, so the trainer has
    nothing to warn about; the reduce dtype becomes the stored default the gradient sums read."""
    with caplog.at_level(logging.WARNING, logger="ttml.trainers.sft_trainer"):
        trainer = SFTTrainer(
            _toy_llama(),
            [],
            None,
            _config(),
            optimizer={"type": "AdamW", "lr": 1e-3},
            peft_config=LORA,
            precision=FP32_PRECISION,
        )

    trainable, frozen = _split(trainer.model)
    assert trainable and frozen, "premise: LoRA leaves a frozen base and trainable adapters"
    assert _dtypes(trainable) == {ttnn.DataType.FLOAT32}
    assert _dtypes(frozen) == {ttnn.DataType.BFLOAT16}
    assert _reduce_default() == ttnn.DataType.FLOAT32
    assert "frozen parameter" not in caplog.text


@pytest.mark.requires_device
def test_none_means_no_block(gradient_reduce_default):
    """precision=None is applied as a config without the block: the parameters keep the model's dtype and no reduce
    dtype is stored, so one stored earlier in the process is cleared, as create_optimizer does."""
    gradient_reduce_default(ttnn.DataType.FLOAT32)

    trainer = SFTTrainer(_toy_llama(), [], None, _config(), optimizer={"type": "AdamW", "lr": 1e-3})

    assert _dtypes(trainer.model.parameters()) == {ttnn.DataType.BFLOAT16}
    assert _reduce_default() is None


@pytest.mark.requires_device
def test_rejects_precision_with_a_ready_optimizer(expect_error):
    """A ready optimizer's state is already sized for the parameters' current dtype, so precision can't follow it."""
    model = _toy_llama()
    optimizer = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": 1e-3}, model.parameters())

    with expect_error(ValueError, "this optimizer already exists"):
        SFTTrainer(model, [], None, _config(), optimizer=optimizer, precision=FP32_PRECISION)


@pytest.mark.requires_device
def test_warns_when_a_frozen_parameter_is_stored_in_fp32(gradient_reduce_default, caplog):
    """Applying the block before LoraModel wraps the model converts the base too, and LoraModel keeps those tensors, so
    the frozen base would cost fp32 plus its bf16 view: the trainer says so."""
    model = _toy_llama()
    apply_precision(model, FP32_PRECISION)

    with caplog.at_level(logging.WARNING, logger="ttml.trainers.sft_trainer"):
        trainer = SFTTrainer(
            model,
            [],
            None,
            _config(),
            optimizer={"type": "AdamW", "lr": 1e-3},
            peft_config=LORA,
            precision=FP32_PRECISION,
        )

    _, frozen = _split(trainer.model)
    assert _dtypes(frozen) == {ttnn.DataType.FLOAT32}, "premise: the base was converted before the wrap"
    assert "frozen parameter(s)" in caplog.text and "stored in fp32" in caplog.text


class _StepLosses(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_step_end(self, trainer, step, loss, lr):
        self.losses.append(loss)


@pytest.mark.requires_device
def test_trains_fp32_master_weights_with_fp32_gradient_accumulation(gradient_reduce_default):
    """Two steps of two micro-batches each with fused AdamW: the gradients add up in fp32 from the second micro-batch,
    AdamW reads them as they are, and every fp32 master weight moves."""
    model = _toy_llama()
    before = {name: p.to_numpy(ttnn.DataType.FLOAT32) for name, p in model.parameters().items()}
    losses = _StepLosses()

    SFTTrainer(
        model,
        _loader(),
        None,
        _config(max_steps=2, gradient_accumulation_steps=2),
        optimizer={"type": "AdamW", "lr": 1e-3},
        callbacks=[losses],
        precision=FP32_PRECISION,
    ).train()

    assert len(losses.losses) == 2 and all(np.isfinite(losses.losses)), losses.losses
    parameters = model.parameters()
    assert _dtypes(parameters) == {ttnn.DataType.FLOAT32}
    assert {p.get_grad().dtype for p in parameters.values()} == {ttnn.DataType.FLOAT32}
    moved = [
        name for name, p in parameters.items() if not np.array_equal(p.to_numpy(ttnn.DataType.FLOAT32), before[name])
    ]
    assert len(moved) == len(parameters), f"unchanged: {sorted(set(parameters) - set(moved))}"
    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
def test_weights_loaded_after_the_trainer_is_built_keep_their_fp32_values(gradient_reduce_default):
    """Full fine-tuning from an fp32 file: build the trainer, which applies the block once, then load. What checkpoints
    and the Python loaders write with (``assign_numpy``) lands in the dtype a parameter is stored in, so fp32 master
    weights get the values unrounded, the stored reduce dtype stays, and the optimizer built before the load steps the
    loaded weights."""
    model = _toy_llama()
    losses = _StepLosses()
    trainer = SFTTrainer(
        model,
        _loader(),
        None,
        _config(),
        optimizer={"type": "AdamW", "lr": 1e-3},
        callbacks=[losses],
        precision=FP32_PRECISION,
    )

    rng = np.random.default_rng(1)
    loaded = {}
    for name, parameter in model.parameters().items():
        loaded[name] = (0.02 * rng.standard_normal(parameter.shape())).astype(np.float32)
        ttml.autograd.assign_numpy(parameter, loaded[name])
    some = next(iter(loaded.values()))
    assert not np.array_equal(some, some.astype(ml_dtypes.bfloat16).astype(np.float32)), "premise: not bf16 values"
    for name, parameter in model.parameters().items():
        np.testing.assert_array_equal(parameter.to_numpy(ttnn.DataType.FLOAT32), loaded[name], err_msg=name)
    assert _reduce_default() == ttnn.DataType.FLOAT32

    trainer.train()

    assert len(losses.losses) == 1 and np.isfinite(losses.losses[0]), losses.losses
    parameters = model.parameters()
    assert _dtypes(parameters) == {ttnn.DataType.FLOAT32}
    moved = [
        name for name, p in parameters.items() if not np.array_equal(p.to_numpy(ttnn.DataType.FLOAT32), loaded[name])
    ]
    assert len(moved) == len(parameters), f"unchanged: {sorted(set(parameters) - set(moved))}"
    ttml.autograd.AutoContext.get_instance().reset_graph()
