# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for ``GRPOTrainer`` on a Tenstorrent device.

The goal is to exercise the full GRPO loop (rollout with per-token
``log pi_old`` -> reward -> advantages -> train-mode forward+backward ->
optimizer step -> callbacks) in the smallest configuration that still
produces a non-degenerate gradient update.

Speed strategy:
  * Tiny random-init Llama (1 layer, hidden=64, head_dim=32).
  * Skip the HuggingFace weight download: ``_download_hf_repo`` in
    ``ttml.trainers.grpo_trainer.grpo_ttml_model`` is patched to return a directory holding only the tiny
    model's ``config.json``, and ``load_from_safetensors`` to do nothing, so the model keeps its random init.
  * ``max_completion_length=4`` so autoregressive generation is cheap.
  * Two optimizer steps (``num_iterations=1``,
    ``prompts_to_train=4 * gradient_accumulation_steps``). On this single device
    ``per_device_train_batch_size=4`` and ``num_generations=2`` give a
    per-micro-batch prompt count of ``4 * 1 / 2 = 2``, so the generation batch
    has ``gradient_accumulation_steps`` micro-batches of 2 prompts.
  * ``num_generations=2`` is the minimum that yields a non-zero advantage
    (group of 1 -> mean == reward -> advantage == 0 -> loss == 0).

Note on CI: the tokenizer load uses ``meta-llama/Llama-3.2-1B-Instruct``,
which is a gated HuggingFace repo. To run this in CI without an
``HF_TOKEN`` secret, swap ``HF_MODEL_ID`` below for a non-gated mirror
(e.g. ``unsloth/Llama-3.2-1B-Instruct``).
"""

from __future__ import annotations

import collections
import contextlib
import functools
import json
import os
import queue
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
import torch
import yaml

import ttml
import ttnn
from datasets import Dataset
from safetensors.numpy import save_file
from transformers import AutoTokenizer, LlamaConfig as HFLlamaConfig, LlamaForCausalLM

from ttml.common.config import DeviceConfig, TransformerConfig, load_config
from ttml.common.utils import build_mesh, repeat_interleave
from ttml.models.llama import Llama, LlamaConfig, load_from_safetensors
from ttml.modules import RunMode
from ttml.trainers import GRPOConfig, GRPOTrainer, RolloutBatch, RolloutSampler, TrainerCallback, get_grpo_config
from ttml.trainers.grpo_trainer import (
    RemoteRolloutConfig,
    grpo_ttml_model,
    layout_microbatch,
    place_old_nlog_probs,
    save_checkpoint,
)
from ttml.trainers.grpo_trainer import grpo_trainer as grpo_trainer_module
from ttml.trainers.grpo_trainer.remote_rollout import (
    async_rollout,
    mpi_rollout,
    rollout_queue,
    threaded_weight_bridge,
)
from ttml.trainers.grpo_trainer.remote_rollout.async_remote_rollout_batch_source import AsyncRemoteRolloutBatchSource
from ttml.trainers.grpo_trainer.remote_rollout.rollout_queue import RolloutQueue
from ttml.trainers.grpo_trainer.remote_rollout.sync_remote_rollout_batch_source import SyncRemoteRolloutBatchSource
from ttml.trainers.grpo_trainer.remote_rollout.transport import _torch_save_bytes, send_frame
from ttml.trainers.grpo_trainer.remote_rollout.weight_bridge import TTML_RANK, TTT_RANK
from ttml.trainers.grpo_trainer.rollout_batch_source import (
    InProcessRolloutBatchSource,
    RolloutBatchSource,
    ScoredRolloutBatch,
    pad_logprobs,
    score_rollout,
    validate_rollout_batch,
)
from ttml.trainers.grpo_trainer.ttml_rollout_sampler import TTMLRolloutSampler

from remote_rollout._loopback import LoopbackContext as _LoopbackContext

HF_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"  # not gated


# The model setup reads the architecture from the checkpoint's config.json; these only set the run.
TINY_TRANSFORMER_CONFIG = TransformerConfig(
    {
        "transformer_config": {
            "model_type": "llama",
            "dropout_prob": 0.0,
            "max_sequence_length": 128,
            "runner_type": "memory_efficient",
        }
    }
)

LLAMA_1B_TRANSFORMER_CONFIG = TransformerConfig(
    {
        "transformer_config": {
            "model_type": "llama",
            "dropout_prob": 0.0,
            "max_sequence_length": 1024,
            "runner_type": "memory_efficient",
        }
    }
)

# Llama-3.2-1B shrunk to one narrow layer.
TINY_HF_CONFIG = {
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 128256,
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": True,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
    },
}

DEVICE_CONFIG = DeviceConfig(
    {
        "device_config": {
            "enable_ddp": False,
            "mesh_shape": [1, 1],
        }
    }
)

CAPITALS_SYSTEM_PROMPT = (
    "You are a precise geography assistant.\n"
    "Given a country, reply with exactly one word: its capital city in English.\n"
    "Feel free to describe the capital city or the country."
)


@pytest.fixture(autouse=True)
def _reuse_open_device(monkeypatch):
    """Override ``grpo_ttml_model.open_device`` to reuse the already-open
    AutoContext device instead of calling ``open_device`` again.

    Other tests in ``tests/python/`` lazily open the AutoContext device on
    first tensor use and never close it. When pytest collects this file
    alongside them, the device is already open by the time we get here, so
    the default ``open_device`` would trip ``open_device was called after
    the device was created``. Reusing the live device sidesteps the issue
    without leaking device-management code into the test body.
    """
    current_device = lambda *args: ttml.autograd.AutoContext.get_instance().get_device()  # noqa: E731
    monkeypatch.setattr(grpo_ttml_model, "open_device", current_device)


class _RecordingCallback(TrainerCallback):
    """Records hook invocations so we can assert the trainer drove them."""

    def __init__(self) -> None:
        self.train_begin = 0
        self.before_step = 0
        self.step_end = 0
        self.train_end = 0
        self.first_step_metrics: dict | None = None
        self.last_step_metrics: dict | None = None
        self.final_step_time_s: float | None = None

    def on_train_begin(self, trainer):
        self.train_begin += 1

    def on_before_optimizer_step(self, trainer):
        self.before_step += 1

    def on_step_end(self, trainer, step, **kwargs):
        self.step_end += 1
        self.last_step_metrics = {"step": step, **kwargs}
        if self.first_step_metrics is None:
            self.first_step_metrics = self.last_step_metrics

    def on_train_end(self, trainer):
        self.train_end += 1
        self.final_step_time_s = trainer.metrics.get("step_time_s")


def _patch_tiny_checkpoint(monkeypatch, checkpoint_dir, **config_overrides) -> str:
    """Point ``grpo_ttml_model`` at *checkpoint_dir*, holding only the tiny model's ``config.json``, and skip
    the weight load so the model keeps its random init.

    The download patch goes on ``_download_hf_repo``, which wraps ``huggingface_hub.snapshot_download``;
    that one is left alone because transformers also uses it to fetch the tokenizer this test loads.
    ``load_from_safetensors`` is bound with ``from ... import``, so it is patched on ``grpo_ttml_model``,
    not on ``ttml.models.llama``.
    """
    (checkpoint_dir / "config.json").write_text(json.dumps({**TINY_HF_CONFIG, **config_overrides}))
    monkeypatch.setattr(grpo_ttml_model, "_download_hf_repo", lambda model_source: str(checkpoint_dir))
    monkeypatch.setattr(grpo_ttml_model, "load_from_safetensors", lambda *args, **kwargs: None)
    return str(checkpoint_dir)


@pytest.fixture
def patch_llama_weight_loading(monkeypatch, tmp_path_factory):
    """Skip the HF download / safetensors load so the tiny model keeps random init; return its config directory."""
    return _patch_tiny_checkpoint(monkeypatch, tmp_path_factory.mktemp("tiny_llama"))


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "lr_scheduler_type, warmup_steps, use_clip_grad_norm, gradient_accumulation_steps, expected_first_lr_scale",
    [
        ("constant", 0, False, 1, 1.0),
        ("cosine", 0, True, 2, 1.0),
        ("constant_with_warmup", 2, False, 1, 0.5),
    ],
)
def test_grpo_trainer_one_step_smoke(
    patch_llama_weight_loading,
    tmp_path,
    lr_scheduler_type,
    warmup_steps,
    use_clip_grad_norm,
    gradient_accumulation_steps,
    expected_first_lr_scale,
):
    """Two GRPO optimizer steps on a tiny random-init Llama.

    Asserts the loop reaches every callback hook, produces finite metrics,
    and actually mutates model weights via ``optimizer.step``.
    """
    np.random.seed(0)

    grpo_cfg = GRPOConfig(
        epsilon=0.2,
        per_device_train_batch_size=4,
        num_iterations=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=1,
        output_dir=str(tmp_path),
        checkpointing=False,
        checkpoint_interval=999,
        prompts_to_train=4 * gradient_accumulation_steps,
        temperature=1.0,
        max_completion_length=4,
        num_generations=2,
        warmup_steps=warmup_steps,
        rollout_backend="ttml_in_process_sync",
        lr_scheduler_type=lr_scheduler_type,
        use_clip_grad_norm=use_clip_grad_norm,
    )

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    user_prompts = ["What is 1+1?", "Name a color.", "Name a fruit.", "What is 2+2?"]
    dataset = Dataset.from_dict(
        {
            "prompt": [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in user_prompts * 2
            ],
            "answer": ["2", "blue", "apple", "4"] * 2,
        }
    )

    reward_calls: list[list[float]] = []

    def reward_func(completions, **_kwargs):
        # Fixed per-position rewards keep advantages non-zero (group mean != reward)
        # without depending on what the random model actually generates.
        rewards = [float(i % 2) for i in range(len(completions))]
        reward_calls.append(rewards)
        return rewards

    optimizer_dict = {
        "type": "MorehAdamW",
        "lr": 1.0e-3,
        "beta1": 0.9,
        "beta2": 0.99,
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "kahan_summation": False,
    }

    recorder = _RecordingCallback()
    trainer = GRPOTrainer(
        transformer_config=TINY_TRANSFORMER_CONFIG,
        device_config=DEVICE_CONFIG,
        model_source=HF_MODEL_ID,
        dataset_func=lambda: dataset,
        config=grpo_cfg,
        reward_func=reward_func,
        optimizer_dict=optimizer_dict,
        callbacks=[recorder],
    )
    sampler = trainer.rollout_sampler
    assert isinstance(sampler, TTMLRolloutSampler)

    # Snapshot a single parameter so we can prove training mutated it.
    params = trainer.model.parameters()
    assert params, "tiny model should expose at least one parameter"
    snapshot_name, snapshot_param = next(iter(params.items()))
    before = snapshot_param.to_numpy(ttnn.DataType.FLOAT32).copy()

    trainer.train()

    assert trainer.model.get_run_mode() == RunMode.TRAIN, "the sampler must leave the shared model in train mode"
    assert sampler.weight_version == 1, "trainer should publish a new weight version after the first batch"

    assert recorder.train_begin == 1, "on_train_begin should fire exactly once"
    assert recorder.before_step == 2, "on_before_optimizer_step should fire once per step"
    assert recorder.step_end == 2, "on_step_end should fire once per step"
    assert recorder.train_end == 1, "on_train_end should fire exactly once"

    assert reward_calls, "reward_func was never invoked"
    # reward_func runs once per generation (effective) batch, which spans
    # gradient_accumulation_steps micro-batches of per_device_train_batch_size *
    # num_devices completions each. Query num_devices the same way
    # GRPOTrainer.train does so this holds for single- and multi-device meshes.
    num_devices = ttml.autograd.AutoContext.get_instance().get_device().get_num_devices()
    expected_completions = grpo_cfg.per_device_train_batch_size * num_devices * grpo_cfg.gradient_accumulation_steps
    assert (
        len(reward_calls[0]) == expected_completions
    ), f"expected {expected_completions} completions per batch, got {len(reward_calls[0])}"

    metrics = recorder.last_step_metrics
    assert metrics is not None
    assert recorder.first_step_metrics["lr"] == pytest.approx(
        optimizer_dict["lr"] * expected_first_lr_scale
    ), "first-step LR does not match the scheduler contract"
    for key in (
        "reward_mean",
        "reward_std",
        "mean_completion_len",
        "generation_time_s",
        "reward_time_s",
        "rollout_wait_s",
        "weight_sync_time_s",
        "logprob_mismatch_mean",
        "logprob_mismatch_max",
        "clip_frac",
    ):
        assert key in metrics, f"missing metric {key}"
        assert np.isfinite(metrics[key]), f"metric {key} is not finite: {metrics[key]}"

    step_time_s = recorder.final_step_time_s
    assert step_time_s is not None, "missing metric step_time_s"
    assert np.isfinite(step_time_s) and step_time_s > 0.0, f"step_time_s is not a positive duration: {step_time_s}"

    after = trainer.model.parameters()[snapshot_name].to_numpy(ttnn.DataType.FLOAT32)
    assert before.shape == after.shape
    assert not np.array_equal(before, after), (
        f"parameter {snapshot_name!r} was unchanged after one optimizer step; "
        "either backward did not run or the gradient was identically zero"
    )

    ttml.autograd.AutoContext.get_instance().reset_graph()


def _assert_same_parameters(a, b):
    pa, pb = a.parameters(), b.parameters()
    assert sorted(pa) == sorted(pb)
    for name in pa:
        np.testing.assert_array_equal(
            pa[name].to_numpy(ttnn.DataType.FLOAT32), pb[name].to_numpy(ttnn.DataType.FLOAT32), err_msg=name
        )


@pytest.mark.requires_device
def test_weights_ref_hf_dict_matches_the_transfer_format(patch_llama_weight_loading):
    """Checks the contract in LLAMA_WEIGHT_TRANSFER.md section 3 on a tiny Llama."""
    model, _ = grpo_ttml_model.setup_ttml_model(TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, HF_MODEL_ID)
    cfg = model.config
    with torch.device("meta"):
        hf_model = LlamaForCausalLM(
            HFLlamaConfig(
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                num_hidden_layers=cfg.num_hidden_layers,
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=cfg.num_key_value_heads,
                tie_word_embeddings=True,
            )
        )
    hf_shapes = {key: tuple(value.shape) for key, value in hf_model.state_dict().items()}
    params = model.parameters()

    exported = grpo_ttml_model.weights_ref_hf_dict(model)

    assert exported.keys() == hf_shapes.keys()
    assert len(exported) == 3 + 9 * cfg.num_hidden_layers
    live_buffers = {param.get_value().buffer_address() for param in params.values()}
    for key, tensor in exported.items():
        assert isinstance(tensor, ttnn.Tensor), key
        assert tensor.dtype == ttnn.bfloat16, key
        assert tensor.layout == ttnn.TILE_LAYOUT, key
        assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG, key
        assert all(isinstance(p, ttnn.PlacementReplicate) for p in tensor.tensor_topology().placements()), key
        hf_shape = hf_shapes[key]
        assert tuple(tensor.shape) == (1,) * (4 - len(hf_shape)) + hf_shape, key
        is_slice = key.endswith(
            ("q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight")
        )
        assert (tensor.buffer_address() in live_buffers) != is_slice, key
    assert exported["model.embed_tokens.weight"].buffer_address() == exported["lm_head.weight"].buffer_address()

    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
def test_save_checkpoint_round_trip_llama(patch_llama_weight_loading, tmp_path):
    model, tokenizer = grpo_ttml_model.setup_ttml_model(TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, HF_MODEL_ID)
    # Distinct values in every parameter, so a fresh init cannot pass for a resumed one.
    rng = np.random.default_rng(0)
    for param in model.parameters().values():
        ttml.autograd.assign_numpy(param, rng.standard_normal(param.shape()).astype(np.float32))
    save_checkpoint(model, 1, str(tmp_path), tokenizer=tokenizer, model_source=patch_llama_weight_loading)

    ckpt_dir = str(tmp_path / "checkpoints" / "grpo_step_1")
    loaded, _ = grpo_ttml_model.setup_ttml_model(TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, ckpt_dir)
    _assert_same_parameters(model, loaded)

    ttml.autograd.AutoContext.get_instance().reset_graph()


FP32_PRECISION = {"master_dtype": "float32", "reduce_dtype": "float32"}


def _stored_dtypes(model) -> set:
    return {p.get_value(ttml.autograd.PreferredPrecision.NATIVE).dtype for p in model.parameters().values()}


@pytest.mark.requires_device
def test_setup_ttml_model_resumes_fp32_master_weights_bit_for_bit(
    patch_llama_weight_loading, tmp_path, gradient_reduce_default
):
    """The precision block is applied right after the model is built, before a trainer checkpoint loads, so fp32
    master weights resume exactly. Without the block the same checkpoint loads into bf16 parameters, rounded."""
    model, tokenizer = grpo_ttml_model.setup_ttml_model(
        TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, HF_MODEL_ID, precision=FP32_PRECISION
    )
    assert _stored_dtypes(model) == {ttnn.DataType.FLOAT32}
    rng = np.random.default_rng(0)
    for param in model.parameters().values():
        ttml.autograd.assign_numpy(param, rng.standard_normal(param.shape()).astype(np.float32))
    save_checkpoint(model, 1, str(tmp_path), tokenizer=tokenizer, model_source=patch_llama_weight_loading)
    ckpt_dir = str(tmp_path / "checkpoints" / "grpo_step_1")

    resumed, _ = grpo_ttml_model.setup_ttml_model(
        TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, ckpt_dir, precision=FP32_PRECISION
    )
    assert _stored_dtypes(resumed) == {ttnn.DataType.FLOAT32}
    _assert_same_parameters(model, resumed)

    rounded, _ = grpo_ttml_model.setup_ttml_model(TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, ckpt_dir)
    assert _stored_dtypes(rounded) == {ttnn.DataType.BFLOAT16}
    name, saved = next(iter(model.parameters().items()))
    assert not np.array_equal(
        rounded.parameters()[name].to_numpy(ttnn.DataType.FLOAT32), saved.to_numpy(ttnn.DataType.FLOAT32)
    ), "premise: without the block the checkpoint's fp32 values are rounded"

    ttml.autograd.AutoContext.get_instance().reset_graph()


class _GradientDtypes(TrainerCallback):
    """Records the gradients' dtypes right before each optimizer step."""

    def __init__(self) -> None:
        self.per_step: list[set] = []

    def on_before_optimizer_step(self, trainer):
        self.per_step.append(
            {p.get_grad().dtype for p in trainer.model.parameters().values() if p.is_grad_initialized()}
        )


@pytest.mark.requires_device
def test_grpo_trainer_trains_fp32_master_weights_with_fp32_gradient_accumulation(
    patch_llama_weight_loading, tmp_path, gradient_reduce_default
):
    """The config's precision block reaches the model before its weights load: fused AdamW then steps fp32 master
    weights on gradients added up in fp32 over two micro-batches, and every parameter moves."""
    grpo_cfg = GRPOConfig(
        epsilon=0.2,
        per_device_train_batch_size=4,
        num_iterations=1,
        gradient_accumulation_steps=2,
        logging_steps=1,
        output_dir=str(tmp_path),
        checkpointing=False,
        checkpoint_interval=999,
        prompts_to_train=8,
        temperature=1.0,
        max_completion_length=4,
        num_generations=2,
        warmup_steps=0,
        rollout_backend="ttml_in_process_sync",
        lr_scheduler_type="constant",
        precision=FP32_PRECISION,
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    prompts = ["What is 1+1?", "Name a color.", "Name a fruit.", "What is 2+2?"] * 2
    dataset = Dataset.from_dict(
        {
            "prompt": [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in prompts
            ],
            "answer": ["2", "blue", "apple", "4"] * 2,
        }
    )
    gradients = _GradientDtypes()

    trainer = GRPOTrainer(
        transformer_config=TINY_TRANSFORMER_CONFIG,
        device_config=DEVICE_CONFIG,
        model_source=HF_MODEL_ID,
        dataset_func=lambda: dataset,
        config=grpo_cfg,
        reward_func=lambda completions, **_: [float(i % 2) for i in range(len(completions))],
        optimizer_dict={"type": "AdamW", "lr": 1.0e-3},
        callbacks=[gradients],
    )
    assert _stored_dtypes(trainer.model) == {ttnn.DataType.FLOAT32}
    assert ttml.autograd.AutoContext.get_instance().get_gradient_reduce_dtype() == ttnn.DataType.FLOAT32
    before = {name: p.to_numpy(ttnn.DataType.FLOAT32) for name, p in trainer.model.parameters().items()}

    trainer.train()

    assert gradients.per_step and all(
        dtypes == {ttnn.DataType.FLOAT32} for dtypes in gradients.per_step
    ), gradients.per_step
    parameters = trainer.model.parameters()
    assert _stored_dtypes(trainer.model) == {ttnn.DataType.FLOAT32}
    moved = [
        name for name, p in parameters.items() if not np.array_equal(p.to_numpy(ttnn.DataType.FLOAT32), before[name])
    ]
    assert len(moved) == len(parameters), f"unchanged: {sorted(set(parameters) - set(moved))}"

    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_grpo_trainer_rejects_a_whole_config_as_optimizer_dict(expect_error):
    """``create_optimizer`` applies a whole config's precision block, which here would run after the weights loaded.
    The trainer takes that block from ``GRPOConfig.precision`` and only the optimizer's own dict as
    ``optimizer_dict``."""
    whole_config = {
        "training_config": {"optimizer": {"type": "AdamW", "lr": 1.0e-3}, "precision": {"master_dtype": "float32"}}
    }
    with expect_error(ValueError, "must be the optimizer's own dict"):
        GRPOTrainer(
            transformer_config=None,
            device_config=None,
            model_source="unused",
            dataset_func=lambda: None,
            config=None,
            optimizer_dict=whole_config,
        )


@pytest.mark.parametrize(
    "precision, error, message",
    [
        ({"master_dtpye": "float32"}, RuntimeError, "Unknown key precision.master_dtpye"),
        ({"reduce_dtype": "float16"}, RuntimeError, 'precision.reduce_dtype must be "bfloat16" or "float32"'),
        ("float32", TypeError, "precision must be a dict"),
    ],
)
def test_grpo_config_checks_its_precision_block(expect_error, precision, error, message):
    """A typo in the precision block stops the config, before a trainer opens a device and builds a model."""
    with expect_error(error, message):
        GRPOConfig(**_GRPO_CONFIG_FIELDS, rollout_backend="ttml_in_process_sync", precision=precision)


def test_get_grpo_config_reads_the_precision_block_of_the_training_config(expect_error):
    """The block has one place in the YAML, ``training_config.precision`` next to the optimizer. ``get_grpo_config``
    puts it into ``GRPOConfig.precision``, which the trainer applies and wandb logs with the rest of the config."""
    raw = load_config("training_configs/grpo_boolq_llama_1b_ddp_4dev.yaml")
    assert get_grpo_config(raw).precision is None  # the shipped config has no block

    raw["training_config"]["precision"] = dict(FP32_PRECISION)
    cfg = get_grpo_config(raw)
    assert cfg.precision == FP32_PRECISION
    logged = grpo_trainer_module._build_wandb_config(
        SimpleNamespace(config=cfg, model_source="unused", optimizer_dict=raw["training_config"]["optimizer"])
    )
    assert logged["precision"] == FP32_PRECISION and logged["optimizer"] == raw["training_config"]["optimizer"]

    raw["training_config"]["precision"] = {"master_dtpye": "float32"}
    with expect_error(RuntimeError, "Unknown key precision.master_dtpye"):
        get_grpo_config(raw)

    del raw["training_config"]["precision"]
    raw["training_config"]["grpo_config"]["precision"] = dict(FP32_PRECISION)
    with expect_error(ValueError, "belongs under training_config.precision"):
        get_grpo_config(raw)


@pytest.mark.requires_device
def test_policy_forward_is_causal(patch_llama_weight_loading):
    """A change to the last token changes no earlier -log pi; full attention would change them."""
    model, _ = grpo_ttml_model.setup_ttml_model(TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, HF_MODEL_ID)
    policy = SimpleNamespace(
        model=model, config=SimpleNamespace(temperature=1.0), _batch_layout=SimpleNamespace(mapper=None)
    )
    inputs = np.tile(np.arange(1, 33, dtype=np.uint32), (2, 1))
    inputs[1, -1] = 7

    nlog = GRPOTrainer._compute_policy_nlog_probs(policy, inputs, inputs.copy()).to_numpy(ttnn.DataType.FLOAT32)

    np.testing.assert_array_equal(nlog[0, :-1], nlog[1, :-1])
    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
def test_setup_ttml_model_rejects_a_checkpoint_missing_a_parameter(patch_llama_weight_loading, tmp_path, expect_error):
    """A checkpoint saved under older parameter names must not resume with those weights at init."""
    model, tokenizer = grpo_ttml_model.setup_ttml_model(TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, HF_MODEL_ID)
    save_checkpoint(model, 1, str(tmp_path), tokenizer=tokenizer, model_source=patch_llama_weight_loading)
    checkpoint_dir = tmp_path / "checkpoints" / "grpo_step_1"
    params = ttml.NamedParameters()
    for name, param in model.parameters().items():
        if not name.endswith("/qkv_linear/weight"):
            params[name] = param
    ttml.checkpointing.save_checkpoint(str(checkpoint_dir / grpo_ttml_model.CHECKPOINT_FILE), model_params=params)

    with expect_error(RuntimeError, r"param\(s\) left at init"):
        grpo_ttml_model.setup_ttml_model(TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, str(checkpoint_dir))
    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
def test_rejects_a_tokenizer_larger_than_the_checkpoint_vocab(monkeypatch, tmp_path, expect_error):
    """A token id past the checkpoint's vocab has no embedding row, so the model setup refuses it up front."""
    _patch_tiny_checkpoint(monkeypatch, tmp_path, vocab_size=128000)  # the tokenizer has 128256 tokens
    with expect_error(ValueError, "the tokenizer has 128256 tokens but the checkpoint embeds 128000"):
        grpo_ttml_model.setup_ttml_model(TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, HF_MODEL_ID)


@pytest.mark.requires_device
def test_never_samples_a_checkpoint_row_past_the_tokenizer(monkeypatch, tmp_path):
    """Rows past the tokenizer decode to nothing, so the logits mask covers them. Here every such row outscores
    the real ones, so one left unmasked gets sampled."""
    _patch_tiny_checkpoint(monkeypatch, tmp_path, vocab_size=128256 + 8, tie_word_embeddings=False)
    model, tokenizer = grpo_ttml_model.setup_ttml_model(TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, HF_MODEL_ID)
    num_tokens = len(tokenizer)
    assert model.config.vocab_size > num_tokens, "premise: the checkpoint has rows past the tokenizer"

    lm_head = np.zeros((model.padded_vocab_size, model.config.hidden_size), np.float32)
    lm_head[num_tokens:] = np.random.default_rng(0).normal(0.0, 4.0, lm_head[num_tokens:].shape)
    ttml.autograd.assign_numpy(model.fc.weight.tensor, lm_head)

    batch = TTMLRolloutSampler(model, tokenizer, 8, 1.0, 1).generate([[1, 2, 3, 4]] * 4)

    assert all(token < num_tokens for completion in batch.completions for token in completion), batch.completions
    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_save_checkpoint_header_records_given_learning_rate(monkeypatch, tmp_path):
    headers = []
    monkeypatch.setattr(ttml.checkpointing, "save_checkpoint", lambda path, header, **_: headers.append(header))
    optimizer = SimpleNamespace(get_lr=lambda: 0.75)

    save_checkpoint(SimpleNamespace(parameters=lambda: {}), 2, str(tmp_path), optimizer=optimizer, learning_rate=0.5)

    assert headers[0]["learning_rate"] == 0.5


def test_setup_records_num_generation_batches(monkeypatch):
    class _SetupTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

    monkeypatch.setattr(grpo_trainer_module, "create_optimizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(grpo_trainer_module, "_build_lr_scheduler", lambda *args, **kwargs: None)

    trainer = object.__new__(GRPOTrainer)
    trainer.config = GRPOConfig(**{**_HOST_GRPO_CONFIG, "prompts_to_train": 8})
    trainer.model = SimpleNamespace(parameters=lambda: {})
    trainer.tokenizer = _SetupTokenizer()
    trainer._batch_layout = SimpleNamespace(num_shards=1)
    trainer.optimizer_dict = {}
    trainer.callbacks = []
    trainer.reward_funcs = [lambda completions, **_: [0.0] * len(completions)]
    trainer.rollout_sampler = _EchoSampler()
    trainer._rollout_client = None

    dataset = Dataset.from_dict({"prompt": ["hello"] * trainer.config.prompts_to_train})
    trainer._setup(dataset)

    assert trainer._num_generation_batches == 4


def _save_old_format_checkpoint(directory):
    save_file({"Llama/fc/weight": np.zeros((2, 2), np.float32)}, str(directory / "model.safetensors"))


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_setup_ttml_model_rejects_old_checkpoint_format(tmp_path, expect_error, model_type):
    _save_old_format_checkpoint(tmp_path)
    transformer_config = TransformerConfig({"transformer_config": {"model_type": model_type}})
    with expect_error(ValueError, "old GRPO checkpoint format"):
        grpo_ttml_model.setup_ttml_model(transformer_config, DEVICE_CONFIG, str(tmp_path))


@pytest.mark.requires_device
def test_llama_strict_load_raises_on_unloaded_parameters(tmp_path, expect_error):
    _save_old_format_checkpoint(tmp_path)
    model = Llama(LlamaConfig.from_hf(TINY_HF_CONFIG, max_position_embeddings=128))
    with expect_error(RuntimeError, "the checkpoint lacks tensors the model needs"):
        load_from_safetensors(model, tmp_path)


def test_layout_microbatch_aligns_old_logprobs_mask_and_targets():
    """pi_old columns, mask columns and target tokens refer to the same
    completion tokens on ragged prompts/completions, with pi_old sign-flipped.
    """
    pad = 0
    prompts = [[11, 12, 13], [21, 22], [31, 32, 33, 34, 35]]
    completions = [[101, 102], [201, 202, 203, 204], [301]]
    max_completion_length = 6
    logprobs = np.zeros((len(prompts), max_completion_length), dtype=np.float32)
    for r, c in enumerate(completions):
        logprobs[r, : len(c)] = -np.arange(1, len(c) + 1, dtype=np.float32) - 10.0 * r

    inputs, targets, mask, Tp = layout_microbatch(prompts, completions, pad)
    old = place_old_nlog_probs(prompts, completions, logprobs, Tp)

    assert Tp == 32
    assert inputs.shape == targets.shape == mask.shape == old.shape == (len(prompts), Tp)
    for r, (p, c) in enumerate(zip(prompts, completions)):
        seq = p + c
        L = len(seq) - 1
        np.testing.assert_array_equal(inputs[r, :L], seq[:-1])
        np.testing.assert_array_equal(targets[r, :L], seq[1:])
        assert np.all(inputs[r, L:] == pad) and np.all(targets[r, L:] == pad)

        cols = np.flatnonzero(mask[r])
        np.testing.assert_array_equal(cols, np.arange(len(p) - 1, len(p) - 1 + len(c)))
        np.testing.assert_array_equal(targets[r, cols], c)
        np.testing.assert_array_equal(old[r, cols], -logprobs[r, : len(c)])
        assert np.all(old[r, cols] > 0.0)
        assert np.all(np.delete(old[r], cols) == 0.0)


_GRPO_CONFIG_FIELDS = dict(
    epsilon=0.2,
    per_device_train_batch_size=4,
    num_iterations=1,
    gradient_accumulation_steps=1,
    logging_steps=1,
    output_dir="",
    checkpointing=False,
    checkpoint_interval=1,
    prompts_to_train=2,
    temperature=1.0,
    max_completion_length=4,
    num_generations=2,
    warmup_steps=0,
)


def test_grpo_config_requires_rollout_backend(expect_error):
    with expect_error(TypeError, "rollout_backend"):
        GRPOConfig(**_GRPO_CONFIG_FIELDS)


def test_grpo_config_rejects_unknown_rollout_backend(expect_error):
    with expect_error(ValueError, "'rollout_backend' must be one of"):
        GRPOConfig(**_GRPO_CONFIG_FIELDS, rollout_backend="vllm")


def test_grpo_config_rejects_negative_max_staleness(expect_error):
    with expect_error(ValueError, "'max_staleness' must be >= 0"):
        GRPOConfig(**_GRPO_CONFIG_FIELDS, rollout_backend="ttml_in_process_sync", max_staleness=-1)


def test_grpo_config_max_staleness_defaults_to_weight_sync_every():
    cfg = _remote_grpo_config(rollout_backend="ttt_remote_async", weight_sync_every=2, num_iterations=1)
    assert cfg.max_staleness == 2


def test_get_grpo_config_reads_rollout_backend():
    cfg = get_grpo_config(
        {"training_config": {"grpo_config": {**_GRPO_CONFIG_FIELDS, "rollout_backend": "ttml_in_process_sync"}}}
    )
    assert cfg.rollout_backend == "ttml_in_process_sync"


_TRAINING_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "training_configs"


@pytest.mark.parametrize("path", sorted(_TRAINING_CONFIGS.glob("grpo_*.yaml")), ids=lambda path: path.name)
def test_get_grpo_config_reads_the_committed_grpo_configs(path):
    assert get_grpo_config(yaml.safe_load(path.read_text())).rollout_backend in grpo_trainer_module.ROLLOUT_BACKENDS


def test_layout_microbatch_rejects_short_prompt(expect_error):
    with expect_error(ValueError, "Prompt is too short"):
        layout_microbatch([[1]], [[2, 3]], pad_token=0)


def _to_capitals_chat_prompt(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": CAPITALS_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@pytest.mark.requires_device
@pytest.mark.slow
def test_capitals_one_by_one_equals_single_batch():
    """Greedy generation must give the same output one-by-one and batched.

    Loads the real Llama-3.2-1B-Instruct weights (no monkey-patch) and runs
    the same four prompts through ``TTMLRolloutSampler.generate`` twice:
    once one prompt at a time, once as a single batch. With temperature=0
    and ``completions_per_prompt=1`` the per-token log p must match up to the
    first different token (a bf16 near tie can flip the greedy pick); any drift
    indicates a batching / padding / mask bug in the generation path.
    """
    model, tokenizer = grpo_ttml_model.setup_ttml_model(LLAMA_1B_TRANSFORMER_CONFIG, DEVICE_CONFIG, HF_MODEL_ID)
    sampler = TTMLRolloutSampler(model, tokenizer, 256, 0.0, 1)

    user_prompts = [
        "The capital of France is",
        "The capital of Portugal is",
        "The capital of United Kingdom is",
        "The capital of Czech Republic is",
    ]
    prompts = [tokenizer.encode(_to_capitals_chat_prompt(tokenizer, p), add_special_tokens=False) for p in user_prompts]

    singles = [sampler.generate([prompt]) for prompt in prompts]
    batched = sampler.generate(prompts)
    assert len(batched.completions) == len(prompts)

    for b, single in enumerate(singles):
        assert len(single.completions) == 1
        one, many = single.completions[0], batched.completions[b]
        first_diff = next((i for i, (x, y) in enumerate(zip(one, many)) if x != y), min(len(one), len(many)))
        n = min(first_diff + 1, batched.logprobs.shape[1])
        np.testing.assert_allclose(
            batched.logprobs[b][:n],
            single.logprobs[0][:n],
            atol=0.1,
            err_msg=f"row {b}: single={tokenizer.decode(one)!r} batch={tokenizer.decode(many)!r}",
        )
    # Agreeing outputs also pass on a wrong but deterministic load; the right answers don't.
    for capital, single, many in zip(["Paris", "Lisbon", "London", "Prague"], singles, batched.completions):
        for completion in (single.completions[0], many):
            text = tokenizer.decode(completion, skip_special_tokens=False)
            assert capital in text, f"expected {capital!r} in {text!r}"

    ttml.autograd.AutoContext.get_instance().reset_graph()


_HOST_GRPO_CONFIG = dict(
    epsilon=0.2,
    per_device_train_batch_size=4,
    num_iterations=1,
    gradient_accumulation_steps=1,
    logging_steps=1,
    output_dir="",
    checkpointing=False,
    checkpoint_interval=999,
    prompts_to_train=2,
    temperature=1.0,
    max_completion_length=4,
    num_generations=2,
    warmup_steps=0,
    rollout_backend="ttml_in_process_sync",
)


@pytest.mark.parametrize("lr_scheduler_type", ["constant", "constant_with_warmup", "cosine", "cosine_with_min_lr"])
def test_grpo_config_accepts_trl_lr_scheduler_types(lr_scheduler_type):
    cfg = GRPOConfig(**_HOST_GRPO_CONFIG, lr_scheduler_type=lr_scheduler_type, min_lr_rate=0.1)
    assert cfg.lr_scheduler_type == lr_scheduler_type


@pytest.mark.parametrize(
    "override, message",
    [
        ({"lr_scheduler_type": "linear"}, "'lr_scheduler_type' must be one of"),
        ({"min_lr_rate": 1.5}, r"'min_lr_rate' must be in \[0.0, 1.0\]"),
        ({"use_clip_grad_norm": True, "clip_grad_norm_max_norm": 0.0}, "'clip_grad_norm_max_norm' must be > 0"),
        ({"warmup_steps": -2}, "'warmup_steps' must be a non-negative int"),
        ({"warmup_steps": 0.1}, "'warmup_steps' must be a non-negative int"),
        ({"lr_scheduler_type": "cosine_with_min_lr"}, "'min_lr_rate' is required"),
    ],
)
def test_grpo_config_rejects_invalid_lr_and_clip(expect_error, override, message):
    with expect_error(ValueError, message):
        GRPOConfig(**{**_HOST_GRPO_CONFIG, **override})


def test_grpo_config_rejects_weight_sync_every_for_in_process(expect_error):
    with expect_error(ValueError, "'weight_sync_every' does not apply to rollout_backend='ttml_in_process_sync'"):
        GRPOConfig(**_HOST_GRPO_CONFIG, weight_sync_every=2)


@pytest.mark.parametrize("num_iterations", [0, -1])
def test_grpo_config_rejects_num_iterations_below_one(expect_error, num_iterations):
    with expect_error(ValueError, "'num_iterations' must be > 0"):
        GRPOConfig(**{**_HOST_GRPO_CONFIG, "num_iterations": num_iterations})


@pytest.mark.parametrize("weight_sync_every, num_iterations", [(1, 2), (3, 2), (4, 3)])
def test_grpo_config_rejects_async_weight_sync_every_not_a_multiple_of_num_iterations(
    expect_error, weight_sync_every, num_iterations
):
    with expect_error(ValueError, "needs 'weight_sync_every' to be a multiple of 'num_iterations'"):
        _remote_grpo_config(
            rollout_backend="ttt_remote_async", weight_sync_every=weight_sync_every, num_iterations=num_iterations
        )


@pytest.mark.parametrize("weight_sync_every, num_iterations, max_staleness", [(2, 1, 0), (4, 2, 1), (6, 2, 3)])
def test_grpo_config_rejects_async_max_staleness_below_weight_sync_every_minus_num_iterations(
    expect_error, weight_sync_every, num_iterations, max_staleness
):
    with expect_error(ValueError, "training waits for a fresh batch from inference"):
        _remote_grpo_config(
            rollout_backend="ttt_remote_async",
            weight_sync_every=weight_sync_every,
            num_iterations=num_iterations,
            max_staleness=max_staleness,
        )


@pytest.mark.parametrize(
    "rollout_backend, weight_sync_every, num_iterations, max_staleness",
    [
        ("ttt_remote_sync", 3, 2, 0),
        ("ttt_remote_sync", 2, 1, 0),
        ("ttt_remote_async", 2, 2, 0),
        ("ttt_remote_async", 6, 3, 3),
        ("ttt_remote_async", 2, 1, 1),
        ("ttt_remote_async", 6, 2, 4),
    ],
)
def test_grpo_config_accepts_remote_weight_sync_schedule(
    rollout_backend, weight_sync_every, num_iterations, max_staleness
):
    cfg = _remote_grpo_config(
        rollout_backend=rollout_backend,
        weight_sync_every=weight_sync_every,
        num_iterations=num_iterations,
        max_staleness=max_staleness,
    )
    assert cfg.weight_sync_every == weight_sync_every


def test_validate_clip_grad_norm_rejects_sharded_params(monkeypatch, expect_error):
    from ttml.common.utils import validate_clip_grad_norm

    replicated, sharded = object(), object()
    monkeypatch.setattr(
        ttml.Sharding, "from_tensor", lambda tensor: SimpleNamespace(is_fully_replicated=tensor is not sharded)
    )

    validate_clip_grad_norm(SimpleNamespace(parameters=lambda: {"a": replicated, "b": replicated}))
    with expect_error(ValueError, "not supported with sharded parameters"):
        validate_clip_grad_norm(SimpleNamespace(parameters=lambda: {"a": replicated, "w": sharded}))


@pytest.mark.parametrize(
    "lr_scheduler_type, warmup_steps, total_steps, min_lr_rate, expected",
    [
        ("constant", 0, 3, 0.0, [1.0, 1.0, 1.0]),
        ("constant", 2, 3, 0.0, [1.0, 1.0, 1.0]),
        ("constant_with_warmup", 1, 3, 0.0, [1.0, 1.0, 1.0]),
        ("constant_with_warmup", 2, 4, 0.0, [0.5, 1.0, 1.0, 1.0]),
        ("constant_with_warmup", 4, 5, 0.0, [0.25, 0.5, 0.75, 1.0, 1.0]),
        ("cosine", 0, 5, 0.0, [1.0, 0.904508, 0.654508, 0.345492, 0.095492]),
        ("cosine_with_min_lr", 0, 5, 0.5, [1.0, 0.952254, 0.827254, 0.672746, 0.547746]),
        ("cosine_with_min_lr", 0, 5, 0.9, [1.0, 0.990451, 0.965451, 0.934549, 0.909549]),
        ("cosine_with_min_lr", 2, 6, 0.1, [0.5, 1.0, 1.0, 0.868198, 0.55, 0.231802]),
        ("cosine", 0, 1, 0.0, [1.0]),
        ("cosine", 4, 2, 0.0, [0.25, 0.5]),
    ],
)
def test_build_lr_scheduler_trajectory(lr_scheduler_type, warmup_steps, total_steps, min_lr_rate, expected):
    from ttml.trainers.grpo_trainer.grpo_trainer import _build_lr_scheduler

    base_lr = 0.1
    opt = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": base_lr}, ttml.NamedParameters())
    cfg = GRPOConfig(
        **{
            **_HOST_GRPO_CONFIG,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_steps": warmup_steps,
            "min_lr_rate": min_lr_rate,
        }
    )
    sched = _build_lr_scheduler(opt, cfg, total_steps)
    lrs = []
    for _ in range(total_steps):
        lrs.append(opt.get_lr() / base_lr)
        if sched is not None:
            sched.step()
    assert lrs == pytest.approx(expected, abs=1e-6)


_GENERATE_TIME_S = 1.0
_SAVE_CHECKPOINT_TIME_S = 4.0
_ON_SAVE_TIME_S = 2.0
_WEIGHT_SYNC_TIME_S = 8.0
_ROLLOUTS_DROPPED = 3
_REWARDS = [0.0, 1.0, 0.0, 1.0]
_BEHAVIOR_LOGPROB = -0.5
_BASE_LR = 1.0


def _host_scored_batch(version=0, rollouts_dropped=0):
    """A scored batch of 2 prompts x 2 generations, as _run_train_loop_on_host trains on."""
    rollout = RolloutBatch(
        weight_version=version,
        prompts=[[1, 2], [1, 2], [3, 4], [3, 4]],
        completions=[[7], [7], [7], [7]],
        logprobs=np.full((4, _HOST_GRPO_CONFIG["max_completion_length"]), _BEHAVIOR_LOGPROB, dtype=np.float32),
    )
    return ScoredRolloutBatch(
        rollout, np.array(_REWARDS, dtype=np.float32), {}, _GENERATE_TIME_S, 0.0, rollouts_dropped
    )


def _run_train_loop_on_host(monkeypatch, num_batches=1, source=None, **config_overrides):
    """Run ``GRPOTrainer.train()`` on host fakes: ``num_batches`` scored batches of 2 prompts x 2 generations,
    ``num_iterations=2``, a checkpoint every step. The fake scheduler halves the LR each step."""
    from ttml.trainers import GRPOMonitor

    run = SimpleNamespace(
        clock=0.0, batches_drawn=0, events=[], pi_old=[], saved_lrs=[], metrics_by_step={}, update_weights_steps=[]
    )
    monkeypatch.setattr(grpo_trainer_module, "time", SimpleNamespace(perf_counter=lambda: run.clock))

    class _Source(RolloutBatchSource):
        def __iter__(self):
            run.batches_drawn += 1
            for _ in range(num_batches):
                run.clock += _GENERATE_TIME_S
                yield _host_scored_batch(rollouts_dropped=_ROLLOUTS_DROPPED)

        def update_weights(self, model, step):
            run.update_weights_steps.append(step)
            run.clock += _WEIGHT_SYNC_TIME_S
            return _WEIGHT_SYNC_TIME_S

    class _Optimizer:
        lr = _BASE_LR

        def get_lr(self):
            return self.lr

        def step(self):
            pass

        def zero_grad(self):
            pass

    class _Recorder(TrainerCallback):
        def on_step_end(self, trainer, step, **kwargs):
            run.events.append(("on_step_end", step))
            if step == 1:
                trainer.metrics["step_1_only"] = 1.0

        def on_save(self, trainer, step, path):
            run.events.append(("on_save", step))
            run.clock += _ON_SAVE_TIME_S

    class _Monitor(GRPOMonitor):
        def on_step_end(self, trainer, step, **kwargs):
            run.events.append(("monitor", step))
            run.metrics_by_step[step] = kwargs

    def fake_save_checkpoint(model, step, output_dir, **kwargs):
        run.events.append(("save_checkpoint", step))
        run.saved_lrs.append(kwargs["learning_rate"])
        run.clock += _SAVE_CHECKPOINT_TIME_S

    cfg = GRPOConfig(
        **{
            **_HOST_GRPO_CONFIG,
            "num_iterations": 2,
            "checkpointing": True,
            "checkpoint_interval": 1,
            **config_overrides,
        }
    )
    optimizer, monitor = _Optimizer(), _Monitor(cfg)

    def fake_build_policy(self, transformer_config, device_config, dataset_func):
        self.model = SimpleNamespace(get_run_mode=lambda: RunMode.TRAIN)
        self.tokenizer = _IdsTokenizer()
        self._batch_layout = SimpleNamespace(
            fsdp_enabled=False, ddp_context_enabled=False, ddp_enabled=False, composer=None
        )
        self._optimizer = optimizer
        self._lr_scheduler = SimpleNamespace(step=lambda: setattr(optimizer, "lr", optimizer.lr / 2))
        self._monitor_cb = monitor
        self._completions_per_microbatch = 4
        self._num_generation_batches = num_batches
        self.rollout_batch_source = source or _Source()

    monkeypatch.setattr(grpo_trainer_module, "init_distributed_training", lambda config, device_config: None)
    monkeypatch.setattr(grpo_trainer_module, "is_rollout_rank", lambda config: False)
    monkeypatch.setattr(GRPOTrainer, "_build_policy", fake_build_policy)
    monkeypatch.setattr(
        GRPOTrainer, "_compute_loss_and_backward", lambda self, nlog_new, m, old, adv, n: run.pi_old.append(old)
    )
    monkeypatch.setattr(
        GRPOTrainer,
        "_compute_policy_nlog_probs",
        lambda self, inputs, targets: SimpleNamespace(to_numpy=lambda *a, **k: np.zeros(inputs.shape, np.float32)),
    )
    monkeypatch.setattr(grpo_trainer_module, "save_checkpoint", fake_save_checkpoint)

    GRPOTrainer(
        transformer_config=SimpleNamespace(model_type="llama"),
        device_config=SimpleNamespace(enable_fsdp=False),
        model_source=HF_MODEL_ID,
        dataset_func=lambda: None,
        config=cfg,
        reward_funcs=[lambda completions, **_: list(_REWARDS)],
        optimizer_dict={"type": "AdamW"},  # never built: _build_policy is replaced
        callbacks=[_Recorder(), monitor],
    ).train()
    return run


def test_train_loop_reuses_pi_old_across_mini_epochs(monkeypatch):
    run = _run_train_loop_on_host(monkeypatch)
    assert run.batches_drawn == 1
    mini_epoch_1_pi_old, mini_epoch_2_pi_old = run.pi_old
    # With num_iterations > 1, pi_old must not be recomputed between mini-epochs.
    np.testing.assert_array_equal(mini_epoch_1_pi_old, mini_epoch_2_pi_old)


def test_train_loop_updates_weights_once_per_batch_except_the_last(monkeypatch):
    assert _run_train_loop_on_host(monkeypatch, num_batches=2).update_weights_steps == [2]


def test_train_loop_checkpoints_between_step_end_and_monitor(monkeypatch):
    run = _run_train_loop_on_host(monkeypatch)
    for step in (1, 2):
        step_events = [event for event, s in run.events if s == step]
        assert step_events == ["on_step_end", "save_checkpoint", "on_save", "monitor"]


def test_train_loop_mini_epoch_metrics(monkeypatch):
    step_1, step_2 = _run_train_loop_on_host(monkeypatch).metrics_by_step.values()
    assert step_1["reward_mean"] == step_2["reward_mean"] == np.mean(_REWARDS)
    assert step_1["generation_time_s"] == _GENERATE_TIME_S
    assert step_2["generation_time_s"] == 0.0
    assert step_1["rollout_wait_s"] == _GENERATE_TIME_S
    assert step_2["rollout_wait_s"] == step_2["reward_time_s"] == 0.0
    assert step_1["_Recorder_time_s"] == step_2["_Recorder_time_s"] == _ON_SAVE_TIME_S
    assert "step_1_only" in step_1 and "step_1_only" not in step_2
    assert step_1["logprob_mismatch_mean"] == step_1["logprob_mismatch_max"] == 0.5
    assert step_1["clip_frac"] == 0.5
    assert (step_1["rollouts_dropped"], step_2["rollouts_dropped"]) == (_ROLLOUTS_DROPPED, 0)
    assert step_1["rollout_staleness_mean"] == step_1["rollout_staleness_max"] == 0
    assert step_2["rollout_staleness_mean"] == step_2["rollout_staleness_max"] == 1


def test_monitor_emits_the_interval_max_of_rollout_staleness_max():
    from ttml.trainers.grpo_trainer.grpo_trainer import _aggregate, _MetricStats

    stats = _MetricStats()
    for value in (0.0, 2.0, 1.0):
        stats.push(value)
    assert _aggregate("rollout_staleness_max", stats) == 2.0
    assert _aggregate("rollout_staleness_mean", stats) == 1.0


def test_clipping_stats_counts_tokens_whose_clipped_term_is_the_min():
    from ttml.trainers.grpo_trainer.grpo_trainer import clipping_stats

    new = np.array([[-0.5, 0.0, 9.0], [0.5, 0.0, 9.0]], dtype=np.float32)
    mask = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.float32)
    mismatch, clipped = clipping_stats(np.zeros_like(new), new, mask, np.array([1.0, -1.0]), eps=0.2)
    np.testing.assert_array_equal(mismatch, [0.5, 0.0, 0.5, 0.0])
    np.testing.assert_array_equal(clipped, [True, False, True, False])


def test_train_loop_step_time_includes_checkpoint(monkeypatch):
    step_1, step_2 = _run_train_loop_on_host(monkeypatch).metrics_by_step.values()
    assert step_1["step_time_s"] == _GENERATE_TIME_S + _SAVE_CHECKPOINT_TIME_S + _ON_SAVE_TIME_S
    assert step_2["step_time_s"] == _SAVE_CHECKPOINT_TIME_S + _ON_SAVE_TIME_S


def test_train_loop_counts_the_weight_push_in_the_next_step(monkeypatch):
    step_1, step_2, step_3, step_4 = _run_train_loop_on_host(monkeypatch, num_batches=2).metrics_by_step.values()
    assert step_1["weight_sync_time_s"] == step_2["weight_sync_time_s"] == step_4["weight_sync_time_s"] == 0.0
    assert step_3["weight_sync_time_s"] == _WEIGHT_SYNC_TIME_S
    assert step_3["step_time_s"] == step_1["step_time_s"] + _WEIGHT_SYNC_TIME_S


def test_train_loop_checkpoint_records_step_lr(monkeypatch):
    run = _run_train_loop_on_host(monkeypatch)
    step_lrs = [metrics["lr"] for metrics in run.metrics_by_step.values()]
    assert run.saved_lrs == step_lrs == [_BASE_LR, _BASE_LR / 2]


@pytest.mark.requires_device
@pytest.mark.parametrize("temperature", [0.0, 1.0, 0.5])
def test_compute_policy_nlog_probs_divides_logits_by_temperature(temperature):
    B, Tp, V = 2, 32, 128
    rng = np.random.default_rng(0)
    logits_np = rng.normal(scale=3.0, size=(B, 1, Tp, V)).astype(np.float32)
    targets = rng.integers(0, V, size=(B, Tp)).astype(np.uint32)
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.config = SimpleNamespace(temperature=temperature)
    trainer._batch_layout = SimpleNamespace(mapper=None)
    trainer.model = lambda x, mask: ttml.autograd.Tensor.from_numpy(logits_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
    trainer.model.config = SimpleNamespace(use_composite_sdpa=False)

    nlog = trainer._compute_policy_nlog_probs(np.zeros((B, Tp), dtype=np.uint32), targets)

    z = logits_np[:, 0] / (temperature if temperature > 0 else 1.0)
    z_max = z.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.exp(z - z_max).sum(axis=-1)) + z_max[..., 0]
    expected = log_sum_exp - np.take_along_axis(z, targets[..., None].astype(np.int64), axis=-1)[..., 0]
    np.testing.assert_allclose(nlog.to_numpy(ttnn.DataType.FLOAT32), expected, rtol=2e-2, atol=5e-2)
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _make_muon_with_adamw():
    from ttml.common.muon_optimizer import MuonWithAdamW

    return MuonWithAdamW({"muon": {"lr": 0.02}, "adamw": {"lr": 0.0003}}, ttml.NamedParameters())


@pytest.mark.parametrize(
    "lr_scheduler_type, warmup_steps",
    [("constant", 0), ("constant", 2), ("constant_with_warmup", 0), ("constant_with_warmup", 1)],
)
def test_build_lr_scheduler_returns_none_for_constant_lr(lr_scheduler_type, warmup_steps):
    from ttml.trainers.grpo_trainer.grpo_trainer import _build_lr_scheduler

    cfg = GRPOConfig(**{**_HOST_GRPO_CONFIG, "lr_scheduler_type": lr_scheduler_type, "warmup_steps": warmup_steps})
    assert _build_lr_scheduler(_make_muon_with_adamw(), cfg, 4) is None


@pytest.mark.parametrize(
    "lr_scheduler_type, warmup_steps, expected_first_lr_scale",
    [("constant_with_warmup", 2, 0.5), ("cosine", 0, 1.0), ("cosine", 2, 0.5)],
)
def test_build_lr_scheduler_supports_schedules_for_muon_with_adamw(
    lr_scheduler_type, warmup_steps, expected_first_lr_scale
):
    from ttml.trainers.grpo_trainer.grpo_trainer import _build_lr_scheduler

    optimizer = _make_muon_with_adamw()
    base_lrs = optimizer.get_group_lrs()
    cfg = GRPOConfig(**{**_HOST_GRPO_CONFIG, "lr_scheduler_type": lr_scheduler_type, "warmup_steps": warmup_steps})
    scheduler = _build_lr_scheduler(optimizer, cfg, 4)

    assert scheduler is not None
    assert optimizer.get_group_lrs() == pytest.approx([lr * expected_first_lr_scale for lr in base_lrs], abs=1e-9)


@pytest.mark.parametrize(
    "warmup_steps, total_steps, expected_scales",
    [
        (0, 6, [1.0, 0.93971, 0.775, 0.55, 0.325, 0.16029, 0.1]),
        (2, 6, [0.5, 1.0, 1.0, 0.8682, 0.55, 0.2318, 0.1]),
    ],
)
def test_build_lr_scheduler_cosine_with_min_lr_scales_each_group_relatively(warmup_steps, total_steps, expected_scales):
    from ttml.trainers.grpo_trainer.grpo_trainer import _build_lr_scheduler

    optimizer = _make_muon_with_adamw()
    base_lrs = optimizer.get_group_lrs()
    cfg = GRPOConfig(
        **{
            **_HOST_GRPO_CONFIG,
            "lr_scheduler_type": "cosine_with_min_lr",
            "warmup_steps": warmup_steps,
            "min_lr_rate": 0.1,
        }
    )
    scheduler = _build_lr_scheduler(optimizer, cfg, total_steps=total_steps)
    assert scheduler is not None

    observed_scales = []
    for step in range(total_steps + 1):
        group_scales = [lr / base_lr for lr, base_lr in zip(optimizer.get_group_lrs(), base_lrs)]
        assert group_scales == pytest.approx([group_scales[0]] * len(group_scales), abs=1e-6)
        observed_scales.append(group_scales[0])
        if step < total_steps:
            scheduler.step()
    assert observed_scales == pytest.approx(expected_scales, abs=2e-3)


class _LoopbackBridge:
    def connect(self):
        pass

    def send_weights(self, weights):
        self.weights = weights

    def receive_weights(self):
        return [self.weights]

    def barrier(self):
        pass


class _EchoSampler(RolloutSampler):
    """Returns each prompt as its completion; an empty prompt raises."""

    def __init__(self):
        self._version = -1

    def generate(self, prompts):
        if any(not p for p in prompts):
            raise ValueError("empty prompt")
        logprobs = np.full((len(prompts), 4), -0.5, dtype=np.float32)
        return RolloutBatch(weight_version=self._version, prompts=prompts, completions=prompts, logprobs=logprobs)

    @property
    def weight_version(self):
        return self._version

    def update_weights(self, weights, version):
        if version <= self._version:
            raise ValueError(f"version {version} <= {self._version}")
        self._version = version

    def close(self):
        self.closed = True


@pytest.fixture
def mpi_loopback(monkeypatch):
    """Replace MPI with a _LoopbackContext; returns (queues, atexit hooks)."""
    queues = collections.defaultdict(queue.Queue)
    at_exit = []
    loopback = _LoopbackContext(queues)
    monkeypatch.setattr(mpi_rollout, "_mpi_send_bytes", loopback.send)
    monkeypatch.setattr(mpi_rollout, "_mpi_recv_bytes", loopback.recv)
    monkeypatch.setattr(
        mpi_rollout,
        "_require_distributed_context",
        lambda owner: TTT_RANK if owner == "MPIRolloutServer" else TTML_RANK,
    )
    # Without real MPI: no os._exit, and no shutdown at pytest exit.
    monkeypatch.setattr(mpi_rollout, "exit_on_error", contextlib.nullcontext)

    def unregister(fn):
        at_exit[:] = [f for f in at_exit if f != fn]

    monkeypatch.setattr(mpi_rollout.atexit, "register", lambda fn, *args: at_exit.append(fn))
    monkeypatch.setattr(mpi_rollout.atexit, "unregister", unregister)
    return queues, at_exit


def _start_rollout_server(sampler):
    """Connect a client to an MPIRolloutServer serving ``sampler`` in a thread."""
    bridge = _LoopbackBridge()
    server = mpi_rollout.MPIRolloutServer(peer_rank=TTML_RANK, bridge=bridge, sampler=sampler)
    client = mpi_rollout.MPIRolloutClient(peer_rank=TTT_RANK, bridge=bridge)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return client, thread


def test_mpi_rollout_loopback(mpi_loopback, expect_error):
    """Real MPIRolloutServer (in a thread) against the real MPIRolloutClient, with MPI
    replaced by one in-memory queue per tag. In order:

    1. send_weights delivers the version, and the client consumes the transfer response.
    2. remote_generate returns completions, log-probs cut to each completion's length, and the version.
    3. A raising generate() reaches the client as OP_ERROR: a RuntimeError with rank 1's traceback.
    4. A raising update_weights() reaches the client the same way.
    5. After both errors the protocol is still in sync: the next remote_generate works.
    """
    _, at_exit = mpi_loopback
    client, thread = _start_rollout_server(_EchoSampler())
    assert at_exit == [os._exit, client.shutdown]

    client.send_weights({}, version=0)
    assert client.remote_generate([[1, 2], [3]]) == ([[1, 2], [3]], [[-0.5, -0.5], [-0.5]], 0)
    with expect_error(RuntimeError, r"rollout rank 1 failed:[\s\S]*ValueError: empty prompt"):
        client.remote_generate([[1], []])
    with expect_error(RuntimeError, r"rollout rank 1 failed:[\s\S]*ValueError: version 0 <= 0"):
        client.send_weights({}, version=0)
    _, _, version = client.remote_generate([[4]])
    assert version == 0

    client.shutdown()
    thread.join(timeout=30)
    assert at_exit == [client.shutdown]


class _IdsTokenizer:
    def decode(self, ids, **kwargs):
        return str(ids)


def _failing_reward(completions, **kwargs):
    raise ValueError("reward failed")


def _zero_reward(completions, **kwargs):
    return [0.0] * len(completions)


class _FailingTrainBegin(TrainerCallback):
    def on_train_begin(self, trainer):
        raise ValueError("on_train_begin failed")


def _remote_grpo_config(**overrides):
    return GRPOConfig(
        **{
            **_HOST_GRPO_CONFIG,
            "rollout_backend": "ttt_remote_sync",
            "remote_rollout": RemoteRolloutConfig(mesh_shape=[1, 1], max_batch_size=1, max_seq_len=64),
            **overrides,
        }
    )


def _trainer_with_remote_rollout(monkeypatch, client, reward_funcs, callbacks=()):
    """A real GRPOTrainer on the ttt_remote_sync backend whose rollouts go through ``client``.
    Device, fabric and MPI setup are patched out."""

    def build_policy(self, transformer_config, device_config, dataset_func):
        self.rollout_batch_source = SyncRemoteRolloutBatchSource(
            client=client,
            prompts=[[1, 2]],
            extra_columns={},
            batch_prompts=1,
            tokenizer=_IdsTokenizer(),
            reward_funcs=reward_funcs,
            num_generations=1,
            max_completion_length=4,
            weight_sync_every=1,
        )

    monkeypatch.setattr(grpo_trainer_module, "init_distributed_training", lambda config, device_config: None)
    monkeypatch.setattr(grpo_trainer_module, "is_rollout_rank", lambda config: False)
    monkeypatch.setattr(GRPOTrainer, "_build_policy", build_policy)
    monkeypatch.setattr(grpo_ttml_model, "weights_ref_hf_dict", lambda model: {})
    return GRPOTrainer(
        transformer_config=SimpleNamespace(model_type="llama"),
        device_config=SimpleNamespace(enable_fsdp=False),
        model_source=HF_MODEL_ID,
        dataset_func=lambda: None,
        config=_remote_grpo_config(),
        reward_funcs=reward_funcs,
        optimizer_dict={"type": "AdamW"},  # never built: _build_policy is replaced
        callbacks=list(callbacks),
    )


@pytest.mark.parametrize(
    "reward_func, callbacks, message",
    [(_failing_reward, [], "reward failed"), (_zero_reward, [_FailingTrainBegin()], "on_train_begin failed")],
)
def test_failed_trainer_shuts_down_the_rollout_server(
    mpi_loopback, monkeypatch, expect_error, reward_func, callbacks, message
):
    """A trainer that fails before or during training sends OP_SHUTDOWN; the rollout server stops without closing
    the sampler, and the atexit backstop sends nothing more."""
    queues, at_exit = mpi_loopback
    sampler = _EchoSampler()
    client, server_thread = _start_rollout_server(sampler)
    trainer = _trainer_with_remote_rollout(monkeypatch, client, reward_funcs=[reward_func], callbacks=callbacks)

    with expect_error(ValueError, message):
        trainer.train()

    server_thread.join(timeout=30)
    assert not server_thread.is_alive()
    assert not getattr(sampler, "closed", False)
    assert at_exit == [client.shutdown]
    at_exit[0]()
    assert queues[(TTT_RANK, mpi_rollout._INFER_REQ_HDR_TAG)].empty()


def test_serve_rollouts_closes_the_sampler_and_the_mesh(monkeypatch):
    closed_meshes = []
    monkeypatch.setattr(ttnn, "close_mesh_device", closed_meshes.append)
    trainer = object.__new__(GRPOTrainer)
    sampler = _EchoSampler()
    trainer._rollout_server = SimpleNamespace(serve_forever=lambda: None)
    trainer.rollout_sampler, trainer._rollout_mesh = sampler, "mesh"
    trainer._serve_rollouts()
    assert sampler.closed and closed_meshes == ["mesh"]


@pytest.mark.parametrize(
    "model_type, enable_fsdp, local_source, message",
    [
        ("mistral", False, False, "supports Llama and Qwen3 models only"),
        ("qwen3", True, False, "does not support FSDP"),
        ("llama", False, True, "needs a HuggingFace Hub model id"),
    ],
)
def test_validate_remote_inputs_rejects(tmp_path, expect_error, model_type, enable_fsdp, local_source, message):
    with expect_error(ValueError, message):
        grpo_trainer_module._validate_remote_inputs(
            _remote_grpo_config(),
            SimpleNamespace(model_type=model_type),
            SimpleNamespace(enable_fsdp=enable_fsdp),
            str(tmp_path) if local_source else HF_MODEL_ID,
        )


def test_register_hub_config_downloads_only_unregistered_names(monkeypatch, tmp_path):
    from ttml.trainers.grpo_trainer.remote_rollout import ttt_rollout_sampler

    downloads = []
    monkeypatch.setattr(ttt_rollout_sampler.ModelArgs, "LOCAL_HF_PARAMS", {"Llama-3.2-1B-Instruct": "registered"})
    monkeypatch.setattr(ttt_rollout_sampler, "snapshot_download", lambda **kw: downloads.append(kw) or str(tmp_path))

    ttt_rollout_sampler._register_hub_config("unsloth/Llama-3.2-1B-Instruct")
    ttt_rollout_sampler._register_hub_config("Qwen/Qwen3-0.6B")

    assert downloads == [{"repo_id": "Qwen/Qwen3-0.6B", "allow_patterns": ["config.json"]}]
    assert ttt_rollout_sampler.ModelArgs.LOCAL_HF_PARAMS == {
        "Llama-3.2-1B-Instruct": "registered",
        "Qwen3-0.6B": str(tmp_path),
    }


@pytest.mark.parametrize("rollout_mesh", [[1, 1], [1, 2]])
@pytest.mark.parametrize("trainer_mesh", [[1, 1], [1, 2]])
@pytest.mark.parametrize("rank", [0, 1])
def test_init_distributed_training_sets_fabric_on_multi_device_ranks(monkeypatch, rank, trainer_mesh, rollout_mesh):
    """Only a rank whose own mesh has more than one device enables fabric."""
    from ttml.trainers.grpo_trainer import distributed

    calls = []
    monkeypatch.setattr(ttnn, "distributed_context_is_initialized", lambda: True)
    monkeypatch.setattr(ttnn, "distributed_context_get_size", lambda: 2)
    monkeypatch.setattr(ttnn, "distributed_context_get_rank", lambda: rank)
    monkeypatch.setattr(ttnn, "set_fabric_config", calls.append)
    config = _remote_grpo_config()
    config.remote_rollout.mesh_shape = rollout_mesh

    distributed.init_distributed_training(config, SimpleNamespace(total_devices=lambda: trainer_mesh[1]))

    own_mesh = rollout_mesh if rank == distributed.TTT_RANK else trainer_mesh
    assert calls == ([ttnn.FabricConfig.FABRIC_2D] if own_mesh[1] > 1 else [])


class _RepeatSampler(_EchoSampler):
    """Returns each prompt twice, as its two completions."""

    def generate(self, prompts):
        return super().generate(repeat_interleave(prompts, 2))


def test_in_process_source_scores_each_batch():
    answers = []

    def length_reward(completions, answer, **_):
        answers.append(answer)
        return [float(len(c)) for c in completions]

    source = InProcessRolloutBatchSource(
        sampler=_RepeatSampler(),
        prompts=[[1], [2, 3], [4], [5, 6]],
        extra_columns={"answer": ["a", "b", "c", "d"]},
        batch_prompts=2,
        tokenizer=_IdsTokenizer(),
        reward_funcs=[length_reward],
        num_generations=2,
        max_completion_length=4,
    )
    batches = list(source)

    assert [b.rollout.prompts for b in batches] == [[[1], [1], [2, 3], [2, 3]], [[4], [4], [5, 6], [5, 6]]]
    assert answers == [["a", "a", "b", "b"], ["c", "c", "d", "d"]]
    np.testing.assert_array_equal(batches[0].rewards, [3.0, 3.0, 6.0, 6.0])


def test_score_rollout_rejects_unexpanded_prompts(expect_error):
    rollout = RolloutBatch(
        weight_version=0, prompts=[[1], [2]], completions=[[7], [7]], logprobs=np.zeros((2, 4), np.float32)
    )
    with expect_error(RuntimeError, "each prompt repeated num_generations times"):
        score_rollout(
            rollout,
            [[1], [2]],
            {},
            tokenizer=_IdsTokenizer(),
            reward_funcs=[lambda completions, **_: [0.0] * len(completions)],
            num_generations=2,
            generation_time_s=0.0,
        )


@pytest.mark.parametrize(
    "completions, logprobs, message",
    [
        ([[7]], np.zeros((1, 4), np.float32), "expected 2 rows"),
        ([[7], [7]], np.zeros((2, 3), np.float32), "logprobs must be a float32 array"),
        ([[7], [7]], np.zeros((2, 4), np.float64), "logprobs must be a float32 array"),
        ([[7], [7] * 5], np.zeros((2, 4), np.float32), "completion length 5 > max_completion_length 4"),
        ([[7], [7]], np.array([[0.5, 0, 0, 0], [0, 0, 0, 0]], np.float32), "invalid behavior log-probs"),
        ([[7], [7]], np.array([[np.nan, 0, 0, 0], [0, 0, 0, 0]], np.float32), "invalid behavior log-probs"),
    ],
)
def test_validate_rollout_batch_rejects(expect_error, completions, logprobs, message):
    rollout = RolloutBatch(
        weight_version=0, prompts=[[1]] * len(completions), completions=completions, logprobs=logprobs
    )
    with expect_error(RuntimeError, message):
        validate_rollout_batch(rollout, num_prompts=1, num_generations=2, max_completion_length=4)


class _FakeRolloutClient:
    """Records weight pushes; remote_generate echoes the prompts at the version the rollout rank holds."""

    def __init__(self, applies_pushes=True):
        self.applies_pushes = applies_pushes
        self.pushed = []
        self.version = -1

    def send_weights(self, weights, version):
        self.pushed.append(version)
        if self.applies_pushes:
            self.version = version

    def remote_generate(self, prompts):
        return prompts, [[-0.5] * len(p) for p in prompts], self.version


def _sync_source(monkeypatch, client, weight_sync_every):
    monkeypatch.setattr(grpo_ttml_model, "weights_ref_hf_dict", lambda model: {})
    return SyncRemoteRolloutBatchSource(
        client=client,
        prompts=[[1], [2], [3], [4]],
        extra_columns={},
        batch_prompts=1,
        tokenizer=_IdsTokenizer(),
        reward_funcs=[lambda completions, **_: [0.0] * len(completions)],
        num_generations=1,
        max_completion_length=4,
        weight_sync_every=weight_sync_every,
    )


def test_in_process_source_update_weights_returns_zero_duration_and_updates_sampler_version():
    class _VersionSampler(RolloutSampler):
        def __init__(self):
            self._version = -1

        @property
        def weight_version(self):
            return self._version

        def generate(self, prompts):
            raise AssertionError("generate is not used in this test")

        def update_weights(self, weights, version):
            self._version = version

    sampler = _VersionSampler()
    source = InProcessRolloutBatchSource(
        sampler=sampler,
        prompts=[[1]],
        extra_columns={},
        batch_prompts=1,
        tokenizer=_IdsTokenizer(),
        reward_funcs=[lambda completions, **_: [0.0] * len(completions)],
        num_generations=1,
        max_completion_length=4,
    )

    assert source.update_weights(None, 7) == 0.0
    assert sampler.weight_version == 7


def test_sync_remote_source_pushes_every_weight_sync_every_steps(monkeypatch):
    client = _FakeRolloutClient()
    source = _sync_source(monkeypatch, client, weight_sync_every=2)

    first_push_duration = source.update_weights(None, 0)
    versions = []
    skipped_durations = []
    push_durations = [first_push_duration]
    for step, scored in enumerate(source, start=1):
        versions.append(scored.rollout.weight_version)
        if step < 4:
            duration_s = source.update_weights(None, step)
            if step == 2:
                push_durations.append(duration_s)
            else:
                skipped_durations.append(duration_s)

    assert versions == [0, 0, 2, 2]
    assert client.pushed == [0, 2]
    assert skipped_durations == [0.0, 0.0]
    assert all(duration_s >= 0.0 for duration_s in push_durations)


def test_sync_remote_source_rejects_a_stale_rollout_version(monkeypatch, expect_error):
    source = _sync_source(monkeypatch, _FakeRolloutClient(applies_pushes=False), weight_sync_every=1)
    source.update_weights(None, 0)
    with expect_error(RuntimeError, "generated with weight_version -1, expected 0"):
        next(iter(source))


def _scored_batch(version):
    completions = [[7, 8], [9]]
    return ScoredRolloutBatch(
        rollout=RolloutBatch(
            weight_version=version,
            prompts=[[1], [1]],
            completions=completions,
            logprobs=pad_logprobs(completions, [[-0.5] * len(c) for c in completions], 4),
        ),
        rewards=np.array([1.0, 0.5], dtype=np.float32),
        reward_components={"length": np.array([2.0, 1.0], dtype=np.float32)},
        generation_time_s=0.25,
        reward_time_s=0.125,
    )


def test_rollout_queue_loopback(monkeypatch, thread_errors):
    queues = collections.defaultdict(queue.Queue)
    monkeypatch.setattr(rollout_queue, "duplicate_distributed_context", lambda: _LoopbackContext(queues))
    producer = RolloutQueue.producer(peer_rank=TTML_RANK, capacity=2, num_generations=2)
    consumer = RolloutQueue.consumer(peer_rank=TTT_RANK, capacity=2, num_generations=2)
    producer.connect()
    consumer.connect()

    sent = [_scored_batch(version) for version in range(3)]
    received = []

    def send_all():
        for scored in sent:
            producer.push(scored)
        producer.close()

    assert _finishes(send_all)
    assert _finishes(lambda: received.extend(consumer.pop() for _ in range(len(sent) + 1)))
    assert received.pop() is None
    assert _finishes(consumer.close)
    assert json.loads(rollout_queue._encode(sent[0], 2))["prompts"] == [[1]]

    for got, want in zip(received, sent):
        assert got.rollout.weight_version == want.rollout.weight_version
        assert got.rollout.prompts == want.rollout.prompts
        assert got.rollout.completions == want.rollout.completions
        assert got.rollout.logprobs.dtype == np.float32
        np.testing.assert_array_equal(got.rollout.logprobs, want.rollout.logprobs)
        np.testing.assert_array_equal(got.rewards, want.rewards)
        np.testing.assert_array_equal(got.reward_components["length"], want.reward_components["length"])
        assert (got.generation_time_s, got.reward_time_s) == (want.generation_time_s, want.reward_time_s)


def test_trace_capture_lock_is_held_from_begin_to_end(monkeypatch):
    lock = threaded_weight_bridge._capture_lock
    seen = []
    monkeypatch.setattr(
        threaded_weight_bridge, "_begin_trace_capture", lambda *a, **k: seen.append(lock.locked()) or "tid"
    )
    monkeypatch.setattr(threaded_weight_bridge, "_end_trace_capture", lambda *a, **k: seen.append(lock.locked()))
    assert threaded_weight_bridge._begin_trace_capture_locked("mesh", cq_id=0) == "tid"
    assert lock.locked()
    threaded_weight_bridge._end_trace_capture_locked("mesh", "tid", cq_id=0)
    assert not lock.locked() and seen == [True, True]


def test_trace_capture_lock_is_released_when_begin_raises(monkeypatch, expect_error):
    def fail(*args, **kwargs):
        raise RuntimeError("begin failed")

    monkeypatch.setattr(threaded_weight_bridge, "_begin_trace_capture", fail)
    with expect_error(RuntimeError, "begin failed"):
        threaded_weight_bridge._begin_trace_capture_locked("mesh", cq_id=0)
    assert not threaded_weight_bridge._capture_lock.locked()


def _wait_for_waiters(lock, count):
    deadline = time.monotonic() + 10
    while len(lock._waiters) < count:
        assert time.monotonic() < deadline, f"{count} threads did not wait for the lock in 10 s"
        time.sleep(0.01)


class _Interrupted(BaseException):
    pass


@contextlib.contextmanager
def _interrupt_main_thread_when(ready, action=lambda: None):
    """Sends SIGUSR1 to the main thread once ready() is true; its handler calls action() and raises _Interrupted.

    _thread.interrupt_main() does not wake a thread that is blocked in Lock.acquire(); a real signal does.
    """

    def handler(signum, frame):
        action()
        raise _Interrupted("interrupted")

    def send():
        deadline = time.monotonic() + 10
        while not ready() and time.monotonic() < deadline:
            time.sleep(0.01)
        if ready():
            signal.pthread_kill(threading.main_thread().ident, signal.SIGUSR1)

    sender = threading.Thread(target=send, daemon=True)
    previous = signal.signal(signal.SIGUSR1, handler)
    try:
        sender.start()
        yield
    finally:
        sender.join(10)
        signal.signal(signal.SIGUSR1, previous)


def test_fifo_lock_gives_the_lock_in_request_order():
    lock = threaded_weight_bridge._FifoLock()
    order = []

    def take(name):
        with lock:
            order.append(name)

    lock.acquire()
    threads = []
    for count, name in enumerate(("B", "C"), start=1):
        threads.append(threading.Thread(target=take, args=(name,), daemon=True))
        threads[-1].start()
        _wait_for_waiters(lock, count)
    lock.release()
    take("A")
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert order == ["B", "C", "A"]
    assert not lock.locked()


def test_fifo_lock_interrupted_wait_leaves_the_queue(expect_error):
    lock = threaded_weight_bridge._FifoLock()
    lock.acquire()
    with _interrupt_main_thread_when(lambda: len(lock._waiters) == 1), expect_error(_Interrupted, "interrupted"):
        lock.acquire()
    assert len(lock._waiters) == 0 and lock.locked()
    lock.release()
    assert not lock.locked()


def test_fifo_lock_interrupted_wait_passes_on_the_lock_it_was_given(expect_error):
    lock = threaded_weight_bridge._FifoLock()
    taken = []

    def take_after_the_main_thread():
        _wait_for_waiters(lock, 1)
        with lock:
            taken.append(True)

    lock.acquire()
    other = threading.Thread(target=take_after_the_main_thread, daemon=True)
    other.start()
    # The handler's release() gives the lock to the main thread's waiter just before the interrupt.
    with _interrupt_main_thread_when(lambda: len(lock._waiters) == 2, action=lock.release):
        with expect_error(_Interrupted, "interrupted"):
            lock.acquire()
    other.join(10)
    assert taken == [True]
    assert not lock.locked() and len(lock._waiters) == 0


def test_fifo_lock_release_when_unlocked_raises(expect_error):
    with expect_error(RuntimeError, "release unlocked lock"):
        threaded_weight_bridge._FifoLock().release()


def test_threaded_weight_bridge_receiver_locks_trace_capture_until_close(monkeypatch):
    monkeypatch.setattr(ttnn, "begin_trace_capture", threaded_weight_bridge._begin_trace_capture)
    monkeypatch.setattr(ttnn, "end_trace_capture", threaded_weight_bridge._end_trace_capture)
    monkeypatch.setattr(threaded_weight_bridge, "exit_on_error", contextlib.nullcontext)
    queues = collections.defaultdict(queue.Queue)
    monkeypatch.setattr(threaded_weight_bridge, "duplicate_distributed_context", lambda: _LoopbackContext(queues))
    bridge = threaded_weight_bridge.ThreadedWeightBridge

    sender = bridge.sender(peer_rank=TTT_RANK, mesh_device=None)
    sender.connect()
    assert ttnn.begin_trace_capture is threaded_weight_bridge._begin_trace_capture
    sender.close()  # sends the 0-length manifest that ends the receiver loop

    receiver = bridge.receiver(peer_rank=TTML_RANK, submeshes=[None])
    receiver.connect()
    assert ttnn.begin_trace_capture is threaded_weight_bridge._begin_trace_capture_locked
    assert ttnn.end_trace_capture is threaded_weight_bridge._end_trace_capture_locked
    receiver.close()
    assert ttnn.begin_trace_capture is threaded_weight_bridge._begin_trace_capture
    assert ttnn.end_trace_capture is threaded_weight_bridge._end_trace_capture


def test_threaded_weight_bridge_receiver_waits_for_the_cq0_reads_before_the_next_writes(monkeypatch, thread_errors):
    calls = []
    monkeypatch.setattr(ttnn, "begin_trace_capture", threaded_weight_bridge._begin_trace_capture)
    monkeypatch.setattr(ttnn, "end_trace_capture", threaded_weight_bridge._end_trace_capture)
    monkeypatch.setattr(ttnn, "allocate_tensor_on_device", lambda *a, **k: "pad")
    monkeypatch.setattr(ttnn, "from_torch", lambda *a, **k: "host")
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, pad, cq_id: calls.append(("write", cq_id)))
    monkeypatch.setattr(ttnn, "record_event", lambda mesh, cq_id: calls.append(("record", cq_id)) or f"E{cq_id}")
    monkeypatch.setattr(ttnn, "wait_for_event", lambda cq_id, event: calls.append(("wait", cq_id, event)))
    queues = collections.defaultdict(queue.Queue)
    monkeypatch.setattr(threaded_weight_bridge, "duplicate_distributed_context", lambda: _LoopbackContext(queues))
    trainer = _LoopbackContext(queues)
    receiver = threaded_weight_bridge.ThreadedWeightBridge.receiver(peer_rank=TTML_RANK, submeshes=["submesh"])
    receiver.connect()

    def send(blob, len_tag, body_tag):
        send_frame(trainer.send, blob, TTT_RANK, len_tag, body_tag)

    def receive_two_versions():
        entry = {"key": "w", "shape": [1], "dtype": "BFLOAT16", "layout": "TILE"}
        for version in (1, 2):
            manifest = json.dumps({"weight_version": version, "entries": [entry]}).encode()
            send(manifest, threaded_weight_bridge._TAG_MANIFEST_LEN, threaded_weight_bridge._TAG_MANIFEST_BODY)
            send(
                _torch_save_bytes(torch.zeros(1)),
                threaded_weight_bridge._TAG_WEIGHT_LEN,
                threaded_weight_bridge._TAG_WEIGHT_BODY,
            )
            with receiver.receive_weights(block=True) as (got, _):
                assert got == version
        send(b"", threaded_weight_bridge._TAG_MANIFEST_LEN, threaded_weight_bridge._TAG_MANIFEST_BODY)

    assert _finishes(receive_two_versions)
    assert _finishes(receiver.close)

    one_version = [("write", 1), ("record", 1), ("wait", 0, "E1"), ("record", 0)]
    assert calls == one_version + [("wait", 1, "E0")] + one_version


def test_threaded_weight_bridge_sender_reads_device_0_on_cq1_without_blocking(monkeypatch):
    calls = []
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda pad: [f"{pad}/device0"])
    monkeypatch.setattr(
        ttnn,
        "from_device",
        lambda tensor, blocking, queue_id: calls.append(("from_device", tensor, blocking, queue_id)) or "host",
    )
    monkeypatch.setattr(ttnn, "record_event", lambda mesh, cq_id: calls.append(("record", mesh, cq_id)) or "E1")
    monkeypatch.setattr(ttnn, "event_synchronize", lambda event: calls.append(("synchronize", event)))
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor: calls.append(("to_torch", tensor)) or "torch")
    sender = threaded_weight_bridge.ThreadedWeightBridge.sender(peer_rank=TTT_RANK, mesh_device="mesh")

    assert sender._read_device0("pad") == "torch"
    assert calls == [
        ("from_device", "pad/device0", False, 1),
        ("record", "mesh", 1),
        ("synchronize", "E1"),
        ("to_torch", "host"),
    ]


class _ScriptedBridge:
    """receive_weights yields the scripted results in order; the peer counts as closed once they run out."""

    def __init__(self, received):
        self._received = list(received)
        self.blocks = []
        self.connected = self.closed = False

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True

    @contextlib.contextmanager
    def receive_weights(self, block):
        self.blocks.append(block)
        yield self._received.pop(0) if self._received else None

    @property
    def peer_closed(self):
        return not self._received


class _RecordingQueue:
    def __init__(self):
        self.pushed = []
        self.connected = self.closed = False

    def connect(self):
        self.connected = True

    def push(self, scored, timeout=None):
        self.pushed.append(scored)
        return True

    def close(self):
        self.closed = True


def _async_rollout_server(bridge, queue, sampler, reward_funcs):
    rollout_batch_source = InProcessRolloutBatchSource(
        sampler=sampler,
        prompts=[[1], [2]],
        extra_columns={"answer": ["a", "b"]},
        batch_prompts=1,
        tokenizer=_IdsTokenizer(),
        reward_funcs=reward_funcs,
        num_generations=1,
        max_completion_length=4,
    )
    return async_rollout.AsyncRolloutServer(bridge=bridge, queue=queue, rollout_batch_source=rollout_batch_source)


def test_async_rollout_server_rejects_an_empty_dataset(expect_error):
    config = _remote_grpo_config(rollout_backend="ttt_remote_async")
    with expect_error(ValueError, "needs a non-empty dataset"):
        GRPOTrainer._build_rollout_server(SimpleNamespace(config=config), None, lambda: [])


@pytest.mark.parametrize("enable_ddp", [False, True])
def test_async_rollout_server_batch_prompts_match_the_trainer(monkeypatch, enable_ddp):
    from ttml.trainers.grpo_trainer import distributed as grpo_distributed

    device_config = DeviceConfig({"device_config": {"mesh_shape": [1, 2], "enable_ddp": enable_ddp}})
    context = SimpleNamespace(
        get_device=lambda: SimpleNamespace(get_num_devices=device_config.total_devices),
        is_parallelism_context_initialized=lambda: False,
        initialize_distributed_context=lambda *args: None,
    )
    fake_ttml = SimpleNamespace(
        autograd=SimpleNamespace(AutoContext=SimpleNamespace(get_instance=lambda: context)),
        maybe_mesh=lambda: build_mesh(device_config),
        core=SimpleNamespace(
            distributed=SimpleNamespace(
                shard_tensor_to_mesh_mapper=lambda device, dim: None,
                concat_mesh_to_tensor_composer=lambda device, dim: None,
            )
        ),
    )
    monkeypatch.setattr(grpo_distributed, "ttml", fake_ttml)
    monkeypatch.setattr(grpo_trainer_module, "ttml", fake_ttml)
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: [len(text)])
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(threaded_weight_bridge.ThreadedWeightBridge, "receiver", lambda **kwargs: None)
    monkeypatch.setattr(RolloutQueue, "producer", lambda **kwargs: None)
    monkeypatch.setattr(async_rollout, "AsyncRolloutServer", lambda **kwargs: kwargs["rollout_batch_source"])
    dataset = Dataset.from_dict({"prompt": [f"prompt {i}" for i in range(8)]})
    trainer = SimpleNamespace(
        config=_remote_grpo_config(rollout_backend="ttt_remote_async", prompts_to_train=8),
        model_source=HF_MODEL_ID,
        rollout_sampler=SimpleNamespace(submeshes=[None]),
        reward_funcs=[],
    )
    trainer._prompt_batches = functools.partial(GRPOTrainer._prompt_batches, trainer)

    source = GRPOTrainer._build_rollout_server(trainer, device_config, lambda: dataset)
    num_shards = grpo_distributed.resolve_batch_layout().num_shards
    assert num_shards == (2 if enable_ddp else 1)
    assert source._batch_prompts == trainer._prompt_batches(dataset, tokenizer, num_shards)[2]


def test_async_rollout_server_walks_prompts_with_the_newest_weights(monkeypatch):
    monkeypatch.setattr(async_rollout, "_require_distributed_context", lambda owner: TTT_RANK)
    monkeypatch.setattr(async_rollout, "exit_on_error", contextlib.nullcontext)
    answers = []

    def answer_reward(completions, answer, **_):
        answers.append(answer)
        return [0.0] * len(completions)

    bridge = _ScriptedBridge([(0, [{}]), None, (1, [{}]), None])
    queue_ = _RecordingQueue()
    sampler = _EchoSampler()
    server = _async_rollout_server(bridge, queue_, sampler, [answer_reward])
    assert bridge.connected and queue_.connected
    server.serve_forever()

    assert [s.rollout.weight_version for s in queue_.pushed] == [0, 0, 1]
    assert [s.rollout.prompts for s in queue_.pushed] == [[[1]], [[2]], [[1]]]
    assert answers == [["a"], ["b"], ["a"]]
    assert bridge.blocks == [True, False, False, False]
    assert bridge.closed and queue_.closed
    assert not getattr(sampler, "closed", False)


def test_async_rollout_server_atexit_backstop_is_unregistered_after_clean_exit(monkeypatch):
    monkeypatch.setattr(async_rollout, "_require_distributed_context", lambda owner: TTT_RANK)
    monkeypatch.setattr(async_rollout, "exit_on_error", contextlib.nullcontext)
    registrations = []

    def register(fn, *args):
        registrations.append((fn, args))

    def unregister(fn):
        for i, (registered_fn, _args) in enumerate(registrations):
            if registered_fn is fn:
                registrations.pop(i)
                break

    monkeypatch.setattr(async_rollout.atexit, "register", register)
    monkeypatch.setattr(async_rollout.atexit, "unregister", unregister)

    bridge = _ScriptedBridge([])
    queue_ = _RecordingQueue()
    server = _async_rollout_server(bridge, queue_, _EchoSampler(), [lambda completions, **_: [0.0] * len(completions)])
    assert registrations == [(async_rollout.os._exit, (1,))]

    server.serve_forever()

    assert registrations == []
    assert bridge.closed and queue_.closed


def test_async_rollout_client_and_server_check_their_rank(monkeypatch, expect_error):
    bridge, queue_ = _ScriptedBridge([]), _RecordingQueue()
    monkeypatch.setattr(async_rollout, "_require_distributed_context", lambda owner: TTT_RANK)
    with expect_error(RuntimeError, "AsyncRolloutClient must run on TTML_RANK=0"):
        async_rollout.AsyncRolloutClient(bridge=bridge, queue=queue_)
    monkeypatch.setattr(async_rollout, "_require_distributed_context", lambda owner: TTML_RANK)
    with expect_error(RuntimeError, "AsyncRolloutServer must run on TTT_RANK=1"):
        _async_rollout_server(bridge, queue_, _EchoSampler(), [])
    assert not bridge.connected and not queue_.connected


def test_async_rollout_client_shuts_down_at_exit(monkeypatch):
    monkeypatch.setattr(async_rollout, "_require_distributed_context", lambda owner: TTML_RANK)
    registered = []
    monkeypatch.setattr(async_rollout.atexit, "register", registered.append)
    bridge, queue_ = _ScriptedBridge([]), _RecordingQueue()
    client = async_rollout.AsyncRolloutClient(bridge=bridge, queue=queue_)
    assert registered == [client.shutdown]
    registered[0]()
    assert bridge.closed and queue_.closed


class _BlockingSendContext(_LoopbackContext):
    """A _LoopbackContext whose send returns only once the peer has received the message, as MPI_Send may."""

    def send(self, data, dest, tag):
        super().send(data, dest, tag)
        self._queues[(dest, tag)].join()

    def recv(self, nbytes, source, tag):
        data = super().recv(nbytes, source, tag)
        self._queues[(1 - source, tag)].task_done()
        return data


class _HostPad:
    """A ttnn.Tensor of the ThreadedWeightBridge, kept in host memory."""

    dtype, layout = ttnn.bfloat16, ttnn.TILE_LAYOUT

    def __init__(self, shape):
        self.shape = [int(d) for d in shape]
        self.value = torch.zeros(self.shape)

    def memory_config(self):
        return ttnn.DRAM_MEMORY_CONFIG


class _CountingSampler(_EchoSampler):
    def __init__(self):
        super().__init__()
        self.installed = []

    def update_weights(self, weights, version):
        super().update_weights(weights, version)
        self.installed.append(version)


def _finishes(fn, timeout_s=10.0):
    """Runs ``fn`` in a thread; returns False if it does not return in ``timeout_s``, raises what ``fn`` raised."""
    errors = []

    def run():
        try:
            fn()
        except BaseException as e:
            errors.append(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if errors:
        raise errors[0]
    return not thread.is_alive()


@pytest.fixture
def thread_errors(monkeypatch):
    """exit_on_error of the remote rollout modules records what a thread raised; the test fails if one did."""
    errors = []

    @contextlib.contextmanager
    def record_errors():
        try:
            yield
        except BaseException as e:
            errors.append(e)

    for module in (async_rollout, rollout_queue, threaded_weight_bridge):
        monkeypatch.setattr(module, "exit_on_error", record_errors)
    yield errors
    assert not errors, f"a transport thread raised: {errors!r}"


def _patch_host_ttnn(monkeypatch):
    """ttnn calls of the ThreadedWeightBridge on _HostPads; returns the pads that ttnn.deallocate got."""
    deallocated = []
    monkeypatch.setattr(ttnn, "begin_trace_capture", threaded_weight_bridge._begin_trace_capture)
    monkeypatch.setattr(ttnn, "end_trace_capture", threaded_weight_bridge._end_trace_capture)
    monkeypatch.setattr(ttnn, "allocate_tensor_on_device", lambda shape, *a, **k: _HostPad(shape))
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)
    monkeypatch.setattr(ttnn, "copy", lambda src, dst: setattr(dst, "value", src.value.clone()))
    monkeypatch.setattr(ttnn, "record_event", lambda mesh, cq_id: None)
    monkeypatch.setattr(ttnn, "wait_for_event", lambda cq_id, event: None)
    monkeypatch.setattr(ttnn, "from_device", lambda pad, blocking, queue_id: pad)
    monkeypatch.setattr(ttnn, "event_synchronize", lambda event: None)
    monkeypatch.setattr(ttnn, "to_torch", lambda pad: pad.value.clone())
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda pad: [pad])
    monkeypatch.setattr(ttnn, "from_torch", lambda tensor, **k: tensor)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, pad, cq_id: setattr(pad, "value", host))
    monkeypatch.setattr(threaded_weight_bridge, "_validate_source_tensor", lambda key, tensor: None)
    return deallocated


def test_threaded_weight_bridge_sender_close_deallocates_the_pads(monkeypatch, thread_errors):
    deallocated = _patch_host_ttnn(monkeypatch)
    queues = collections.defaultdict(queue.Queue)
    monkeypatch.setattr(threaded_weight_bridge, "duplicate_distributed_context", lambda: _LoopbackContext(queues))
    sender = threaded_weight_bridge.ThreadedWeightBridge.sender(peer_rank=TTT_RANK, mesh_device="mesh")
    sender.connect()
    sender.send_weights({"a": _HostPad([1]), "b": _HostPad([2])}, version=0)
    pads = list(sender._pads.values())
    assert _finishes(sender.close)
    assert deallocated == pads and sender._pads is None


@pytest.fixture(params=[1, 3], ids=["capacity1", "capacity3"])
def async_rollout_over_blocking_sends(request, monkeypatch, thread_errors):
    """The real AsyncRolloutClient and AsyncRolloutServer over blocking sends; the server serves in a thread.

    Returns (client, sampler, server thread) once both local rollout queues are full of version-0 batches.
    """
    _patch_host_ttnn(monkeypatch)
    bridge_queues, rollout_queues = collections.defaultdict(queue.Queue), collections.defaultdict(queue.Queue)
    monkeypatch.setattr(
        threaded_weight_bridge, "duplicate_distributed_context", lambda: _BlockingSendContext(bridge_queues)
    )
    monkeypatch.setattr(rollout_queue, "duplicate_distributed_context", lambda: _BlockingSendContext(rollout_queues))
    monkeypatch.setattr(
        async_rollout,
        "_require_distributed_context",
        lambda owner: TTML_RANK if owner == "AsyncRolloutClient" else TTT_RANK,
    )
    monkeypatch.setattr(async_rollout.atexit, "register", lambda fn, *args: None)
    monkeypatch.setattr(async_rollout.atexit, "unregister", lambda fn: None)

    sampler = _CountingSampler()
    producer = RolloutQueue.producer(peer_rank=TTML_RANK, capacity=request.param, num_generations=1)
    consumer = RolloutQueue.consumer(peer_rank=TTT_RANK, capacity=request.param, num_generations=1)
    server = _async_rollout_server(
        threaded_weight_bridge.ThreadedWeightBridge.receiver(peer_rank=TTML_RANK, submeshes=["submesh"]),
        producer,
        sampler,
        [_zero_reward],
    )
    client = async_rollout.AsyncRolloutClient(
        bridge=threaded_weight_bridge.ThreadedWeightBridge.sender(peer_rank=TTT_RANK, mesh_device="mesh"),
        queue=consumer,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        client.send_weights({"w": _HostPad([1])}, version=0)
        deadline = time.monotonic() + 30
        while not (producer._local.full() and consumer._local.full()):
            assert not thread_errors, f"a transport thread raised: {thread_errors!r}"
            assert time.monotonic() < deadline, "the rollout queues did not fill in 30 s"
            time.sleep(0.01)
        yield client, sampler, server_thread
    finally:
        _finishes(client.shutdown)
        server_thread.join(10)
        assert not server_thread.is_alive(), "AsyncRolloutServer did not stop"
        assert not thread_errors, f"a transport thread raised: {thread_errors!r}"


def _queue_full_warnings(caplog):
    return [
        r for when in ("setup", "call") for r in caplog.get_records(when) if "rollout queue was full" in r.getMessage()
    ]


def test_async_rollout_weight_pushes_continue_while_the_rollout_queue_is_full(
    async_rollout_over_blocking_sends, caplog
):
    client, sampler, server_thread = async_rollout_over_blocking_sends

    def push_versions_1_to_3():
        for version in (1, 2, 3):
            client.send_weights({"w": _HostPad([1])}, version=version)

    def pop_until_version_3():
        while client.pop().rollout.weight_version != 3:
            pass

    assert _finishes(push_versions_1_to_3)
    assert _finishes(pop_until_version_3)
    assert _finishes(client.shutdown)
    server_thread.join(10)
    assert not server_thread.is_alive()
    assert sampler.installed == [0, 1, 2, 3]
    assert len(_queue_full_warnings(caplog)) == 1


def test_async_rollout_shuts_down_with_a_weight_push_in_flight_and_a_full_rollout_queue(
    async_rollout_over_blocking_sends,
):
    client, sampler, server_thread = async_rollout_over_blocking_sends
    client.send_weights({"w": _HostPad([1])}, version=1)
    client.send_weights({"w": _HostPad([1])}, version=2)
    assert _finishes(client.shutdown)
    server_thread.join(10)
    assert not server_thread.is_alive()
    assert sampler.installed == [0, 1, 2]


@pytest.mark.parametrize("weight_sync_every, num_iterations, max_staleness", [(1, 1, 0), (2, 2, 0)])
def test_async_remote_source_trains_every_batch_over_a_full_rollout_queue(
    monkeypatch, async_rollout_over_blocking_sends, weight_sync_every, num_iterations, max_staleness
):
    client, sampler, _ = async_rollout_over_blocking_sends

    def slow_copy(host, pad, cq_id):
        time.sleep(0.3)
        pad.value = host

    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", slow_copy)
    monkeypatch.setattr(grpo_ttml_model, "weights_ref_hf_dict", lambda model: {"w": _HostPad([1])})
    num_batches = 6
    source = AsyncRemoteRolloutBatchSource(
        client=client,
        num_batches=num_batches,
        batch_prompts=1,
        num_generations=1,
        max_completion_length=4,
        weight_sync_every=weight_sync_every,
        max_staleness=max_staleness,
    )
    trained = []

    def train():
        # Version 0 is already pushed; push at the batch boundaries as GRPOTrainer._train_policy does.
        for batch_index, scored in enumerate(source, start=1):
            trained.append(scored.rollout.weight_version)
            if batch_index < num_batches:
                source.update_weights(None, batch_index * num_iterations)

    assert _finishes(train, timeout_s=30)
    assert len(trained) == num_batches
    assert trained == sorted(trained) and trained[-1] > 0


class _FakeAsyncClient:
    """pop() returns batches generated at the scripted versions, then None; records the pushed versions."""

    def __init__(self, versions, make_batch=_scored_batch):
        self._batches = [make_batch(version) for version in versions]
        self.pushed = []

    def pop(self):
        return self._batches.pop(0) if self._batches else None

    def send_weights(self, weights, *, version):
        self.pushed.append(version)

    def shutdown(self):
        pass


def _async_source(monkeypatch, client, num_batches, max_staleness, weight_sync_every=1, batch_prompts=1):
    monkeypatch.setattr(grpo_ttml_model, "weights_ref_hf_dict", lambda model: {})
    return AsyncRemoteRolloutBatchSource(
        client=client,
        num_batches=num_batches,
        batch_prompts=batch_prompts,
        num_generations=2,
        max_completion_length=4,
        weight_sync_every=weight_sync_every,
        max_staleness=max_staleness,
    )


_ASYNC_HOST_CONFIG = dict(
    rollout_backend="ttt_remote_async",
    remote_rollout=RemoteRolloutConfig(mesh_shape=[1, 1], max_batch_size=4, max_seq_len=64),
)


def _staleness_and_drops(run):
    return [(m["rollout_staleness_max"], m["rollouts_dropped"]) for m in run.metrics_by_step.values()]


def test_async_remote_source_drops_stale_rollouts(monkeypatch):
    client = _FakeAsyncClient([0, 0, 1, 1, 2], make_batch=_host_scored_batch)
    source = _async_source(monkeypatch, client, num_batches=3, max_staleness=0, batch_prompts=2)

    run = _run_train_loop_on_host(monkeypatch, num_batches=3, source=source, **_ASYNC_HOST_CONFIG, num_iterations=1)
    assert _staleness_and_drops(run) == [(0, 0), (0, 1), (0, 1)]
    assert client.pushed == [0, 1, 2]


def test_async_remote_source_counts_staleness_in_optimizer_steps(monkeypatch):
    client = _FakeAsyncClient([0, 0, 2, 4], make_batch=_host_scored_batch)
    source = _async_source(monkeypatch, client, num_batches=3, max_staleness=1, weight_sync_every=2, batch_prompts=2)

    run = _run_train_loop_on_host(
        monkeypatch, num_batches=3, source=source, **_ASYNC_HOST_CONFIG, weight_sync_every=2, max_staleness=1
    )
    assert _staleness_and_drops(run) == [(0, 0), (1, 0), (0, 1), (1, 0), (0, 0), (1, 0)]
    assert client.pushed == [0, 2, 4]


def test_async_remote_source_update_weights_returns_zero_when_sync_not_due(monkeypatch):
    client = _FakeAsyncClient([0])
    source = _async_source(monkeypatch, client, num_batches=1, max_staleness=0, weight_sync_every=3)

    source.update_weights(None, 0)
    assert source.update_weights(None, 1) == 0.0
    assert client.pushed == [0]


def test_async_remote_source_raises_when_the_queue_closes(monkeypatch, expect_error):
    source = _async_source(monkeypatch, _FakeAsyncClient([]), num_batches=1, max_staleness=0)
    with expect_error(RuntimeError, "rollout rank closed the rollout queue"):
        next(iter(source))


def test_async_remote_source_rejects_a_batch_with_more_prompts_than_planned(expect_error):
    source = AsyncRemoteRolloutBatchSource(
        client=_FakeAsyncClient([0]),
        num_batches=1,
        batch_prompts=1,
        num_generations=1,
        max_completion_length=4,
        weight_sync_every=1,
        max_staleness=0,
    )
    with expect_error(RuntimeError, "expected 1 rows, got 2 prompts"):
        next(iter(source))


QWEN3_MODEL_ID = "Qwen/Qwen3-0.6B"
QWEN3_TRANSFORMER_CONFIG = TransformerConfig(
    {"transformer_config": {"model_type": "qwen3", "max_sequence_length": 256, "runner_type": "memory_efficient"}}
)


class _ReplayedRollouts(RolloutBatchSource):
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def update_weights(self, model, step):
        return 0.0


class _FirstBlockGrads(TrainerCallback):
    def __init__(self):
        self.grads = {}

    def on_before_optimizer_step(self, trainer):
        for name, param in trainer.model.parameters().items():
            if "/blocks/0/" in name:
                grad = param.get_grad_tensor()
                self.grads[name] = np.asarray(ttml.Sharding.from_tensor(grad).gather(grad), dtype=np.float32)


def _first_block_grads_after_one_step(fresh_device_mesh, device_config, gradient_accumulation_steps, scored=None):
    """One GRPO step of Qwen3-0.6B on a fresh mesh; generates the scored batch unless ``scored`` is given."""
    recorder = _FirstBlockGrads()
    mesh = build_mesh(device_config)
    with fresh_device_mesh(mesh.shape, mesh.axis_names, what="GRPO device-count invariance"):
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_ID)
        countries = ["France", "Portugal", "Japan", "Peru"]
        dataset = Dataset.from_dict(
            {"prompt": [_to_capitals_chat_prompt(tokenizer, f"The capital of {c} is") for c in countries]}
        )
        config = GRPOConfig(
            **{
                **_HOST_GRPO_CONFIG,
                "prompts_to_train": 4,
                "max_completion_length": 8,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "disable_default_monitor": True,
            }
        )
        trainer = GRPOTrainer(
            transformer_config=QWEN3_TRANSFORMER_CONFIG,
            device_config=device_config,
            model_source=QWEN3_MODEL_ID,
            dataset_func=lambda: dataset,
            config=config,
            reward_func=lambda completions, **_: [float(i * i) for i in range(len(completions))],
            optimizer_dict={"type": "AdamW", "lr": 1e-5},
            callbacks=[recorder],
        )
        scored = scored or list(trainer.rollout_batch_source)
        trainer.rollout_batch_source = _ReplayedRollouts(scored)
        trainer.train()
        del trainer
        ttml.autograd.AutoContext.get_instance().reset_graph()
    return recorder.grads, scored


@pytest.mark.requires_device
@pytest.mark.parametrize("parallelism", ["enable_ddp", "enable_fsdp"])
def test_grpo_gradients_match_across_device_counts(fresh_device_mesh, parallelism):
    """Gradients don't depend on the device count.

    One step on the same 8 completions: 1 device runs two micro-batches of 4, and a
    [1, 2] mesh runs one micro-batch with 4 completions per device. The first block's
    gradients must match within 2%; bf16 sums them in a different order, while a wrong
    loss normalization would be off by about 2x.
    """
    one_device = DeviceConfig({"device_config": {"mesh_shape": [1, 1]}})
    two_devices = DeviceConfig({"device_config": {parallelism: True, "mesh_shape": [1, 2]}})

    actual, scored = _first_block_grads_after_one_step(fresh_device_mesh, two_devices, 1)
    expected, _ = _first_block_grads_after_one_step(fresh_device_mesh, one_device, 2, scored)

    assert expected and actual.keys() == expected.keys()
    for name, grad in expected.items():
        error = np.linalg.norm(actual[name] - grad) / np.linalg.norm(grad)
        assert error < 2e-2, f"{name}: relative gradient error {error:.3g}"


QWEN3_BUILDS = {
    "one_device": {"mesh_shape": [1, 1]},
    "fsdp_lazy": {"enable_fsdp": True, "mesh_shape": [1, 2]},
    "fsdp_eager": {"enable_fsdp": True, "mesh_shape": [1, 2], "lazy_parameter_init": False},
}


def _tiny_fp32_qwen3_checkpoint(directory) -> str:
    """A two-layer random Qwen3 in HF format, saved in fp32, with Qwen3-0.6B's tokenizer next to it."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    Qwen3ForCausalLM(config).save_pretrained(directory)
    AutoTokenizer.from_pretrained(QWEN3_MODEL_ID).save_pretrained(directory)
    return str(directory)


def _gathered(model) -> dict:
    """Every parameter as one host array in its stored dtype, whatever its mesh layout."""
    return {name: ttml.Sharding.from_tensor(param).gather(param) for name, param in model.parameters().items()}


def _assert_same_values(actual: dict, expected: dict) -> None:
    assert sorted(actual) == sorted(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values, err_msg=name)


@pytest.mark.requires_device
@pytest.mark.parametrize("build", list(QWEN3_BUILDS))
def test_qwen3_builds_load_and_resume_fp32_master_weights(fresh_device_mesh, tmp_path, gradient_reduce_default, build):
    """Every Qwen3 build shards, applies the precision block and then loads: one device, FSDP built lazily (the
    default) and FSDP built eagerly. From an fp32 HF file each gets fp32 master weights with the file's values
    unrounded, the FSDP builds hold what the one-device build holds, and a trainer checkpoint resumes bit for bit."""
    hf_dir = _tiny_fp32_qwen3_checkpoint(tmp_path / "hf")

    def setup(device_config, model_source):
        return grpo_ttml_model.setup_ttml_model(
            QWEN3_TRANSFORMER_CONFIG, device_config, model_source, precision=FP32_PRECISION
        )

    def on_fresh_mesh(device_config):
        mesh = build_mesh(device_config)
        return fresh_device_mesh(mesh.shape, mesh.axis_names, what=f"Qwen3 with fp32 master weights ({build})")

    one_device_values = None
    if build != "one_device":
        one_device = DeviceConfig({"device_config": QWEN3_BUILDS["one_device"]})
        with on_fresh_mesh(one_device):
            model, _ = setup(one_device, hf_dir)
            one_device_values = _gathered(model)
            del model
            ttml.autograd.AutoContext.get_instance().reset_graph()

    device_config = DeviceConfig({"device_config": QWEN3_BUILDS[build]})
    with on_fresh_mesh(device_config):
        model, tokenizer = setup(device_config, hf_dir)
        assert _stored_dtypes(model) == {ttnn.DataType.FLOAT32}
        loaded = _gathered(model)
        # Rounded to bf16 on the way in, the random weights would all be bf16 values. The norm weights are ones.
        random_weights = {name: values for name, values in loaded.items() if np.ptp(values) > 0}
        assert len(random_weights) > len(loaded) // 2
        for name, values in random_weights.items():
            assert not np.array_equal(values, values.astype(ml_dtypes.bfloat16).astype(np.float32)), name
        if one_device_values is not None:
            _assert_same_values(loaded, one_device_values)

        save_checkpoint(model, 1, str(tmp_path), tokenizer=tokenizer, model_source=hf_dir)
        del model
        resumed, _ = setup(device_config, str(tmp_path / "checkpoints" / "grpo_step_1"))
        assert _stored_dtypes(resumed) == {ttnn.DataType.FLOAT32}
        _assert_same_values(_gathered(resumed), loaded)
        del resumed
        ttml.autograd.AutoContext.get_instance().reset_graph()
