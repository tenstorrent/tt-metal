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
  * Skip the HuggingFace weight download by monkey-patching
    ``_download_hf_repo`` and ``load_from_safetensors`` in
    ``ttml.trainers.grpo_trainer.grpo_ttml_model`` to no-ops; the model keeps its random init.
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
import os
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ttml
import ttnn
from datasets import Dataset
from safetensors.numpy import save_file
from transformers import AutoTokenizer, LlamaConfig as HFLlamaConfig, LlamaForCausalLM

from ttml.common.config import DeviceConfig, TransformerConfig
from ttml.common.utils import build_mesh, repeat_interleave
from ttml.models.llama import load_from_safetensors
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
from ttml.trainers.grpo_trainer.remote_rollout import mpi_rollout, sync_remote_rollout_batch_source
from ttml.trainers.grpo_trainer.remote_rollout.sync_remote_rollout_batch_source import SyncRemoteRolloutBatchSource
from ttml.trainers.grpo_trainer.remote_rollout.weight_bridge import TTML_RANK, TTT_RANK
from ttml.trainers.grpo_trainer.rollout_batch_source import (
    InProcessRolloutBatchSource,
    RolloutBatchSource,
    score_rollout,
    validate_rollout_batch,
)
from ttml.trainers.grpo_trainer.ttml_rollout_sampler import TTMLRolloutSampler

HF_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"  # not gated


TINY_TRANSFORMER_CONFIG = TransformerConfig(
    {
        "transformer_config": {
            "model_type": "llama",
            "num_heads": 2,
            "num_groups": 1,
            "embedding_dim": 64,
            "intermediate_dim": 128,
            "dropout_prob": 0.0,
            "num_blocks": 1,
            "weight_tying": "enabled",
            # Overwritten by ``len(tokenizer)`` inside the sampler ctor.
            "vocab_size": 32000,
            "max_sequence_length": 128,
            "runner_type": "memory_efficient",
            "theta": 500000.0,
            "rope_scaling": {
                "scaling_factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_context_length": 8192,
            },
        }
    }
)

LLAMA_1B_TRANSFORMER_CONFIG = TransformerConfig(
    {
        "transformer_config": {
            "model_type": "llama",
            "num_heads": 32,
            "num_groups": 8,
            "embedding_dim": 2048,
            "intermediate_dim": 8192,
            "dropout_prob": 0.0,
            "num_blocks": 16,
            "weight_tying": "enabled",
            "vocab_size": 32000,
            "max_sequence_length": 1024,
            "runner_type": "memory_efficient",
            "theta": 500000.0,
            "rope_scaling": {
                "scaling_factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_context_length": 8192,
            },
        }
    }
)

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


@pytest.fixture
def patch_llama_weight_loading(monkeypatch):
    """Skip the HF download / safetensors load so the tiny model keeps random init.

    Both names are patched on ``grpo_ttml_model`` only. ``_download_hf_repo`` wraps
    ``huggingface_hub.snapshot_download``, which is left alone because transformers
    also uses it to fetch the tokenizer this test loads. ``load_from_safetensors`` is
    bound there with ``from ... import``, so patching ``ttml.models.llama`` would not
    reach it.
    """
    monkeypatch.setattr(grpo_ttml_model, "_download_hf_repo", lambda model_source: "/tmp/unused")
    monkeypatch.setattr(grpo_ttml_model, "load_from_safetensors", lambda *args, **kwargs: None)


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
        is_kv_slice = key.endswith(("k_proj.weight", "v_proj.weight"))
        assert (tensor.buffer_address() in live_buffers) != is_kv_slice, key
    assert exported["model.embed_tokens.weight"].buffer_address() == exported["lm_head.weight"].buffer_address()

    def to_torch(tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])

    kv_dim = cfg.num_key_value_heads * cfg.hidden_size // cfg.num_attention_heads
    for i in range(cfg.num_hidden_layers):
        kv = to_torch(params[f"Llama/blocks/{i}/attention/kv_linear/weight"].get_value())
        assert torch.equal(to_torch(exported[f"model.layers.{i}.self_attn.k_proj.weight"]), kv[:, :, :kv_dim])
        assert torch.equal(to_torch(exported[f"model.layers.{i}.self_attn.v_proj.weight"]), kv[:, :, kv_dim:])

    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
def test_save_checkpoint_round_trip_llama(patch_llama_weight_loading, tmp_path):
    model, tokenizer = grpo_ttml_model.setup_ttml_model(TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, HF_MODEL_ID)
    save_checkpoint(model, 1, str(tmp_path), tokenizer=tokenizer, model_source=HF_MODEL_ID)

    ckpt_dir = str(tmp_path / "checkpoints" / "grpo_step_1")
    loaded, _ = grpo_ttml_model.setup_ttml_model(TINY_TRANSFORMER_CONFIG, DEVICE_CONFIG, ckpt_dir)
    _assert_same_parameters(model, loaded)

    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_save_checkpoint_header_records_given_learning_rate(monkeypatch, tmp_path):
    headers = []
    monkeypatch.setattr(ttml.checkpointing, "save_checkpoint", lambda path, header, **_: headers.append(header))
    optimizer = SimpleNamespace(get_lr=lambda: 0.75)

    save_checkpoint(SimpleNamespace(parameters=lambda: {}), 2, str(tmp_path), optimizer=optimizer, learning_rate=0.5)

    assert headers[0]["learning_rate"] == 0.5


def _save_old_format_checkpoint(directory):
    save_file({"Llama/fc/weight": np.zeros((2, 2), np.float32)}, str(directory / "model.safetensors"))


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_setup_ttml_model_rejects_old_checkpoint_format(tmp_path, expect_error, model_type):
    _save_old_format_checkpoint(tmp_path)
    transformer_config = TransformerConfig({"transformer_config": {"model_type": model_type}})
    with expect_error(ValueError, "old GRPO checkpoint format"):
        grpo_ttml_model.setup_ttml_model(transformer_config, DEVICE_CONFIG, str(tmp_path))


def test_llama_strict_load_raises_on_unloaded_parameters(tmp_path, expect_error):
    _save_old_format_checkpoint(tmp_path)
    model = SimpleNamespace(parameters=lambda: {"Llama/fc/weight": None})
    config = SimpleNamespace(
        use_tp=False, num_attention_heads=1, num_key_value_heads=1, weight_tying=None, num_hidden_layers=0
    )
    with expect_error(RuntimeError, "1 model parameters were NOT loaded"):
        load_from_safetensors(model, tmp_path, config, strict=True)


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


def test_get_grpo_config_reads_rollout_backend():
    cfg = get_grpo_config(
        {"training_config": {"grpo_config": {**_GRPO_CONFIG_FIELDS, "rollout_backend": "ttml_in_process_sync"}}}
    )
    assert cfg.rollout_backend == "ttml_in_process_sync"


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
    and ``completions_per_prompt=1`` the outputs must match exactly; any drift
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

    def decode(batch):
        return [tokenizer.decode(c, skip_special_tokens=False) for c in batch.completions]

    single_outputs = []
    for prompt in prompts:
        completions = decode(sampler.generate([prompt]))
        assert len(completions) == 1
        single_outputs.append(completions[0])

    batched_outputs = decode(sampler.generate(prompts))
    assert len(batched_outputs) == len(prompts)

    assert batched_outputs == single_outputs, (
        "Mismatch between one-by-one and batched outputs.\n" f"single={single_outputs}\n" f"batch={batched_outputs}"
    )

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
_REWARDS = [0.0, 1.0, 0.0, 1.0]
_BEHAVIOR_LOGPROB = -0.5
_BASE_LR = 1.0


def _run_train_loop_on_host(monkeypatch, num_batches=1):
    """Run ``GRPOTrainer.train()`` on host fakes: ``num_batches`` scored batches of 2 prompts x 2 generations,
    ``num_iterations=2``, a checkpoint every step. The fake scheduler halves the LR each step."""
    from ttml.trainers import GRPOMonitor
    from ttml.trainers.grpo_trainer.rollout_batch_source import ScoredRolloutBatch

    run = SimpleNamespace(
        clock=0.0, batches_drawn=0, events=[], pi_old=[], saved_lrs=[], metrics_by_step={}, update_weights_steps=[]
    )
    monkeypatch.setattr(grpo_trainer_module, "time", SimpleNamespace(perf_counter=lambda: run.clock))

    max_len = _HOST_GRPO_CONFIG["max_completion_length"]
    rollout = RolloutBatch(
        weight_version=0,
        prompts=[[1, 2], [1, 2], [3, 4], [3, 4]],
        completions=[[7], [7], [7], [7]],
        logprobs=np.full((4, max_len), _BEHAVIOR_LOGPROB, dtype=np.float32),
    )

    class _Source(RolloutBatchSource):
        def __iter__(self):
            run.batches_drawn += 1
            for _ in range(num_batches):
                run.clock += _GENERATE_TIME_S
                yield ScoredRolloutBatch(rollout, np.array(_REWARDS, dtype=np.float32), {}, _GENERATE_TIME_S, 0.0)

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

    cfg = GRPOConfig(**{**_HOST_GRPO_CONFIG, "num_iterations": 2, "checkpointing": True, "checkpoint_interval": 1})
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
        self.rollout_batch_source = _Source()

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
        optimizer_dict={},
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
    """Replace MPI with one in-memory queue per tag; returns (queues, atexit hooks)."""
    queues = collections.defaultdict(queue.Queue)
    at_exit = []
    monkeypatch.setattr(mpi_rollout, "_mpi_send_bytes", lambda data, peer, tag: queues[tag].put(bytes(data)))
    monkeypatch.setattr(mpi_rollout, "_mpi_recv_bytes", lambda n, peer, tag: queues[tag].get(timeout=30))
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


def _remote_grpo_config():
    return GRPOConfig(
        **{
            **_HOST_GRPO_CONFIG,
            "rollout_backend": "ttt_remote_sync",
            "remote_rollout": RemoteRolloutConfig(mesh_shape=[1, 1], max_batch_size=1, max_seq_len=64),
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
    monkeypatch.setattr(sync_remote_rollout_batch_source, "weights_ref_hf_dict", lambda model: {})
    return GRPOTrainer(
        transformer_config=SimpleNamespace(model_type="llama"),
        device_config=SimpleNamespace(enable_fsdp=False),
        model_source=HF_MODEL_ID,
        dataset_func=lambda: None,
        config=_remote_grpo_config(),
        reward_funcs=reward_funcs,
        optimizer_dict={},
        callbacks=list(callbacks),
    )


@pytest.mark.parametrize(
    "reward_func, callbacks, message",
    [(_failing_reward, [], "reward failed"), (_zero_reward, [_FailingTrainBegin()], "on_train_begin failed")],
)
def test_failed_trainer_shuts_down_the_rollout_server(
    mpi_loopback, monkeypatch, expect_error, reward_func, callbacks, message
):
    """A trainer that fails before or during training sends OP_SHUTDOWN; the rollout server stops and closes its
    sampler, and the atexit backstop sends nothing more."""
    queues, at_exit = mpi_loopback
    sampler = _EchoSampler()
    client, server_thread = _start_rollout_server(sampler)
    trainer = _trainer_with_remote_rollout(monkeypatch, client, reward_funcs=[reward_func], callbacks=callbacks)

    with expect_error(ValueError, message):
        trainer.train()

    server_thread.join(timeout=30)
    assert not server_thread.is_alive()
    assert sampler.closed
    assert at_exit == [client.shutdown]
    at_exit[0]()
    assert queues[mpi_rollout._INFER_REQ_HDR_TAG].empty()


@pytest.mark.parametrize(
    "model_type, enable_fsdp, local_source, message",
    [
        ("mistral", False, False, "supports Llama and Qwen3 models only"),
        ("llama", True, False, "does not support FSDP"),
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


def test_ttt_stop_and_pad_delegates_to_shared_stop_token_resolver(monkeypatch):
    from ttml.trainers.grpo_trainer.remote_rollout import ttt_presets

    tokenizer = SimpleNamespace(pad_token_id=7, eos_token_id=3, unk_token_id=0)
    called = []
    monkeypatch.setattr(ttt_presets.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(
        ttt_presets,
        "resolve_stop_token_ids",
        lambda tok: called.append(tok) or {tokenizer.eos_token_id, tokenizer.pad_token_id, 11},
    )

    stop_ids, pad_token_id = ttt_presets.stop_and_pad(HF_MODEL_ID)

    assert called == [tokenizer]
    assert stop_ids == [3, 7, 11]
    assert pad_token_id == tokenizer.pad_token_id


def test_ttt_stop_and_pad_includes_tokenizer_pad_id_in_stop_set(monkeypatch):
    from ttml.common.utils import resolve_stop_token_ids
    from ttml.trainers.grpo_trainer.remote_rollout import ttt_presets

    class _FakeTokenizer:
        eos_token_id = 5
        pad_token_id = 17
        unk_token_id = 0

        def __init__(self):
            self._ids = {
                "<|eot_id|>": 31,
                "<|end_of_text|>": 5,
                "<|eom_id|>": self.unk_token_id,
                "<|im_end|>": -1,
                "<|endoftext|>": 63,
            }

        def convert_tokens_to_ids(self, token):
            return self._ids.get(token, self.unk_token_id)

    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(ttt_presets.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)

    stop_ids, pad_token_id = ttt_presets.stop_and_pad(HF_MODEL_ID)
    expected_stop_ids = resolve_stop_token_ids(tokenizer)

    assert pad_token_id == tokenizer.pad_token_id
    assert tokenizer.pad_token_id in expected_stop_ids
    assert set(stop_ids) == expected_stop_ids
    assert stop_ids == sorted(stop_ids)


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
    monkeypatch.setattr(sync_remote_rollout_batch_source, "weights_ref_hf_dict", lambda model: {})
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
