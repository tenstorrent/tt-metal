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
    ``snapshot_download`` and ``load_from_safetensors`` in
    ``ttml.trainers.grpo_trainer.ttml_rollout_sampler`` to no-ops; the model keeps its random init.
  * ``max_completion_length=4`` so autoregressive generation is cheap.
  * Exactly one optimizer step (``gradient_accumulation_steps=1``,
    ``num_iterations=1``, ``prompts_to_train=2``). On this single device
    ``per_device_train_batch_size=4`` and ``num_generations=2`` give a
    per-micro-batch prompt count of ``4 * 1 / 2 = 2``, and with
    ``gradient_accumulation_steps=1`` the generation batch is also 2 prompts
    (one micro-batch per optimizer step).
  * ``num_generations=2`` is the minimum that yields a non-zero advantage
    (group of 1 -> mean == reward -> advantage == 0 -> loss == 0).

Note on CI: the tokenizer load uses ``meta-llama/Llama-3.2-1B-Instruct``,
which is a gated HuggingFace repo. To run this in CI without an
``HF_TOKEN`` secret, swap ``HF_MODEL_ID`` below for a non-gated mirror
(e.g. ``unsloth/Llama-3.2-1B-Instruct``).
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pytest

import ttml
import ttnn
from datasets import Dataset
from transformers import AutoTokenizer

from ttml.common.config import DeviceConfig, TransformerConfig
from ttml.modules import RunMode
from ttml.trainers import GRPOConfig, GRPOTrainer, TrainerCallback, get_grpo_config
from ttml.trainers.grpo_trainer import layout_microbatch, place_old_nlog_probs
from ttml.trainers.grpo_trainer.ttml_rollout_sampler import TTMLRolloutSampler


# The ``LlamaGRPOCompleter`` reference implementation lives under the examples
# tree, not under ``ttml`` proper. Surface its package on the import path so
# this test can use it.
_EXAMPLES_DIR = os.path.join(
    os.environ.get("TT_METAL_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    "tt-train",
    "sources",
    "examples",
)
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from grpo.utils.llama_completer import LlamaCompletionCtx, LlamaGRPOCompleter  # noqa: E402


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
    """Override ``TTMLRolloutSampler.setup_device`` (and the reference
    ``LlamaGRPOCompleter.setup_device``) to reuse the already-open AutoContext
    device instead of calling ``open_device`` again.

    Other tests in ``tests/python/`` lazily open the AutoContext device on
    first tensor use and never close it. When pytest collects this file
    alongside them, the device is already open by the time we get here, so
    the default ``setup_device`` would trip ``open_device was called after
    the device was created``. Reusing the live device sidesteps the issue
    without leaking device-management code into the test body.
    """
    reuse = lambda self, device_config: ttml.autograd.AutoContext.get_instance().get_device()  # noqa: E731
    monkeypatch.setattr(TTMLRolloutSampler, "setup_device", reuse)
    monkeypatch.setattr(LlamaGRPOCompleter, "setup_device", reuse)


class _RecordingCallback(TrainerCallback):
    """Records hook invocations so we can assert the trainer drove them."""

    def __init__(self) -> None:
        self.train_begin = 0
        self.before_step = 0
        self.step_end = 0
        self.train_end = 0
        self.last_step_metrics: dict | None = None

    def on_train_begin(self, trainer):
        self.train_begin += 1

    def on_before_optimizer_step(self, trainer):
        self.before_step += 1

    def on_step_end(self, trainer, step, **kwargs):
        self.step_end += 1
        self.last_step_metrics = {"step": step, **kwargs}

    def on_train_end(self, trainer):
        self.train_end += 1


@pytest.fixture
def patch_llama_weight_loading(monkeypatch):
    """Skip the HF download / safetensors load so the tiny model keeps random init.

    The sampler module binds both names with ``from ... import``, so they must be
    patched on that module, not on ``huggingface_hub`` / ``ttml.models.llama``.
    """
    from ttml.trainers.grpo_trainer import ttml_rollout_sampler

    monkeypatch.setattr(ttml_rollout_sampler, "snapshot_download", lambda *args, **kwargs: "/tmp/unused")
    monkeypatch.setattr(ttml_rollout_sampler, "load_from_safetensors", lambda *args, **kwargs: None)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "lr_scheduler_type, warmup_steps, use_clip_grad_norm, expected_first_lr_scale",
    [
        ("constant", 0, False, 1.0),
        ("cosine", 0, True, 1.0),
        ("constant_with_warmup", 2, False, 0.5),
    ],
)
def test_grpo_trainer_one_step_smoke(
    patch_llama_weight_loading, tmp_path, lr_scheduler_type, warmup_steps, use_clip_grad_norm, expected_first_lr_scale
):
    """One full GRPO optimizer step on a tiny random-init Llama.

    Asserts the loop reaches every callback hook, produces finite metrics,
    and actually mutates model weights via ``optimizer.step``.
    """
    np.random.seed(0)

    grpo_cfg = GRPOConfig(
        epsilon=0.2,
        per_device_train_batch_size=4,
        num_iterations=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        output_dir=str(tmp_path),
        checkpointing=False,
        checkpoint_interval=999,
        prompts_to_train=2,
        temperature=1.0,
        max_completion_length=4,
        num_generations=2,
        warmup_steps=warmup_steps,
        rollout_source="ttml",
        lr_scheduler_type=lr_scheduler_type,
        use_clip_grad_norm=use_clip_grad_norm,
    )

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    user_prompts = ["What is 1+1?", "Name a color."]
    dataset = Dataset.from_dict(
        {
            "prompt": [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in user_prompts
            ],
            "answer": ["2", "blue"],
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
        dataset=dataset,
        config=grpo_cfg,
        reward_func=reward_func,
        optimizer_dict=optimizer_dict,
        callbacks=[recorder],
    )
    sampler = trainer.rollout_sampler
    assert isinstance(sampler, TTMLRolloutSampler)
    assert trainer.model is sampler.model and trainer.tokenizer is sampler.tokenizer

    # Snapshot a single parameter so we can prove training mutated it.
    params = trainer.model.parameters()
    assert params, "tiny model should expose at least one parameter"
    snapshot_name, snapshot_param = next(iter(params.items()))
    before = snapshot_param.to_numpy(ttnn.DataType.FLOAT32).copy()

    trainer.train()

    assert trainer.model.get_run_mode() == RunMode.TRAIN, "the sampler must leave the shared model in train mode"
    assert sampler.weight_version == 1, "trainer should publish the new weight version after the optimizer step"

    assert recorder.train_begin == 1, "on_train_begin should fire exactly once"
    assert recorder.before_step == 1, "on_before_optimizer_step should fire once for the single step"
    assert recorder.step_end == 1, "on_step_end should fire once for the single step"
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
    assert metrics["lr"] == pytest.approx(
        optimizer_dict["lr"] * expected_first_lr_scale
    ), "first-step LR does not match the scheduler contract"
    for key in ("reward_mean", "reward_std", "mean_completion_len", "generation_time_s"):
        assert key in metrics, f"missing metric {key}"
        assert np.isfinite(metrics[key]), f"metric {key} is not finite: {metrics[key]}"

    # ``step_time_s`` is sealed after the non-monitor ``on_step_end`` callbacks
    # and cleared by the end-of-step metrics reset, so read it from the row the
    # default GRPOMonitor wrote.
    with open(tmp_path / "grpo_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, f"expected one metrics row, got {len(rows)}"
    step_time_s = float(rows[0]["step_time_s"])
    assert np.isfinite(step_time_s) and step_time_s > 0.0, f"step_time_s is not a positive duration: {step_time_s}"

    after = trainer.model.parameters()[snapshot_name].to_numpy(ttnn.DataType.FLOAT32)
    assert before.shape == after.shape
    assert not np.array_equal(before, after), (
        f"parameter {snapshot_name!r} was unchanged after one optimizer step; "
        "either backward did not run or the gradient was identically zero"
    )

    ttml.autograd.AutoContext.get_instance().reset_graph()


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


def test_grpo_config_requires_rollout_source(expect_error):
    with expect_error(TypeError, "rollout_source"):
        GRPOConfig(**_GRPO_CONFIG_FIELDS)


def test_grpo_config_rejects_unknown_rollout_source(expect_error):
    with expect_error(ValueError, "'rollout_source' must be one of"):
        GRPOConfig(**_GRPO_CONFIG_FIELDS, rollout_source="vllm")


def test_get_grpo_config_reads_rollout_source():
    cfg = get_grpo_config({"training_config": {"grpo_config": {**_GRPO_CONFIG_FIELDS, "rollout_source": "ttml"}}})
    assert cfg.rollout_source == "ttml"


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
    the same four prompts through ``LlamaGRPOCompleter.generate_str`` twice:
    once one prompt at a time, once as a single batch. With temperature=0
    and ``num_generations=1`` the outputs must match exactly; any drift
    indicates a batching / padding / mask bug in the generation path.
    """
    completer = LlamaGRPOCompleter(
        ctx=LlamaCompletionCtx(
            max_tokens_to_complete=256,
            temperature=0.0,
            completions_per_prompt=1,
        ),
        transformer_config=LLAMA_1B_TRANSFORMER_CONFIG,
        device_config=DEVICE_CONFIG,
        model_source=HF_MODEL_ID,
    )

    tokenizer = completer.tokenizer
    user_prompts = [
        "The capital of France is",
        "The capital of Portugal is",
        "The capital of United Kingdom is",
        "The capital of Czech Republic is",
    ]
    prompts = [_to_capitals_chat_prompt(tokenizer, p) for p in user_prompts]

    single_outputs = []
    for prompt in prompts:
        completions = completer.generate_str([prompt])
        assert len(completions) == 1
        single_outputs.append(completions[0])

    batched_outputs = completer.generate_str(prompts)
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


def test_validate_clip_grad_norm_rejects_sharded_params(monkeypatch, expect_error):
    from types import SimpleNamespace

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
_REWARDS = [0.0, 1.0, 0.0, 1.0]
_BASE_LR = 1.0


def _run_train_loop_on_host(monkeypatch):
    """Run ``GRPOTrainer.train()`` on host fakes: one rollout of 2 prompts x 2 generations,
    ``num_iterations=2``, a checkpoint every step. The fake scheduler halves the LR each step."""
    from types import SimpleNamespace

    from ttml.trainers import GRPOMonitor, grpo_trainer

    run = SimpleNamespace(clock=0.0, nlog_calls=0, events=[], pi_old=[], saved_lrs=[], metrics_by_step={})
    monkeypatch.setattr(grpo_trainer, "time", SimpleNamespace(perf_counter=lambda: run.clock))

    class _Tensor:
        def set_requires_grad(self, _):
            pass

    class _Completer:
        def generate(self, prompts):
            run.clock += _GENERATE_TIME_S
            return [[7] for _ in range(len(prompts) * 2)]

        def compute_nlog_probs(self, prompts, completions):
            run.nlog_calls += 1
            return _Tensor(), _Tensor()

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

    def fake_setup(self):
        self.model = SimpleNamespace(train=lambda: None, eval=lambda: None)
        self._tokenizer = SimpleNamespace(decode=lambda ids, **_: str(ids))
        self._optimizer = optimizer
        self._lr_scheduler = SimpleNamespace(step=lambda: setattr(optimizer, "lr", optimizer.lr / 2))
        self._monitor_cb = monitor
        self._completions_per_microbatch = 4
        self._generation_batch_prompts = 2
        self._prompts = [[1], [2]]
        self._extra_dataset_columns = {}

    monkeypatch.setattr(GRPOTrainer, "_setup", fake_setup)
    monkeypatch.setattr(
        GRPOTrainer, "_compute_loss_and_backward", lambda self, p, c, adv, nlog, mask, n: run.pi_old.append(nlog)
    )
    monkeypatch.setattr(grpo_trainer, "save_checkpoint", fake_save_checkpoint)

    GRPOTrainer(
        completer=_Completer(),
        dataset=None,
        config=cfg,
        reward_func=lambda completions, **_: list(_REWARDS),
        optimizer_dict={},
        callbacks=[_Recorder(), monitor],
    ).train()
    return run


def test_train_loop_reuses_pi_old_across_mini_epochs(monkeypatch):
    run = _run_train_loop_on_host(monkeypatch)
    assert run.nlog_calls == 1
    mini_epoch_1_pi_old, mini_epoch_2_pi_old = run.pi_old
    # With num_iterations > 1, pi_old must not be recomputed between mini-epochs.
    assert mini_epoch_1_pi_old is mini_epoch_2_pi_old


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
    assert step_1["_Recorder_time_s"] == step_2["_Recorder_time_s"] == _ON_SAVE_TIME_S
    assert "step_1_only" in step_1 and "step_1_only" not in step_2


def test_train_loop_step_time_includes_checkpoint(monkeypatch):
    step_1, step_2 = _run_train_loop_on_host(monkeypatch).metrics_by_step.values()
    assert step_1["step_time_s"] == _GENERATE_TIME_S + _SAVE_CHECKPOINT_TIME_S + _ON_SAVE_TIME_S
    assert step_2["step_time_s"] == _SAVE_CHECKPOINT_TIME_S + _ON_SAVE_TIME_S


def test_train_loop_checkpoint_records_step_lr(monkeypatch):
    run = _run_train_loop_on_host(monkeypatch)
    step_lrs = [metrics["lr"] for metrics in run.metrics_by_step.values()]
    assert run.saved_lrs == step_lrs == [_BASE_LR, _BASE_LR / 2]


def _make_muon_with_adamw():
    from ttml.common.muon_optimizer import MuonWithAdamW

    return MuonWithAdamW({"muon": {"lr": 0.02}, "adamw": {"lr": 0.0003}}, ttml.NamedParameters())


@pytest.mark.parametrize(
    "lr_scheduler_type, warmup_steps",
    [("constant", 0), ("constant", 2), ("constant_with_warmup", 0), ("constant_with_warmup", 1)],
)
def test_build_lr_scheduler_returns_none_for_constant_lr(lr_scheduler_type, warmup_steps):
    from ttml.trainers.grpo_trainer import _build_lr_scheduler

    cfg = GRPOConfig(**{**_HOST_GRPO_CONFIG, "lr_scheduler_type": lr_scheduler_type, "warmup_steps": warmup_steps})
    assert _build_lr_scheduler(_make_muon_with_adamw(), cfg, 4) is None


@pytest.mark.parametrize(
    "lr_scheduler_type, warmup_steps, expected_first_lr_scale",
    [("constant_with_warmup", 2, 0.5), ("cosine", 0, 1.0), ("cosine", 2, 0.5)],
)
def test_build_lr_scheduler_supports_schedules_for_muon_with_adamw(
    lr_scheduler_type, warmup_steps, expected_first_lr_scale
):
    from ttml.trainers.grpo_trainer import _build_lr_scheduler

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
    from ttml.trainers.grpo_trainer import _build_lr_scheduler

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
