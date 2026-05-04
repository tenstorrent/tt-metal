# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for ``GRPOTrainer`` on a Tenstorrent device.

The goal is to exercise the full GRPO loop (generation -> reward ->
advantages -> old-prob forward -> train-mode forward+backward -> optimizer
step -> callbacks) in the smallest configuration that still produces a
non-degenerate gradient update.

Speed strategy:
  * Tiny random-init Llama (1 layer, hidden=64, head_dim=32).
  * Skip the HuggingFace weight download by monkey-patching
    ``snapshot_download`` and ``load_from_safetensors`` in
    ``utils.llama_completer`` to no-ops; the model keeps its random init.
  * ``max_completion_length=4`` so autoregressive generation is cheap.
  * Exactly one optimizer step (``gradient_accumulation_steps=1``,
    ``num_iterations=1``, ``prompts_to_train=2``, ``batch_size=2``).
  * ``num_generations=2`` is the minimum that yields a non-zero advantage
    (group of 1 -> mean == reward -> advantage == 0 -> loss == 0).

Note on CI: the tokenizer load uses ``meta-llama/Llama-3.2-1B-Instruct``,
which is a gated HuggingFace repo. To run this in CI without an
``HF_TOKEN`` secret, swap ``HF_MODEL_ID`` below for a non-gated mirror
(e.g. ``unsloth/Llama-3.2-1B-Instruct``).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import ttml
import ttnn
from datasets import Dataset

from ttml.common.config import DeviceConfig, TransformerConfig
from ttml.trainers import GRPOConfig, GRPOTrainer, TrainerCallback


# The ``LlamaGRPOCompleter`` reference implementation lives under the
# examples tree, not under ``ttml`` proper. Surface its package on the
# import path so this test can use it without copy-pasting the completer.
_GRPO_EXAMPLES_DIR = os.path.join(
    os.environ.get("TT_METAL_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    "tt-train",
    "sources",
    "examples",
    "grpo",
)
if _GRPO_EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _GRPO_EXAMPLES_DIR)

from utils.llama_completer import LlamaCompletionCtx, LlamaGRPOCompleter  # noqa: E402


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
            # Overwritten by ``len(tokenizer)`` inside the completer ctor.
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
    """Override ``LlamaGRPOCompleter.setup_device`` to reuse the already-open
    AutoContext device instead of calling ``open_device`` again.

    Other tests in ``tests/python/`` lazily open the AutoContext device on
    first tensor use and never close it. When pytest collects this file
    alongside them, the device is already open by the time we get here, so
    the default ``setup_device`` would trip ``open_device was called after
    the device was created``. Reusing the live device sidesteps the issue
    without leaking device-management code into the test body.
    """
    monkeypatch.setattr(
        LlamaGRPOCompleter,
        "setup_device",
        lambda self, device_config: ttml.autograd.AutoContext.get_instance().get_device(),
    )


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
    """Skip the HF download / safetensors load so the tiny model keeps random init."""
    from utils import llama_completer

    monkeypatch.setattr(llama_completer, "snapshot_download", lambda *args, **kwargs: "/tmp/unused")
    monkeypatch.setattr(llama_completer, "load_from_safetensors", lambda *args, **kwargs: None)


@pytest.mark.requires_device
def test_grpo_trainer_one_step_smoke(patch_llama_weight_loading, tmp_path):
    """One full GRPO optimizer step on a tiny random-init Llama.

    Asserts the loop reaches every callback hook, produces finite metrics,
    and actually mutates model weights via ``optimizer.step``.
    """
    np.random.seed(0)

    completer = LlamaGRPOCompleter(
        ctx=LlamaCompletionCtx(
            max_tokens_to_complete=4,
            temperature=1.0,
            completions_per_prompt=2,
        ),
        transformer_config=TINY_TRANSFORMER_CONFIG,
        device_config=DEVICE_CONFIG,
        model_source=HF_MODEL_ID,
    )

    # Snapshot a single parameter so we can prove training mutated it.
    params = completer.model.parameters()
    assert params, "tiny model should expose at least one parameter"
    snapshot_name, snapshot_param = next(iter(params.items()))
    before = snapshot_param.to_numpy(ttnn.DataType.FLOAT32).copy()

    grpo_cfg = GRPOConfig(
        epsilon=0.2,
        batch_size=2,
        micro_batch_size=4,
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
        warmup_steps=0,
    )

    tokenizer = completer.tokenizer
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
    GRPOTrainer(
        completer=completer,
        dataset=dataset,
        config=grpo_cfg,
        reward_func=reward_func,
        optimizer_dict=optimizer_dict,
        callbacks=[recorder],
    ).train()

    assert recorder.train_begin == 1, "on_train_begin should fire exactly once"
    assert recorder.before_step == 1, "on_before_optimizer_step should fire once for the single step"
    assert recorder.step_end == 1, "on_step_end should fire once for the single step"
    assert recorder.train_end == 1, "on_train_end should fire exactly once"

    assert reward_calls, "reward_func was never invoked"
    expected_completions = grpo_cfg.batch_size * grpo_cfg.num_generations
    assert (
        len(reward_calls[0]) == expected_completions
    ), f"expected {expected_completions} completions per batch, got {len(reward_calls[0])}"

    metrics = recorder.last_step_metrics
    assert metrics is not None
    for key in ("reward_mean", "reward_std", "mean_completion_len", "step_time_s", "generation_time_s"):
        assert key in metrics, f"missing metric {key}"
        assert np.isfinite(metrics[key]), f"metric {key} is not finite: {metrics[key]}"

    after = completer.model.parameters()[snapshot_name].to_numpy(ttnn.DataType.FLOAT32)
    assert before.shape == after.shape
    assert not np.array_equal(before, after), (
        f"parameter {snapshot_name!r} was unchanged after one optimizer step; "
        "either backward did not run or the gradient was identically zero"
    )

    ttml.autograd.AutoContext.get_instance().reset_graph()


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
