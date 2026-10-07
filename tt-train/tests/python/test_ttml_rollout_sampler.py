# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests for ``TTMLRolloutSampler``.

Follows the fixture and prompt patterns already in ``test_grpo_trainer.py``.
Each test builds a model from real HuggingFace weights with ``setup_ttml_model``,
wraps it in a ``TTMLRolloutSampler`` and asserts on the returned
:class:`RolloutBatch`.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttml

from ttml.common.config import DeviceConfig, TransformerConfig
from ttml.modules import RunMode
from ttml.trainers.grpo_trainer import RolloutBatch, grpo_ttml_model
from ttml.trainers.grpo_trainer.grpo_ttml_model import setup_ttml_model
from ttml.trainers.grpo_trainer.ttml_rollout_sampler import TTMLRolloutSampler


HF_LLAMA_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"  # not gated
HF_QWEN3_MODEL_ID = "Qwen/Qwen3-0.6B"  # small Qwen3 for CI-friendly generation

MAX_COMPLETION_LENGTH = 32


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

# Only ``max_sequence_length`` and ``runner_type`` are consulted for Qwen3 — the
# architecture itself is read from the HF config of ``model_source``.
QWEN3_TRANSFORMER_CONFIG = TransformerConfig(
    {
        "transformer_config": {
            "model_type": "qwen3",
            "num_heads": 1,  # unused
            "num_groups": 1,  # unused
            "embedding_dim": 1,  # unused
            "intermediate_dim": 1,  # unused
            "dropout_prob": 0.0,
            "num_blocks": 1,  # unused
            "vocab_size": 1,  # unused
            "max_sequence_length": 512,
            "runner_type": "memory_efficient",
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
    """Override ``grpo_ttml_model.open_device`` to reuse an already-open
    ``AutoContext`` device rather than calling ``open_device`` again.

    Other tests in ``tests/python/`` lazily open the ``AutoContext`` device on
    first tensor use and never close it. When pytest collects this file
    alongside them, the device is already open by the time we get here, so a
    real ``open_device`` would trip ``open_device was called after the device
    was created``.

    Llama just needs the current device. Qwen3 also needs a named mesh so the
    ``ttml.mesh().has_axis`` checks work; mirror what ``ttml.open_device_mesh``
    does at the Python level minus the ``open_device`` call.
    """

    def _open_current_device(model_kind, device_config):
        if model_kind == "qwen3":
            from ttml.common.utils import build_mesh
            import ttml._mesh as _mesh_module

            _mesh_module._mesh = build_mesh(device_config)
        return ttml.autograd.AutoContext.get_instance().get_device()

    monkeypatch.setattr(grpo_ttml_model, "open_device", _open_current_device)


def _to_capitals_chat_prompt(tokenizer, user_text: str, **template_kwargs) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": CAPITALS_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


def _assert_rollout_batch_ok(batch: RolloutBatch, *, expected_shape) -> None:
    """Shared shape / sign / metadata invariants for the acceptance tests."""
    assert isinstance(batch, RolloutBatch)
    assert batch.weight_version == 0, f"expected weight_version=0 (default), got {batch.weight_version}"
    assert batch.logprobs.shape == expected_shape, f"logprobs shape {batch.logprobs.shape} != expected {expected_shape}"
    assert batch.logprobs.dtype == np.float32, f"logprobs dtype {batch.logprobs.dtype} != float32"

    # Log-probs are non-positive by definition; using `<=` (not strict `<`)
    # avoids a flaky failure when a near-certain pick rounds to 0.0.
    lp = batch.logprobs
    assert np.all(lp <= 0.0), (
        f"all logprobs must be <= 0 (log-probs are non-positive); " f"found positive values (max = {lp.max():.4g})"
    )

    # Padded (post-completion) positions must be exactly 0.0 — the sampler
    # right-pads with zeros so downstream masking treats them as ignored.
    for b, completion in enumerate(batch.completions):
        c_len = len(completion)
        if c_len < expected_shape[1]:
            pad_slice = lp[b, c_len:]
            assert np.all(pad_slice == 0.0), (
                f"row {b}: expected padded logprob positions [{c_len}:] to be 0.0, "
                f"got nonzero max = {np.abs(pad_slice).max():.4g}"
            )

    # At least one non-padded position must be finite AND non-zero magnitude.
    # This catches: (a) a sign flip that would forward `-log p` unchanged
    # (values would be >= 0, failing the check above OR having only positive
    # magnitudes never seen here), and (b) the degenerate case where every
    # logprob is exactly 0.0 (which would silently pass the `<= 0` check).
    saw_nontrivial = False
    for b, completion in enumerate(batch.completions):
        c_len = len(completion)
        if c_len == 0:
            continue
        row_lp = lp[b, :c_len]
        if np.any(np.isfinite(row_lp) & (np.abs(row_lp) > 1e-6)):
            saw_nontrivial = True
            break
    assert saw_nontrivial, (
        "expected at least one non-padded logprob to be finite and non-zero magnitude; "
        "either every completion has length 0 or every captured logprob is zero"
    )


@pytest.mark.requires_device
@pytest.mark.slow
def test_llama_rollout_sampler_capital_of_france():
    """Llama sampler answers "The capital of France is" with a completion
    containing "Paris", and returns a well-formed :class:`RolloutBatch`.
    """
    model, tokenizer = setup_ttml_model(LLAMA_1B_TRANSFORMER_CONFIG, DEVICE_CONFIG, HF_LLAMA_MODEL_ID)
    sampler = TTMLRolloutSampler(model, tokenizer, MAX_COMPLETION_LENGTH, 0.0, 1)

    prompt_str = _to_capitals_chat_prompt(tokenizer, "The capital of France is")
    prompt_ids = tokenizer.encode(prompt_str)

    batch = sampler.generate([prompt_ids])
    assert model.get_run_mode() == RunMode.TRAIN, "generate() must restore the model's run mode"

    completion_str = tokenizer.decode(batch.completions[0], skip_special_tokens=True)
    assert "paris" in completion_str.lower(), f"expected 'Paris' in Llama completion, got: {completion_str!r}"

    _assert_rollout_batch_ok(batch, expected_shape=(1, MAX_COMPLETION_LENGTH))

    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
@pytest.mark.slow
def test_qwen3_rollout_sampler_capital_of_france():
    """Qwen3 sampler answers "The capital of France is" with a completion
    containing "Paris". Exercises the Qwen3 dispatch end-to-end
    (Qwen3 KV cache + ``past_key_values=`` forward + Qwen3-shaped decode mask).
    """
    model, tokenizer = setup_ttml_model(QWEN3_TRANSFORMER_CONFIG, DEVICE_CONFIG, HF_QWEN3_MODEL_ID)
    sampler = TTMLRolloutSampler(model, tokenizer, MAX_COMPLETION_LENGTH, 0.0, 1)

    prompt_str = _to_capitals_chat_prompt(tokenizer, "The capital of France is", enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt_str)

    batch = sampler.generate([prompt_ids])
    assert model.get_run_mode() == RunMode.TRAIN, "generate() must restore the model's run mode"

    completion_str = tokenizer.decode(batch.completions[0], skip_special_tokens=True)
    assert "paris" in completion_str.lower(), f"expected 'Paris' in Qwen3 completion, got: {completion_str!r}"

    _assert_rollout_batch_ok(batch, expected_shape=(1, MAX_COMPLETION_LENGTH))

    ttml.autograd.AutoContext.get_instance().reset_graph()
