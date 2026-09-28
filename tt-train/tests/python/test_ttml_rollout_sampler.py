# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests for :class:`ttml.trainers.grpo_trainer.TTMLRolloutSampler`.

Follows the fixture and prompt patterns already in ``test_grpo_trainer.py``.
Each test loads real HuggingFace weights via one of the existing
``GRPOCompleter`` implementations (the shortest path to a live ttml model
on device today), then builds a ``TTMLRolloutSampler`` from that completer's
resolved mesh state and asserts on the returned :class:`RolloutBatch`.

The ``GRPOCompleter`` classes are used only as a temporary source of already-
resolved mesh state / weight loading here. Once ``GRPOCompleter`` is removed,
these tests should be re-pointed at whatever replaces it as the model-loading
entry point — the sampler assertions themselves do not depend on
``GRPOCompleter``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import ttml

from ttml.common.config import DeviceConfig, TransformerConfig
from ttml.trainers.grpo_trainer import TTMLRolloutSampler, RolloutBatch


# The completer implementations live under the examples tree, not under
# ``ttml`` proper. Add its parent to ``sys.path`` so this test can import them
# without copy-pasting the completer.
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
from utils.qwen3_completer import Qwen3CompletionCtx, Qwen3GRPOCompleter  # noqa: E402


HF_LLAMA_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"  # not gated
HF_QWEN3_MODEL_ID = "Qwen/Qwen3-0.6B"  # small Qwen3 for CI-friendly generation


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

# Only ``max_sequence_length`` is consulted by ``Qwen3GRPOCompleter`` — the
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
    """Override both completers' ``setup_device`` to reuse an already-open
    ``AutoContext`` device rather than calling ``open_device`` again.

    Other tests in ``tests/python/`` lazily open the ``AutoContext`` device on
    first tensor use and never close it. When pytest collects this file
    alongside them, the device is already open by the time we get here, so a
    real ``setup_device`` (which calls ``open_device``) would trip
    ``open_device was called after the device was created``. Reusing the live
    device sidesteps the issue without leaking device-management code into
    the test body.

    Llama's ``setup_device`` just needs to return the current device.
    Qwen3's also needs to install a named mesh so its ``self._mesh.has_axis``
    checks work; mirror what ``ttml.open_device_mesh`` does at the Python
    level minus the ``open_device`` call.
    """
    monkeypatch.setattr(
        LlamaGRPOCompleter,
        "setup_device",
        lambda self, device_config: ttml.autograd.AutoContext.get_instance().get_device(),
    )

    def _qwen3_reuse_setup_device(self, device_config):
        from ttml.common.utils import build_mesh
        import ttml._mesh as _mesh_module

        mesh = build_mesh(device_config)
        self._mesh = mesh
        _mesh_module._mesh = mesh
        return ttml.autograd.AutoContext.get_instance().get_device()

    monkeypatch.setattr(Qwen3GRPOCompleter, "setup_device", _qwen3_reuse_setup_device)


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


def _build_llama_sampler(completer: LlamaGRPOCompleter, *, max_tokens: int) -> TTMLRolloutSampler:
    cfg = completer.transformer_config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.embedding_dim // cfg.num_heads)
    return TTMLRolloutSampler(
        model_kind="llama",
        model=completer.model,
        tokenizer=completer.tokenizer,
        mesh_device=completer._mesh_device,
        dp_mapper=completer._dp_mapper,
        dp_composer=completer._dp_composer,
        num_devices=completer._num_devices,
        seed_axes=completer._seed_axes,
        num_layers=cfg.num_blocks,
        max_seq_len=cfg.max_sequence_length,
        max_tokens_to_complete=max_tokens,
        max_completion_length=max_tokens,
        temperature=0.0,
        completions_per_prompt=1,
        num_kv_groups=cfg.num_groups,
        head_dim=head_dim,
    )


def _build_qwen3_sampler(completer: Qwen3GRPOCompleter, *, max_tokens: int) -> TTMLRolloutSampler:
    return TTMLRolloutSampler(
        model_kind="qwen3",
        model=completer._model,
        tokenizer=completer._ctx._tokenizer,
        mesh_device=completer._mesh_device,
        dp_mapper=completer._dp_mapper,
        dp_composer=completer._dp_composer,
        num_devices=completer._num_devices,
        seed_axes=completer._seed_axes,
        num_layers=completer._config.num_hidden_layers,
        max_seq_len=completer._max_seq_len,
        max_tokens_to_complete=max_tokens,
        max_completion_length=max_tokens,
        temperature=0.0,
        completions_per_prompt=1,
    )


def _assert_rollout_batch_ok(batch: RolloutBatch, *, expected_shape, first_batch_id: int) -> None:
    """Shared shape / sign / metadata invariants for the acceptance tests."""
    assert isinstance(batch, RolloutBatch)
    assert batch.batch_id == first_batch_id, f"expected batch_id={first_batch_id}, got {batch.batch_id}"
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
    completer = LlamaGRPOCompleter(
        ctx=LlamaCompletionCtx(
            max_tokens_to_complete=32,
            temperature=0.0,
            completions_per_prompt=1,
        ),
        transformer_config=LLAMA_1B_TRANSFORMER_CONFIG,
        device_config=DEVICE_CONFIG,
        model_source=HF_LLAMA_MODEL_ID,
    )

    max_tokens = 32
    sampler = _build_llama_sampler(completer, max_tokens=max_tokens)

    tokenizer = completer.tokenizer
    prompt_str = _to_capitals_chat_prompt(tokenizer, "The capital of France is")
    prompt_ids = tokenizer.encode(prompt_str)

    batch = sampler.generate([prompt_ids])

    completion_str = tokenizer.decode(batch.completions[0], skip_special_tokens=True)
    assert "paris" in completion_str.lower(), f"expected 'Paris' in Llama completion, got: {completion_str!r}"

    _assert_rollout_batch_ok(batch, expected_shape=(1, max_tokens), first_batch_id=0)

    # Second call bumps batch_id monotonically.
    batch2 = sampler.generate([prompt_ids])
    assert batch2.batch_id == 1, f"expected monotonic batch_id, got {batch2.batch_id}"

    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
@pytest.mark.slow
def test_qwen3_rollout_sampler_capital_of_france():
    """Qwen3 sampler answers "The capital of France is" with a completion
    containing "Paris". Exercises the ``model_kind="qwen3"`` dispatch end-to-end
    (Qwen3 KV cache + ``past_key_values=`` forward + Qwen3-shaped decode mask).
    """
    completer = Qwen3GRPOCompleter(
        ctx=Qwen3CompletionCtx(
            max_tokens_to_complete=32,
            temperature=0.0,
            completions_per_prompt=1,
        ),
        transformer_config=QWEN3_TRANSFORMER_CONFIG,
        device_config=DEVICE_CONFIG,
        model_source=HF_QWEN3_MODEL_ID,
    )

    max_tokens = 32
    sampler = _build_qwen3_sampler(completer, max_tokens=max_tokens)

    tokenizer = completer._ctx._tokenizer
    prompt_str = _to_capitals_chat_prompt(tokenizer, "The capital of France is", enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt_str)

    batch = sampler.generate([prompt_ids])

    completion_str = tokenizer.decode(batch.completions[0], skip_special_tokens=True)
    assert "paris" in completion_str.lower(), f"expected 'Paris' in Qwen3 completion, got: {completion_str!r}"

    _assert_rollout_batch_ok(batch, expected_shape=(1, max_tokens), first_batch_id=0)

    batch2 = sampler.generate([prompt_ids])
    assert batch2.batch_id == 1, f"expected monotonic batch_id, got {batch2.batch_id}"

    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
@pytest.mark.slow
def test_llama_rollout_sampler_matches_completer_no_padding():
    """Parity between ``LlamaGRPOCompleter.generate`` and
    ``TTMLRolloutSampler.generate`` on a single prompt (no padding).

    With ``batch_size == 1`` and ``completions_per_prompt == 1``:
      * ``max_prompt_len == len_b == Np`` — the pad region ``[len_b, Np)``
        is empty on both paths.
      * RoPE positions align — real content sits at ``[0, len_b)`` in both
        the completer's left-padded window (``pad_lengths[0] == 0``) and
        the sampler's right-padded window.
      * The "left-pad vs. right-pad" difference therefore collapses to a
        no-op, so strict token parity is achievable.

    ``temperature = 0.0`` makes sampling deterministic (``sample_op`` skips
    Gumbel noise). This isolates and validates the decode-mask shape change
    (the trailing-pad zeroing mirroring today's leading-pad zeroing) from
    every other moving part.

    Padding-side changes DO shift RoPE positions when padding is present,
    so strict token parity is not attempted for the multi-prompt / variable-
    length case — that is the intentional scope carve-out.
    """
    completer = LlamaGRPOCompleter(
        ctx=LlamaCompletionCtx(
            max_tokens_to_complete=16,
            temperature=0.0,
            completions_per_prompt=1,
        ),
        transformer_config=LLAMA_1B_TRANSFORMER_CONFIG,
        device_config=DEVICE_CONFIG,
        model_source=HF_LLAMA_MODEL_ID,
    )

    max_tokens = 16
    sampler = _build_llama_sampler(completer, max_tokens=max_tokens)

    tokenizer = completer.tokenizer
    prompt_str = _to_capitals_chat_prompt(tokenizer, "The capital of France is")
    prompt_ids = tokenizer.encode(prompt_str)

    completer_completions = completer.generate([prompt_ids])
    assert len(completer_completions) == 1

    sampler_batch = sampler.generate([prompt_ids])
    assert len(sampler_batch.completions) == 1

    completer_tokens = completer_completions[0]
    sampler_tokens = sampler_batch.completions[0]

    # Compare first N tokens (both are trimmed at first stop, so we only
    # need to compare up to the shorter length).
    min_len = min(len(completer_tokens), len(sampler_tokens))
    assert min_len > 0, "both paths returned empty completions"
    assert completer_tokens[:min_len] == sampler_tokens[:min_len], (
        f"decode-mask parity broken (no padding case):\n"
        f"  completer: {completer_tokens[:min_len]}\n"
        f"  sampler:   {sampler_tokens[:min_len]}"
    )

    ttml.autograd.AutoContext.get_instance().reset_graph()
