# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Layer-3 regression for PR #57765: Generator.process_decode_output_host must NOT fall back
to the ``torch.ones(logits_i.shape)`` placeholder in
[models/ttt_compat/tt/generator.py](../tt/generator.py) when the sampling op
returned real per-token logprobs. The exact bug this PR fixes was that on 1x1 meshes the
sampling op emitted ``(tokens, None)`` even with logprobs enabled (the calculator only
supported 8/32-device meshes), triggering that fallback for every prefill and decode step.

These tests stub ``self.model[i].process_output_decode`` so no device weights are required;
they exercise ``process_decode_output_host`` in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from models.ttt_compat.tt.generator import Generator

# ---------------------------------------------------------------------------
# Stubs for self.model[i] / self.model_args[i]
# ---------------------------------------------------------------------------


class _StubDecodeModel:
    """Minimal self.model[i] that satisfies the two call shapes of process_decode_output_host.

    - is_tokens=True, is_log_probs=False  -> returns self._tokens (treated as "logits_i")
    - is_tokens=True, is_log_probs=True   -> returns self._log_probs
    """

    def __init__(self, *, tokens: torch.Tensor, log_probs: torch.Tensor):
        self._tokens = tokens
        self._log_probs = log_probs

    def process_output_decode(
        self,
        tt_out,  # noqa: ARG002 - stub ignores the ttnn handle
        B,  # noqa: ARG002
        S: int = 1,  # noqa: ARG002
        is_tokens: bool = False,
        is_log_probs: bool = False,
    ):
        if is_log_probs:
            return self._log_probs
        return self._tokens


@dataclass
class _StubModelArgs:
    max_batch_size: int


def _make_stub_generator(stub_model: _StubDecodeModel, max_batch_size: int) -> Generator:
    """Build just enough of a Generator to call process_decode_output_host.

    Generator.__init__ is heavy (defaultdicts, trace state, etc.) but only ``self.model``,
    ``self.model_args``, and ``self.data_parallel`` matter for process_decode_output_host.
    """
    gen = object.__new__(Generator)
    gen.model = [stub_model]
    gen.model_args = [_StubModelArgs(max_batch_size=max_batch_size)]
    gen.data_parallel = 1
    gen.mesh_device = None  # unused for the single-device, non-topk path
    return gen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


BATCH_SIZE = 32


def test_process_decode_output_host_forwards_real_logprobs_when_sampling_returned_tuple():
    """PR #57765 regression: when ``tt_out[i] = (tokens, log_probs_tensor)``, the real
    logprobs must flow through the pipeline unchanged. If this breaks, every greedy 1x1 caller
    silently sees the constant-1.0 placeholder again.
    """
    expected_log_probs = torch.linspace(-10.0, -0.1, steps=BATCH_SIZE)
    stub_tokens = torch.arange(BATCH_SIZE, dtype=torch.int64)
    gen = _make_stub_generator(
        _StubDecodeModel(tokens=stub_tokens, log_probs=expected_log_probs),
        max_batch_size=BATCH_SIZE,
    )

    # Simulate what TTSampling.forward returns on greedy + enable_log_probs=True on 1x1:
    # (tokens_handle, log_probs_handle) -- the stub ignores both handles.
    tt_out = [(object(), object())]

    tokens, log_probs = gen.process_decode_output_host(tt_out, is_tokens=True)

    # The output must match the stubbed logprobs byte-for-byte -- the fallback would overwrite
    # with torch.ones(BATCH_SIZE), which is not what we return above.
    assert torch.equal(
        log_probs, expected_log_probs
    ), f"logprobs changed in flight: got {log_probs[:4]} vs expected {expected_log_probs[:4]}"
    assert not torch.allclose(log_probs, torch.ones_like(log_probs)), (
        "process_decode_output_host returned the constant-1.0 placeholder; "
        "the torch.ones fallback must not fire on the old-path logprobs branch."
    )
    # Tokens pass through unchanged.
    assert torch.equal(tokens, stub_tokens)


def test_process_decode_output_host_uses_placeholder_when_sampling_returned_no_logprobs():
    """Pins today's complement: when the sampling op returned ``(tokens, None)`` the
    constant-1.0 placeholder is still what callers see. This is the exact signature that
    PR #57765 avoids for greedy+logprobs; here we confirm it still fires when
    logprobs are DISABLED (temperature=0 + no enable_log_probs).
    """
    stub_tokens = torch.arange(BATCH_SIZE, dtype=torch.int64)
    # The log_probs stub value is unused in this branch -- the generator short-circuits to
    # torch.ones once it sees lp is None.
    gen = _make_stub_generator(
        _StubDecodeModel(tokens=stub_tokens, log_probs=torch.empty(0)),
        max_batch_size=BATCH_SIZE,
    )

    tt_out = [(object(), None)]
    tokens, log_probs = gen.process_decode_output_host(tt_out, is_tokens=True)

    assert torch.equal(tokens, stub_tokens)
    assert torch.allclose(log_probs, torch.ones_like(log_probs)), (
        "The torch.ones placeholder must still be the result when the sampling op returned "
        "(tokens, None); otherwise some other codepath started synthesising logprobs."
    )
    assert log_probs.shape == stub_tokens.shape


if __name__ == "__main__":  # pragma: no cover - allow adhoc runs
    pytest.main([__file__, "-v"])
