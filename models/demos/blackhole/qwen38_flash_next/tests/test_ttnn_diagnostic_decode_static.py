# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device checks for the narrow nonqualifying decode adapter."""

from __future__ import annotations

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import diagnostic_decode
from models.demos.blackhole.qwen38_flash_next.ttnn.decode import (
    Qwen38OrdinaryEmissionTiming,
    Qwen38OrdinaryInvocationTiming,
    Qwen38OrdinarySessionStatus,
    Qwen38OrdinaryStopReason,
    Qwen38OrdinaryTimingPhase,
    Qwen38OrdinaryToken,
)


def _event(token_id: int = 11751) -> Qwen38OrdinaryToken:
    invocation = Qwen38OrdinaryInvocationTiming(42, 0, 1, 2, 3, 7)
    timing = Qwen38OrdinaryEmissionTiming(
        phase=Qwen38OrdinaryTimingPhase.TTFT,
        input_token_count=1,
        model_call_ns=2,
        explicit_sync_ns=4,
        host_overhead_ns=1,
        end_to_end_ns=7,
        invocations=(invocation,),
    )
    return Qwen38OrdinaryToken(
        request_id=1,
        token_index=0,
        token_id=token_id,
        cache_position=1,
        stop_reason=Qwen38OrdinaryStopReason.MAX_NEW_TOKENS,
        timing=timing,
    )


class _FakeSession:
    instances: list["_FakeSession"] = []
    begin_error: BaseException | None = None

    def __init__(self, target, **kwargs) -> None:
        self.target = target
        self.kwargs = kwargs
        self.status = Qwen38OrdinarySessionStatus.IDLE
        self.begin_calls = []
        self.close_calls = 0
        type(self).instances.append(self)

    def begin(self, prompt, *, max_new_tokens: int) -> Qwen38OrdinaryToken:
        self.begin_calls.append((tuple(prompt), max_new_tokens))
        if type(self).begin_error is not None:
            self.status = Qwen38OrdinarySessionStatus.POISONED
            raise type(self).begin_error
        self.status = Qwen38OrdinarySessionStatus.FINISHED
        return _event()

    def close(self) -> None:
        self.close_calls += 1
        self.status = Qwen38OrdinarySessionStatus.CLOSED


@pytest.fixture(autouse=True)
def _reset_fake_session(monkeypatch):
    _FakeSession.instances.clear()
    _FakeSession.begin_error = None
    monkeypatch.setattr(diagnostic_decode, "Qwen38OrdinaryDecodeSession", _FakeSession)


def test_first_token_executes_exact_one_step_and_is_explicitly_nonqualifying() -> None:
    target = object()
    provenance = object()
    sync = object()
    clock = object()

    result = diagnostic_decode.decode_first_nonqualifying_token(
        target,
        42,
        expected_provenance=provenance,
        expected_physical_ids=(1, 0, 2, 3),
        expected_identity_key="identity",
        eos_token_ids=(2,),
        synchronize=sync,
        clock_ns=clock,
    )

    assert result.input_token_id == 42
    assert result.token_id == 11751
    assert result.cache_position == 1
    assert result.qualification == "NONQUALIFYING"
    assert result.layers_executed == 48
    assert result.terminal_mechanic == "terminal_hyper_connection_rms_norm_and_untied_lm_head"
    assert result.sampling_mechanic == "tp4_vocab_sharded_sparse_argmax"

    session = _FakeSession.instances[0]
    assert session.target is target
    assert session.kwargs == {
        "expected_provenance": provenance,
        "expected_physical_ids": (1, 0, 2, 3),
        "expected_identity_key": "identity",
        "eos_token_ids": (2,),
        "synchronize": sync,
        "clock_ns": clock,
    }
    assert session.begin_calls == [((42,), 1)]
    assert session.close_calls == 1


def test_poisoned_diagnostic_does_not_attempt_state_reuse(expect_error) -> None:
    _FakeSession.begin_error = RuntimeError("layer failure")

    with expect_error(RuntimeError, "layer failure"):
        diagnostic_decode.decode_first_nonqualifying_token(
            object(),
            42,
            expected_provenance=object(),
            expected_physical_ids=(1, 0, 2, 3),
            expected_identity_key="identity",
            eos_token_ids=(2,),
        )

    session = _FakeSession.instances[0]
    assert session.status is Qwen38OrdinarySessionStatus.POISONED
    assert session.close_calls == 0


@pytest.mark.parametrize("token", (True, -1, 248320, "42"))
def test_invalid_input_is_rejected_before_constructing_a_session(token, expect_error) -> None:
    with expect_error(ValueError, "input_token_id"):
        diagnostic_decode.decode_first_nonqualifying_token(
            object(),
            token,
            expected_provenance=object(),
            expected_physical_ids=(1, 0, 2, 3),
            expected_identity_key="identity",
            eos_token_ids=(2,),
        )
    assert _FakeSession.instances == []
