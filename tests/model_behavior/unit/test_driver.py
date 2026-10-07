# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import replace
from functools import partial

import pytest

from tests.model_behavior.driver import Request, RequestDriver, RequestState, assert_same_tokens
from tests.model_behavior.test_request_lifecycle import test_admission_preserves_surviving_request as admission_scenario
from tests.model_behavior.test_request_lifecycle import (
    test_completed_request_state_does_not_leak_on_slot_reuse as reuse_scenario,
)
from tests.model_behavior.test_request_lifecycle import test_mixed_request_replay as mixed_replay_scenario

replay_scenario = partial(mixed_replay_scenario, batch_layout="sparse")


class RecordingAdapter:
    """Host scheduling oracle, not a substitute for model/device validation."""

    capacity = 32
    vocab_size = 256

    def __init__(self, leak_on_admission=False, leak_on_reuse=False):
        self.calls = []
        self.leak_on_admission = leak_on_admission
        self.leak_on_reuse = leak_on_reuse
        self.perturb_survivor = False
        self.dirty = False

    def encode(self, prompt):
        return tuple(prompt.encode())

    def prefill(self, admitted):
        self.calls.append(("prefill", deepcopy(admitted)))
        if any(state.request.request_id == "replacement" for state in admitted):
            self.perturb_survivor = self.leak_on_admission
        if any(state.request.request_id == "previous-owner" for state in admitted):
            self.dirty = self.leak_on_reuse
        return {state.slot: (state.prompt_tokens[0] + int(self.dirty)) % self.vocab_size for state in admitted}

    def decode(self, active, *, reset_batch):
        self.calls.append(("decode", deepcopy(active), reset_batch))
        return {
            state.slot: (
                state.output_tokens[-1] + 1 + int(self.perturb_survivor and state.request.request_id == "survivor")
            )
            % self.vocab_size
            for state in active
        }


def test_admission_completion_and_survivor_history():
    adapter = RecordingAdapter()
    driver = RequestDriver(adapter)
    driver.admit([(31, Request("short", "xy", max_tokens=2)), (0, Request("long", "abc", max_tokens=5))])
    driver.step()
    assert set(driver.active) == {0}
    assert driver.requests["short"].output_tokens == [120, 121]
    driver.admit([(31, Request("replacement", "pq", max_tokens=3))])
    driver.step()
    _, active, reset = adapter.calls[-1]
    assert reset
    assert [state.slot for state in active] == [0, 31]
    assert active[0].prompt_tokens == (97, 98, 99)
    assert active[0].output_tokens == [97, 98]
    assert active[0].position == 4
    assert active[1].output_tokens == [112]
    driver.drain()
    assert not driver.active
    assert len(driver.requests["long"].output_tokens) == 5
    assert len(driver.requests["replacement"].output_tokens) == 3
    resets = [call[2] for call in adapter.calls if call[0] == "decode"]
    assert resets == [True, True, False, True]
    assert [event["phase"] for event in driver.events if event["request_id"] == "replacement"] == [
        "prefill",
        "decode",
        "decode",
    ]


def test_completion_notifies_public_release_hook_for_prefill_and_decode_once():
    adapter = RecordingAdapter()
    driver = RequestDriver(adapter)
    released = []
    adapter.release_request = lambda slot: released.append((slot, len(driver.active[slot].output_tokens)))
    driver.admit([(31, Request("one-token", "x", max_tokens=1)), (0, Request("survivor", "y", max_tokens=3))])
    assert released == [(31, 1)]
    driver.step()
    assert released == [(31, 1)]
    driver.admit([(31, Request("replacement", "z", max_tokens=2))])
    driver.drain()
    assert released == [(31, 1), (0, 3), (31, 2)]


@pytest.mark.parametrize("scenario", [replay_scenario, admission_scenario, reuse_scenario])
def test_scenarios_accept_isolated_requests(scenario):
    scenario(RequestDriver(RecordingAdapter()))


@pytest.mark.parametrize(
    "scenario,adapter,phase",
    [
        (admission_scenario, RecordingAdapter(leak_on_admission=True), "decode"),
        (reuse_scenario, RecordingAdapter(leak_on_reuse=True), "prefill"),
    ],
)
def test_scenarios_detect_cross_request_state_leaks(scenario, adapter, phase):
    # Root expect_error depends on TT fixtures; these tests run with --confcutdir.
    with pytest.raises(AssertionError, match=phase):  # allow-pytest.raises: host-only isolated suite
        scenario(RequestDriver(adapter))


def test_comparison_rejects_truncated_output():
    first = RequestState(Request("a", "x", max_tokens=2), 0, (1,), [2, 3])
    second = RequestState(replace(first.request, request_id="b"), 0, (1,), [2])
    with pytest.raises(  # allow-pytest.raises: host-only isolated suite
        AssertionError, match="generated token 1.*decode"
    ):
        assert_same_tokens(first, second)


def test_slot_collision_fails_before_model_call():
    adapter = RecordingAdapter()
    driver = RequestDriver(adapter)
    driver.admit([(0, Request("a", "x"))])
    calls = len(adapter.calls)
    with pytest.raises(ValueError, match="unavailable"):  # allow-pytest.raises: host-only isolated suite
        driver.admit([(0, Request("b", "y"))])
    assert len(adapter.calls) == calls
