# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sampled-token logprob alignment against logits from the same model step."""

from dataclasses import replace

import pytest

from tests.model_behavior.driver import Request, Sampling, assert_same_tokens, run_batch
from tests.model_behavior.test_sampling_behavior import PENALTY_PROMPTS, spaced_slots


@pytest.fixture(autouse=True)
def require_device_logprobs(request_driver):
    if not request_driver.adapter.logprob_phases:
        pytest.skip(request_driver.adapter.logprob_skip_reason)


def assert_logprobs_align(state, atol=0.05, start_index=0):
    assert len(state.samples) == state.request.max_tokens - start_index, "Missing per-token logprob observations"
    assert [sample.token_id for sample in state.samples] == state.output_tokens[
        start_index:
    ], "Logprob token IDs are misaligned"
    for index, sample in enumerate(state.samples, start_index):
        error = abs(sample.logprob - sample.reference_logprob)
        assert error <= atol, (
            f"{state.request.request_id}, slot {state.slot}, token {index} "
            f"({'prefill' if index == 0 else 'decode'}), ID {sample.token_id}: "
            f"logprob {sample.logprob} vs CPU log_softmax {sample.reference_logprob}; {error=} > {atol=}"
        )


@pytest.mark.parametrize("seeded", [True, False], ids=["seeded", "unseeded"])
@pytest.mark.parametrize("layout", ["full", "mixed-flags"])
def test_sampled_token_logprobs_align(request_driver, seeded, layout):
    driver = request_driver
    slots = list(range(driver.adapter.capacity)) if layout == "full" else spaced_slots(driver.adapter.capacity, 4)[::-1]
    states = run_batch(
        driver,
        [
            (
                slot,
                Request(
                    f"logprobs-{i}",
                    PENALTY_PROMPTS[i % len(PENALTY_PROMPTS)],
                    Sampling(temperature=0.8, seed=i if seeded else None, enable_log_probs=layout == "full" or i != 1),
                    max_tokens=8,
                ),
            )
            for i, slot in enumerate(slots)
        ],
    )
    for state in states:
        if state.request.sampling.enable_log_probs:
            assert_logprobs_align(state, start_index=0 if "prefill" in driver.adapter.logprob_phases else 1)
            driver.checks.append(
                dict(
                    request_id=state.request.request_id,
                    max_logprob_error=max(abs(s.logprob - s.reference_logprob) for s in state.samples),
                )
            )
    if seeded:
        without = run_batch(
            driver,
            [
                (
                    state.slot,
                    replace(
                        state.request,
                        request_id=f"without-{i}",
                        sampling=replace(state.request.sampling, enable_log_probs=False),
                    ),
                )
                for i, state in enumerate(states)
            ],
        )
        for a, b in zip(states, without):
            assert_same_tokens(a, b)


def test_greedy_sampled_token_logprobs_align(request_driver):
    driver = request_driver
    states = run_batch(
        driver,
        [
            (slot, Request(f"greedy-logprobs-{i}", prompt, Sampling(enable_log_probs=True), max_tokens=8))
            for i, (slot, prompt) in enumerate(
                zip(spaced_slots(driver.adapter.capacity, min(4, driver.adapter.capacity)), PENALTY_PROMPTS)
            )
        ],
    )
    for state in states:
        assert_logprobs_align(state, start_index=0 if "prefill" in driver.adapter.logprob_phases else 1)
