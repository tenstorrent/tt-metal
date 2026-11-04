# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sampling contracts at the real-model boundary, without a serving scheduler."""

from dataclasses import replace

import pytest

from tests.model_behavior.driver import Request, Sampling, assert_same_tokens, assert_varied_tokens, run_batch

DIVERSITY_PROMPT = "Pick one random lowercase letter from a to z. Reply with that letter only."
PENALTY_PROMPTS = (
    "She opened the door and",
    "The reason is that the",
    "He said that the",
    "I think the answer is maybe",
    "After a while, she",
    "It was a",
    "The book was",
    "a b c a b c a b c",
)


def spaced_slots(capacity, count):
    if capacity < count:
        pytest.skip(f"This scenario needs {count} physical slots")
    return [0] if count == 1 else [i * (capacity - 1) // (count - 1) for i in range(count)]


def comparison_slots(capacity, count):
    """Preserve all sensitivity prompts on single-user CI configurations."""
    return spaced_slots(capacity, count) if capacity >= count else [i % capacity for i in range(count)]


def run_waves(driver, placements):
    states = []
    for start in range(0, len(placements), driver.adapter.capacity):
        states.extend(run_batch(driver, placements[start : start + driver.adapter.capacity]))
    return states


@pytest.mark.parametrize("seeded", [True, False], ids=["distinct-seeds", "unseeded"])
def test_stochastic_requests_vary(request_driver, seeded):
    driver = request_driver
    slots = spaced_slots(driver.adapter.capacity, 8)
    first = run_batch(
        driver,
        [
            (slot, Request(f"first-{i}", DIVERSITY_PROMPT, Sampling(temperature=2.0, seed=i if seeded else None)))
            for i, slot in enumerate(slots)
        ],
    )
    second = run_batch(
        driver,
        [(state.slot, replace(state.request, request_id=f"second-{i}")) for i, state in enumerate(first)],
    )
    if getattr(driver.adapter, "stochastic_prefill", True):
        assert_varied_tokens(first, first_token=True)
    assert_varied_tokens(first)
    if seeded:
        for a, b in zip(first, second):
            assert_same_tokens(a, b)
        shifted = run_batch(
            driver,
            [
                (
                    state.slot,
                    replace(
                        state.request,
                        request_id=f"shifted-{i}",
                        sampling=replace(state.request.sampling, seed=i + 1000),
                    ),
                )
                for i, state in enumerate(first)
            ],
        )
        assert any(a.output_tokens != b.output_tokens for a, b in zip(first, shifted)), "Changing seeds had no effect"
    else:
        assert any(
            a.output_tokens != b.output_tokens for a, b in zip(first, second)
        ), "Unseeded requests replayed exactly"


@pytest.mark.parametrize("seed", [0, 1])
def test_identical_seed_requests_match(request_driver, seed):
    driver = request_driver
    if driver.adapter.duplicate_seed_policy != "identical":
        pytest.skip("This backend deliberately salts duplicate request seeds")
    slots = spaced_slots(driver.adapter.capacity, 4)
    first = run_batch(
        driver,
        [
            (slot, Request(f"first-{i}", DIVERSITY_PROMPT, Sampling(temperature=1.0, seed=seed)))
            for i, slot in enumerate(slots)
        ],
    )
    second = run_batch(
        driver,
        [(state.slot, replace(state.request, request_id=f"second-{i}")) for i, state in enumerate(first)],
    )
    for state in first[1:] + second:
        assert_same_tokens(first[0], state)


@pytest.mark.parametrize("seed", [0, -1])
def test_single_seeded_request_replays(request_driver, seed):
    driver = request_driver
    request = Request("first", DIVERSITY_PROMPT, Sampling(temperature=1.5, seed=seed))
    baseline = run_batch(driver, [(driver.adapter.capacity - 1, request)])[0]
    for repeat in range(3):
        replay = run_batch(driver, [(baseline.slot, replace(request, request_id=f"replay-{repeat}"))])[0]
        assert_same_tokens(baseline, replay)


def test_unseeded_single_request_varies(request_driver):
    driver = request_driver
    states = []
    for repeat in range(8):
        states.extend(run_batch(driver, [(0, Request(f"draw-{repeat}", DIVERSITY_PROMPT, Sampling(temperature=2.0)))]))
    if getattr(driver.adapter, "stochastic_prefill", True):
        assert_varied_tokens(states, first_token=True)
    assert_varied_tokens(states)


def test_seeded_requests_replay_after_slot_permutation(request_driver):
    driver = request_driver
    slots = spaced_slots(driver.adapter.capacity, 4)
    first = run_batch(
        driver,
        [
            (slot, Request(f"first-{i}", prompt, Sampling(temperature=0.8, seed=100 + i)))
            for i, (slot, prompt) in enumerate(zip(slots, PENALTY_PROMPTS))
        ],
    )
    second = run_batch(
        driver,
        [
            (slot, replace(state.request, request_id=f"moved-{i}"))
            for i, (slot, state) in enumerate(zip(slots[::-1], first))
        ],
    )
    for a, b in zip(first, second):
        assert_same_tokens(a, b)


def test_top_k_one_matches_greedy(request_driver):
    driver = request_driver
    slots = spaced_slots(driver.adapter.capacity, min(4, driver.adapter.capacity))
    baseline = run_batch(
        driver, [(slot, Request(f"greedy-{i}", prompt)) for i, (slot, prompt) in enumerate(zip(slots, PENALTY_PROMPTS))]
    )
    top1 = run_batch(
        driver,
        [
            (
                state.slot,
                replace(state.request, request_id=f"top1-{i}", sampling=Sampling(temperature=2.0, top_k=1, seed=i)),
            )
            for i, state in enumerate(baseline)
        ],
    )
    for a, b in zip(baseline, top1):
        assert_same_tokens(a, b)


def test_top_p_changes_model_sampling(request_driver):
    driver = request_driver
    slots = comparison_slots(driver.adapter.capacity, 8)
    baseline = run_waves(
        driver,
        [
            (slot, Request(f"full-{i}", DIVERSITY_PROMPT, Sampling(temperature=2.0, seed=i)))
            for i, slot in enumerate(slots)
        ],
    )
    restricted = run_waves(
        driver,
        [
            (
                state.slot,
                replace(
                    state.request, request_id=f"restricted-{i}", sampling=replace(state.request.sampling, top_p=0.1)
                ),
            )
            for i, state in enumerate(baseline)
        ],
    )
    assert any(
        a.output_tokens != b.output_tokens for a, b in zip(baseline, restricted)
    ), "Restricting top_p had no effect"


@pytest.mark.parametrize(
    "penalty,value", [("repetition_penalty", 2.5), ("presence_penalty", 2.0), ("frequency_penalty", 2.0)]
)
def test_penalty_changes_model_output(request_driver, penalty, value):
    driver = request_driver
    slots = comparison_slots(driver.adapter.capacity, len(PENALTY_PROMPTS))
    baseline = run_waves(
        driver,
        [
            (slot, Request(f"baseline-{i}", prompt, max_tokens=24))
            for i, (slot, prompt) in enumerate(zip(slots, PENALTY_PROMPTS))
        ],
    )
    control = run_waves(
        driver, [(state.slot, replace(state.request, request_id=f"control-{i}")) for i, state in enumerate(baseline)]
    )
    for a, b in zip(baseline, control):
        assert_same_tokens(a, b)
    penalized = run_waves(
        driver,
        [
            (state.slot, replace(state.request, request_id=f"penalized-{i}", sampling=Sampling(**{penalty: value})))
            for i, state in enumerate(baseline)
        ],
    )
    changed = [a.request.request_id for a, b in zip(baseline, penalized) if a.output_tokens != b.output_tokens]
    driver.checks.append({"penalty": penalty, "value": value, "changed_requests": changed})
    # This is an integration sensitivity check, not a distribution/penalty-math
    # test. One responsive prompt suffices; exact arithmetic belongs to synthetic
    # logits, where model confidence cannot mask a correctly applied penalty.
    assert changed, f"{penalty}={value} had no effect on any of {len(baseline)} prompts"


@pytest.mark.parametrize(
    "penalty,value", [("repetition_penalty", 2.5), ("presence_penalty", 2.0), ("frequency_penalty", 2.0)]
)
def test_neighbor_penalties_preserve_unpenalized_request(request_driver, penalty, value):
    driver = request_driver
    if driver.adapter.capacity < 2:
        pytest.skip("Neighbor isolation requires at least two physical slots")
    baseline = run_batch(
        driver,
        [
            (slot, Request(f"clean-{slot}", "a b c a b c a b c", max_tokens=16))
            for slot in range(driver.adapter.capacity)
        ],
    )
    mixed = run_batch(
        driver,
        [
            (
                state.slot,
                replace(state.request, request_id=f"mixed-{i}", sampling=Sampling(**({penalty: value} if i else {}))),
            )
            for i, state in enumerate(baseline)
        ],
    )
    assert_same_tokens(baseline[0], mixed[0])
