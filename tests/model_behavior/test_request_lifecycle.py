# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from tests.model_behavior.driver import Request, Sampling, assert_same_tokens


@pytest.mark.parametrize("batch_layout", ["sparse", "full"])
def test_mixed_request_replay(request_driver, batch_layout):
    driver = request_driver
    capacity = driver.adapter.capacity
    if capacity < 4:
        pytest.skip("Mixed replay requires at least four physical slots")
    slots = (0, capacity // 2 - 1, capacity // 2, capacity - 1)
    requests = [
        Request("greedy", "Continue counting: one, two, three,"),
        Request("seeded", "Write a short story about a blue bird.", Sampling(temperature=0.8, seed=17)),
        Request(
            "repetition",
            "The garden was quiet and the garden was green.",
            Sampling(temperature=0.8, seed=17, repetition_penalty=1.2),
        ),
        Request(
            "frequency",
            "List a few things you might find in a kitchen.",
            Sampling(temperature=0.7, seed=23, presence_penalty=0.5, frequency_penalty=0.5),
        ),
    ]
    if batch_layout == "full":
        templates = requests + [
            Request("top1", "Name a season.", Sampling(temperature=2.0, top_k=1, seed=0)),
            Request("top5", "Invent a name for a city.", Sampling(temperature=1.5, top_k=5, seed=31)),
            Request("nucleus", "The reason is that the", Sampling(temperature=1.0, top_p=0.5, seed=43)),
            Request("low-temp", "Count from five:", Sampling(temperature=0.01, seed=47)),
        ]
        slots = tuple(range(capacity))
        requests = [replace(templates[i % len(templates)], request_id=f"full-{i}") for i in slots]
    driver.admit(list(zip(slots, requests)))
    driver.drain()
    replay = [replace(request, request_id=request.request_id + "-replay") for request in requests]
    driver.admit(list(zip(slots, replay)))
    driver.drain()
    for first, second in zip(requests, replay):
        assert_same_tokens(driver.requests[first.request_id], driver.requests[second.request_id])


def test_admission_preserves_surviving_request(request_driver):
    driver = request_driver
    if driver.adapter.capacity < 2:
        pytest.skip("Admission isolation requires two physical slots")
    neighbor_slot = driver.adapter.capacity - 1
    survivor = Request(
        "control",
        "Continue the story: the traveler opened the old wooden door and",
        Sampling(temperature=0.8, seed=101, repetition_penalty=1.2, frequency_penalty=0.5),
    )
    neighbor = Request("neighbor", "Count slowly from one to ten.")
    driver.admit([(0, survivor), (neighbor_slot, neighbor)])
    driver.drain()

    driver.admit(
        [
            (0, replace(survivor, request_id="survivor")),
            (neighbor_slot, replace(neighbor, request_id="short-neighbor", max_tokens=4)),
        ]
    )
    for _ in range(3):
        driver.step()
    assert neighbor_slot not in driver.active
    before_admission = list(driver.requests["survivor"].output_tokens)
    driver.admit(
        [
            (
                neighbor_slot,
                Request(
                    "replacement",
                    "Name a few colors seen at sunset.",
                    Sampling(temperature=0.9, seed=303, presence_penalty=1.0),
                    max_tokens=9,
                ),
            )
        ]
    )
    assert driver.requests["survivor"].output_tokens == before_admission
    driver.drain()
    assert_same_tokens(driver.requests["control"], driver.requests["survivor"])


def test_completed_request_state_does_not_leak_on_slot_reuse(request_driver):
    driver = request_driver
    slot = driver.adapter.capacity - 1
    # Unseeded greedy exercises the internal sampler trace too. Explicitly
    # seeded sampling deliberately bypasses that trace in the current backend.
    target = Request(
        "baseline",
        "The little boat crossed the lake and",
        Sampling(repetition_penalty=1.1, presence_penalty=0.5, frequency_penalty=0.5),
    )
    driver.admit([(slot, target)])
    driver.drain()
    driver.admit(
        [
            (
                slot,
                Request(
                    "previous-owner",
                    "Repeat these words: red red blue blue green green.",
                    Sampling(temperature=0.9, seed=909, repetition_penalty=1.5, frequency_penalty=1.0),
                    max_tokens=20,
                ),
            )
        ]
    )
    driver.drain()
    driver.admit([(slot, replace(target, request_id="reused"))])
    driver.drain()
    assert_same_tokens(driver.requests["baseline"], driver.requests["reused"])
