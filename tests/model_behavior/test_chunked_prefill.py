# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Needle recall after real, observed nonzero prefill resume offsets."""

from dataclasses import replace

import pytest

from tests.model_behavior.driver import Request, run_batch

pytestmark = pytest.mark.timeout(1800)

FILLER = "The archive contains ordinary notes about weather, gardens, rivers, and roads. "
NEEDLES = ("violet compass 7319", "copper lantern 4826", "silver meadow 9053", "amber pebble 1648")


def needle_prompt(adapter, needle, min_tokens):
    def build(repeats):
        return (
            "Read the following archive and remember its secret passphrase.\n\n"
            + FILLER * (repeats // 2)
            + f"\nThe secret passphrase is: {needle}. Remember it exactly.\n"
            + FILLER * (repeats - repeats // 2)
            + "\nWhat is the secret passphrase? Reply with only the passphrase."
        )

    # Size by the model's real tokenizer, including its instruct wrapper.
    low, high = 1, 1
    while len(adapter.encode(build(high))) < min_tokens:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if len(adapter.encode(build(middle))) < min_tokens:
            low = middle + 1
        else:
            high = middle
    return build(low)


def assert_needle_recalled(adapter, state, needle):
    assert len(state.output_tokens) == state.request.max_tokens, "Incomplete needle response"
    text = adapter.decode_tokens(state.output_tokens)
    assert needle in text.lower(), f"{state.request.request_id}, slot {state.slot}: expected {needle!r}, got {text!r}"
    return text


@pytest.mark.parametrize("scenario", ["solo", "long", "shared-cache"])
def test_chunked_prefill_recalls_needle(request_driver, scenario):
    driver = request_driver
    if not getattr(driver.adapter, "supports_chunked_prefill", True):
        pytest.skip("This model cannot resume an externally supplied prefix; internal chunking is a different contract")
    if scenario == "shared-cache" and len(driver.adapter.long_context_slots) < 2:
        pytest.skip("Shared-cache recall requires at least two long-context slots")
    slots = driver.adapter.long_context_slots if scenario == "shared-cache" else driver.adapter.long_context_slots[:1]
    placements = []
    for i, slot in enumerate(slots):
        prompt = needle_prompt(driver.adapter, NEEDLES[i], 6000 if scenario == "long" else 1800 + 128 * i)
        placements.append((slot, Request(f"whole-{i}", prompt, max_tokens=24)))
    whole = run_batch(driver, placements)
    for i, state in enumerate(whole):
        assert_needle_recalled(driver.adapter, state, NEEDLES[i])

    # Replace the baseline's KV contents before rebuilding prefixes. Otherwise
    # a broken chunk write could read the correct answer left by the control.
    run_batch(
        driver,
        [
            (
                state.slot,
                replace(
                    state.request,
                    request_id=f"previous-owner-{i}",
                    prompt=state.request.prompt.replace(NEEDLES[i], "obsolete ticket 0000"),
                    max_tokens=1,
                ),
            )
            for i, state in enumerate(whole)
        ],
    )
    start_event = len(driver.adapter.prefill_events)
    chunked = run_batch(
        driver,
        [
            (
                state.slot,
                replace(
                    state.request,
                    request_id=f"chunked-{i}",
                    prefill_chunk_ends=(1024, 3072) if scenario == "long" else (1024 + 128 * i,),
                ),
            )
            for i, state in enumerate(whole)
        ],
    )
    for i, state in enumerate(chunked):
        if scenario == "long":
            assert len(state.prompt_tokens) - state.request.prefill_chunk_ends[-1] > 2048
        text = assert_needle_recalled(driver.adapter, state, NEEDLES[i])
        events = [
            event
            for event in driver.adapter.prefill_events[start_event:]
            if event["request_id"] == state.request.request_id
        ]
        expected = [0, *state.request.prefill_chunk_ends]
        assert [event["start"] for event in events] == expected, f"Incomplete chunk schedule: {events}"
        assert [event["end"] for event in events] == [*state.request.prefill_chunk_ends, len(state.prompt_tokens)]
        for event in events[1:]:
            assert any(
                call["start"] == event["start"] for call in event["observed"]
            ), f"Resume was silently ignored: {event}"
        driver.checks.append(dict(request_id=state.request.request_id, needle=NEEDLES[i], response=text, chunks=events))
