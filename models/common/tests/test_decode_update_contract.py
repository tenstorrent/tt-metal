# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for the sampling half of the explicit vLLM decode update contract.

import random
from types import SimpleNamespace

from models.common.sampling.generator import (
    SamplingGenerator,
    SeedManager,
    should_align_decode_seed_counters,
)


def _fake_sampling_generator():
    calls = []
    fake = SimpleNamespace(
        tt_sampling=SimpleNamespace(max_batch_size=4),
        reset_sampling_params=lambda params: calls.append(("params", params)),
        reset_prompt_tokens=lambda tokens: calls.append(("prompt", tokens)),
        reset_output_state=lambda tokens: calls.append(("output", tokens)),
    )
    return fake, calls


def test_sampling_state_can_reset_without_reloading_params():
    fake, calls = _fake_sampling_generator()

    SamplingGenerator.apply_decode_state(
        fake,
        [object()],
        reload_sampling_params=False,
        reset_sampling_state=True,
        prompt_tokens="prompt",
        output_tokens="output",
    )

    assert calls == [("prompt", "prompt"), ("output", "output")]


def test_no_sampling_updates_is_a_true_noop():
    fake, calls = _fake_sampling_generator()

    SamplingGenerator.apply_decode_state(
        fake,
        [object()],
        reload_sampling_params=False,
        reset_sampling_state=False,
    )

    assert calls == []


def test_legacy_reset_batch_still_rebuilds_sampling_state(monkeypatch):
    fake, calls = _fake_sampling_generator()
    formatted = object()
    monkeypatch.setattr(
        "models.common.sampling.generator.format_sampling_params",
        lambda params, max_batch_size: formatted,
    )

    SamplingGenerator.apply_decode_state(
        fake,
        [object()],
        reset_batch=True,
        prompt_tokens="prompt",
        output_tokens="output",
    )

    assert calls == [
        ("params", formatted),
        ("prompt", "prompt"),
        ("output", "output"),
    ]


def test_decode_batch_reset_clears_removed_and_reseeds_identity_slot():
    manager = SeedManager.__new__(SeedManager)
    manager.max_batch_size = 3
    manager.seeds = [11, None, 22]
    manager.seed_counters = [5, 6, 7]
    manager.rngs = [random.Random(1), random.Random(2), random.Random(3)]
    manager._seed_active = True
    manager._reseted = False
    before_identity_state = manager.rngs[1].getstate()

    manager.reset_decode_batch([None, None, None], [1])

    assert manager.seeds == [None, None, None]
    assert manager.seed_counters == [0, 0, 0]
    assert manager.rngs[1].getstate() != before_identity_state
    assert not manager._seed_active
    assert manager._reseted


def test_legacy_seed_alignment_policy_is_preserved():
    assert should_align_decode_seed_counters(
        explicit_contract=False,
        reset_sampling_state=False,
        legacy_alignment=True,
    )
    assert not should_align_decode_seed_counters(
        explicit_contract=True,
        reset_sampling_state=False,
        legacy_alignment=True,
    )
    assert should_align_decode_seed_counters(
        explicit_contract=True,
        reset_sampling_state=True,
        legacy_alignment=False,
    )
