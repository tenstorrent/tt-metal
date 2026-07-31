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


def test_unseeded_decode_reset_loads_fresh_device_seed(monkeypatch):
    manager = SeedManager.__new__(SeedManager)
    manager.max_batch_size = 1
    manager.seeds = [None]
    manager.seed_counters = [4]
    manager.rngs = [random.Random(1)]
    manager._seed_active = False
    manager._reseted = False
    manager._needs_skip = False
    manager._active_request_seed = False
    manager._seed_mapper = None
    seed_buffer = object()
    manager.tt_sampling = SimpleNamespace(seeds_tt_tensor=seed_buffer)
    before = manager.rngs[0].getstate()
    host_seed_tensor = object()
    uploads = []

    monkeypatch.setattr(manager, "_next_unseeded_device_seed", lambda: 123)
    monkeypatch.setattr(
        "models.common.sampling.generator.ttnn.from_torch",
        lambda *args, **kwargs: host_seed_tensor,
    )
    monkeypatch.setattr(
        "models.common.sampling.generator.ttnn.copy_host_to_device_tensor",
        lambda host, device: uploads.append((host, device)),
    )

    # The conditional path would see None == None and do nothing. A decode
    # state reset must be unconditional so the following get_new_values()
    # enters its init state and uploads a fresh device seed.
    manager.reset_seed_from_slots([None], [0])
    manager.get_new_values([0])

    assert manager.seed_counters == [0]
    assert manager.rngs[0].getstate() != before
    assert uploads == [(host_seed_tensor, seed_buffer)]
    assert manager._needs_skip
    assert not manager._reseted


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
