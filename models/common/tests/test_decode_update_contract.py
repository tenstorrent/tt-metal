# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import inspect
import random
from types import SimpleNamespace

import torch

from models.common.sampling.generator import SamplingGenerator, SeedManager


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


def test_sampling_update_commands_are_required_and_have_no_legacy_alias():
    params = inspect.signature(SamplingGenerator.apply_decode_state).parameters

    assert params["reload_sampling_params"].default is inspect.Parameter.empty
    assert params["reset_sampling_state"].default is inspect.Parameter.empty
    assert "reset_batch" not in params


def test_qwen_vl_slot_remap_moves_persistent_rope_deltas():
    from models.demos.qwen25_vl.tt.generator import Generator as Qwen25Generator
    from models.demos.qwen3_vl.tt.generator import Generator as Qwen3Generator

    for generator_cls in (Qwen25Generator, Qwen3Generator):
        generator = SimpleNamespace(
            model=SimpleNamespace(
                rope_setup=SimpleNamespace(
                    batch_size=4,
                    rope_deltas=torch.tensor([10, 20, 30, 40]),
                )
            )
        )

        generator_cls.remap_rope_deltas(generator, [3, 1, 2, 3])

        assert generator.model.rope_setup.rope_deltas.tolist() == [40, 20, 30, 40]


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
