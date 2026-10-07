# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host checks for the request-seed boundary used by traced device sampling."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from models.demos.llama31_8b_qb2.tt.generator import LlamaGenerator
from models.demos.llama31_8b_qb2.tt.generator_vllm import LlamaForCausalLM


def make_generator(batch=1):
    generator = LlamaGenerator.__new__(LlamaGenerator)
    generator.mesh_device = None
    generator.max_batch_size = batch
    generator.model = SimpleNamespace(supported_context=131072)
    generator.sampler = SimpleNamespace(reset_params=Mock(), seeds_tt_tensor=object())
    generator._copy = Mock()
    return generator


def sampling_params(seeds):
    return SimpleNamespace(top_k=20, top_p=1.0, temperature=0.7, seed=seeds)


@pytest.mark.parametrize(
    "seed",
    [
        0,
        42,
        2**31 - 131072 - 2,
        2**31,
        -(2**31) - 1,
        4294967168,
        2**32 - 1,
        2**60 + 7,
        -(2**60) + 19,
        2**63 - 1,
        -(2**63),
    ],
)
@pytest.mark.parametrize("position", [0, 127, 131071])
def test_serving_seed_stays_safe_through_remaining_context(seed, position):
    generator = make_generator()
    adapter = LlamaForCausalLM(generator)
    adapter._sampling(sampling_params([seed]), [position])

    uploaded, target, counter = generator._copy.call_args.args
    assert uploaded.dtype == torch.int32
    assert uploaded.shape == (32,)
    assert target is generator.sampler.seeds_tt_tensor
    assert counter == "seed_refreshes"
    # Every subsequent sample advances this seed once on device. Include the
    # final increment after the last supported token position.
    final_seed = uploaded.long() + generator.model.supported_context - position
    assert (uploaded >= 0).all()
    assert (final_seed < 2**31 - 1).all()
    if seed in (0, 42):
        assert uploaded[0].item() == seed + position
    generator.set_sampling(seed=seed)
    initial = generator._copy.call_args.args[0]
    assert torch.equal(uploaded.long(), initial.long() + position)


def test_seed_replay_and_slot_remapping_preserve_each_request_stream():
    generator = make_generator(batch=3)
    adapter = LlamaForCausalLM(generator)
    seeds = [4294967168, 2**63 - 1, -(2**63)]
    positions = [127, 11, 5]

    adapter._sampling(sampling_params(seeds), positions)
    first = generator._copy.call_args.args[0].clone()
    adapter._sampling(sampling_params([7, 8, 9]), [40, 50, 60])
    adapter._sampling(sampling_params(seeds), positions)
    assert torch.equal(generator._copy.call_args.args[0], first)

    order = [2, 0, 1]
    adapter._sampling(sampling_params([seeds[i] for i in order]), [positions[i] + 1 for i in order])
    assert torch.equal(generator._copy.call_args.args[0][:3], first[order] + 1)


def test_parameter_refresh_preserves_advancing_device_seed():
    generator = make_generator()
    adapter = LlamaForCausalLM(generator)
    adapter._sampling(sampling_params([2**63 - 1]), [127], reset_state=False)
    generator._copy.assert_not_called()
    generator.sampler.reset_params.assert_called_once()


@pytest.mark.parametrize("seed", [2**32 - 1, 2**63 - 1, -(2**63)])
def test_direct_generator_seed_leaves_room_for_full_context(seed):
    generator = make_generator()
    generator.set_sampling(seed=seed)
    uploaded = generator._copy.call_args.args[0]
    assert (uploaded >= 0).all()
    assert (uploaded.long() + generator.model.supported_context < 2**31 - 1).all()
