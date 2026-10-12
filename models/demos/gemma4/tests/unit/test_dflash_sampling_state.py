# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host coverage for sampling state when dFlash returns a fused argmax token."""

from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tests.unit.dflash_contract_harness import (
    _ordinary,
    _prefill,
    _propose,
    _sampling,
    _start_solo,
    _table,
    _tensor,
    _verify,
    make_expect_error,
)
from models.demos.gemma4.tests.unit.dflash_sampling_harness import attach_sampling_state


@pytest.fixture
def expect_error():
    return make_expect_error()


def test_fused_bootstrap_resets_sampling_before_the_first_proposal_declines(model):
    sampler = model.model[0].sampling
    _prefill(model, key=10)
    _ordinary(
        model,
        [3],
        [2],
        [10],
        result="sampled",
        sampling_params=_sampling(1, temperature=1.0, top_k=20, presence_penalty=0.5, seed=7),
    )
    assert sampler.uploads[-1]["k"][0] == 20 and sampler._penalties_active
    model.release_request(0)
    _prefill(model, key=20)
    model.model_args[0].max_seq_len = 8
    greedy = _sampling(1, seed=19)
    out = _ordinary(model, [3], [2], [20], sampling_params=greedy)
    assert out.tolist() == [150]
    assert len(sampler.uploads) == 2 and sampler.uploads[-1]["k"][0] == 1
    assert not sampler._penalties_active
    assert sampler.seed_manager.seeds[0] == 19 and sampler.seed_manager.positions[0] == 2
    assert len([event for event in sampler.seed_manager.events if event[0] == "draw"]) == 1

    proposal = _propose(model, [[150, -1, -1, -1, -1, -1]], [[3, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [0]
    _ordinary(
        model,
        [150],
        [3],
        [20],
        result="plain",
        sampling_params=greedy,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    assert len(sampler.uploads) == 2 and sampler.uploads[-1]["k"][0] == 1
    assert not sampler._penalties_active
    assert sampler.seed_manager.seeds[0] == 19 and sampler.seed_manager.positions[0] == 3
    assert len([event for event in sampler.seed_manager.events if event[0] == "draw"]) == 2
    assert model.decode_commands[-1] == dict(
        reload_inputs=True, reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False
    )


@pytest.mark.parametrize("step", ["ordinary", "draftless-verify", "retained-verify"])
def test_fused_outputs_apply_sampling_commands_and_remap_once_without_drawing(model, step):
    sampler = model.model[0].sampling
    if step == "retained-verify":
        _start_solo(model)
    else:
        _prefill(model, slot=1)
    sampler.seed_manager.events.clear()
    model.sampling_commands.clear()
    params = _sampling(2, seed=[17, None] if step != "retained-verify" else [None, 17])
    commands = dict(reload_inputs=True, reload_page_table=False, reload_sampling_params=True, reset_sampling_state=True)
    kwargs = dict(sampling_params=params, slot_remap=[1, 0], **commands)
    if step == "ordinary":
        _ordinary(model, [3, 0], [2, -1], [10, 0], **kwargs)
        row, position = 0, 2
    elif step == "draftless-verify":
        _verify(model, [[3, 0, 0, 0, 0, 0], [-1] * 6], [list(range(2, 8)), [-1] * 6], [0, 0], [10, 0], **kwargs)
        row, position = 0, 2
    else:
        _verify(
            model,
            [[-1] * 6, [150, 201, 202, 203, 204, 205]],
            [[-1] * 6, list(range(3, 9))],
            [0, 5],
            [0, 10],
            **kwargs,
        )
        row, position = 1, 3
    assert len(model.sampling_commands) == 1
    assert {name: model.sampling_commands[0][name] for name in commands if name != "reload_page_table"} == {
        name: value for name, value in commands.items() if name != "reload_page_table"
    }
    events = sampler.seed_manager.events
    assert len([event for event in events if event[0] == "remap"]) == 1
    assert not any(event[0] == "draw" for event in events)
    assert sampler.seed_manager.seeds[row] == 17
    assert sampler.seed_manager.positions[row] == position


def test_retained_fused_output_preserves_false_sampling_commands(model):
    _start_solo(model)
    sampler = model.model[0].sampling
    before = len(sampler.uploads)
    sampler.seed_manager.events.clear()
    _verify(
        model,
        [[150, 201, 202, 203, 204, 205]],
        [list(range(3, 9))],
        [5],
        [10],
        reload_inputs=False,
        reload_page_table=True,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    assert len(sampler.uploads) == before
    assert model.sampling_commands[-1]["reload_inputs"] is False
    assert model.sampling_commands[-1]["reload_sampling_params"] is False
    assert model.sampling_commands[-1]["reset_sampling_state"] is False
    assert not any(event[0] in {"reset", "align", "draw"} for event in sampler.seed_manager.events)


def test_host_verify_rejection_preserves_seed_state_before_remap(model, expect_error):
    _start_solo(model)
    sampler = model.model[0].sampling
    sampler.seed_manager.seeds[0] = 23
    sampler.seed_manager.events.clear()
    model.sampling_commands.clear()
    with expect_error(ValueError, "cannot apply host sampling controls"):
        _verify(
            model,
            [[-1] * 6, [150, 201, 202, 203, 204, 205]],
            [[-1] * 6, list(range(3, 9))],
            [0, 5],
            [0, 10],
            sampling=False,
            slot_remap=[1, 0],
        )
    assert sampler.seed_manager.seeds[0] == 23
    assert sampler.seed_manager.seeds[1] is None
    assert sampler.seed_manager.events == []
    assert model.sampling_commands == []


@pytest.mark.parametrize("data_parallel,sampling_dp", [(1, 1), (2, 1), (1, 2)])
def test_shared_preparation_keeps_parameters_history_and_seeds_with_their_sampling_groups(
    adapter, data_parallel, sampling_dp
):
    model = SimpleNamespace(model=[SimpleNamespace(sampling_dp=sampling_dp) for _ in range(data_parallel)])
    attach_sampling_state(model)
    rows = 32 * max(data_parallel, sampling_dp)
    top_k = [1 + row % 20 for row in range(rows)]
    params = _sampling(rows, temperature=1.0, top_k=top_k, presence_penalty=0.5, seed=list(range(100, 100 + rows)))
    positions = torch.arange(rows, dtype=torch.int32)
    positions[-1] = -1
    prompts = torch.arange(2 * rows, dtype=torch.int32).reshape(rows, 2)
    outputs = torch.arange(rows, dtype=torch.int32).reshape(rows, 1)
    active = model._prepare_decode_sampling_state(
        params,
        start_pos=torch.chunk(positions, data_parallel),
        prompt_tokens=prompts,
        output_tokens=outputs,
        slot_remap=list(range(rows)),
        reload_inputs=True,
        reload_sampling_params=True,
        reset_sampling_state=True,
    )
    per_model = rows // data_parallel
    for index, target in enumerate(model.model):
        sampler = target.sampling
        start, end = index * per_model, (index + 1) * per_model
        assert sampler.uploads[-1]["k"] == top_k[start:end]
        assert torch.equal(sampler.prompt_resets[-1], prompts[start:end])
        assert torch.equal(sampler.output_resets[-1], outputs[start:end])
        expected_active = [row for row in range(per_model) if int(positions[start + row]) >= 0]
        assert active[index] == expected_active
        assert [sampler.seed_manager.seeds[row] for row in expected_active] == [
            100 + start + row for row in expected_active
        ]
        assert not any(event[0] == "draw" for event in sampler.seed_manager.events)


@pytest.mark.parametrize(
    "missing", ["reload_inputs", "reload_page_table", "reload_sampling_params", "reset_sampling_state"]
)
def test_fused_decode_requires_each_command_before_mutating_state(model, expect_error, missing):
    _prefill(model)
    commands = dict(reload_inputs=True, reload_page_table=False, reload_sampling_params=True, reset_sampling_state=True)
    del commands[missing]
    events = list(model.events)
    with expect_error(TypeError, missing):
        model.decode_forward(
            tokens=_tensor([[3]]),
            start_pos=_tensor([2]),
            page_table=_table([10]),
            kv_cache=model.kv_cache,
            sampling_params=_sampling(1),
            **commands,
        )
    assert model.events == events
    assert model.sampling_commands == []
