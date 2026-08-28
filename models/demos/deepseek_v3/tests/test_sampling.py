#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import random
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.sampling import SamplingGenerator, SamplingParams, SeedManager, format_sampling_params
from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.tt.generator_vllm import DeepseekV3ForCausalLM
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, get_fabric_config, make_deepseek_sampling_args


def _make_lm_head_sharded_logits(torch_input, mesh_device):
    return ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )


def _extract_all_tokens(tt_out_tok, mesh_device, batch_size_per_row):
    composed = ttnn.to_torch(
        tt_out_tok,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=tuple(mesh_device.shape)),
    )
    if composed.ndim == 4:
        if tt_out_tok.shape[-2] == batch_size_per_row:
            tokens = composed[:, :, :, 0]
        elif tt_out_tok.shape[-1] == batch_size_per_row:
            tokens = composed[:, :, 0, :batch_size_per_row]
        else:
            tokens = composed
        tokens = tokens.reshape(-1)
    else:
        tokens = composed.reshape(-1)
    batch_size = batch_size_per_row * int(mesh_device.shape[0])
    return tokens[:batch_size].to(torch.int64)


def _sample_device_tokens(mesh_device, ccl, args, torch_input, user_params):
    batch_size = USERS_PER_ROW * int(mesh_device.shape[0])
    tt_input = _make_lm_head_sharded_logits(torch_input, mesh_device)
    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=ccl)
    params = format_sampling_params(user_params, max_batch_size=batch_size)
    sampling.reset_sampling_params(params)
    sampling.reset_prompt_tokens(torch.zeros((USERS_PER_ROW, 1), dtype=torch.int64))
    sampling.reset_output_state(torch.zeros((USERS_PER_ROW, 1), dtype=torch.int64))
    sampling.seed_manager.reset_seed(params.seed, list(range(batch_size)))
    sampling.seed_manager.get_new_values()
    tt_tokens, _ = sampling.sample(tt_input, enable_trace=False)
    device_tokens = _extract_all_tokens(tt_tokens, mesh_device, USERS_PER_ROW)
    ttnn.deallocate(tt_tokens)
    ttnn.deallocate(tt_input)
    return device_tokens


class _FakeSeedManager:
    def __init__(self, max_batch_size):
        self.max_batch_size = max_batch_size
        self.reset_calls = []
        self.new_value_calls = []

        self.conditional_reset_calls = []
        self.align_calls = []
        self.remap_calls = []
        self.deactivate_calls = []

    def reset_seed_from_slots(self, seeds, user_ids):
        self.reset_calls.append((seeds, user_ids))

    def reset_seed_from_slots_if_needed(self, seeds, user_ids):
        self.conditional_reset_calls.append((seeds, user_ids))

    def align_seed_counters_to_positions(self, seeds, user_ids, positions):
        self.align_calls.append((seeds, user_ids, positions))

    def apply_slot_remap(self, remap):
        self.remap_calls.append(remap)

    def deactivate_slots_except(self, live_slots):
        self.deactivate_calls.append(live_slots)

    def get_new_values(self, user_slots):
        self.new_value_calls.append(user_slots)


class _FakeSamplingGenerator:
    def __init__(self, *, padded_batch_size=32, sampling_dp=2):
        self.tt_sampling = SimpleNamespace(max_batch_size=padded_batch_size, _sampling_dp=sampling_dp)
        self.seed_manager = _FakeSeedManager(padded_batch_size * sampling_dp)
        self.decode_state_calls = []

    def apply_decode_state(self, sampling_param_chunks, **kwargs):
        self.decode_state_calls.append((sampling_param_chunks, kwargs))


def _fake_deepseek_generator(*, batch_size_per_row=8, sampling_dp=2):
    class _FakeDeepseekGenerator:
        _apply_sampling_state = DeepseekGenerator._apply_sampling_state
        _sampling_device_slot = DeepseekGenerator._sampling_device_slot
        _sampling_device_slots = DeepseekGenerator._sampling_device_slots
        _sampling_device_positions = DeepseekGenerator._sampling_device_positions
        _sampling_params_for_user_slots = DeepseekGenerator._sampling_params_for_user_slots
        _sampling_history_for_user_slots = DeepseekGenerator._sampling_history_for_user_slots
        _sampling_device_history = DeepseekGenerator._sampling_device_history
        _sampling_device_slot_remap = DeepseekGenerator._sampling_device_slot_remap
        _apply_sampling_slot_remap = DeepseekGenerator._apply_sampling_slot_remap
        _sampling_device_seed_slots = DeepseekGenerator._sampling_device_seed_slots
        _to_local_sampling_params = DeepseekGenerator._to_local_sampling_params
        _normalize_sampling_params_for_batch = DeepseekGenerator._normalize_sampling_params_for_batch

        def __init__(self):
            self.batch_size_per_row = batch_size_per_row
            self.batch_size = batch_size_per_row * sampling_dp
            self.sampling_generator = _FakeSamplingGenerator(padded_batch_size=32, sampling_dp=sampling_dp)

    return _FakeDeepseekGenerator()


def test_deepseek_reset_sampling_state_does_not_preformat_sampling_params():
    batch_size = 64
    batch_size_per_row = 32
    generator = _fake_deepseek_generator(batch_size_per_row=batch_size_per_row, sampling_dp=2)
    sampling_generator = generator.sampling_generator
    sampling_params = SamplingParams(
        temperature=[0.6] * batch_size,
        top_k=[32] * batch_size,
        top_p=[0.95] * batch_size,
        seed=[1234] * batch_size,
    )

    DeepseekGenerator._apply_sampling_state(
        generator,
        sampling_params,
        batch_size,
        batch_size_per_row,
        reload_sampling_params=True,
        reset_sampling_state=True,
    )

    [(sampling_param_chunks, kwargs)] = sampling_generator.decode_state_calls
    assert len(sampling_param_chunks) == 2
    assert sampling_param_chunks[0].temperature[0] == 0.6
    assert sampling_param_chunks[1].temperature[0] == 0.6
    first_chunk_temperature = format_sampling_params(
        sampling_param_chunks[0], max_batch_size=batch_size_per_row
    ).temperature[0]
    assert first_chunk_temperature == pytest.approx(1 / 0.6)
    assert kwargs["reload_sampling_params"] is True
    assert kwargs["reset_sampling_state"] is True
    assert kwargs["prompt_tokens"].shape == (batch_size, 1)
    assert torch.all(kwargs["prompt_tokens"] == -1)
    assert kwargs["output_tokens"] is None
    [seed_reset] = sampling_generator.seed_manager.reset_calls
    assert seed_reset == ([1234] * batch_size, list(range(batch_size)))


def test_deepseek_sampling_user_slots_map_to_row_padded_device_slots():
    generator = _fake_deepseek_generator(batch_size_per_row=8, sampling_dp=3)

    assert DeepseekGenerator._sampling_device_slots(generator, [0, 7, 8, 15, 16, 23]) == [0, 7, 32, 39, 64, 71]


def test_deepseek_sampling_seed_reset_uses_row_padded_device_slots():
    batch_size = 32
    generator = _fake_deepseek_generator(batch_size_per_row=8, sampling_dp=4)
    sampling_params = SamplingParams(
        temperature=[0.0] * batch_size,
        top_k=[1] * batch_size,
        top_p=[1.0] * batch_size,
        seed=list(range(100, 100 + batch_size)),
    )

    DeepseekGenerator._apply_sampling_state(
        generator,
        sampling_params,
        batch_size,
        generator.batch_size_per_row,
        reload_sampling_params=True,
        reset_sampling_state=True,
    )

    [(seeds, user_ids)] = generator.sampling_generator.seed_manager.reset_calls
    assert user_ids == list(range(128))
    assert seeds[:8] == list(range(100, 108))
    assert seeds[8:32] == [None] * 24
    assert seeds[32:40] == list(range(108, 116))
    assert seeds[40:64] == [None] * 24
    assert seeds[64:72] == list(range(116, 124))
    assert seeds[72:96] == [None] * 24
    assert seeds[96:104] == list(range(124, 132))
    assert seeds[104:] == [None] * 24


def test_deepseek_unseeded_sampling_reset_initializes_all_device_slots():
    batch_size = 32
    generator = _fake_deepseek_generator(batch_size_per_row=16, sampling_dp=2)
    sampling_params = SamplingParams(
        temperature=[0.6] * batch_size,
        top_k=[32] * batch_size,
        top_p=[0.95] * batch_size,
        seed=None,
    )

    DeepseekGenerator._apply_sampling_state(
        generator,
        sampling_params,
        batch_size,
        generator.batch_size_per_row,
        reload_sampling_params=True,
        reset_sampling_state=True,
    )

    [(seeds, user_ids)] = generator.sampling_generator.seed_manager.reset_calls
    assert seeds == [None] * 64
    assert user_ids == list(range(64))


def test_deepseek_sampling_reset_aligns_active_row_padded_seed_positions():
    batch_size = 32
    generator = _fake_deepseek_generator(batch_size_per_row=16, sampling_dp=2)
    sampling_params = SamplingParams(
        temperature=[0.6] * batch_size,
        top_k=[32] * batch_size,
        top_p=[0.95] * batch_size,
        seed=list(range(100, 100 + batch_size)),
    )
    positions = torch.tensor([10] + [-1] * 15 + [20] + [-1] * 15)

    DeepseekGenerator._apply_sampling_state(
        generator,
        sampling_params,
        batch_size,
        generator.batch_size_per_row,
        reload_sampling_params=False,
        reset_sampling_state=True,
        user_slots=[0, 16],
        positions=positions,
    )

    [(seeds, user_ids)] = generator.sampling_generator.seed_manager.reset_calls
    assert user_ids == [0, 32]
    [(aligned_seeds, aligned_user_ids, device_positions)] = generator.sampling_generator.seed_manager.align_calls
    assert aligned_seeds == seeds
    assert aligned_user_ids == [0, 32]
    assert device_positions[0] == 10
    assert device_positions[32] == 20
    assert device_positions[1:32] == [-1] * 31


def test_deepseek_sampling_histories_use_row_padded_device_layout():
    generator = _fake_deepseek_generator(batch_size_per_row=2, sampling_dp=2)
    compact = torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]])

    padded = DeepseekGenerator._sampling_device_history(generator, compact)

    assert padded.shape == (64, 2)
    assert padded[0].tolist() == [10, 11]
    assert padded[1].tolist() == [20, 21]
    assert padded[32].tolist() == [30, 31]
    assert padded[33].tolist() == [40, 41]
    assert torch.all(padded[2:32] == -1)
    assert torch.all(padded[34:] == -1)


def test_deepseek_prefill_sampling_state_uses_nonidentity_stable_slots():
    generator = _fake_deepseek_generator(batch_size_per_row=16, sampling_dp=2)
    generator.sampling_params = SamplingParams(
        temperature=[1.0] * generator.batch_size,
        top_k=[1] * generator.batch_size,
        top_p=[1.0] * generator.batch_size,
        seed=[None] * generator.batch_size,
    )
    sampling_params = SamplingParams(
        temperature=[0.6, 0.8],
        top_k=[16, 32],
        top_p=[0.9, 0.95],
        seed=[111, 222],
    )
    stable_slots = [17, 0]
    prompt_tokens = torch.tensor([[10, 11], [20, -1]])

    stable_params = DeepseekGenerator._sampling_params_for_user_slots(
        generator,
        sampling_params,
        stable_slots,
    )
    stable_history = DeepseekGenerator._sampling_history_for_user_slots(
        generator,
        prompt_tokens,
        stable_slots,
    )
    DeepseekGenerator._apply_sampling_state(
        generator,
        stable_params,
        generator.batch_size,
        generator.batch_size_per_row,
        reload_sampling_params=True,
        reset_sampling_state=True,
        user_slots=stable_slots,
        prompt_tokens=stable_history,
        preserve_unlisted_slots=True,
    )

    [(seeds, active_slots)] = generator.sampling_generator.seed_manager.reset_calls
    assert active_slots == [33, 0]
    assert seeds[33] == 111
    assert seeds[0] == 222
    [(_, commands)] = generator.sampling_generator.decode_state_calls
    assert commands["prompt_tokens"][33].tolist() == [10, 11]
    assert commands["prompt_tokens"][0].tolist() == [20, -1]
    assert commands["sampling_state_slots"] == [33, 0]
    assert generator.sampling_generator.seed_manager.deactivate_calls == []


def test_deepseek_prefill_preserves_live_slots_when_admitting_new_request():
    generator = _fake_deepseek_generator(batch_size_per_row=4, sampling_dp=2)
    generator.sampling_params = SamplingParams(
        temperature=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        top_k=[1, 2, 3, 4, 5, 6, 7, 8],
        top_p=[0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58],
        presence_penalty=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
        frequency_penalty=[0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
        repetition_penalty=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        seed=list(range(100, 108)),
        enable_log_probs=[False, True, False, True, False, True, False, True],
        num_logprobs=list(range(8)),
    )
    incoming = SamplingParams(
        temperature=[0.95],
        top_k=[42],
        top_p=[0.99],
        presence_penalty=[0.25],
        frequency_penalty=[0.35],
        repetition_penalty=[1.95],
        seed=[999],
        enable_log_probs=[False],
        num_logprobs=[9],
    )

    stable = DeepseekGenerator._sampling_params_for_user_slots(generator, incoming, [5])

    assert stable.temperature == [0.1, 0.2, 0.3, 0.4, 0.5, 0.95, 0.7, 0.8]
    assert stable.top_k == [1, 2, 3, 4, 5, 42, 7, 8]
    assert stable.top_p == [0.51, 0.52, 0.53, 0.54, 0.55, 0.99, 0.57, 0.58]
    assert stable.presence_penalty == [0.01, 0.02, 0.03, 0.04, 0.05, 0.25, 0.07, 0.08]
    assert stable.frequency_penalty == [0.11, 0.12, 0.13, 0.14, 0.15, 0.35, 0.17, 0.18]
    assert stable.repetition_penalty == [1.1, 1.2, 1.3, 1.4, 1.5, 1.95, 1.7, 1.8]
    assert stable.seed == [100, 101, 102, 103, 104, 999, 106, 107]
    assert stable.enable_log_probs == [False, True, False, True, False, False, False, True]
    assert stable.num_logprobs == [0, 1, 2, 3, 4, 9, 6, 7]


def test_deepseek_prompt_history_excludes_padding_tokens():
    tokens = torch.tensor([[10, 11, 99, 99], [20, 21, 22, 99]])
    lengths = torch.tensor([2, 3])

    history = DeepseekGenerator._prompt_history(tokens, lengths)

    assert history.tolist() == [[10, 11, -1, -1], [20, 21, 22, -1]]


def test_deepseek_sampling_reset_uses_vllm_penalty_histories():
    batch_size = 32
    generator = _fake_deepseek_generator(batch_size_per_row=16, sampling_dp=2)
    sampling_params = SamplingParams(
        temperature=[0.6] * batch_size,
        top_k=[32] * batch_size,
        top_p=[0.95] * batch_size,
        presence_penalty=[0.5] * batch_size,
        seed=[1234] * batch_size,
    )
    prompt_tokens = torch.arange(batch_size * 2).reshape(batch_size, 2)
    output_tokens = torch.arange(batch_size).reshape(batch_size, 1)

    DeepseekGenerator._apply_sampling_state(
        generator,
        sampling_params,
        batch_size,
        generator.batch_size_per_row,
        reload_sampling_params=True,
        reset_sampling_state=True,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )

    [(_, commands)] = generator.sampling_generator.decode_state_calls
    assert commands["prompt_tokens"][[0, 1, 32, 33]].tolist() == prompt_tokens[[0, 1, 16, 17]].tolist()
    assert commands["output_tokens"][[0, 1, 32, 33]].tolist() == output_tokens[[0, 1, 16, 17]].tolist()


def test_deepseek_sampling_params_can_reload_without_state_reset():
    batch_size = 32
    generator = _fake_deepseek_generator(batch_size_per_row=16, sampling_dp=2)
    sampling_params = SamplingParams(
        temperature=[0.7] * batch_size,
        top_k=[16] * batch_size,
        top_p=[0.9] * batch_size,
        seed=[1234] * batch_size,
    )

    DeepseekGenerator._apply_sampling_state(
        generator,
        sampling_params,
        batch_size,
        generator.batch_size_per_row,
        reload_sampling_params=True,
        reset_sampling_state=False,
        user_slots=[0, 16],
    )

    [(_, commands)] = generator.sampling_generator.decode_state_calls
    assert commands["reload_sampling_params"] is True
    assert commands["reset_sampling_state"] is False
    assert generator.sampling_generator.seed_manager.reset_calls == []
    [(_, user_ids)] = generator.sampling_generator.seed_manager.conditional_reset_calls
    assert user_ids == [0, 32]


def test_deepseek_sampling_retires_finished_tail_slot_seed_state():
    generator = _fake_deepseek_generator(batch_size_per_row=16, sampling_dp=2)
    sampling_params = SamplingParams(
        temperature=[0.7] * 32,
        top_k=[16] * 32,
        top_p=[0.9] * 32,
        seed=[1234] * 32,
    )

    DeepseekGenerator._apply_sampling_state(
        generator,
        sampling_params,
        32,
        generator.batch_size_per_row,
        reload_sampling_params=True,
        reset_sampling_state=True,
        user_slots=[0, 16],
    )

    assert generator.sampling_generator.seed_manager.deactivate_calls == [[0, 32]]


def test_deepseek_sampling_update_obeys_explicit_commands_without_value_comparison():
    sampling_params = SamplingParams(temperature=0.6, top_k=32, top_p=0.95, seed=None)
    update_calls = []
    generator = SimpleNamespace(
        batch_size=1,
        batch_size_per_row=1,
        sampling_params=sampling_params,
        sampling_generator=object(),
        _to_local_sampling_params=lambda params: params,
        _normalize_sampling_params_for_batch=lambda params, batch_size: params,
        _get_sampling_value=lambda value, index: value[index] if isinstance(value, list) else value,
        _apply_sampling_state=lambda params, batch_size, batch_size_per_row, **commands: update_calls.append(
            (params, batch_size, batch_size_per_row, commands)
        ),
    )

    DeepseekGenerator._validate_and_initialize_sampling(
        generator,
        sampling_params,
        sample_on_device=True,
        reload_sampling_params=False,
        reset_sampling_state=True,
    )

    assert update_calls == [
        (
            sampling_params,
            1,
            1,
            {
                "reload_sampling_params": False,
                "reset_sampling_state": True,
                "user_slots": None,
                "positions": None,
                "prompt_tokens": None,
                "output_tokens": None,
                "preserve_unlisted_slots": False,
            },
        )
    ]


def test_deepseek_slot_remap_maps_users_to_row_padded_seed_slots():
    generator = _fake_deepseek_generator(batch_size_per_row=2, sampling_dp=2)

    remap = DeepseekGenerator._sampling_device_slot_remap(generator, [3, 1, 2, 3])

    assert len(remap) == 64
    assert remap[0] == 33
    assert remap[1] == 1
    assert remap[32] == 32
    assert remap[33] == 33
    assert remap[2:32] == list(range(2, 32))


def test_deepseek_slot_remap_moves_seeded_state_across_padded_rows():
    generator = _fake_deepseek_generator(batch_size_per_row=2, sampling_dp=2)
    manager = SeedManager.__new__(SeedManager)
    manager.max_batch_size = 64
    manager.seeds = [None] * 64
    manager.seed_counters = [0] * 64
    manager.seed_salts = [0] * 64
    manager.rngs = [random.Random(i) for i in range(64)]
    manager._seed_active = True
    manager.seeds[33] = 1234
    manager.seed_counters[33] = 17
    manager.seed_salts[33] = 2
    old_rng_state = manager.rngs[33].getstate()
    generator.sampling_generator.seed_manager = manager

    DeepseekGenerator._apply_sampling_slot_remap(generator, [3, 1, 2, 3])

    assert manager.seeds[0] == 1234
    assert manager.seed_counters[0] == 17
    assert manager.seed_salts[0] == 2
    assert manager.rngs[0].getstate() == old_rng_state
    assert manager.seeds[33] is None
    assert manager.seed_counters[33] == 0
    assert manager.seed_salts[33] == 0


def test_deepseek_host_sampling_remap_is_noop_without_device_sampler():
    generator = SimpleNamespace(sampling_generator=None)

    DeepseekGenerator._apply_sampling_slot_remap(generator, [0])


def test_deepseek_sampling_applies_remap_before_update_and_advance():
    events = []
    sampling_params = SamplingParams(temperature=0.6, top_k=32, top_p=0.95, seed=1234)
    generator = SimpleNamespace(
        _apply_sampling_slot_remap=lambda remap: events.append(("remap", remap)),
        _validate_and_initialize_sampling=lambda *args, **kwargs: events.append(("update", kwargs)),
        _sample_tokens_device=lambda *args, **kwargs: events.append(("advance", kwargs)) or "tokens",
    )

    result = DeepseekGenerator.sample_decode_on_device(
        generator,
        "logits",
        sampling_params=sampling_params,
        slot_remap=[1, 1],
        user_slots=[0],
        reload_sampling_params=True,
        reset_sampling_state=True,
    )

    assert result == "tokens"
    assert [event[0] for event in events] == ["remap", "update", "advance"]


def test_deepseek_host_sampling_applies_dormant_remap_after_success(monkeypatch):
    events = []
    generator = DeepseekV3ForCausalLM.__new__(DeepseekV3ForCausalLM)
    generator.model_run_config_decode = object()
    generator._apply_sampling_slot_remap = lambda remap: events.append(("remap", remap))
    monkeypatch.setattr(
        DeepseekGenerator,
        "decode_forward",
        lambda *args, **kwargs: events.append("decode") or torch.zeros((1, 1, 1, 8)),
    )

    output = generator.decode_forward(
        tokens=torch.zeros((1, 1), dtype=torch.int64),
        start_pos=torch.zeros((1,), dtype=torch.int64),
        read_from_device=False,
        slot_remap=[0],
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )

    assert output.shape == (1, 1, 8)
    assert events == ["decode", ("remap", [0])]


def test_deepseek_host_sampling_does_not_consume_remap_on_failure(monkeypatch, expect_error):
    events = []
    generator = DeepseekV3ForCausalLM.__new__(DeepseekV3ForCausalLM)
    generator.model_run_config_decode = object()
    generator._apply_sampling_slot_remap = lambda remap: events.append(("remap", remap))
    monkeypatch.setattr(
        DeepseekGenerator,
        "decode_forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decode")),
    )

    with expect_error(RuntimeError, "decode"):
        generator.decode_forward(
            tokens=torch.zeros((1, 1), dtype=torch.int64),
            start_pos=torch.zeros((1,), dtype=torch.int64),
            read_from_device=False,
            slot_remap=[0],
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )

    assert events == []


@torch.no_grad()
@pytest.mark.parametrize(
    "sampling_params",
    [
        {"temperature": 0.0, "top_k": 32, "top_p": 0.00, "seed": 42},
        {"temperature": 0.0, "top_k": 32, "top_p": 0.95, "seed": 42},
        {"temperature": 1.0, "top_k": 1, "top_p": 0.00, "seed": 42},  # top-k=1 (always argmax)
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": get_fabric_config()}], indirect=True)
def test_deepseek_device_sampling_argmax_path(mesh_device, ccl, hf_config, device_params, sampling_params):
    vocab_size = int(hf_config.vocab_size)
    args = make_deepseek_sampling_args(mesh_device, vocab_size=vocab_size, pad_logits_to_power_of_2=True)
    batch_size = USERS_PER_ROW * int(mesh_device.shape[0])
    seed = int(sampling_params.get("seed", 0))
    torch.manual_seed(seed)
    torch_input = torch.randn(1, 1, batch_size, args.padded_vocab_size) * 0.01
    forced_tokens = torch.tensor([(u * 1237 + 31) % vocab_size for u in range(batch_size)], dtype=torch.int64)
    batch_indices = torch.arange(batch_size, dtype=torch.int64)
    torch_input[0, 0, batch_indices, forced_tokens] = 50.0
    if args.padded_vocab_size > vocab_size:
        torch_input[:, :, :, vocab_size:] = -float("inf")

    user_params = SamplingParams(
        temperature=[sampling_params["temperature"]] * batch_size,
        top_k=[sampling_params["top_k"]] * batch_size,
        top_p=[sampling_params["top_p"]] * batch_size,
        seed=[seed] * batch_size,
    )
    device_tokens = _sample_device_tokens(mesh_device, ccl, args, torch_input, user_params)

    assert device_tokens.numel() == batch_size, (
        f"Expected {batch_size} sampled tokens, got {device_tokens.numel()}. "
        "This usually indicates incorrect mesh token reconstruction."
    )
    assert torch.equal(
        device_tokens, forced_tokens
    ), "Device sampling generator produced tokens mismatch for LM-head sharded DeepSeek logits."


@torch.no_grad()
@pytest.mark.parametrize("device_params", [{"fabric_config": get_fabric_config()}], indirect=True)
@pytest.mark.parametrize("use_tracing", [False, True], ids=["no_trace", "trace_mode"])
def test_deepseek_device_sampling_stochastic_behavior(mesh_device, ccl, hf_config, device_params, use_tracing):
    vocab_size = int(hf_config.vocab_size)
    args = make_deepseek_sampling_args(mesh_device, vocab_size=vocab_size)
    batch_size = USERS_PER_ROW * int(mesh_device.shape[0])

    torch_input = torch.full((1, 1, batch_size, args.padded_vocab_size), -1e9, dtype=torch.float32)
    candidate_tokens = torch.tensor([3, 7, 11, 19], dtype=torch.int64)
    candidate_logits = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    torch_input[0, 0, :, candidate_tokens] = candidate_logits
    if args.padded_vocab_size > vocab_size:
        torch_input[:, :, :, vocab_size:] = -float("inf")

    num_samples = 100
    per_user_seeds = [1000 + u for u in range(batch_size)]
    user_params = SamplingParams(
        temperature=[1.0] * batch_size,
        top_k=[4] * batch_size,
        top_p=[0.95] * batch_size,
        seed=per_user_seeds,
    )

    tt_input = _make_lm_head_sharded_logits(torch_input, mesh_device)
    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=ccl)
    params = format_sampling_params(user_params, max_batch_size=batch_size)
    sampling.reset_sampling_params(params)
    sampling.reset_prompt_tokens(torch.zeros((USERS_PER_ROW, 1), dtype=torch.int64))
    sampling.reset_output_state(torch.zeros((USERS_PER_ROW, 1), dtype=torch.int64))
    sampling.seed_manager.reset_seed(params.seed, list(range(batch_size)))

    sampled_tokens = []
    try:
        for _ in range(num_samples):
            sampling.seed_manager.get_new_values()
            tt_tokens, tt_log_probs = sampling.sample(tt_input, enable_trace=use_tracing)
            device_tokens = _extract_all_tokens(tt_tokens, mesh_device, USERS_PER_ROW)
            sampled_tokens.append(int(device_tokens[0].item()))
            # In trace mode, sampling reuses captured output tensors across iterations.
            # Deallocating those per-step breaks subsequent trace replays.
            if not use_tracing:
                ttnn.deallocate(tt_tokens)
                if tt_log_probs is not None:
                    ttnn.deallocate(tt_log_probs)
    finally:
        if use_tracing:
            # Release cached trace metadata/tensors before exiting the test.
            sampling.reset_trace()
        ttnn.deallocate(tt_input)

    candidate_set = set(candidate_tokens.tolist())
    sampled_set = set(sampled_tokens)
    assert sampled_set.issubset(
        candidate_set
    ), f"Sampled tokens outside candidate set. got={sorted(sampled_set)}, expected subset of {sorted(candidate_set)}"
    assert len(sampled_set) >= 2, (
        f"Only {len(sampled_set)} unique token(s) in {num_samples} samples; sampling may be stuck. "
        f"sampled_set={sorted(sampled_set)}"
    )
