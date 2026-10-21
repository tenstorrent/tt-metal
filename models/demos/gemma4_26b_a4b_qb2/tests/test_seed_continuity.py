# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise adapter seed continuity with the pinned plugin's count-free calls."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import ttnn
from models.common.sampling.generator import SamplingGenerator as CommonSamplingGenerator
from models.demos.gemma4_26b_a4b_qb2.tt.generator import Gemma4Generator, _Gemma4SamplingGenerator
from models.demos.gemma4_26b_a4b_qb2.tt.generator_vllm import Gemma4ForCausalLM


class SamplingGenerator:
    _reset_sampling_seeds = Gemma4Generator._reset_sampling_seeds

    def __init__(self):
        self.mesh = None
        self.model = SimpleNamespace(layer_indices=[0], max_seq_len=2048)
        self.prefill_trace_enabled = False
        self.prefill_prepared = None
        self.host_sampling = False
        self._release_trace = Mock()
        self.sampler = SimpleNamespace(tt_sampling=SimpleNamespace(seeds_tt_tensor=torch.zeros(32, dtype=torch.int32)))
        self.draws = []

    def _copy(self, source, target, counter):
        target.copy_(source)

    def configure_sampling(self, params, *, seed_offsets, **kwargs):
        self._reset_sampling_seeds(list(params.seed) + [0] * (32 - len(params.seed)), seed_offsets)

    restore_sampling_state = configure_sampling

    def prefill_forward(self, *args, **kwargs):
        return torch.zeros(1, 1)

    def sample_prefill(self, logits):
        self.draws.append(self.sampler.tt_sampling.seeds_tt_tensor.clone())
        return torch.zeros(32, 1, dtype=torch.int64)

    def decode_forward(self, *args, **kwargs):
        self.sampler.tt_sampling.seeds_tt_tensor.add_(1)
        return self.sample_prefill(None)

    def can_reuse_serving_decode(self, *args, **kwargs):
        return False


def params(seeds):
    return SimpleNamespace(seed=list(seeds), temperature=[0.8] * len(seeds), top_k=[20] * len(seeds))


def decode(adapter, positions, seeds, *, reset):
    return adapter.decode_forward(
        torch.ones(len(positions), 1, dtype=torch.int64),
        torch.tensor(positions),
        None,
        object(),
        page_tables_per_layer=[torch.zeros(len(positions), 64, dtype=torch.int32)],
        sampling_params=params(seeds),
        reset_batch=reset,
        read_from_device=False,
    )


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("GEMMA4_EAGER_PREFILL_DECODE_REUSE", "0")
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda value: [value])
    monkeypatch.setattr(ttnn, "to_torch", lambda value: value)
    return Gemma4ForCausalLM(SamplingGenerator(), 4)


def test_sampled_survivor_keeps_seed_after_batch_reorder(adapter):
    decode(adapter, [8, 12], [42, 99], reset=True)
    for _ in range(7):
        # Steady-state async host positions may lag; device RNG must keep advancing.
        decode(adapter, [8, 12], [42, 99], reset=False)
    before = adapter.generator.draws[-1]
    decode(adapter, [20, 16], [99, 42], reset=True)
    after = adapter.generator.draws[-1]
    assert after[:2].tolist() == (before[[1, 0]] + 1).tolist()


def test_sampled_survivor_keeps_seed_when_other_request_finishes(adapter):
    decode(adapter, [8, 12], [42, 99], reset=True)
    before = adapter.generator.draws[-1]
    decode(adapter, [13], [99], reset=True)
    assert adapter.generator.draws[-1][0] == before[1] + 1


def test_layout_only_reset_does_not_restart_seed(adapter):
    decode(adapter, [8], [42], reset=True)
    before = adapter.generator.draws[-1]
    decode(adapter, [9], [42], reset=True)
    assert adapter.generator.draws[-1][0] == before[0] + 1


def test_prefill_and_first_decode_use_adjacent_seeds(adapter):
    adapter.prefill_forward(
        torch.ones(1, 8, dtype=torch.int64),
        None,
        object(),
        [8],
        page_tables_per_layer=[torch.zeros(1, 64, dtype=torch.int32)],
        sampling_params=params([42]),
    )
    before = adapter.generator.draws[-1]
    decode(adapter, [8], [42], reset=True)
    assert adapter.generator.draws[-1][0] == before[0] + 1


@pytest.mark.parametrize("grammar_on", [None, False, True])
@pytest.mark.parametrize("feedback", [False, True])
def test_sampling_wrapper_forwards_common_arguments(monkeypatch, grammar_on, feedback):
    sampler = object.__new__(_Gemma4SamplingGenerator)
    sampler.bind_public_tokens(object(), object())
    sampler.format_public_tokens = Mock()
    sample = Mock(return_value=object())
    monkeypatch.setattr(CommonSamplingGenerator, "_run_sampling", sample)
    logits = object()
    arguments = dict(penalties_on=False, tt_out_tok=sampler._feedback_tokens if feedback else None, count_tokens=False)
    if grammar_on is not None:
        arguments["grammar_on"] = grammar_on

    assert sampler._run_sampling(logits, **arguments) is sample.return_value
    sample.assert_called_once_with(logits, **arguments)
    assert sampler.format_public_tokens.call_count == int(feedback)
