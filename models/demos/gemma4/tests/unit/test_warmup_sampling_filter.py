# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The decode-warmup sampling sweep drops the request shapes a mesh's sampler opts out of
(2-D fractured weights: no penalty / top-k programs), keeping greedy and None. Pure Python."""

from types import SimpleNamespace

from models.common.sampling.sampling_params import SamplingParams
from models.demos.gemma4.tt.generator_utils import filter_warmup_sampling_params

B = 4
PEN = SamplingParams(
    temperature=[1.0] * B,
    top_k=[10] * B,
    top_p=[0.9] * B,
    presence_penalty=[1.2] * B,
    frequency_penalty=[1.2] * B,
    repetition_penalty=[1.5] * B,
)
PLAIN = SamplingParams(
    temperature=[1.0] * B,
    top_k=[10] * B,
    top_p=[0.9] * B,
    presence_penalty=None,
    frequency_penalty=None,
    repetition_penalty=None,
)
GREEDY = SamplingParams(temperature=[0.0] * B, top_k=[1] * B, top_p=[1.0] * B)
SWEEP = [PEN, PLAIN, GREEDY, None]


def test_full_sampler_keeps_the_whole_sweep():
    assert filter_warmup_sampling_params(SWEEP, SimpleNamespace()) == SWEEP
    assert filter_warmup_sampling_params(SWEEP, None) == SWEEP


def test_no_penalties_drops_only_the_penalty_configs():
    sampler = SimpleNamespace(_allow_penalties_sampling=False, _allow_topk_sampling=True)
    assert filter_warmup_sampling_params(SWEEP, sampler) == [PLAIN, GREEDY, None]


def test_no_topk_keeps_only_greedy_and_none():
    sampler = SimpleNamespace(_allow_penalties_sampling=False, _allow_topk_sampling=False)
    assert filter_warmup_sampling_params(SWEEP, sampler) == [GREEDY, None]
