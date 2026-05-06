# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from models.common.sampling.generator import SamplingParams, format_sampling_params
from models.common.sampling.tt_sampling import TTSampling


def test_temperature_zero_normalizes_to_force_argmax_params():
    params = format_sampling_params(
        SamplingParams(temperature=0.0, top_k=32, top_p=0.08, seed=0),
        max_batch_size=32,
    )

    assert params.temperature == [1.0] * 32
    assert params.top_k == [1] * 32
    assert params.top_p == [1.0] * 32
    assert params.seed == [0] * 32


def test_normalized_greedy_params_enable_ttsampling_force_argmax():
    params = format_sampling_params(
        SamplingParams(temperature=0.0, top_k=32, top_p=0.08, seed=0),
        max_batch_size=32,
    )
    sampling = object.__new__(TTSampling)
    sampling._allow_force_argmax_sampling = True

    assert sampling._is_force_argmax_sampling(params.top_k, params.top_p, params.temperature)
