# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the sampling parameters the adapter hands the device sampler.

No device is opened. The adapter accepts stochastic requests with top_k 1..64
(serving_capabilities["max_device_top_k"], so the plugin keeps them on device), but
the device sampler (ttnn.sampling) takes k in (0, 32] only: the shared
format_sampling_params(params, 32) caps k at 32, so a request top_k of 33..64 reaches
the device as 32. Also covers greedy and padding rows and reading the plugin's
TTSamplingParams.
"""

from collections import Counter
from dataclasses import dataclass, fields

import pytest
import torch

from models.common.sampling.generator import SamplingParams, format_sampling_params
from agentic_models.models.k2_horizon_mova_36b_a4b.vllm.tt.generator import MAX_DEVICE_TOP_K, K2HorizonGenerator

TOP_K_ERROR = r"top_k in 1\.\.64"
DEVICE_TOP_K_LIMIT = 32  # ttnn.sampling: k in (0, 32]


def stochastic(top_k, temperature=1.0):
    n = len(top_k)
    return SamplingParams(temperature=[temperature] * n, top_k=list(top_k), top_p=[0.95] * n)


def test_device_path_accepts_top_k_up_to_64():
    # The plugin routes a request to the device sampler when its top_k <= max_device_top_k.
    assert MAX_DEVICE_TOP_K == 64
    assert K2HorizonGenerator.serving_capabilities["max_device_top_k"] == MAX_DEVICE_TOP_K


@pytest.mark.parametrize("k", [1, 8, 32, 33, 40, 63, 64])
def test_validation_accepts_top_k_1_to_64(k):
    K2HorizonGenerator.validate_sampling_params(stochastic([k]))
    K2HorizonGenerator.validate_sampling_params(stochastic([k], temperature=0.6))


@pytest.mark.parametrize("k", [65, 128, 250000, 0, -1])
def test_validation_rejects_stochastic_top_k_outside_1_to_64(k):
    with pytest.raises(ValueError, match=TOP_K_ERROR):  # allow-pytest.raises: no tt-metal conftest
        K2HorizonGenerator.validate_sampling_params(stochastic([k]))


@pytest.mark.parametrize("k", [1, 8, 31, 32])
def test_top_k_up_to_32_reaches_device_unchanged(k):
    formatted = format_sampling_params(stochastic([k]), 32)
    assert formatted.top_k[0] == k
    assert formatted.top_k[1:] == [1] * 31


@pytest.mark.parametrize("k", [33, 40, 48, 64])
def test_top_k_33_to_64_reaches_device_as_32(k):
    # GPQA's lm-eval task asks for top_k 40: accepted by the adapter, sampled with k=32.
    params = stochastic([k])
    K2HorizonGenerator.validate_sampling_params(params)
    formatted = format_sampling_params(params, 32)
    assert formatted.top_k[0] == DEVICE_TOP_K_LIMIT
    assert formatted.top_k[1:] == [1] * 31


@pytest.mark.parametrize("k", [1, 40, 64, 0, -1, 250000])
def test_greedy_rows_keep_greedy_encoding(k):
    params = SamplingParams(temperature=[0.0], top_k=[k], top_p=[1.0])
    K2HorizonGenerator.validate_sampling_params(params)
    formatted = format_sampling_params(params, 32)
    assert formatted.top_k[0] == 1
    assert formatted.top_p[0] == 0.0 and formatted.temperature[0] == 1.0


def test_mixed_batch_and_padding_rows():
    params = SamplingParams(
        temperature=[0.0, 0.7, 1.0, 0.6, 1.0, 0.0],
        top_k=[40, 40, 64, 8, 33, 250000],
        top_p=[1.0, 0.95, 0.95, 0.8, 1.0, 1.0],
        seed=[1, 2, 3, 4, 5, 6],
    )
    K2HorizonGenerator.validate_sampling_params(params)
    formatted = format_sampling_params(params, 32)
    assert len(formatted.top_k) == 32
    assert formatted.top_k[:6] == [1, 32, 32, 8, 32, 1]
    assert formatted.top_k[6:] == [1] * 26
    assert formatted.top_p[:6] == [0.0, 0.95, 0.95, 0.8, 1.0, 0.0]
    assert formatted.seed[:6] == [1, 2, 3, 4, 5, 6]


class RecordingSampler:
    """Stands in for SamplingGenerator: records the top_k each upload sends to the device."""

    def __init__(self):
        self.uploads = []

    def apply_decode_state(self, chunks, *, reload_sampling_params, reset_sampling_state, **kwargs):
        if reload_sampling_params:
            # The shared implementation formats the raw request itself.
            self.reset_sampling_params(format_sampling_params(chunks[0], 32))

    def reset_sampling_params(self, formatted):
        self.uploads.append(list(formatted.top_k))


@pytest.mark.parametrize(
    "top_k, device_top_k",
    [([40, 64], [32, 32]), ([16, 32], [16, 32]), ([33, 8], [32, 8])],
    ids=["above-32", "within-32", "mixed"],
)
def test_decode_reload_uploads_top_k_capped_at_32(top_k, device_top_k):
    generator = K2HorizonGenerator.__new__(K2HorizonGenerator)
    generator.sampler = RecordingSampler()
    generator.counters = Counter()
    params = SamplingParams(temperature=[1.0, 0.7], top_k=top_k, top_p=[0.95, 0.95])
    generator.update_sampling_state(params, start_pos=None, reload_params=True, reset_state=False)
    # One upload per reload; the generator does not upload a second, wider k.
    assert len(generator.sampler.uploads) == 1
    assert generator.sampler.uploads[-1][:2] == device_top_k
    assert generator.formatted_sampling_params.top_k[:2] == device_top_k
    # A decode step without a reload uploads nothing.
    generator.update_sampling_state(params, start_pos=None, reload_params=False, reset_state=False)
    assert len(generator.sampler.uploads) == 1


class StopAfterUpload(Exception):
    """Raised by the set_active_slots stub so prefill stops before the device seed code."""


def test_prefill_uploads_top_k_capped_at_32_into_slots():
    generator = K2HorizonGenerator.__new__(K2HorizonGenerator)
    generator.sampler = RecordingSampler()
    generator.formatted_sampling_params = format_sampling_params(SamplingParams(0.0, 1, 1.0), 32)

    def stop(slots):
        raise StopAfterUpload(slots)

    generator.set_active_slots = stop
    params = SamplingParams(temperature=[1.0, 0.7, 0.9], top_k=[40, 64, 16], top_p=[0.95, 0.95, 0.95])
    with pytest.raises(StopAfterUpload):  # allow-pytest.raises: no tt-metal conftest
        generator.prepare_prefill_sampling(torch.zeros(3, 8, dtype=torch.long), [8, 8, 8], [3, 5, 9], params)
    assert len(generator.sampler.uploads) == 1
    upload = generator.sampler.uploads[-1]
    assert upload[3] == 32 and upload[5] == 32 and upload[9] == 16
    # Slots without a new request keep their greedy encoding.
    assert [k for slot, k in enumerate(upload) if slot not in (3, 5, 9)] == [1] * 29
    assert generator.formatted_sampling_params.top_k == upload


def test_frozen_plugin_style_params_round_trip():
    # Frozen, and without log_probs_enable_temperature, like the plugin's TTSamplingParams.
    @dataclass(frozen=True)
    class PluginParams:
        temperature: list
        top_k: list
        top_p: list
        presence_penalty: float = 0.0
        frequency_penalty: float = 0.0
        repetition_penalty: float = 1.0
        seed: int = 0
        num_logprobs: int = 0
        enable_log_probs: bool = False

    formatted = format_sampling_params(PluginParams(temperature=[1.0, 1.0], top_k=[40, 8], top_p=[0.95, 0.95]), 32)
    assert isinstance(formatted, PluginParams)
    assert formatted.top_k[:2] == [32, 8]


def test_adapter_fills_fields_missing_from_plugin_params():
    model_input = pytest.importorskip("vllm_tt_plugin.model_input")
    from agentic_models.models.k2_horizon_mova_36b_a4b.vllm.tt.generator_vllm import K2HorizonForCausalLM, _plugin_field

    adapter = K2HorizonForCausalLM()
    adapter.host_compat = False
    params = model_input.TTSamplingParams(
        temperature=[1.0, 0.0],
        top_k=[40, 1],
        top_p=[0.95, 1.0],
        seed=[3, 4],
        enable_log_probs=[True, False],
        num_logprobs=[5, -1],
    )
    converted = adapter._sampling(params)
    assert isinstance(converted, SamplingParams)
    # The adapter keeps the request top_k; the shared formatter caps it for the device.
    assert converted.top_k == [40, 1] and converted.seed == [3, 4]
    assert converted.num_logprobs == [5, 0]
    # A field newer than the plugin's dataclass takes the common default.
    plugin_fields = {f.name for f in fields(model_input.TTSamplingParams)}
    for field in fields(SamplingParams):
        if field.name not in plugin_fields:
            assert getattr(converted, field.name) == field.default, field.name
    assert format_sampling_params(converted, 32).top_k[:2] == [32, 1]

    # A missing required field still fails loudly.
    @dataclass
    class NoTopK:
        temperature: float = 1.0

    top_k = next(f for f in fields(SamplingParams) if f.name == "top_k")
    with pytest.raises(AttributeError):  # allow-pytest.raises: no tt-metal conftest
        _plugin_field(NoTopK(), top_k)
