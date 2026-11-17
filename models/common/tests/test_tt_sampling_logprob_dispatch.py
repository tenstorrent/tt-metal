# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for greedy token selection with logprob reporting."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.sampling.tt_sampling import TTSampling


@pytest.mark.parametrize("enabled", [False, True])
def test_argmax_logprobs_use_original_shards_and_preserve_tokens(monkeypatch, enabled):
    sampler = object.__new__(TTSampling)
    sampler._force_argmax_sampling = True
    sampler._force_argmax_sub_core_grids = None
    sampler.mesh_device = SimpleNamespace(get_num_devices=lambda: 8)
    sampler.tt_ccl = SimpleNamespace(
        get_and_cycle_barrier_semaphore_handle=lambda axis: None,
        get_and_cycle_ag_semaphore_handles=lambda axis: None,
    )
    sampler.argmax_chunks_per_sync = 10
    sampler.argmax_num_workers_per_link = 1
    sampler._get_sampling_cluster_axis = lambda: 1
    sampler._get_force_argmax_all_gather_config = lambda axis: (1, ttnn.Topology.Linear)
    sampler._can_slice_valid_vocab_for_argmax = lambda: False
    sampler._mask_invalid_vocab_logits = lambda value: value
    sampler._untilize_for_argmax = lambda value: value
    shards = SimpleNamespace(memory_config=lambda: ttnn.DRAM_MEMORY_CONFIG)
    gathered, probabilities = object(), object()
    tokens = SimpleNamespace(shape=(1, 1, 32))
    canonical_tokens = object()
    calls = []

    def calculate(logits, indices):
        assert logits is shards, "Logprob normalization must not count replicated full vocabularies"
        assert indices is canonical_tokens
        calls.append(True)
        return probabilities

    def argmax(value, **kwargs):
        assert value is gathered
        return tokens

    def reshape(value, shape):
        assert value is tokens and shape == (1, 1, 1, 32)
        return canonical_tokens

    sampler.log_probs_calculator = SimpleNamespace(enable_log_probs=enabled, calculate_log_probs=calculate)
    monkeypatch.setattr(ttnn.experimental, "all_gather_async", lambda *a, **kw: gathered)
    monkeypatch.setattr(ttnn, "argmax", argmax)
    monkeypatch.setattr(ttnn, "reshape", reshape)
    result_tokens, result_probs = sampler.forward(shards)
    assert result_tokens is tokens
    assert result_probs is (probabilities if enabled else None)
    assert len(calls) == int(enabled)


@pytest.mark.parametrize("enabled,use_topk,expected", [(False, True, True), (True, False, True), (True, True, False)])
def test_topk_reporting_requires_candidate_pipeline(enabled, use_topk, expected):
    sampler = object.__new__(TTSampling)
    sampler._allow_force_argmax_sampling = True
    sampler.log_probs_calculator = SimpleNamespace(enable_log_probs=enabled, _use_topk_logprobs=use_topk)
    assert sampler._is_force_argmax_sampling([1], [1.0], [1.0]) is expected


def test_enabling_topk_reporting_refreshes_candidate_parameters(monkeypatch):
    sampler = object.__new__(TTSampling)
    sampler._allow_force_argmax_sampling = True
    sampler.max_top_k = 32
    sampler._sampling_dp = 1
    sampler._greedy_col = object()
    sampler._scale_logprobs = object()
    sampler.k_tensor, sampler.p_tensor, sampler.temp_tensor = object(), object(), object()
    calculator = SimpleNamespace(enable_log_probs=False, _use_topk_logprobs=True)
    calculator.set_log_probs_mode = lambda enabled, **kw: setattr(calculator, "enable_log_probs", enabled)
    sampler.log_probs_calculator = calculator
    copied = []
    monkeypatch.setattr(ttnn, "from_torch", lambda value, **kw: value)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda source, dest: copied.append((source, dest)))
    sampler.reset_params(k=[1], p=[1.0], temp=[1.0], enable_log_probs=True)
    assert sampler.force_argmax_sampling is False
    assert len(copied) == 5
    assert copied[0][1] is sampler.k_tensor
    assert torch.equal(copied[0][0], torch.tensor([1]))
    assert copied[-1][1] is sampler._scale_logprobs


def test_scalar_log_probs_enable_temperature_broadcasts_scale_tensor(monkeypatch):
    sampler = object.__new__(TTSampling)
    sampler._allow_force_argmax_sampling = False
    sampler._is_force_argmax_sampling = lambda k, p, temp: False
    sampler._normalize_device_params = lambda k, temp: (torch.as_tensor(k), torch.as_tensor(temp))
    sampler._sampling_dp = 1
    sampler.mesh_device = SimpleNamespace(get_num_devices=lambda: 1)
    sampler.cluster_shape = (1, 1)
    sampler._param_dims = (None, None)
    sampler._greedy_col_dims = lambda: (None, None)
    sampler.k_tensor = object()
    sampler.p_tensor = object()
    sampler.temp_tensor = object()
    sampler._greedy_col = object()
    sampler._scale_logprobs = object()
    sampler.log_probs_calculator = SimpleNamespace(
        _use_topk_logprobs=False,
        set_log_probs_mode=lambda *args, **kwargs: None,
    )

    copied_tensors = []
    monkeypatch.setattr(ttnn, "from_torch", lambda value, **kwargs: torch.as_tensor(value))
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, device: copied_tensors.append((host, device)))

    temp = [0.7] * 5
    sampler.reset_params(k=[1] * 5, p=[0.0] * 5, temp=temp, log_probs_enable_temperature=True)

    # copy order: k, p, temp, greedy_col, scale_logprobs
    assert len(copied_tensors) == 5
    scale_host, scale_device = copied_tensors[-1]
    assert scale_device is sampler._scale_logprobs
    torch.testing.assert_close(scale_host, torch.tensor(temp).reshape(1, 1, -1, 1).to(torch.bfloat16), rtol=0, atol=0)
