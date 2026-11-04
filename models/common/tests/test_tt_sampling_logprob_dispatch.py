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
    sampler.k_tensor, sampler.p_tensor, sampler.temp_tensor = object(), object(), object()
    calculator = SimpleNamespace(enable_log_probs=False, _use_topk_logprobs=True)
    calculator.set_log_probs_mode = lambda enabled, **kw: setattr(calculator, "enable_log_probs", enabled)
    sampler.log_probs_calculator = calculator
    copied = []
    monkeypatch.setattr(ttnn, "from_torch", lambda value, **kw: value)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda source, dest: copied.append((source, dest)))
    sampler.reset_params(k=[1], p=[1.0], temp=[1.0], enable_log_probs=True)
    assert sampler.force_argmax_sampling is False
    assert len(copied) == 4
    assert copied[0][1] is sampler.k_tensor
    assert torch.equal(copied[0][0], torch.tensor([1]))
