# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


from types import SimpleNamespace

import torch

from models.common.llm_runtime.tensor_resources import best_effort_deallocate_owned_tensors
from models.common.modules.sampling import sampling_1d
from models.common.modules.sampling.sampling_1d import Sampling1D


class _ReusingCalculator:
    enable_log_probs = True

    def __init__(self):
        self.output_tensor = torch.zeros(1, 1, 1, 32)

    def calculate_log_probs(self, logits, tokens):
        self.output_tensor.fill_(float(logits))
        return self.output_tensor


def _sampler(monkeypatch, calculator, deallocated):
    sampler = object.__new__(Sampling1D)
    sampler.config = SimpleNamespace(sub_core_grids=None, sampling_memory_config=None)
    sampler._mask_invalid_vocab_logits = lambda value: value
    sampler._topk = lambda value: (object(), object())
    sampler._prepare_topk_memory = lambda values, indices: (values, indices)
    sampler._slice_user_rows = lambda tensor, rows, memory_config: (tensor, False)
    sampler._index_offsets, sampler._seeds, sampler._user_ids = object(), object(), object()
    sampler._sampling_sub_core_grids = None
    sampler._log_probs_calculator = calculator
    shape = SimpleNamespace(shape=(1, 1, 32, 32))
    monkeypatch.setattr(sampling_1d.ttnn, "typecast", lambda value, **kw: value if isinstance(value, float) else shape)
    monkeypatch.setattr(sampling_1d.ttnn, "add", lambda *a, **kw: object())
    monkeypatch.setattr(sampling_1d.ttnn, "deallocate", lambda value: deallocated.append(value))
    monkeypatch.setattr(sampling_1d.ttnn, "to_memory_config", lambda value, *a, **kw: value)
    monkeypatch.setattr(sampling_1d.ttnn, "untilize", lambda value, **kw: value)
    monkeypatch.setattr(sampling_1d.ttnn, "manual_seed", lambda **kw: None)
    monkeypatch.setattr(sampling_1d.ttnn, "sampling", lambda *a, **kw: object())
    monkeypatch.setattr(sampling_1d.ttnn, "clone", lambda value, **kw: value.clone(), raising=False)
    return sampler


def _sample(sampler, logits):
    return sampler._sample_topk(logits, object(), object(), object(), None, None)[1]


def test_an_earlier_request_keeps_its_logprobs_after_a_later_request_runs(monkeypatch):
    sampler = _sampler(monkeypatch, _ReusingCalculator(), [])
    first = _sample(sampler, -1.0)
    second = _sample(sampler, -2.0)
    assert torch.all(first == -1.0), "request 1 read back request 2's logprobs"
    assert torch.all(second == -2.0)


def test_releasing_a_result_leaves_the_calculator_output_alive(monkeypatch):
    calculator, deallocated = _ReusingCalculator(), []
    sampler = _sampler(monkeypatch, calculator, deallocated)
    monkeypatch.setattr(
        "models.common.llm_runtime.tensor_resources.ttnn",
        SimpleNamespace(Tensor=torch.Tensor, deallocate=deallocated.append),
    )
    result = _sample(sampler, -1.0)
    assert not best_effort_deallocate_owned_tensors((object(), result))
    assert any(value is result for value in deallocated)
    assert not any(value is calculator.output_tensor for value in deallocated)
