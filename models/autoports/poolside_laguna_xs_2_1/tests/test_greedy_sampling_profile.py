# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hardware-independent guards for Laguna's qualified generic greedy path.

The compact distributed-argmax experiment is intentionally opt-in: P150x2
qualification found it correct but slower than generic k=1 at both B=1 and B=32.
These tests prevent it from silently returning to Laguna's standalone or served
hot paths.
"""

from types import SimpleNamespace

import torch

from models.autoports.poolside_laguna_xs_2_1.tt import generator_vllm as gv
from models.autoports.poolside_laguna_xs_2_1.tt.generator import LagunaGenerator
from models.common.modules.sampling.sampling_1d import Sampling1DConfig


class _FakeSampler:
    def __init__(self):
        self.calls = []

    def decode_forward(self, _shards, **kwargs):
        self.calls.append(kwargs)


def test_compact_argmax_is_default_off():
    assert Sampling1DConfig(vocab_size=100_352).use_compact_argmax is False


def test_standalone_greedy_keeps_generic_topk1_params():
    sampler = _FakeSampler()
    generator = object.__new__(LagunaGenerator)
    generator._sampler = lambda _batch: sampler
    k, p, temp = object(), object(), object()
    generator._greedy_params = {1: (k, p, temp)}
    output = object()

    assert generator._greedy_sample(object(), 1, output) is output
    assert sampler.calls == [{"k": k, "p": p, "temp": temp, "tt_out_tok": output}]


def _capture_bridge(monkeypatch):
    sampler = _FakeSampler()
    marker = object()
    gen = SimpleNamespace(
        _rep=lambda *_args, **_kwargs: marker,
        _sampler=lambda _batch: sampler,
    )
    model = SimpleNamespace(
        embed_decode=lambda _tok: marker,
        decode_layers=lambda *_args, **_kwargs: marker,
        lm_head_shards_decode=lambda _hidden: marker,
    )
    bridge = object.__new__(gv.LagunaForCausalLM)
    bridge._decode = {}
    bridge.gen = gen
    bridge.model = model
    bridge.mesh_device = marker

    monkeypatch.setattr(gv.ttnn, "reshape", lambda tensor, _shape: tensor)
    monkeypatch.setattr(gv.ttnn, "plus_one", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gv.ttnn, "synchronize_device", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gv.ttnn, "begin_trace_capture", lambda *_args, **_kwargs: 7)
    monkeypatch.setattr(gv.ttnn, "end_trace_capture", lambda *_args, **_kwargs: None)

    state = bridge._decode_state(1, [], marker)
    return sampler, state


def test_served_decode_trace_keeps_generic_sampling(monkeypatch):
    sampler, state = _capture_bridge(monkeypatch)

    assert len(sampler.calls) == 2  # compile + capture
    assert all(call["k"] is not None and call["p"] is not None and call["temp"] is not None for call in sampler.calls)
    assert all(call["seeds"] is not None for call in sampler.calls)
    assert set(state) >= {"k", "p", "t", "seeds", "last_sp_key"}


def test_greedy_sampling_buffers_still_encode_topk1():
    bridge = object.__new__(gv.LagunaForCausalLM)
    params = SimpleNamespace(temperature=[0.0], top_k=[0], top_p=[1.0], seed=[None])

    k, p, temp, seeds = bridge._sampling_buffers_from_params(params, 1)

    assert torch.equal(k, torch.tensor([1], dtype=torch.int32))
    assert torch.equal(p, torch.tensor([1.0], dtype=torch.float32))
    assert torch.equal(temp, torch.tensor([1.0], dtype=torch.float32))
    assert torch.equal(seeds, torch.tensor([0], dtype=torch.int32))
