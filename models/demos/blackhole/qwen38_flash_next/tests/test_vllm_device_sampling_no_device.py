# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The vLLM adapter's device-sampling policy mapping and draw stream (host logic, no device)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests import test_vllm_async_decode_no_device as async_tests
from models.demos.blackhole.qwen38_flash_next.tests.test_qwen38_vllm_no_device import (  # noqa: F401
    ALLOCATED_CONTEXT,
    PAGE_TABLE,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import (
    Qwen38AsyncStep,
    Qwen38DeviceSamplingState,
    Qwen38ForCausalLM,
    Qwen38HostSamplingPolicy,
    Qwen38VLLMSlot,
    device_policy_fingerprint,
    device_policy_of_contract,
    greedy_batch_sampling_params,
    slot_draw_stream,
)


def host(**overrides) -> Qwen38HostSamplingPolicy:
    values = dict(temperature=1.0, top_k=20, top_p=0.95)
    values.update(overrides)
    return Qwen38HostSamplingPolicy(**values)


def test_checkpoint_default_is_accepted():
    policy, reason = device_policy_of_contract(host())
    assert reason == ""
    assert (policy.temperature, policy.top_k, policy.top_p, policy.min_p) == (1.0, 20, 0.95, 0.0)
    assert not policy.greedy and policy.presence_penalty == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        dict(temperature=4.0),
        dict(temperature=0.01),
        dict(top_k=1),
        dict(top_k=32),
        dict(top_p=1.0),
        dict(top_p=0.01),
        dict(seed=7),
    ],
)
def test_accepted(overrides):
    policy, reason = device_policy_of_contract(host(**overrides))
    assert policy is not None and reason == ""


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (dict(temperature=0.0), "greedy"),
        (dict(temperature=4.5), "temperature"),
        (dict(top_k=0), "top_k"),
        (dict(top_k=-1), "top_k"),
        (dict(top_k=33), "top_k"),
        (dict(top_k=248320), "top_k"),
        (dict(top_p=0.0), "top_p"),
        (dict(top_p=1.5), "top_p"),
        (dict(frequency_penalty=0.5), "frequency"),
        (dict(repetition_penalty=1.1), "repetition"),
        (dict(presence_penalty=0.5), "presence"),
        (dict(logprobs=True), "logprobs"),
    ],
)
def test_refused(overrides, reason):
    policy, why = device_policy_of_contract(host(**overrides))
    assert policy is None and why == reason


def test_stream_same_seed_same_draws():
    a, b = slot_draw_stream(1234), slot_draw_stream(1234)
    draws = [a.next_uniform() for _ in range(8)]
    assert draws == [b.next_uniform() for _ in range(8)]
    assert all(0.0 <= u < 1.0 for u in draws) and len(set(draws)) == 8
    assert a.draws == 8
    assert [slot_draw_stream(1235).next_uniform()] != draws[:1]


def test_stream_without_seed_is_fresh():
    seeds = {slot_draw_stream(None).seed for _ in range(8)}
    assert len(seeds) == 8
    assert all(0 <= s < 2**64 for s in seeds)


def test_fingerprint_changes_on_any_field():
    base, _ = device_policy_of_contract(host())
    assert device_policy_fingerprint(base) == device_policy_fingerprint(device_policy_of_contract(host())[0])
    variants = [
        dataclasses.replace(base, temperature=0.7),
        dataclasses.replace(base, top_k=21),
        dataclasses.replace(base, top_p=0.9),
        dataclasses.replace(base, min_p=0.05),
        dataclasses.replace(base, presence_penalty=0.5),
        dataclasses.replace(base, greedy=True),
    ]
    prints = {device_policy_fingerprint(v) for v in variants} | {device_policy_fingerprint(base)}
    assert len(prints) == len(variants) + 1


def test_slot_state_defaults_are_independent():
    a, b = Qwen38VLLMSlot(), Qwen38VLLMSlot()
    assert a.device_sampling == Qwen38DeviceSamplingState()
    assert a.device_sampling is not b.device_sampling


# -- the routing of a request over the chain's greedy set, sampler set and the host sampler ------------------------------

ORDER = async_tests.ORDER


class SamplerChain(async_tests.AsyncChain):
    """The async fake chain with the device sampler's surface: ``ttsampler.set_slot_params``, ``prepare_sampled_step``,
    and a TAIL / HEAD that log which set they replay."""

    def __init__(self) -> None:
        super().__init__()
        self.ttsampler = SimpleNamespace(set_slot_params=lambda *args: ORDER.append(f"set:{args}"), seeds_active=True)

    def prepare_sampled_step(self) -> None:
        ORDER.append("prepare")

    def execute_head(self, residue: int, sampled: bool = False) -> None:
        super().execute_head(residue)
        if sampled:
            ORDER[-1] = f"head:{residue}:s"

    def execute_tail(self, residue: int, sampled: bool = False) -> None:
        super().execute_tail(residue)
        if sampled:
            ORDER[-1] = f"tail:{residue}:s"


def params(temperature=1.0, top_k=20, top_p=0.95, seed=None):
    return SimpleNamespace(
        temperature=[temperature],
        top_k=[top_k],
        top_p=[top_p],
        presence_penalty=[0.0],
        frequency_penalty=[0.0],
        repetition_penalty=[1.0],
        seed=[seed],
        num_logprobs=[0],
        enable_log_probs=[False],
    )


class Driver:
    def __init__(self, monkeypatch, *, device_sampler=True, async_decode=True) -> None:
        monkeypatch.setattr(vllm_module, "DEVICE_SAMPLER", device_sampler)
        monkeypatch.setattr(vllm_module, "ASYNC_DECODE", async_decode)
        ORDER.clear()
        self.chain = SamplerChain()
        self.model = Qwen38ForCausalLM(self.chain, allocated_context=ALLOCATED_CONTEXT)
        self.ids = list(range(1000, 1010))
        logits, _ = self.model.prefill_forward(
            tokens=torch.tensor([self.ids], dtype=torch.int32),
            page_table=PAGE_TABLE,
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[len(self.ids)],
            start_pos=torch.zeros(1, dtype=torch.int32),
            empty_slots=[0],
            sampling_params=greedy_batch_sampling_params(1),
        )
        self.token = int(logits[0, 0])

    def call(self, sampling, *, reload=True, **extra):
        position = len(self.model.slot.committed)
        return self.model.decode_forward(
            tokens=torch.tensor([[self.token]], dtype=torch.int32),
            start_pos=torch.tensor([position], dtype=torch.int32),
            page_table=PAGE_TABLE,
            kv_cache=None,
            enable_trace=True,
            read_from_device=False,
            sampling_params=sampling,
            reload_inputs=reload,
            **extra,
        )

    def sets(self) -> list[str]:
        return [op for op in ORDER if op.startswith("set:")]

    def check_prepared(self) -> int:
        """Every sampler-set TAIL follows a ``prepare``; returns how many ran."""

        tails = [i for i, op in enumerate(ORDER) if op.endswith(":s") and op.startswith("tail")]
        for i in tails:
            before = [op for op in ORDER[:i] if op.startswith("tail") or op == "prepare"]
            assert before and before[-1] == "prepare", ORDER[: i + 1]
        assert ORDER.count("prepare") == len(tails)
        return len(tails)


@pytest.fixture(autouse=True)
def _fake_ttnn(monkeypatch):
    ORDER.clear()
    ttnn = vllm_module.ttnn
    monkeypatch.setattr(ttnn, "event_synchronize", lambda event: ORDER.append(f"wait:{event.name}"), raising=False)
    monkeypatch.setattr(ttnn, "to_torch", lambda host: host, raising=False)


def test_default_policy_lean_steps_use_the_sampler_set(monkeypatch):
    d = Driver(monkeypatch, async_decode=False)
    for _ in range(3):
        out = d.call(params())
        assert isinstance(out, torch.Tensor) and tuple(out.shape) == (1, 1)
        assert not [op for op in ORDER if op.startswith("full_gather")]
    assert d.sets() == [f"set:(0, 1.0, 20, 0.95, {d.model.slot.device_sampling.stream.seed})"]
    assert d.model.slot.device_sampling.sampled
    assert d.check_prepared() == 3
    after = ORDER[ORDER.index(d.sets()[0]) :]
    tails = [op for op in after if op.startswith("tail:")]
    assert len(tails) == 3 and all(op.endswith(":s") for op in tails)


def test_default_policy_steady_steps_run_async_on_the_sampler_set(monkeypatch):
    d = Driver(monkeypatch)
    d.call(params())  # the first (lean, blocking) step admits the policy
    handles = [d.call(params(), reload=False) for _ in range(3)]
    assert all(isinstance(h, Qwen38AsyncStep) for h in handles)
    assert len(d.sets()) == 1
    assert d.check_prepared() == 4
    heads = [op for op in ORDER[ORDER.index(d.sets()[0]) :] if op.startswith("head:")]
    assert len(heads) == 4
    assert all(op.endswith(":s") for op in heads)


def test_greedy_keeps_the_greedy_set(monkeypatch):
    d = Driver(monkeypatch)
    d.call(params(0.0, 1, 0.0))
    handle = d.call(params(0.0, 1, 0.0), reload=False)
    assert isinstance(handle, Qwen38AsyncStep)
    assert not d.sets() and "prepare" not in ORDER
    assert not any(op.endswith(":s") for op in ORDER)
    assert not d.model.slot.device_sampling.sampled


def test_host_fallback_keeps_the_blocking_full_row_path(monkeypatch):
    d = Driver(monkeypatch)
    out = d.call(params(top_k=-1))  # top_k off: the device table does not cover it
    assert isinstance(out, torch.Tensor)
    out = d.call(params(top_k=-1), reload=False)
    assert isinstance(out, torch.Tensor) and not isinstance(out, Qwen38AsyncStep)
    assert not d.sets() and "prepare" not in ORDER and not any(op.endswith(":s") for op in ORDER)
    assert [op for op in ORDER if op.startswith("full_gather")]
    assert not d.model.slot.device_sampling.sampled


def test_slot_params_are_written_on_admission_and_changes_only(monkeypatch):
    d = Driver(monkeypatch, async_decode=False)
    for _ in range(3):
        d.call(params(seed=5))
    assert d.sets() == ["set:(0, 1.0, 20, 0.95, 5)"]
    d.call(params(0.7, 20, 0.95, seed=5))  # policy change
    d.call(params(0.7, 20, 0.95, seed=5))
    assert len(d.sets()) == 2 and d.sets()[1] == "set:(0, 0.7, 20, 0.95, 5)"
    d.call(params(0.7, 20, 0.95, seed=6))  # seed change
    assert len(d.sets()) == 3
    d.call(params(0.7, 20, 0.95, seed=6), reset_sampling_state=True)  # the plugin's reset
    assert len(d.sets()) == 4
    d.call(params(0.0, 1, 0.0))  # to greedy: no write, the greedy set
    assert len(d.sets()) == 4 and not d.model.slot.device_sampling.sampled
    d.check_prepared()
    assert [op for op in ORDER if op.startswith("tail:")][-1].count(":") == 1  # the greedy step's TAIL


def test_a_policy_change_between_steady_steps_goes_through_the_blocking_step(monkeypatch):
    d = Driver(monkeypatch)
    d.call(params(seed=1))
    assert isinstance(d.call(params(seed=1), reload=False), Qwen38AsyncStep)
    out = d.call(params(0.7, seed=1), reload=False)  # a new policy: quiesce, write, blocking sampled step
    assert isinstance(out, torch.Tensor)
    assert len(d.sets()) == 2
    assert isinstance(d.call(params(0.7, seed=1), reload=False), Qwen38AsyncStep)
    d.check_prepared()


def test_a_new_request_writes_the_params_again(monkeypatch):
    d = Driver(monkeypatch, async_decode=False)
    d.call(params(seed=3))
    d.model.release_request(0)
    d.model.prefill_forward(
        tokens=torch.tensor([d.ids], dtype=torch.int32),
        page_table=PAGE_TABLE,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[len(d.ids)],
        start_pos=torch.zeros(1, dtype=torch.int32),
        empty_slots=[0],
        sampling_params=greedy_batch_sampling_params(1),
    )
    d.call(params(seed=3))
    assert len(d.sets()) == 2


@pytest.mark.parametrize("async_decode", (False, True))
def test_knob_off_never_touches_the_sampler(monkeypatch, async_decode):
    d = Driver(monkeypatch, device_sampler=False, async_decode=async_decode)
    for _ in range(2):
        d.call(params())
    if async_decode:
        assert not isinstance(d.call(params(), reload=False), Qwen38AsyncStep)  # sampled: the blocking path
    assert not d.sets() and "prepare" not in ORDER and not any(op.endswith(":s") for op in ORDER)
    assert not d.model.slot.device_sampling.sampled
    assert [op for op in ORDER if op.startswith("full_gather")]
