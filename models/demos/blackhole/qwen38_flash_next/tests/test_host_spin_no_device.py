# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_BATCH_SPIN on a fake ttnn: no device.

Flag off: the lane step replays with the blocking execute and makes no other ttnn call.  Flag on: a non-blocking
execute, a non-blocking token-row read into a NaN host tensor, a poll that exits when the fake device completes the
read, then the full-mesh event synchronize, then the same post-wait calls as before.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy
import pytest

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_batch_decode as batch
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_host_spin as spin
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm_module


class FakeHost:
    """A host tensor that turns finite after ``complete_after`` polls (0: already finished)."""

    def __init__(self, calls, complete_after=3):
        self.calls, self.left, self.polls = calls, complete_after, 0

    def to_numpy(self):
        self.polls += 1
        self.calls.append(("poll",))
        self.left -= 1
        return numpy.array([7.0 if self.left < 0 else float("nan")])


class FakeTTNN:
    float32 = "f32"
    TILE_LAYOUT = "tile"

    def __init__(self, complete_after=3):
        self.calls, self.complete_after, self.hosts = [], complete_after, []

    def from_torch(self, tensor, dtype=None, layout=None):
        self.calls.append(("from_torch", tuple(tensor.shape), dtype, layout))
        host = FakeHost(self.calls, self.complete_after)
        self.hosts.append(host)
        return host

    def copy_device_to_host_tensor(self, shard, host, blocking=True):
        self.calls.append(("copy", shard, blocking))

    def from_device(self, shard, blocking=True):
        self.calls.append(("from_device", shard, blocking))
        return "host"

    def get_device_tensors(self, tensor):
        return [("shard", tensor)]

    def record_event(self, mesh, cq_id=None):
        self.calls.append(("record_event", mesh, cq_id))
        return "event"

    def event_synchronize(self, event):
        self.calls.append(("event_synchronize", event))

    def _ttnn_execute_trace(self, mesh, trace_id, cq_id=0, blocking=True):
        self.calls.append(("execute", mesh, trace_id, cq_id, blocking))


@pytest.fixture
def fake(monkeypatch):
    ttnn = FakeTTNN()
    monkeypatch.setattr(spin, "ttnn", ttnn)
    monkeypatch.setattr(batch, "ttnn", ttnn)
    monkeypatch.setattr(vllm_module, "ttnn", ttnn)
    monkeypatch.setattr(batch, "STEP_SPLIT", True)
    monkeypatch.setattr(batch, "LANE_PLE_FAST", True)
    return ttnn


def _decoder(calls):
    bucket = SimpleNamespace(trace_id=11, ls="bucket_ls")
    model = SimpleNamespace(
        write_lane_ple_rows=lambda ls, tokens, contexts, **kw: calls.append(("ple",)) or list(contexts),
        write_lane_tokens=lambda ls, tokens, **kw: calls.append(("tok",)),
        mark_lane_ple_fenced=lambda: calls.append(("fenced",)),
        prove_lane_ple_reads=lambda proofs: calls.append(("proved",)),
    )
    return SimpleNamespace(
        model=model,
        ls=SimpleNamespace(token_row="row", position=SimpleNamespace(advance_replayed=lambda: calls.append(("adv",)))),
        mesh="mesh",
        lanes=4,
        positions=[0, 0, 0, 0],
        contexts=[None] * 4,
        bucket_sets={4: bucket},
        bucket_steps={4: 0},
        bucket_for_step=lambda: 4,
        active=None,
        step_split={},
    )


def _enqueue(decoder, **kwargs):
    return batch.Qwen38PlainLaneDecoder.enqueue_step(decoder, [1, 2, 3, 4], **kwargs)


def test_flag_off_is_the_blocking_execute(fake, monkeypatch):
    monkeypatch.setattr(spin, "SPIN", False)
    decoder = _decoder(fake.calls)
    assert _enqueue(decoder) == 4
    assert [c for c in fake.calls if c[0] not in ("ple", "tok", "adv")] == [
        ("execute", "mesh", 11, 0, True),
        ("fenced",),
        ("proved",),
    ]
    assert set(decoder.step_split) >= {"ple", "tok", "exe", "adv", "read"}


def test_flag_on_spins_after_a_non_blocking_execute(fake, monkeypatch):
    monkeypatch.setattr(spin, "SPIN", True)
    decoder = _decoder(fake.calls)
    assert _enqueue(decoder) == 4
    names = [c[0] for c in fake.calls]
    execute = fake.calls.index(("execute", "mesh", 11, 0, False))
    # execute (non-blocking), the NaN host, the non-blocking read, the polls, the mesh event wait, then the fence
    assert names[execute:] == [
        "execute",
        "from_torch",
        "copy",
        "poll",
        "poll",
        "poll",
        "poll",
        "record_event",
        "event_synchronize",
        "fenced",
        "proved",
        "adv",
    ]
    assert fake.calls[execute + 2] == ("copy", ("shard", "row"), False)
    assert fake.calls[execute + 1][1] == (1, 1, 1, 32)
    assert fake.hosts[0].polls == 4  # NaN three times, then the value: the loop left when the read completed
    assert decoder.step_split["exe"] >= 0.0


def test_flag_on_non_blocking_enqueue_does_not_spin(fake, monkeypatch):
    monkeypatch.setattr(spin, "SPIN", True)
    decoder = _decoder(fake.calls)
    _enqueue(decoder, blocking=False)
    assert ("execute", "mesh", 11, 0, False) in fake.calls
    assert not [c for c in fake.calls if c[0] in ("poll", "copy", "record_event", "event_synchronize")]


def test_spin_times_out(fake, monkeypatch, expect_error):
    monkeypatch.setattr(spin, "SPIN", True)
    fake.complete_after = 10**9
    host = spin.start_read("shard")
    with expect_error(TimeoutError):
        spin.wait_read(host, timeout_s=0.0)


def test_wait_read_is_a_no_op_when_off(fake, monkeypatch):
    monkeypatch.setattr(spin, "SPIN", False)
    assert spin.wait_read(object()) == 0
    assert fake.calls == []


@pytest.mark.parametrize("on", [False, True])
def test_enqueue_token_read(fake, monkeypatch, on):
    monkeypatch.setattr(spin, "SPIN", on)
    decoder = _decoder(fake.calls)
    host, adapter, plugin = batch.Qwen38PlainLaneDecoder.enqueue_token_read(decoder)
    names = [c[0] for c in fake.calls]
    if on:
        assert names == ["from_torch", "copy", "record_event", "record_event"]
        assert fake.calls[1][2] is False
    else:
        assert fake.calls[0] == ("from_device", ("shard", "row"), False)
        assert names == ["from_device", "record_event", "record_event"]


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("kind", ["chain", "lane"])
def test_wait_handle_order(fake, monkeypatch, on, kind):
    monkeypatch.setattr(spin, "SPIN", on)
    host = FakeHost(fake.calls, 2)
    handle = SimpleNamespace(kind=kind, host=host, adapter_event="adapter")
    chain = SimpleNamespace(
        event_synchronize=lambda e: fake.calls.append(("chain_sync", e)),
        mark_ple_fenced=lambda: fake.calls.append(("chain_fenced",)),
    )
    adapter = SimpleNamespace(
        chain=chain, decoder=None, _async_pending=None, slot=SimpleNamespace(failed=False), batch_failed=False
    )
    vllm_module.Qwen38ForCausalLM._wait_handle(adapter, handle)
    names = [c[0] for c in fake.calls]
    polls = ["poll"] * 3 if on else []
    last = ["chain_sync", "chain_fenced"] if kind == "chain" else ["event_synchronize"]
    assert names == polls + last
