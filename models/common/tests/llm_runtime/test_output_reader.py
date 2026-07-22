# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref
from dataclasses import FrozenInstanceError

import torch

import ttnn
from models.common.llm_runtime.output_reader import OutputReader, PendingRead


class FakeDeviceTensor:
    def __init__(self, name: str):
        self.name = name
        self.cpu_calls = []
        self.host_value = object()

    def cpu(self, *, blocking: bool):
        self.cpu_calls.append(blocking)
        return self.host_value


class FailingDeviceTensor:
    def __init__(self):
        self.cpu_calls = []

    def cpu(self, *, blocking: bool):
        self.cpu_calls.append(blocking)
        raise RuntimeError("copy failed")


def _install_events(monkeypatch):
    events = []
    synchronized = []

    def record_event(device, queue_id):
        event = object()
        events.append((device, queue_id, event))
        return event

    monkeypatch.setattr(ttnn, "record_event", record_event)
    monkeypatch.setattr(ttnn, "event_synchronize", synchronized.append)
    return events, synchronized


def test_blocking_read_returns_nested_host_payload_without_retention(monkeypatch):
    events, synchronized = _install_events(monkeypatch)
    reader = OutputReader("mesh")
    output = FakeDeviceTensor("output")
    log_probs = FakeDeviceTensor("log_probs")

    host = reader.read((output, log_probs), blocking=True)

    assert host == (output.host_value, log_probs.host_value)
    assert output.cpu_calls == [True]
    assert log_probs.cpu_calls == [True]
    assert events == []
    assert synchronized == []
    assert reader.pending_count == 0


def test_synchronized_read_submits_every_copy_then_synchronizes_device_once(monkeypatch):
    synchronized = []
    monkeypatch.setattr(ttnn, "synchronize_device", synchronized.append)
    reader = OutputReader("mesh")
    output = FakeDeviceTensor("output")
    log_probs = FakeDeviceTensor("log_probs")

    host = reader.read_synchronized((output, log_probs))

    assert host == (output.host_value, log_probs.host_value)
    assert output.cpu_calls == [False]
    assert log_probs.cpu_calls == [False]
    assert synchronized == ["mesh"]
    assert reader.pending_count == 0


def test_async_read_retains_destination_and_event_until_completion(monkeypatch):
    events, synchronized = _install_events(monkeypatch)
    reader = OutputReader("mesh")
    output = FakeDeviceTensor("output")

    pending = reader.submit(output)

    assert isinstance(pending, PendingRead)
    assert pending.value is output.host_value
    assert pending.events == (events[0][2],)
    assert events[0][:2] == ("mesh", 0)
    assert output.cpu_calls == [False]
    assert not pending.completed
    assert reader.pending_reads == (pending,)

    assert reader.complete(pending) is output.host_value
    assert synchronized == [events[0][2]]
    assert pending.completed
    assert reader.pending_count == 0

    assert reader.complete(pending) is output.host_value
    assert synchronized == [events[0][2]]


def test_async_read_can_be_completed_by_exact_unwrapped_value(monkeypatch):
    events, synchronized = _install_events(monkeypatch)
    reader = OutputReader("mesh")
    output = FakeDeviceTensor("output")
    pending = reader.submit(output)

    assert reader.complete(pending.value) is pending.value
    assert synchronized == [events[0][2]]
    assert pending.completed
    assert reader.pending_count == 0


def test_reader_rejects_another_readers_pending_handle(monkeypatch, expect_error):
    _install_events(monkeypatch)
    first_reader = OutputReader("first_mesh")
    second_reader = OutputReader("second_mesh")
    foreign_pending = first_reader.submit(FakeDeviceTensor("first"))
    own_pending = second_reader.submit(FakeDeviceTensor("second"))

    with expect_error(ValueError, "not owned"):
        second_reader.complete(foreign_pending)

    assert second_reader.pending_reads == (own_pending,)
    first_reader.drain()
    second_reader.drain()


def test_reader_rejects_another_readers_completed_pending_handle(monkeypatch, expect_error):
    _install_events(monkeypatch)
    first_reader = OutputReader("first_mesh")
    second_reader = OutputReader("second_mesh")
    foreign_pending = first_reader.submit(torch.tensor([1]))

    assert foreign_pending.completed
    with expect_error(ValueError, "not owned"):
        second_reader.complete(foreign_pending)


def test_drain_completes_every_pending_read_and_is_idempotent(monkeypatch, expect_error):
    events, synchronized = _install_events(monkeypatch)
    reader = OutputReader("mesh")
    first = reader.submit(FakeDeviceTensor("first"))
    second = reader.submit(FakeDeviceTensor("second"))

    with expect_error(FrozenInstanceError, ""):
        first.completed = True

    reader.drain()

    assert synchronized == [events[0][2], events[1][2]]
    assert first.completed
    assert second.completed
    assert reader.pending_count == 0

    reader.drain()
    assert synchronized == [events[0][2], events[1][2]]


def test_async_host_only_payload_is_already_complete(monkeypatch):
    events, synchronized = _install_events(monkeypatch)
    reader = OutputReader("mesh")
    host_tensor = torch.tensor([1, 2])

    pending = reader.submit((host_tensor, None))

    assert pending.completed
    assert torch.equal(pending.value[0], host_tensor)
    assert pending.value[1] is None
    assert pending.events == ()
    assert reader.pending_count == 0
    assert reader.complete(pending) is pending.value
    assert events == []
    assert synchronized == []


def test_nested_dict_and_list_preserve_shape(monkeypatch):
    _install_events(monkeypatch)
    reader = OutputReader("mesh")
    output = FakeDeviceTensor("output")
    log_probs = FakeDeviceTensor("log_probs")

    pending = reader.submit({"outputs": [output], "log_probs": log_probs})

    assert pending.value == {"outputs": [output.host_value], "log_probs": log_probs.host_value}
    reader.complete(pending)


def test_record_event_failure_synchronizes_device(monkeypatch):
    device_synchronizations = []
    monkeypatch.setattr(ttnn, "record_event", lambda *_: (_ for _ in ()).throw(RuntimeError("record failed")))
    monkeypatch.setattr(ttnn, "synchronize_device", device_synchronizations.append)
    reader = OutputReader("mesh")

    try:
        reader.submit(FakeDeviceTensor("output"))
    except RuntimeError as error:
        assert str(error) == "record failed"
    else:
        raise AssertionError("record_event failure was not propagated")

    assert device_synchronizations == ["mesh"]
    assert reader.pending_count == 0


def test_partial_nested_async_copy_failure_synchronizes_while_destinations_are_retained(monkeypatch, expect_error):
    destination_refs = []
    device_synchronizations = []

    class HostDestination:
        pass

    class EphemeralDeviceTensor:
        def cpu(self, *, blocking: bool):
            destination = HostDestination()
            destination_refs.append(weakref.ref(destination))
            return destination

    def synchronize_device(mesh_device):
        assert destination_refs[0]() is not None
        device_synchronizations.append(mesh_device)

    monkeypatch.setattr(ttnn, "synchronize_device", synchronize_device)
    reader = OutputReader("mesh")
    failing = FailingDeviceTensor()

    with expect_error(RuntimeError, "copy failed"):
        reader.submit((EphemeralDeviceTensor(), failing))

    assert failing.cpu_calls == [False]
    assert device_synchronizations == ["mesh"]
    assert reader.pending_count == 0
