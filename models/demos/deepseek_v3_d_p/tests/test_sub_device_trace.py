# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.demos.deepseek_v3_d_p.utils import sub_device_trace


class _FakeMeshDevice:
    def __init__(self, events):
        self.events = events
        self.active_manager = "default"

    def load_sub_device_manager(self, manager_id):
        self.events.append(("load", manager_id))
        self.active_manager = manager_id

    def clear_loaded_sub_device_manager(self):
        self.events.append(("clear", self.active_manager))
        self.active_manager = "default"


def test_release_walks_trace_owners_without_firing_replay_acks(monkeypatch):
    events = []
    mesh_device = _FakeMeshDevice(events)
    controller = sub_device_trace.SubDeviceTraceController(mesh_device)
    controller.set_layer_ack_callback(lambda layer: events.append(("ack", layer)))
    controller._program = [
        (controller._TRACE, "default-before"),
        (controller._LOAD, "overlap"),
        (controller._TRACE, "overlap-trace"),
        (controller._ACK, 17),
        (controller._CLEAR, None),
        (controller._TRACE, "default-after"),
    ]

    def release_trace(actual_mesh, trace_id):
        assert actual_mesh is mesh_device
        events.append(("release", mesh_device.active_manager, trace_id))

    monkeypatch.setattr(sub_device_trace.ttnn, "release_trace", release_trace)

    controller.release()

    assert events == [
        ("release", "default", "default-before"),
        ("load", "overlap"),
        ("release", "overlap", "overlap-trace"),
        ("clear", "overlap"),
        ("release", "default", "default-after"),
    ]
    assert controller._program == []

    controller.release()
    assert len(events) == 5
