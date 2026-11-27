# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the live-tracer registry (no device needed)."""

import gc

from models.tt_dit.utils.tracing import Tracer


class _StubDevice:
    def __init__(self, device_id: int) -> None:
        self._id = device_id

    def id(self) -> int:
        return self._id


def test_release_all_visits_every_live_tracer(monkeypatch):
    released = []
    monkeypatch.setattr(Tracer, "release_trace", lambda self: released.append(self))

    a = Tracer(lambda: None, device=_StubDevice(9000))
    b = Tracer(lambda: None, devices=[_StubDevice(9001), _StubDevice(9002)])
    Tracer.release_all()

    assert a in released
    assert b in released


def test_release_all_skips_dropped_tracer(monkeypatch):
    released = []
    monkeypatch.setattr(Tracer, "release_trace", lambda self: released.append(id(self)))

    kept = Tracer(lambda: None, device=_StubDevice(9000))
    dropped = Tracer(lambda: None, device=_StubDevice(9000))
    dropped_id = id(dropped)
    del dropped
    gc.collect()
    Tracer.release_all()

    assert id(kept) in released
    assert dropped_id not in released


def test_release_all_continues_past_a_failing_release(monkeypatch):
    import models.tt_dit.utils.tracing as tracing

    released = []

    def release_trace(device, trace_id):
        if device.id() == 9100:
            msg = "mesh device is closed"
            raise RuntimeError(msg)
        released.append(device.id())

    monkeypatch.setattr(tracing.ttnn, "release_trace", release_trace)

    failing = Tracer(lambda: None, device=_StubDevice(9100))
    healthy = Tracer(lambda: None, device=_StubDevice(9101))
    for tracer, device_id in ((failing, 9100), (healthy, 9101)):
        tracer._trace_ids = (object(),)
        monkeypatch.setitem(Tracer._traces_live, device_id, 1)

    Tracer.release_all()

    assert released == [9101]
    assert failing not in tracing._live
    assert healthy in tracing._live
