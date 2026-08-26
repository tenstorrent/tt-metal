# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Program-cache contract for capture_peak_memory.

A NO_DISPATCH capture can leave the device program cache holding entries for programs that
were never dispatched; a later vector that hits one reads back garbage. These tests pin the
resulting contract:

- NO_DISPATCH captures purge the cache, whether the capture succeeded or not.
- NORMAL captures dispatch for real, so their cache entries are valid and are kept.
- A purge that fails is surfaced, never swallowed.

They mock ttnn.graph and the device, so they need no hardware.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from sweep_utils.memory_utils import capture_peak_memory


class FakeGridSize:
    x = 8
    y = 8


class FakeDevice:
    def __init__(self, clear_raises=False):
        self.clear_calls = 0
        self._clear_raises = clear_raises

    def compute_with_storage_grid_size(self):
        return FakeGridSize()

    def clear_program_cache(self):
        self.clear_calls += 1
        if self._clear_raises:
            raise RuntimeError("device wedged")


class PassingModule:
    @staticmethod
    def run(device, **kwargs):
        return [True, 0]


class FailingModule:
    @staticmethod
    def run(device, **kwargs):
        raise RuntimeError("op blew up mid-capture")


@pytest.fixture
def fake_graph(monkeypatch):
    """Stub ttnn.graph so the capture path runs without a device."""
    import ttnn

    class FakePerCore:
        peak_total = 4096
        peak_cb = 1024
        peak_l1 = 3072

    modes = []
    monkeypatch.setattr(ttnn.graph, "begin_graph_capture", lambda mode: modes.append(mode))
    monkeypatch.setattr(ttnn.graph, "end_graph_capture", lambda: object())
    monkeypatch.setattr(ttnn.graph, "extract_resource_usage_per_core", lambda _g: FakePerCore())
    monkeypatch.setattr(ttnn.graph, "extract_peak_L1_memory_usage", lambda _g: 8192)
    return modes


def test_no_dispatch_capture_purges_cache(fake_graph):
    device = FakeDevice()
    metrics = capture_peak_memory(PassingModule, {}, device, use_no_dispatch=True)
    assert metrics is not None
    assert device.clear_calls == 1


def test_normal_capture_keeps_cache(fake_graph):
    """NORMAL dispatches for real, so its cache entries are valid and must survive."""
    device = FakeDevice()
    metrics = capture_peak_memory(PassingModule, {}, device, use_no_dispatch=False)
    assert metrics is not None
    assert device.clear_calls == 0


def test_no_dispatch_purges_cache_even_when_the_test_fails(fake_graph):
    """The undispatched entries are already in the cache once the op path has been walked."""
    device = FakeDevice()
    capture_peak_memory(FailingModule, {}, device, use_no_dispatch=True)
    assert device.clear_calls == 1


def test_failure_before_capture_starts_does_not_touch_cache(fake_graph, monkeypatch):
    """Nothing has run yet, so there is nothing to purge -- and no attribute to blow up on."""
    import ttnn

    def explode(_mode):
        raise RuntimeError("capture could not start")

    monkeypatch.setattr(ttnn.graph, "begin_graph_capture", explode)
    device = FakeDevice()
    assert capture_peak_memory(PassingModule, {}, device, use_no_dispatch=True) is None
    assert device.clear_calls == 0


def test_purge_failure_is_surfaced_not_swallowed(fake_graph, expect_error):
    """Returning normally here would hand a poisoned cache to the next vector."""
    device = FakeDevice(clear_raises=True)
    with expect_error(RuntimeError, "undispatched programs"):
        capture_peak_memory(PassingModule, {}, device, use_no_dispatch=True)
    assert device.clear_calls == 1
