# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: a traced packed forward adds one capture per real-chunk count below forward_chunks only while the free
trace region still holds a full capture; the counts left without one replay the full capture."""

from types import SimpleNamespace

import pytest

from models.demos.deepseek_v3_d_p.tt import tt_prefill_runtime as runtime_module
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

MB = 2**20


class _Region:
    def __init__(self, size):
        self.size, self.used = size, 0


class _Controller:
    """A capture that takes ``size`` bytes of the shared region; trace_bytes() is the whole region in use, as
    SubDeviceTraceController.trace_bytes() reads it."""

    num_segments = 1

    def __init__(self, region, size):
        self.region, self.size = region, size

    def begin_capture(self):
        pass

    def end_capture(self):
        self.region.used += self.size

    def trace_bytes(self):
        return self.region.used


def _captures(monkeypatch, region_mb, full_mb, partial_mb, used_before_mb=0):
    region = _Region(region_mb * MB)
    region.used = used_before_mb * MB  # e.g. another capture made earlier on the device
    rt = object.__new__(TtPrefillRuntime)
    rt.config = SimpleNamespace(forward_chunks=4, use_trace=True, num_layers=1, is_last_rank=True)
    rt.mesh_device = SimpleNamespace(num_program_cache_entries=lambda: 0)
    rt.model = SimpleNamespace(
        packed_trace_skips_empty_positions=True, set_trace_controller=lambda c: None, last_indexer_indices=None
    )
    rt._controller = _Controller(region, full_mb * MB)
    rt._trace_d2h_service = None
    rt._controllers_by_real_count = {}
    rt._forward_traced = lambda kv: None
    rt._forward_traced_with_empty_tail = lambda kv, real, capture: None
    view = lambda mesh, kind: SimpleNamespace(num_banks=1, total_bytes_free_per_bank=region.size - region.used)
    monkeypatch.setattr(runtime_module.ttnn, "get_memory_view", view)
    monkeypatch.setattr(runtime_module.ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(runtime_module, "SubDeviceTraceController", lambda mesh: _Controller(region, partial_mb * MB))
    monkeypatch.setattr(
        runtime_module, "get_tt_ccl", lambda mesh: SimpleNamespace(set_shared_rs_input_keepalive=lambda t: None)
    )
    rt._capture_trace_packed(None)
    return sorted(rt._controllers_by_real_count)


@pytest.mark.parametrize(
    "region_mb, full_mb, partial_mb, used_before_mb, counts",
    [
        (100, 40, 10, 0, [1, 2, 3]),  # 40 MB left before the last capture: still one full capture's worth
        (100, 40, 25, 0, [1]),  # 35 MB left after the first: the 2- and 3-real forwards replay the full capture
        (100, 20, 10, 30, [1, 2, 3]),  # region used before the full capture is not part of the full capture
        (40, 40, 10, 0, []),
    ],
)
def test_captures_stop_when_the_region_cannot_hold_a_full_capture(
    monkeypatch, region_mb, full_mb, partial_mb, used_before_mb, counts
):
    assert _captures(monkeypatch, region_mb, full_mb, partial_mb, used_before_mb) == counts
