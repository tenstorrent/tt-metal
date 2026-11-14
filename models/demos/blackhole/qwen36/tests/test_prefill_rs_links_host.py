# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of QWEN36_PREFILL_RS_LINKS (tp_common.prefill_rs_links) and of how tt_all_reduce consumes it on the
1x8 TP mesh (reduce_scatter_minimal_async num_links). Needs ttnn importable (no device; the CCL op is monkeypatched).

  pytest -q models/demos/blackhole/qwen36/tests/test_prefill_rs_links_host.py
"""

import types

import pytest
import ttnn

from models.demos.blackhole.qwen36.tt import tp_common as tc
from models.tt_transformers.tt import ccl


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(tc, "_PREFILL_RS_LINKS", None)
    monkeypatch.delenv("QWEN36_PREFILL_RS_LINKS", raising=False)
    yield


@pytest.mark.parametrize("raw", [None, "", "0", "-1", " "])
def test_default_is_auto(monkeypatch, raw):
    if raw is not None:
        monkeypatch.setenv("QWEN36_PREFILL_RS_LINKS", raw)
    assert tc.prefill_rs_links() is None
    assert tc.prefill_rs_links(4096) is None


def test_set_two_links_prefill_rows_only(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_RS_LINKS", "2")
    assert tc.prefill_rs_links() == 2
    assert tc.prefill_rs_links(2048) == 2
    assert tc.prefill_rs_links(33) == 2
    # decode / verify widths (<= 1 tile of rows, e.g. B=8 x (K+1)=4 = 32) keep the auto-detect.
    assert tc.prefill_rs_links(32) is None
    assert tc.prefill_rs_links(1) is None


def test_read_once(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_RS_LINKS", "2")
    assert tc.prefill_rs_links(64) == 2
    monkeypatch.setenv("QWEN36_PREFILL_RS_LINKS", "1")
    assert tc.prefill_rs_links(64) == 2


def test_bad_value_raises(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_RS_LINKS", "two")
    with pytest.raises(ValueError):
        tc.prefill_rs_links()


class _FakeTensor:
    shape = (1, 1, 2048, 640)

    def is_sharded(self):
        return False

    def deallocate(self, *_):
        pass


class _FakeCCL:
    def __init__(self, axis_links):
        self.axis_links = axis_links
        self.queried = []

    def get_num_links(self, cluster_axis=None):
        self.queried.append(cluster_axis)
        return self.axis_links[cluster_axis]

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis=None):
        return None

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        return None


@pytest.mark.parametrize("links", [None, 2])
def test_tt_all_reduce_1x8_uses_requested_links(monkeypatch, links):
    """On the (1, 8) mesh qwen36 calls tt_all_reduce(cluster_axis=0); auto-detect queries the size-1 axis 0 (1 link
    here), an explicit num_reduce_scatter_links overrides it. cluster_axis=1 would skip the CCL entirely."""
    seen = {}

    def fake_rs(t, **kw):
        seen.update(kw)
        return "reduced"

    monkeypatch.setattr(ttnn.experimental, "reduce_scatter_minimal_async", fake_rs, raising=False)
    mesh = types.SimpleNamespace(shape=(1, 8))
    tt_ccl = _FakeCCL({0: 1, 1: 2, None: 1})
    out = ccl.tt_all_reduce(
        _FakeTensor(),
        mesh,
        tt_ccl,
        cluster_axis=0,
        dim=3,
        num_reduce_scatter_links=links,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert out == "reduced"
    assert seen["num_links"] == (1 if links is None else links)
    assert seen["dim"] == 3
