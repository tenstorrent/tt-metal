# SPDX-License-Identifier: Apache-2.0
"""Device-free trace and request-state lifecycle tests for the Laguna adapter."""

from __future__ import annotations

import pytest

from models.autoports.poolside_laguna_xs_2_1.tt import generator_vllm as gv


def _bare_adapter():
    adapter = gv.LagunaForCausalLM.__new__(gv.LagunaForCausalLM)
    adapter._closed = False
    adapter.mesh_device = object()
    adapter._decode = {}
    adapter._verify_dec = {}
    adapter._dflash_controller = None
    adapter._dflash_cache = None
    adapter._dflash_core = None
    adapter._dflash_tok = None
    adapter._dflash_request_id = None
    adapter._pf = None
    adapter._pf_pt = {}
    adapter._pf_fill_pt = {}
    adapter._pf_pt_groups = {}
    adapter._pf_fill_pt_groups = {}
    adapter._spec = None
    adapter._spec_tok = None
    adapter._spec_buf = []
    adapter._spec_hist = []
    adapter._spec_prefill_seq = []
    return adapter


def test_close_releases_all_traces_before_dflash_and_is_idempotent(monkeypatch):
    adapter = _bare_adapter()
    events = []
    adapter._decode = {1: {"tid": 11, "buffer": object()}}
    adapter._verify_dec = {3: {"tid": 22, "buffer": object()}}
    adapter._pf_pt = {(1, 1): object()}
    adapter._spec_buf = [7]

    class Controller:
        def close(self):
            events.append("dflash-close")

    adapter._dflash_controller = Controller()
    adapter._dflash_cache = object()
    adapter._dflash_core = object()
    adapter._dflash_tok = object()

    monkeypatch.setattr(gv.ttnn, "release_trace", lambda mesh, trace_id: events.append(f"trace-{trace_id}"))

    adapter.close()
    adapter.close()

    assert events == ["trace-11", "trace-22", "dflash-close"]
    assert adapter._closed
    assert adapter._decode == {}
    assert adapter._verify_dec == {}
    assert adapter._pf_pt == {}
    assert adapter._spec_buf == []
    assert adapter._dflash_controller is None
    assert adapter._dflash_cache is None
    assert adapter._dflash_core is None
    assert adapter._dflash_tok is None


def test_trace_release_failure_is_reported_and_can_be_retried(monkeypatch, expect_error):
    adapter = _bare_adapter()
    adapter._decode = {1: {"tid": 33}}
    attempts = []

    def release(mesh, trace_id):
        attempts.append(trace_id)
        if len(attempts) == 1:
            raise RuntimeError("injected release failure")

    monkeypatch.setattr(gv.ttnn, "release_trace", release)

    with expect_error(RuntimeError, "failed to release Laguna TT trace"):
        adapter.close()
    assert not adapter._closed
    assert adapter._decode[1]["tid"] == 33

    adapter.close()
    assert attempts == [33, 33]
    assert adapter._closed
    assert adapter._decode == {}


@pytest.mark.parametrize("per_layer", (False, True))
def test_kv_reallocation_releases_traces_before_first_new_tensor(monkeypatch, per_layer):
    adapter = _bare_adapter()
    events = []
    adapter._kv_dtype = object()
    adapter._HYBRID_KV_CACHE_GROUPS_ENABLED = False
    adapter._release_decode_traces = lambda: events.append("release-traces")
    adapter._report_dram = lambda *args, **kwargs: None

    def allocate(*args, **kwargs):
        events.append("allocate-tensor")
        return object()

    monkeypatch.setattr(gv.ttnn, "from_torch", allocate)
    monkeypatch.setattr(gv, "_replicate", lambda mesh: None)

    shape = (1, 1, 32, 64)
    if per_layer:
        adapter.allocate_kv_cache_per_layer([(shape, object(), 0)])
    else:
        adapter.allocate_kv_cache(shape, object(), 1)

    assert events == ["release-traces", "allocate-tensor", "allocate-tensor"]
