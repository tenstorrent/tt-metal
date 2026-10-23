# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layers handed the same host page table share one persistent device buffer,
so refreshing the tables costs one copy per kv-cache group, not per layer."""


import torch

import models.demos.gemma4.tt.model as model_module
from models.demos.gemma4.tt.model import Gemma4Model


class _Dev:
    _n = 0

    def __init__(self, shape):
        self.shape = shape
        _Dev._n += 1
        self.uid = _Dev._n


def _model(monkeypatch, copies):
    monkeypatch.setenv("GEMMA4_SHARE_PAGE_TABLES", "1")  # sharing is opt-in
    m = Gemma4Model.__new__(Gemma4Model)
    m.mesh_config = None
    monkeypatch.setattr(m, "_page_table_torch_to_ttnn", lambda pt, layer_idx=None: _Dev(list(pt.shape)), raising=False)
    monkeypatch.setattr(m, "_replicate_to_mesh_mapper", lambda: None, raising=False)
    monkeypatch.setattr(model_module.ttnn, "from_torch", lambda t, **kw: t)
    monkeypatch.setattr(model_module.ttnn, "copy_host_to_device_tensor", lambda host, dev: copies.append(dev.uid))
    return m


def _tables(seed):
    sliding = torch.arange(4 * 8, dtype=torch.int32).reshape(4, 8) + seed
    full = torch.arange(4 * 8, dtype=torch.int32).reshape(4, 8) + 100 + seed
    return [sliding, sliding, sliding, full, sliding, full]


def test_layers_with_one_host_table_share_one_device_buffer(monkeypatch):
    copies = []
    m = _model(monkeypatch, copies)
    persistent = m._page_tables_to_ttnn(_tables(0))
    assert len(persistent) == 6
    assert persistent[0] is persistent[1] is persistent[2] is persistent[4]
    assert persistent[3] is persistent[5] and persistent[3] is not persistent[0]


def test_update_writes_each_shared_buffer_once_and_skips_unchanged_tables(monkeypatch):
    copies = []
    m = _model(monkeypatch, copies)
    m.update_persistent_per_layer_page_tables(_tables(0))
    assert sorted(copies) == sorted({d.uid for d in m._persistent_per_layer_page_tables}) and len(copies) == 2
    copies.clear()
    m.update_persistent_per_layer_page_tables(_tables(0))
    assert copies == []
    m.update_persistent_per_layer_page_tables(_tables(7))
    assert len(copies) == 2


def test_layers_that_stop_sharing_content_get_their_own_buffers(monkeypatch):
    copies = []
    m = _model(monkeypatch, copies)
    m.update_persistent_per_layer_page_tables(_tables(0))
    tables = _tables(3)
    tables[1] = tables[1].clone() + 1000  # layer 1 diverges from the sliding group
    copies.clear()
    m.update_persistent_per_layer_page_tables(tables)
    persistent = m._persistent_per_layer_page_tables
    assert persistent[0] is not persistent[1]
    assert m._invalidate_decode_traces_after_page_table_realloc is True
    # Rebuilt unshared: every layer owns a buffer and each was written (the one
    # copy made before the divergence was noticed lands on a replaced buffer).
    assert len({id(d) for d in persistent}) == 6
    assert len(copies) >= 6 and set(d.uid for d in persistent) <= set(copies)


def test_lane_parallel_prefill_table_shards_to_one_row_per_column():
    # A lanes-row table is the lane-parallel prefill layout: one row per column, like a
    # single-row table; it must not be sized as the lane_slots-row frame (that grew the
    # persistent buffers and dropped the decode traces after every lane-router prefill).
    rows = model_module.lane_page_table_device_rows
    assert rows(1, 4, 32) == 1
    assert rows(4, 4, 32) == 1
    assert rows(8, 4, 32) == 32
    assert rows(128, 4, 32) == 32
    assert rows(4, 4, 1) == 1  # single-slot lanes: the frame is lanes rows


def test_lane_parallel_prefill_table_does_not_regrow_its_buffer(monkeypatch):
    from types import SimpleNamespace

    copies = []
    m = _model(monkeypatch, copies)
    m.mesh_config = SimpleNamespace(lane_sharded=True, lanes=4, lane_shard_mapper=lambda dev, dim: None)
    m.lane_slots = 32
    m.mesh_device = None
    m._invalidate_decode_traces_after_page_table_realloc = False
    lane_parallel = [torch.arange(4 * 8, dtype=torch.int32).reshape(4, 8)] * 2  # one row per lane
    first = m._page_tables_to_ttnn(lane_parallel)
    again = m._page_tables_to_ttnn([t + 1 for t in lane_parallel])
    assert again[0] is first[0]  # same batch key, same device rows: no reallocation
    assert m._invalidate_decode_traces_after_page_table_realloc is False
    frame = [torch.zeros((128, 8), dtype=torch.int32)] * 2  # the decode frame keys its own buffers ...
    m._page_tables_to_ttnn(frame)
    # ... but a batch key first seen after the process's first key allocates where captured replays
    # scribble, so the next decode re-captures (the first key, the warmup's own, needs nothing).
    assert m._invalidate_decode_traces_after_page_table_realloc is True
    assert m._g4_page_table_realloc_reason.startswith("new batch key")
