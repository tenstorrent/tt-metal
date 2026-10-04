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
