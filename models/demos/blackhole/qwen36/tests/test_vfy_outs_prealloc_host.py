# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host checks of QWEN36_SPEC_VFY_OUTS_PREALLOC's helpers (no device; ttnn.copy / deallocate faked)."""
import types

import pytest

from models.demos.blackhole.qwen36.tt import model as qmodel
from models.demos.blackhole.qwen36.tt import tp_common
_M = qmodel.Qwen36Model


class _T:
    def __init__(self, name, sharded=False):
        self.name, self._sharded = name, sharded

    def memory_config(self):
        return types.SimpleNamespace(is_sharded=lambda: self._sharded)


@pytest.fixture
def fake_ttnn(monkeypatch):
    log = []
    monkeypatch.setattr(qmodel.ttnn, "copy", lambda a, b: log.append(("copy", a.name, b.name)))
    monkeypatch.setattr(qmodel.ttnn, "deallocate", lambda t: log.append(("free", t.name)))
    return log


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_VFY_OUTS_PREALLOC", raising=False)
    assert not tp_common.spec_vfy_outs_prealloc_enabled()
    monkeypatch.setenv("QWEN36_SPEC_VFY_OUTS_PREALLOC", "1")
    assert tp_common.spec_vfy_outs_prealloc_enabled()


def test_keep_interleaved_only():
    pre = {}
    a, s = _T("ids"), _T("normed", sharded=True)
    _M._vfy_pre_keep(pre, "ids", a)
    _M._vfy_pre_keep(pre, "normed", s)
    _M._vfy_pre_keep(pre, "topk_ids", None)
    assert pre == {"ids": a}
    assert _M._vfy_pre_has(pre, a) and not _M._vfy_pre_has(pre, s) and not _M._vfy_pre_has(None, a)


def test_into_copies_then_frees(fake_ttnn):
    dst, t = _T("pre_ids"), _T("trace_ids")
    out = _M._vfy_pre_into({"ids": dst}, "ids", t)
    assert out is dst
    assert fake_ttnn == [("copy", "trace_ids", "pre_ids"), ("free", "trace_ids")]


def test_into_without_destination_is_identity(fake_ttnn):
    t = _T("trace_rows")
    assert _M._vfy_pre_into({}, "rows", t) is t
    assert _M._vfy_pre_into(None, "rows", t) is t
    assert fake_ttnn == []
