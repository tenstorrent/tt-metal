# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host checks of QWEN36_SPEC_DR_READBACK_TOLERANT / QWEN36_SPEC_DR_OUTS_PREALLOC (no device)."""
import types

import pytest
import torch

from models.demos.blackhole.qwen36.tt import tp_common
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

# seen on device (K1-gate gated serving, bucketed step): rows past the bucket hold any 32-bit word
STEPS = [[728, 279, 0, 0], [264, 3672, 3208953432, 1058094935], [5357, 11, 0, 0]]


def _eng():
    e = types.SimpleNamespace(B=4, K=3, _dr_outs=[], _vb=2)
    e._ksp_width = lambda: 2
    return e


def test_flags_default_off(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_DR_OUTS_PREALLOC", raising=False)
    monkeypatch.delenv("QWEN36_SPEC_DR_READBACK_TOLERANT", raising=False)
    assert not tp_common.spec_dr_outs_prealloc_enabled()
    assert not tp_common.spec_dr_readback_tolerant_enabled()


def test_default_still_raises(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_DR_READBACK_TOLERANT", raising=False)
    with pytest.raises(RuntimeError):
        MTPSpecEngine._drafts_from_steps(_eng(), STEPS)


def test_tolerant_zeroes_only_out_of_range(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_DR_READBACK_TOLERANT", "1")
    e = _eng()
    d = MTPSpecEngine._drafts_from_steps(e, STEPS)
    assert d.dtype == torch.int32 and tuple(d.shape) == (4, 3)
    assert d[0].tolist() == [728, 264, 5357] and d[1].tolist() == [279, 3672, 11]
    assert d[2, 1] == 0 and d[3, 1] == 1058094935
    assert e._dr_readback_oob == 1


def test_tolerant_identical_on_clean_steps(monkeypatch):
    clean = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    monkeypatch.delenv("QWEN36_SPEC_DR_READBACK_TOLERANT", raising=False)
    a = MTPSpecEngine._drafts_from_steps(_eng(), clean)
    monkeypatch.setenv("QWEN36_SPEC_DR_READBACK_TOLERANT", "1")
    b = MTPSpecEngine._drafts_from_steps(_eng(), clean)
    assert torch.equal(a, b)
