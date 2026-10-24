# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the 4-chip column-mesh defaults (tt/mesh_orientation.py): the (1,4) -> (4,1) reshape
decision of a vLLM build, the TP-axis default per mesh shape and the dispatch-prefill defaults, with every env
escape hatch. No device, no ttnn import.

    pytest models/demos/gemma4/tests/unit/test_mesh_orientation_host.py --noconftest -p no:cacheprovider
"""

import pytest

from models.demos.gemma4.tt import mesh_orientation as mo

_ENVS = (mo.COLUMN_MESH_ENV, mo.TP_AXIS_ENV, mo.DISPATCH_ENV, mo.DISPATCH_MIN_TOKENS_ENV)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)


def test_column_mesh_detection():
    assert mo.is_column_mesh((4, 1)) and mo.is_column_mesh((8, 1))
    for s in ((1, 4), (1, 1), (2, 4), (8, 4), (1, 8)):
        assert not mo.is_column_mesh(s)


def test_tp_axis_default_and_override(monkeypatch):
    assert mo.tp_axis_for((4, 1)) == 0
    assert mo.tp_axis_for((1, 4)) == 1
    assert mo.tp_axis_for((2, 4)) == 1
    assert mo.tp_axis_for(None) == 1
    monkeypatch.setenv(mo.TP_AXIS_ENV, "0")
    assert mo.tp_axis_for((1, 4)) == 0
    monkeypatch.setenv(mo.TP_AXIS_ENV, "1")
    assert mo.tp_axis_for((4, 1)) == 1


def test_reshape_decision(monkeypatch):
    assert mo.should_reshape_to_column((1, 4), blackhole=True, moe=True) == (4, 1)
    # already a column mesh, dense models, Wormhole, DP > 1, lane fold, other shapes: keep
    assert mo.should_reshape_to_column((4, 1), blackhole=True, moe=True) is None
    assert mo.should_reshape_to_column((1, 4), blackhole=True, moe=False) is None
    assert mo.should_reshape_to_column((1, 4), blackhole=False, moe=True) is None
    assert mo.should_reshape_to_column((1, 4), blackhole=True, moe=True, data_parallel=2) is None
    assert mo.should_reshape_to_column((1, 4), blackhole=True, moe=True, lanes=True) is None
    for s in ((1, 8), (2, 4), (8, 4), (1, 1)):
        assert mo.should_reshape_to_column(s, blackhole=True, moe=True) is None
    monkeypatch.setenv(mo.COLUMN_MESH_ENV, "0")
    assert mo.should_reshape_to_column((1, 4), blackhole=True, moe=True) is None


def test_dispatch_defaults(monkeypatch):
    assert mo.dispatch_prefill_default((4, 1), blackhole=True)
    assert mo.dispatch_min_tokens_default((4, 1), blackhole=True) == 128
    assert not mo.dispatch_prefill_default((1, 4), blackhole=True)
    assert mo.dispatch_min_tokens_default((1, 4), blackhole=True) == 256
    assert not mo.dispatch_prefill_default((4, 1), blackhole=False)
    assert not mo.dispatch_prefill_default((8, 4), blackhole=True)
    monkeypatch.setenv(mo.DISPATCH_ENV, "0")
    assert not mo.dispatch_prefill_default((4, 1), blackhole=True)
    assert mo.dispatch_min_tokens_default((4, 1), blackhole=True) == 256
    monkeypatch.setenv(mo.DISPATCH_ENV, "1")
    assert mo.dispatch_prefill_default((1, 4), blackhole=True)
    monkeypatch.setenv(mo.DISPATCH_MIN_TOKENS_ENV, "512")
    assert mo.dispatch_min_tokens_default((1, 4), blackhole=True) == 512
