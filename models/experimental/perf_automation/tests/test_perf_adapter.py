# SPDX-License-Identifier: Apache-2.0
"""PipelineDecodeAdapter: trace+1cq decode adapter (no CQ1 input staging)."""

import sys
import types


def _fake_ttnn():
    m = types.ModuleType("ttnn")
    m.from_device = lambda buf: ("host_seed", buf)
    m.copy_host_to_device_tensor = lambda host, dev, cq_id=0: None
    return m


class _PipeNeither:
    def decode_step(self, state):
        return state


def test_neither_no_write_inputs_means_1cq(monkeypatch):
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn())
    from agent.perf_adapter import PipelineDecodeAdapter

    a = PipelineDecodeAdapter(lambda dev: _PipeNeither())
    a.setup(device=object())
    assert not hasattr(a, "write_inputs")  # trace+1cq: no CQ1 input staging is ever bound


def test_resolve_mesh_shape_reads_env_else_default(monkeypatch):
    from agent.perf_adapter import resolve_mesh_shape

    monkeypatch.delenv("TT_PERF_MESH_ROWS", raising=False)
    monkeypatch.delenv("TT_PERF_MESH_COLS", raising=False)
    assert resolve_mesh_shape(default_rows=1, default_cols=4) == (1, 4)  # unset -> source default

    monkeypatch.setenv("TT_PERF_MESH_ROWS", "1")
    monkeypatch.setenv("TT_PERF_MESH_COLS", "1")
    assert resolve_mesh_shape(default_rows=1, default_cols=4) == (1, 1)  # env wins -> single chip

    monkeypatch.setenv("TT_PERF_MESH_ROWS", "2")
    monkeypatch.setenv("TT_PERF_MESH_COLS", "2")
    assert resolve_mesh_shape(default_rows=1, default_cols=4) == (2, 2)  # env wins -> planned split

    monkeypatch.setenv("TT_PERF_MESH_ROWS", "notanint")
    assert resolve_mesh_shape(default_rows=1, default_cols=4) == (1, 4)  # bad env -> default
