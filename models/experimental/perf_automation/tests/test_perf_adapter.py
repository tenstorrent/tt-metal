# SPDX-License-Identifier: Apache-2.0
"""PipelineDecodeAdapter: the generic persistent-buffer 2CQ write path (#2)."""
import sys
import types


def _fake_ttnn(calls):
    m = types.ModuleType("ttnn")
    m.from_device = lambda buf: ("host_seed", buf)

    def copy(host, dev, cq_id=0):
        calls.append((host, dev, cq_id))

    m.copy_host_to_device_tensor = copy
    return m


class _PipeBuffer:
    def __init__(self):
        self.buf = object()

    def decode_step(self, state):
        return state

    def decode_prefill(self, ids):
        return {"s": 1}

    def decode_input_buffer(self, state):
        return self.buf


class _PipeWriteInputs:
    def __init__(self):
        self.wrote = []

    def decode_step(self, state):
        return state

    def decode_write_inputs(self, state):
        self.wrote.append(state)


class _PipeNeither:
    def decode_step(self, state):
        return state


def test_prefers_persistent_buffer_writes_same_buf_on_cq1(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn(calls))
    from agent.perf_adapter import PipelineDecodeAdapter

    pipe = _PipeBuffer()
    a = PipelineDecodeAdapter(lambda dev: pipe, prompt_ids=[1])
    a.setup(device=object())
    assert hasattr(a, "write_inputs")
    a.write_inputs()
    a.write_inputs()
    assert len(calls) == 2
    for _host, dev, cq in calls:
        assert dev is pipe.buf and cq == 1  # in-place into the SAME persistent buffer, on cq1


def test_falls_back_to_model_write_inputs(monkeypatch):
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn([]))
    from agent.perf_adapter import PipelineDecodeAdapter

    pipe = _PipeWriteInputs()
    a = PipelineDecodeAdapter(lambda dev: pipe)
    a.setup(device=object())
    assert hasattr(a, "write_inputs")
    a.write_inputs()
    assert pipe.wrote  # model-authored staging used when no decode_input_buffer


def test_neither_no_write_inputs_means_1cq(monkeypatch):
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn([]))
    from agent.perf_adapter import PipelineDecodeAdapter

    a = PipelineDecodeAdapter(lambda dev: _PipeNeither())
    a.setup(device=object())
    assert not hasattr(a, "write_inputs")  # measure_adapter auto -> single-CQ


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
