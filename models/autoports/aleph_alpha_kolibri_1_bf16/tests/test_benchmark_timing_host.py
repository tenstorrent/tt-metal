# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of deferred completion and FIFO instrumentation."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def test_phase_completion(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tt/benchmark_timing.py"
    spec = importlib.util.spec_from_file_location("models.autoports.aleph_alpha_kolibri_1_bf16.tt.timing_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Runner:
        def execute_model(self, *_):
            return None

        def _forward_with_model_input(self, model_input):
            return SimpleNamespace(model_input=model_input)

        def _finish_front_packed_sync(self, *args, fwd=None):
            return "sync-completed"

    class Deferred:
        def _get_output_impl(self):
            return "device-readback-and-host-format-completed"

    class Controller:
        def __init__(self, r):
            self.runner = r

        def submit_async_decode(self, *a, **kw):
            out = Deferred()
            out._submission = SimpleNamespace(tt_out=[])
            return out

    class Generator:
        def bind(self, *a, **kw):
            pass

        def replay(self, *a, **kw):
            pass

        def copy(self, *a, **kw):
            pass

    gen_mod = ModuleType("models.autoports.aleph_alpha_kolibri_1_bf16.tt.generator")
    gen_mod.KolibriGenerator = Generator
    monkeypatch.setitem(sys.modules, gen_mod.__name__, gen_mod)
    runner_mod = ModuleType("vllm_tt_plugin.model_runner")
    runner_mod.TTModelRunner = Runner
    async_mod = ModuleType("vllm_tt_plugin.async_decode")
    async_mod.TTAsyncDecodeController = Controller
    async_mod.AsyncTTModelRunnerOutput = Deferred
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin.model_runner", runner_mod)
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin.async_decode", async_mod)
    monkeypatch.setenv("KOLIBRI_PHASE_EVENTS", "unused")
    with patch.object(module, "ThreadingHTTPServer"), patch.object(module.threading, "Thread"):
        module.install()
    r = Runner()
    r.model = SimpleNamespace(
        batch_size=32, capacity=1048576, generator=SimpleNamespace(state=SimpleNamespace(host_page_tables={}))
    )
    r.execute_model()
    mi = SimpleNamespace(row_req_ids=["a"], input_positions=[0], prompt_lens=[33], intermediate_prefill_mask=[False])
    fwd = r._forward_with_model_input(mi)
    assert not module._EVENTS
    assert r._finish_front_packed_sync(None, fwd=fwd) == "sync-completed"
    r.execute_model()
    mi = SimpleNamespace(row_req_ids=["a"], input_positions=[33], prompt_lens=None, intermediate_prefill_mask=None)
    deferred = Controller(r).submit_async_decode(mi)
    assert len(module._EVENTS) == 1
    assert deferred._get_output_impl() == "device-readback-and-host-format-completed"
    assert [e["phase"] for e in module._EVENTS] == ["prefill", "decode"]
    assert all(e["end"] >= e["start"] for e in module._EVENTS)
    r.execute_model()
    pending = SimpleNamespace(
        row_req_ids=["b"], input_positions=[512], prompt_lens=[1024], intermediate_prefill_mask=[True]
    )
    fwd = r._forward_with_model_input(pending)
    r._finish_front_packed_sync(None, fwd=fwd)
    assert module._EVENTS[-1]["device_completion_observed"] is False
    assert "end" not in module._EVENTS[-1]
    r.execute_model()
    late = Controller(r).submit_async_decode(mi)
    late._get_output_impl()
    assert module._EVENTS[-1]["unfenced_prefill_at_decode_submission"] == [2]
    assert module._EVENTS[-1]["covers_pending_prefills"] == [2]
