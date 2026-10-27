# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A pipeline adapter publish-hf generates must decode through the trace when vLLM asks for one.

kolibri_1, 2026-10-26: its card reports 39.47 tok/s/user traced, but the package it published served
about 8 tok/s. vLLM's plugin warms the generator up twice and the second pass asks for traced decode
(warmup_model_decode(enable_trace=True), then decode_forward(..., enable_trace=True) for every token);
the adapter template returned early from the trace warmup and swallowed enable_trace, so every token
ran the eager step -- the speed the model was optimized AWAY from. Nemotron's hand-written adapter has
the same shape and is model code, so it is out of scope here and must never be overwritten.

No device: ttnn and the demo pipeline are fakes that record the order of calls.
"""

import json
import logging
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import publish_hf as ph  # noqa: E402

PMOD = "fake_demo_pipeline_for_adapter_test"


class FakePipe:
    def __init__(self, calls, traceable=True, capture_raises=False):
        self.calls = calls
        self.config = types.SimpleNamespace(vocab_size=4, bos_token_id=1)
        self.batch = 32
        self.tok_u32 = object()
        self.prefill_len = None
        self._capture_raises = capture_raises
        if not traceable:
            self.decode_trace_step = None

    def _write(self, name, t, dtype, layout):
        self.calls.append(("write", name, int(t.reshape(-1)[0])))

    def decode_prefill(self, ids):
        self.prefill_len = 32 * ((len(ids[0]) + 31) // 32)
        self.calls.append(("prefill", self.prefill_len))
        return {"logits": "prefill_logits"}

    def decode_step(self, state):
        self.calls.append(("eager",))
        return {"logits": "eager_logits"}

    def decode_trace_step(self):
        if self._capture_raises:
            raise RuntimeError("trace region too small")
        self.calls.append(("record",))
        return "trace_logits"


@pytest.fixture
def adapter(monkeypatch):
    """(build, calls): build(**pipe_kwargs) returns a generator the way the plugin creates one."""
    calls = []
    ids = iter(range(1, 100))

    fake_ttnn = types.ModuleType("ttnn")
    fake_ttnn.uint32, fake_ttnn.ROW_MAJOR_LAYOUT = "uint32", "row_major"

    def begin_trace_capture(dev, cq_id=0):
        tid = next(ids)
        calls.append(("begin", tid))
        return tid

    fake_ttnn.begin_trace_capture = begin_trace_capture
    fake_ttnn.end_trace_capture = lambda dev, tid, cq_id=0: calls.append(("end", tid))
    fake_ttnn.execute_trace = lambda dev, tid, cq_id=0, blocking=True: calls.append(("execute", tid))
    fake_ttnn.release_trace = lambda dev, tid: calls.append(("release", tid))
    monkeypatch.setitem(sys.modules, "ttnn", fake_ttnn)

    pipe_kwargs = {}
    fake_pmod = types.ModuleType(PMOD)
    fake_pmod.KV_CAPACITY = 2048
    fake_pmod.build_pipeline = lambda dev, layers=None: FakePipe(calls, **pipe_kwargs)

    def read_host(t):
        calls.append(("read", t))
        return torch.zeros(4)

    fake_pmod.read_host = read_host
    monkeypatch.setitem(sys.modules, PMOD, fake_pmod)

    src = ph._pipeline_adapter_src("fake_demo", "FakeForCausalLM", "TTFakeForCausalLM", PMOD, 2048)
    mod = types.ModuleType("rendered_adapter")
    exec(compile(src, "rendered_adapter.py", "exec"), mod.__dict__)

    def build(**kw):
        pipe_kwargs.clear()
        pipe_kwargs.update(kw)
        gen = mod.TTFakeForCausalLM.initialize_vllm_model(None, "mesh", 1, max_seq_len=1024)
        # The plugin's two-phase warmup: compile everything eagerly, then capture traces.
        gen.warmup_model_prefill(enable_trace=False)
        gen.warmup_model_decode(enable_trace=False)
        gen.already_warmed_up_prefill = False
        gen.warmup_model_prefill(enable_trace=True)
        gen.warmup_model_decode(enable_trace=True)
        return gen

    return build, calls


def _request(gen, prompt_len, tokens, enable_trace=True):
    gen.prefill_forward(torch.ones((1, prompt_len), dtype=torch.int64), None, prompt_lens=[prompt_len])
    for tok in tokens:
        gen.decode_forward(torch.tensor([[tok]]), None, None, enable_trace=enable_trace)


def test_every_decode_token_replays_the_trace_captured_at_warmup(adapter):
    build, calls = adapter
    gen = build()
    assert calls[-3:] == [("begin", 1), ("record",), ("end", 1)], "the trace warmup must capture decode"

    del calls[:]
    _request(gen, 8, [7, 9, 11])

    assert ("eager",) not in calls, "a traced request ran the eager decode step"
    replays = [c for c in calls if c[0] == "execute"]
    assert replays == [("execute", 1)] * 3
    for tok in (7, 9, 11):
        write = calls.index(("write", "tok_u32", tok))
        assert calls[write + 1] == ("execute", 1), "vLLM's token must be written before the replay reads it"
    assert calls.count(("read", "trace_logits")) == 3


def test_a_request_without_enable_trace_decodes_eagerly(adapter):
    build, calls = adapter
    gen = build()
    del calls[:]
    _request(gen, 8, [7, 9], enable_trace=False)
    assert calls.count(("eager",)) == 2
    assert not [c for c in calls if c[0] == "execute"]


def test_a_new_prefill_length_releases_the_trace_and_captures_it_again(adapter):
    build, calls = adapter
    gen = build()
    del calls[:]
    _request(gen, 40, [7])  # pads to 64; warmup only compiled the 32-token prefill
    assert calls[:2] == [("prefill", 64), ("release", 1)]
    assert calls[-6:] == [
        ("write", "tok_u32", 7),
        ("begin", 2),
        ("record",),
        ("end", 2),
        ("execute", 2),
        ("read", "trace_logits"),
    ]

    del calls[:]
    _request(gen, 50, [9])  # also pads to 64: the trace is kept
    assert ("release", 2) not in calls and ("begin", 3) not in calls
    assert ("execute", 2) in calls


def test_a_failed_capture_is_undone_and_reported_then_decode_runs_eagerly(adapter, caplog):
    build, calls = adapter
    with caplog.at_level(logging.WARNING):
        gen = build(capture_raises=True)
    assert calls[-3:] == [("begin", 1), ("end", 1), ("release", 1)], "a half-recorded trace must be closed"
    assert "decode trace capture failed" in caplog.text and "trace region too small" in caplog.text

    del calls[:]
    _request(gen, 8, [7])
    assert ("eager",) in calls and not [c for c in calls if c[0] in ("begin", "execute")]


def test_a_pipeline_without_a_trace_hook_says_it_decodes_eagerly(adapter, caplog):
    build, calls = adapter
    with caplog.at_level(logging.WARNING):
        gen = build(traceable=False)
    assert "has no decode_trace_step" in caplog.text

    del calls[:]
    _request(gen, 8, [7])
    assert ("eager",) in calls and not [c for c in calls if c[0] in ("begin", "execute")]


def _bundle(tmp_path, slug, adapter_src):
    (tmp_path / "models" / "demos" / slug / "tt").mkdir(parents=True)
    (tmp_path / "models" / "demos" / slug / "tt" / "pipeline.py").write_text("def build_pipeline(device):\n    pass\n")
    bundle = tmp_path / "models" / "demos" / slug / "vllm_bundle" / slug
    bundle.mkdir(parents=True)
    (bundle / "vllm_metadata.json").write_text(
        json.dumps({"arch": "Kolibri1ForCausalLM", "main_class": "adapter:TTKolibri1ForCausalLM"})
    )
    (bundle / "adapter.py").write_text(adapter_src)
    return bundle


def test_an_eager_copy_of_the_template_is_re_rendered(tmp_path):
    old = (
        "class TTKolibri1ForCausalLM:\n"
        "    def initialize_vllm_model(cls): ...\n"
        "    def prefill_forward(self): ...\n"
        "    def decode_forward(self): ...\n"
        "    def warmup_model_prefill(self, *, enable_trace=False):\n"
        "        if enable_trace:\n"
        "            " + ph._EAGER_PIPELINE_ADAPTER_MARK + "\n"
    )
    bundle = _bundle(tmp_path, "kolibri_1", old)
    assert ph._adapter_is_working(bundle)

    assert ph._refresh_eager_pipeline_adapter(tmp_path, "kolibri_1", bundle)
    new = (bundle / "adapter.py").read_text()
    assert "class TTKolibri1ForCausalLM" in new
    assert "from models.demos.kolibri_1.tt.pipeline import build_pipeline" in new
    assert "ttnn.execute_trace" in new and ph._EAGER_PIPELINE_ADAPTER_MARK not in new
    assert ph._adapter_is_working(bundle)
    assert not ph._refresh_eager_pipeline_adapter(tmp_path, "kolibri_1", bundle), "re-rendering must settle"


def test_a_hand_written_adapter_is_never_replaced(tmp_path):
    hand = (
        "class TTNemotronHForCausalLM:\n"
        "    def initialize_vllm_model(cls): ...\n"
        "    def prefill_forward(self): ...\n"
        "    def decode_forward(self): ...\n"
        "    def warmup_model_decode(self, *, enable_trace=False):\n"
        "        if enable_trace:\n"
        "            return  # eager-only generator; trace warmup is a no-op\n"
    )
    bundle = _bundle(tmp_path, "nemotron", hand)
    assert not ph._refresh_eager_pipeline_adapter(tmp_path, "nemotron", bundle)
    assert (bundle / "adapter.py").read_text() == hand
