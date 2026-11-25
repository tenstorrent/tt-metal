# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for Tracer(hold_capture_frees=True) (ltx-rt task #6000).

A trace rewrites its intermediates' addresses on every replay. With the hold, the DRAM a capture
frees stays reserved until the trace is released, so a later job's buffers cannot land on it. This
checks the Python side with a fake ttnn: the hold is set only around begin_trace_capture, and only
H3's audio decoder asks for it. LTX's mel decoder, vocoders and BWE keep it off (task #6623): the hold
reserves the capture's intermediates without reuse, and the LTX 6 s 1080p cold warm-up then ran out
of DRAM on cs04 in the Gemma warm-up after the audio capture pass. Run directly; no device or quality
acceptance.
"""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACING = ROOT / "utils/tracing.py"
AUDIO = [ROOT / "models/audio_vae" / f for f in ("vocoder_ltx.py", "mel_decoder_ltx.py", "bwe_ltx.py")]
H3_DECODER = ROOT / "models/audio_vae/minimax_h3/decoder_minimax_h3_audio.py"


class FakeTensor:
    pass


class FakeDevice:
    def __init__(self, i=0):
        self._id = i

    def id(self):
        return self._id


def fake_ttnn(log, *, with_hold=True):
    m = types.ModuleType("ttnn")
    m.Tensor = FakeTensor
    m.MeshDevice = FakeDevice
    m.bfloat16 = "bf16"
    held = {}
    state = {"hold": {}}
    tid = iter(range(100, 200))

    def begin_trace_capture(d, cq_id=0):
        t = next(tid)
        log.append(("begin", d.id(), state["hold"].get(d.id(), False)))
        if state["hold"].get(d.id()):
            held[t] = 4096
        return t

    m.begin_trace_capture = begin_trace_capture
    m.end_trace_capture = lambda d, t, cq_id=0: log.append(("end", d.id()))
    m.release_trace = lambda d, t: log.append(("release", d.id()))
    m.execute_trace = lambda d, t, cq_id=0, blocking=True: log.append(("exec", d.id()))
    if with_hold:

        def set_trace_capture_hold(d, on):
            state["hold"][d.id()] = on
            log.append(("hold", d.id(), on))

        m.set_trace_capture_hold = set_trace_capture_hold
        m.trace_capture_hold_bytes = lambda d, t: held.get(t, 0)
    return m


def load_tracing(ttnn_mod):
    names = ("ttnn", "loguru", "models.tt_dit.utils", "models.tt_dit.utils.tensor", "models.tt_dit.utils.trace_store")
    saved = {k: sys.modules.get(k) for k in names}
    loguru = types.ModuleType("loguru")
    loguru.logger = types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None,
                                          warning=lambda *a, **k: None)
    utils = types.ModuleType("models.tt_dit.utils")
    utils.tensor = types.ModuleType("models.tt_dit.utils.tensor")
    utils.trace_store = types.ModuleType("models.tt_dit.utils.trace_store")
    utils.trace_store.enabled = lambda: False
    sys.modules.update({"ttnn": ttnn_mod, "loguru": loguru, "models.tt_dit.utils": utils,
                        "models.tt_dit.utils.tensor": utils.tensor,
                        "models.tt_dit.utils.trace_store": utils.trace_store})
    try:
        spec = importlib.util.spec_from_file_location("tracing_under_test", TRACING)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class TraceCaptureHoldTest(unittest.TestCase):
    def capture(self, *, hold, with_hold=True):
        log = []
        tracing = load_tracing(fake_ttnn(log, with_hold=with_hold))
        t = tracing.Tracer(lambda: None, device=FakeDevice(), prep_run=False, hold_capture_frees=hold)
        t._record(cq_id=0, blocking_execution=True, execute=False)
        return t, log

    def test_hold_wraps_only_begin(self):
        t, log = self.capture(hold=True)
        self.assertEqual(log, [("hold", 0, True), ("begin", 0, True), ("hold", 0, False), ("end", 0)])
        self.assertEqual(t.capture_hold_bytes, 4096)
        t.release_trace()
        self.assertEqual(log[-1], ("release", 0))

    def test_no_hold_by_default(self):
        t, log = self.capture(hold=False)
        self.assertEqual(log, [("begin", 0, False), ("end", 0)])
        self.assertEqual(t.capture_hold_bytes, 0)

    def test_hold_without_binding_refuses(self):
        with self.assertRaises(RuntimeError):
            self.capture(hold=True, with_hold=False)

    def traced_capture(self, hold_attr):
        log = []
        tracing = load_tracing(fake_ttnn(log))

        class Module:
            def __init__(self):
                self.mesh_device = FakeDevice()
                if hold_attr is not None:
                    self.hold_capture_frees = hold_attr

            @tracing.traced_function(
                device=lambda self: self.mesh_device,
                prep_run=False,
                clone_prep_inputs=False,
                hold_capture_frees=lambda self: getattr(self, "hold_capture_frees", False),
            )
            def forward(self, n):
                return None

        Module().forward(3, traced=True)
        return [e for e in log if e[0] in ("hold", "begin")]

    def test_callable_hold_is_per_instance(self):
        self.assertEqual(self.traced_capture(None), [("begin", 0, False)])
        self.assertEqual(self.traced_capture(False), [("begin", 0, False)])
        self.assertEqual(
            self.traced_capture(True), [("hold", 0, True), ("begin", 0, True), ("hold", 0, False)]
        )

    def test_audio_decoders_hold_per_instance(self):
        # The shared audio modules resolve the hold on the instance; none hard-codes it on.
        for path in AUDIO:
            calls = [
                n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "traced_function"
            ]
            self.assertTrue(calls, path.name)
            for c in calls:
                kw = {k.arg: k.value for k in c.keywords}
                self.assertIn("hold_capture_frees", kw, path.name)
                self.assertIsInstance(kw["hold_capture_frees"], ast.Lambda, path.name)
                self.assertIn("getattr(self, 'hold_capture_frees', False)", ast.unparse(kw["hold_capture_frees"]))
            self.assertNotIn(".hold_capture_frees = True", path.read_text(), path.name)

    def test_only_h3_opts_in(self):
        self.assertIn("self.decoder.hold_capture_frees = True", H3_DECODER.read_text())
        ltx_adapter = (ROOT / "models/audio_vae/audio_decoder_ltx.py").read_text()
        self.assertNotIn("hold_capture_frees", ltx_adapter)

if __name__ == "__main__":
    unittest.main()
