# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU lifetime controls executing production encoder methods with a fake Tracer."""
import ast
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[2]
VAE = ROOT / "models/vae/vae_ltx.py"
PIPE = ROOT / "pipelines/ltx/pipeline_ltx_distilled.py"


def methods(path, class_name, names, namespace):
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), "missing production lifetime method"
    source = "from __future__ import annotations\n" + ast.unparse(ast.Module(nodes, type_ignores=[]))
    exec(compile(source, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


class EncoderTraceTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        events = self.events

        class Tracer:
            def __init__(self, fn, **kwargs):
                self.fn, self.trace_captured = fn, False
                self.kwargs = kwargs

            def prepare(self, sample, h, w):
                events.append(("prepare", h, w))
                self.inputs = (sample.clone(), h, w)
                return self.fn(*self.inputs)

            def capture_prepared(self):
                events.append(("capture", *self.inputs[1:]))
                self.trace_captured = True

            def __call__(self, sample, h, w):
                events.append(("replay", h, w))
                return self.fn(sample, h, w)

            def release_trace(self):
                events.append(("release", *self.inputs[1:]))
                self.trace_captured = False

        ns = dict(
            torch=torch,
            Tracer=Tracer,
            logger=Mock(),
            _eager_stages=lambda: {s.strip() for s in os.environ.get("LTX_GEN_EAGER_STAGES", "").split(",")},
            rearrange=lambda x, *a, **kw: x,
            conv_pad_in_channels=lambda x: x,
            conv_pad_height=lambda x, n: (x, x.shape[2]),
            conv_pad_width=lambda x, n: (x, x.shape[3]),
            typed_tensor_2dshard=lambda x, *a, **kw: x,
            fast_device_to_host=lambda x, *a, **kw: x,
            ttnn=SimpleNamespace(ROW_MAJOR_LAYOUT="row", bfloat16="bf16", to_device=lambda x, dev: x),
        )
        cls = type("Encoder", (), methods(VAE, "LTXVideoEncoder", ["forward", "capture_traces", "release_trace"], ns))
        self.enc = cls()
        self.enc.traced, self.enc.capture_open = True, True
        self.enc.patch_size = 1
        self.enc.mesh_device, self.enc.ccl_manager = object(), object()
        self.enc.parallel_config = SimpleNamespace(
            height_parallel=SimpleNamespace(factor=1, mesh_axis=0),
            width_parallel=SimpleNamespace(factor=1, mesh_axis=1),
        )
        self.enc._encode_tracers, self.enc._encode_logical_hw = {}, {}
        self.enc._set_device_gather = lambda on: events.append(("gather", on))

        def encode(x, h, w):
            self.enc._last_logical_hw = (h, w)
            events.append(("encode", h, w))
            return x

        self.enc.encode_device = encode

    def image(self, size=4):
        return torch.arange(3 * size * size).reshape(1, 3, 1, size, size).float()

    def prepare_two(self):
        for size in (4, 8):
            torch.testing.assert_close(self.enc.forward(self.image(size)), self.image(size))
        self.enc.capture_open = False

    def test_all_shapes_prepared_before_capture_and_replay(self):
        self.prepare_two()
        self.assertFalse(any(e[0] == "capture" for e in self.events))
        self.enc.capture_traces()
        kinds = [e[0] for e in self.events]
        self.assertLess(max(i for i, k in enumerate(kinds) if k == "prepare"), kinds.index("capture"))
        self.assertEqual(kinds.count("capture"), 2)
        self.enc.capture_traces()
        self.assertEqual(sum(e[0] == "capture" for e in self.events), 2)
        torch.testing.assert_close(self.enc.forward(self.image()), self.image())
        self.assertIn(("replay", 4, 4), self.events)
        for tracer in self.enc._encode_tracers.values():
            self.assertTrue(tracer.kwargs["prep_run"] and tracer.kwargs["clone_prep_inputs"])

    def test_late_shape_and_uncaptured_shape_stay_eager(self):
        self.prepare_two()
        self.enc.forward(self.image())  # prepared but not captured
        self.enc.capture_traces()
        before = len(self.enc._encode_tracers)
        torch.testing.assert_close(self.enc.forward(self.image(12)), self.image(12))
        self.assertEqual(len(self.enc._encode_tracers), before)
        self.assertNotIn(("prepare", 12, 12), self.events)
        self.assertIn(("encode", 12, 12), self.events)

    def test_encoder_eager_optout_and_explicit_eager(self):
        with patch.dict(os.environ, {"LTX_GEN_EAGER_STAGES": " s1, enc "}):
            self.enc.forward(self.image())
        self.enc.forward(self.image(8), traced=False)
        self.assertEqual(self.enc._encode_tracers, {})
        self.assertFalse(any(e[0] == "prepare" for e in self.events))

    def test_release_clears_both_maps_and_allows_recapture(self):
        self.prepare_two()
        self.enc.capture_traces()
        self.enc.release_trace()
        self.assertEqual(self.enc._encode_tracers, {})
        self.assertEqual(self.enc._encode_logical_hw, {})
        self.assertEqual(sum(e[0] == "release" for e in self.events), 2)
        self.enc.capture_open = True
        self.enc.forward(self.image())
        self.enc.capture_traces()
        self.assertEqual(sum(e[0] == "capture" for e in self.events), 3)

    def test_gather_restored_on_capture_exception_and_interrupt(self):
        self.prepare_two()
        tracer = next(iter(self.enc._encode_tracers.values()))
        for error in (RuntimeError("capture"), KeyboardInterrupt()):
            with patch.object(tracer, "capture_prepared", side_effect=error):
                with self.assertRaises(type(error)) as caught:
                    self.enc.capture_traces()
                self.assertIs(caught.exception, error)
                self.assertEqual(self.events[-1], ("gather", False))

    def test_pipeline_captures_encoder_after_dit_and_gemma(self):
        self.prepare_two()
        capture = methods(
            PIPE, "LTXDistilledPipeline", ["_capture_warmup_traces"], dict(os=os, time=time, logger=Mock())
        )["_capture_warmup_traces"]
        pipe = SimpleNamespace(
            _warmup_capture_pending=[
                dict(
                    stages=("s1", "s2"),
                    kf_anchors=None,
                    i2v_families=False,
                    num_frames=9,
                    height=8,
                    width=8,
                    ref_num_frames=None,
                )
            ],
            _device_resident=lambda _: False,
            _capture_warmup_family=lambda *a, **kw: self.events.append(("dit",)),
            gemma_encoder_pair=SimpleNamespace(
                open_trace_gate=lambda: None, capture_trace=lambda: self.events.append(("gemma",))
            ),
            vae_encoder=self.enc,
        )
        capture(pipe)
        kinds = [e[0] for e in self.events]
        self.assertLess(kinds.index("dit"), kinds.index("gemma"))
        self.assertLess(kinds.index("gemma"), kinds.index("capture"))
        self.assertEqual(pipe._warmup_capture_pending, [])

    def test_early_warmup_prepares_two_shapes_and_honors_optout(self):
        # Execute the production early encoder branch without unrelated DiT warmup.
        tree = ast.parse(PIPE.read_text())
        block = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "self.vae_encoder is not None and warm_encoders"
        )
        code = compile(ast.Module([block], type_ignores=[]), str(PIPE), "exec")
        for eager, expected in (("", [False, False, True, True]), ("enc", [False, False])):
            with self.subTest(eager=eager), patch.dict(os.environ, {"LTX_GEN_EAGER_STAGES": eager}):
                calls = []
                enc = SimpleNamespace(traced=True, capture_open=False)
                pipe = SimpleNamespace(vae_encoder=enc, _warmup_encode=lambda h, w: calls.append(enc.traced))
                exec(
                    code,
                    dict(self=pipe, warm_encoders=True, height=8, width=8, os=os, logger=Mock(), ref_num_frames=None),
                )
                self.assertEqual(calls, expected)
                self.assertFalse(enc.capture_open)


if __name__ == "__main__":
    unittest.main()
