# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU encoder lifetime controls, including real Tracer transitions with fake native transfers."""
import ast
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from typing import overload

from test_ltx_audio_trace_control import definitions

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
                enc = SimpleNamespace(traced=True, capture_open=False, _encode_tracers={})
                pipe = SimpleNamespace(vae_encoder=enc, _warmup_encode=lambda h, w: calls.append(enc.traced))
                exec(
                    code,
                    dict(self=pipe, warm_encoders=True, height=8, width=8, os=os, logger=Mock(), ref_num_frames=None),
                )
                self.assertEqual(calls, expected)
                self.assertFalse(enc.capture_open)


class EncoderLifecycleTests(unittest.TestCase):
    """Real Tracer state transitions; only native tensor transfers/capture are fakes."""

    image = EncoderTraceTests.image

    def setUp(self):
        EncoderTraceTests.setUp(self)
        self.created = []
        self.live = {}
        self.active = None
        self.serial = 0
        native = SimpleNamespace(Tensor=torch.Tensor, clone=lambda x: x.clone())

        def begin(*args, **kwargs):
            self.serial += 1
            self.active = self.serial
            self.live[self.active] = []
            return self.active

        native.begin_trace_capture = begin
        native.end_trace_capture = lambda *a, **kw: setattr(self, "active", None)
        native.release_trace = lambda dev, tid: self.live.pop(tid)
        native.execute_trace = lambda dev, tid, **kw: [op() for op in self.live[tid]]
        ns = definitions(
            ROOT / "utils/tracing.py",
            {"Tracer", "_tree_map", "_verify_value", "_clone_tensor"},
            dict(ttnn=native, overload=overload, logger=Mock(), _kernel_prewarm_capturing=False,
                 _TRACER_VALID_INPUT_TYPES=(torch.Tensor, int, float, str, bool, type(None))),
        )
        self.tracer_type = ns["Tracer"]
        # Native transfer/copy boundaries only; prepare/staging/capture/replay/release are production.
        self.tracer_type._tensor_to_device = lambda t, value, **kw: value.clone() if torch.is_tensor(value) else value
        self.tracer_type._update_input = lambda t, prev, new, **kw: prev.copy_(new) if torch.is_tensor(prev) else None
        self.enc.mesh_device = SimpleNamespace(id=lambda: 1)

        def factory(*args, **kwargs):
            tracer = self.tracer_type(*args, **kwargs)
            self.created.append(tracer)
            return tracer

        self.enc.forward.__globals__["Tracer"] = factory

        def encode(x, h, w):
            self.enc._last_logical_hw = (h, w)
            self.events.append(("encode", h, w))
            out = x.clone()
            if self.active is not None:
                self.live[self.active].append(lambda: out.copy_(x))
            return out

        self.enc.encode_device = encode
        self.warmup = next(
            n for n in ast.walk(ast.parse(PIPE.read_text()))
            if isinstance(n, ast.If) and ast.unparse(n.test) == "self.vae_encoder is not None and warm_encoders"
        )
        release = methods(
            ROOT / "pipelines/ltx/pipeline_ltx.py", "LTXPipeline", ["release_traces"],
            dict(StateTensor=Mock()),
        )["release_traces"]
        self.pipe = SimpleNamespace(
            vae_encoder=self.enc, vae_decoder=Mock(), _audio_adapter=Mock(),
            transformer=None, _trace_state={},
            _warmup_encode=lambda h, w: self.enc.forward(self.image(h)),
        )
        self.pipe.release_traces = lambda: release(self.pipe)

    def warm(self):
        exec(compile(ast.Module([self.warmup], type_ignores=[]), str(PIPE), "exec"),
             dict(self=self.pipe, warm_encoders=True, height=8, width=8, os=os, logger=Mock(), ref_num_frames=None))

    def assert_released(self, tracers):
        for tracer in tracers:
            self.assertFalse(tracer._prepared)
            self.assertFalse(tracer.trace_captured)
            self.assertEqual(tracer.inputs, {})
            self.assertIsNone(tracer._outputs)

    def check_reset(self, captured):
        self.warm()
        if captured:
            self.enc.capture_traces()
        old = list(self.created)
        self.pipe.release_traces()
        self.pipe.release_traces()
        self.assertEqual(self.enc._encode_tracers, {})
        self.assertEqual(self.enc._encode_logical_hw, {})
        self.assertEqual(self.live, {})
        self.assert_released(old)
        self.assertEqual(self.pipe.vae_decoder.release_trace.call_count, 2)
        self.assertEqual(self.pipe._audio_adapter.release_traces.call_count, 2)
        self.warm()
        self.enc.capture_traces()
        torch.testing.assert_close(self.enc.forward(self.image() + 3), self.image() + 3)
        self.assertEqual(sum(t.replay_count for t in self.created), 1)
        self.assertEqual(len(self.created), 4)

    def test_pipeline_cleanup_prepared_only(self):
        self.check_reset(False)

    def test_pipeline_cleanup_captured(self):
        self.check_reset(True)

    def test_capture_only_prewarm_then_normal_warmup(self):
        # Execute the real cold-start reset statements, following the capture-only prep pass.
        self.warm()
        old = list(self.created)
        tree = ast.parse((ROOT / "pipelines/ltx/pipeline_ltx.py").read_text())
        reset = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr)
                     and ast.unparse(n) == "self.release_traces()")
        exec(compile(ast.Module([reset], type_ignores=[]), "cold_reset", "exec"), dict(self=self.pipe))
        self.assert_released(old)
        self.warm()
        self.enc.capture_traces()
        self.assertEqual(len(self.created), 4)
        self.assertTrue(all(t.trace_captured for t in self.created[2:]))

    def check_prepare_failure(self, error_type, phase, release_error=False):
        error = error_type("primary preparation failure")
        original = self.enc.encode_device
        # Use the actual failed prepare transition, including staged inputs.
        def fail(*args):
            raise error
        self.enc.encode_device = fail
        if phase == "stage":
            original_stage = self.tracer_type._stage_inputs
            def stage_then_fail(tracer, *args):
                original_stage(tracer, *args)
                raise error
            fault = patch.object(self.tracer_type, "_stage_inputs", stage_then_fail)
        else:
            fault = patch.object(self.enc, "encode_device", fail)
        original_release = self.tracer_type.release_trace
        def release(tracer):
            original_release(tracer)
            if release_error:
                raise RuntimeError("secondary cleanup failure")
        with fault, patch.object(self.tracer_type, "release_trace", release):
            with self.assertRaises(error_type) as caught:
                self.enc.forward(self.image())
        self.assertIs(caught.exception, error)
        self.assertEqual(self.enc._encode_tracers, {})
        self.assertEqual(self.enc._encode_logical_hw, {})
        self.assert_released(self.created)
        self.assertEqual(self.events[-1], ("gather", False))
        self.enc.encode_device = original
        self.enc.forward(self.image())
        self.enc.capture_traces()
        torch.testing.assert_close(self.enc.forward(self.image() + 2), self.image() + 2)
        self.assertEqual(self.created[-1].replay_count, 1)

    def check_warmup_failure(self, error_type, shape, fail_eager=False, release_error=False):
        # Existing valid state is unrelated to this two-shape transaction and must survive.
        self.enc.forward(self.image(12))
        self.enc.capture_traces()
        existing = dict(self.enc._encode_tracers)
        logical = dict(self.enc._encode_logical_hw)
        start = len(self.created)
        original = self.enc.encode_device
        error = error_type("warmup failure")
        def fail(x, h, w):
            if h == shape and self.enc.traced != fail_eager:
                raise error
            return original(x, h, w)
        original_release = self.tracer_type.release_trace
        def release(tracer):
            original_release(tracer)
            if release_error:
                raise RuntimeError("secondary warmup cleanup failure")
        with patch.object(self.enc, "encode_device", fail), patch.object(self.tracer_type, "release_trace", release):
            with self.assertRaises(error_type) as caught:
                self.warm()
        self.assertIs(caught.exception, error)
        self.assertFalse(self.enc.capture_open)
        self.assertEqual(self.enc._encode_tracers, existing)
        self.assertEqual(self.enc._encode_logical_hw, logical)
        self.assert_released(self.created[start:])
        self.assertEqual(len(self.live), 1)
        # Late shapes remain eager even after interruption.
        count = len(self.created)
        self.enc.forward(self.image(16))
        self.assertEqual(len(self.created), count)
        self.warm()
        self.enc.capture_traces()
        for size in (4, 8, 12):
            torch.testing.assert_close(self.enc.forward(self.image(size) + 1), self.image(size) + 1)
        self.assertEqual(len(self.enc._encode_tracers), 3)

    def test_warmup_rollback_cleanup_failure_preserves_runtime_error(self):
        self.check_warmup_failure(RuntimeError, 8, release_error=True)

    def test_warmup_rollback_cleanup_failure_preserves_interrupt(self):
        self.check_warmup_failure(KeyboardInterrupt, 8, release_error=True)


def _preparation_case(error_type, phase, release_error):
    def test(self):
        self.check_prepare_failure(error_type, phase, release_error)
    return test


def _warmup_case(error_type, shape, fail_eager):
    def test(self):
        self.check_warmup_failure(error_type, shape, fail_eager)
    return test


for _error in (RuntimeError, KeyboardInterrupt):
    for _phase in ("stage", "prep"):
        for _secondary in (False, True):
            setattr(EncoderLifecycleTests, f"test_prepare_{_error.__name__}_{_phase}_cleanup_fault_{_secondary}",
                    _preparation_case(_error, _phase, _secondary))
    for _shape in (4, 8):
        for _eager in (False, True):
            setattr(EncoderLifecycleTests, f"test_warmup_{_error.__name__}_{_shape}_eager_{_eager}",
                    _warmup_case(_error, _shape, _eager))


if __name__ == "__main__":
    unittest.main()
