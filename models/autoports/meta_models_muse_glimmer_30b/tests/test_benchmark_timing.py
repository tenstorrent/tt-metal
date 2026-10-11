# SPDX-License-Identifier: Apache-2.0
"""Host-only timing contract checks; no TTNN/plugin import or device access.

Run directly with Python to avoid the checkout's hardware-facing conftest.
The fake plugin exposes the inspected standalone plugin's current signatures.
"""

import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Vector:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return copy.deepcopy(self.values)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TimingContract(unittest.TestCase):
    def setUp(self):
        class Output:
            def __init__(self):
                self.completed = False

            def _get_output_impl(self):
                self.completed = True
                return "completed"

        class Controller:
            def __init__(self, runner):
                self.runner = runner

            def submit_async_decode(self, model_input, *, steady_decode_fast_path):
                self.runner.model.decode_forward(
                    start_pos=model_input.input_positions,
                    reload_inputs=model_input.reload_inputs,
                    sampling_params=object(),
                )
                return Output()

        class Runner:
            def __init__(self, model):
                self.model = model
                self.async_decode = Controller(self)

            def execute_model(self, model_input):
                if model_input.prompt_lens is None:
                    return self.async_decode.submit_async_decode(model_input, steady_decode_fast_path=True)
                return self._forward_with_model_input(model_input)

            def _forward_with_model_input(self, model_input):
                return types.SimpleNamespace(model_input=model_input)

            def _finish_front_packed_sync(self, grammar_output, *, fwd):
                return "sync completed"

        class Model:
            batch_size = 32

            def __init__(self):
                self.generator = types.SimpleNamespace(_host_positions=[-1] * 32)

            def decode_forward(self, *, start_pos, reload_inputs, sampling_params):
                positions = start_pos.tolist() if reload_inputs else self.generator._host_positions
                self.generator._host_positions = [p + 1 if p >= 0 else -1 for p in positions]

        self.timing = load("isolated_benchmark_timing", ROOT / "tt/benchmark_timing.py")
        fake_async = types.ModuleType("vllm_tt_plugin.async_decode")
        fake_async.AsyncTTModelRunnerOutput = Output
        fake_async.TTAsyncDecodeController = Controller
        fake_runner = types.ModuleType("vllm_tt_plugin.model_runner")
        fake_runner.TTModelRunner = Runner
        self.model = Model()
        self.runner = Runner(self.model)
        with patch.dict(
            sys.modules,
            {
                "vllm_tt_plugin.async_decode": fake_async,
                "vllm_tt_plugin.model_runner": fake_runner,
            },
        ), patch.dict("os.environ", {"MUSE_PHASE_EVENTS": "/not-written-in-host-test"}), patch.object(
            self.timing, "ThreadingHTTPServer"
        ), patch.object(
            self.timing.threading.Thread, "start"
        ):
            self.timing.install(self.model, {})

    @staticmethod
    def prefill(request="request", *, start=0, end=4096, intermediate=False):
        return types.SimpleNamespace(
            row_req_ids=[request],
            input_positions=Vector([start]),
            prompt_lens=Vector([end]),
            intermediate_prefill_mask=Vector([intermediate]),
        )

    @staticmethod
    def decode(positions, *, request="request", reload=True):
        return types.SimpleNamespace(
            row_req_ids=[request],
            input_positions=Vector(positions),
            prompt_lens=None,
            intermediate_prefill_mask=None,
            reload_inputs=reload,
        )

    def sync(self, mi):
        fwd = self.runner.execute_model(mi)
        self.runner._finish_front_packed_sync(None, fwd=fwd)

    def test_final_prefill_observes_completion_and_copies_metadata(self):
        mi = self.prefill()
        self.sync(mi)
        mi.row_req_ids[0] = "changed"
        mi.input_positions.values[0] = 99
        (event,) = self.timing._EVENTS
        self.assertEqual(event["requests"], ["request"])
        self.assertEqual(event["positions"], [0])
        self.assertTrue(event["device_completion_observed"])
        self.assertGreaterEqual(event["end"], event["start"])

    def test_intermediate_prefill_has_no_fabricated_completion(self):
        self.sync(self.prefill(end=2048, intermediate=True))
        (event,) = self.timing._EVENTS
        self.assertFalse(event["device_completion_observed"])
        self.assertNotIn("end", event)
        self.assertEqual(self.timing._PENDING_PREFILLS, [0])
        self.sync(self.prefill(start=2048))
        self.assertEqual(self.timing._EVENTS[-1]["covers_pending_prefills"], [0])
        self.assertEqual(self.timing._PENDING_PREFILLS, [])

    def test_older_decode_cannot_cover_newer_intermediate_prefill(self):
        older = self.runner.execute_model(self.decode([4096] + [-1] * 31))
        self.sync(self.prefill("new request", end=2048, intermediate=True))
        older._get_output_impl()
        self.assertEqual(self.timing._EVENTS[-1]["covers_pending_prefills"], [])
        self.assertEqual(self.timing._PENDING_PREFILLS, [1])
        newer = self.runner.execute_model(self.decode([4096] + [-1] * 31, reload=False))
        newer._get_output_impl()
        self.assertEqual(self.timing._EVENTS[-1]["unfenced_prefill_at_decode_submission"], [1])
        self.assertEqual(self.timing._EVENTS[-1]["covers_pending_prefills"], [1])

    def test_overlapped_decode_uses_each_pre_replay_host_position(self):
        stale = self.decode([4096] + [-1] * 31)
        first = self.runner.execute_model(stale)
        stale.reload_inputs = False
        second = self.runner.execute_model(stale)
        third = self.runner.execute_model(stale)
        # Resolve out of order to prove the output owns immutable submission data.
        second._get_output_impl()
        first._get_output_impl()
        third._get_output_impl()
        ordered = sorted(self.timing._EVENTS, key=lambda e: e["observation_index"])
        self.assertEqual([e["positions"][0] for e in ordered], [4096, 4097, 4098])
        self.assertEqual([e["scheduler_positions"][0] for e in ordered], [4096] * 3)
        self.assertEqual(self.model.generator._host_positions[0], 4099)
        self.assertTrue(all(e["device_completion_observed"] for e in ordered))


class PhaseEnvelopeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subsets = types.ModuleType("benchmark_stage.subsets")
        subsets.digest = lambda value: "unused"
        work = types.ModuleType("benchmark_work")
        work.work = lambda events: "unused"
        with patch.dict(
            sys.modules,
            {
                "requests": types.ModuleType("requests"),
                "benchmark_stage.subsets": subsets,
                "benchmark_work": work,
            },
        ):
            cls.collector = load("isolated_collect_roofline", ROOT / "tools/collect_roofline.py")

    def test_serial_phases_assign_all_internal_gaps(self):
        result = self.collector.phase_times(
            [
                dict(phase="prefill", start=1.0, end=3.0, device_completion_observed=True),
                dict(phase="decode", start=4.0, end=6.0, device_completion_observed=True),
                dict(phase="decode", start=7.0, end=8.0, device_completion_observed=True),
            ]
        )
        self.assertEqual(result["host_envelope_seconds"], 7.0)
        self.assertEqual(result["separate_phase_seconds"], {"prefill": 2.0, "decode": 5.0})

    def test_crossing_phase_intervals_require_bounded_accounting(self):
        result = self.collector.phase_times(
            [
                dict(phase="decode", start=1.0, end=4.0, device_completion_observed=True),
                dict(phase="prefill", start=3.0, end=7.0, device_completion_observed=True),
            ]
        )
        self.assertIsNone(result["separate_phase_seconds"])
        self.assertEqual(result["host_envelope_seconds"], 6.0)

    def test_unfinished_prefill_uses_later_completion_envelope(self):
        result = self.collector.phase_times(
            [
                dict(phase="prefill", start=1.0, host_output_time=2.0, device_completion_observed=False),
                dict(
                    phase="decode",
                    start=3.0,
                    end=7.0,
                    device_completion_observed=True,
                    unfenced_prefill_at_decode_submission=[0],
                ),
            ]
        )
        self.assertIsNone(result["separate_phase_seconds"])
        self.assertEqual(result["host_envelope_seconds"], 6.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
