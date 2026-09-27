# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only completion observer tests; fake plugin, no device imports or waits."""

import contextlib
import io
import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import benchmark_runtime as runtime


class Positions:
    def __init__(self, rows):
        self.rows = rows

    def reshape(self, _):
        return self

    def tolist(self):
        return self.rows


def model_input(positions, lengths=None, reset=False):
    return NS(
        input_positions=Positions(positions), prompt_lens=lengths, reset_batch=reset, perform_device_sampling=True
    )


def set_requests(runner, ids):
    runner.input_batch = NS(req_ids=ids, num_reqs=len(ids))
    runner.requests = {
        rid: NS(prompt_token_ids=[1] * 4, sampling_params=NS(max_tokens=128, temperature=0, top_k=-1)) for rid in ids
    }


class FakeModel:
    def __init__(self):
        self.generator = NS(
            model=NS(
                layers=[object()] * 30,
                config=NS(num_hidden_layers=30),
                layer_indices=list(range(30)),
                max_seq_len=262144,
                precision_summary=lambda: {"weights": "selected"},
            ),
            mesh=NS(shape=(1, 4)),
            tokenizer=NS(init_kwargs={"revision": "pinned"}),
        )
        self.max_batch_size = 32


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.recorder = runtime.Recorder({"test": True})
        self.runner = NS()
        set_requests(self.runner, ["gemma4-a", "gemma4-b"])

    def test_positions_follow_request_ids_across_layout_changes(self):
        r = self.recorder
        sid = r.begin(self.runner, model_input([0, 0], [4, 6]))
        r.complete(sid)
        sid = r.begin(self.runner, model_input([4, 6, -1], reset=True))
        r.complete(sid)
        set_requests(self.runner, ["gemma4-b", "gemma4-a"])
        sid = r.begin(self.runner, model_input([7, 5, -1], reset=True))
        r.complete(sid)
        snap = r.snapshot(True)
        self.assertEqual([e["positions"] for e in snap["events"] if e["event"] == "dispatch"], [[], [4, 6], [7, 5]])
        self.assertEqual(snap["errors"], [])
        self.assertEqual(snap["pending"], [])
        self.assertEqual(snap["requests"]["gemma4-a"]["prompt_tokens"], 4)
        self.assertEqual(snap["events"][-1]["wire_rows"], 3)

    def test_stale_steady_positions_advance_without_device_read(self):
        r = self.recorder
        r.complete(r.begin(self.runner, model_input([0, 0], [4, 6])))
        r.complete(r.begin(self.runner, model_input([4, 6])))
        r.complete(r.begin(self.runner, model_input([4, 6])))
        self.assertEqual(r.events[-1]["positions"], [5, 7])
        self.assertEqual(r.errors, [])

    def test_unobserved_and_mixed_cohorts(self):
        set_requests(self.runner, ["ordinary"])
        self.assertIsNone(self.recorder.begin(self.runner, model_input([0], [4])))
        set_requests(self.runner, ["gemma4-a", "ordinary"])
        self.recorder.begin(self.runner, model_input([0, 0], [4, 4]))
        self.assertIn("mixed observed and unobserved request cohort", self.recorder.errors)

    def test_bad_reset_and_continuation_are_visible(self):
        r = self.recorder
        r.begin(self.runner, model_input([1, 0], [4, 6]))
        r.begin(self.runner, model_input([90, 6], reset=True))
        self.assertEqual(len(r.errors), 2)
        self.assertEqual(len(r.snapshot()["pending"]), 2)

    def test_prefill_start_excludes_previous_async_drain(self):
        self.runner._gemma_benchmark_entry_ns = 100
        self.recorder.last_completion = 200
        self.recorder.begin(self.runner, model_input([0, 0], [4, 6]))
        self.assertEqual(self.recorder.events[-1]["timestamp_ns"], 200)


class InstallationTests(unittest.TestCase):
    def setUp(self):
        class Runner:
            def execute_model(self, value):
                self.execute_calls += 1
                return value

            def _forward_with_model_input(self, value):
                self.forward_calls += 1
                if self.fail_forward:
                    raise RuntimeError("forward failed")
                return NS(input=value)

            def _finish_nondp_sync(self, grammar, *, fwd):
                self.finish_calls += 1
                if self.fail_finish:
                    raise RuntimeError("finish failed")
                return fwd

        class Output:
            def __init__(self, controller):
                self._controller = controller
                self.calls = 0
                self.done = False
                self.fail = False
                self.value = object()

            def _get_output_impl(self):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("read failed")
                return self.value

            def get_output(self):
                if not self.done:
                    self.cached = self._get_output_impl()
                    self.done = True
                return self.cached

        class Controller:
            def __init__(self, runner):
                self.runner = runner
                self.submit_calls = 0

            def submit_async_non_dp_decode(self, value, *, steady_decode_fast_path):
                self.submit_calls += 1
                return Output(self)

        async_module = types.ModuleType("vllm_tt_plugin.async_decode")
        async_module.AsyncTTModelRunnerOutput = Output
        async_module.TTAsyncDecodeController = Controller
        runner_module = types.ModuleType("vllm_tt_plugin.model_runner")
        runner_module.TTModelRunner = Runner
        self.module_patch = patch.dict(
            sys.modules,
            {
                "vllm_tt_plugin": types.ModuleType("vllm_tt_plugin"),
                "vllm_tt_plugin.async_decode": async_module,
                "vllm_tt_plugin.model_runner": runner_module,
            },
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.model = FakeModel()
        with patch.object(runtime, "serve_control") as serving, contextlib.redirect_stdout(io.StringIO()):
            self.recorder = runtime.install(self.model, "/not-created.sock")
        serving.assert_called_once_with(self.recorder, "/not-created.sock")
        self.runner = Runner()
        self.runner.model = self.model
        self.runner.execute_calls = self.runner.forward_calls = self.runner.finish_calls = 0
        self.runner.fail_forward = self.runner.fail_finish = False
        set_requests(self.runner, ["gemma4-a"])
        self.controller = Controller(self.runner)

    def test_sync_return_identity_and_call_counts_preserved(self):
        value = object()
        self.assertIs(self.runner.execute_model(value), value)
        fwd = self.runner._forward_with_model_input(model_input([0], [4]))
        self.assertEqual(len(self.recorder.pending), 1)
        self.assertIs(self.runner._finish_nondp_sync(None, fwd=fwd), fwd)
        self.assertEqual((self.runner.execute_calls, self.runner.forward_calls, self.runner.finish_calls), (1, 1, 1))
        self.assertEqual([e["event"] for e in self.recorder.events], ["dispatch", "completion"])
        self.assertEqual(self.recorder.events[0]["submission_id"], self.recorder.events[1]["submission_id"])

    def test_async_delayed_completion_pairs_original_cohorts(self):
        self.recorder.next_positions["gemma4-a"] = 4
        a = self.controller.submit_async_non_dp_decode(model_input([4]), steady_decode_fast_path=True)
        set_requests(self.runner, ["gemma4-b"])
        b = self.controller.submit_async_non_dp_decode(model_input([20]), steady_decode_fast_path=True)
        self.assertEqual(len(self.recorder.pending), 2)
        self.assertEqual((a.calls, b.calls), (0, 0))
        self.assertIs(a.get_output(), a.value)
        self.assertIs(b.get_output(), b.value)
        self.assertIs(a.get_output(), a.value)
        self.assertEqual((a.calls, b.calls, self.controller.submit_calls), (1, 1, 2))
        self.assertEqual(self.recorder.pending, {})
        finished = [e for e in self.recorder.events if e["event"] == "completion"]
        self.assertEqual([e["request_ids"] for e in finished], [["gemma4-a"], ["gemma4-b"]])
        self.assertEqual([e["positions"] for e in finished], [[4], [20]])

    def test_other_model_not_recorded(self):
        self.runner.model = FakeModel()
        fwd = self.runner._forward_with_model_input(model_input([0], [4]))
        self.runner._finish_nondp_sync(None, fwd=fwd)
        out = self.controller.submit_async_non_dp_decode(model_input([4]), steady_decode_fast_path=True)
        self.assertIs(out.get_output(), out.value)
        self.assertEqual(self.recorder.events, [])

    def test_forward_exception_preserves_uncompleted_dispatch(self):
        self.runner.fail_forward = True
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            self.runner._forward_with_model_input(model_input([0], [4]))
        self.assertEqual(len(self.recorder.pending), 1)
        self.assertEqual(self.runner.forward_calls, 1)

    def test_sync_finalize_exception_preserves_uncompleted_dispatch(self):
        fwd = self.runner._forward_with_model_input(model_input([0], [4]))
        self.runner.fail_finish = True
        with self.assertRaisesRegex(RuntimeError, "finish failed"):
            self.runner._finish_nondp_sync(None, fwd=fwd)
        self.assertEqual(len(self.recorder.pending), 1)
        self.assertEqual(self.runner.finish_calls, 1)

    def test_async_finalize_exception_preserves_uncompleted_dispatch(self):
        out = self.controller.submit_async_non_dp_decode(model_input([4]), steady_decode_fast_path=True)
        out.fail = True
        with self.assertRaisesRegex(RuntimeError, "read failed"):
            out.get_output()
        self.assertEqual(len(self.recorder.pending), 1)
        self.assertEqual(out.calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
