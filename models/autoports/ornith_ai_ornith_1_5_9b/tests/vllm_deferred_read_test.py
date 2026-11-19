# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of actual deferred-output methods, without importing TTNN.

The fake CQ copies a source at read enqueue time. This deliberately exposes
borrowed-output overwrite bugs when a second decode mutates that source.
"""

import ast
import os
import unittest
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from weakref import WeakValueDictionary

import torch

from models.autoports.ornith_ai_ornith_1_5_9b.tt.decode_contract import DecodeCommands, DecodeSubmission


class FakeShard:
    def __init__(self, value, calls, index):
        self.value, self.calls, self.index = value, calls, index

    def cpu(self, *, blocking):
        if blocking:
            raise AssertionError("Output transfer must be nonblocking")
        self.calls.append(("read", self.index))
        return self.value.clone()


class ReachedPreflight(Exception):
    pass


class DeferredReadTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.events = []

        def record_event(device, cq):
            event = object()
            self.events.append(event)
            self.calls.append(("event", event))
            return event

        fake_ttnn = SimpleNamespace(
            get_device_tensors=lambda tensor: tensor.shards,
            record_event=record_event,
            event_synchronize=lambda event: self.calls.append(("sync", event)),
            synchronize_device=lambda device: self.calls.append(("device_sync", device)),
            release_trace=lambda device, trace: self.calls.append(("release", trace)),
            to_torch=lambda tensor: tensor.value.clone() if isinstance(tensor, FakeShard) else tensor,
        )
        source = Path(
            os.environ.get("ORNITH_DEFERRED_READ_SOURCE", Path(__file__).resolve().parents[1] / "tt/generator.py")
        )
        cls = next(
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "OrnithGenerator"
        )
        wanted = {
            "_submit_read",
            "read_decode_output",
            "process_decode_output_host",
            "decode_forward",
            "replay",
            "close",
        }
        cls.bases = []
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {
            "ttnn": fake_ttnn,
            "DecodeSubmission": DecodeSubmission,
            "DecodeCommands": DecodeCommands,
            "Integral": Integral,
        }
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
        self.gen = namespace["OrnithGenerator"]()
        self.gen.device = object()
        self.gen.model = SimpleNamespace(vocab_size=8)
        self.gen.closed = False
        self.gen.next_submission_id = 0
        self.gen.output_handles = WeakValueDictionary()
        self.gen.submissions = {}
        self.gen.counters = {"readbacks": 0, "releases": 0}
        cache = [[object(), object()]]
        self.state = SimpleNamespace(
            tokens=self.tensor(torch.tensor([[[[11, 12]]]])),
            logits=self.tensor(torch.arange(4).reshape(1, 1, 1, 4).expand(1, 1, 2, 4).float()),
            slots=(0, 1),
            cache=cache,
            batch_state=SimpleNamespace(active=(True, True)),
            forward_initialized=True,
            sampling_params_initialized=True,
            sampling_state_initialized=True,
            model_trace=101,
            sample_trace=102,
            model_traces={False: 101},
            sample_traces={(False, False): 102},
            trace_logits={False: object()},
        )
        self.gen.states = {(id(cache), (0, 1)): self.state}

    def tensor(self, values):
        return SimpleNamespace(
            shards=[FakeShard(values.clone() + 100 * index, self.calls, index) for index in range(4)]
        )

    def submit(self, *, logits=False):
        return self.gen._submit_read(self.state, host_sampling=logits)

    def test_forward_output_registration_does_not_read_or_sync(self):
        handle = self.submit()
        self.assertIs(handle.token_tensor, self.state.tokens)
        self.assertEqual(self.gen.submissions[handle.submission_id], (handle, None, None))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.gen.counters["readbacks"], 0)

    def test_token_read_uses_first_shard_and_is_enqueued_once(self):
        handle = self.submit()
        first, events = self.gen.read_decode_output(handle, async_read=True)
        again, repeated_events = self.gen.read_decode_output(handle, async_read=True)
        self.assertIs(first, handle)
        self.assertIs(again, handle)
        self.assertEqual(events, repeated_events)
        self.assertEqual(len(events), 1)
        self.assertEqual([call for call in self.calls if call[0] == "read"], [("read", 0)])
        self.assertFalse(any(call[0] in ("sync", "device_sync") for call in self.calls))
        self.assertEqual(self.gen.counters["readbacks"], 1)

    def test_logits_read_uses_all_shards(self):
        handle = self.submit(logits=True)
        self.assertIs(handle.logits_tensor, self.state.logits)
        self.assertEqual(self.calls, [])
        self.gen.read_decode_output(handle, async_read=True)
        self.assertEqual([call for call in self.calls if call[0] == "read"], [("read", i) for i in range(4)])
        self.assertEqual(self.gen.counters["readbacks"], 1)

    def test_sync_read_waits_after_enqueue_without_duplicate_transfer(self):
        handle = self.submit()
        self.assertIs(self.gen.read_decode_output(handle), handle)
        self.assertEqual([call[0] for call in self.calls], ["read", "event", "sync"])
        self.gen.read_decode_output(handle)
        self.assertEqual(sum(call[0] == "read" for call in self.calls), 1)

    def test_two_queued_reads_preserve_outputs_when_device_source_changes(self):
        first = self.submit()
        self.gen.read_decode_output(first, async_read=True)
        self.state.tokens.shards[0].value += 20
        second = self.submit()
        self.gen.read_decode_output(second, async_read=True)
        self.state.tokens.shards[0].value += 1000
        output2 = self.gen.process_decode_output_host(second, is_tokens=True)
        output1 = self.gen.process_decode_output_host(first, is_tokens=True)
        self.assertEqual(output1.tolist(), [[11], [12]])
        self.assertEqual(output2.tolist(), [[31], [32]])
        self.assertFalse(self.gen.submissions)
        before = list(self.calls)
        self.assertIs(self.gen.process_decode_output_host(first, is_tokens=True), output1)
        self.assertEqual(self.calls, before)
        self.assertEqual(self.gen.read_decode_output(first, async_read=True), (first, []))

    def test_unknown_handle_is_rejected_without_read(self):
        unknown = DecodeSubmission(999, "tokens", 2, (0, 1), 2, token_tensor=self.state.tokens)
        with self.assertRaises((ValueError, RuntimeError)):
            self.gen.read_decode_output(unknown, async_read=True)
        self.assertEqual(self.calls, [])

    def test_closed_generator_rejects_unread_handle(self):
        handle = self.submit()
        self.gen.closed = True
        with self.assertRaises((ValueError, RuntimeError)):
            self.gen.read_decode_output(handle, async_read=True)
        self.assertEqual(self.calls, [])

    def call_next_decode(self):
        def preflight(**kwargs):
            raise ReachedPreflight()

        self.gen._decode_preflight = preflight
        return self.gen.decode_forward(
            object(),
            object(),
            page_table=object(),
            kv_cache=self.state.cache,
            sampling_params=object(),
            read_from_device=False,
            reload_inputs=False,
        )

    def test_unqueued_read_blocks_next_decode_before_preflight_or_writes(self):
        self.submit()
        with self.assertRaises((ValueError, RuntimeError)):
            self.call_next_decode()
        self.assertEqual(self.calls, [])

    def test_queued_read_permits_steady_decode_overlap(self):
        handle = self.submit()
        self.gen.read_decode_output(handle, async_read=True)
        with self.assertRaises(ReachedPreflight):
            self.call_next_decode()
        self.assertFalse(any(call[0] in ("sync", "device_sync") for call in self.calls))

    def replay(self):
        def execute(state, *, host_sampling):
            self.calls.append(("execute", host_sampling))
            state.tokens.shards[0].value.add_(20)

        self.gen._execute = execute
        return self.gen.replay(self.state)

    def test_unqueued_read_blocks_direct_replay_without_execution_or_transfer(self):
        self.submit()
        before = dict(self.gen.counters)
        with self.assertRaisesRegex(ValueError, "Queue deferred"):
            self.replay()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.gen.counters, before)
        self.assertEqual(self.state.tokens.shards[0].value.reshape(-1).tolist(), [11, 12])

    def test_closed_generator_blocks_direct_replay_without_execution_or_transfer(self):
        self.gen.closed = True
        before = dict(self.gen.counters)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.replay()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.gen.counters, before)

    def test_queued_read_permits_direct_replay_and_preserves_previous_output(self):
        handle = self.submit()
        self.gen.read_decode_output(handle, async_read=True)
        self.calls.clear()
        result = self.replay()
        self.assertEqual(result.tolist(), [31, 32])
        self.assertEqual(self.calls, [("execute", False)])
        self.assertEqual(self.gen.counters["readbacks"], 2)
        previous = self.gen.process_decode_output_host(handle, is_tokens=True)
        self.assertEqual(previous.tolist(), [[11], [12]])

    def test_close_drains_unqueued_work_before_releasing_traces(self):
        self.submit()
        self.gen.close()
        self.assertEqual([call[0] for call in self.calls], ["device_sync", "release", "release"])
        self.assertTrue(self.gen.closed)
        self.assertEqual(self.gen.counters["releases"], 2)
        self.assertEqual(self.gen.counters["readbacks"], 0)
        previous = list(self.calls)
        self.gen.close()
        self.assertEqual(previous, self.calls)

    def test_close_waits_for_queued_reads_before_release(self):
        handle = self.submit()
        self.gen.read_decode_output(handle, async_read=True)
        self.gen.close()
        kinds = [call[0] for call in self.calls]
        self.assertIn("sync", kinds)
        self.assertLess(kinds.index("sync"), kinds.index("release"))


if __name__ == "__main__":
    unittest.main()
