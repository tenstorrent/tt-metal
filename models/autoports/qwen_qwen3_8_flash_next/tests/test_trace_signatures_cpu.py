# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""AST-extracted production trace selection; no TTNN import or devices."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


class TraceSignatureTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "tt/generator.py"
        owner = next(
            n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "QwenGenerator"
        )
        names = ("_capture", "_sample", "_decode_body", "_restore_inactive_tokens")
        nodes = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.bodies, self.active_calls, self.decode_calls = {}, [], []
        self.replaying, self.active_slots, self.active_uploads = False, (), 0
        self.fills, self.row_writes = [], []

        def record(body, signature=None):
            key = len(self.bodies) + 1
            self.bodies[key] = body
            body()  # Capture executes Python once; replay executes device work only.
            return key

        def replay(mesh, key, **kw):
            self.replaying = True
            try:
                self.bodies[key]()
            finally:
                self.replaying = False

        def fill(source, value, *, optional_tensor):
            self.assertIs(source, optional_tensor)
            self.fills.append(value)
            return optional_tensor.fill_(value)

        def write_row(source, destination, offset):
            self.row_writes.append(offset)
            destination[:, :, offset : offset + source.shape[-2]].copy_(source)

        fake = SimpleNamespace(
            reshape=torch.reshape,
            concat=lambda xs, dim: torch.cat(xs, dim=dim),
            copy=lambda src, dst: dst.copy_(src),
            full_like=fill,
            execute_trace=replay,
        )
        ns = {"ttnn": fake, "write_token_chunk": write_row}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
        cls = type("ActualTraceMethods", (), {name: ns[name] for name in names})
        self.g = g = cls()
        g.mesh = object()
        g.max_batch_size = 1
        g.prefill_checkpoints = SimpleNamespace(invalidate=lambda slots: None)
        g.traces, g.sampling_traces = {}, {}
        g.counters = {"captures": 0, "sampling_replays": 0}
        g._record_trace = record
        g.decode_tokens = torch.arange(32).reshape(1, 1, 1, 32).float()
        g.saved_tokens = torch.zeros_like(g.decode_tokens)
        g.word_suffix = torch.zeros(1, 1, 15)
        g.sample_logits = torch.zeros(1, 1, 32, 4)
        g.inactive_logits = torch.full((1, 1, 1, 4), -100.0)
        g.caches = list(range(32))

        def decode(tokens, words, *, caches):
            self.decode_calls.append(tuple(caches))
            return tuple(torch.full((1, 1, 1, 4), float(slot + 10)) for slot in caches)

        g.model = SimpleNamespace(decode_batch=decode)

        def set_active(slots):
            # A recorded trace does not execute this Python scheduler method.
            if self.replaying:
                return
            slots = tuple(sorted(slots))
            self.active_calls.append(slots)
            if slots != self.active_slots:
                self.active_uploads += 1
                self.active_slots = slots

        def restore():
            inactive = [i for i in range(32) if i not in self.active_slots]
            g.decode_tokens.flatten()[inactive] = g.saved_tokens.flatten()[inactive]

        g.seed_state = SimpleNamespace(set_active_slots=set_active, forward=lambda: None)
        g.token_feedback = SimpleNamespace(forward=restore)
        g.seed_skip = torch.zeros(32)

        def sample(logits, *, penalties_on, grammar_on, tt_out_tok):
            tt_out_tok.copy_(torch.arange(32).reshape(1, 1, 1, 32) + 100)

        g.sampler = SimpleNamespace(
            _penalties_active=False,
            _log_probs_active=False,
            tt_sampling=SimpleNamespace(force_argmax_sampling=False, seeds_tt_tensor=torch.ones(32)),
            _run_sampling=sample,
        )

    def test_standalone_b1_integer_capture_reuses_and_rejects_batched_bypass(self):
        first = self.g._capture((0,))
        self.assertEqual(first, self.g._capture((0,)))
        self.assertEqual(list(self.g.traces), [(0,)])
        self.assertEqual(self.g.counters["captures"], 1)
        self.bodies[first]()
        self.assertTrue(torch.equal(self.g.sample_logits[0, 0, 0], torch.full((4,), 10.0)))
        with self.assertRaisesRegex(ValueError, "_replay_decode"):
            self.g._capture((1,))
        self.g.max_batch_size = 2
        with self.assertRaisesRegex(ValueError, "_replay_decode"):
            self.g._capture((0,))

    def test_sampler_permutation_reuses_trace_and_preserves_inactive_tokens(self):
        self.g._sample((7, 2))
        first = self.g.decode_tokens.clone()
        self.g._sample((2, 7))
        self.assertEqual(list(self.g.sampling_traces), [(False, False)])
        self.assertEqual(self.g.counters, {"captures": 1, "sampling_replays": 2})
        self.assertTrue(torch.equal(first, self.g.decode_tokens))
        expected = torch.arange(32).float()
        expected[[2, 7]] += 100
        self.assertTrue(torch.equal(self.g.decode_tokens.flatten(), expected))
        self.g._sample((2, 8))
        self.assertEqual(len(self.g.sampling_traces), 1)
        expected[8] = 108
        self.assertTrue(torch.equal(self.g.decode_tokens.flatten(), expected))
        self.assertEqual(self.active_uploads, 2)
        self.g._sample((8, 2))
        self.assertEqual(self.active_uploads, 2, "Same membership must not upload on each token")
        self.g._sample(())
        self.assertTrue(torch.equal(self.g.decode_tokens.flatten(), expected))
        self.assertEqual(len(self.g.sampling_traces), 1)
        self.g.sampler._penalties_active = True
        self.g._sample((31,))
        self.assertEqual(set(self.g.sampling_traces), {(False, False), (True, False)})
        self.g.sampler._penalties_active = False
        self.g._sample((0,))
        self.assertEqual(len(self.g.sampling_traces), 2, "Retained semantic config should reuse its original trace")

    def test_row_assembly_crossover_preserves_exact_physical_rows_and_clears_poison(self):
        for slots in ((31,), (7, 0), tuple(range(24)), tuple(range(25)), tuple(range(32)), (2,)):
            with self.subTest(slots=slots):
                self.g.sample_logits[:, :, ::2] = float("nan")
                self.g.sample_logits[:, :, 1::2] = float("inf")
                self.fills.clear()
                self.row_writes.clear()
                self.g._decode_body(slots)
                expected = torch.full_like(self.g.sample_logits, -100.0)
                for slot in slots:
                    expected[:, :, slot] = slot + 10
                self.assertTrue(torch.equal(expected, self.g.sample_logits))
                self.assertEqual(self.fills, [-100.0] if len(slots) <= 24 else [])
                self.assertEqual(self.row_writes, list(slots) if len(slots) <= 24 else [])


if __name__ == "__main__":
    unittest.main()
