# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host admission controls for fixed-step resident decode; no TTNN import."""

import ast
import unittest
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

import torch


class ResidentTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "tt/generator.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OrnithGenerator")
        cls.bases = []
        cls.body = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name
            in (
                "decode_resident",
                "_invalidate_resident",
                "_validate_resident_window",
                "_decode_preflight",
                "_refresh_resident_page_bounds",
                "_remap_forward_inputs",
            )
        ]
        ns = {"Integral": Integral}
        exec(
            compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns
        )  # noqa: S102 -- trusted local source, no TTNN import
        self.gen = ns["OrnithGenerator"]()
        self.gen.cache = [[object(), object()]]
        self.state = SimpleNamespace(
            cache=self.gen.cache,
            model_traces={False: 1},
            host_positions=(10, -1),
            host_tables=torch.tensor([[0], [-1]], dtype=torch.int32),
            position_advances=2,
            resident_ready=True,
            tokens=object(),
        )
        self.gen._refresh_resident_page_bounds(self.state)
        self.gen.states = {(id(self.gen.cache), (0, 1)): self.state}
        self.gen.closed, self.gen.submissions, self.gen.max_seq_len = False, {}, 16
        self.calls = []

        def execute(state, *, host_sampling):
            self.calls.append((state, host_sampling))
            state.position_advances += 1

        self.gen._execute = execute

    def test_exact_context_boundary_and_inactive_rows(self):
        self.assertIs(self.gen.decode_resident(4), self.state.tokens)
        self.assertEqual(len(self.calls), 4)
        with self.assertRaises(ValueError):
            self.gen.decode_resident(1)
        self.assertEqual(len(self.calls), 4)

    def test_reject_before_mutation(self):
        for steps in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.gen.decode_resident(steps)
        with self.assertRaises(ValueError):
            self.gen.decode_resident(5)
        self.state.resident_ready = False
        with self.assertRaises(ValueError):
            self.gen.decode_resident(1)
        self.assertEqual(self.calls, [])

    def test_reset_invalidates_only_matching_cache(self):
        other = [[object(), object()]]
        peer = SimpleNamespace(cache=other, resident_ready=True)
        self.gen.states[(id(other), (0, 1))] = peer
        self.gen._invalidate_resident(self.gen.cache)
        self.assertFalse(self.state.resident_ready)
        self.assertTrue(peer.resident_ready)
        with self.assertRaises(ValueError):
            self.gen.decode_resident(1)

    def test_pending_submission_and_closed_lifetime(self):
        self.gen.submissions[0] = (SimpleNamespace(completion=None), None, None)
        with self.assertRaises(ValueError):
            self.gen.decode_resident(1)
        self.gen.submissions.clear()
        self.gen.closed = True
        with self.assertRaises(RuntimeError):
            self.gen.decode_resident(1)
        self.assertEqual(self.calls, [])

    def test_external_cache_must_have_prepared_state(self):
        with self.assertRaises(ValueError):
            self.gen.decode_resident(1, kv_cache=[[object(), object()]])
        self.assertEqual(self.calls, [])

    def sparse_state(self):
        self.gen.max_seq_len = 128
        self.state.host_positions = (31, -1)
        self.state.position_advances = 0
        self.state.host_tables = torch.tensor([[0, -1, 2, 3], [-1, -1, -1, -1]], dtype=torch.int32)
        self.gen._refresh_resident_page_bounds(self.state)

    def test_last_mapped_then_first_unmapped(self):
        self.sparse_state()
        self.gen.decode_resident(1)
        self.assertEqual(self.state.position_advances, 1)
        with self.assertRaisesRegex(ValueError, "causal page"):
            self.gen.decode_resident(1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.state.position_advances, 1)

    def test_entire_window_rejected_before_any_submission(self):
        self.sparse_state()
        for steps in (2, 34, 65):
            with self.assertRaisesRegex(ValueError, "causal page"):
                self.gen.decode_resident(steps)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.state.position_advances, 0)

    def test_changed_table_and_inactive_unmapped_rows(self):
        self.sparse_state()
        self.state.host_tables[0, 1] = 1
        self.gen._refresh_resident_page_bounds(self.state)
        self.gen.decode_resident(65)
        self.assertEqual(len(self.calls), 65)
        self.assertEqual(self.state.host_positions[1], -1)

    def test_ordinary_resident_preflight_checks_pages_and_context(self):
        self.sparse_state()
        self.gen.model = SimpleNamespace(layers=[SimpleNamespace(kind="linear_attention")])
        commands = SimpleNamespace(
            reload_inputs=False,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
            slot_remap=None,
        )

        def preflight():
            return self.gen._decode_preflight(
                commands=commands,
                previous=self.state,
                tokens=None,
                start_pos=None,
                page_table=None,
                kv_cache=self.gen.cache,
                capacity=2,
                sampling_params=None,
                prompt_tokens=None,
                output_tokens=None,
            )

        preflight()  # Position 31 is legal, despite inactive unmapped row.
        self.state.position_advances = 1
        with self.assertRaisesRegex(ValueError, "causal page"):
            preflight()
        self.state.host_tables[0, 1] = 1
        self.gen._refresh_resident_page_bounds(self.state)
        preflight()
        self.state.position_advances = 97
        with self.assertRaisesRegex(ValueError, "context"):
            preflight()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.state.position_advances, 97)

    def test_unchanged_table_does_not_touch_tensor_on_replay(self):
        self.sparse_state()
        self.state.host_tables = object()  # Scalar cached bound is sufficient.
        self.gen.decode_resident(1)
        with self.assertRaisesRegex(ValueError, "causal page"):
            self.gen.decode_resident(1)
        self.assertEqual(len(self.calls), 1)

    def test_changed_mapping_shrinks_or_extends_cached_window(self):
        self.sparse_state()
        for row, bound in (([0, 1, 2, 3], 4), ([0, -1, 2, 3], 1), ([-1, 1, 2, 3], 0)):
            self.state.host_tables = torch.tensor([row, [-1] * 4], dtype=torch.int32)
            self.gen._refresh_resident_page_bounds(self.state)
            self.assertEqual(self.state.mapped_prefix_pages, (bound, 0))
            before = (self.state.position_advances, len(self.calls))
            if bound < 2:
                with self.assertRaisesRegex(ValueError, "causal page"):
                    self.gen.decode_resident(2)
                self.assertEqual(before, (self.state.position_advances, len(self.calls)))
            else:
                self.gen._validate_resident_window(self.state, 2)

    def test_cached_admission_matches_full_prefix_scan(self):
        import random

        rng = random.Random(45)
        self.gen.max_seq_len = 128
        for _ in range(200):
            rows = [[rng.choice([-1, 0, 3, 17]) for _ in range(4)] for _ in range(2)]
            self.state.host_tables = torch.tensor(rows, dtype=torch.int32)
            self.state.host_positions = (rng.randrange(128), rng.choice([-1, rng.randrange(128)]))
            self.state.position_advances = rng.randrange(5)
            self.gen._refresh_resident_page_bounds(self.state)
            steps = rng.randrange(1, 40)
            valid = all(
                pos < 0
                or (
                    pos + self.state.position_advances + steps - 1 < 128
                    and all(
                        page >= 0 for page in rows[slot][: (pos + self.state.position_advances + steps - 1) // 32 + 1]
                    )
                )
                for slot, pos in enumerate(self.state.host_positions)
            )
            if valid:
                self.gen._validate_resident_window(self.state, steps)
            else:
                with self.assertRaises(ValueError):
                    self.gen._validate_resident_window(self.state, steps)
        self.assertEqual(self.calls, [])

    def test_remap_permutes_cached_bounds_and_positions_together(self):
        self.sparse_state()
        self.state.host_tables = torch.tensor([[0, 1, 2, 3], [4, -1, 6, 7]], dtype=torch.int32)
        self.state.host_positions = (32, 31)
        self.gen._refresh_resident_page_bounds(self.state)
        self.state.positions = [torch.tensor([32]), torch.tensor([31])]
        self.state.position_snapshots = [torch.zeros(1, dtype=torch.int64) for _ in range(2)]
        self.state.tables = list(self.state.host_tables.clone())
        self.state.table_snapshots = [torch.zeros(4, dtype=torch.int32) for _ in range(2)]
        self.state.token_remap_indices = object()
        self.state.token_slots = SimpleNamespace(remap=lambda indices: None)
        self.state.batch_state = SimpleNamespace(set_active=lambda active: None)
        self.gen.sampling_state = SimpleNamespace(_copy_host=lambda source, target: None)
        globals_ = type(self.gen)._remap_forward_inputs.__globals__
        globals_["ttnn"] = SimpleNamespace(copy=lambda source, target: target.copy_(source))
        self.gen._remap_forward_inputs(self.state, (1, 0))
        self.assertEqual(self.state.host_positions, (31, 32))
        self.assertEqual(self.state.mapped_prefix_pages, (1, 4))
        self.gen._validate_resident_window(self.state, 1)
        with self.assertRaisesRegex(ValueError, "causal page"):
            self.gen.decode_resident(2)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
