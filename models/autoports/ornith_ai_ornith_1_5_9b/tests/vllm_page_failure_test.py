# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU failure injection for persistent decode page mappings; no TTNN import."""

import ast
import os
import unittest
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.ornith_ai_ornith_1_5_9b.tt.decode_contract import DecodeCommands


class PageFailureTests(unittest.TestCase):
    def setUp(self):
        source = Path(os.environ.get("ORNITH_PAGE_FAILURE_SOURCE", Path(__file__).parents[1] / "tt/generator.py"))
        cls = next(
            n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "OrnithGenerator"
        )
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "decode_forward")
        block = next(
            n
            for n in forward.body
            if isinstance(n, ast.If) and ast.unparse(n.test) == "commands.reload_inputs or commands.reload_page_table"
        )
        refresh = ast.parse("def refresh(self, state, tables, commands):\n    pass\n").body[0]
        refresh.body = [block]
        methods = {
            "decode_forward",
            "replay",
            "decode_resident",
            "capture",
            "_execute",
            "_validate_resident_window",
            "_remap_forward_inputs",
            "prepare_decode",
        }
        cls.bases = []
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        namespace = {"torch": torch, "Integral": Integral, "DecodeCommands": DecodeCommands}
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[cls, refresh], type_ignores=[])), str(source), "exec"),
            namespace,
        )
        self.refresh = namespace["refresh"]
        self.namespace = namespace
        self.gen = namespace["OrnithGenerator"]()
        self.calls = []
        self.table = torch.tensor([[0, 1, -1], [3, 4, 5]], dtype=torch.int32)
        self.cache = [[object(), object()]]
        self.state = SimpleNamespace(
            host_tables=self.table.clone(),
            tables=[row[None].clone() for row in self.table],
            page_tables_valid=True,
            model_traces={False: 1},
            host_positions=(10, -1),
            resident_ready=True,
            position_advances=0,
            cache=self.cache,
        )
        self.gen.closed = False
        self.gen.submissions = {}
        self.gen.states = {(id(self.cache), (0, 1)): self.state}
        self.gen.cache = self.cache
        self.gen.max_seq_len = 96
        self.gen._copy = self.copy
        self.gen._refresh_resident_page_bounds = lambda state: self.calls.append("bounds")
        self.gen._decode_preflight = lambda **kwargs: self.fail("poisoned decode reached preflight")

    def copy(self, value, target, counter, signed=False):
        self.calls.append("copy")
        target.copy_(value)

    def call_refresh(self, table):
        self.refresh(self.gen, self.state, table, SimpleNamespace(reload_inputs=False, reload_page_table=True))

    def poison_with_partial_failure(self):
        changed = self.table.clone()
        changed[0, 2], changed[1, 1] = 7, 8
        attempts = 0

        def failing_copy(value, target, counter, signed=False):
            nonlocal attempts
            attempts += 1
            # Deliberately mutate even the failing destination: its contents are unknown.
            self.copy(value, target, counter, signed)
            if attempts == 2:
                raise RuntimeError("injected copy failure")

        self.gen._copy = failing_copy
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.call_refresh(changed)
        self.gen._copy = self.copy
        return changed

    def test_partial_refresh_poison_survives_old_and_new_table_retries(self):
        changed = self.poison_with_partial_failure()
        self.assertFalse(self.state.page_tables_valid)
        self.assertTrue(torch.equal(self.state.host_tables, self.table))
        self.assertFalse(torch.equal(torch.cat(self.state.tables), self.table))
        self.calls.clear()
        for table in (self.table, changed):
            with self.assertRaisesRegex(RuntimeError, "recreate the generator"):
                self.call_refresh(table)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.state.page_tables_valid)

    def test_success_and_unchanged_refresh_preserve_copy_elision(self):
        self.call_refresh(self.table.clone())
        self.assertEqual(self.calls, ["bounds"])
        changed = self.table.clone()
        changed[0, 2] = 7
        self.calls.clear()
        self.call_refresh(changed)
        self.assertEqual(self.calls, ["copy", "bounds"])
        self.assertTrue(self.state.page_tables_valid)
        self.assertTrue(torch.equal(torch.cat(self.state.tables), changed))

    def test_failed_metadata_commit_also_poisons_state(self):
        def fail_bounds(state):
            raise RuntimeError("injected bounds failure")

        self.gen._refresh_resident_page_bounds = fail_bounds
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.call_refresh(self.table.clone())
        self.assertFalse(self.state.page_tables_valid)

    def test_public_decode_rejects_poison_before_validation_or_writes(self):
        self.poison_with_partial_failure()
        self.calls.clear()
        for reload_inputs in (False, True):
            with self.assertRaisesRegex(RuntimeError, "recreate the generator"):
                self.gen.decode_forward(
                    object(),
                    object(),
                    page_table=object(),
                    kv_cache=self.cache,
                    reload_inputs=reload_inputs,
                    reload_page_table=not reload_inputs,
                )
        self.assertEqual(self.calls, [])

    def test_replay_capture_resident_and_prepare_reject_before_device_calls(self):
        self.poison_with_partial_failure()
        self.calls.clear()
        operations = (
            lambda: self.gen.replay(self.state),
            lambda: self.gen.capture(self.state),
            lambda: self.gen.decode_resident(1),
            lambda: self.gen._execute(self.state),
            lambda: self.gen._validate_resident_window(self.state, 1),
            lambda: self.gen.prepare_decode(kv_cache=self.cache, page_table=self.table, slots=(0, 1)),
        )
        for operation in operations:
            with self.assertRaisesRegex(RuntimeError, "recreate the generator"):
                operation()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.gen.closed)  # Normal close/drain/release remains available.

    def test_failed_page_remap_also_poisons_state(self):
        self.state.tokens = object()
        self.state.token_remap_indices = object()
        self.state.token_slots = SimpleNamespace(remap=lambda index: None)
        self.gen.sampling_state = SimpleNamespace(_copy_host=lambda *args: None)
        self.state.positions = [torch.tensor([0]), torch.tensor([1])]
        self.state.position_snapshots = [value.clone() for value in self.state.positions]
        self.state.table_snapshots = [row.clone() for row in self.state.tables]
        attempts = 0

        def copy(source, target):
            nonlocal attempts
            attempts += 1
            target.copy_(source)
            # Four position copies, two page snapshots, first replacement, then failed second replacement.
            if attempts == 8:
                raise RuntimeError("injected remap copy failure")

        self.namespace["ttnn"] = SimpleNamespace(copy=copy)
        with self.assertRaisesRegex(RuntimeError, "injected remap"):
            self.gen._remap_forward_inputs(self.state, (1, 0))
        self.assertFalse(self.state.page_tables_valid)
        with self.assertRaisesRegex(RuntimeError, "recreate the generator"):
            self.gen._remap_forward_inputs(self.state, (1, 0))
        self.assertEqual(attempts, 8)


if __name__ == "__main__":
    unittest.main()
