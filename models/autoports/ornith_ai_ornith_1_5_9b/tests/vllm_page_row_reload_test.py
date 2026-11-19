# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only exercise of the production decode table-refresh block."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


class PageRowReloadTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "tt/generator.py"
        cls = next(
            n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "OrnithGenerator"
        )
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "decode_forward")
        block = next(
            n
            for n in forward.body
            if isinstance(n, ast.If) and ast.unparse(n.test) == "commands.reload_inputs or commands.reload_page_table"
        )
        method = ast.parse("def refresh(self, state, tables, commands):\n    pass\n").body[0]
        method.body = [block]
        namespace = {"torch": torch}
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"),
            namespace,
        )
        self.refresh = namespace["refresh"]
        self.calls = []
        self.table = torch.tensor([[0, 1, -1], [3, 4, 5], [-1, -1, -1]], dtype=torch.int32)
        self.state = SimpleNamespace(host_tables=self.table.clone(), tables=[row[None].clone() for row in self.table])

        def copy(value, target, counter, signed=False):
            self.calls.append((value.clone(), counter, signed))
            target.copy_(value)

        self.owner = SimpleNamespace(_copy=copy, _refresh_resident_page_bounds=lambda state: None)

    def call(self, table, *, inputs=False, pages=True):
        self.refresh(self.owner, self.state, table, SimpleNamespace(reload_inputs=inputs, reload_page_table=pages))

    def test_equal_distinct_table_skips_copy_for_both_reload_modes(self):
        for inputs in (False, True):
            self.call(self.table.clone(), inputs=inputs, pages=not inputs)
        self.assertEqual(self.calls, [])

    def test_one_changed_row_updates_exactly_that_device_row(self):
        changed = self.table.clone()
        changed[0, 2] = 7
        self.call(changed)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(torch.equal(self.state.tables[0], changed[0:1]))
        self.assertTrue(torch.equal(self.state.tables[1], self.table[1:2]))
        self.assertTrue(torch.equal(self.state.host_tables, changed))
        self.call(changed.clone())
        self.assertEqual(len(self.calls), 1)
        changed[1, 1] = 8  # Mutating caller storage cannot mutate our snapshot.
        self.assertEqual(int(self.state.host_tables[1, 1]), 4)
        self.call(changed)
        self.assertEqual(len(self.calls), 2)

    def test_new_hole_and_inactive_row_changes_are_not_skipped(self):
        changed = self.table.clone()
        changed[1, 0] = -1
        changed[2, 2] = 10
        self.call(changed, inputs=True, pages=False)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(torch.equal(torch.cat(self.state.tables), changed))

    def test_no_reload_ignores_even_changed_host_table(self):
        self.call(self.table + 20, inputs=False, pages=False)
        self.assertEqual(self.calls, [])
        self.assertTrue(torch.equal(self.state.host_tables, self.table))

    def test_post_remap_snapshot_is_used(self):
        remap = [1, 0, 2]
        self.state.host_tables = self.state.host_tables[remap].clone()
        self.state.tables = [self.state.tables[i] for i in remap]
        self.call(self.table[remap].clone())
        self.assertEqual(self.calls, [])
        changed = self.table[remap].clone()
        changed[1, 2] = 7
        self.call(changed)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(torch.equal(torch.cat(self.state.tables), changed))


if __name__ == "__main__":
    unittest.main()
