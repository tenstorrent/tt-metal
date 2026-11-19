# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Actual streaming prefill helper lifetime, without TTNN/device imports."""
import ast
import copy
import unittest
import weakref
from pathlib import Path
from types import MethodType, SimpleNamespace


class RowBuffer:
    def __init__(self, index):
        self.index = index


class PrefillSamplingLifetimeTest(unittest.TestCase):
    def run_stream(self, slots, counts, *, sample=True, retain_slice=False):
        source = Path(__file__).resolve().parents[1] / "tt/generator.py"
        tree = ast.parse(source.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "prefill_forward")
        branch = copy.deepcopy(
            [n for n in method.body if isinstance(n, ast.If) and ast.unparse(n.test) == "return_device"][-1]
        )
        terminal = ast.parse("def assemble(self, slots, sampling_params): pass").body[0]
        terminal.body = [branch]
        write = copy.deepcopy(
            next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_write_prefill_last")
        )
        if retain_slice:
            write.body.insert(-1, ast.parse("self.retained_test_slice = last").body[0])
        row_refs, events = [], []

        class Output:
            def __getitem__(self, key):
                row = RowBuffer(key[2].start)
                row_refs.append(weakref.ref(row))
                events.append(("slice", row.index))
                return row

        persistent = object()
        canvas = {slot: -100 for slot in range(32)}

        def write_row(row, destination, slot):
            self.assertIs(destination, persistent)
            self.assertIsNotNone(row_refs[-1]())
            canvas[slot] = row.index
            events.append(("write", slot))

        def no_live_temporaries(phase):
            if any(ref() is not None for ref in row_refs):
                raise RuntimeError("Temporary row allocation still alive at " + phase)
            events.append((phase,))

        def sample_rows(actual_slots):
            self.assertEqual(actual_slots, slots)
            no_live_temporaries("sampling replay")

        generator = SimpleNamespace(
            prefill_output=Output(), sample_logits=persistent, _sample=sample_rows, _read_tokens=lambda: [111, 222]
        )
        namespace = dict(
            return_device=True,
            sample_slots=slots,
            sample_mask=[True] * len(slots),
            write_token_chunk=write_row,
            torch=SimpleNamespace(tensor=lambda value, **kw: value, int64=object()),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[write, terminal], type_ignores=[])), str(source), "exec"
            ),
            namespace,
        )
        generator._write_prefill_last = MethodType(namespace["_write_prefill_last"], generator)
        for slot, count in zip(slots, counts):
            no_live_temporaries("prefill replay")
            generator._write_prefill_last(count, slot)
        # The next replay must be safe even when there is no sampled return.
        no_live_temporaries("prefill replay")
        result = namespace["assemble"](generator, slots, object() if sample else None)
        self.assertTrue(all(ref() is None for ref in row_refs))
        for slot, count in zip(slots, counts):
            self.assertEqual(canvas[slot], count - 1)
        self.assertTrue(all(canvas[slot] == -100 for slot in range(32) if slot not in slots))
        return result, persistent

    def test_all_six_last_row_layouts_release_before_next_prefill_replay(self):
        for count in (1, 2, 4, 8, 16, 32):
            result, _ = self.run_stream((0,), (count,))
            self.assertEqual(result, [[111]])

    def test_mixed_reordered_slots_release_before_sampling(self):
        result, _ = self.run_stream((1, 0), (1, 32))
        self.assertEqual(result, [[222], [111]])

    def test_logits_only_returns_persistent_destination(self):
        result, persistent = self.run_stream((0,), (4,), sample=False)
        self.assertIs(result, persistent)

    def test_negative_control_detects_retained_slice_before_next_trace(self):
        with self.assertRaisesRegex(RuntimeError, "Temporary row allocation still alive"):
            self.run_stream((0, 1), (2, 32), retain_slice=True)


if __name__ == "__main__":
    unittest.main()
