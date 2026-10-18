# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact-shape routing and qualified expert arithmetic checks."""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def tree(relative):
    return ast.parse((ROOT / relative).read_text())


class HybridHostTests(unittest.TestCase):
    def test_only_exact1024_uses_helper_and_original_fallback_args_survive(self):
        node = next(
            n for n in tree("tt/prefill_view.py").body if isinstance(n, ast.ClassDef) and n.name == "_PrefillDecoder"
        )
        calls = []

        class Base:
            def prefill_forward(self, value, **kwargs):
                calls.append(("original", value, kwargs))
                return "original"

        def forward1024(self, value, **kwargs):
            calls.append(("hybrid", value, kwargs))
            return "hybrid"

        namespace = dict(MultichipDecoder=Base, forward1024=forward1024)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "production_prefill_dispatch", "exec"), namespace)
        model = namespace["_PrefillDecoder"]()
        model.prefill_hybrid_enabled = True
        model.ple = None
        positions, slots, tokens, pages = (object() for _ in range(4))
        for length in (1, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4096):
            value = SimpleNamespace(shape=(1, 1, length, 2560))
            result = model.prefill_forward(
                value, current_positions=positions, slots=slots, tokens=tokens, page_table=pages
            )
            self.assertEqual(result, "hybrid" if length == 1024 else "original")
            self.assertIs(calls[-1][1], value)
            self.assertIs(calls[-1][2]["slots"], slots)
            self.assertIs(calls[-1][2]["tokens"], tokens)
            self.assertIs(calls[-1][2]["page_table"], pages)
        model.prefill_hybrid_enabled = False
        self.assertEqual(
            model.prefill_forward(SimpleNamespace(shape=(1, 1, 1024, 2560)), current_positions=positions), "original"
        )
        model.prefill_hybrid_enabled = True
        with self.assertRaises(ValueError):
            model.prefill_forward(SimpleNamespace(shape=(1, 1, 1024, 2560)), current_positions=positions)

    def test_non_moe_preparation_never_expands_above512(self):
        node = next(
            n for n in tree("tt/prefill_view.py").body if isinstance(n, ast.FunctionDef) and n.name == "_decoder"
        )
        prefix = copy.deepcopy(node)
        prefix.body = prefix.body[:3] + [
            ast.Return(
                value=ast.Tuple(
                    elts=[ast.Name(id=n, ctx=ast.Load()) for n in ("hybrid_enabled", "lengths", "chunk_size")],
                    ctx=ast.Load(),
                )
            )
        ]
        namespace = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[prefix], type_ignores=[])),
                "production_prepare_prefix",
                "exec",
            ),
            namespace,
        )
        enabled, lengths, size = namespace["_decoder"](SimpleNamespace(batch_size=32), {}, [1, 128, 256, 512, 1024])
        self.assertTrue(enabled)
        self.assertEqual(lengths, [1, 128, 256, 512])
        self.assertEqual(size, 512)
        self.assertFalse(namespace["_decoder"](SimpleNamespace(batch_size=1), {}, [512, 1024])[0])

    def test_group_arithmetic_identical_to_hardware_qualified_component(self):
        baseline = next(
            n
            for n in tree("tests/benchmark_prefill_moe1024_candidate.py").body
            if isinstance(n, ast.FunctionDef) and n.name == "group1024"
        )
        candidate = copy.deepcopy(
            next(
                n for n in tree("tt/prefill_hybrid.py").body if isinstance(n, ast.FunctionDef) and n.name == "group1024"
            )
        )
        self.assertIsInstance(candidate.body.pop(0), ast.If)  # exact1024 public guard
        candidate.body = [
            n
            for n in candidate.body
            if not (
                isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == "hybrid_capacity_rows" for t in n.targets)
            )
        ]
        self.assertEqual(ast.dump(baseline, include_attributes=False), ast.dump(candidate, include_attributes=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
