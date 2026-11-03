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

        for groups in ([1024], [1024, 2048]):  # default, and QWEN_PREFILL_MOE_GROUP=2048
            namespace = dict(MultichipDecoder=Base, forward1024=forward1024, moe_group_sizes=lambda: groups)
            exec(compile(ast.Module(body=[node], type_ignores=[]), "production_prefill_dispatch", "exec"), namespace)
            model = namespace["_PrefillDecoder"]()
            model.prefill_hybrid_enabled = True
            model.ple = None
            positions, slots, tokens, pages = (object() for _ in range(4))
            for length in (1, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2048, 4096):
                value = SimpleNamespace(shape=(1, 1, length, 2560))
                result = model.prefill_forward(
                    value, current_positions=positions, slots=slots, tokens=tokens, page_table=pages
                )
                self.assertEqual(result, "hybrid" if length in groups else "original")
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
        # Production adds bit-identical data-movement variants (pipelined dispatch/compaction, packed
        # capacity; toggles in prefill_moe_compact.py, verified on device). The arithmetic must stay the
        # qualified component's: the same ttnn compute calls in the same order.
        arithmetic = {"linear", "topk", "softmax", "add", "multiply", "sum", "unified_routed_expert_moe", "permute"}

        def compute_calls(fn):
            names = []
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in arithmetic:
                    names.append((n.lineno, n.col_offset, n.func.attr))
            return [name for *_, name in sorted(names)]

        self.assertEqual(compute_calls(baseline), compute_calls(candidate))


if __name__ == "__main__":
    unittest.main(verbosity=2)
