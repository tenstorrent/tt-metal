# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real AST-extracted table planner + real CPU pool validation, without TTNN."""

import argparse
import ast
import hashlib
import json
import statistics
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_8_flash_next.tt.cache_pool import SharedKVPool
from models.autoports.qwen_qwen3_8_flash_next.tt.slot_state import remap_slot_mapping

ROOT = Path(__file__).resolve().parents[1]


def methods():
    source = ROOT / "tt/generator.py"
    owner = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "QwenGenerator"
    )
    nodes = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in ("_table_plan", "_bind")]
    namespace = dict(torch=torch, remap_slot_mapping=remap_slot_mapping)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return type("CpuGenerator", (), {n.name: namespace[n.name] for n in nodes})


class CountingPool(SharedKVPool):
    def normalize_tables(self, *args, **kwargs):
        self.normalizations += 1
        return super().normalize_tables(*args, **kwargs)

    def validate_active_pages(self, *args, **kwargs):
        self.validations += 1
        return super().validate_active_pages(*args, **kwargs)


def fixture(*, batch=2, capacity=128, layers=2, position=31):
    g = methods()()
    g.max_batch_size, g.capacity = 2, capacity
    pool = CountingPool(
        [((max(16, capacity // 32 + 3), 1, 32, 256), "bfloat16", layer) for layer in range(layers)],
        allocate_pair=lambda _, spec: tuple(
            SimpleNamespace(shape=spec.shape, dtype="bfloat16", layout="tile") for _ in range(2)
        ),
    )
    pool.normalizations = pool.validations = 0
    g.shared_pool = pool
    width = ((capacity + 127) // 128) * 4
    g.caches = [
        SimpleNamespace(
            position=0,
            shared_kv_pool=pool,
            layers=[
                SimpleNamespace(
                    attention=SimpleNamespace(
                        page_table=torch.zeros(1, width, dtype=torch.int32),
                        position=torch.zeros(1, dtype=torch.int32),
                        k_cache=pool.pair_for_attention_layer(layer)[0],
                    )
                )
                for layer in range(layers)
            ],
        )
        for _ in range(2)
    ]
    g.page_tables = {}
    g.counters = dict(page_table_refreshes=0, position_refreshes=0)
    g._refresh = lambda target, value: target.copy_(value)
    table = torch.zeros(batch, width, dtype=torch.int32)
    for row in range(batch):
        table[row, :4] = torch.arange(row * 4 + 1, row * 4 + 5)
    normalized = pool.normalize_tables(
        batch_size=batch, capacity=capacity, page_table=table, lengths=[position] * batch
    )
    g._bind(normalized, g.caches[:batch], tuple(range(batch)), [position] * batch)
    pool.normalizations = pool.validations = 0
    return g, table


def plan(g, table, **kwargs):
    kwargs.setdefault("remap", (0, 1))
    kwargs.setdefault("reuse_decode_snapshot", True)
    return g._table_plan(table, g.caches, (0, 1), [32, 32], **kwargs)


class SnapshotFastPathTest(unittest.TestCase):
    def test_prefill_reset_rebind_restores_snapshot_even_with_identical_tables(self):
        g, table = fixture()
        # Prefill plans before resetting a start==0 request. The default must
        # return a table so the later _bind can restore the deleted snapshot.
        tables = g._table_plan(table[:1], [g.caches[0]], (0,), [32])
        self.assertIsNotNone(tables)
        self.assertEqual(g.shared_pool.normalizations, 1)
        del g.page_tables[0]
        g.caches[0].position = 0
        g._bind(tables, [g.caches[0]], (0,), [32])
        self.assertTrue(torch.equal(g.page_tables[0][0], table[:1]))
        self.assertNotEqual(g.page_tables[0][0].data_ptr(), tables[0].data_ptr())
        self.assertEqual(g.caches[0].position, 32)

    def test_only_decode_call_explicitly_enables_snapshot_reuse(self):
        owner = next(
            n
            for n in ast.parse((ROOT / "tt/generator.py").read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "QwenGenerator"
        )
        calls = [
            (method.name, node)
            for method in owner.body
            if isinstance(method, ast.FunctionDef)
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_table_plan"
        ]
        enabled = [
            (name, kw.value.value) for name, call in calls for kw in call.keywords if kw.arg == "reuse_decode_snapshot"
        ]
        self.assertEqual(enabled, [("decode_forward", True)])

    def test_clone_snapshot_equal_values_skip_both_validations(self):
        g, table = fixture()
        self.assertNotEqual(g.page_tables[0][0].data_ptr(), table.data_ptr())
        self.assertIsNone(plan(g, table.clone()))
        self.assertEqual((g.shared_pool.normalizations, g.shared_pool.validations), (0, 0))
        # Disable only the optimization, keeping identical decode metadata.
        slow = plan(g, table, reuse_decode_snapshot=False)
        self.assertEqual((g.shared_pool.normalizations, g.shared_pool.validations), (1, 2))
        self.assertTrue(all(torch.equal(layer, table) for layer in slow))

    def test_mutable_strided_view_changed_future_id_is_not_identity_cached(self):
        g, table = fixture()
        backing = torch.zeros(2, 8, dtype=torch.int32)
        view = backing[:, ::2]
        view.copy_(table)
        self.assertIsNone(plan(g, view))
        view[0, 3] = 16
        with self.assertRaisesRegex(ValueError, "outside"):
            plan(g, view)
        self.assertEqual(g.shared_pool.normalizations, 1)
        self.assertEqual(g.page_tables[0][0][0, 3].item(), 4)
        view[0, 3] = 9
        changed = plan(g, view)
        self.assertEqual(changed[0][0, 3].item(), 9)
        self.assertEqual(g.shared_pool.normalizations, 2)

    def test_page_crossing_revalidates_overlap_with_retained_inactive_slot(self):
        g, table = fixture(position=32)
        table[0, 1] = table[1, 0]
        # Valid future reservations can overlap; they cannot become live together.
        normalized = g.shared_pool.normalize_tables(batch_size=2, capacity=128, page_table=table, lengths=[32, 32])
        g._bind(normalized, g.caches, (0, 1), [32, 32])
        g.shared_pool.normalizations = g.shared_pool.validations = 0
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            g._table_plan(table[:1], [g.caches[0]], (0,), [33], remap=(0, 1), reuse_decode_snapshot=True)
        self.assertEqual((g.shared_pool.normalizations, g.shared_pool.validations), (1, 2))

    def test_per_layer_modes_override_common_and_preserve_submission_row_order(self):
        g, table = fixture()
        second = table.clone()
        second[:, 3] += 1
        normalized = g.shared_pool.normalize_tables(
            batch_size=2, capacity=128, page_tables_by_layer=[table, second], lengths=[31, 31]
        )
        g._bind(normalized, g.caches, (0, 1), [31, 31])
        g.shared_pool.normalizations = g.shared_pool.validations = 0
        self.assertIsNone(plan(g, torch.full_like(table, 999), page_tables_per_layer=[table, second]))
        per_layer = [torch.cat([torch.full_like(x[:1], 999), x[:1]]) for x in (table, second)]
        self.assertIsNone(
            g._table_plan(
                None,
                [g.caches[0]],
                (0,),
                [32],
                page_tables_per_layer=per_layer,
                table_rows=[1],
                submission_rows=2,
                remap=(0, 1),
                reuse_decode_snapshot=True,
            )
        )
        self.assertEqual(g.shared_pool.normalizations, 0)
        per_layer[1][1, 2] = 16
        with self.assertRaisesRegex(ValueError, "outside"):
            g._table_plan(
                None,
                [g.caches[0]],
                (0,),
                [32],
                page_tables_per_layer=per_layer,
                table_rows=[1],
                submission_rows=2,
                remap=(0, 1),
                reuse_decode_snapshot=True,
            )

    def test_missing_snapshot_and_reset_take_full_path(self):
        g, table = fixture()
        del g.page_tables[1]
        with self.assertRaisesRegex(ValueError, "Retained"):
            g._table_plan(table[:1], [g.caches[0]], (0,), [32], remap=(0, 1), reuse_decode_snapshot=True)
        self.assertEqual(g.shared_pool.normalizations, 1)
        g.caches[1].position = 0
        fresh = g._table_plan(table[1:], [g.caches[1]], (1,), [1], remap=(0, 1), reuse_decode_snapshot=True)
        self.assertIsNotNone(fresh)
        self.assertEqual(g.shared_pool.normalizations, 2)

    def test_nonidentity_remap_validates_instead_of_reusing_snapshot(self):
        g, table = fixture()
        value = plan(g, table.flip(0), remap=(1, 0))
        self.assertTrue(torch.equal(value[0], table.flip(0)))
        self.assertEqual(g.shared_pool.normalizations, 1)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            g._table_plan(table[:1], [g.caches[0]], (0,), [32], remap=(1, 0))

    def test_short_and_negative_suffix_tables_require_normalization(self):
        g, table = fixture()
        short = plan(g, table[:, :1])
        self.assertTrue(torch.equal(short[0][:, 1:], torch.zeros(2, 3, dtype=torch.int32)))
        negative = table.clone()
        negative[:, 1:] = -1
        self.assertTrue(torch.equal(plan(g, negative)[0], short[0]))
        self.assertEqual(g.shared_pool.normalizations, 2)
        with self.assertRaisesRegex(ValueError, "width"):
            plan(g, torch.cat([table, table[:, :1]], dim=1))

    def test_structural_and_dtype_checks_precede_fast_path(self):
        for dtype in (torch.float32, torch.bool):
            g, table = fixture()
            with self.assertRaisesRegex(ValueError, "integer dtype"):
                plan(g, table.to(dtype))
            self.assertEqual(g.shared_pool.normalizations, 0)
        g, table = fixture()
        for value, text in ((table[0], "cover"), (table[:1], "cover"), (table.to("meta"), "host")):
            with self.assertRaisesRegex(ValueError, text):
                plan(g, value)
        with self.assertRaisesRegex(ValueError, "one page table"):
            plan(g, table, page_tables_per_layer=[table])
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            plan(g, table, page_tables_per_layer=[table, table.float()])

    def test_invalid_lengths_and_duplicate_slots_do_not_bypass_validation(self):
        for lengths in ([True, 32], [31.0, 32], [-1, 32], [129, 32], [32]):
            g, table = fixture()
            with self.assertRaises(ValueError):
                g._table_plan(table, g.caches, (0, 1), lengths, remap=(0, 1), reuse_decode_snapshot=True)
            self.assertEqual(g.shared_pool.normalizations, 1)
        g, table = fixture()
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            g._table_plan(
                table[:1].repeat(2, 1), [g.caches[0]] * 2, (0, 0), [32, 32], remap=(0, 1), reuse_decode_snapshot=True
            )

    def test_pool_ownership_mismatch_disables_fast_path(self):
        g, table = fixture()
        g.caches[1].shared_kv_pool = object()
        self.assertIsNotNone(plan(g, table))
        self.assertEqual(g.shared_pool.normalizations, 1)

    def test_equal_valued_float_remap_does_not_bypass_original_type_check(self):
        g, table = fixture()
        with self.assertRaisesRegex(ValueError, "integer slot"):
            plan(g, table, remap=(0.0, 1.0))

    def test_private_cache_path_is_unchanged(self):
        g, table = fixture()
        g.shared_pool = None
        self.assertTrue(torch.equal(plan(g, table)[0], table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh report; preserve prior failures")
    torch.set_num_threads(8)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SnapshotFastPathTest)
    )
    report = dict(
        passed=result.wasSuccessful(),
        tests_run=result.testsRun,
        failures=[text for _, text in result.failures + result.errors],
        benchmarks=[],
        scope="CPU-only real AST-extracted _table_plan and real SharedKVPool normalization/overlap checks; no device/HTTP or P2 pass claim",
        method_sha256=hashlib.sha256((ROOT / "tt/generator.py").read_bytes()).hexdigest(),
    )
    if result.wasSuccessful():
        for batch in (1, 2):
            g, table = fixture(batch=batch, capacity=262144, layers=12, position=105)
            times = {"unchanged_snapshot": [], "original_validation": []}
            for repeat in range(args.repeats + 3):
                for name in times if repeat % 2 == 0 else reversed(times):
                    start = time.perf_counter_ns()
                    out = g._table_plan(
                        table,
                        g.caches[:batch],
                        tuple(range(batch)),
                        [106] * batch,
                        remap=(0, 1),
                        reuse_decode_snapshot=name == "unchanged_snapshot",
                    )
                    ms = (time.perf_counter_ns() - start) / 1e6
                    assert (out is None) == (name == "unchanged_snapshot")
                    if repeat >= 3:
                        times[name].append(ms)
            medians = {name: statistics.median(samples) for name, samples in times.items()}
            report["benchmarks"].append(
                dict(
                    batch=batch,
                    capacity=262144,
                    attention_layers=12,
                    columns=8192,
                    current_live_positions=[105] * batch,
                    next_lengths=[106] * batch,
                    threads=torch.get_num_threads(),
                    median_ms=medians,
                    saved_ms=medians["original_validation"] - medians["unchanged_snapshot"],
                    raw_ms=times,
                )
            )
    report["ttnn_imported"] = any(n == "ttnn" or n.startswith("ttnn.") for n in sys.modules)
    assert not report["ttnn_imported"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("passed", "tests_run", "benchmarks", "ttnn_imported")}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
