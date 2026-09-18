"""Source-route fault tests; no Torch/native import or device work."""
import ast
import importlib.abc
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "ttnn", "tt_lib", "tt_d_gen", "transformers", "numpy"):
            raise AssertionError("native import forbidden: " + fullname)


sys.meta_path.insert(0, BlockNative())
import run_owner
from edge_checks import EffectCheck, check_values, seed_value
from edge_coverage import PAGE_BYTES, RUNTIME_CALLS


class PrimaryError(Exception):
    pass


class CleanupError(Exception):
    pass


def seed_probe(failures, primary):
    calls = []
    errors = []

    def call(name):
        calls.append(name)
        if name in failures:
            raise CleanupError(name)

    tensors = iter(
        [SimpleNamespace(deallocate=lambda force: call("k")), SimpleNamespace(deallocate=lambda force: call("v"))]
    )
    tt = SimpleNamespace(
        bfloat16="bf16",
        TILE_LAYOUT="tile",
        DRAM_MEMORY_CONFIG="dram",
        ReplicateTensorToMesh=lambda mesh: None,
        from_torch=lambda *a, **k: next(tensors),
        synchronize_device=lambda mesh: call("sync"),
    )
    torch = SimpleNamespace(bfloat16="bf16", full=lambda *a, **k: None)

    def writer(*a, **k):
        if primary:
            raise PrimaryError("writer")

    kwargs = {"cleanup_errors": errors} if "cleanup_errors" in run_owner.seed_cache.__code__.co_varnames else {}
    try:
        run_owner.seed_cache(tt, torch, None, None, writer, **kwargs)
    except BaseException as exc:
        return calls, errors, exc
    raise AssertionError("fault should stop seed setup")


def finally_probe(failures, primary="primary compute error"):
    calls = []
    reports = []
    gc_index = [0]

    def call(name):
        calls.append(name)
        if name in failures:
            raise CleanupError(name)

    def collect():
        name = "service" if gc_index[0] == 0 else "collect"
        gc_index[0] += 1
        call(name)

    report = dict(
        acks=[],
        published=[],
        errors=[primary] if primary else [],
        cleanup_errors=[],
        status="runtime_edges_complete",
        owner_cleanup_complete=False,
    )
    env = dict(
        vars(run_owner),
        report=report,
        recorder=SimpleNamespace(rows=[]),
        sink=None,
        service=object(),
        channel=object(),
        producer=SimpleNamespace(shutdown=lambda: call("producer")),
        router=SimpleNamespace(stop=lambda: call("router")),
        saved={"value": SimpleNamespace(close=lambda: call("saved"))},
        cache=SimpleNamespace(
            k=SimpleNamespace(deallocate=lambda force: call("cache.k")),
            v=SimpleNamespace(deallocate=lambda force: call("cache.v")),
        ),
        runtime=SimpleNamespace(model=SimpleNamespace(close=lambda: call("model"))),
        mesh=object(),
        gc=SimpleNamespace(collect=collect),
        ttnn=SimpleNamespace(
            synchronize_device=lambda mesh: call("mesh.sync"),
            set_fabric_config=lambda mode: call("fabric.disable"),
            close_mesh_device=lambda mesh: call("mesh.close"),
            FabricConfig=SimpleNamespace(DISABLED="disabled"),
        ),
        write_json=lambda path, value: reports.append(dict(value)),
        output=Path("/unused"),
    )
    node = next(
        n
        for n in ast.parse(Path(run_owner.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    block = next(n for n in node.body if isinstance(n, ast.Try))
    exec(compile(ast.Module(body=block.finalbody, type_ignores=[]), str(run_owner.__file__), "exec"), env)
    return calls, reports[0]


class CleanupTests(unittest.TestCase):
    # A primary writer failure survives secondary temporary-release failures, and both releases are attempted.
    def test_seed_primary_survives_each_temporary_release_failure(self):
        for failures in ({"v"}, {"k"}, {"v", "k"}):
            with self.subTest(failures=failures):
                calls, errors, exc = seed_probe(failures, True)
                self.assertEqual(calls, ["v", "k"])
                self.assertIsInstance(exc, PrimaryError)
                self.assertEqual(len(errors), len(failures))

    # A failed release after a successful writer must stop initialization and report cleanup failure.
    def test_seed_cleanup_only_failure_is_not_success(self):
        calls, errors, exc = seed_probe({"v"}, False)
        self.assertEqual(calls, ["v", "k"])
        self.assertTrue(errors)
        self.assertIsInstance(exc, RuntimeError)

    # Each actual owner-finally route attempts every later action once, even after any injected release failure.
    def test_all_finally_actions_continue_and_mesh_closes_last(self):
        expected = [
            "service",
            "producer",
            "router",
            "saved",
            "cache.k",
            "cache.v",
            "model",
            "collect",
            "mesh.sync",
            "fabric.disable",
            "mesh.close",
        ]
        for failure in expected:
            with self.subTest(failure=failure):
                calls, report = finally_probe({failure})
                self.assertEqual(calls, expected)
                self.assertEqual(report["errors"], ["primary compute error"])
                self.assertEqual(len(report["cleanup_errors"]), 1)
                self.assertFalse(report["gate_passed"])
                self.assertFalse(report["owner_cleanup_complete"])

    # Multiple release faults are aggregated while the primary compute failure remains unchanged.
    def test_multiple_finally_failures_are_aggregated(self):
        calls, report = finally_probe({"producer", "cache.k", "model", "mesh.sync"})
        self.assertEqual(calls[-1], "mesh.close")
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(len(report["cleanup_errors"]), 4)
        self.assertEqual(report["errors"], ["primary compute error"])

    # Cleanup errors alone fail an otherwise completed run, while an entirely clean route can pass.
    def test_cleanup_only_failure_and_clean_success(self):
        _, bad = finally_probe({"saved"}, primary=None)
        self.assertFalse(bad["gate_passed"])
        _, good = finally_probe(set(), primary=None)
        self.assertTrue(good["gate_passed"])
        self.assertTrue(good["owner_cleanup_complete"])


class StructuralWriteTests(unittest.TestCase):
    # A changed packed page whose valid row is all zero must fail the structural write check.
    def test_zero_selected_valid_row_rejected(self):
        check = EffectCheck(RUNTIME_CALLS[4], lambda raw: [[0.0] * 128 for _ in range(32)])
        check.count = 1
        with self.assertRaisesRegex(ValueError, "zero valid"):
            check.accept((0, 0, 0, 32), b"a" * PAGE_BYTES, b"b" * PAGE_BYTES)

    # Changed padding alone cannot prove a valid write: every valid row must replace its original seeded sentinel.
    def test_seeded_valid_row_with_changed_padding_rejected(self):
        value = seed_value(8, 0, 0)
        rows = [[value] * 128] + [[0.0] * 128 for _ in range(31)]
        check = EffectCheck(RUNTIME_CALLS[4], lambda raw: rows)
        check.count = ((0 * 16 + 8) * 32 + 0) * 64 + 2
        with self.assertRaisesRegex(ValueError, "seed sentinel"):
            check.accept((8, 0, 0, 64), b"a" * PAGE_BYTES, b"b" * PAGE_BYTES)

    # Legitimate zeros within a nonzero row and exact zero padding remain allowed, with no numerical threshold.
    def test_nonzero_valid_structure_and_zero_padding_allowed(self):
        rows = [[0.0] * 127 + [0.25]] + [[0.0] * 128 for _ in range(31)]
        self.assertEqual(check_values(rows, valid_rows=1)["valid_values"], 128)


if __name__ == "__main__":
    unittest.main()
