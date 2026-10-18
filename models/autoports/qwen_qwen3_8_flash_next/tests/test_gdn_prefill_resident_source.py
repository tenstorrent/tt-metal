# SPDX-License-Identifier: Apache-2.0
"""CPU-only production extraction and fallback guards; no TTNN imports."""

import ast
import hashlib
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
HELPER = ROOT / "tt/gdn_prefill_resident.py"
EVIDENCE = ROOT / "doc/benchmark/repair_deadline/gdn_scalar_tail_initial"
PIN = "ea2eba307627c3ac5946ec0718e31904698c87bb37c1c4d1211f4ee7d05f1df3"


def function(path, name):
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(old)
    return source.replace(old, new)


def literals():
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(HELPER.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("READER", "COMPUTE", "NATIVE_CONTRACT_SHA256")
    }


class ResidentProductionTests(unittest.TestCase):
    def test_embedded_kernels_match_frozen_combination(self):
        self.assertEqual(
            hashlib.sha256((TESTS / "benchmark_gdn_scalar_tail_candidate.py").read_bytes()).hexdigest(), PIN
        )
        data = literals()
        for key, filename in (("READER", "candidate_reader.cpp"), ("COMPUTE", "candidate_compute.cpp")):
            self.assertEqual(data[key], (EVIDENCE / filename).read_text())
        self.assertEqual(
            hashlib.sha256(data["READER"].encode()).hexdigest(),
            "320d8ca1229086f647e43d5b2990bf66f0c33b46152a9f9b99cda32e9ed924cc",
        )
        self.assertEqual(
            hashlib.sha256(data["COMPUTE"].encode()).hexdigest(),
            "d4f8fc6dab54860662ed2c4400180c4a9cb6a0df3ed42e935f14ac44790c1122",
        )

    def test_launcher_ast_matches_selected_candidate(self):
        original = function(ROOT / "tt/gdn_prefill_qk.py", "_rm_recurrence")
        namespace = dict(
            inspect=types.SimpleNamespace(getsource=lambda _: original),
            textwrap=textwrap,
            baseline=types.SimpleNamespace(_rm_recurrence=None),
            qualified=types.SimpleNamespace(replace=replace_once),
        )
        exec(function(TESTS / "benchmark_gdn_resident_candidate.py", "launcher_source"), namespace)
        expected = namespace["launcher_source"]()
        transform = dict(replace_once=replace_once)
        exec(function(TESTS / "benchmark_gdn_scalar_cache_candidate.py", "transform_launcher"), transform)
        expected = transform["transform_launcher"](expected).replace("self._half_column_poison", "0")
        expected = expected.replace("def resident_recurrence(", "def _resident_recurrence(")
        actual = function(HELPER, "_resident_recurrence").replace(
            "    _record_signature(self, batch, length, workers, config, cbs)\n", ""
        )
        self.assertEqual(ast.dump(ast.parse(actual)), ast.dump(ast.parse(expected)))

    def test_native_pins_and_no_runtime_rewriting(self):
        data = literals()
        self.assertEqual(len(data["NATIVE_CONTRACT_SHA256"]), 7)
        for relative, expected in data["NATIVE_CONTRACT_SHA256"].items():
            self.assertEqual(hashlib.sha256((ROOT.parents[2] / relative).read_bytes()).hexdigest(), expected)
        tree = ast.parse(HELPER.read_text())
        self.assertFalse(
            any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("exec", "eval")
                for n in ast.walk(tree)
            )
        )
        self.assertFalse(
            any(
                isinstance(n, ast.ImportFrom) and ("tests" in (n.module or "") or "inspect" in (n.module or ""))
                for n in ast.walk(tree)
            )
        )

    def test_real_super_fallback_preserves_short_and_disabled_routes(self):
        tree = ast.parse(HELPER.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))

        class Original:
            def _recurrence(self, *args):
                return ("original", args)

        namespace = dict(PrefillQKRowMajorGDN=Original)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(HELPER), "exec"), namespace)
        obj = namespace["PrefillResidentGDN"]()
        obj.prefill_qk_rm_enabled = obj.recurrence_broadcast = True
        for enabled, batch, length in (
            (False, 1, 512),
            (True, 1, 1),
            (True, 1, 127),
            (True, 1, 129),
            (True, 1, 1024),
            (True, 4, 512),
        ):
            obj.prefill_resident_enabled = enabled
            q = types.SimpleNamespace(shape=(batch, length, 512))
            self.assertEqual(obj._recurrence(q, None, None, None, None)[0], "original")

    def test_view_enables_only_owner32_and_retains_qk_attestation(self):
        source = (ROOT / "tt/prefill_view.py").read_text()
        tree = ast.parse(source)
        decoder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_decoder")
        branch = next(
            n
            for n in decoder.body
            if isinstance(n, ast.If) and ast.unparse(n.test) == "owner.kind == 'linear_attention'"
        )
        assignments = {ast.unparse(n.targets[0]): n.value for n in branch.body if isinstance(n, ast.Assign)}
        self.assertEqual(ast.unparse(assignments["attention.__class__"]), "PrefillResidentGDN")
        self.assertEqual(ast.unparse(assignments["attention.prefill_resident_enabled"]), "owner.batch_size == 32")
        self.assertEqual(ast.literal_eval(assignments["attention.prefill_resident_signatures"]), {})
        self.assertTrue(ast.literal_eval(assignments["attention.prefill_qk_rm_enabled"]))
        self.assertEqual(ast.literal_eval(assignments["attention.prefill_qk_rm_signatures"]), {})
        summary = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "runtime_summary")
        returned = next(n.value for n in summary.body if isinstance(n, ast.Return))
        entries = {kw.arg: kw.value for kw in returned.keywords}
        self.assertIn("gdn_prefill_qk_rm", entries)

        def layer(kind, enabled=None):
            attention = types.SimpleNamespace(prefill_resident_signatures={"512": {"sequence": 512}})
            if enabled is not None:
                attention.prefill_resident_enabled = enabled
            return types.SimpleNamespace(kind=kind, attention=attention)

        owner = types.SimpleNamespace(layers=[layer("linear_attention"), layer("attention")])
        model = types.SimpleNamespace(
            layer_indices=[0, 3], layers=[layer("linear_attention", True), layer("attention")]
        )
        observed = eval(
            compile(ast.Expression(entries["gdn_prefill_resident"]), "resident_summary", "eval"),
            {"self": types.SimpleNamespace(owner=owner, model=model)},
        )
        self.assertEqual(
            observed, [{"layer": 0, "enabled": True, "owner_enabled": False, "signatures": [{"sequence": 512}]}]
        )

    def test_metadata_uses_actual_descriptor_values(self):
        namespace = dict(_PROVENANCE={"proof": "fake"})
        exec(function(HELPER, "_record_signature"), namespace)
        fmt = lambda index: types.SimpleNamespace(buffer_index=index, page_size=4096)
        state = types.SimpleNamespace(total_size=16384, format_descriptors=[fmt(0), fmt(12)])
        scalar = types.SimpleNamespace(total_size=8192, format_descriptors=[fmt(16)])
        config = types.SimpleNamespace(
            dst_full_sync_en=True, fp32_dest_acc_en=True, math_approx_mode=False, math_fidelity="MathFidelity.HiFi4"
        )
        obj = types.SimpleNamespace(prefill_resident_signatures={})
        namespace["_record_signature"](obj, 1, 512, 96, config, [state, scalar])
        row = obj.prefill_resident_signatures["512"]
        self.assertEqual((row["workers"], row["state_tiles"], row["scalar_cache_bytes_per_worker"]), (96, 4, 8192))
        scalar.total_size = 4096
        namespace["_record_signature"](obj, 1, 512, 80, config, [state, scalar])
        self.assertEqual(obj.prefill_resident_signatures["512"]["workers"], 80)
        self.assertEqual(obj.prefill_resident_signatures["512"]["scalar_cache_bytes_per_worker"], 4096)


if __name__ == "__main__":
    unittest.main(verbosity=2)
