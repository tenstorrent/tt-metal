# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of this migration's exact production delta and fail-closed guards.

Reads the pinned baseline from git and current production source; never opens a
cache, imports TTNN, loads a tensor, or invokes the migration publisher.
"""

import argparse
import ast
import copy
import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "tools/benchmark_migrate_qsa_fold_cache.py"
SPEC = importlib.util.spec_from_file_location("qsa_fold_cache_migration_host", SOURCE)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class QsaFoldMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = subprocess.check_output(
            ["git", "-C", str(TOOL.ROOT), "show", f"{TOOL.OLD_COMMIT}:{TOOL.MULTICHIP.relative_to(TOOL.ROOT)}"]
        )
        cls.new = TOOL.MULTICHIP.read_bytes()

    def identities(self):
        path = str(TOOL.MULTICHIP)
        old = {
            "transform_sha256": {path: TOOL.digest_bytes(self.old), "/cache.py": "cache", "/precision.py": "precision"},
            "native_sha256": {"/native.so": "native"},
            "precision_sha256": "policy",
            "source_seal": {"source": "unchanged"},
        }
        new = copy.deepcopy(old)
        new["transform_sha256"][path] = TOOL.digest_bytes(self.new)
        return old, new

    def test_exact_current_production_delta_and_producers(self):
        proof = TOOL.validate_source_delta(self.old, self.new)
        self.assertEqual(proof["allowed_changed_node"], "ParallelQSA._sparse_attention")
        self.assertEqual(proof["old_full_ast_sha256"], proof["normalized_new_ast_sha256"])
        self.assertEqual(set(proof["unchanged_producer_ast_sha256"]), {"MeshOps", "ParallelMoE"})

    def test_changed_qualified_method_is_rejected(self):
        replacements = (
            (b"self._decode_phase and b > 1 and s == 1", b"b > 1 and s == 1"),
            (b"self._decode_phase and b > 1 and s == 1", b"self._decode_phase and b >= 1 and s == 1"),
            (b"self._decode_fold_vectorized = vectorized_decode", b"self._decode_fold_vectorized = True"),
            (b"chunk = ttnn.permute(q, (2, 1, 0, 3))", b"chunk = ttnn.permute(q, (0, 1, 2, 3))"),
            (
                b"out = ttnn.permute(out[:, :6, :, :], (2, 0, 1, 3))",
                b"out = ttnn.permute(out[:, :5, :, :], (2, 0, 1, 3))",
            ),
            (b"scale=0.0625,", b"scale=0.125,"),
        )
        for old, new in replacements:
            with self.subTest(replacement=new):
                self.assertIn(old, self.new)
                altered = self.new.replace(old, new)
                with self.assertRaisesRegex(ValueError, "exact qualified QSA fold AST"):
                    TOOL.validate_source_delta(self.old, altered)

    def test_all_other_classes_methods_imports_and_globals_are_rejected(self):
        for name in ("MeshOps", "ParallelMoE", "ParallelGatedDeltaNet", "ParallelQSA", "MultichipDecoder"):
            tree = ast.parse(self.new)
            node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            node.body.append(ast.parse("unexpected_cache_change = True").body[0])
            with self.subTest(class_name=name):
                with self.assertRaisesRegex(ValueError, "outside the exact qualified QSA fold method"):
                    TOOL.validate_source_delta(self.old, ast.unparse(tree).encode())
        for extra in (b"\nimport unexpected_dependency\n", b"\nunexpected_global = 1\n", b"\ndef unexpected(): pass\n"):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, "outside the exact qualified QSA fold method"):
                    TOOL.validate_source_delta(self.old, self.new + extra)

    def test_duplicate_class_or_method_is_rejected(self):
        tree = ast.parse(self.new)
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ParallelQSA")
        method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "_sparse_attention")
        duplicate_class = copy.deepcopy(tree)
        duplicate_class.body.append(copy.deepcopy(node))
        node.body.append(copy.deepcopy(method))
        for altered in (tree, duplicate_class):
            with self.assertRaisesRegex(ValueError, "Expected exactly one"):
                TOOL.validate_source_delta(self.old, ast.unparse(altered).encode())

    def test_old_source_is_pinned_and_unchanged_delta_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pinned QSA baseline"):
            TOOL.validate_source_delta(self.old + b"\n", self.new)
        with self.assertRaisesRegex(ValueError, "exact qualified QSA fold AST"):
            TOOL.validate_source_delta(self.old, self.old)

    def test_private_verifier_retains_historical_behavior_and_checks(self):
        original = TOOL.VERIFIER_SOURCE.read_bytes()
        private = TOOL._load_verifier()
        historical_spec = importlib.util.spec_from_file_location("historical_gdn_migration", TOOL.VERIFIER_SOURCE)
        historical = importlib.util.module_from_spec(historical_spec)
        historical_spec.loader.exec_module(historical)
        self.assertEqual(TOOL.digest_bytes(original), TOOL.VERIFIER_SHA256)
        self.assertEqual(historical.OLD_COMMIT, "eca11b6710363892fc6ccbb44af2374ac5293cad")
        self.assertEqual(private.OLD_COMMIT, TOOL.OLD_COMMIT)
        self.assertIs(private.validate_source_delta, TOOL.validate_source_delta)
        for name in ("migrate", "validate_identity_delta", "verify_entries", "validate_entry", "file_state"):
            self.assertEqual(getattr(private, name).__code__.co_code, getattr(historical, name).__code__.co_code)
        self.assertEqual(TOOL.VERIFIER_SOURCE.read_bytes(), original)

    def test_reused_identity_guard_rejects_all_other_dependency_changes(self):
        verifier = TOOL._load_verifier()
        old, new = self.identities()
        verifier.validate_identity_delta(old, new, TOOL.MULTICHIP, self.old, self.new)
        for field, key in (
            ("transform_sha256", "/cache.py"),
            ("transform_sha256", "/precision.py"),
            ("native_sha256", "/native.so"),
            ("source_seal", "source"),
            ("precision_sha256", None),
        ):
            altered = copy.deepcopy(new)
            if key is None:
                altered[field] = "changed"
            else:
                altered[field][key] = "changed"
            with self.subTest(field=field, key=key):
                with self.assertRaisesRegex(ValueError, "Unexpected cache identity"):
                    verifier.validate_identity_delta(old, altered, TOOL.MULTICHIP, self.old, self.new)

    def test_wrong_index_or_identity_fails_before_cache_access(self):
        for identity, digest in (("0" * 64, TOOL.OLD_INDEX_SHA256), (TOOL.OLD_IDENTITY, "0" * 64)):
            args = argparse.Namespace(
                cache_root=Path("/must-not-be-accessed"), old_identity=identity, expected_old_index_sha256=digest
            )
            with patch.object(TOOL, "_load_verifier", side_effect=AssertionError("Verifier must not load")):
                with self.assertRaisesRegex(ValueError, "only the pinned previous identity"):
                    TOOL.migrate(args)

    def test_modified_historical_verifier_is_rejected_before_execution(self):
        with patch.object(Path, "read_bytes", return_value=b"raise RuntimeError('must not execute')"):
            with self.assertRaisesRegex(ValueError, "Historical cache verifier differs"):
                TOOL._load_verifier()


if __name__ == "__main__":
    unittest.main()
