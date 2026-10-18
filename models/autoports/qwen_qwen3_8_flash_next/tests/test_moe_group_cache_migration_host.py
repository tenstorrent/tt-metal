# SPDX-License-Identifier: Apache-2.0
"""Exact source-proof guards only: never deserialize tensors or mutate a cache."""

import ast
import copy
import importlib.util
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "moe_cache_migration_host", ROOT / "tools/benchmark_migrate_moe_group_cache.py"
)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
SOURCE = ROOT / "tt/multichip_decoder.py"
REPO = ROOT.parents[2]
OLD = subprocess.check_output(["git", "show", f"{TOOL.OLD_COMMIT}:{SOURCE.relative_to(REPO)}"], cwd=REPO)
NEW = SOURCE.read_bytes()


class MoEGroupMigrationTests(unittest.TestCase):
    def test_real_frozen_source_normalizes_exactly(self):
        proof = TOOL.validate_source_delta(OLD, NEW)
        self.assertEqual(proof["old_full_ast_sha256"], proof["normalized_new_ast_sha256"])
        self.assertEqual(set(proof["unchanged_producer_ast_sha256"]), {"MeshOps", "ParallelMoE.__init__"})
        verifier = TOOL._load_verifier()
        self.assertEqual(verifier.OLD_COMMIT, TOOL.OLD_COMMIT)
        self.assertIs(verifier.validate_source_delta, TOOL.validate_source_delta)

    def test_any_producer_consumer_helper_or_global_change_rejected(self):
        for scope in ("producer", "consumer", "helper", "global", "import", "duplicate"):
            tree = ast.parse(NEW)
            cls = TOOL.unique(tree.body, ast.ClassDef, "ParallelMoE")
            if scope == "producer":
                TOOL.unique(cls.body, ast.FunctionDef, "__init__").body.append(ast.parse("changed = 1").body[0])
            elif scope == "consumer":
                TOOL.unique(cls.body, ast.FunctionDef, "group_packed").body.append(ast.parse("changed = 1").body[0])
            elif scope == "helper":
                TOOL.unique(tree.body, ast.FunctionDef, "_expert_capacity_source").body.append(
                    ast.parse("changed = 1").body[0]
                )
            elif scope == "duplicate":
                tree.body.append(copy.deepcopy(TOOL.unique(tree.body, ast.FunctionDef, "_expert_capacity_source")))
            else:
                tree.body.append(ast.parse("changed_global = 1" if scope == "global" else "import fractions").body[0])
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                TOOL.validate_source_delta(OLD, ast.unparse(ast.fix_missing_locations(tree)).encode())
        with self.assertRaises(ValueError):
            TOOL.validate_source_delta(OLD + b"\n", NEW)

    def test_gdn_constructor_consumers_and_helper_reject_drift(self):
        for method in ("__init__", "prefill_forward", "decode_forward", "_chunk"):
            tree = ast.parse(NEW)
            cls = TOOL.unique(tree.body, ast.ClassDef, "ParallelGatedDeltaNet")
            TOOL.unique(cls.body, ast.FunctionDef, method).body.append(ast.parse("changed = 1").body[0])
            with self.subTest(method=method), self.assertRaises(ValueError):
                TOOL.validate_source_delta(OLD, ast.unparse(ast.fix_missing_locations(tree)).encode())
        with patch.object(TOOL, "GDN_HELPER_SHA256", "changed"), self.assertRaises(ValueError):
            TOOL.validate_source_delta(OLD, NEW)

    def test_identity_native_precision_and_seal_guards_retained(self):
        verifier = TOOL._load_verifier()
        old = dict(
            transform_sha256={"/source.py": TOOL.digest_bytes(OLD), "/cache.py": "cache", "/precision.py": "precision"},
            native_sha256={"/native.so": "native"},
            precision_sha256="policy",
            source_seal={"source": "unchanged"},
        )
        new = copy.deepcopy(old)
        new["transform_sha256"]["/source.py"] = TOOL.digest_bytes(NEW)
        verifier.validate_identity_delta(old, new, "/source.py", OLD, NEW)
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
            with self.subTest(field=field, key=key), self.assertRaises(ValueError):
                verifier.validate_identity_delta(old, altered, "/source.py", OLD, NEW)

    def test_unpinned_index_or_identity_rejected_before_loading_verifier(self):
        for key in ("old_identity", "expected_old_index_sha256"):
            args = SimpleNamespace(old_identity=TOOL.OLD_IDENTITY, expected_old_index_sha256=TOOL.OLD_INDEX_SHA256)
            setattr(args, key, "changed")
            with patch.object(TOOL, "_load_verifier") as load, self.subTest(key=key), self.assertRaises(ValueError):
                TOOL.migrate(args)
            load.assert_not_called()
        with patch.object(TOOL, "VERIFIER_SHA256", "changed"), self.assertRaises(ValueError):
            TOOL._load_verifier()
