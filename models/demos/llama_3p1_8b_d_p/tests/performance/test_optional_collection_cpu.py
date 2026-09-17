# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional benchmark collection must not import tensor libraries without valid configuration."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[5]
BENCHMARK = Path(__file__).with_name("test_long_context_performance.py")
CHILD = r"""
import importlib.abc, os, sys
class NoTensorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "ttnn", "tt_metal", "tt_lib", "numpy", "transformers", "safetensors"}:
            raise AssertionError("FORBIDDEN_TENSOR_IMPORT:" + fullname)
sys.meta_path.insert(0, NoTensorImports())
sys.path.insert(0, sys.argv[1])
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
import pytest
raise SystemExit(pytest.main(["--noconftest", "-c", os.path.join(sys.argv[3], "pytest.ini"), "--rootdir", sys.argv[3],
                             "-q", "test_benchmark.py", "test_unrelated.py"]))
"""


class OptionalCollectionTests(unittest.TestCase):
    def collect(self, config_text=None, *, supplied=False, empty=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pytest.ini").write_text("[pytest]\n")
            benchmark = root / "test_benchmark.py"
            benchmark.write_bytes(BENCHMARK.read_bytes())
            unrelated = root / "test_unrelated.py"
            unrelated.write_text("def test_unrelated():\n    assert 2 + 2 == 4\n")
            env = dict(os.environ)
            env.pop("LLAMA_LONG_CONTEXT_PERF_CONFIG", None)
            env.pop("PYTEST_ADDOPTS", None)
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            if supplied:
                path = root / "config.json"
                path.write_text(config_text or "")
                env["LLAMA_LONG_CONTEXT_PERF_CONFIG"] = "" if empty else str(path)
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", CHILD, str(PACKAGE), str(benchmark), str(root), str(unrelated)],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout + result.stderr
            self.assertNotIn("FORBIDDEN_TENSOR_IMPORT", output)
            return result.returncode, output

    # Ordinary collection skips only this unconfigured module and still executes an unrelated test.
    def test_absent_config_skips_optional_module_only(self):
        code, output = self.collect()
        self.assertEqual(code, 0, output)
        self.assertIn("1 passed, 1 skipped", output)

    # A supplied malformed file is a collection error, never an optional-module skip.
    def test_invalid_config_remains_hard_before_tensor_imports(self):
        code, output = self.collect("{", supplied=True)
        self.assertEqual(code, 2, output)
        self.assertIn("JSONDecodeError", output)
        self.assertNotIn("1 skipped", output)

    # An explicit closed site request is refused before any tensor library can be imported.
    def test_closed_config_remains_hard_before_tensor_imports(self):
        code, output = self.collect(
            json.dumps({"schema_version": 1, "context_length": 16384, "authorized": False}), supplied=True
        )
        self.assertEqual(code, 2, output)
        self.assertIn("Caller-provided site contract remains closed", output)
        self.assertNotIn("1 skipped", output)

    # An explicitly empty environment variable is invalid input rather than an absent opt-in.
    def test_empty_supplied_config_is_not_silently_skipped(self):
        code, output = self.collect(supplied=True, empty=True)
        self.assertEqual(code, 2, output)
        self.assertIn("An explicit performance config is required", output)
        self.assertNotIn("1 skipped", output)
