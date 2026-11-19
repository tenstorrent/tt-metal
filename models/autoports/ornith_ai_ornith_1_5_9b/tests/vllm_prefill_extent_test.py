# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host contract for compact padding; numerical acceptance uses the device probe."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class PrefillExtentTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "tt/prefill_runtime.py"
        tree = ast.parse(source.read_text())
        method = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_execution_height"
        )
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        self.select = namespace["_execution_height"]
        self.runtime = SimpleNamespace(
            profile_spec={"linear": (512, 4096, 8192, 32768), "full": (512, 4096, 8192, 16384)}
        )

    def test_large_profiles_keep_large_numerical_branch_and_cover_real_rows(self):
        for kind, canonical in (("linear_attention", 32768), ("full_attention", 16384)):
            for valid in (1, 2, 31, 32, 33, 511, 512, 1033, 8191, 8192, 8193, canonical):
                with self.subTest(kind=kind, valid=valid):
                    height = self.select(self.runtime, kind, canonical, valid)
                    self.assertGreater(height, 4096)
                    self.assertGreaterEqual(height, valid)
                    self.assertEqual(height, 8192 if valid <= 8192 else canonical)

    def test_existing_small_profiles_do_not_change_policy(self):
        for height in (1, 32, 511, 512, 1024, 4096, 8192):
            self.assertEqual(self.select(self.runtime, "linear_attention", height, 1), height)

    def test_missing_prepared_profile_retains_canonical_execution(self):
        self.runtime.profile_spec["full"] = (512, 4096, 16384)
        self.assertEqual(self.select(self.runtime, "full_attention", 16384, 512), 16384)
        self.assertEqual(self.select(self.runtime, "linear_attention", 32768, 512), 8192)


if __name__ == "__main__":
    unittest.main()
