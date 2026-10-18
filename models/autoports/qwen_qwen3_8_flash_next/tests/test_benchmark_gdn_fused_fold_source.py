# SPDX-License-Identifier: Apache-2.0
"""CPU-only source guards; never import TTNN or compile a device kernel."""

import ast
import hashlib
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "benchmark_gdn_fused_fold_candidate.py"
HEADER = HERE / "benchmark_gdn_fused_fold_compute.hpp"
BASELINE = HERE / "benchmark_gdn_resident_compute.cpp"


class FoldSourceTests(unittest.TestCase):
    def transform(self):
        tree = ast.parse(CANDIDATE.read_text())
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "transform_body"
        )
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(CANDIDATE), "exec"), namespace)
        return namespace["transform_body"], function

    def test_only_two_original_fold_sites_change_and_reverse_exactly(self):
        transform, function = self.transform()
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in function.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        }
        original = BASELINE.read_text()
        changed = transform(original)
        self.assertEqual(changed.count("half_product_fold(i, 6, 4);"), 2)
        self.assertEqual(changed.replace(assignments["new"], assignments["old"]), original)
        self.assertIn("for (uint32_t i = 0; i < 4; ++i)", original)
        self.assertIn("if (i == 0)", assignments["new"])
        self.assertIn("half_mul(i, 6, 4);", assignments["new"])

    def test_missing_or_extra_fold_site_rejected(self):
        transform, _ = self.transform()
        original = BASELINE.read_text()
        for changed in (original.replace("half_add(4, 5, 4);", "broken;", 1), original + original):
            with self.subTest(change=len(changed)), self.assertRaises(ValueError):
                transform(changed)

    def test_explicit_opcodes_operand_order_and_latency_slots(self):
        code = "\n".join(line.split("//")[0] for line in HEADER.read_text().splitlines())
        self.assertNotIn("TTI_SFPMAD", code)
        self.assertNotIn("sfpi::vFloat", code)
        self.assertEqual(code.count("TTI_SFPMUL("), 1)
        self.assertEqual(code.count("TTI_SFPADD("), 1)
        self.assertEqual(code.count("TT_SFPLOAD("), 3)
        self.assertEqual(code.count("TT_SFPSTORE("), 1)
        markers = [
            "TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0)",
            "TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, accumulator * 64 + offset)",
            "TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG2, 0)",
            "TTI_SFPNOP;",
            "TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, accumulator * 64 + offset)",
        ]
        positions = [code.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("VectorMode::None", code)

    def test_half_face_addresses_fit_fullsync_and_baseline_is_pinned(self):
        offsets = [face * 32 + row * 2 for face in range(2) for row in range(8)]
        self.assertEqual(offsets, list(range(0, 16, 2)) + list(range(32, 48, 2)))
        for tile in (1, 2, 3, 4, 6):
            self.assertTrue(all(tile * 64 <= tile * 64 + off < (tile + 1) * 64 for off in offsets))
            self.assertLess(max(tile * 64 + off for off in offsets), 8 * 64)
        tree = ast.parse(CANDIDATE.read_text())
        pins = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "PINS" for target in node.targets)
        )
        literal_pins = {
            ast.literal_eval(key): ast.literal_eval(value)
            for key, value in zip(pins.keys, pins.values)
            if key is not None
        }
        for filename, expected in literal_pins.items():
            self.assertEqual(hashlib.sha256((HERE / filename).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
