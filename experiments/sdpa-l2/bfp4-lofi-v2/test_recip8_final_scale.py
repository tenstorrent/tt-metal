"""CPU-only source-contract checks; these do not substitute for device JIT."""

import ast
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STREAM = (
    ROOT
    / "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
)


class FinalScaleContract(unittest.TestCase):
    def test_fixed_cb_mapping_and_callsites(self):
        compute = (HERE / "recip8_streaming/compute.cpp").read_text()
        self.assertIn("2,4,2,4,false,0,1,2,6,3,14,4,5,16,15>", re.sub(r"\s+", "", compute))
        stream = STREAM.read_text()
        init_calls = re.findall(r"mul_bcast_cols_init\(([^;]+)\);", stream)
        self.assertEqual(
            [re.sub(r"\s+", "", call) for call in init_calls],
            ["out_in_cb,bcast_cb", "cur_out_cb,scratch_cb"],
        )
        execute_calls = re.findall(r"mul_tiles_bcast_cols\(([^;]+)\);", stream)
        self.assertEqual(len(execute_calls), 4)
        self.assertEqual(sum(bool(re.search(r",\s*scratch_cb\s*,", call)) for call in execute_calls), 1)
        self.assertEqual(sum(bool(re.search(r",\s*bcast_cb\s*,", call)) for call in execute_calls), 3)
        self.assertEqual(len(re.findall(r"salad_correct_fused<[^;]+?prev\.sum,\s*cb_exp_max_diff,", stream)), 3)
        self.assertRegex(stream, r"cb_recip_scratch,\s*cb_normalized_out,")

    def test_macro_scope_and_matching_fidelity(self):
        compute = (HERE / "recip8_streaming/compute.cpp").read_text()
        start = compute.index('#include "final_scale.hpp"')
        end = compute.index("#define mul_bcast_cols_init lofi_safe_mul_bcast_cols_init", start)
        self.assertLess(compute.index("compute_common.hpp"), start)
        self.assertIn("compute_streaming.hpp", compute[start:end])
        helper = (HERE / "recip8_streaming/final_scale.hpp").read_text()
        self.assertEqual(helper.count("if (b == 5)"), 2)
        self.assertEqual(helper.count("MathFidelity::HiFi4"), 2)
        self.assertIn("lofi_safe_mul_bcast_cols_init(a, b, line)", helper)
        self.assertIn("lofi_safe_mul_tiles_bcast_cols<fp32>(a, b, ai, bi, dst)", helper)
        self.assertIn("#ifdef RESIDENT_MAIN", helper)
        self.assertNotIn("APPROX", helper)

    def test_source_pins_and_cli_guard(self):
        text = (HERE / "recip8_fullchip.py").read_text()
        tree = ast.parse(text)
        source_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "source_files")
        env = dict(Path=Path, ROOT=ROOT, HERE=HERE, __file__=str(HERE / "recip8_fullchip.py"))
        exec(compile(ast.Module(body=[source_fn], type_ignores=[]), "source_files", "exec"), env)
        for destination in ("main_bf16", "fast_bf16"):
            paths = env["source_files"](destination)
            self.assertTrue(all(p.is_file() for p in paths))
            self.assertIn(HERE / "recip8_streaming/final_scale.hpp", paths)
        expected_guard = ast.parse('not args.final_scale_hifi4 or args.destination == "fast_bf16"', mode="eval").body
        self.assertIn(
            ast.dump(expected_guard),
            [ast.dump(node.test) for node in ast.walk(tree) if isinstance(node, ast.Assert)],
        )
        configs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ttnn"
            and node.func.attr == "ComputeConfigDescriptor"
        ]
        self.assertEqual(len(configs), 1)
        keywords = {kw.arg: kw.value for kw in configs[0].keywords}
        self.assertEqual(len(keywords), len(configs[0].keywords))
        self.assertNotIn(None, keywords, "Config **kwargs could override audited values")
        self.assertIs(ast.literal_eval(keywords["dst_full_sync_en"]), False)
        self.assertIs(ast.literal_eval(keywords["math_approx_mode"]), True)
        self.assertEqual(
            ast.dump(keywords["math_fidelity"]),
            ast.dump(ast.parse("ttnn.MathFidelity.LoFi", mode="eval").body),
        )


if __name__ == "__main__":
    unittest.main()
