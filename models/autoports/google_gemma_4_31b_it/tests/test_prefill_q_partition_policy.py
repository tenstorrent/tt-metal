"""Host-only regression for production prefill Q-chunk policy selection.

The test executes the geometry assignment and SDPA keyword expressions
extracted from MultichipDecoder source with small doubles; it imports no TTNN
or model modules.
"""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(
    os.environ.get(
        "GEMMA_MULTICHIP_DECODER_SOURCE",
        Path(__file__).parents[1] / "tt/multichip_decoder.py",
    )
)
OPTIMIZED_SOURCE = SOURCE.with_name("optimized_decoder.py")


def _production_nodes():
    tree = ast.parse(SOURCE.read_text())
    decoder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MultichipDecoder")
    constructor = next(
        node for node in decoder.body if isinstance(node, ast.FunctionDef) and node.name == "from_state_dict"
    )
    geometry = next(
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "obj"
            and target.attr == "geometry"
            for target in node.targets
        )
    )
    optimized_tree = ast.parse(OPTIMIZED_SOURCE.read_text())
    optimized_decoder = next(
        node for node in optimized_tree.body if isinstance(node, ast.ClassDef) and node.name == "OptimizedDecoder"
    )
    forward = next(
        node
        for node in optimized_decoder.body
        if isinstance(node, ast.FunctionDef) and node.name == "prefill_chunk_forward"
    )
    sdpa_call = next(
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "SDPAProgramConfig"
    )
    keywords = {keyword.arg: keyword.value for keyword in sdpa_call.keywords}
    return geometry, keywords["q_chunk_size"], keywords["k_chunk_size"]


def _geometry(*, fidelity, cache_dtype, sliding):
    assignment, _, _ = _production_nodes()
    obj = SimpleNamespace(residual_sharded=False, sliding_window=sliding)
    ttnn = SimpleNamespace(
        bfloat16="BF16",
        MathFidelity=SimpleNamespace(HiFi2="HiFi2", HiFi4="HiFi4"),
    )
    namespace = {"obj": obj, "ttnn": ttnn, "sdpa_fidelity": fidelity, "cache_dtype": cache_dtype}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return obj.geometry


def _evaluate_production_q(*, geometry, attention_start):
    _, q_expr, _ = _production_nodes()
    return eval(
        compile(ast.Expression(body=q_expr), str(SOURCE), "eval"),
        {"attention_start": attention_start, "self": SimpleNamespace(geometry=geometry)},
    )


class PrefillQPartitionPolicyTest(unittest.TestCase):
    def test_hifi4_selects_q32_for_sliding_and_global_attention(self):
        for sliding in (False, True):
            with self.subTest(sliding=sliding):
                geometry = _geometry(fidelity="HiFi4", cache_dtype="BFP8", sliding=sliding)
                self.assertEqual(geometry["prefill_q"], 32)
                self.assertEqual(_evaluate_production_q(geometry=geometry, attention_start=0), 32)
                self.assertEqual(_evaluate_production_q(geometry=geometry, attention_start=32), 32)

    def test_hifi2_head_policy_retains_existing_q128(self):
        geometry = _geometry(fidelity="HiFi2", cache_dtype="BFP8", sliding=True)
        self.assertEqual(geometry["prefill_q"], 128)
        self.assertEqual(_evaluate_production_q(geometry=geometry, attention_start=0), 128)
        self.assertEqual(_evaluate_production_q(geometry=geometry, attention_start=32), 32)

    def test_hifi2_bf16_global_and_sliding_geometry_is_preserved(self):
        global_geometry = _geometry(fidelity="HiFi2", cache_dtype="BF16", sliding=False)
        sliding_geometry = _geometry(fidelity="HiFi2", cache_dtype="BF16", sliding=True)
        self.assertEqual(global_geometry["prefill_q"], 64)
        self.assertEqual(sliding_geometry["prefill_q"], 128)

    def test_k_chunk_remains_literal_32_and_q32_aligned_path_is_uniform(self):
        _, _, k_expr = _production_nodes()
        self.assertIsInstance(k_expr, ast.Constant)
        self.assertEqual(k_expr.value, 32)
        q32_geometry = _geometry(fidelity="HiFi4", cache_dtype="BF16", sliding=False)
        self.assertEqual(q32_geometry["prefill_q"], 32)
        self.assertEqual(_evaluate_production_q(geometry=q32_geometry, attention_start=0), 32)
        self.assertEqual(_evaluate_production_q(geometry=q32_geometry, attention_start=128), 32)


if __name__ == "__main__":
    unittest.main()
