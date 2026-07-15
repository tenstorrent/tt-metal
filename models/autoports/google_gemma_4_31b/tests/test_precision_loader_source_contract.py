# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Source-only precision-loader contracts; safe without importing TTNN."""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TT_DIR = ROOT / "models/autoports/google_gemma_4_31b/tt"


def _source(name: str) -> str:
    return (TT_DIR / name).read_text(encoding="utf-8")


def _method_source(filename: str, class_name: str, method_name: str) -> str:
    source = _source(filename)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return ast.get_source_segment(source, item)
    raise AssertionError(f"missing {class_name}.{method_name}")


class PrecisionLoaderSourceContractTests(unittest.TestCase):
    def test_single_device_loader_separates_prefill_and_decode_attention_dtypes(self):
        source = _method_source("optimized_decoder.py", "OptimizedDecoder", "from_state_dict")
        self.assertIn("wqkv = _load_weight", source)
        self.assertIn("o_proj = _load_weight", source)
        self.assertGreaterEqual(source.count("dtype=optimization_policy.attention_weight_dtype"), 2)
        self.assertGreaterEqual(source.count("dtype=optimization_policy.resolved_attention_qkv_weight_dtype"), 4)
        self.assertIn("dtype=optimization_policy.resolved_attention_o_weight_dtype", source)
        self.assertIn("math_fidelity=optimization_policy.resolved_attention_qkv_math_fidelity", source)
        self.assertIn("math_fidelity=optimization_policy.resolved_attention_o_math_fidelity", source)

    def test_single_device_decode_consumes_separate_compute_configs(self):
        source = _method_source("optimized_decoder.py", "OptimizedDecoder", "_decode_attention")
        self.assertIn("compute_kernel_config=self.attention_qkv_compute", source)
        self.assertIn("compute_kernel_config=self.attention_o_compute", source)

    def test_tp4_measured_loader_already_consumes_resolved_decode_dtypes(self):
        source = _method_source("multichip_decoder.py", "MultichipDecoder", "from_state_dict")
        self.assertIn("attention_dtype=optimization_policy.attention_weight_dtype", source)
        self.assertGreaterEqual(source.count("dtype=optimization_policy.resolved_attention_qkv_weight_dtype"), 2)
        self.assertIn("dtype=optimization_policy.resolved_attention_o_weight_dtype", source)
        self.assertIn("math_fidelity=optimization_policy.resolved_attention_qkv_math_fidelity", source)
        self.assertIn("math_fidelity=optimization_policy.resolved_attention_o_math_fidelity", source)

    def test_tp4_material_mlp_lm_and_layer_policy_fields_reach_construction(self):
        mlp_source = _method_source("multichip_decoder.py", "_TPOptimizedSharedMLP", "__init__")
        self.assertGreaterEqual(mlp_source.count("dtype=policy.mlp_gate_up_weight_dtype"), 5)
        self.assertGreaterEqual(mlp_source.count("dtype=policy.mlp_down_weight_dtype"), 2)
        self.assertIn("math_fidelity=policy.mlp_gate_up_math_fidelity", mlp_source)
        self.assertIn("math_fidelity=policy.mlp_down_math_fidelity", mlp_source)

        model_source = _method_source("model.py", "Gemma4FullModel", "__init__")
        self.assertGreaterEqual(model_source.count("dtype=self.config.lm_head_weight_dtype"), 2)
        self.assertIn("math_fidelity=self.config.lm_head_math_fidelity", model_source)
        self.assertIn("layer_policy_overrides.get(", model_source)
        self.assertIn("communication_dtype=self.config.decode_ccl_dtype", model_source)
        self.assertIn("prefill_communication_dtype=self.config.prefill_ccl_dtype", model_source)
        self.assertIn("residual_dtype=self.config.residual_dtype", model_source)

    def test_activation_kv_logits_and_sampling_fields_reach_runtime_operations(self):
        embed_source = _method_source("model.py", "Gemma4FullModel", "embed_tokens")
        self.assertIn("self.config.activation_dtype", embed_source)

        cache_source = _method_source("multichip_decoder.py", "MultichipDecoder", "init_paged_kv_cache")
        self.assertIn("cache_dtype=self.policy.kv_cache_dtype", cache_source)

        sharded_terminal = _method_source("model.py", "Gemma4FullModel", "_project_sharded_lm_head_tile")
        terminal = _method_source("model.py", "Gemma4FullModel", "_terminal")
        self.assertIn("dtype=self.config.logits_dtype", sharded_terminal)
        self.assertIn("dtype=self.config.logits_dtype", terminal)

        generator_source = _method_source("generator.py", "Gemma4Generator", "__init__")
        self.assertIn("gather_values_dtype=self.model.config.sampling_dtype", generator_source)

    def test_runtime_summary_reads_constructed_tensor_dtypes(self):
        source = _method_source("model.py", "Gemma4FullModel", "precision_runtime_summary")
        for expression in (
            "attention_weights.wqkv.dtype",
            "attention_weights.o_proj.dtype",
            "decode_qkv.dtype",
            "layer.decode_o_proj.dtype",
            "mlp.gate_prefill.dtype",
            "mlp.down_prefill.dtype",
            "decode_gate_up.dtype",
            "mlp.down_decode.dtype",
            "physical_lm_head.dtype",
        ):
            self.assertIn(expression, source)

    def test_eager_sampling_uses_selected_sampling_dtype(self):
        source = _method_source("generator.py", "Gemma4Generator", "_get_eager_sampler")
        self.assertIn("gather_values_dtype=self.model.config.sampling_dtype", source)
        self.assertNotIn("gather_values_dtype=ttnn.float32", source)


if __name__ == "__main__":
    unittest.main()
