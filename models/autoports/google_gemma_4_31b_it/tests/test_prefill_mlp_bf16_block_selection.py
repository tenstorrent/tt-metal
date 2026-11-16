# SPDX-License-Identifier: Apache-2.0
"""Host-only regression checks for prefill MLP L1 block selection.

The test AST-extracts OptimizedDecoder._linear from production source and runs
that method with recording TTNN doubles. It imports neither TTNN nor model code.
"""

import ast
import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(os.environ.get("GEMMA_OPTIMIZED_DECODER_SOURCE", Path(__file__).parents[1] / "tt/optimized_decoder.py"))


def _actual_linear_method():
    tree = ast.parse(SOURCE.read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OptimizedDecoder")
    return next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "_linear")


def _call_linear(*, name, activation_dtype, weight_dtype="BFP8", fp32=False, decoding=False):
    method = _actual_linear_method()
    calls = []
    input_tensor = SimpleNamespace(shape=(1, 1, 16, 128))
    weight = SimpleNamespace(shape=(1, 1, 128, 512), dtype=weight_dtype)
    kernel = SimpleNamespace(fp32_dest_acc_en=fp32)

    def minimal_config(**kwargs):
        calls.append(("minimal_config", kwargs))
        return SimpleNamespace(**kwargs)

    def minimal_matmul(**kwargs):
        calls.append(("minimal_matmul", kwargs))
        return SimpleNamespace(shape=(1, 1, 16, 512))

    def dram_config(**kwargs):
        calls.append(("decode_config", kwargs))
        return SimpleNamespace(**kwargs)

    def linear(*args, **kwargs):
        calls.append(("decode_linear", kwargs))
        return SimpleNamespace(shape=(1, 1, 16, 512))

    ttnn = SimpleNamespace(
        bfloat8_b="BFP8",
        bfloat16="BF16",
        L1_MEMORY_CONFIG="L1",
        DRAM_MEMORY_CONFIG="DRAM",
        L1_WIDTH_SHARDED_MEMORY_CONFIG="L1_WIDTH_SHARDED",
        CoreCoord=lambda x, y: (x, y),
        MinimalMatmulConfig=minimal_config,
        MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig=dram_config,
        experimental=SimpleNamespace(minimal_matmul=minimal_matmul),
        to_memory_config=lambda tensor, _memory: tensor,
        linear=linear,
    )
    env = {"ttnn": ttnn, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), env)
    group = "attention" if name.startswith("self_attn.") else "down" if name == "mlp.down_proj" else "mlp"
    decoder = SimpleNamespace(
        weights={name + ".weight": weight},
        linear_kernels={group: kernel},
        decoding=decoding,
        prefill_l1=False,
        prefill_activation_dtype=activation_dtype,
        residual_dtype="BF16",
        mesh_device=SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10)),
        output_padding=set(),
        geometry={name: {"block": 8, "cores": 8, "readers": 2}},
        banks=8,
        _working_memory=lambda _x, _cores: "WORKING",
    )
    result = env["_linear"](decoder, input_tensor, name)
    assert sys.modules.get("ttnn") is not ttnn, "Test unexpectedly registered its TTNN double as a real import"
    return calls, result


class PrefillMlpBlockSelectionTest(unittest.TestCase):
    def test_p8_bfp8_activation_keeps_mlp_n16(self):
        for name in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            with self.subTest(name=name):
                calls, _ = _call_linear(name=name, activation_dtype="BFP8")
                config = next(kwargs for kind, kwargs in calls if kind == "minimal_config")
                self.assertEqual(config["N_block_size"], 16)

    def test_p10_bf16_activation_selects_n8_for_all_mlp_projections(self):
        for name in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            with self.subTest(name=name):
                calls, _ = _call_linear(name=name, activation_dtype="BF16")
                config = next(kwargs for kind, kwargs in calls if kind == "minimal_config")
                self.assertEqual(config["N_block_size"], 8)

    def test_attention_geometry_does_not_follow_mlp_activation_policy(self):
        calls, _ = _call_linear(name="self_attn.o_proj", activation_dtype="BF16")
        config = next(kwargs for kind, kwargs in calls if kind == "minimal_config")
        self.assertEqual(config["N_block_size"], 16)

    def test_existing_bf16_weight_and_fp32_accumulation_guards_still_select_n8(self):
        for options in ({"weight_dtype": "BF16"}, {"fp32": True}):
            with self.subTest(options=options):
                calls, _ = _call_linear(name="mlp.gate_proj", activation_dtype="BFP8", **options)
                config = next(kwargs for kind, kwargs in calls if kind == "minimal_config")
                self.assertEqual(config["N_block_size"], 8)

    def test_decode_continues_using_existing_decode_geometry(self):
        calls, _ = _call_linear(name="mlp.down_proj", activation_dtype="BF16", decoding=True)
        self.assertFalse(any(kind == "minimal_config" for kind, _ in calls))
        decode = next(kwargs for kind, kwargs in calls if kind == "decode_config")
        self.assertEqual(decode["in0_block_w"], 8)
        self.assertEqual(decode["per_core_N"], 2)
        self.assertTrue(any(kind == "decode_linear" for kind, _ in calls))


if __name__ == "__main__":
    unittest.main()
