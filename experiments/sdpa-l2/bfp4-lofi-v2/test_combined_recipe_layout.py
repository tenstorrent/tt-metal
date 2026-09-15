"""Stdlib descriptor/preprocessing schedule tests; no TTNN/device execution."""

import ast
import contextlib
import io
import hashlib
import math
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from test_vtransposed_layout import Device, Tensor, environment

HERE = Path(__file__).resolve().parent


def combined_environment():
    scope = environment("Vtransposed_fullchip.py")
    scope["__file__"] = str(HERE / "combined_recipe_fullchip.py")
    events, quantizers, rotations = [], [], []

    def quantizer(device, src, **kwargs):
        index = len(quantizers) % 3
        name = ("q_quantization", "k_quantization", "v_quantization")[index]
        quantizers.append(dict(name=name, source=src, options=kwargs))
        return Tensor(src.shape, kwargs.get("output_format", "b4")), lambda: events.append(name), {}

    def rotate(device, src, width):
        assert width == 16
        name = ("q_rotation", "k_rotation")[len(rotations) % 2]
        rotations.append(src)
        return Tensor(src.shape, "bf16"), lambda: events.append(name), dict(block_size=16)

    real_transpose_builder = scope["build_transpose"]

    def transpose(device, src, cores):
        out, invoke, metadata = real_transpose_builder(device, src, cores)

        def call():
            events.append("v_transpose")
            invoke()

        return out, call, metadata

    scope.update(
        PREP=NS(build=quantizer),
        B4_PREP=NS(build=quantizer),
        ADAPT=NS(build=quantizer),
        HADAMARD=NS(build=rotate),
        build_transpose=transpose,
    )
    tree = ast.parse((HERE / "combined_recipe_fullchip.py").read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build"]
    exec(compile(tree, "combined_recipe_fullchip.py", "exec"), scope)
    return scope, events, quantizers


class CombinedRecipeLayout(unittest.TestCase):
    def test_source_pins(self):
        root = HERE.parents[2]

        def extract(path, name, **extra):
            tree = ast.parse(path.read_text())
            tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
            scope = dict(Path=Path, ROOT=root, HERE=HERE, __file__=str(path), hashlib=hashlib, **extra)
            exec(compile(tree, str(path), "exec"), scope)
            return scope[name]

        baseline = environment("Vtransposed_fullchip.py")
        hadamard = extract(HERE / "hadamard_preprocess.py", "source_files")
        adaptive = extract(HERE / "adaptive_bfp4_round.py", "source_pins", DIRECTORY=HERE / "adaptive_bfp4_round")
        sources = extract(
            HERE / "combined_recipe_fullchip.py",
            "source_files",
            BASE=NS(source_files=baseline["source_files"]),
            HADAMARD=NS(source_files=hadamard),
            ADAPT=NS(source_pins=adaptive),
        )("fast_bf16")
        self.assertTrue(all(path.is_file() for path in sources))
        for path in (
            "combined_recipe_fullchip.py",
            "Vtransposed_fullchip.py",
            "hadamard_preprocess.py",
            "adaptive_bfp4_round/compute.cpp",
            "vtransposed/pv_transpose.hpp",
        ):
            self.assertIn(HERE / path, sources)
        self.assertIn(HERE.parent / "frontier-accuracy-v1/run.py", sources)

    def test_independent_flags_and_stage_order(self):
        baseline = environment("Vtransposed_fullchip.py")
        inputs = [Tensor([1, 2, 1024, 128]) for _ in range(3)]
        for formats in ("b4_b4", "b8_b4", "b4_b8", "b8_b8"):
            for adaptive in (("none", "baseline", "minus", "pm") if formats.endswith("b4") else ("none",)):
                for h16 in (False, True):
                    for axis in (False, True):
                        for grid7 in (False, True):
                            scope, events, quantizers = combined_environment()
                            args = NS(
                                destination="fast_bf16",
                                denom_only=False,
                                kv_formats=formats,
                                length=1024,
                                heads=2,
                                cores=4,
                                check_preprocess=False,
                                read_barrier_tiles=2,
                                grid7_exp=grid7,
                                v_transposed=axis,
                                h16=h16,
                                adaptive_v=adaptive,
                            )
                            with contextlib.redirect_stdout(io.StringIO()):
                                control = baseline["build"](Device(), args, inputs)
                                result = scope["build"](Device(), args, inputs)
                            for field in (
                                "cb_audit",
                                "input_slots",
                                "assignments",
                                "q_chunk",
                                "k_chunk",
                                "head_dim",
                                "defines",
                            ):
                                self.assertEqual(control[-1][field], result[-1][field])
                            self.assertEqual(result[-1]["input_slots"], 2)
                            self.assertEqual(result[-1]["score_scale"], 1 / (math.sqrt(128) * (16 if h16 else 1)))
                            self.assertEqual(result[-1]["adaptive_v"], adaptive)
                            self.assertFalse(result[-1]["adaptive_k"])
                            expected_stages = ["q_rotation", "k_rotation"] if h16 else []
                            expected_stages += ["v_transpose"] if axis else []
                            expected_stages += ["q_quantization", "k_quantization", "v_quantization"]
                            self.assertEqual(list(result[4].stages), expected_stages)
                            events.clear()
                            result[4]()
                            self.assertEqual(events, expected_stages)
                            self.assertEqual(
                                quantizers[2]["source"].shape, (1, 2, 128, 1024) if axis else (1, 2, 1024, 128)
                            )
                            self.assertNotIn("search", quantizers[0]["options"])
                            self.assertNotIn("search", quantizers[1]["options"])
                            self.assertEqual(quantizers[2]["options"].get("search", "none"), adaptive)
                            result[3]()
                            descriptor = scope["calls"][-1]
                            encoded = struct.unpack("I", struct.pack("f", result[-1]["score_scale"]))[0]
                            self.assertEqual(descriptor.kernels[2].compile_time_args, [2, encoded, 8])

    def test_rejects_non_bf16_full_compensation_and_adaptive_v8(self):
        scope, _, _ = combined_environment()
        args = NS(destination="fast_bf16", denom_only=False, adaptive_v="pm", kv_formats="b4_b8")
        with self.assertRaisesRegex(AssertionError, "Adaptive V requires"):
            scope["build"](Device(), args, [])
        args.destination = "main_bf16"
        with self.assertRaises(AssertionError):
            scope["build"](Device(), args, [])
        args.destination, args.denom_only = "fast_bf16", True
        with self.assertRaises(AssertionError):
            scope["build"](Device(), args, [])


if __name__ == "__main__":
    unittest.main()
