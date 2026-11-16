# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run directly with Python: CPU-only production AST extraction, no TTNN imports.

Exercises real formatter/decode-reload code and the production primitive call.
This intentionally does not claim candidate allocation or device correctness.
"""

import ast
import dataclasses
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
GENERATOR = ROOT / "models/common/sampling/generator.py"
SAMPLER = ROOT / "models/common/sampling/tt_sampling.py"


def extract_formatter(source):
    tree = ast.parse(source)
    names = {"SamplingParams", "format_sampling_params"}
    body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = dict(
        dataclass=dataclasses.dataclass,
        fields=dataclasses.fields,
        replace=dataclasses.replace,
        List=List,
        clamp=lambda value, low, high: min(max(value, low), high),
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(GENERATOR), "exec"), namespace)
    return namespace


def class_method(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


class SamplingCapacityHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = extract_formatter(GENERATOR.read_text())

    def test_default32_matches_expected_field_behavior(self):
        # Explicit field expectations remain meaningful after commits and without Git.
        # Cases cover scalar greedy, scalar sampling and mixed lanes; padding stays greedy.
        cases = (
            (0.0, [1.0], [0.0], [False]),
            (0.7, [1 / 0.7], [0.95], [True]),
            ([0.0, 0.7, 1.0], [1.0, 1 / 0.7, 1.0], [0.0, 0.95, 0.95], [False, True, True]),
        )
        capped_k = {-1: 32, 0: 32, 1: 1, 20: 20, 32: 32, 33: 32, 40: 32, 64: 32, 248320: 32}
        for temperature, inverse_temperature, nucleus, stochastic in cases:
            active = len(inverse_temperature)
            for k, sampled_k in capped_k.items():
                with self.subTest(temperature=temperature, k=k):
                    params = dict(temperature=temperature, top_k=k, top_p=0.95, presence_penalty=0.75, seed=42)
                    expected = dict(
                        temperature=inverse_temperature + [1.0] * (32 - active),
                        top_k=[sampled_k if random else 1 for random in stochastic] + [1] * (32 - active),
                        top_p=nucleus + [0.0] * (32 - active),
                        presence_penalty=[0.75] * active + [0.0] * (32 - active),
                        frequency_penalty=[0.0] * 32,
                        repetition_penalty=[1.0] * 32,
                        seed=[42] + [None] * 31,
                        enable_log_probs=[False] * 32,
                        num_logprobs=[0] * 32,
                        log_probs_enable_temperature=[False] * 32,
                    )
                    new = self.current["format_sampling_params"](self.current["SamplingParams"](**params), 32)
                    explicit = self.current["format_sampling_params"](
                        self.current["SamplingParams"](**params), 32, max_top_k=32
                    )
                    self.assertEqual(dataclasses.asdict(new), expected)
                    self.assertEqual(dataclasses.asdict(explicit), expected)

    def test_explicit64_preserves_k40_and_greedy_sentinel(self):
        params = self.current["SamplingParams"](
            temperature=[0.7, 0.0, 0.7, 0.7, 0.7], top_k=[40, 40, -1, 65, 33], top_p=0.95, seed=[42]
        )
        before = dataclasses.asdict(params)
        formatted = self.current["format_sampling_params"](params, 32, max_top_k=64)
        self.assertEqual(formatted.top_k, [40, 1, 64, 64, 33] + [1] * 27)
        self.assertEqual(formatted.top_p[1], 0.0)
        self.assertEqual(formatted.temperature[1], 1.0)
        self.assertEqual(formatted.seed, [42] + [None] * 31)
        self.assertEqual(dataclasses.asdict(params), before)

    def test_invalid_capacity_fails_before_parameter_access(self):
        for capacity in (0, 16, 40, 128, None, True, 32.0, "64"):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(ValueError, "max_top_k must be 32 or 64"):
                self.current["format_sampling_params"](None, 32, max_top_k=capacity)

    def test_constructor_validates_capacity_before_device_access(self):
        init = class_method(SAMPLER, "TTSampling", "__init__")
        cls = ast.ClassDef(name="TTSampling", bases=[], keywords=[], body=[init], decorator_list=[])
        namespace = {}
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(SAMPLER), "exec"), namespace
        )
        for capacity in (0, 16, 40, 128, None, True, 32.0, "64"):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(ValueError, "max_top_k must be 32 or 64"):
                namespace["TTSampling"](mesh_device=None, tt_ccl=None, args=SimpleNamespace(max_top_k=capacity))

    def test_decode_reload_keeps_capacity_single_and_row_sharded(self):
        method = class_method(GENERATOR, "SamplingGenerator", "apply_decode_state")
        namespace = dict(self.current, torch=SimpleNamespace(Tensor=object))
        namespace["SAMPLING_PARAM_FIELDS"] = tuple(
            field.name for field in dataclasses.fields(namespace["SamplingParams"])
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(GENERATOR), "exec"), namespace)
        for capacity in (32, 64):
            for chunks in (1, 2):
                with self.subTest(capacity=capacity, chunks=chunks):
                    owner = SimpleNamespace(
                        tt_sampling=SimpleNamespace(max_batch_size=32, max_top_k=capacity),
                        validate_decode_state_commands=Mock(),
                        commit_decode_state_commands=Mock(),
                        reset_sampling_params=Mock(),
                    )
                    params = namespace["SamplingParams"](temperature=[0.7] * 32, top_k=40, top_p=0.95)
                    namespace["apply_decode_state"](
                        owner, [params] * chunks, reload_sampling_params=True, reset_sampling_state=False
                    )
                    formatted = owner.reset_sampling_params.call_args.args[0]
                    self.assertEqual(formatted.top_k, [min(40, capacity)] * (32 * chunks))

    def test_actual_primitive_call_passes_only_explicit64(self):
        tree = ast.parse(SAMPLER.read_text())
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ttnn"
            and node.func.attr == "sampling"
        )
        code = compile(ast.Expression(body=call), str(SAMPLER), "eval")
        for capacity in (32, 64):
            sampling = Mock(return_value="output")
            owner = SimpleNamespace(
                max_top_k=capacity, k_tensor="k", p_tensor="p", temp_tensor="temp", _sampling_sub_core_grids="grid"
            )
            namespace = dict(
                ttnn=SimpleNamespace(sampling=sampling),
                self=owner,
                sampling_values="values",
                topk_global_indices_interleaved_untilised="indices",
                tt_out_tok="out",
            )
            self.assertEqual(eval(code, namespace), "output")
            expected = dict(k="k", p="p", temp="temp", sub_core_grids="grid", output_tensor="out")
            if capacity == 64:
                expected["max_top_k"] = 64
            sampling.assert_called_once_with("values", "indices", **expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
