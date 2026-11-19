# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""AST-extracted real generator/formatter tests without a TTNN import."""

import ast
import unittest
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch

from models.autoports.qwen_qwen3_8_flash_next.tt.host_sampling import HostSamplingParams


class SamplingParamsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = Path(__file__).resolve().parents[1]
        common = model.parents[1] / "common/sampling/generator.py"
        namespace = {
            "dataclass": dataclass,
            "fields": fields,
            "replace": replace,
            "List": List,
            "torch": torch,
            "clamp": lambda value, lo, hi: min(max(value, lo), hi),
            "HostSamplingParams": HostSamplingParams,
        }
        nodes = [
            node
            for node in ast.parse(common.read_text()).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in ("SamplingParams", "format_sampling_params")
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(common), "exec"), namespace)
        generator = model / "tt/generator.py"
        owner = next(
            node
            for node in ast.parse(generator.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "QwenGenerator"
        )
        setter = next(
            node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "set_sampling_params"
        )
        exec(compile(ast.Module(body=[setter], type_ignores=[]), str(generator), "exec"), namespace)
        cls.setter = staticmethod(namespace[setter.name])
        configure = next(
            node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_configure_sampling"
        )
        exec(compile(ast.Module(body=[configure], type_ignores=[]), str(generator), "exec"), namespace)
        cls.configure = staticmethod(namespace[configure.name])
        cls.formatter = staticmethod(namespace["format_sampling_params"])
        cls.params = namespace["SamplingParams"]

    def setUp(self):
        self.calls = []
        self.gen = SimpleNamespace(max_batch_size=4, sampler=SimpleNamespace(reset_sampling_params=self.calls.append))
        self.setter(self.gen)

    def test_required_defaults_and_initial_greedy(self):
        formatted = self.gen.formatted_sampling_params
        self.assertEqual(formatted.top_k, [1] * 32)
        self.assertEqual(formatted.top_p, [0.0] * 32)
        self.assertEqual(self.gen.request_seeds, [None] * 32)

    def test_plugin_ducktype_scatter_and_none_logprobs(self):
        self.setter(
            self.gen,
            self.params(temperature=[1.0] * 4, top_k=16, top_p=0.9, seed=[101, 102, 103, 104], num_logprobs=None),
        )
        incoming = SimpleNamespace(
            temperature=[0.7, 1.5], top_k=[16, 32], top_p=[0.85, 0.95], seed=[-4, 2**61], num_logprobs=None
        )
        self.setter(self.gen, incoming, slots=[3, 1])
        result = self.gen.formatted_sampling_params
        self.assertEqual(self.gen.request_seeds[:4], [101, 2**61, 103, -4])
        self.assertEqual(result.top_k[:4], [16, 32, 16, 16])
        self.assertAlmostEqual(result.temperature[3], 1 / 0.7)
        self.assertAlmostEqual(result.temperature[1], 1 / 1.5)
        self.assertEqual(result.num_logprobs, [0] * 32)

    def test_scalar_params_broadcast_over_explicit_slots(self):
        self.setter(self.gen, self.params(temperature=0.8, top_k=12, top_p=0.7), slots=[2, 0])
        value = self.gen.formatted_sampling_params
        self.assertEqual(value.top_k[:4], [12, 1, 12, 1])
        self.assertAlmostEqual(value.temperature[0], 1 / 0.8)
        self.assertAlmostEqual(value.temperature[2], 1 / 0.8)

    def test_decode_order_helper_seeds_both_physical_rows(self):
        from .decode_submission_order_check import set_matched_greedy_params

        self.gen.set_sampling_params = lambda *args, **kwargs: self.setter(self.gen, *args, **kwargs)
        # Demonstrate the actual old setter contract: an unscoped scalar is B1.
        self.setter(self.gen, self.params(temperature=0.0, top_k=1, top_p=1.0, seed=0))
        self.assertEqual(self.gen.request_seeds[:2], [0, None])
        set_matched_greedy_params(self.gen, self.params)
        self.assertEqual(self.gen.request_seeds[:2], [0, 0])
        self.assertEqual(self.gen.formatted_sampling_params.top_k[:2], [1, 1])
        set_matched_greedy_params(self.gen, self.params)
        self.assertEqual(self.gen.request_seeds[:2], [0, 0])

    def test_invalid_requests_reject_before_sampler_mutation(self):
        invalid = [
            dict(top_k=41),
            dict(top_k=64),
            dict(top_k=0),
            dict(top_k=-1),
            dict(top_k=2.5),
            dict(top_p=1.2),
            dict(top_p=-0.1),
            dict(temperature=float("inf")),
            dict(temperature=[1.0, float("nan")], top_k=[16, 16], top_p=[0.9, 0.9]),
            dict(temperature=[1.0, 1.0], top_k=[16, 16], top_p=[0.9, 0.9], seed=[1, 0.5]),
            dict(frequency_penalty=0.5),
            dict(enable_log_probs=True),
        ]
        before = self.gen.formatted_sampling_params
        calls = len(self.calls)
        for patch in invalid:
            kwargs = dict(temperature=1.0, top_k=16, top_p=0.9)
            kwargs.update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.setter(self.gen, self.params(**kwargs))
            self.assertIs(self.gen.formatted_sampling_params, before)
            self.assertEqual(len(self.calls), calls)

    def test_request_lengths_and_slot_indices_reject_before_mutation(self):
        calls = len(self.calls)
        for slots in ([1, 1], [4], [1.0], []):
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                self.setter(self.gen, self.params(temperature=1.0, top_k=16, top_p=0.9), slots=slots)
        with self.assertRaises(ValueError):
            self.setter(self.gen, self.params(temperature=[1.0, 1.0], top_k=[16], top_p=[0.9, 0.9]), slots=[0, 1])
        self.assertEqual(len(self.calls), calls)

    def test_wide_admission_preserves_every_new_logical_k(self):
        for k in range(33, 41):
            self.setter(self.gen, self.params(temperature=1.0, top_k=k, top_p=0.95, seed=42), slots=[2])
            self.assertEqual(self.gen.formatted_sampling_params.top_k[2], k)
            self.assertEqual(self.gen.formatted_sampling_params.top_p[2], 0.95)
            self.assertEqual(self.gen.request_seeds[2], 42)
        default = self.formatter(self.params(temperature=1.0, top_k=40, top_p=0.95), 32)
        wide = self.formatter(self.params(temperature=1.0, top_k=40, top_p=0.95), 32, max_top_k=40)
        self.assertEqual(default.top_k[0], 32)
        self.assertEqual(wide.top_k[0], 40)

    def test_inactive_wide_metadata_is_neutral_and_reactivated_without_loss(self):
        self.configure(self.gen, (0,))
        calls = len(self.calls)
        self.setter(self.gen, self.params(temperature=1.0, top_k=40, top_p=0.95), slots=[1])
        self.assertEqual(len(self.calls), calls, "setter must defer uploads until actual sampling membership is known")
        self.configure(self.gen, (0,))
        self.assertEqual(len(self.calls), calls, "inactive metadata must not cause a redundant upload")
        self.assertEqual(self.gen.formatted_sampling_params.top_k[1], 40)
        self.assertEqual(self.gen._effective_sampling_params.top_k, [1] * 32)
        self.configure(self.gen, (0, 1))
        self.assertEqual(self.calls[-1].top_k[:4], [1, 40, 1, 1])
        self.assertEqual(self.calls[-1].top_p[1], 0.95)
        self.configure(self.gen, (0,))
        self.assertEqual(self.calls[-1].top_k, [1] * 32)
        self.assertEqual(self.gen.formatted_sampling_params.top_k[1], 40)
        configuration = self.gen._sampling_configuration
        calls = len(self.calls)
        for _ in range(10):
            self.configure(self.gen, (0,))
        self.assertIs(self.gen._sampling_configuration, configuration)
        self.assertEqual(len(self.calls), calls)

    def test_equal_scheduler_params_do_not_refresh_sampling_buffers(self):
        params = self.params(temperature=[1.0, 1.0], top_k=[20, 40], top_p=[0.9, 0.95], seed=[42, 43])
        self.setter(self.gen, params)
        self.configure(self.gen, (0, 1))
        calls = len(self.calls)
        version = self.gen._sampling_params_version
        self.setter(self.gen, params)
        self.configure(self.gen, (1, 0))
        self.assertEqual(self.gen._sampling_params_version, version)
        self.assertEqual(len(self.calls), calls)

    def test_explicit_host_mode_keeps_device_params(self):
        before = self.gen.formatted_sampling_params
        calls = len(self.calls)
        self.setter(
            self.gen,
            self.params(temperature=0.7, top_k=100, top_p=0.95, seed=-9, frequency_penalty=0.2),
            host_sampling=True,
        )
        self.assertTrue(self.gen.sampling_host_only)
        self.assertEqual(self.gen.host_sampling_params.seed, -9)
        self.assertIs(self.gen.formatted_sampling_params, before)
        self.assertEqual(len(self.calls), calls)


if __name__ == "__main__":
    unittest.main()
