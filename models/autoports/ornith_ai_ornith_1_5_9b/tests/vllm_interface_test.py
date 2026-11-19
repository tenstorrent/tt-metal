# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host boundary regressions using actual adapter and parameter definitions.

Extract their AST definitions without importing TTNN or initializing vLLM. This
keeps host contract checks independent of a concurrent native rebuild.
"""

import ast
import importlib.util
import json
import os
import time
import unittest
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[2]


def actual_class(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


plugin_spec = importlib.util.find_spec("vllm_tt_plugin")
if plugin_spec is None or not plugin_spec.submodule_search_locations:
    raise ImportError("Installed standalone vllm_tt_plugin is required for interface contract checks")
plugin_source = Path(next(iter(plugin_spec.submodule_search_locations))) / "model_input.py"
parameter_namespace = {"torch": torch, "dataclass": dataclass}
TTSamplingParams = actual_class(plugin_source, "TTSamplingParams", parameter_namespace)
SamplingParams = actual_class(REPO / "models/common/sampling/generator.py", "SamplingParams", parameter_namespace)
OrnithForCausalLM = actual_class(
    MODEL / "tt/generator_vllm.py",
    "OrnithForCausalLM",
    {
        "torch": torch,
        "json": json,
        "os": os,
        "time": time,
        "fields": fields,
        "SamplingParams": SamplingParams,
        "ttnn": SimpleNamespace(FabricConfig=SimpleNamespace(FABRIC_1D="FABRIC_1D")),
    },
)


class FakeGenerator:
    def __init__(self):
        self.cache = object()
        self.params = None
        self.counters = {}
        self.prefill_calls = []

    def prefill_forward(self, tokens, **kwargs):
        self.prefill_calls.append(kwargs)
        self.params = kwargs["sampling_params"]
        if self.params is None:
            return torch.zeros((tokens.shape[0], 1, 8))
        assert isinstance(self.params, SamplingParams)
        assert hasattr(self.params, "log_probs_enable_temperature")
        return torch.tensor([[17]])

    def decode_forward(self, tokens, positions, **kwargs):
        self.params = kwargs["sampling_params"]
        assert isinstance(self.params, SamplingParams)
        assert "rope_deltas_all_users" not in kwargs
        return "device_submission"


class InterfaceTest(unittest.TestCase):
    def setUp(self):
        self.adapter = OrnithForCausalLM()
        self.adapter.generator = FakeGenerator()
        self.adapter.max_batch_size = 3
        self.adapter.max_seq_len = 128
        self.params = TTSamplingParams(
            temperature=[0.0],
            top_k=[1],
            top_p=[1.0],
            seed=[42],
            num_logprobs=[0],
            enable_log_probs=[False],
        )

    def test_plugin_prefill_parameters_and_text_rope(self):
        result, deltas = self.adapter.prefill_forward(
            torch.tensor([[1, 2, 3]]),
            torch.zeros((1, 8), dtype=torch.int32),
            self.adapter.generator.cache,
            [3],
            original_prompt_lens=[3],
            sampling_params=self.params,
            pixel_values=[None],
            image_grid_thw=[None],
        )
        self.assertEqual(result.tolist(), [[17]])
        self.assertTrue(torch.all(deltas == 0))
        self.assertEqual(self.adapter.generator.params.seed, [42])
        self.assertEqual(self.adapter.generator.prefill_calls[-1]["original_prompt_lens"], [3])

    def test_plugin_decode_parameters_and_text_rope(self):
        result = self.adapter.decode_forward(
            object(),
            object(),
            object(),
            self.adapter.generator.cache,
            sampling_params=self.params,
            rope_deltas_all_users=[0, 0, 0],
            reload_inputs=False,
        )
        self.assertEqual(result, "device_submission")

    def test_host_prefill_is_already_host_logits(self):
        result, deltas = self.adapter.prefill_forward(
            torch.tensor([[1, 2, 3]]),
            torch.zeros((1, 8), dtype=torch.int32),
            self.adapter.generator.cache,
            [3],
            original_prompt_lens=[3],
        )
        self.assertEqual(tuple(result.shape), (1, 1, 8))
        self.assertEqual(tuple(deltas.shape), (1, 1))

    def test_real_media_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.prefill_forward(
                torch.tensor([[1]]),
                None,
                self.adapter.generator.cache,
                [1],
                original_prompt_lens=[1],
                sampling_params=self.params,
                pixel_values=[torch.ones(1)],
            )

    def test_nonzero_text_rope_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.decode_forward(
                object(),
                object(),
                object(),
                self.adapter.generator.cache,
                sampling_params=self.params,
                rope_deltas_all_users=[1],
            )

    def test_original_boundaries_are_forwarded_without_inference(self):
        for original in ([1], [3], [10]):
            with self.subTest(original=original):
                self.adapter.prefill_forward(
                    torch.tensor([[1, 2, 3]]),
                    None,
                    self.adapter.generator.cache,
                    [3],
                    original_prompt_lens=original,
                )
                self.assertEqual(self.adapter.generator.prefill_calls[-1]["original_prompt_lens"], original)

    def test_missing_or_invalid_original_metadata_fails_before_generator(self):
        for original in (None, (), (3,), torch.tensor([3]), [True], [3.0], [], [3, 3], [-1], [129]):
            with self.subTest(original=original):
                with self.assertRaises((TypeError, ValueError)):
                    self.adapter.prefill_forward(
                        torch.tensor([[1, 2, 3]]),
                        None,
                        self.adapter.generator.cache,
                        [3],
                        original_prompt_lens=original,
                    )
                self.assertEqual(self.adapter.generator.prefill_calls, [])

    def test_mixed_final_intermediate_preserves_known_horizons(self):
        self.adapter.prefill_forward(
            torch.ones((2, 5), dtype=torch.int64),
            None,
            self.adapter.generator.cache,
            [3, 5],
            original_prompt_lens=[11, 5],
        )
        call = self.adapter.generator.prefill_calls[-1]
        self.assertEqual(call["request_lens"], [11, 5])
        self.assertEqual(call["discard_logits"], [True, False])

    def test_replay_beyond_original_keeps_unknown_horizon(self):
        self.adapter.prefill_forward(
            torch.ones((2, 5), dtype=torch.int64),
            None,
            self.adapter.generator.cache,
            [3, 5],
            original_prompt_lens=[11, 4],
        )
        call = self.adapter.generator.prefill_calls[-1]
        self.assertIsNone(call["request_lens"])
        self.assertEqual(call["discard_logits"], [True, False])

    def test_original_boundary_and_inactive_rows_do_not_discard(self):
        self.adapter.prefill_forward(
            torch.ones((2, 5), dtype=torch.int64),
            None,
            self.adapter.generator.cache,
            [0, 5],
            original_prompt_lens=[0, 5],
        )
        call = self.adapter.generator.prefill_calls[-1]
        self.assertEqual(call["request_lens"], [0, 5])
        self.assertEqual(call["discard_logits"], [False, False])

    def test_original_boundary_probe_uses_adapter_horizon_translation(self):
        # Execute the diagnostic's actual nested helper through the actual
        # compatibility wrapper and adapter; no generator/device import.
        probe = ast.parse((MODEL / "tests/vllm_original_boundary_probe.py").read_text())
        helpers = [
            node for node in ast.walk(probe) if isinstance(node, ast.FunctionDef) and node.name in ("packed", "prefill")
        ]
        builder = ast.parse((MODEL / "tests/vllm_adapter_decode_probe.py").read_text())
        wrapper = next(
            node
            for node in ast.walk(builder)
            if isinstance(node, ast.FunctionDef) and node.name == "prefill_with_metadata"
        )
        namespace = {"torch": torch, "adapter": self.adapter, "params": self.params, "patch": patch}
        exec(compile(ast.Module(body=[wrapper] + helpers, type_ignores=[]), "probe_helpers", "exec"), namespace)
        namespace["gen"] = SimpleNamespace(
            prefill_forward=namespace["prefill_with_metadata"],
            page_table=torch.zeros((3, 8), dtype=torch.int32),
            cache=self.adapter.generator.cache,
            sampling_state=SimpleNamespace(reset_history=lambda **kwargs: None),
        )
        for originals, expected_horizon, expected_discard in (
            ([6, 4], [6, 4], [True, False]),
            ([2, 3], None, [False, False]),
        ):
            with self.subTest(originals=originals):
                namespace["prefill"]([[1, 2, 3], [4, 5, 6, 7]], originals, slots=(0, 1), device=False)
                forwarded = self.adapter.generator.prefill_calls[-1]
                self.assertEqual(forwarded["original_prompt_lens"], originals)
                self.assertEqual(forwarded["request_lens"], expected_horizon)
                self.assertEqual(forwarded["discard_logits"], expected_discard)


if __name__ == "__main__":
    unittest.main()
