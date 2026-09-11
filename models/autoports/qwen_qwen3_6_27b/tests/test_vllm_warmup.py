# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exercise startup phase ordering without importing vLLM or opening a device."""

import ast
import os
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch


def adapter():
    path = Path(__file__).resolve().parents[1] / "tt/generator_vllm.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen36ForCausalLM")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and "warmup" in node.name]
    namespace = dict(
        torch=torch,
        os=os,
        time=time,
        ttnn=types.SimpleNamespace(synchronize_device=Mock()),
        SamplingParams=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    module = ast.Module(body=methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    result = types.SimpleNamespace(
        max_seq_len=4608,
        mesh_device=None,
        _prefill_warmup_complete=False,
        _decode_compile_complete=False,
        _decode_warmup_complete=False,
        _warmup_decode_inputs=None,
        _decode_ready=True,
    )
    result.generator = types.SimpleNamespace(
        MAX_PREFILL_TOKENS=262144,
        page_table_host=torch.zeros((32, 72), dtype=torch.int32),
        model=types.SimpleNamespace(batch=32),
        setup_token_out_decode=Mock(),
        warmup_token_out_decode=Mock(),
        _release_traces=Mock(),
    )
    result._require_generator = lambda: result.generator
    result.prefill_forward = Mock(return_value=((torch.tensor([42]), None), torch.zeros(1)))
    for method in methods:
        setattr(result, method.name, types.MethodType(namespace[method.name], result))
    return result


class WarmupTests(unittest.TestCase):
    def test_ragged_lengths_keep_logical_length_and_pad_physical_input(self):
        model = adapter()
        with patch.dict(os.environ, {"QWEN36_WARMUP_PREFILL_LENGTHS": "65,128,65"}):
            model.warmup_model_prefill(kv_cache="cache")
        calls = model.prefill_forward.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(tuple(calls[0].args[0].shape), (1, 96))
        self.assertEqual(calls[0].kwargs["prompt_lens"], [65])
        self.assertEqual(calls[0].kwargs["empty_slots"], [0])
        self.assertEqual(calls[0].kwargs["page_table"][0].tolist(), list(range(72)))

    def test_compile_and_capture_phases_are_separate_and_idempotent(self):
        model = adapter()
        with patch.dict(os.environ, {"QWEN36_WARMUP_PREFILL_LENGTHS": "128"}):
            model.warmup_model_prefill(kv_cache="cache")
            model.warmup_model_decode(kv_cache="cache", enable_trace=False)
            model.warmup_model_decode(kv_cache="cache", enable_trace=False)
            model.warmup_model_prefill(kv_cache="cache", enable_trace=True)
            model.warmup_model_decode(kv_cache="cache", enable_trace=True)
            model.warmup_model_decode(kv_cache="cache", enable_trace=True)
        model.generator.warmup_token_out_decode.assert_called_once()
        model.generator.setup_token_out_decode.assert_called_once()
        model.prefill_forward.assert_called_once()
        self.assertFalse(model._decode_ready, "real scheduler inputs must still be reloaded")
        tokens, positions = model.generator.setup_token_out_decode.call_args.args
        self.assertEqual(tokens[0].item(), 42)
        self.assertEqual(positions.tolist(), [128] + [-1] * 31)

    def test_bad_length_fails_before_any_device_work(self):
        for value in ("0", "4608", "-1", "abc"):
            model = adapter()
            with patch.dict(os.environ, {"QWEN36_WARMUP_PREFILL_LENGTHS": value}):
                with self.assertRaises(ValueError):
                    model.warmup_model_prefill(kv_cache="cache")
            model.prefill_forward.assert_not_called()

    def test_capture_phase_rejects_missing_compile_phase(self):
        model = adapter()
        with self.assertRaises(RuntimeError):
            model.warmup_model_prefill(kv_cache="cache", enable_trace=True)
        with self.assertRaises(RuntimeError):
            model.warmup_model_decode(kv_cache="cache", enable_trace=True)

    def test_failed_prefill_does_not_mark_startup_ready(self):
        model = adapter()
        model.prefill_forward.side_effect = RuntimeError("warmup failed")
        with patch.dict(os.environ, {"QWEN36_WARMUP_PREFILL_LENGTHS": "128"}):
            with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                model.warmup_model_prefill(kv_cache="cache")
        self.assertFalse(model._prefill_warmup_complete)
        model.generator.setup_token_out_decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
