# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU contract for the Gemma encoder pairs' ``capture_trace``; no TTNN import.

Run directly with Python to avoid device-aware repository conftest imports. The production
``capture_trace``/``encode`` methods run on a shell whose device graph only records ``traced``.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
GEMMA4 = ROOT / "encoders" / "gemma4" / "encoder_pair.py"
GEMMA3 = ROOT / "encoders" / "gemma3" / "encoder_pair.py"


def _methods(path, names):
    tree = ast.parse(path.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names), f"{path.name}: expected {names}"
    for node in nodes:
        node.decorator_list = []
    fake_ttnn = SimpleNamespace(
        uint32=None,
        ROW_MAJOR_LAYOUT=None,
        from_torch=lambda t, **_: SimpleNamespace(shape=tuple(t.shape)),
        get_device_tensors=lambda t: [t],
        to_torch=lambda t: torch.zeros(1),
    )
    namespace = {"ttnn": fake_ttnn, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _shell(path, *, encoder_trace=True, gate_open=True):
    seq = 8
    mask = torch.ones(1, seq, dtype=torch.long)
    calls = []

    def encode_device(self, *inputs, traced):
        calls.append(traced)
        return "video", "audio"

    pair = SimpleNamespace(
        _encoder_trace=encoder_trace,
        _trace_gate_open=gate_open,
        _trace_captured=False,
        _sequence_length=seq,
        mesh_device=None,
        gemma_encoder=SimpleNamespace(build_attn_mask=lambda *_: None),
        feature_extractor=SimpleNamespace(build_mask=lambda *_: None),
        video_connector=SimpleNamespace(build_indices=lambda *_: (None, None)),
        tokenize=lambda prompt: (torch.zeros(1, seq, dtype=torch.long), mask),
        tokenizer=lambda prompt, **_: SimpleNamespace(
            input_ids=torch.zeros(1, seq, dtype=torch.long), attention_mask=mask
        ),
    )
    pair._encode_device = MethodType(encode_device, pair)
    names = ["capture_trace", "encode"] + (["_encode_prompt_device"] if path == GEMMA3 else [])
    for name, fn in _methods(path, names).items():
        setattr(pair, name, MethodType(fn, pair))
    return pair, calls


class CaptureTraceContract(unittest.TestCase):
    def test_captures_once_when_gate_open(self):
        for path in (GEMMA4, GEMMA3):
            with self.subTest(pair=path.parent.name):
                pair, calls = _shell(path)
                pair.capture_trace()
                pair.capture_trace()
                # Capture, then one replay to build the input-copy programs.
                self.assertEqual(calls, [True, True])
                pair.encode(["a new prompt"])
                self.assertEqual(calls, [True, True, True])

    def test_no_capture_while_gate_closed(self):
        for path in (GEMMA4, GEMMA3):
            with self.subTest(pair=path.parent.name):
                pair, calls = _shell(path, gate_open=False)
                pair.capture_trace()
                self.assertEqual(calls, [])
                pair.encode(["eager"])
                self.assertEqual(calls, [False])
                self.assertFalse(pair._trace_captured)

    def test_no_capture_without_encoder_trace(self):
        for path in (GEMMA4, GEMMA3):
            with self.subTest(pair=path.parent.name):
                pair, calls = _shell(path, encoder_trace=False)
                pair.capture_trace()
                self.assertEqual(calls, [])

    def test_request_capture_counts(self):
        """A request that captured the trace itself leaves nothing for capture_trace to do."""
        for path in (GEMMA4, GEMMA3):
            with self.subTest(pair=path.parent.name):
                pair, calls = _shell(path)
                pair.encode(["first request"])
                pair.capture_trace()
                self.assertEqual(calls, [True])


if __name__ == "__main__":
    unittest.main()
