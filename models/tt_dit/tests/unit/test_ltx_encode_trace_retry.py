# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU contract: a failed image-encoder trace prep is not cached and the warmup gate always closes.

A cached unprepared tracer made every later encode of that shape eager and made capture_traces()
raise for every shape; a capture gate left open let a later gen prepare an encode trace after other
persistent allocations (#3354). Source-level check, no TTNN import; run directly with Python.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
VAE = ROOT / "models" / "vae" / "vae_ltx.py"
PIPE = ROOT / "pipelines" / "ltx" / "pipeline_ltx_distilled.py"


def _calls(node, attr):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == attr]


class EncodeTraceRetry(unittest.TestCase):
    def preparation_block(self):
        tree = ast.parse(VAE.read_text())
        encoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LTXVideoEncoder")
        forward = next(n for n in encoder.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        blocks = [
            n
            for n in ast.walk(forward)
            if isinstance(n, ast.If)
            and any(isinstance(t, ast.Attribute) and t.attr == "capture_open" for t in ast.walk(n.test))
            and _calls(n, "prepare")
        ]
        self.assertEqual(len(blocks), 1, "expected the capture-gated preparation block")
        return blocks[0]

    def test_tracer_cached_only_after_prepare(self):
        block = self.preparation_block()
        # The owning try handles preparation failures; nested tries may reset gather mode.
        tries = [t for t in block.body if isinstance(t, ast.Try) and _calls(t, "prepare")]
        self.assertEqual(len(tries), 1)
        prep = tries[0]
        self.assertTrue(any(_calls(h, "release_trace") for h in prep.handlers), "failed prepare must release")
        self.assertTrue(any(isinstance(h.body[-1], ast.Raise) for h in prep.handlers), "failed prepare must re-raise")
        stores = [
            n
            for n in ast.walk(block)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Subscript) and getattr(t.value, "attr", None) == "_encode_tracers" for t in n.targets
            )
        ]
        self.assertEqual(len(stores), 1)
        self.assertGreater(stores[0].lineno, prep.end_lineno, "tracer cached before prepare succeeded")

    def test_failed_prepare_releases_and_same_shape_retries(self):
        code = compile(ast.Module(body=self.preparation_block().body, type_ignores=[]), str(VAE), "exec")
        for error_type in (RuntimeError, KeyboardInterrupt):
            for release_fails in (False, True):
                with self.subTest(error=error_type.__name__, release_fails=release_fails):
                    error = error_type("prepare failed")
                    tracer = Mock()
                    tracer.prepare.side_effect = error
                    if release_fails:
                        tracer.release_trace.side_effect = RuntimeError("release failed")
                    encoder = SimpleNamespace(
                        encode_device=Mock(),
                        mesh_device=object(),
                        _set_device_gather=Mock(),
                        _encode_tracers={},
                        _encode_logical_hw={},
                        _last_logical_hw=(3, 4),
                    )
                    sample = object()
                    namespace = dict(
                        self=encoder,
                        Tracer=Mock(return_value=tracer),
                        logger=Mock(),
                        key="same-shape",
                        sample_tt=sample,
                        logical_h=3,
                        logical_w=4,
                    )
                    with self.assertRaises(error_type) as raised:
                        exec(code, namespace)
                    self.assertIs(raised.exception, error, "cleanup must preserve the primary error")
                    tracer.prepare.assert_called_once_with(sample, 3, 4)
                    tracer.release_trace.assert_called_once_with()
                    self.assertEqual(encoder._encode_tracers, {})
                    self.assertEqual(encoder._encode_logical_hw, {})
                    self.assertEqual([c.args[0] for c in encoder._set_device_gather.call_args_list], [True, False])
                    retry = Mock()
                    namespace["Tracer"].return_value = retry
                    exec(code, namespace)
                    retry.prepare.assert_called_once_with(sample, 3, 4)
                    retry.release_trace.assert_not_called()
                    self.assertIs(encoder._encode_tracers["same-shape"], retry)
                    self.assertEqual(encoder._encode_logical_hw["same-shape"], (3, 4))
                    self.assertIs(namespace["sample_tt"], retry.prepare.return_value)
                    self.assertEqual([c.args[0] for c in encoder._set_device_gather.call_args_list], [True, False] * 2)

    def test_capture_gate_closed_in_finally(self):
        tree = ast.parse(PIPE.read_text())
        closes = [
            t
            for t in ast.walk(tree)
            if isinstance(t, ast.Try)
            and any(
                isinstance(s, ast.Assign)
                and getattr(s.targets[0], "attr", None) == "capture_open"
                and getattr(s.value, "value", None) is False
                for s in t.finalbody
            )
        ]
        self.assertEqual(len(closes), 1, "capture_open=False must sit in a finally")
        self.assertTrue(_calls(closes[0], "_warmup_encode"))


if __name__ == "__main__":
    unittest.main()
