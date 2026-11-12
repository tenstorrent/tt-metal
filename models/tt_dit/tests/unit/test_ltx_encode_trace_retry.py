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

ROOT = Path(__file__).resolve().parents[2]
VAE = ROOT / "models" / "vae" / "vae_ltx.py"
PIPE = ROOT / "pipelines" / "ltx" / "pipeline_ltx_distilled.py"


def _calls(node, attr):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == attr]


class EncodeTraceRetry(unittest.TestCase):
    def test_tracer_cached_only_after_prepare(self):
        tree = ast.parse(VAE.read_text())
        tries = [t for t in ast.walk(tree) if isinstance(t, ast.Try) and _calls(t, "prepare")]
        self.assertEqual(len(tries), 1)
        prep = tries[0]
        self.assertTrue(any(_calls(h, "release_trace") for h in prep.handlers), "failed prepare must release")
        self.assertTrue(any(isinstance(h.body[-1], ast.Raise) for h in prep.handlers), "failed prepare must re-raise")
        stores = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Subscript) and getattr(t.value, "attr", None) == "_encode_tracers" for t in n.targets)
        ]
        self.assertEqual(len(stores), 1)
        self.assertGreater(stores[0].lineno, prep.end_lineno, "tracer cached before prepare succeeded")

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
