# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check the actual packing helper with CPU operations and independent histories."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def convolution(row, history, *args, **kwargs):
    taps, widths = args[:4], args[4:]
    joined = torch.cat([history, row], dim=1)
    out = sum(joined[:, i : i + row.shape[1], :] * tap for i, tap in enumerate(taps))
    return F.silu(out).split(widths, dim=-1)


class PackedPrefillConvTests(unittest.TestCase):
    def test_retained_rows_and_user_isolation(self):
        source = Path(__file__).resolve().parents[2] / "tt" / "decode_conv.py"
        tree = ast.parse(source.read_text())
        fn = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "packed_prefill_conv"
        )
        ops = SimpleNamespace(
            pad=lambda x, pads, value: F.pad(x, tuple(v for pair in reversed(pads) for v in pair), value=value),
            concat=lambda xs, dim: torch.cat(xs, dim=dim),
            reshape=torch.reshape,
            experimental=SimpleNamespace(kda=SimpleNamespace(qkv_causal_conv1d_silu=convolution)),
            QkvCausalConv1dSiluProgramConfig=lambda **kw: kw,
        )
        namespace = {"ttnn": ops}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
        packed = namespace[fn.name]
        torch.manual_seed(19)
        widths = (32, 32, 64)
        for batch in (2, 8, 16):
            for length in (32, 128):
                with self.subTest(batch=batch, length=length):
                    row = torch.randn(batch, length, sum(widths))
                    history = torch.randn(batch, 3, sum(widths))
                    taps = [torch.randn(1, 1, sum(widths)) for _ in range(4)]
                    expected = convolution(row, history, *taps, *widths)
                    actual = packed(row, history, taps, widths, None)
                    for a, b in zip(actual, expected):
                        self.assertTrue(torch.equal(a, b))
                    history[batch // 2] *= -3
                    row[batch // 2] += 17
                    changed = packed(row, history, taps, widths, None)
                    keep = [i for i in range(batch) if i != batch // 2]
                    for a, b in zip(actual, changed):
                        self.assertTrue(torch.equal(a[keep], b[keep]))


if __name__ == "__main__":
    unittest.main()
