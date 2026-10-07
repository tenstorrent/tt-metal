# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU regression for LTXPipeline.decode_latents' defer_yuv hand-off; no TTNN binary or device is imported.

Run directly with Python to avoid the repository's device-aware pytest conftest.
"""

import ast
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _decode_latents():
    path = Path(__file__).resolve().parents[2] / "pipelines" / "ltx" / "pipeline_ltx.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LTXPipeline")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "decode_latents")
    namespace = {
        "torch": torch,
        "os": __import__("os"),
        "logger": SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None),
        "Watchdog": lambda name: contextlib.nullcontext(),
        "log_dram": lambda *a, **k: None,
    }
    # Execute the production method, with only its device dependencies replaced.
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["decode_latents"]


class DiffVaeLike:
    """Mirrors DiffVAEDecoder.forward: YUV output, but no defer_yuv keyword."""

    supports_yuv = True
    trace_yuv_output = False

    def __init__(self):
        self.calls = []

    def __call__(self, sample, *, output_type="float"):
        self.calls.append({"output_type": output_type})
        return np.zeros((1, 6, 4), dtype=np.uint8)


class ConvVaeLike(DiffVaeLike):
    supports_defer_yuv = True

    def __call__(self, sample, *, output_type="float", defer_yuv=False):
        self.calls.append({"output_type": output_type, "defer_yuv": defer_yuv})
        return np.zeros((1, 6, 4), dtype=np.uint8)


def _pipeline(decoder):
    return SimpleNamespace(vae_decoder=decoder, in_channels=2, mesh_device=None, dynamic_load=False)


class DecodeDeferYuvTest(unittest.TestCase):
    def setUp(self):
        self.decode = _decode_latents()
        self.latent = torch.zeros(1, 1 * 2 * 2, 2)

    def test_decoder_without_defer_kwarg_gets_plain_call(self):
        dec = DiffVaeLike()
        out = self.decode(_pipeline(dec), self.latent, 1, 2, 2, output_type="yuv", defer_yuv=True)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(dec.calls, [{"output_type": "yuv"}])

    def test_decoder_with_defer_kwarg_gets_it(self):
        dec = ConvVaeLike()
        self.decode(_pipeline(dec), self.latent, 1, 2, 2, output_type="yuv", defer_yuv=True)
        self.assertEqual(dec.calls, [{"output_type": "yuv", "defer_yuv": True}])

    def test_default_path_passes_no_defer_kwarg(self):
        dec = ConvVaeLike()
        self.decode(_pipeline(dec), self.latent, 1, 2, 2, output_type="yuv")
        self.assertEqual(dec.calls, [{"output_type": "yuv", "defer_yuv": False}])


if __name__ == "__main__":
    unittest.main()
