# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU regression for LTXPipeline.decode_latents' defer_yuv hand-off (LTX_AUDIO_OVERLAP); no device is used.

The production method is executed from source with only its device dependencies replaced.
"""

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def decode_latents():
    path = Path(__file__).resolve().parents[2] / "pipelines" / "ltx" / "pipeline_ltx.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LTXPipeline")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "decode_latents")
    method.decorator_list = []  # the CPU-thread cap only matters on a real pipeline
    namespace = {
        "torch": torch,
        "logger": SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None),
        "Watchdog": lambda name: contextlib.nullcontext(),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["decode_latents"]


class PlainYuvDecoder:
    """A YUV decoder whose forward has no defer_yuv keyword."""

    def __init__(self):
        self.calls = []

    def __call__(self, sample, *, output_type="float"):
        self.calls.append({"output_type": output_type})
        return np.zeros((1, 6, 4), dtype=np.uint8)


class DeferringDecoder(PlainYuvDecoder):
    supports_defer_yuv = True

    def __call__(self, sample, *, output_type="float", defer_yuv=False):
        self.calls.append({"output_type": output_type, "defer_yuv": defer_yuv})
        return np.zeros((1, 6, 4), dtype=np.uint8)


def _pipeline(decoder):
    return SimpleNamespace(vae_decoder=decoder, in_channels=2)


LATENT = torch.zeros(1, 1 * 2 * 2, 2)


def test_decoder_without_defer_kwarg_gets_plain_call(decode_latents):
    dec = PlainYuvDecoder()
    out = decode_latents(_pipeline(dec), LATENT, 1, 2, 2, output_type="yuv", defer_yuv=True)
    assert isinstance(out, np.ndarray)
    assert dec.calls == [{"output_type": "yuv"}]


def test_decoder_with_defer_kwarg_gets_it(decode_latents):
    dec = DeferringDecoder()
    decode_latents(_pipeline(dec), LATENT, 1, 2, 2, output_type="yuv", defer_yuv=True)
    assert dec.calls == [{"output_type": "yuv", "defer_yuv": True}]


def test_default_path_passes_no_defer_kwarg(decode_latents):
    dec = DeferringDecoder()
    decode_latents(_pipeline(dec), LATENT, 1, 2, 2, output_type="yuv")
    assert dec.calls == [{"output_type": "yuv", "defer_yuv": False}]
