"""Phase 3 PCC test: HiFiGAN vocoder (HifiganGenerator) vs reference module.

Self-contained — loads reference XTTS-v2 and compares the TTNN vocoder against
xtts.hifigan_decoder.waveform_decoder. Device needs l1_small_size>0 for conv ops.

Env: see test_gpt_stack.py docstring (bare recipe works).
"""

import os
import sys

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "tt")))
import hifigan_generator as HG  # noqa: E402
from model_config import load_reference_state_dict  # noqa: E402

PCC = 0.99


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def reference():
    model, _ = load_reference_state_dict("cpu")
    return model.hifigan_decoder.waveform_decoder


@pytest.mark.parametrize("frames", [16])
def test_hifigan_generator(device, reference, frames):
    gen = reference
    torch.manual_seed(0)
    x = torch.randn(1, 1024, frames) * 0.3  # GPT latents
    g = torch.randn(1, 512, 1) * 0.3  # speaker embedding

    with torch.no_grad():
        ref = gen(x, g=g)  # [1, 1, frames*256]

    p = HG.load_generator_params(gen, device)
    got = HG.hifigan_generator(x, g, p, device)

    ok, msg = comp_pcc(ref, got, PCC)
    print(f"\nhifigan_generator: {msg}  (ref {tuple(ref.shape)}, got {tuple(got.shape)})")
    assert ok, msg
