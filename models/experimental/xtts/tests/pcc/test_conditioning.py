"""Phase 2 PCC tests: conditioning encoder + perceiver resampler vs reference.

Self-contained — loads the reference XTTS-v2 model and compares each TTNN module
against the real reference module (xtts.gpt.conditioning_encoder / .conditioning_perceiver).

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
import perceiver_resampler as PR  # noqa: E402
import conditioning_encoder as CE  # noqa: E402
from model_config import load_reference_state_dict  # noqa: E402

PCC = 0.99


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def reference():
    model, sd = load_reference_state_dict("cpu")
    return model.gpt, sd


@pytest.mark.parametrize("seq_len", [345])
def test_conditioning_encoder(device, reference, seq_len):
    gpt, sd = reference
    ref_mod = gpt.conditioning_encoder
    torch.manual_seed(0)
    mel = torch.randn(1, CE.SPEC, seq_len) * 0.5

    with torch.no_grad():
        ref = ref_mod(mel)  # [1, 1024, S]

    p = CE.load_encoder_params(sd, device)
    mel_tt = ttnn.from_torch(mel.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    got = ttnn.to_torch(CE.conditioning_encoder(mel_tt, p)).float()  # [1, S, 1024]
    got = got.transpose(1, 2)  # -> [1, 1024, S] to match reference

    ok, msg = comp_pcc(ref, got, PCC)
    print(f"\nconditioning_encoder: {msg}")
    assert ok, msg


@pytest.mark.parametrize("seq_len", [345])
def test_perceiver_resampler(device, reference, seq_len):
    gpt, sd = reference
    ref_mod = gpt.conditioning_perceiver
    torch.manual_seed(0)
    context = torch.randn(1, seq_len, PR.DIM) * 0.1

    with torch.no_grad():
        ref = ref_mod(context)  # [1, 32, 1024]

    p = PR.load_perceiver_params(sd, device)
    ctx_tt = ttnn.from_torch(context.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    got = ttnn.to_torch(PR.perceiver_resampler(ctx_tt, p)).float()

    ok, msg = comp_pcc(ref, got, PCC)
    print(f"\nperceiver: {msg}")
    assert ok, msg
