# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""Portable + device tests: config/shape logic on CPU, TT encoder and greedy decode vs the CPU FP32 reference.

Run: pytest models/experimental/parakeet/tests/test_parakeet.py [-m "not device"]
"""

import numpy as np
import pytest

from models.experimental.parakeet.reference.torch_parakeet import row_nrmse, strip_pad
from models.experimental.parakeet.tests.conftest import make_deterministic_mels

NRMSE_GATE_SYNTHETIC = 0.10  # relaxed for random-noise mel (real-speech gate is 0.04)

CASES = make_deterministic_mels()


# ------------------------------------------------------------------ CPU only
def test_config_consistent(hf_config):
    from models.experimental.parakeet.tt import ParakeetConfig

    c = ParakeetConfig.from_dict(hf_config)
    assert c.head_dim * c.heads == c.hidden
    assert 2**c.n_sub_convs == c.sub_factor
    assert c.blank == hf_config["blank_token_id"] and c.vocab == hf_config["vocab_size"]
    assert list(c.durations) == list(hf_config["durations"])


def test_sub_length_matches_torch_conv(hf_config):
    torch = pytest.importorskip("torch")
    from models.experimental.parakeet.tt import ParakeetConfig

    c = ParakeetConfig.from_dict(hf_config)
    pad = (c.sub_kernel - 1) // 2
    conv = torch.nn.Conv1d(1, 1, c.sub_kernel, stride=c.sub_stride, padding=pad)
    for n in (1, 7, 8, 201, 225, 440, 1636, 4000):
        x = torch.zeros(1, 1, n)
        for _ in range(c.n_sub_convs):
            x = conv(x)
        assert c.sub_length(n) == x.shape[-1]


def test_rel_positional_encoding():
    pytest.importorskip("torch")
    from models.experimental.parakeet.tt import rel_positional_encoding

    pe = rel_positional_encoding(5, 16).numpy()
    assert pe.shape == (9, 16)
    zero = pe[4]
    np.testing.assert_allclose(zero[0::2], 0.0, atol=1e-7)
    np.testing.assert_allclose(zero[1::2], 1.0, atol=1e-7)


def test_unsupported_precision_fails(weights_path, hf_config):
    from models.experimental.parakeet.tt import SUPPORTED_PRECISIONS, create_backend

    assert "bfp8_b" not in SUPPORTED_PRECISIONS
    with pytest.raises(ValueError):
        create_backend(weights_path, hf_config, device=None, precision="bfp8_b")


# ------------------------------------------------------------------ TT device
@pytest.mark.device
@pytest.mark.parametrize("name", sorted(CASES))
def test_encoder_nrmse(name, tt_model, reference):
    mel, lens = CASES[name]
    out = tt_model.encode(mel, lens)["encoder"]
    ref = reference.encode(mel, lens)["encoder"]
    sub = [tt_model.cfg.sub_length(int(n)) for n in lens]
    assert out.shape[0] == ref.shape[0] and out.shape[2] == ref.shape[2]
    assert out.shape[1] == tt_model.cfg.sub_length(mel.shape[1])
    worst = max(row_nrmse(ref, out, sub))
    assert worst <= NRMSE_GATE_SYNTHETIC, f"{name}: encoder NRMSE {worst:.4f} > {NRMSE_GATE_SYNTHETIC}"


@pytest.mark.device
@pytest.mark.parametrize("name", sorted(CASES))
def test_transcribe_deterministic(name, tt_model):
    mel, lens = CASES[name]
    before = mel.copy()
    a = tt_model.transcribe(mel, lens)["tokens"]
    b = tt_model.transcribe(mel, lens)["tokens"]
    assert a.shape == b.shape and (a == b).all()
    assert (mel == before).all(), "inputs must not be mutated"


@pytest.mark.device
@pytest.mark.parametrize("name", sorted(CASES))
def test_fast_decode_matches_reference_path(name, tt_model):
    if not tt_model.fast_decode:
        pytest.skip("model was built with fast_decode off")
    mel, lens = CASES[name]
    fast = tt_model.transcribe(mel, lens)["tokens"]
    tt_model.fast_decode = False
    try:
        slow = tt_model.transcribe(mel, lens)["tokens"]
    finally:
        tt_model.fast_decode = True
    assert fast.shape == slow.shape and (fast == slow).all()


@pytest.mark.device
def test_batch_matches_single(tt_model):
    mel, lens = CASES["synthetic_b2"]
    pad = tt_model.cfg.pad
    batched = tt_model.transcribe(mel, lens)["tokens"]
    for b in range(mel.shape[0]):
        single = tt_model.transcribe(mel[b : b + 1], lens[b : b + 1])["tokens"]
        assert strip_pad(batched[b], pad) == strip_pad(single[0], pad), f"row {b}"


# ------------------------------------------------------------------ E2E with real checkpoint
@pytest.mark.device
def test_transcribe_real_checkpoint(tt_model, reference):
    """End-to-end: generate a deterministic mel in-test, transcribe on device, compare tokens with CPU FP32."""
    rng = np.random.default_rng(42)
    mel = rng.standard_normal((1, 200, 128)).astype(np.float32)
    lens = np.array([200], dtype=np.int64)

    out = tt_model.transcribe(mel, lens)["tokens"]
    ref = reference.transcribe(mel, lens)["tokens"]
    pad = tt_model.cfg.pad
    got = strip_pad(out[0], pad)
    expected = strip_pad(ref[0], pad)
    assert got == expected, f"token mismatch: got {got}, expected {expected}"
