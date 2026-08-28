# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device PCC for the TTNN codec decoder (Block 3) vs the CPU reference.

The reference is itself validated against upstream (30/30, all 8 decoder stages bit-exact or
PCC ~1.0), so matching it here is a real correctness statement, not a self-comparison.

Skips cleanly without ttnn, a device, or the checkpoint.

    pytest -svv models/experimental/voxtral_tts/tests/test_codec_pcc.py
"""

import os

import pytest
import torch

from models.experimental.voxtral_tts.reference import voxtral_codec_ref as ref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT, pcc
from models.experimental.voxtral_tts.tests.reference_helpers import long_frame_cases, real_frames_long

ttnn = pytest.importorskip("ttnn", reason="ttnn not importable")
# Every test here opens a device, so `slow` joins the checkpoint guard: `-m "not slow"` is
# meant to be the host-only subset, and it used to still run these.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT),
                       reason=f"no checkpoint at {DEFAULT_CKPT}"),
]

WAVE_PCC = 0.999  # the gate XTTS-v2's HiFi-GAN port shipped at (0.99946)
# Real codes are far kinder than synthetic ones, so the real-input gates are tighter. Measured over
# all 15 captured utterances (1754 frames, 140 s): PCC 0.999959 .. 1.000000, worst-sample 0.77-3.22%.
REAL_LONG_PCC = 0.9999
REAL_LONG_WORST_PCT = 5.0  # above the 64-frame test's 2%: a whole utterance draws more samples
STAGE_PCC = 0.996  # per-stage; the final window-16 stage amplifies inherited error (see below)


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0, l1_small_size=65536)
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def pair(device):
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    return TtVoxtralCodecDecoder(device), ref.load_codec_state()


@pytest.mark.parametrize("n_frames", [8, 24, 64])
def test_waveform_pcc(pair, n_frames):
    gen, w = pair
    codes = ref.make_synthetic_codes(n_frames)
    got = gen(codes)
    exp = ref.reference_decode(codes, w)
    assert got.shape == exp.shape == (1, 1, n_frames * 1920)
    p = pcc(got, exp)
    assert p > WAVE_PCC, f"waveform PCC {p:.6f} at T={n_frames}"


def test_quantizer_is_exact(pair):
    """The semantic gather runs on host precisely so this stays exact — a bf16 device embedding
    would inject ~0.4% before a deep conv stack that does not cancel error."""
    gen, w = pair
    codes = ref.make_synthetic_codes(16)
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    got = TtVoxtralCodecDecoder._chw(gen.quantizer_decode(codes))
    assert pcc(got, ref.quantizer_decode(codes, w)) > 0.99999


def test_every_stage_matches(pair):
    """Bisects the 8 decoder stages, so a regression localises to one conv or one 2-layer
    transformer rather than 'the audio sounds wrong'."""
    gen, w = pair
    codes = ref.make_synthetic_codes(24)
    _, stages = gen(codes, return_stages=True)
    lat = ref.quantizer_decode(codes, w)
    x = ref.causal_conv1d(lat, w["decoder_blocks.0.conv.weight"], 3, 1, "replicate")
    assert pcc(stages["after_input_conv"], x) > 0.9999
    for stage, tf_i in enumerate(ref.DEC_TF_BLOCKS):
        x = ref.codec_transformer(x.permute(0, 2, 1), w, tf_i, 2,
                                  ref.decoder_window_sizes()[stage]).permute(0, 2, 1)
        p = pcc(stages[f"after_tf{tf_i}"], x)
        assert p > STAGE_PCC, f"after_tf{tf_i} PCC {p:.6f}"
        if stage < 3:
            ci = ref.DEC_CONV_BLOCKS[stage + 1]
            x = ref.causal_conv_transpose1d(x, w[f"decoder_blocks.{ci}.conv.weight"], 4, 2)
            p = pcc(stages[f"after_up{ci}"], x)
            assert p > 0.9999, f"after_up{ci} PCC {p:.6f}"


def test_final_stage_is_not_itself_lossy(pair):
    """The window-16 stage shows the lowest in-chain PCC, which could look like a bug in it.
    Fed the REFERENCE's input it matches at ~0.99998 like every other stage, so the drop is
    inherited error being amplified (the same effect XTTS-v2 found in the Perceiver), not a
    defect in this stage. Guards against 'fixing' the wrong thing."""
    gen, w = pair
    codes = ref.make_synthetic_codes(24)
    lat = ref.quantizer_decode(codes, w)
    x = ref.causal_conv1d(lat, w["decoder_blocks.0.conv.weight"], 3, 1, "replicate")
    for s, tf in enumerate((1, 3, 5)):
        x = ref.codec_transformer(x.permute(0, 2, 1), w, tf, 2, ref.decoder_window_sizes()[s]).permute(0, 2, 1)
        ci = ref.DEC_CONV_BLOCKS[s + 1]
        x = ref.causal_conv_transpose1d(x, w[f"decoder_blocks.{ci}.conv.weight"], 4, 2)
    exp = ref.codec_transformer(x.permute(0, 2, 1), w, 7, 2, 16)

    L = x.shape[2]
    xd = ttnn.from_torch(x.permute(0, 2, 1).reshape(1, L, 1024).contiguous(),
                         dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=gen.device)
    seq = xd
    for li in range(2):
        seq = gen._block(seq, gen.layers[(7, li)], 16)  # _block takes the WINDOW; it builds/chunks itself
    got = ttnn.to_torch(seq).float().reshape(1, L, 1024)
    assert pcc(got, exp) > 0.9999, "stage 7 is lossy in isolation — this IS a bug in stage 7"


def test_shipped_precision_holds_the_gate(device):
    """Precision is no longer switchable -- fp32 weights, bf16 attention, chosen by a sweep whose
    table is in the codec module docstring. The three rejected combinations used to be pinned here
    by parametrizing weight_dtype/attn_dtype; those constructor kwargs are gone, so what is left to
    guard is that the ONE shipped combination still clears the gate. The finding that motivated the
    default -- bf16 weights with fp32 attention scoring 0.998757 at T=469, the only combination
    below 0.999 -- is recorded in that same table.
    """
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    gen = TtVoxtralCodecDecoder(device)
    w = ref.load_codec_state()
    codes = ref.make_synthetic_codes(64)
    p = pcc(gen(codes), ref.reference_decode(codes, w))
    assert p > 0.999, f"shipped fp32 weights / bf16 attention PCC {p:.6f}"


def test_real_speech_frames_decode_correctly(device):
    """Decode REAL model output, not synthetic codes.

    This exists because it was missing: the real-speech check (PCC 0.999987, ASR WER 0.0%) was
    run against the FIRST working version, and then bf16 attention, chunked attention, length
    bucketing and prepared conv weights all landed without it being re-run. Nothing caught that,
    because every gate was synthetic-codes-vs-reference. Real codes have very different statistics
    from uniform random ones, so they exercise the numerics differently.

    The fixture is 64 frames of genuine Block 1+2 output (int16, ~5 KB), so this needs no backbone.
    """
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    fx = os.path.join(os.path.dirname(__file__), "real_frames_fixture.pt")
    if not os.path.exists(fx):
        pytest.skip("real_frames_fixture.pt missing")
    frames = torch.load(fx).long()
    codes = ref.strip_offset_and_trim(frames)
    exp = ref.reference_decode(codes, ref.load_codec_state())
    got = TtVoxtralCodecDecoder(device)(codes)  # DEFAULT config, as callers get it
    assert got.shape == exp.shape
    p = pcc(got, exp)
    assert p > 0.9999, f"real-speech PCC {p:.6f}"
    # also bound the worst single sample, which PCC can hide
    peak = exp.abs().max().item()
    assert (got - exp).abs().max().item() < 0.02 * peak, "worst-sample error above 2% of peak"


@pytest.mark.parametrize("case", long_frame_cases())
def test_real_utterance_decodes_correctly(pair, case):
    """Real frames over a WHOLE utterance, 16 to 470 frames (1.3 to 37.6 s), against fp32.

    The other real-input test is one 64-frame capture, and the 469-frame checks compare the chunked
    and unchunked device paths to each other. This is the only comparison of full-length output to
    the reference, so it is what would catch an error the two device paths share.
    """
    gen, w = pair
    codes = ref.strip_offset_and_trim(real_frames_long(case))
    T = codes.shape[2]
    exp = ref.reference_decode(codes, w)
    got = gen(codes)
    assert got.shape == exp.shape == (1, 1, T * 1920)
    p = pcc(got, exp)
    assert p > REAL_LONG_PCC, f"case {case} PCC {p:.6f} at T={T}"
    worst = (got - exp).abs().max().item() / exp.abs().max().item() * 100
    assert worst < REAL_LONG_WORST_PCT, f"case {case} worst-sample {worst:.2f}% at T={T}"


def test_causal_padding_matches_torch(pair):
    """replicate/reflect left-padding is built from slice+concat because ttnn.pad is
    constant-only and there is no flip. Easy to get backwards; compare against torch."""
    import torch.nn.functional as F

    gen, _ = pair
    x = torch.randn(1, 1, 11, 32)
    xd = ttnn.from_torch(x.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=gen.device)
    for mode, pad in (("replicate", 2), ("reflect", 6)):
        got = ttnn.to_torch(gen._pad_causal(xd, pad, mode)).float()
        exp = F.pad(x.permute(0, 3, 1, 2).reshape(1, 32, 11), (pad, 0), mode=mode)
        exp = exp.reshape(1, 32, 1, 11 + pad).permute(0, 2, 3, 1)
        assert torch.allclose(got, exp, atol=1e-6), f"{mode} pad mismatch"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-svv", __file__]))
