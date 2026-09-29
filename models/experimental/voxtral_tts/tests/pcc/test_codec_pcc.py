# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device PCC for the TTNN codec decoder (codec) vs the CPU reference.

The reference is itself validated against upstream, so matching it is a real correctness check.
Skips cleanly without ttnn, a device, or the checkpoint.

    pytest -svv models/experimental/voxtral_tts/tests/pcc/test_codec_pcc.py
"""

import os

import pytest
import torch

from models.experimental.voxtral_tts.reference import voxtral_codec_ref as ref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT, pcc
from models.experimental.voxtral_tts.tests.reference_helpers import FRAMES, long_frame_cases, real_frames_long

ttnn = pytest.importorskip("ttnn", reason="ttnn not importable")
# Every test here opens a device, so it is `slow`: `-m "not slow"` is the host-only subset.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT), reason=f"no checkpoint at {DEFAULT_CKPT}"),
]

# Gate constants, each just below what the codec reaches on its inputs.
WAVE_PCC = 0.999  # synthetic codes
REAL_LONG_PCC = 0.9999  # real codes are kinder than synthetic, so the real-input gates are tighter
REAL_LONG_WORST_PCT = 5.0  # above the 64-frame test's 2%: a whole utterance draws more samples
STAGE_PCC = 0.996  # per-stage; the window-16 stage amplifies inherited error (test_final_stage_...)


@pytest.fixture(scope="module")
def device():
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import L1_SMALL_SIZE

    d = ttnn.open_device(device_id=0, l1_small_size=L1_SMALL_SIZE)
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
    """The semantic gather runs on host, in fp32, so this stays exact."""
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
        x = ref.codec_transformer(x.permute(0, 2, 1), w, tf_i, 2, ref.decoder_window_sizes()[stage]).permute(0, 2, 1)
        p = pcc(stages[f"after_tf{tf_i}"], x)
        assert p > STAGE_PCC, f"after_tf{tf_i} PCC {p:.6f}"
        if stage < 3:
            ci = ref.DEC_CONV_BLOCKS[stage + 1]
            x = ref.causal_conv_transpose1d(x, w[f"decoder_blocks.{ci}.conv.weight"], 4, 2)
            p = pcc(stages[f"after_up{ci}"], x)
            assert p > 0.9999, f"after_up{ci} PCC {p:.6f}"


def test_final_stage_is_not_itself_lossy(pair):
    """Feeds stage 7 (window 16) the REFERENCE's input: its low in-chain PCC is amplified
    inherited error, not a defect in the stage."""
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
    xd = ttnn.from_torch(
        x.permute(0, 2, 1).reshape(1, L, 1024).contiguous(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=gen.device,
    )
    seq = xd
    for li in range(2):
        seq = gen._block(seq, gen.layers[(7, li)], 16)  # _block takes the WINDOW; it builds/chunks itself
    got = ttnn.to_torch(seq).float().reshape(1, L, 1024)
    assert pcc(got, exp) > 0.9999, "stage 7 is lossy in isolation — this IS a bug in stage 7"


def test_shipped_precision_holds_the_gate(device):
    """The fixed precision (fp32 weights, bf16 attention) clears the waveform gate."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    gen = TtVoxtralCodecDecoder(device)
    w = ref.load_codec_state()
    codes = ref.make_synthetic_codes(64)
    p = pcc(gen(codes), ref.reference_decode(codes, w))
    assert p > 0.999, f"shipped fp32 weights / bf16 attention PCC {p:.6f}"


def test_real_speech_frames_decode_correctly(device):
    """Decode REAL model output (64-frame fixture of the backbone and flow model codes), not
    synthetic codes; gates PCC and worst sample."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    fx = FRAMES
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
    """Real frames over a WHOLE utterance against fp32: the only full-length comparison to the
    reference, so it catches an error both device paths share."""
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
