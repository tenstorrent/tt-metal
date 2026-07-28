# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device PCC for the TTNN codec decoder (Block 3) vs the CPU reference.

The reference is itself validated against upstream (30/30, all 8 decoder stages bit-exact or
PCC ~1.0), so matching it here is a real correctness statement, not a self-comparison.

Skips cleanly without ttnn, a device, or the checkpoint.

    pytest -svv models/experimental/voxtral_tts/tests/test_codec_ttnn_pcc.py
"""

import os

import pytest
import torch

from models.experimental.voxtral_tts.reference import voxtral_codec_ref as ref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT, pcc

ttnn = pytest.importorskip("ttnn", reason="ttnn not importable")
pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT), reason=f"no checkpoint at {DEFAULT_CKPT}")

WAVE_PCC = 0.999  # the gate XTTS-v2's HiFi-GAN port shipped at (0.99946)
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


@pytest.mark.parametrize(
    "weight_dtype,attn_dtype,gate",
    [
        ("float32", "float32", 0.999),   # all-fp32 baseline
        ("float32", "bfloat16", 0.999),  # the default
        ("bfloat16", "bfloat16", 0.999), # opt-in faster variant
    ],
)
def test_dtype_variants_hold_the_gate(device, weight_dtype, attn_dtype, gate):
    """Pins the precision sweep. bf16 WEIGHTS with fp32 attention is deliberately absent: it
    measured 0.998757 at T=469, below the gate, which is why weight_dtype defaults to fp32."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    gen = TtVoxtralCodecDecoder(device, weight_dtype=getattr(ttnn, weight_dtype),
                                attn_dtype=getattr(ttnn, attn_dtype))
    w = ref.load_codec_state()
    codes = ref.make_synthetic_codes(64)
    p = pcc(gen(codes), ref.reference_decode(codes, w))
    assert p > gate, f"{weight_dtype}/{attn_dtype} PCC {p:.6f}"


def test_bf16_weights_alone_is_below_gate(device):
    """Documents WHY weight_dtype defaults to fp32: bf16 weights with fp32 attention is the one
    combination that fails. If a future ttnn makes this pass, the default should be revisited."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    gen = TtVoxtralCodecDecoder(device, weight_dtype=ttnn.bfloat16, attn_dtype=ttnn.float32)
    w = ref.load_codec_state()
    codes = ref.make_synthetic_codes(469)
    p = pcc(gen(codes), ref.reference_decode(codes, w))
    assert p < 0.999, f"bf16 weights + fp32 attn now scores {p:.6f} (was 0.998757) — revisit the default"


@pytest.mark.parametrize("n_frames", [64, 469])
def test_chunked_matches_unchunked(device, n_frames):
    """Chunking must be EXACT, not an approximation: attention is causal AND windowed, so a slab
    starting `window` positions early has all the context its kept rows need. Compares the two
    paths directly rather than both against the reference, so a shared error cannot hide."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    codes = ref.make_synthetic_codes(n_frames)
    un = TtVoxtralCodecDecoder(device, chunk_min=None)(codes)
    ch = TtVoxtralCodecDecoder(device, chunk_min=0, slab=512)(codes)
    assert pcc(ch, un) > 0.9999, f"chunked diverges from unchunked at T={n_frames}"


def test_bias_cache_does_not_grow_with_utterance_length(device):
    """Every chunk is padded to `slab`, so above the threshold the cache holds exactly ONE bias
    per window no matter how many different lengths are decoded. Before this, first/last chunk
    lengths varied (the last with S mod C), so each new length added biases AND a kernel compile.

    Stages whose S <= slab still run unchunked and get an SxS bias, so a few length-specific
    entries remain -- that is the conv-side bucketing work, tracked separately."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    gen = TtVoxtralCodecDecoder(device)
    for n in (700, 1000, 1200):  # all well above slab, so every stage chunks
        gen(ref.make_synthetic_codes(n))
    chunked = {(S, w) for (S, w, _) in gen._bias_cache if S == gen.slab}
    assert len(chunked) <= 4, f"expected <=4 slab biases (one per window), got {sorted(chunked)}"
    assert len(gen._bias_cache) <= 6, f"cache grew to {len(gen._bias_cache)} across 3 lengths"


@pytest.mark.parametrize("n_frames", [64, 65, 130, 469])
def test_bucketing_preserves_length_and_accuracy(device, n_frames):
    """Bucketing pads T up to a grid so the 5 convs stop recompiling per length (each distinct T
    otherwise costs 5 new conv programs, measured 1-5 s each). The output must still be trimmed to
    exactly T frames and match the reference."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    w = ref.load_codec_state()
    codes = ref.make_synthetic_codes(n_frames)
    exp = ref.reference_decode(codes, w)
    got = TtVoxtralCodecDecoder(device, bucket=128)(codes)
    assert got.shape == exp.shape == (1, 1, n_frames * 1920), "bucketed output not trimmed back to T"
    assert pcc(got, exp) > WAVE_PCC


def test_bucketing_pads_with_last_frame_not_zeros(device):
    """The pad repeats the final frame rather than zero-filling: zeros are a hard edge to the
    causal convs, and the transposed convs overlap, so a pathological tail could in principle
    reach the kept region. Checks the two agree, i.e. the choice is not load-bearing but is safe."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    codes = ref.make_synthetic_codes(70)  # 70 -> bucket 128, so 58 frames of padding
    bucketed = TtVoxtralCodecDecoder(device, bucket=128)(codes)
    plain = TtVoxtralCodecDecoder(device, bucket=None)(codes)
    assert pcc(bucketed, plain) > 0.999, "padding is leaking into the kept region"


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


def test_slab_is_tile_aligned():
    """TILE_LAYOUT pads every dim to 32. A slab of 272 (= 256 chunk + 16 window) silently becomes
    288, wasting a row and a column of tiles — pick the SLAB aligned and derive the chunk from it."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import SLAB

    assert SLAB % 32 == 0, f"slab {SLAB} is not tile-aligned"


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
