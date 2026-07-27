# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block 3 (Voxtral Codec decoder) reference tests.

Structural + wiring tests run always (random weights at real checkpoint shapes; ~150M fits in
RAM so the FULL decoder runs). Numerical tests need the checkpoint.

Also pins the two facts that shape the whole port plan: the codec ENCODER is absent from the
released checkpoint, and the codec's norm_eps is 1e-2.

    pytest -svv models/experimental/voxtral_tts/tests/test_codec_pcc.py
"""

import os

import pytest
import torch

from models.experimental.voxtral_tts.reference import voxtral_codec_ref as ref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    ACOUSTIC_CODEBOOK_SIZE,
    CODEC_DIM,
    CODEC_NORM_EPS,
    DEC_CONV_BLOCKS,
    DEC_TF_BLOCKS,
    DEFAULT_CKPT,
    END_AUDIO_ID,
    LATENT_DIM,
    N_AUDIO_SPECIAL,
    NUM_CODEBOOKS,
    PATCH_SIZE,
    SEMANTIC_DIM,
    fold_weight_norm,
    load_manifest,
    pcc,
    random_state_from_manifest,
)

PREFIX = "audio_tokenizer."
UPSAMPLE = PATCH_SIZE * 8  # 240 patch x 8 from the three stride-2 transposed convs = 1920/frame
needs_ckpt = pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT), reason=f"no checkpoint at {DEFAULT_CKPT}")


@pytest.fixture(scope="module")
def w():
    """Full-size random codec decoder, weight_norm folded exactly as the real loader does."""
    raw = random_state_from_manifest(PREFIX, seed=0)
    s = {k: v for k, v in raw.items() if ".parametrizations.weight.original" not in k}
    for base in [f"decoder_blocks.{i}.conv" for i in DEC_CONV_BLOCKS] + ["output_proj.conv"]:
        s[base + ".weight"] = fold_weight_norm(raw, base)
    s["semantic_embedding"] = raw["quantizer.semantic_codebook.embedding_sum"] / raw[
        "quantizer.semantic_codebook.cluster_usage"
    ].clamp(min=1e-5).unsqueeze(-1)
    return s


# ---------------------------------------------------------------------------------------
# The finding that drives the port plan
# ---------------------------------------------------------------------------------------
def test_codec_encoder_is_absent_from_released_checkpoint():
    """The public checkpoint ships NO encoder tensors, so waveform -> codes is impossible and
    voice cloning from arbitrary reference audio cannot be built or validated. Only the 20
    shipped voice_embedding presets work. If a future release adds them this test will fail —
    which is the notification we want."""
    man = load_manifest()
    enc = [k for k in man if k.startswith((PREFIX + "input_proj", PREFIX + "encoder_blocks"))]
    assert enc == [], f"encoder weights appeared ({len(enc)} tensors) — Block 4 is now portable"


def test_codec_norm_eps_is_1e_2():
    """params.json really does say norm_eps 0.01 for the codec (vs 1e-5 elsewhere). Guard it so
    nobody 'fixes' it into 1e-5 and silently changes every RMSNorm in the decoder."""
    assert CODEC_NORM_EPS == 1e-2


# ---------------------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------------------
def test_decoder_block_layout_matches_checkpoint():
    """decoder_blocks is an nn.ModuleList of mixed types; the indices must be exactly
    conv at 0,2,4,6 and 2-layer transformers at 1,3,5,7."""
    man = load_manifest()
    conv_idx = sorted({int(k.split(".")[2]) for k in man if ".conv.parametrizations" in k and "decoder_blocks" in k})
    tf = {}
    for k in man:
        parts = k.split(".")
        if k.startswith(PREFIX + "decoder_blocks") and len(parts) > 4 and parts[3] == "layers":
            tf.setdefault(int(parts[2]), set()).add(int(parts[4]))
    assert tuple(conv_idx) == DEC_CONV_BLOCKS
    assert tuple(sorted(tf)) == DEC_TF_BLOCKS
    assert all(v == {0, 1} for v in tf.values()), "each transformer stage must hold exactly 2 layers"


def test_conv_shapes():
    man = load_manifest()
    g = lambda k: tuple(man[PREFIX + k]["shape"])
    assert g("decoder_blocks.0.conv.parametrizations.weight.original1") == (CODEC_DIM, LATENT_DIM, 3)
    for i in (2, 4, 6):  # ConvTranspose1d weight is [in, out, k]
        assert g(f"decoder_blocks.{i}.conv.parametrizations.weight.original1") == (CODEC_DIM, CODEC_DIM, 4)
    assert g("output_proj.conv.parametrizations.weight.original1") == (PATCH_SIZE, CODEC_DIM, 7)


def test_no_conv_biases():
    """Every conv in the decoder is bias-free (use_bias=False upstream)."""
    man = load_manifest()
    assert not [k for k in man if k.startswith(PREFIX) and k.endswith(".conv.bias")]


def test_weight_norm_fold_reproduces_direction_and_magnitude():
    """Folding g,v at dim=0 must give per-output-channel norm == g."""
    v = torch.randn(8, 4, 3)
    g = torch.rand(8, 1, 1) + 0.5
    folded = fold_weight_norm(
        {"c.parametrizations.weight.original0": g, "c.parametrizations.weight.original1": v}, "c"
    )
    assert folded.shape == v.shape
    assert torch.allclose(folded.flatten(1).norm(dim=1), g.flatten(), atol=1e-5)


def test_decoder_windows_are_2_4_8_16():
    """Derived, not hard-coded: the decoder inherits the encoder's final (narrowest) window and
    doubles it per upsample, so stage 0 is the NARROWEST."""
    assert ref.decoder_window_sizes() == (2, 4, 8, 16)


def test_alibi_slopes_geometric():
    s = ref.alibi_slopes(8)
    assert s.shape == (8,)
    assert torch.isclose(s[0], torch.tensor(1.0))
    ratios = s[1:] / s[:-1]
    assert torch.allclose(ratios, torch.full((7,), 0.5), atol=1e-6), "ratio must be 2^(-8/8) = 0.5"


# ---------------------------------------------------------------------------------------
# Attention bias: ALiBi + causal + sliding window folded into one additive term
# ---------------------------------------------------------------------------------------
def test_attention_bias_is_causal_and_windowed():
    S, window = 10, 3
    b = ref.attention_bias(S, window)
    assert b.shape == (1, 8, S, S)
    ninf = float("-inf")
    for i in range(S):
        for j in range(S):
            if j > i:
                assert b[0, 0, i, j] == ninf, f"future position {j} visible from {i}"
            elif i - j > window:
                assert b[0, 0, i, j] == ninf, f"position {j} outside window from {i}"
            else:
                assert b[0, 0, i, j] == pytest.approx(float(j - i)), "head 0 (slope 1) must be j-i"
    assert (b[0, :, 0, 0] == 0).all(), "self-attention must be unbiased"


def test_attention_bias_rows_are_never_fully_masked():
    """Every query must keep at least its own key, or softmax produces NaN."""
    for S, window in ((1, 2), (5, 2), (40, 2), (40, 16)):
        b = ref.attention_bias(S, window)
        assert torch.isfinite(b).any(dim=-1).all(), f"a fully-masked row at S={S}, window={window}"


# ---------------------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------------------
def test_quantizer_decode_shapes_and_acoustic_range(w):
    codes = ref.make_synthetic_codes(n_frames=6)
    lat = ref.quantizer_decode(codes, w)
    assert lat.shape == (1, LATENT_DIM, 6)
    ac = lat[:, SEMANTIC_DIM:]
    assert ac.min() >= -1.0 - 1e-6 and ac.max() <= 1.0 + 1e-6, "FSQ rescale must land in [-1, 1]"


def test_fsq_rescale_inverts_block2_quantization(w):
    """Round-trip every one of the 21 levels: Block 2 quantizes, Block 3 must rescale back to a
    value that re-quantizes to the same code. If these two drift the audio degrades silently."""
    lvl = ACOUSTIC_CODEBOOK_SIZE
    codes = torch.arange(lvl).view(1, 1, lvl).expand(1, NUM_CODEBOOKS - 1, lvl).contiguous()
    full = torch.cat([torch.zeros(1, 1, lvl, dtype=torch.long), codes], dim=1)
    lat = ref.quantizer_decode(full, w)[:, SEMANTIC_DIM:]
    requant = (((lat + 1) / 2) * (lvl - 1)).round().long()
    assert torch.equal(requant, codes)


def test_strip_offset_and_trim_cuts_at_end_audio():
    frames = torch.full((6, NUM_CODEBOOKS), 5, dtype=torch.long)
    frames[4, 0] = END_AUDIO_ID  # EOA in codebook 0 at frame 4
    out = ref.strip_offset_and_trim(frames)
    assert out.shape == (1, NUM_CODEBOOKS, 4), "must cut at the first END_AUDIO"
    assert (out == 5 - N_AUDIO_SPECIAL).all(), "special-token offset not removed"


# ---------------------------------------------------------------------------------------
# Full block
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_frames", [1, 4, 24])
def test_decode_produces_exactly_1920_samples_per_frame(w, n_frames):
    codes = ref.make_synthetic_codes(n_frames)
    wav = ref.reference_decode(codes, w)
    assert wav.shape == (1, 1, n_frames * UPSAMPLE), f"expected {UPSAMPLE} samples/frame @ 24 kHz"
    assert torch.isfinite(wav).all()


def test_decode_is_causal_in_frames(w):
    """Appending frames must not change the audio already produced for the earlier ones. This is
    what makes the chunked/streaming decode upstream uses legitimate."""
    codes = ref.make_synthetic_codes(12)
    short = ref.reference_decode(codes[:, :, :8], w)
    full = ref.reference_decode(codes, w)
    n = short.shape[-1]
    assert pcc(short, full[..., :n]) > 0.999, "later frames changed earlier audio"


def test_stage_shapes_upsample_by_two_each(w):
    """Length must go T -> 2T -> 4T -> 8T across the three transposed convs, and the channel
    count must stay 1024 until output_proj widens it to the 240-sample patch."""
    codes = ref.make_synthetic_codes(8)
    x = ref.causal_conv1d(ref.quantizer_decode(codes, w), w["decoder_blocks.0.conv.weight"], 3, 1, "replicate")
    assert x.shape == (1, CODEC_DIM, 8)
    expect = [16, 32, 64]
    for stage, ci in enumerate((2, 4, 6)):
        x = ref.codec_transformer(x.permute(0, 2, 1), w, DEC_TF_BLOCKS[stage], 2,
                                  ref.decoder_window_sizes()[stage]).permute(0, 2, 1)
        x = ref.causal_conv_transpose1d(x, w[f"decoder_blocks.{ci}.conv.weight"], 4, 2)
        assert x.shape == (1, CODEC_DIM, expect[stage]), f"stage {stage} length wrong"


def test_causal_conv_preserves_length_at_stride_one(w):
    x = torch.randn(1, LATENT_DIM, 17)
    out = ref.causal_conv1d(x, w["decoder_blocks.0.conv.weight"], 3, 1, "replicate")
    assert out.shape[-1] == 17, "stride-1 causal conv must preserve length"


def test_causal_conv_transpose_trims_right(w):
    """k=4, stride=2 gives 2T+2 raw; the trim must remove exactly (k - stride) = 2 from the right."""
    x = torch.randn(1, CODEC_DIM, 5)
    raw = torch.nn.functional.conv_transpose1d(x, w["decoder_blocks.2.conv.weight"], None, stride=2)
    out = ref.causal_conv_transpose1d(x, w["decoder_blocks.2.conv.weight"], 4, 2)
    assert raw.shape[-1] == 12 and out.shape[-1] == 10
    assert torch.equal(out, raw[..., :10]), "trim must come off the right, not the left"


@needs_ckpt
def test_real_weights_decode_runs():
    w_real = ref.load_codec_state()
    codes = ref.make_synthetic_codes(16)
    wav = ref.reference_decode(codes, w_real)
    assert wav.shape == (1, 1, 16 * UPSAMPLE)
    assert torch.isfinite(wav).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-svv", __file__]))
