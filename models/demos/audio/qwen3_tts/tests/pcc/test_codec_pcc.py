# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the TTNN Qwen3-TTS codec decoder: codes in, waveform out.

Block boundary: codes [1, 16, T] -> waveform [1, 1, 1920 * T], a 1920x upsample.

**On test input.** The talker needed a real prompt because random embeddings sit outside
the activation distribution its weights were trained on. That argument does not carry here:
codes index learned codebooks, so any valid code produces an in-distribution latent by
construction. What random codes lack is temporal coherence, which changes how loud the
result is but not whether the input is legitimate. So the main measurements use random
valid codes, and a separate test covers real frames from the reference models, which happen
to be the harder case.

Measured on Blackhole P150, bf16:

    stages, random codes at 8 frames      0.995 to 0.999997
    waveform, random codes                0.995
    waveform, real frames at 4 frames     0.954    quiet and short, the pessimal case
    waveform, a real 210-frame utterance  0.9956   and indistinguishable by ear

Why the real 4-frame case scores lower: its waveform has rms 0.043 against 0.145 for random
codes, so a fixed error is a larger share of it, and 4 frames gives the error less to average
over. The 210-frame utterance is quiet too (rms 0.049) and still reaches 0.9956, so length is
what rescues it.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.audio.qwen3_tts import weights
from models.demos.audio.qwen3_tts.reference.qwen3_codec_ref import CodecDecoderReference
from models.demos.audio.qwen3_tts.tests.reference_helpers import codebook_size, codec_frames
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_codec import (
    MASK_FILL,
    TtCodecDecoder,
    preprocess_codec_parameters,
    quantizer_decode,
    windowed_causal_mask,
)

STAGE_PCC = 0.99
# Wormhole: `decoder.3` measured 0.9895 on this seed, 0.9963 on another; HiFi3 no better.
WORMHOLE_STAGE_PCC = 0.985
WAVEFORM_PCC = 0.99

# Real frames are quiet and the fixture is short, so this is the pessimal case rather than
# the representative one. Measured 0.954; see the module docstring.
QUIET_WAVEFORM_PCC = 0.94

# How close a bucketed decode must stay to the exact one. Measured 0.99969; the difference
# is the accumulation order a different parallelisation gives, not a structural one.
BUCKETING_PCC = 0.999

FRAMES = 8

# The pipeline's 64 KB: at 32 KB, two decode lengths ran Wormhole out of L1_SMALL.
DEVICE_PARAMS = [{"l1_small_size": 65536}]
STAGES = ["pre_conv", "pre_transformer", "upsample.0.1", "upsample.1.1"] + [f"decoder.{i}" for i in range(7)]


def random_codes(frames=FRAMES, seed=0):
    """Valid codes: every id indexes a real codebook entry."""
    torch.manual_seed(seed)
    return torch.randint(0, codebook_size(), (1, 16, frames))


@pytest.fixture(scope="module")
def reference():
    return CodecDecoderReference()


def _to_device(device, tensor):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _pcc(want, got):
    return float(str(comp_pcc(want.reshape(1, -1), got.reshape(1, -1), pcc=0.0)[1]))


# ── host ────────────────────────────────────────────────────────────────────


def test_quantizer_decode_matches_the_reference(reference):
    """The codebooks are stored as a sum and a count, so the usable table is their quotient."""
    codes = random_codes()
    want = reference.model.quantizer.decode(codes)  # [1, 512, T]
    got = quantizer_decode(codes, weights.load_codec_decoder_state(), weights.codec_decoder_config())

    assert got.shape == (1, codes.shape[-1], want.shape[1])
    assert torch.allclose(want, got.permute(0, 2, 1), atol=1e-4), (want - got.permute(0, 2, 1)).abs().max()


def test_real_frames_stay_inside_the_codec_vocabulary():
    """A talker control id reaching the decoder would index past the end of its codebooks."""
    frames = codec_frames()
    assert int(frames.max()) < codebook_size()
    assert int(frames.min()) >= 0


def test_window_limits_how_far_back_a_position_sees():
    window = 5
    mask = windowed_causal_mask(12, window)[0, 0]

    assert mask[7, 7] == 0, "a position must see itself"
    assert mask[7, 3] == 0, f"and {window - 1} positions back"
    assert mask[7, 2] == MASK_FILL, "but no further"
    assert mask[3, 4] == MASK_FILL, "and never forwards"


def test_the_upsample_factor_is_1920():
    """4 from upsampling_ratios then 480 from upsample_rates: 12.5 Hz frames to 24 kHz."""
    config = weights.codec_decoder_config()
    ratios = torch.tensor(config["upsampling_ratios"] + config["upsample_rates"])
    assert int(ratios.prod()) == 1920
    assert weights.codec_config()["decode_upsample_rate"] == 1920, "the config's own figure must agree"


# ── device ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_stages_match_the_reference(device, reference):
    """Every stage from the first convolution to the output waveform."""
    codes = random_codes()
    _, gold = reference(codes, return_intermediates=True)
    model = TtCodecDecoder(device, preprocess_codec_parameters(device))

    latents = model.latents(codes)
    cos, sin, mask = model.host_inputs(latents.shape[1])
    _, got = model(*(_to_device(device, t) for t in (latents, cos, sin, mask)), return_intermediates=True)

    gate = WORMHOLE_STAGE_PCC if device.arch() == ttnn.device.Arch.WORMHOLE_B0 else STAGE_PCC
    failures = []
    for name in STAGES:
        if name not in gold or name not in got:
            continue
        want = gold[name]
        measured = ttnn.to_torch(got[name]).float()
        measured = measured.reshape(1, measured.shape[-2], measured.shape[-1])
        # The reference is channel-first everywhere except the transformer's output.
        measured = measured if name == "pre_transformer" else measured.permute(0, 2, 1)
        passed, message = comp_pcc(want, measured.reshape(want.shape), pcc=gate)
        print(f"  [{name:16s}] {tuple(want.shape)}  {message}")
        if not passed:
            failures.append(f"{name}: {message}")

    assert not failures, "stages below PCC gate: " + "; ".join(failures)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_waveform_matches_the_reference(device, reference):
    """The whole decode, and the length and range it must produce."""
    codes = random_codes()
    gold = reference(codes)
    model = TtCodecDecoder(device, preprocess_codec_parameters(device))

    waveform = model.decode(codes, bucket=1)
    expected_samples = codes.shape[-1] * 1920

    assert waveform.shape == (1, 1, expected_samples), f"got {tuple(waveform.shape)}"
    assert torch.isfinite(waveform).all()
    assert waveform.abs().max() <= 1.0, "the decoder clamps its output"

    measured = _pcc(gold, waveform)
    print(f"waveform {expected_samples} samples  pcc {measured:.6f}")
    assert measured >= WAVEFORM_PCC, f"waveform below {WAVEFORM_PCC}: {measured:.6f}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_real_frames_from_the_model_still_track(device, reference):
    """Frames the reference talker and code predictor actually produced.

    Quiet and short, so the loosest gate in this file. It is here because real frames are
    what the decoder will be handed, and a regression that only showed up on them would
    otherwise slip through the random-code tests above.
    """
    frames = codec_frames()
    gold = reference(frames)
    model = TtCodecDecoder(device, preprocess_codec_parameters(device))

    measured = _pcc(gold, model.decode(frames, bucket=1))
    print(f"real frames pcc {measured:.6f}")
    assert measured >= QUIET_WAVEFORM_PCC, f"real frames below {QUIET_WAVEFORM_PCC}: {measured:.6f}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_bucketing_does_not_change_the_samples_it_keeps(device, reference):
    """Decoding a padded length and trimming back gives the same audio.

    Not bit for bit: `ttnn.conv1d` picks its parallelisation from the input length, so a
    padded decode accumulates in a different order. Measured 0.9997 against the exact
    decode, and no worse against the reference, which puts the difference inside what bf16
    already costs (the decoder's own gate is 0.99).

    What makes it safe structurally is that every convolution here is causal and the
    attention only looks backwards, so no output sample depends on a later frame. What makes
    it worth doing is that every distinct frame count compiles its own programs, and
    tt-metal holds each program's L1_SMALL scratch until the device closes: three lengths
    filled the 64 KB region and the next block to want scratch could not allocate.
    """
    codes = random_codes(FRAMES)  # 8 frames, so a 32-frame bucket pads by 24
    model = TtCodecDecoder(device, preprocess_codec_parameters(device))

    exact = model.decode(codes, bucket=1)
    bucketed = model.decode(codes)
    gold = reference(codes)

    assert bucketed.shape == exact.shape == (1, 1, FRAMES * 1920)
    agreement = _pcc(exact, bucketed)
    print(f"bucketed against exact: pcc {agreement:.6f}, max abs difference {(bucketed - exact).abs().max():.5f}")
    assert agreement >= BUCKETING_PCC, f"bucketing moved the audio: {agreement:.6f}"
    assert _pcc(gold, bucketed) >= WAVEFORM_PCC, "and it must still track the reference"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_one_instance_handles_changing_clip_length(device, reference):
    """Regression: `ttnn.conv1d` prepares weights for the parallelisation it picks, and that
    depends on input length. Caching a prepared weight by name alone silently corrupts the
    next clip of a different length: measured 0.995 falling to 0.104 on the second decode.
    A server reusing one decoder would hit this on its second request.
    """
    model = TtCodecDecoder(device, preprocess_codec_parameters(device))

    results = {}
    for frames in (4, 8, 4):
        codes = random_codes(frames)
        measured = _pcc(reference(codes), model.decode(codes, bucket=1))
        print(f"  {frames} frames on the shared instance: pcc {measured:.6f}")
        results.setdefault(frames, []).append(measured)
        assert measured >= WAVEFORM_PCC, f"{frames} frames below {WAVEFORM_PCC}: {measured:.6f}"

    assert results[4][0] == results[4][1], "revisiting a length must reproduce exactly"
