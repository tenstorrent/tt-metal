# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the TTNN Qwen3-TTS codec encoder: waveform in, codes out.

Block boundary: waveform [1, N, 1] at 24 kHz -> codes [1, 16, ceil(N / 1920)].

Two measurements matter here and they are different questions.

**The latents** are floating point and are gated like any other block. The conv stack, the
transformer and the `downsample` convolution all hold 0.9999 on a 2 s clip, in fp32.

**The codes** are the output of a nearest-neighbour search, so they either match or they do
not, and a near-tie can flip on a rounding error. Scoring is per step with the reference's
codes forced into the residual chain, for the reason the code predictor is scored that way:
one flipped code changes the residual every later codebook sees, so a free run measures the
cascade rather than the port. Measured on Blackhole P150, fp32:

    stages (conv stack, transformer, latents)  0.99997
    codes, per step with the prefix forced     92%
    codes, free running                        62%

Every disagreement is a near-tie. At each one the device picked the reference's second or
third nearest entry out of 2048, and the distance it picked was at most 0.6% further away
than the reference's own choice. `test_the_round_trip_is_as_faithful_as_the_reference` is
the test that says whether any of it matters: re-decoding the device's codes lands as close
to the original clip as re-decoding the reference's, within 0.005 of PCC, which is inside
what the codec itself loses at 12.5 Hz.

**This block runs in fp32**, unlike everything else in this directory. bf16 puts the latents
at 0.9984 and per-step agreement at 75%; see `preprocess_codec_encoder_parameters`.
"""

import numpy as np
import pytest
import torch
from transformers import MimiConfig
from transformers.models.mimi.modeling_mimi import MimiModel

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.audio.qwen3_tts import weights
from models.demos.audio.qwen3_tts.reference.qwen3_codec_encoder_ref import CodecEncoderReference
from models.demos.audio.qwen3_tts.reference.qwen3_codec_ref import CodecDecoderReference
from models.demos.audio.qwen3_tts.tests.reference_helpers import synthetic_voiced_clip
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_codec import replicate_pad
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_codec_encoder import (
    TtCodecEncoder,
    encoder_layer_plan,
    preprocess_codec_encoder_parameters,
)

DEVICE_PARAMS = [{"l1_small_size": 65536}]

STAGE_PCC = 0.999
# Per step, with the reference's own codes in the residual chain. Measured 0.92.
STEP_AGREEMENT = 0.85
# How much further away the device's pick may be than the reference's, as a fraction of the
# reference's distance. Measured worst case 0.006 over 31 disagreements.
MAX_DISTANCE_EXCESS = 0.02
# And how far down the reference's own ordering the device's pick may sit, out of 2048.
MAX_NEIGHBOUR_RANK = 8
# Re-decoding the device's codes against re-decoding the reference's, both measured against
# the clip they came from. Measured 0.952 vs 0.948 on real speech, 0.005 apart.
MAX_FAITHFULNESS_GAP = 0.02

CLIP_SECONDS = 2.0


@pytest.fixture(scope="module")
def reference():
    return CodecEncoderReference()


@pytest.fixture(scope="module")
def clip():
    return synthetic_voiced_clip(seconds=CLIP_SECONDS, voice="low")


def _to_device(device, tensor):
    return ttnn.from_torch(tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def _pcc(want, got):
    return float(str(comp_pcc(want.reshape(1, -1), got.reshape(1, -1), pcc=0.0)[1]))


def _encode(device, model, clip, prefix=None):
    """Latents, intermediates and codes from one forward pass, so tests share the work."""
    samples = clip.shape[0]
    cos, sin, mask = model.host_inputs(model.positions(samples))
    audio = ttnn.from_torch(clip.reshape(1, -1, 1), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    latents, intermediates = model(audio, *(_to_device(device, t) for t in (cos, sin, mask)), return_intermediates=True)
    codes = model.quantize(latents, prefix=prefix)[:, :, : model.frames(samples)]
    return codes, intermediates


@torch.no_grad()
def _residual_chain(reference, clip, codes):
    """The reference's fp32 residual at each codebook, following `codes`.

    What a disagreement has to be judged against: the vector the search was actually
    looking at when it picked. Rebuilt here rather than hooked because the quantizer holds
    no state between steps.
    """
    quantizer = reference.model.quantizer
    semantic = quantizer.semantic_residual_vector_quantizer
    acoustic = quantizer.acoustic_residual_vector_quantizer
    tables = [semantic.layers[0].codebook.embed.clone()]
    tables += [acoustic.layers[index].codebook.embed.clone() for index in range(reference.quantizers - 1)]

    latents = reference.latents(clip)[0].t()  # [T, 512]
    residuals = [(latents @ semantic.input_proj.weight.squeeze(-1).t()).clone()]
    current = latents @ acoustic.input_proj.weight.squeeze(-1).t()
    for index in range(1, reference.quantizers):
        residuals.append(current.clone())
        current = current - tables[index][codes[0, index]]
    return tables, residuals


# ── host ────────────────────────────────────────────────────────────────────


def test_the_encoder_is_exactly_transformers_mimi():
    """The reference has no vendored copy, so this is what keeps it honest.

    Upstream's encoder is a `MimiModel` subclass that nulls the decode half and adds
    nothing. If that stops being true, or the checkpoint stops matching `MimiConfig`, the
    key sets diverge here rather than surfacing later as a PCC miss.
    """
    model = MimiModel(MimiConfig(**weights.codec_encoder_config()))
    model.upsample = None
    model.decoder_transformer = None
    model.decoder = None

    wanted = set(model.state_dict().keys())
    present = set(weights.load_codec_encoder_state(dtype=None).keys())

    assert not wanted - present, f"missing from the checkpoint: {sorted(wanted - present)[:8]}"
    assert not present - wanted, f"not part of the module: {sorted(present - wanted)[:8]}"
    assert len(wanted) == 225, f"expected 225 tensors, got {len(wanted)}"


def test_the_layer_plan_matches_the_checkpoint():
    """Which index of `encoder.layers` is a convolution, and which an activation.

    Derived from the config rather than written out, so it is worth checking that the
    derivation lands on the tensors the file actually holds.
    """
    config = weights.codec_encoder_config()
    state = weights.load_codec_encoder_state(dtype=None)
    plan = encoder_layer_plan(config)

    assert [kind for kind, _, _ in plan] == [
        "conv", "resnet", "elu", "conv", "resnet", "elu", "conv",
        "resnet", "elu", "conv", "resnet", "elu", "conv", "elu", "conv",
    ]  # fmt: skip

    strides = [options["stride"] for kind, _, options in plan if kind == "conv"]
    assert strides == [1, 4, 5, 6, 8, 1], "reversed upsampling_ratios, bracketed by two stride-1 convolutions"

    for kind, index, _ in plan:
        if kind == "conv":
            assert f"encoder.layers.{index}.conv.weight" in state
        elif kind == "resnet":
            assert f"encoder.layers.{index}.block.1.conv.weight" in state
            assert f"encoder.layers.{index}.block.3.conv.weight" in state
        else:
            assert f"encoder.layers.{index}.conv.weight" not in state, "an activation carries no weights"


def test_the_frame_rate_is_12_5_hz():
    """960 from the conv stack and 2 from `downsample`: one code per 1920 samples."""
    config = weights.codec_encoder_config()
    stack = int(torch.tensor(config["upsampling_ratios"]).prod())

    assert stack == 960
    assert 2 * stack == weights.codec_config()["encode_downsample_rate"] == 1920
    assert config["sampling_rate"] / 1920 == config["_frame_rate"] == 12.5


def test_the_reference_trims_to_whole_frames(reference):
    """Partial frames are dropped, matching `Qwen3TTSTokenizerV2Model.encode`."""
    assert reference.frames(1920) == 1
    assert reference.frames(1921) == 2, "a frame is claimed as soon as it starts"
    assert reference.frames(0) == 0


# ── device ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_stages_match_the_reference(device, reference, clip):
    """Every convolution block, every transformer layer, and the latents the codes come from."""
    _, gold = reference(clip, return_intermediates=True)
    model = TtCodecEncoder(device, preprocess_codec_encoder_parameters(device))
    _, got = _encode(device, model, clip)

    # The reference is channel-first through the convolutions and time-major inside the
    # transformer, so only the convolution stages transpose.
    config = weights.codec_encoder_config()
    stages = [(f"encoder.layers.{index}", True) for _, index, _ in encoder_layer_plan(config)]
    stages += [("encoder", True)]
    stages += [(f"encoder_transformer.layers.{index}", False) for index in range(config["num_hidden_layers"])]
    stages += [("encoder_transformer", False), ("downsample", True)]

    failures = []
    for name, channel_first in stages:
        if name not in got or name not in gold:
            continue
        want = gold[name]
        measured = ttnn.to_torch(got[name]).float()
        measured = measured.reshape(1, measured.shape[-2], measured.shape[-1])
        measured = measured.permute(0, 2, 1) if channel_first else measured
        passed, message = comp_pcc(want, measured.reshape(want.shape), pcc=STAGE_PCC)
        print(f"  [{name:30s}] {tuple(want.shape)}  {message}")
        if not passed:
            failures.append(f"{name}: {message}")

    assert not failures, "stages below PCC gate: " + "; ".join(failures)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_the_codes_agree_with_the_reference_step_by_step(device, reference, clip):
    """Per codebook, with the reference's codes in the residual chain.

    Each disagreement is then judged the way a nearest-neighbour miss should be: how much
    further away the device's entry is, and how far down the reference's own ordering it
    sits. A rounding error picking the second of two near-equidistant entries is not a
    porting bug; picking the four hundredth would be.
    """
    gold = reference(clip)
    model = TtCodecEncoder(device, preprocess_codec_encoder_parameters(device))
    codes, _ = _encode(device, model, clip, prefix=gold[0])

    assert codes.shape == gold.shape, f"got {tuple(codes.shape)}, want {tuple(gold.shape)}"
    assert int(codes.min()) >= 0 and int(codes.max()) < reference.config.codebook_size

    agree = codes[0] == gold[0]
    fraction = float(agree.sum()) / agree.numel()
    print(f"per step {int(agree.sum())}/{agree.numel()} ({fraction:.1%})")
    print("  per codebook " + " ".join(str(int(agree[k].sum())) for k in range(gold.shape[1])))

    tables, residuals = _residual_chain(reference, clip, gold)
    excess, ranks = [], []
    for codebook, frame in (~agree).nonzero().tolist():
        distances = (residuals[codebook][frame][None] - tables[codebook]).pow(2).sum(-1).sqrt()
        theirs = distances[gold[0, codebook, frame]]
        mine = distances[codes[0, codebook, frame]]
        excess.append(float((mine - theirs) / theirs))
        ranks.append(int((distances.argsort() == codes[0, codebook, frame]).nonzero()) + 1)

    if excess:
        print(
            f"  {len(excess)} disagreements: distance excess max {max(excess):.5f}, "
            f"median {float(np.median(excess)):.5f}; worst neighbour rank {max(ranks)} of {len(tables[0])}"
        )
        assert max(excess) <= MAX_DISTANCE_EXCESS, f"a disagreement was not a near-tie: {max(excess):.5f}"
        assert max(ranks) <= MAX_NEIGHBOUR_RANK, f"the device left the reference's nearest few: rank {max(ranks)}"

    assert fraction >= STEP_AGREEMENT, f"per-step agreement below {STEP_AGREEMENT:.0%}: {fraction:.1%}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_the_round_trip_is_as_faithful_as_the_reference(device, reference, clip):
    """The test that says whether the code disagreements matter.

    Encode on device, decode with the reference decoder, and compare the result to the clip
    it came from. The codec is lossy at 12.5 Hz, so the reference's own round trip is not
    perfect either; what must hold is that the device's is no worse.
    """
    model = TtCodecEncoder(device, preprocess_codec_encoder_parameters(device))
    codes, _ = _encode(device, model, clip)
    decoder = CodecDecoderReference()

    samples = codes.shape[-1] * 1920
    theirs = decoder(reference(clip)).reshape(-1)[:samples]
    mine = decoder(codes).reshape(-1)[:samples]
    original = clip[:samples]

    reference_pcc = _pcc(original, theirs)
    device_pcc = _pcc(original, mine)
    print(f"round trip against the clip: reference {reference_pcc:.5f}, device {device_pcc:.5f}")

    assert device_pcc >= reference_pcc - MAX_FAITHFULNESS_GAP, (
        f"the device's codes reconstruct worse than the reference's by "
        f"{reference_pcc - device_pcc:.5f}, more than {MAX_FAITHFULNESS_GAP}"
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_replicate_padding_matches_torch(device):
    """`downsample` is the one convolution here that replicates its padding.

    Zero padding instead cost 0.009 of PCC on the latents, which is the tensor the codes
    come from, and it was invisible everywhere else. `ttnn.pad` fills with a constant only,
    so the edge columns are built by hand and checked against torch here.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 7, 4)
    tensor = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    for left, right in ((2, 1), (0, 0), (3, 0), (0, 2)):
        want = torch.nn.functional.pad(x.permute(0, 2, 1), (left, right), mode="replicate").permute(0, 2, 1)
        got = ttnn.to_torch(replicate_pad(tensor, left, right)).float()
        assert got.shape == want.shape, f"({left}, {right}): {tuple(got.shape)} != {tuple(want.shape)}"
        assert torch.equal(got, want), f"({left}, {right}) differs by {(got - want).abs().max()}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_one_instance_handles_changing_clip_length(device, reference):
    """Regression, the decoder's twin: `ttnn.conv1d` prepares weights for a parallelisation
    that depends on input length, so a cache keyed by name alone corrupts the next clip of a
    different length. A server encoding one reference clip after another would hit it.
    """
    model = TtCodecEncoder(device, preprocess_codec_encoder_parameters(device))
    results = {}

    for frames in (8, 12, 8):
        clip = synthetic_voiced_clip(seconds=frames * 1920 / 24000, voice="high")
        gold = reference(clip)
        codes, got = _encode(device, model, clip, prefix=gold[0])
        latents = ttnn.to_torch(got["downsample"]).float().permute(0, 2, 1)

        measured = _pcc(reference.latents(clip), latents)
        agree = float((codes[0] == gold[0]).float().mean())
        print(f"  {frames} frames on the shared instance: latents {measured:.6f}, codes {agree:.1%}")
        results.setdefault(frames, []).append((measured, agree))

        assert measured >= STAGE_PCC, f"{frames} frames below {STAGE_PCC}: {measured:.6f}"
        assert agree >= STEP_AGREEMENT, f"{frames} frames agreed on only {agree:.1%}"

    assert results[8][0] == results[8][1], "revisiting a length must reproduce exactly"
