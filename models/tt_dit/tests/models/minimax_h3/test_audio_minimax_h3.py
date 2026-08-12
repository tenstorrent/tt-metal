# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gate M8d: the MiniMax-H3 audio VAE decode path (BigVGAN).

Production shapes only. The latent is 40 Hz, so 5 s is 207 latent frames and 10 s is 405,
matching the video working points (124 and 243 frames at 24 fps). Stereo is carried as
**batch 2** -- the autoencoder itself is mono and ``conv_post`` emits one channel -- so
nothing here is stereo-aware beyond the batch dimension.

Structure follows ``tests/models/ltx/test_audio_components_ltx.py``: staged sub-block
gates first, then the full stack, with ``assert_quality`` plus the two checks PCC hides on
audio -- a PSNR floor and a mel-spectrogram distance. An aggregate PCC can look healthy
while the spectrum is visibly wrong, which is the failure mode ``vocoder_ltx``'s fp32
mandate exists to prevent, and H3's conv chain is longer than LTX's.

The narrow tail is called out explicitly: H3's decoder halves 1024 down to **8** channels
before ``conv_post``, where LTX's narrowest is 24. ``_AlignedOutConv1d`` pads 8 up to 32
while ``SnakeBeta`` keeps 8, so that boundary gets its own case.

T-parallel decode is deliberately not gated here: it is not wired into the pipeline (which
passes ``parallel_config=None``), and its 8-way shard layout is known-broken. If T-parallel
ever ships, resurrect ``test_audio_decode_t_parallel`` (and its ``_localize_divergence``
helper) and the halo isolation gate ``test_neighbor_pad_t_minimax_h3.py`` from git history.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.audio_ops import Conv1dViaConv3d

# Imported for the decoder gates below -- but note the import itself also runs
# ``register_h3_audio_blockings()``, which is what puts the H3 audio conv shapes into
# ``_FP32_BLOCKINGS``. Without it every conv here would silently fall back to the conservative
# ``C_in_block = 32`` default and measure a *different op* than production.
from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import (
    assert_weight_norm_axes_consistent,
    convert_minimax_h3_audio_state_dict,
    fuse_attention_biases,
    fuse_weight_norm,
    remap_amp_activations,
)
from ....models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from ....utils.check import assert_quality

# The vocoder needs extra L1 scratch, as the LTX audio tests do.
SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

# RMSE/sigma bar, set from measurement rather than convention. Candidate error sources were each
# measured: the MAC fallback in depthwise_tap_filter is **bit-exact** (rel_max 0.0), `Activation1d` is
# pcc 0.9999998 / RMSE 0.17%, and the decoder path contains no SDPA so the bf16 attention island cannot
# explain it. What remains is accumulation across the chain, with the per-op term identified
# rather than inferred: an fp32 conv3d on this hardware is fp32 storage and fp32 accumulate with a
# multiply that keeps only ~11 significand bits, which measures 1.86e-03 on `conv_pre` at production
# blocking and is **flat in the reduction depth**. Over ~130 convolutions plus 126 anti-aliased
# activations that lands at ~10% RMSE while PCC stays at 99.5%.
#
# So PCC plus the perceptual gates (PSNR, log-spectrogram distance) are the meaningful bars here, and
# RMSE is held at a level consistent with the measured chain depth rather than at a value only a
# shallower model could reach.
#
# That chain depth is not a floor, though: `MINIMAX_H3_AUDIO_ACCURATE=1` reaches **0.45 %** RMSE
# (PCC 99.9990 %, PSNR 67.5 dB) for ~3x the stage time, by fixing the three sources this bar's 10.5 %
# is made of -- see `audio_accurate_mode`. This bar describes the **default** path, so it must be
# re-derived if that default ever changes; the accurate path carries its own gate,
# `test_decode_accurate_mode` below.
AUDIO_RELATIVE_RMSE = 0.12

# `conv_pre`: the decoder's widest reduction, Cin 2048 x k 7 = 14336, and the shape every operand-split
# measurement was taken at. Its T is the 5 s latent length, as in the decode gates.
CONV_PRE_SHAPE = (2048, 1024, 7)
CONV_PRE_LATENT_FRAMES = 207

LATENT_CHANNELS = 32
HOP_LENGTH = 800
SAMPLING_RATE = 32000

# 40 Hz latent: 5 s at the video working points' true durations. The 10 s case (405 frames)
# exercises no path 207 doesn't -- same convs, same blockings, longer T -- so only 5 s is gated.
PRODUCTION_LATENT_FRAMES = [pytest.param(207, id="5s")]


def _weights_dir() -> str | None:
    base = os.environ.get("MINIMAX_H3_DIFFUSERS_DIR", "/data/cglagovich/MiniMax-H3-diffusers")
    candidate = os.path.join(base, "audio_vae")
    return candidate if os.path.isfile(os.path.join(candidate, "config.json")) else None


def _config(weights_dir: str) -> dict:
    raw = json.loads(open(os.path.join(weights_dir, "config.json")).read())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _build_reference(load_weights: bool = True):
    """The reference audio VAE, with weight norm still attached (the converter removes it)."""
    weights_dir = _weights_dir()
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio

    config = _config(weights_dir)
    reference = AutoencoderKLMiniMaxH3Audio(**config).eval()
    if load_weights:
        from safetensors.torch import load_file

        reference.load_state_dict(load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")))
    return reference, config


def _psnr(reference: torch.Tensor, test: torch.Tensor) -> float:
    """Peak relative to the reference's own dynamic range, as ``test_audio_ltx.py`` does."""
    mse = torch.mean((reference.float() - test.float()) ** 2).item()
    if mse == 0.0:
        return float("inf")
    peak = reference.abs().max().item()
    if peak == 0.0:
        return float("inf")
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def _log_mel_distance(a: torch.Tensor, b: torch.Tensor, *, n_fft: int = 1024, hop: int = 256) -> float:
    """Mean absolute log-magnitude-spectrogram difference.

    A plain STFT rather than a mel filterbank: the point is to catch spectral degradation
    that an aggregate PCC hides, and that does not need a perceptual weighting.
    """
    window = torch.hann_window(n_fft)
    spectra = []
    for signal in (a, b):
        flat = signal.reshape(-1, signal.shape[-1]).float()
        stft = torch.stft(flat, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        spectra.append(torch.log(stft.abs().clamp_min(1e-5)))
    return (spectra[0] - spectra[1]).abs().mean().item()


def _tt_decoder(config: dict, mesh_device) -> MiniMaxH3AudioDecoder:
    return MiniMaxH3AudioDecoder(
        latent_channels=config["latent_channels"],
        latent_dim=config["latent_dim"],
        decoder_dim=config["decoder_dim"],
        decoder_rates=tuple(config["decoder_rates"]),
        decoder_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
        resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(d) for d in config["resblock_dilation_sizes"]),
        mesh_device=mesh_device,
    )


@pytest.mark.parametrize("num_latent_frames", PRODUCTION_LATENT_FRAMES)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decode(mesh_device, num_latent_frames):
    """The whole decode path against the reference, at a production duration, stereo.

    The latent is drawn from the reference *encoder* rather than from ``randn``: BigVGAN's
    behaviour on out-of-distribution latents is not representative of what it will see.

    ``dec_in_proj`` (the gated path's first op, 32 -> 2048) is checked in isolation first,
    so a projection bug names itself instead of surfacing as an end-to-end quality miss.
    """
    reference, config = _build_reference()
    torch.manual_seed(1)

    num_samples = num_latent_frames * HOP_LENGTH
    waveform = torch.randn(2, 1, num_samples) * 0.1
    with torch.no_grad():
        posterior = reference.encode(waveform).latent_dist
        latents = posterior.mode()[..., :num_latent_frames]
        expected = reference.decode(latents).sample
        expected_proj = reference.dec_in_proj(latents)

    tt_decoder = _tt_decoder(config, mesh_device)
    tt_decoder.load_torch_state_dict(convert_minimax_h3_audio_state_dict(dict(reference.state_dict())), strict=False)

    projected = tt_decoder._project_latents_device(latents)
    assert projected.shape == expected_proj.shape, f"proj {tuple(projected.shape)} != {tuple(expected_proj.shape)}"
    assert_quality(expected_proj, projected, pcc=0.999)

    actual = tt_decoder(latents)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.99, relative_rmse=AUDIO_RELATIVE_RMSE)

    psnr = _psnr(expected, actual)
    mel_distance = _log_mel_distance(expected, actual)
    assert psnr >= 28.0, f"decode PSNR {psnr:.2f} dB < 28 dB"
    assert mel_distance <= 5.0, f"log-spectrogram distance {mel_distance:.3f} > 5.0"

    # The two stereo channels share one mono decoder, so they must track their own latents
    # rather than being an accidental broadcast of one channel.
    left, right = actual[0, 0], actual[1, 0]
    assert not torch.allclose(left, right, atol=1e-4), "stereo channels are identical -- a broadcast bug"


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decode_accurate_mode(mesh_device, monkeypatch):
    """The whole decode path with every precision lever on, at the 5 s shape.

    The levers are explicit constructor arguments on the conv and filter layers
    (``split_mode="full"``, ``tap_matmul=True``, ``prefer_mac=True``); ``MiniMaxH3AudioDecoder``
    derives them from the env helpers once at construction and passes them down explicitly, so
    the env var is set before construction and the test asserts the explicit values actually
    landed on the leaf modules rather than trusting the plumbing.

    Each lever fixes a measured error source (the per-lever tables live in ``audio_ops``):

    - ``split_mode="full"``: operand splitting past the fp32 conv3d multiplier's ~11 significand
      bits, which is flat in the reduction depth and untouched by ``fp32_dest_acc_en``;
    - ``prefer_mac=True``: the depthwise MAC form is exact where ``ttnn.conv1d`` injected
      1.54e-03 through a single ``Activation1d`` downsampler -- the single largest error source
      in the whole decode (7e-08 for ``snake_beta`` and the upsampler);
    - ``tap_matmul=True``: shifted-matmul convs avoid conv3d's partial-sum rounding across
      ``C_in_block``. The trap case on this route is ``causal_narrow_out``
      (64 -> 16, causal): ``_AlignedOutConv1d`` rounds a non-32-multiple ``C_out`` up and
      allocates the bias at the rounded width, which the conv3d route handles inside
      ``prepare_conv3d_weight_state`` and the tap route must do for itself -- ``conv_post``'s
      8 -> 1 tail exercises exactly that path here.

    Documented accurate-mode quality: rel_rmse 0.45 %, PCC 99.9990 %, PSNR 67.5 dB
    (see ``audio_accurate_mode``'s table). Bars carry the file's usual ~1.3x margin.
    """
    monkeypatch.setenv("MINIMAX_H3_AUDIO_ACCURATE", "1")

    reference, config = _build_reference()
    torch.manual_seed(1)

    num_latent_frames = 207  # 5 s, as in test_decode
    waveform = torch.randn(2, 1, num_latent_frames * HOP_LENGTH) * 0.1
    with torch.no_grad():
        posterior = reference.encode(waveform).latent_dist
        latents = posterior.mode()[..., :num_latent_frames]
        expected = reference.decode(latents).sample

    tt_decoder = _tt_decoder(config, mesh_device)
    # The constructor resolved the env helpers once; confirm the explicit kwargs reached the leaves.
    assert tt_decoder.dec_in_proj.split_mode == "full", "split_mode='full' did not land on dec_in_proj"
    assert tt_decoder.dec_in_proj.tap_matmul, "tap_matmul=True did not land on dec_in_proj"
    assert tt_decoder.decoder.conv_post.split_mode == "full", "split_mode='full' did not land on conv_post"
    assert tt_decoder.decoder.act_post.downsample.lowpass.prefer_mac, "prefer_mac=True did not land on act_post"

    tt_decoder.load_torch_state_dict(convert_minimax_h3_audio_state_dict(dict(reference.state_dict())), strict=False)
    actual = tt_decoder(latents)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
    # Measured 0.0045 / 99.9990 % / 67.5 dB; rel_rmse carries ~1.3x margin, PSNR is floored at 60 dB.
    assert_quality(expected, actual, pcc=0.9999, relative_rmse=0.006)
    psnr = _psnr(expected, actual)
    assert psnr >= 60.0, f"accurate-mode PSNR {psnr:.2f} dB < 60 dB"


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_conv_operand_split_improves_precision(mesh_device):
    """Operand splitting really does beat the fp32 conv floor, at `conv_pre`'s production shape.

    The floor is the multiplier, not the accumulator: an fp32 conv3d keeps ~11 significand bits per
    operand and its relative error is flat in the reduction depth, so ``fp32_dest_acc_en`` and HiFi4 --
    both already on -- cannot improve it. Splitting an operand into ``hi = bf16(v)`` and the exact
    residual ``lo = v - hi`` lets a second conv carry the bits the first dropped.

    Gated as an inequality against the ``off`` baseline rather than at absolute values, because the
    absolute numbers are hardware- and blocking-dependent while the *ordering* is the claim. The
    baseline itself does carry a loose absolute ceiling, which is what would catch the blocking
    registration silently regressing: unregistered, this shape measures 2.40e-03 instead of
    1.86e-03 because ``C_in_block`` falls back to 32.
    """
    in_channels, out_channels, kernel = CONV_PRE_SHAPE
    torch.manual_seed(0)
    reference = torch.nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2).float().eval()
    x = torch.randn(2, in_channels, CONV_PRE_LATENT_FRAMES) * 0.1
    with torch.no_grad():
        # fp64 golden, so the reference's own fp32 rounding is excluded from every number below.
        golden = copy.deepcopy(reference).double()(x.double()).float()

    x_device = None
    errors = {}
    for mode in ("off", "weight", "full"):
        # The env var does not reach the layer: MiniMax-H3 modules resolve it once at
        # construction and pass the explicit ``split_mode`` argument, so drive that directly.
        layer = Conv1dViaConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            padding_mode="zeros",
            bias=True,
            mesh_device=mesh_device,
            dtype=ttnn.float32,
            split_mode=mode,
        )
        assert layer.split_mode == mode, f"layer resolved split_mode {layer.split_mode!r}, expected {mode!r}"
        assert (layer.weight_lo is None) == (mode == "off"), f"weight residual allocation wrong for {mode!r}"
        layer.load_torch_state_dict(
            {
                "weight": reference.weight.detach().contiguous(),
                "bias": reference.bias.detach().contiguous(),
            },
            strict=False,
        )
        if x_device is None:
            x_device = ttnn.from_torch(
                x.transpose(1, 2).contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
            )
        actual = ttnn.to_torch(layer(x_device)).float().transpose(1, 2)
        errors[mode] = float((actual.double() - golden.double()).pow(2).mean().sqrt() / golden.double().std())

    logger.info("conv_pre rel_rmse by split mode: " + ", ".join(f"{k}={v:.3e}" for k, v in errors.items()))

    # Guards the blocking registration, not the split: 2.1e-03 sits between production's 1.86e-03 and
    # the 2.40e-03 an unregistered C_in_block=32 fallback would give.
    assert errors["off"] <= 2.1e-3, (
        f"baseline conv_pre error {errors['off']:.3e} is above the production floor -- "
        "C_in_block has probably fallen back to 32, meaning register_h3_audio_blockings() did not run"
    )
    # Measured 1.86e-03 / 1.25e-03 / 1.00e-03; bars carry ~1.3x margin on each ratio.
    assert errors["weight"] <= 0.85 * errors["off"], f"weight-split did not improve: {errors}"
    assert errors["full"] <= 0.70 * errors["off"], f"full split did not improve enough: {errors}"
    assert errors["full"] < errors["weight"], f"full split should beat weight-only: {errors}"


def _tt_encoder(config: dict, mesh_device):
    from ....models.audio_vae.minimax_h3.encoder_minimax_h3_audio import MiniMaxH3AudioEncoder

    return MiniMaxH3AudioEncoder(
        encoder_dim=config["encoder_dim"],
        encoder_rates=tuple(config["encoder_rates"]),
        latent_dim=config["latent_dim"],
        latent_channels=config["latent_channels"],
        num_attention_heads=config["num_attention_heads"],
        mesh_device=mesh_device,
    )


@pytest.mark.parametrize("num_latent_frames", PRODUCTION_LATENT_FRAMES)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_encode(mesh_device, num_latent_frames):
    """The whole encode path -- DAC trunk, ``pre_block``, posterior heads -- vs the reference.

    The pipeline only ever consumes ``mode()``, i.e. the mean, so that carries the tight bar;
    ``logs`` is checked more loosely so a ``logs_proj`` regression still shows up.
    """
    reference, config = _build_reference()
    torch.manual_seed(2)

    waveform = torch.randn(2, 1, num_latent_frames * HOP_LENGTH) * 0.1
    with torch.no_grad():
        posterior = reference.encode(waveform).latent_dist
        expected_mean, expected_logs = posterior.mean, posterior.logs

    tt_encoder = _tt_encoder(config, mesh_device)
    tt_encoder.load_torch_state_dict(convert_minimax_h3_audio_state_dict(dict(reference.state_dict())), strict=False)
    mean, logs = tt_encoder(waveform)

    assert mean.shape == expected_mean.shape, f"mean shape {tuple(mean.shape)} != {tuple(expected_mean.shape)}"
    assert mean.shape[2] == num_latent_frames, f"expected {num_latent_frames} latent frames, got {mean.shape[2]}"
    assert_quality(expected_mean, mean, pcc=0.99, relative_rmse=AUDIO_RELATIVE_RMSE)
    assert_quality(expected_logs, logs, pcc=0.98)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_roundtrip(mesh_device):
    """End to end: encode then decode on device, against the reference's own round trip.

    Compared against the reference round trip rather than the input waveform -- a VAE round
    trip is lossy, and the reference is the contract.
    """
    reference, config = _build_reference()
    torch.manual_seed(3)
    num_latent_frames = 207

    waveform = torch.randn(2, 1, num_latent_frames * HOP_LENGTH) * 0.1
    with torch.no_grad():
        latents = reference.encode(waveform).latent_dist.mode()
        expected = reference.decode(latents).sample

    tt_encoder = _tt_encoder(config, mesh_device)
    tt_decoder = _tt_decoder(config, mesh_device)
    converted = convert_minimax_h3_audio_state_dict(dict(reference.state_dict()))
    tt_encoder.load_torch_state_dict(converted, strict=False)
    tt_decoder.load_torch_state_dict(converted, strict=False)

    mean, _ = tt_encoder(waveform)
    actual = tt_decoder(mean)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    psnr = _psnr(expected, actual)
    assert psnr >= 28.0, f"round-trip PSNR {psnr:.2f} dB < 28 dB"
    assert _log_mel_distance(expected, actual) <= 5.0, "round-trip spectrum drifted"


# -------------------------------------------------------------------- weight-norm fusion and checkpoint conversion
#
# Gate M8d.0: the MiniMax-H3 audio checkpoint conversion. Host only, no device.
#
# This is the cheapest high-value gate in the audio port, because it kills the two bugs
# most likely to produce a decoder that is *well-formed but subtly wrong*:
#
# 1. **The ConvTranspose1d weight-norm axis.** ``torch.nn.utils.weight_norm`` defaults to
#    ``dim=0``, and for ``ConvTranspose1d`` axis 0 is ``in_channels``, not ``out_channels``,
#    because the weight is stored ``(in, out, k)``. Fusing over the wrong axis still yields
#    correctly-shaped weights, so nothing downstream complains -- the audio just sounds
#    wrong, and it gets misattributed to precision.
# 2. **The ``activations`` interleave.** H3 stores six activations per AMP block flat;
#    ``AMPBlock1`` wants two lists of three. Swapping them is equally invisible.
#
# Correctness is measured against torch itself: build the reference module, call
# ``remove_weight_norm``, and compare. That makes the reference the oracle rather than a
# second copy of the same arithmetic.


def test_fuse_weight_norm_matches_torch_conv_transpose1d():
    """The load-bearing case: axis 0 of a ConvTranspose1d weight is ``in_channels``.

    (The trivial Conv1d axis case is subsumed by this test plus the real-checkpoint fusion
    gate below, which fuses every Conv1d in the checkpoint against torch's own arithmetic.)
    """
    torch.manual_seed(1)
    conv = torch.nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2)
    assert conv.weight.shape == (32, 16, 4), "ConvTranspose1d weight is (in, out, k)"
    conv = torch.nn.utils.weight_norm(conv)
    with torch.no_grad():
        conv.weight_g.normal_(1.0, 0.2)
        conv.weight_v.normal_()
    assert conv.weight_g.shape == (32, 1, 1), "weight_g is per-in_channel, confirming dim=0"

    weight_g = conv.weight_g.detach().clone()
    weight_v = conv.weight_v.detach().clone()
    fused = fuse_weight_norm(weight_g, weight_v)
    torch.nn.utils.remove_weight_norm(conv)

    relative = (fused - conv.weight).abs().max().item() / conv.weight.abs().max().item()
    assert relative < 1e-6, f"ConvTranspose1d fusion differs from torch by {relative:.3e}"

    # Show that reducing over the wrong axis is silently type-correct: same shape, wrong
    # values. That is precisely why the axis needs a test rather than a comment.
    wrong_norm = weight_v.transpose(0, 1).flatten(1).norm(dim=1).view(1, -1, 1)
    wrong = weight_g * weight_v / wrong_norm
    assert wrong.shape == fused.shape, "the wrong axis still type-checks -- hence this test"
    assert not torch.allclose(wrong, fused, atol=1e-4), "the two axes agree, so this test proves nothing"


def test_remap_amp_activations_interleaves_correctly():
    """``activations.{0,2,4}`` -> ``acts1.{0,1,2}`` and ``{1,3,5}`` -> ``acts2.{0,1,2}``."""
    state = {f"resblocks.0.activations.{i}.act.alpha": torch.tensor([float(i)]) for i in range(6)}
    remapped = remap_amp_activations(state)

    for i in range(3):
        assert remapped[f"resblocks.0.acts1.{i}.act.alpha"].item() == 2 * i
        assert remapped[f"resblocks.0.acts2.{i}.act.alpha"].item() == 2 * i + 1
    assert not any("activations." in key for key in remapped), "an activations key survived"


def test_fuse_attention_biases_rejects_a_nonzero_k_bias():
    """A ``zero_k_bias`` that is not zero must fail loudly, not be dropped."""
    state = {
        "pre_block.attn.q_bias": torch.ones(8),
        "pre_block.attn.v_bias": torch.full((8,), 2.0),
        "pre_block.attn.zero_k_bias": torch.zeros(8),
    }
    fused = fuse_attention_biases(state)
    bias = fused["pre_block.attn.qkv.bias"]
    assert bias.shape == (24,)
    assert torch.equal(bias[:8], torch.ones(8))
    assert torch.equal(bias[8:16], torch.zeros(8))
    assert torch.equal(bias[16:], torch.full((8,), 2.0))

    state["pre_block.attn.zero_k_bias"] = torch.ones(8)
    with pytest.raises(AssertionError, match="not all zero"):  # allow-pytest.raises: guards a silent data-loss path
        fuse_attention_biases(state)


def test_real_checkpoint_fusion_matches_reference_module():
    """The real 1087-tensor checkpoint: axes hold, every pair fuses, and the fused weights
    equal what the reference's own ``remove_weight_norm`` produces.

    One checkpoint load carries both the key-accounting asserts and the value comparison
    against torch's own arithmetic.
    """
    weights_dir = _weights_dir()
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from safetensors.torch import load_file

    config = {
        k: v
        for k, v in json.loads(open(os.path.join(weights_dir, "config.json")).read()).items()
        if not k.startswith("_")
    }
    reference = AutoencoderKLMiniMaxH3Audio(**config)
    state = load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
    reference.load_state_dict(state)

    assert_weight_norm_axes_consistent(state)
    num_pairs = len([k for k in state if k.endswith(".weight_g")])
    assert num_pairs > 100, f"expected ~172 weight-normed convs, found {num_pairs}"

    converted = convert_minimax_h3_audio_state_dict(state)
    assert not [k for k in converted if k.endswith((".weight_g", ".weight_v"))], "a weight-norm pair survived"
    assert not [k for k in converted if k.endswith(("q_bias", "v_bias", "zero_k_bias"))], "an attn bias survived"
    assert not [k for k in converted if "activations." in k], "an activations key survived"
    # Every fused conv produces exactly one weight, and nothing else is lost.
    assert (
        len(converted) == len(state) - num_pairs - 2
    ), f"key count {len(converted)} does not match {len(state)} minus {num_pairs} g/v pairs and 2 folded biases"

    for module in reference.modules():
        if hasattr(module, "weight_g"):
            torch.nn.utils.remove_weight_norm(module)
    expected = dict(reference.state_dict())

    checked = 0
    worst_key, worst = None, 0.0
    for key, value in expected.items():
        if not key.endswith(".weight") or key not in converted:
            continue
        scale = max(value.abs().max().item(), 1e-12)
        relative = (converted[key] - value).abs().max().item() / scale
        if relative > worst:
            worst_key, worst = key, relative
        checked += 1
    assert checked > 100, f"only compared {checked} fused weights"
    assert worst < 1e-6, f"worst fused weight is {worst_key} at relative {worst:.3e}"


# -------------------------------------------------------------------- traced audio decode
#
# Traced vs untraced audio decode, with a correctness gate between them.
#
# Audio decode measures 1.273 s against a ~0.05 s target. It gets
# nothing from the data-parallelism that carried the visual path -- a 5 s clip is one stream,
# not 224 independent work units -- but it is the opposite kind of workload from the visual
# halves: ~1 MB tensors over many ops, so **host dispatch**, not device time, is expected to
# dominate. ``vocoder_ltx.Vocoder`` says so itself ("the vocoder is ~70% host-bound") and
# carries a `@traced_function` device region plus a ``forward_traced`` entry point; H3's
# decoder exposes the same device region through ``traced=True``.
#
# Traced output must match untraced exactly-ish, so this asserts before it reports timing.


# 300 MB covers the default path but not the precision levers: with ``MINIMAX_H3_AUDIO_ACCURATE=1`` the
# graph grows (the depthwise MAC form does one pass per tap, and the shifted-matmul convs add
# ``3 * kernel_size`` matmuls each), and the trace needs 375463936 B. The failure names the requirement
# exactly -- ``mesh_trace.cpp:80``, "Creating trace buffers of size ... but only ... is allocated" -- so
# size for the larger of the two rather than making the region depend on an env var.
TRACED = [
    pytest.param((1, 1), {"l1_small_size": 65536, "trace_region_size": 450_000_000}, id="single_device"),
]
NUM_LATENT_FRAMES = 207
ITERS = 5


def _best(fn) -> float:
    fn()
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


@pytest.mark.parametrize(("mesh_device", "device_params"), TRACED, indirect=["mesh_device", "device_params"])
def test_audio_decode_traced(mesh_device):
    weights_dir = _weights_dir()
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from loguru import logger
    from safetensors.torch import load_file

    from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict

    config = _config(weights_dir)
    reference = AutoencoderKLMiniMaxH3Audio(**config).eval()
    reference.load_state_dict(load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")))
    converted = convert_minimax_h3_audio_state_dict(dict(reference.state_dict()))

    torch.manual_seed(2)
    decoder = MiniMaxH3AudioDecoder(
        latent_channels=config["latent_channels"],
        latent_dim=config["latent_dim"],
        decoder_dim=config["decoder_dim"],
        decoder_rates=tuple(config["decoder_rates"]),
        decoder_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
        resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(d) for d in config["resblock_dilation_sizes"]),
        mesh_device=mesh_device,
    )
    decoder.load_torch_state_dict(converted, strict=False)

    latents = torch.randn(2, config["latent_channels"], NUM_LATENT_FRAMES) * 0.1

    plain = decoder(latents)
    traced = decoder(latents, traced=True)
    assert traced.shape == plain.shape, f"{traced.shape} != {plain.shape}"

    # Trace replays the same program on the same data, so bit-identical (PSNR inf) is the
    # *expected* result rather than a suspicious one. It is a weak assertion on its own
    # though -- it would read inf just as happily if traced=True had silently fallen through
    # to the untraced path -- so check separately that a tracer was actually captured, and
    # that the output is not trivially zero (which would also give inf).
    tracers = type(decoder.decoder)._forward_device._tracers_keyed.get(decoder.decoder, {})
    assert tracers, "traced=True captured no trace; the PSNR below would be meaningless"
    assert plain.abs().max() > 1e-6, "decoder produced all-zero output; PSNR would be vacuous"

    psnr = _psnr(plain, traced)
    logger.info(f"traced vs untraced PSNR: {psnr:.2f} dB ({len(tracers)} trace(s) captured)")

    untraced_s = _best(lambda: decoder(latents))
    traced_s = _best(lambda: decoder(latents, traced=True))
    logger.info(
        f"PERF audio_decode_5s untraced {untraced_s:.4f} s | traced {traced_s:.4f} s "
        f"-> {untraced_s / traced_s:.2f}x"
    )
    decoder.release_trace()

    assert psnr > 60.0, f"traced output diverges from untraced: PSNR {psnr:.2f} dB"
