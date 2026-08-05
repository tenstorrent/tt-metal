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
"""

import copy
import json
import math
import os

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.audio_ops import Conv1dViaConv3d

# Imported for the decoder gates below -- but note the import itself also runs
# ``register_h3_audio_blockings()``, which is what puts the H3 audio conv shapes into
# ``_FP32_BLOCKINGS``. Without it every conv here would silently fall back to the conservative
# ``C_in_block = 32`` default and measure a *different op* than production (STATE.md am. 111).
from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from ....models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from ....utils.check import assert_quality

# The vocoder needs extra L1 scratch, as the LTX audio tests do.
SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

# RMSE/sigma bar, set from measurement rather than convention. Candidate error sources were each
# measured: the MAC fallback in depthwise_tap_filter is **bit-exact** (rel_max 0.0), `Activation1d` is
# pcc 0.9999998 / RMSE 0.17%, and the decoder path contains no SDPA so the bf16 attention island cannot
# explain it. What remains is accumulation across the chain -- and the per-op term is now identified
# rather than inferred: an fp32 conv3d on this hardware is fp32 storage and fp32 accumulate with a
# multiply that keeps only ~11 significand bits, which measures 1.86e-03 on `conv_pre` at production
# blocking and is **flat in the reduction depth**. Over ~130 convolutions plus 126 anti-aliased
# activations that lands at ~10% RMSE while PCC stays at 99.5%.
#
# So PCC plus the perceptual gates (PSNR, log-spectrogram distance) are the meaningful bars here, and
# RMSE is held at a level consistent with the measured chain depth rather than at a value only a
# shallower model could reach. `MINIMAX_H3_AUDIO_CONV_SPLIT=full` halves it to 5.4% by splitting the
# conv operands (see `conv_split_mode` and STATE.md am. 111-112); this bar deliberately still describes
# the **default** path, so it must be re-derived if that default ever changes.
AUDIO_RELATIVE_RMSE = 0.12

# `conv_pre`: the decoder's widest reduction, Cin 2048 x k 7 = 14336, and the shape every operand-split
# measurement in am. 111 was taken at. Its T is the 5 s latent length, as in the decode gates.
CONV_PRE_SHAPE = (2048, 1024, 7)
CONV_PRE_LATENT_FRAMES = 207

LATENT_CHANNELS = 32
HOP_LENGTH = 800
SAMPLING_RATE = 32000

# 40 Hz latent: 5 s and 10 s at the video working points' true durations.
PRODUCTION_LATENT_FRAMES = [pytest.param(207, id="5s"), pytest.param(405, id="10s")]


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


def test_decoder_geometry_matches_config():
    """Host-only: 7 upsample stages whose product is the 800-sample hop, down to 8 channels.

    Cheap, but it pins the two things an earlier plan draft got wrong -- it recorded six
    stages of ``[5,5,2,2,2,2]`` -- and the narrow tail that the device gates then probe.
    """
    weights_dir = _weights_dir()
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = _config(weights_dir)

    rates = config["decoder_rates"]
    assert len(rates) == 7, f"expected 7 upsample stages, config has {len(rates)}"
    assert math.prod(rates) == HOP_LENGTH, f"rates multiply to {math.prod(rates)}, not {HOP_LENGTH}"
    assert math.prod(config["encoder_rates"]) == HOP_LENGTH, "encoder and decoder rates disagree"
    assert len(config["decoder_kernel_sizes"]) == len(rates), "one kernel size per stage"

    final_channels = config["decoder_dim"] // (2 ** len(rates))
    assert final_channels == 8, f"expected an 8-channel tail, got {final_channels}"
    assert config["sampling_rate"] == SAMPLING_RATE


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_dec_in_proj(mesh_device):
    """The 1x1 latent projection, 32 -> 2048, against the reference."""
    reference, config = _build_reference()
    torch.manual_seed(0)
    latents = torch.randn(2, LATENT_CHANNELS, 64)
    with torch.no_grad():
        expected = reference.dec_in_proj(latents)

    tt_decoder = _tt_decoder(config, mesh_device)
    converted = convert_minimax_h3_audio_state_dict(dict(reference.state_dict()))
    tt_decoder.load_torch_state_dict({k: v for k, v in converted.items() if k.startswith("dec_in_proj.")}, strict=False)
    actual = tt_decoder._project_latents_device(latents)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.999)


@pytest.mark.parametrize("num_latent_frames", PRODUCTION_LATENT_FRAMES)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decode(mesh_device, num_latent_frames):
    """The whole decode path against the reference, at a production duration, stereo.

    The latent is drawn from the reference *encoder* rather than from ``randn``: BigVGAN's
    behaviour on out-of-distribution latents is not representative of what it will see.
    """
    reference, config = _build_reference()
    torch.manual_seed(1)

    num_samples = num_latent_frames * HOP_LENGTH
    waveform = torch.randn(2, 1, num_samples) * 0.1
    with torch.no_grad():
        posterior = reference.encode(waveform).latent_dist
        latents = posterior.mode()[..., :num_latent_frames]
        expected = reference.decode(latents).sample

    tt_decoder = _tt_decoder(config, mesh_device)
    tt_decoder.load_torch_state_dict(convert_minimax_h3_audio_state_dict(dict(reference.state_dict())), strict=False)
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
def test_conv_operand_split_improves_precision(mesh_device, monkeypatch):
    """Operand splitting really does beat the fp32 conv floor, at `conv_pre`'s production shape.

    The floor is the multiplier, not the accumulator: an fp32 conv3d keeps ~11 significand bits per
    operand and its relative error is flat in the reduction depth, so ``fp32_dest_acc_en`` and HiFi4 --
    both already on -- cannot improve it. Splitting an operand into ``hi = bf16(v)`` and the exact
    residual ``lo = v - hi`` lets a second conv carry the bits the first dropped.

    Gated as an inequality against the ``off`` baseline rather than at absolute values, because the
    absolute numbers are hardware- and blocking-dependent while the *ordering* is the claim. The
    baseline itself does carry a loose absolute ceiling, which is what would catch the blocking
    registration silently regressing (am. 111): unregistered, this shape measures 2.40e-03 instead of
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
        # Read at construction, so it must be set before the layer is built.
        monkeypatch.setenv("MINIMAX_H3_AUDIO_CONV_SPLIT", mode)
        layer = Conv1dViaConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            padding_mode="zeros",
            bias=True,
            mesh_device=mesh_device,
            dtype=ttnn.float32,
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
