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

from __future__ import annotations

import copy
import json
import math
import os
import struct
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.audio_ops import Conv1dViaConv3d

# Imported for the decoder gates below -- but note the import itself also runs
# ``register_h3_audio_blockings()``, which is what puts the H3 audio conv shapes into
# ``_FP32_BLOCKINGS``. Without it every conv here would silently fall back to the conservative
# ``C_in_block = 32`` default and measure a *different op* than production (STATE.md am. 111).
from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import (
    assert_weight_norm_axes_consistent,
    convert_minimax_h3_audio_state_dict,
    fuse_attention_biases,
    fuse_weight_norm,
    remap_amp_activations,
)
from ....models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from ....parallel.config import ParallelFactor
from ....parallel.manager import CCLManager
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
# shallower model could reach.
#
# That chain depth is not a floor, though: `MINIMAX_H3_AUDIO_ACCURATE=1` reaches **0.45 %** RMSE
# (PCC 99.9990 %, PSNR 67.5 dB) for ~3x the stage time, by fixing the three sources this bar's 10.5 %
# is made of -- see `audio_accurate_mode` and STATE.md am. 111-113. This bar describes the
# **default** path, so it must be re-derived if that default ever changes.
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
    # Log it: this is the only PSNR measured against the *CPU* reference. The figure quoted in
    # AUDIO_RESULTS.md is scored against `MINIMAX_H3_AUDIO_ACCURATE=1`, i.e. device against device,
    # which cannot see an error that both device paths share. Keep the two distinguishable.
    logger.info(f"decode {num_latent_frames} latents: PSNR {psnr:.2f} dB vs CPU reference, log-mel {mel_distance:.3f}")
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


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_depthwise_mac_is_more_accurate_than_conv1d(mesh_device, monkeypatch):
    """The MAC form of the depthwise filter beats ``ttnn.conv1d`` by orders of magnitude.

    Structural, not incidental: MAC is a sum of elementwise multiplies and adds, and those are exact in
    fp32 here, while ``ttnn.conv1d`` goes through the FPU multiply that keeps ~11 significand bits. This
    was the single largest error source in the whole decode -- one anti-aliased ``Activation1d`` injected
    1.54e-03, all of it from its downsampler, against 7e-08 for ``snake_beta`` and the upsampler.

    Shape is the real stage-1 downsampler: K=12 kaiser-sinc taps at stride 2.
    """
    from ....layers.audio_ops import depthwise_tap_filter

    channels, t_pad, kernel, stride = 512, 2081, 12, 2
    torch.manual_seed(0)
    taps = torch.randn(kernel).tolist()
    x = torch.randn(1, t_pad, channels) * 0.3

    t_out = (t_pad - kernel) // stride + 1
    golden = torch.stack(
        [sum(taps[k] * x[0, k + stride * i, :].double() for k in range(kernel)) for i in range(t_out)]
    ).unsqueeze(0)

    errors = {}
    for prefer_mac in ("0", "1"):
        monkeypatch.setenv("MINIMAX_H3_AUDIO_DEPTHWISE_MAC", prefer_mac)
        x_device = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
        out = depthwise_tap_filter(x_device, taps, stride, mesh_device=mesh_device, dtype=ttnn.float32, cache={})
        actual = ttnn.to_torch(out).float()
        assert actual.shape[1] == t_out, f"T_out {actual.shape[1]} != {t_out}"
        errors[prefer_mac] = float((actual.double() - golden).pow(2).mean().sqrt() / golden.std())

    logger.info(f"depthwise filter rel_rmse: conv1d={errors['0']:.3e} mac={errors['1']:.3e}")
    # This gate used to assert MAC beat conv1d by >100x, which was true when conv1d measured 1.5e-03
    # against MAC's 5.3e-08. It is not true any more, and the reason is the point of the gate: the
    # depthwise kernel now accumulates taps on the SFPU with UnpackToDestFp32 operands, so conv1d is
    # bit-equal to the elementwise form -- both measure ~8e-08 here -- while running 5-26x faster.
    #
    # So the property worth protecting is no longer "MAC wins", it is "neither loses". Asserting the
    # old inequality would now fail *because* the defect is fixed, and asserting equality would be too
    # tight to survive a legitimate kernel change; both being fp32-grade is the real contract.
    assert errors["1"] <= 1e-6, f"MAC form should be fp32-grade, got {errors['1']:.3e}"
    assert errors["0"] <= 1e-6, (
        f"conv1d should be fp32-grade now that the depthwise kernel uses SFPU tap accumulation, "
        f"got {errors['0']:.3e}. A regression here means the SFPU branch or the UnpackToDestFp32 "
        f"operands stopped taking effect -- check `sfpu_fp32_enabled` and the conv2d program factory."
    )


@pytest.mark.parametrize("channels", [8, 16, 24, 32])
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_channel_padding_is_bit_exact(mesh_device, channels):
    """Channel padding must not perturb the data it pads. It used to, at C=8 and C=24.

    `_pad_channels_to_aligned` built the padded tensor with ``ttnn.concat(..., dim=2)``. In fp32 that
    is lossy whenever the row length is not a multiple of the buffer alignment (64 B on Blackhole):
    `build_non_aligned_last_dim_concat` routes such cases through a ``ttnn.transpose(-2,-1)`` round
    trip, and fp32 ROW_MAJOR transpose truncates the mantissa to TF32, returning ``x & 0xFFFFE000``.
    C=8 (32 B row) and C=24 (96 B) hit it; C=16 (64 B) and C=32 (128 B) did not.

    The gate is bit-exactness against CPU rather than a tolerance, because a padding op has no licence
    to change a single bit. C=16 and C=32 are included so the test still covers the aligned path if
    the implementation changes again.

    Note why `test_decode` never caught this despite scoring against the CPU reference: at ~1e-03 the
    truncation sits far below the end-to-end error, which measures 41.4 dB against a 28 dB gate.
    Reverting the fix and re-measuring gives 41.40 dB against 41.41 dB -- the corruption is invisible
    at the model level. A tolerance wide enough for a whole decode cannot see one lossy data-movement
    op, so ops whose contract is exactness need their own bit-exact gates. Hence this one.
    """
    from ....layers.audio_ops import _pad_channels_to_aligned, aligned_channels

    torch.manual_seed(0)
    x = torch.randn(2, 1024, channels) * 0.3
    x_device = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    actual = ttnn.to_torch(_pad_channels_to_aligned(x_device, mesh_device, channel_align=32)).float()

    padded = aligned_channels(channels, 32)
    expected = torch.zeros(2, 1024, padded)
    expected[:, :, :channels] = x
    assert tuple(actual.shape) == tuple(expected.shape), f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    maxdiff = float((actual - expected).abs().max())
    assert torch.equal(actual, expected), (
        f"channel padding at C={channels} is not bit-exact (maxdiff {maxdiff:.3e}). If this is "
        f"~1e-03, the padding is going through a last-dim fp32 concat again and losing 13 mantissa "
        f"bits to TF32; use ttnn.pad, which is both exact and faster."
    )


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(
    ("in_channels", "out_channels", "kernel", "dilation", "padding_mode"),
    [
        (512, 512, 3, 1, "zeros"),  # AMP conv, the most numerous shape in the decoder
        (512, 512, 3, 5, "zeros"),  # dilated
        (2048, 1024, 7, 1, "zeros"),  # conv_pre, the widest reduction
        (64, 16, 7, 1, "causal"),  # non-32-multiple C_out: the bias must be padded like conv3d's is
    ],
    ids=["amp_k3", "amp_k3_d5", "conv_pre", "causal_narrow_out"],
)
def test_tap_matmul_beats_conv3d(mesh_device, monkeypatch, in_channels, out_channels, kernel, dilation, padding_mode):
    """The shifted-matmul form of a stride-1 conv is more accurate than conv3d, at equal split.

    conv3d's residual after operand splitting is partial-sum rounding across ``C_in_block`` -- not the
    operand mantissa, since a 3-way split measures bit-identically to a 2-way one -- and matmul has no
    such blocking. Both formulations run with the same operand split so the comparison isolates the
    formulation.

    ``causal_narrow_out`` is here because it is the case that broke first: ``_AlignedOutConv1d`` rounds a
    non-32-multiple ``C_out`` up and the bias is allocated at the rounded width, which the conv3d route
    handles inside ``prepare_conv3d_weight_state`` and the tap route must do for itself.
    """
    torch.manual_seed(0)
    effective_kernel = (kernel - 1) * dilation + 1
    padding = effective_kernel - 1 if padding_mode == "causal" else effective_kernel // 2
    reference = torch.nn.Conv1d(in_channels, out_channels, kernel, padding=padding, dilation=dilation).float().eval()
    x = torch.randn(2, in_channels, CONV_PRE_LATENT_FRAMES) * 0.3
    with torch.no_grad():
        golden = copy.deepcopy(reference).double()(x.double()).float()[..., :CONV_PRE_LATENT_FRAMES]

    state = {"weight": reference.weight.detach().contiguous(), "bias": reference.bias.detach().contiguous()}
    x_device = ttnn.from_torch(
        x.transpose(1, 2).contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
    )

    errors, shapes = {}, {}
    for tap in ("0", "1"):
        monkeypatch.setenv("MINIMAX_H3_AUDIO_TAP_MATMUL", tap)
        monkeypatch.setenv("MINIMAX_H3_AUDIO_CONV_SPLIT", "full")
        layer = Conv1dViaConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            dilation=dilation,
            padding_mode=padding_mode,
            bias=True,
            mesh_device=mesh_device,
            dtype=ttnn.float32,
        )
        assert layer.tap_matmul == (tap == "1"), f"tap_matmul resolved to {layer.tap_matmul} for flag {tap}"
        layer.load_torch_state_dict(dict(state), strict=False)
        actual = ttnn.to_torch(layer(x_device)).float().transpose(1, 2)
        shapes[tap] = tuple(actual.shape)
        rows = min(actual.shape[-1], golden.shape[-1])
        channels = min(actual.shape[-2], golden.shape[-2])
        errors[tap] = float(
            (actual[:, :channels, :rows].double() - golden[:, :channels, :rows].double()).pow(2).mean().sqrt()
            / golden.double().std()
        )

    logger.info(f"conv3d={errors['0']:.3e} tap_matmul={errors['1']:.3e} shapes={shapes}")
    assert shapes["0"] == shapes["1"], f"the two formulations disagree on output shape: {shapes}"
    # Measured gains 1.8x-3.5x across these shapes; 1.4x leaves margin without being vacuous.
    assert errors["1"] <= errors["0"] / 1.4, f"tap-matmul should beat conv3d: {errors}"


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


def _checkpoint_header(path: str) -> dict:
    """safetensors header only -- shapes without reading 605 MB of tensor data."""
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return {k: v for k, v in json.loads(handle.read(length)).items() if k != "__metadata__"}


def test_fuse_weight_norm_matches_torch_conv1d():
    """``fuse_weight_norm`` == what torch's own ``remove_weight_norm`` leaves behind."""
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(16, 32, kernel_size=7)
    conv = torch.nn.utils.weight_norm(conv)
    with torch.no_grad():
        conv.weight_g.normal_(1.0, 0.2)
        conv.weight_v.normal_()

    fused = fuse_weight_norm(conv.weight_g.detach(), conv.weight_v.detach())
    torch.nn.utils.remove_weight_norm(conv)

    assert fused.shape == conv.weight.shape
    relative = (fused - conv.weight).abs().max().item() / conv.weight.abs().max().item()
    assert relative < 1e-6, f"Conv1d fusion differs from torch by {relative:.3e}"


def test_fuse_weight_norm_matches_torch_conv_transpose1d():
    """The load-bearing case: axis 0 of a ConvTranspose1d weight is ``in_channels``."""
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


def test_real_checkpoint_axes_and_conversion():
    """The real 1087-tensor checkpoint: axis assumptions hold and every pair fuses."""
    weights_dir = _weights_dir()
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    from safetensors.torch import load_file

    state = load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
    assert_weight_norm_axes_consistent(state)

    num_pairs = len([k for k in state if k.endswith(".weight_g")])
    assert num_pairs > 100, f"expected ~172 weight-normed convs, found {num_pairs}"

    converted = convert_minimax_h3_audio_state_dict(state)
    assert not [k for k in converted if k.endswith((".weight_g", ".weight_v"))], "a weight-norm pair survived"
    assert not [k for k in converted if k.endswith(("q_bias", "v_bias", "zero_k_bias"))], "an attn bias survived"
    assert not [k for k in converted if "activations." in k], "an activations key survived"
    # Every fused conv should have produced exactly one weight, and nothing else was lost.
    assert (
        len(converted) == len(state) - num_pairs - 2
    ), f"key count {len(converted)} does not match {len(state)} minus {num_pairs} g/v pairs and 2 folded biases"


def test_real_checkpoint_fusion_matches_reference_module():
    """Fused weights equal what the reference's own ``remove_weight_norm`` produces."""
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

    for module in reference.modules():
        if hasattr(module, "weight_g"):
            torch.nn.utils.remove_weight_norm(module)
    expected = dict(reference.state_dict())

    converted = convert_minimax_h3_audio_state_dict(state)

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


# -------------------------------------------------------------------- T-parallel audio decode
#
# T-parallel audio decode: correctness against the single-device path, and the speedup.
#
# Audio decode is 1.284 s against a ~0.05 s target, is **device-bound** (trace buys 1.07 %,
# STATE.md amendment 60), and runs on **one device**. The visual halves got 32x from
# data-parallelism over `(clip, tile)` work units; a single 5 s audio stream is one unit, so
# none of that applies. The equivalent lever here is sharding the time axis across the mesh --
# which ``vocoder_ltx.Vocoder`` already implements (``parallel_config.factor`` threads through
# ``_upload_BCT``'s T-alignment padding, ``_forward_device``'s partition, and the closing
# T-gather) and ``MiniMaxH3AudioDecoder`` already accepts. The shipping path simply passes
# ``None``.
#
# Sharded output is gated against the unsharded output of the same weights, so a speedup that
# comes from dropping work fails rather than reports.


MESH = [
    pytest.param(
        (4, 8),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "require_exact_physical_num_devices": True,
            "l1_small_size": 65536,
        },
        id="mesh4x8",
    )
]
# (t_factor, mesh_axis). Axis 1 is the 8-wide axis of the 4x8 Galaxy, axis 0 the 4-wide one.
#
# The factor must equal the length of the axis it shards: factor=2 or factor=4 on the 8-wide axis 1
# both die in `_partition_t` at slice_device_operation.cpp:164 ("height begin index aligned to tiles"),
# because the partition indexes by the device's coordinate along the axis and assumes it covers it.
# That is why this list is (4, axis 0) and (8, axis 1) rather than a scan.
#
# `KNOWN_BROKEN` is deliberately empty, and adding to it should be a last resort -- an entry here
# silences the PSNR assert, which is the only thing separating a speedup from a fast wrong answer.
# It formerly held (8, 1) at -6.3 dB, blamed on 256/8 = 32 being exactly one tile per shard. That was
# the wrong suspect: *every* factor was wrong, and the cause was `conv_pre` (2048 -> 1024, k=7)
# returning uninitialized memory under T-sharding while every other conv shape was bit-exact. It now
# runs unsharded on the full sequence -- see the comment on `Vocoder.conv_pre` -- and both factors
# measure 78.7 dB. See audio_perf/ITEM2_RESULT.md.
FACTORS = [(1, 1), (4, 0), (8, 1)]
KNOWN_BROKEN: set[tuple[int, int]] = set()
NUM_LATENT_FRAMES = 207
ITERS = 3


def _best(fn) -> float:
    fn()
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _build(mesh_device, config, converted, parallel_config, ccl_manager):
    decoder = MiniMaxH3AudioDecoder(
        latent_channels=config["latent_channels"],
        latent_dim=config["latent_dim"],
        decoder_dim=config["decoder_dim"],
        decoder_rates=tuple(config["decoder_rates"]),
        decoder_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
        resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(d) for d in config["resblock_dilation_sizes"]),
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    decoder.load_torch_state_dict(converted, strict=False)
    return decoder


def _localize_divergence(baseline, parallel, *, factor: int, logger) -> None:
    """Say *where* a diverging shard layout diverges. "PSNR -10.2 dB" alone names no bug.

    Three shapes of answer, each pointing somewhere different:

    - concentrated at the internal shard boundaries -> the halo exchange is wrong, and each conv in the
      stack needs `kernel_size - 1` samples of its neighbour that it is not getting;
    - uniform across the whole signal -> the shard layout itself is wrong, not its edges;
    - a prefix or suffix only -> the causal padding or the final trim is off by a shard.

    Also reports the best cross-correlation lag: a diverging-but-highly-correlated output at a nonzero
    lag is a *shift*, which is a trim bug rather than a numerics one, and PSNR cannot distinguish those.
    """
    import numpy as np

    error = (parallel - baseline).abs()
    total = baseline.shape[-1]
    logger.info(
        f"  divergence localization, t_factor={factor}: overall mean {error.mean():.6f} "
        f"max {error.max():.4f} against baseline absmax {baseline.abs().max():.4f}"
    )
    per_shard = []
    for shard in range(factor):
        lo, hi = shard * total // factor, (shard + 1) * total // factor
        per_shard.append(float(error[..., lo:hi].mean()))
    logger.info(f"  per-shard mean error: {[f'{v:.6f}' for v in per_shard]}")

    # Boundary vs interior. A halo bug puts the error in a narrow band at each internal boundary.
    window = 128
    boundary, interior = [], []
    for shard in range(1, factor):
        cut = shard * total // factor
        boundary.append(float(error[..., max(0, cut - window) : cut + window].mean()))
    mask = torch.ones(total, dtype=torch.bool)
    for shard in range(1, factor):
        cut = shard * total // factor
        mask[max(0, cut - window) : cut + window] = False
    interior = float(error[..., mask].mean())
    logger.info(
        f"  boundary bands (+-{window}): {[f'{v:.6f}' for v in boundary]}  interior {interior:.6f}  "
        f"ratio {max(boundary) / max(interior, 1e-12):.2f}"
    )

    a = baseline[0, 0].numpy()
    c = parallel[0, 0].numpy()
    best, best_lag = -2.0, None
    for lag in range(-4096, 4097, 32):
        x, y = (a[-lag:], c[: len(c) + lag]) if lag < 0 else (a[: len(a) - lag], c[lag:])
        n = min(len(x), len(y))
        if n < 2048:
            continue
        r = float(np.corrcoef(x[:n], y[:n])[0, 1])
        if r > best:
            best, best_lag = r, lag
    logger.info(
        f"  correlation at lag 0: {float(np.corrcoef(a, c)[0, 1]):.4f}; "
        f"best {best:.4f} at lag {best_lag} (nonzero lag => a shift, i.e. a trim bug)"
    )


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
def test_audio_decode_t_parallel(mesh_device):
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
    latents = torch.randn(2, config["latent_channels"], NUM_LATENT_FRAMES) * 0.1

    baseline_out = None
    baseline_s = None
    results = []
    for factor, axis in FACTORS:
        pc = None if factor <= 1 else ParallelFactor(factor=factor, mesh_axis=axis)
        ccl = None if pc is None else CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
        try:
            decoder = _build(mesh_device, config, converted, pc, ccl)
            out = decoder(latents)
            seconds = _best(lambda: decoder(latents))
        except Exception as exc:  # a factor the stack rejects is a result, not a test failure
            logger.warning(f"t_factor={factor} axis={axis} FAILED: {str(exc)[:160]}")
            results.append((factor, axis, None, None))
            continue

        if baseline_out is None:
            baseline_out, baseline_s = out, seconds
            psnr = float("inf")
        else:
            assert out.shape == baseline_out.shape, f"factor {factor}: {out.shape} != {baseline_out.shape}"
            psnr = _psnr(baseline_out, out)
        results.append((factor, axis, seconds, psnr))
        logger.info(
            f"PERF audio_decode t_factor={factor} axis={axis}: {seconds:.4f} s "
            f"({baseline_s / seconds:.2f}x) PSNR vs 1-device {psnr:.1f} dB"
        )
        if psnr < 40.0 and out is not baseline_out:
            _localize_divergence(baseline_out.float(), out.float(), factor=factor, logger=logger)
        del decoder

    logger.info("=== audio decode T-parallel summary ===")
    for factor, axis, seconds, psnr in results:
        if seconds is None:
            logger.info(f"  t_factor={factor:2d} axis={axis}: unsupported")
        else:
            logger.info(
                f"  t_factor={factor:2d} axis={axis}: {seconds:.4f} s  {baseline_s / seconds:5.2f}x  "
                f"PSNR {psnr:6.1f} dB"
            )

    # The baseline must have run, or there is nothing to compare against and every other factor was
    # skipped for want of a reference. Without this the test PASSES when the whole stack is broken:
    # observed once on a device left dirty by an unrelated crash, where all three factors raised
    # TT_FATAL, each was swallowed as "unsupported", and the run reported green. A gate that cannot
    # fail is worse than no gate.
    baseline_ran = any(seconds is not None and factor == 1 for factor, _, seconds, _ in results)
    assert baseline_ran, (
        "the single-device baseline did not run, so nothing was compared. If this follows a crashed "
        "run, reset the device (`tt-smi -glx_reset`) -- an allocator TT_FATAL here is usually a dirty "
        "device, not a code failure"
    )
    # `baseline_ran` above catches "everything failed" but not the subtler case: if factor 1 raises,
    # the loop's `baseline_out is None` branch promotes the *next* factor to baseline, and that factor
    # then scores PSNR inf against itself and reads as correct. That happened for real -- a fusion
    # crash killed factor 1, factor 4 became the reference, and a -10.1 dB configuration reported inf
    # until the baseline was made to run. The first result must be the unsharded one.
    assert results and results[0][0] == 1 and results[0][2] is not None, (
        f"factor 1 must be the baseline, but the first result that ran was {results[0][:2] if results else None}; "
        "a later factor has been promoted to baseline and is being compared against itself"
    )
    ran = [(f, a) for f, a, seconds, _ in results if seconds is not None and f != 1]
    assert ran, "no parallel factor ran at all; the T-parallel path is entirely unavailable"

    # Any factor that ran must agree with the single-device path; a fast wrong answer fails.
    for factor, axis, seconds, psnr in results:
        if seconds is None or (factor, axis) in KNOWN_BROKEN:
            continue
        assert psnr > 40.0, f"t_factor={factor} axis={axis} diverges from 1-device: PSNR {psnr:.1f} dB"


# -------------------------------------------------------------------- traced audio decode
#
# Traced vs untraced audio decode, with a correctness gate between them.
#
# Audio decode measures 1.273 s against a ~0.05 s target (STATE.md amendment 59). It gets
# nothing from the data-parallelism that carried the visual path -- a 5 s clip is one stream,
# not 224 independent work units -- but it is the opposite kind of workload from the visual
# halves: ~1 MB tensors over many ops, so **host dispatch**, not device time, is expected to
# dominate. ``vocoder_ltx.Vocoder`` says so itself ("the vocoder is ~70% host-bound") and
# already carries a `@traced_function` device region plus a ``forward_traced`` entry point --
# H3's decoder simply called the untraced ``forward_BCT``.
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

    # Where is the 1.2 s? The traced region is only the vocoder's `_forward_device`; the
    # latent projection round-trips through host in the middle of forward
    # (decoder_minimax_h3_audio.py: to_torch -> transpose/contiguous -> re-upload), and the
    # final readback is untraced too. Split them so the next step is not a guess.
    proj_s = _best(lambda: decoder._project_latents_device(latents))
    projected = decoder._project_latents_device(latents)
    voc_s = _best(lambda: decoder.decoder.forward_BCT(projected))
    voc_traced_s = _best(lambda: decoder.decoder.forward_BCT_traced(projected))
    logger.info(
        f"PERF split: dec_in_proj {proj_s:.4f} s | vocoder {voc_s:.4f} s | " f"vocoder traced {voc_traced_s:.4f} s"
    )
    decoder.release_trace()

    assert psnr > 60.0, f"traced output diverges from untraced: PSNR {psnr:.2f} dB"
