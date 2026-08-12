# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gate M8d: the MiniMax-H3 audio VAE (BigVGAN) at production shapes; stereo rides as batch 2.
T-parallel decode is unwired and its 8-way shard layout known-broken -- if it ships, resurrect
``test_audio_decode_t_parallel`` and ``test_neighbor_pad_t_minimax_h3.py`` from git history."""

from __future__ import annotations

import copy
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.audio_ops import Conv1dViaConv3d

# Import side effect: runs register_h3_audio_blockings(); without it every conv silently falls back to C_in_block=32.
from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import (
    assert_weight_norm_axes_consistent,
    convert_minimax_h3_audio_state_dict,
    fuse_attention_biases,
    fuse_weight_norm,
    remap_amp_activations,
)
from ....models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from ....utils.check import assert_quality
from .common import build_audio_decoder, load_config, psnr, weights_subdir

# The vocoder needs extra L1 scratch, as the LTX audio tests do.
SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

AUDIO_RELATIVE_RMSE = 0.12  # measured ~0.105 on the default path (fp32 conv multiplier over ~130 convs)

CONV_PRE_SHAPE = (2048, 1024, 7)  # conv_pre: the decoder's widest reduction, Cin 2048 x k 7
CONV_PRE_LATENT_FRAMES = 207

LATENT_CHANNELS = 32
HOP_LENGTH = 800
SAMPLING_RATE = 32000

PRODUCTION_LATENT_FRAMES = [pytest.param(207, id="5s")]  # 40 Hz latent; 10 s exercises no new path


def _build_reference(load_weights: bool = True):
    """The reference audio VAE, with weight norm still attached (the converter removes it)."""
    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio

    config = load_config(weights_dir)
    reference = AutoencoderKLMiniMaxH3Audio(**config).eval()
    if load_weights:
        from safetensors.torch import load_file

        reference.load_state_dict(load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")))
    return reference, config


def _golden_latents(reference, num_latent_frames: int):
    """``(waveform, latents, expected_decode)``; latents come from the reference encoder, not ``randn``."""
    waveform = torch.randn(2, 1, num_latent_frames * HOP_LENGTH) * 0.1
    with torch.no_grad():
        posterior = reference.encode(waveform).latent_dist
        latents = posterior.mode()[..., :num_latent_frames]
        expected = reference.decode(latents).sample
    return waveform, latents, expected


def _log_mel_distance(a: torch.Tensor, b: torch.Tensor, *, n_fft: int = 1024, hop: int = 256) -> float:
    """Mean absolute log-magnitude-spectrogram difference; catches spectral drift an aggregate PCC hides."""
    window = torch.hann_window(n_fft)
    spectra = []
    for signal in (a, b):
        flat = signal.reshape(-1, signal.shape[-1]).float()
        stft = torch.stft(flat, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        spectra.append(torch.log(stft.abs().clamp_min(1e-5)))
    return (spectra[0] - spectra[1]).abs().mean().item()


def _tt_decoder(config: dict, mesh_device) -> MiniMaxH3AudioDecoder:
    return build_audio_decoder(config, mesh_device)


@pytest.mark.parametrize("num_latent_frames", PRODUCTION_LATENT_FRAMES)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decode(mesh_device, num_latent_frames):
    """The whole decode path against the reference, at a production duration, stereo."""
    reference, config = _build_reference()
    torch.manual_seed(1)

    _, latents, expected = _golden_latents(reference, num_latent_frames)
    with torch.no_grad():
        expected_proj = reference.dec_in_proj(latents)

    tt_decoder = _tt_decoder(config, mesh_device)
    tt_decoder.load_torch_state_dict(convert_minimax_h3_audio_state_dict(dict(reference.state_dict())), strict=False)

    projected = tt_decoder._project_latents_device(latents)
    assert projected.shape == expected_proj.shape, f"proj {tuple(projected.shape)} != {tuple(expected_proj.shape)}"
    assert_quality(expected_proj, projected, pcc=0.999)

    actual = tt_decoder(latents)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.99, relative_rmse=AUDIO_RELATIVE_RMSE)

    psnr_db = psnr(expected, actual)
    mel_distance = _log_mel_distance(expected, actual)
    assert psnr_db >= 28.0, f"decode PSNR {psnr_db:.2f} dB < 28 dB"
    assert mel_distance <= 5.0, f"log-spectrogram distance {mel_distance:.3f} > 5.0"

    left, right = actual[0, 0], actual[1, 0]
    assert not torch.allclose(left, right, atol=1e-4), "stereo channels are identical -- a broadcast bug"


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_decode_accurate_mode(mesh_device, monkeypatch):
    """The whole decode path with every precision lever on (``MINIMAX_H3_AUDIO_ACCURATE=1``)."""
    monkeypatch.setenv("MINIMAX_H3_AUDIO_ACCURATE", "1")

    reference, config = _build_reference()
    torch.manual_seed(1)

    num_latent_frames = 207  # 5 s, as in test_decode
    _, latents, expected = _golden_latents(reference, num_latent_frames)

    tt_decoder = _tt_decoder(config, mesh_device)
    assert tt_decoder.dec_in_proj.split_mode == "full", "split_mode='full' did not land on dec_in_proj"
    assert tt_decoder.dec_in_proj.tap_matmul, "tap_matmul=True did not land on dec_in_proj"
    assert tt_decoder.decoder.conv_post.split_mode == "full", "split_mode='full' did not land on conv_post"
    assert tt_decoder.decoder.act_post.downsample.lowpass.prefer_mac, "prefer_mac=True did not land on act_post"

    tt_decoder.load_torch_state_dict(convert_minimax_h3_audio_state_dict(dict(reference.state_dict())), strict=False)
    actual = tt_decoder(latents)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.9999, relative_rmse=0.006)  # measured 0.0045 / 99.9990% / 67.5 dB
    psnr_db = psnr(expected, actual)
    assert psnr_db >= 60.0, f"accurate-mode PSNR {psnr_db:.2f} dB < 60 dB"


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_conv_operand_split_improves_precision(mesh_device):
    """Operand splitting really does beat the fp32 conv floor, at ``conv_pre``'s production shape."""
    in_channels, out_channels, kernel = CONV_PRE_SHAPE
    torch.manual_seed(0)
    reference = torch.nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2).float().eval()
    x = torch.randn(2, in_channels, CONV_PRE_LATENT_FRAMES) * 0.1
    with torch.no_grad():
        golden = copy.deepcopy(reference).double()(x.double()).float()

    x_device = None
    errors = {}
    for mode in ("off", "weight", "full"):
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

    # 2.1e-03 sits between production's 1.86e-03 and the unregistered C_in_block=32 fallback's 2.40e-03.
    assert errors["off"] <= 2.1e-3, (
        f"baseline conv_pre error {errors['off']:.3e} is above the production floor -- "
        "C_in_block has probably fallen back to 32, meaning register_h3_audio_blockings() did not run"
    )
    # Measured 1.86e-03 / 1.25e-03 / 1.00e-03; ratios carry ~1.3x margin.
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
    """The whole encode path -- DAC trunk, ``pre_block``, posterior heads -- vs the reference."""
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
    """End to end: encode then decode on device, against the reference's own round trip."""
    reference, config = _build_reference()
    torch.manual_seed(3)
    num_latent_frames = 207

    waveform, _, expected = _golden_latents(reference, num_latent_frames)

    tt_encoder = _tt_encoder(config, mesh_device)
    tt_decoder = _tt_decoder(config, mesh_device)
    converted = convert_minimax_h3_audio_state_dict(dict(reference.state_dict()))
    tt_encoder.load_torch_state_dict(converted, strict=False)
    tt_decoder.load_torch_state_dict(converted, strict=False)

    mean, _ = tt_encoder(waveform)
    actual = tt_decoder(mean)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    psnr_db = psnr(expected, actual)
    assert psnr_db >= 28.0, f"round-trip PSNR {psnr_db:.2f} dB < 28 dB"
    assert _log_mel_distance(expected, actual) <= 5.0, "round-trip spectrum drifted"


# -------------------------------------------------------------------- weight-norm fusion and checkpoint conversion (host only)


def test_fuse_weight_norm_matches_torch_conv_transpose1d():
    """The load-bearing case: axis 0 of a ConvTranspose1d weight is ``in_channels``, not out."""
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
    """Real checkpoint: axes hold, every pair fuses, fused weights equal torch's ``remove_weight_norm``."""
    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from safetensors.torch import load_file

    config = load_config(weights_dir)
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

# 450 MB trace region: accurate mode's graph needs 375463936 B; the default path fits in 300 MB.
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
    reference, config = _build_reference()
    converted = convert_minimax_h3_audio_state_dict(dict(reference.state_dict()))

    torch.manual_seed(2)
    decoder = _tt_decoder(config, mesh_device)
    decoder.load_torch_state_dict(converted, strict=False)

    latents = torch.randn(2, config["latent_channels"], NUM_LATENT_FRAMES) * 0.1

    plain = decoder(latents)
    traced = decoder(latents, traced=True)
    assert traced.shape == plain.shape, f"{traced.shape} != {plain.shape}"

    # PSNR inf would also appear if traced=True silently fell through, so confirm a trace was captured.
    tracers = type(decoder.decoder)._forward_device._tracers_keyed.get(decoder.decoder, {})
    assert tracers, "traced=True captured no trace; the PSNR below would be meaningless"
    assert plain.abs().max() > 1e-6, "decoder produced all-zero output; PSNR would be vacuous"

    psnr_db = psnr(plain, traced)
    logger.info(f"traced vs untraced PSNR: {psnr_db:.2f} dB ({len(tracers)} trace(s) captured)")

    untraced_s = _best(lambda: decoder(latents))
    traced_s = _best(lambda: decoder(latents, traced=True))
    logger.info(
        f"PERF audio_decode_5s untraced {untraced_s:.4f} s | traced {traced_s:.4f} s "
        f"-> {untraced_s / traced_s:.2f}x"
    )
    decoder.release_trace()

    assert psnr_db > 60.0, f"traced output diverges from untraced: PSNR {psnr_db:.2f} dB"
