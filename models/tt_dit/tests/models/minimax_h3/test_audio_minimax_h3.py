# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gate M8d: the MiniMax-H3 audio VAE (BigVGAN) at production shapes; stereo rides as batch 2.

T-parallel decode works: ``test_audio_decode_t_parallel`` at the bottom of this file is resurrected from
git history, as the previous version of this docstring asked for. The 8-way shard layout it called
known-broken was `Vocoder.conv_pre` returning uninitialized memory when sharded, plus
``_forward_tap_matmul`` padding shard boundaries locally instead of through the halo; both are fixed, and
the test's PSNR assert is enforced for every factor rather than excused. ``test_neighbor_pad_t_minimax_h3.py``
is still in git history if the halo itself ever needs its own gate again -- it was verified correct here
(exact at pad 1/3/25) while hunting the conv_pre bug."""

from __future__ import annotations

import copy
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.audio_ops import Conv1dViaConv3d, depthwise_tap_filter
from ....models.audio_vae.minimax_h3.blockings_minimax_h3_audio import register_h3_audio_blockings
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
from .common import build_audio_decoder, load_config, psnr, weights_subdir

# The vocoder needs extra L1 scratch, as the LTX audio tests do.
SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

# Encode floor, calibrated before accurate mode became the constructed default (measured ~0.105 with
# fp32 conv multipliers over ~130 convs); the accurate defaults only improve on it.
AUDIO_RELATIVE_RMSE = 0.12

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
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")
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
    """The whole decode path against the reference, at a production duration, stereo.

    Constructor defaults are accurate mode (split_mode='full', tap_matmul, prefer_mac), so the bars
    are the accurate-mode ones: measured 0.0045 rel RMSE / 99.9990% PCC / 67.5 dB PSNR.
    """
    reference, config = _build_reference()
    torch.manual_seed(1)

    _, latents, expected = _golden_latents(reference, num_latent_frames)
    with torch.no_grad():
        expected_proj = reference.dec_in_proj(latents)

    tt_decoder = _tt_decoder(config, mesh_device)
    # The precision levers are the constructed defaults; assert they landed where they matter.
    assert tt_decoder.dec_in_proj.split_mode == "full", "split_mode='full' did not land on dec_in_proj"
    assert tt_decoder.dec_in_proj.tap_matmul, "tap_matmul=True did not land on dec_in_proj"
    assert tt_decoder.decoder.conv_post.split_mode == "full", "split_mode='full' did not land on conv_post"
    assert tt_decoder.decoder.act_post.downsample.lowpass.prefer_mac, "prefer_mac=True did not land on act_post"

    tt_decoder.load_torch_state_dict(convert_minimax_h3_audio_state_dict(dict(reference.state_dict())), strict=False)

    projected = tt_decoder._project_latents_device(latents)
    assert projected.shape == expected_proj.shape, f"proj {tuple(projected.shape)} != {tuple(expected_proj.shape)}"
    assert_quality(expected_proj, projected, pcc=0.999)

    actual = tt_decoder(latents)

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}"
    assert_quality(expected, actual, pcc=0.9999, relative_rmse=0.006)

    psnr_db = psnr(expected, actual)
    mel_distance = _log_mel_distance(expected, actual)
    assert psnr_db >= 60.0, f"decode PSNR {psnr_db:.2f} dB < 60 dB"
    assert mel_distance <= 5.0, f"log-spectrogram distance {mel_distance:.3f} > 5.0"

    left, right = actual[0, 0], actual[1, 0]
    assert not torch.allclose(left, right, atol=1e-4), "stereo channels are identical -- a broadcast bug"


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_depthwise_chunked_conv1d_matches_torch(mesh_device):
    """`depthwise_tap_filter`'s C-chunked `ttnn.conv1d` recovery, against a torch depthwise reference.

    Nothing else covers it: H3 defaults to `prefer_mac=True`, which returns before the conv1d fallback,
    so no decode test touches the slicing, the per-chunk weight, or the concat. Shape is the one the
    decode really chunks (T_pad=166, C=512, K=7, stride=1), where the activation block is C*K wide and
    the slicer runs out of L1 at full C.
    """
    torch.manual_seed(0)
    B, T_pad, C, K, stride = 1, 166, 512, 7, 1

    x = torch.randn(B, T_pad, C, dtype=torch.float32)
    taps = torch.randn(K, dtype=torch.float32)
    x_dev = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)

    cache: dict = {}
    out = depthwise_tap_filter(
        x_dev,
        [float(t) for t in taps],
        stride,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
        cache=cache,
        prefer_mac=False,
    )
    actual = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()

    # Valid depthwise conv over T: one tap vector shared by every channel, groups=C.
    weight = taps.view(1, 1, K).expand(C, 1, K).contiguous()
    expected = torch.nn.functional.conv1d(x.transpose(1, 2), weight, stride=stride, groups=C).transpose(1, 2)

    assert actual.shape == expected.shape, f"{actual.shape} != {expected.shape}"
    psnr_db = psnr(expected, actual)
    logger.info(f"chunked depthwise conv1d vs torch: {psnr_db:.2f} dB")
    assert psnr_db > 60.0, f"chunked depthwise conv1d diverges from torch: {psnr_db:.2f} dB"

    # Prove the chunked path ran: it keys its prepared weight on the chunk width where the unchunked
    # path keys on C. Without this the test would pass on plain conv1d or a silent MAC fallback, i.e.
    # assert torch equivalence while covering none of the code it exists for.
    chunk_widths = [k[1] for k in cache if isinstance(k, tuple) and k and k[0] == "w" and k[1] < C]
    assert chunk_widths, (
        f"expected the C-chunked conv1d recovery to run at C={C}, K={K}, T_pad={T_pad}, but the weight "
        f"cache has no sub-C chunk key: {sorted(k for k in cache if isinstance(k, tuple))}. Either "
        "conv1d now fits at full C (then this shape no longer covers the chunked path and the test needs "
        "a new one), or it fell back to MAC (then the recovery is broken)."
    )
    logger.info(f"chunked into widths {sorted(set(chunk_widths))} from C={C}")


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_conv_operand_split_improves_precision(mesh_device):
    """Operand splitting really does beat the fp32 conv floor, at ``conv_pre``'s production shape."""
    # Raw Conv1dViaConv3d does not register the H3 conv blockings (the H3 audio module constructors
    # do); without this, every conv silently falls back to C_in_block=32 and the "off" bar fails.
    register_h3_audio_blockings()
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


def _tt_encoder(config: dict, mesh_device, split_mode: str = "full"):
    from ....models.audio_vae.minimax_h3.encoder_minimax_h3_audio import MiniMaxH3AudioEncoder

    return MiniMaxH3AudioEncoder(
        encoder_dim=config["encoder_dim"],
        encoder_rates=tuple(config["encoder_rates"]),
        latent_dim=config["latent_dim"],
        latent_channels=config["latent_channels"],
        num_attention_heads=config["num_attention_heads"],
        mesh_device=mesh_device,
        split_mode=split_mode,
    )


@pytest.mark.parametrize(
    "split_mode",
    # "full" is the accurate constructor default; "weight" is what the pipeline ships for ref2va
    # (565 vs 796 ms at 5.17 s, mean PCC 99.978% vs 99.999% -- both far inside the bars below).
    [pytest.param("full", id="full"), pytest.param("weight", id="weight_production")],
)
@pytest.mark.parametrize("num_latent_frames", PRODUCTION_LATENT_FRAMES)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_encode(mesh_device, num_latent_frames, split_mode):
    """The whole encode path -- DAC trunk, ``pre_block``, posterior heads -- vs the reference."""
    reference, config = _build_reference()
    torch.manual_seed(2)

    waveform = torch.randn(2, 1, num_latent_frames * HOP_LENGTH) * 0.1
    with torch.no_grad():
        posterior = reference.encode(waveform).latent_dist
        expected_mean, expected_logs = posterior.mean, posterior.logs

    tt_encoder = _tt_encoder(config, mesh_device, split_mode=split_mode)
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
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")
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

# 450 MB trace region: the accurate-mode (default) decode graph needs 375463936 B.
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
# `KNOWN_BROKEN` is deliberately empty; an entry silences the PSNR assert, which is the only thing
# separating a speedup from a fast wrong answer. It formerly held (8, 1) at -6.3 dB, blamed on one tile
# per shard -- the wrong suspect, since every factor was wrong and the cause was `conv_pre` returning
# uninitialized memory when sharded. Both factors now measure 83.4 dB against the unsharded path
# (1.87x at axis 0, 2.20x at axis 1), and `cpu_vs_device.py` scores sharded at the same PSNR as single
# device (81.89 vs 81.99 dB at the constructed defaults): sharding buys latency, not accuracy.
FACTORS = [(1, 1), (4, 0), (8, 1)]
KNOWN_BROKEN: set[tuple[int, int]] = set()


def _build(mesh_device, config, converted, parallel_config, ccl_manager):
    """The decoder at this file's shared defaults, plus a shard layout.

    Goes through `build_audio_decoder` rather than constructing directly so the precision levers
    (`split_mode`, `tap_matmul`, `prefer_mac`, `max_c_in_block`) stay at whatever the shipping default
    is -- a sharded run must measure the same configuration the unsharded gates do, or a divergence
    could be a lever difference rather than a sharding bug.
    """
    decoder = build_audio_decoder(
        config,
        mesh_device,
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


# ~14 min: three decoder builds plus a decode per factor, against `pytest.ini`'s 300 s default.
@pytest.mark.timeout(2400)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
def test_audio_decode_t_parallel(mesh_device):
    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from loguru import logger
    from safetensors.torch import load_file

    from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict

    config = load_config(weights_dir)
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
        except Exception as exc:
            # Every FACTORS entry is a supported layout, so a raise is a regression, not a "result":
            # recording it as unsupported let factor 4 break while factor 8 kept the test green, since
            # the asserts below need only *some* factor to run. Optional layouts go in KNOWN_BROKEN.
            logger.warning(f"t_factor={factor} axis={axis} FAILED: {str(exc)[:160]}")
            results.append((factor, axis, None, None))
            if (factor, axis) not in KNOWN_BROKEN:
                raise
            continue

        if baseline_out is None:
            baseline_out, baseline_s = out, seconds
            psnr_db = float("inf")
        else:
            assert out.shape == baseline_out.shape, f"factor {factor}: {out.shape} != {baseline_out.shape}"
            # `psnr_db`, not `psnr`: binding the float to `psnr` shadows the imported helper, so the
            # first factor's `float("inf")` makes the second factor's call a TypeError.
            psnr_db = psnr(baseline_out, out)
        results.append((factor, axis, seconds, psnr_db))
        logger.info(
            f"PERF audio_decode t_factor={factor} axis={axis}: {seconds:.4f} s "
            f"({baseline_s / seconds:.2f}x) PSNR vs 1-device {psnr_db:.1f} dB"
        )
        if psnr_db < 40.0 and out is not baseline_out:
            _localize_divergence(baseline_out.float(), out.float(), factor=factor, logger=logger)
        del decoder

    logger.info("=== audio decode T-parallel summary ===")
    for factor, axis, seconds, psnr_db in results:
        if seconds is None:
            logger.info(f"  t_factor={factor:2d} axis={axis}: unsupported")
        else:
            logger.info(
                f"  t_factor={factor:2d} axis={axis}: {seconds:.4f} s  {baseline_s / seconds:5.2f}x  "
                f"PSNR {psnr_db:6.1f} dB"
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
    for factor, axis, seconds, psnr_db in results:
        if seconds is None or (factor, axis) in KNOWN_BROKEN:
            continue
        assert psnr_db > 40.0, f"t_factor={factor} axis={axis} diverges from 1-device: PSNR {psnr_db:.1f} dB"
