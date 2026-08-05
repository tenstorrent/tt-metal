# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gate M8f: MiniMax-H3 VAE roundtrip quality and per-component performance baselines.

Two things live here, and they are separate concerns deliberately:

* **Roundtrip quality.** PCC per component says the port matches the reference; it does not
  say the reconstruction is good. Visual gets a PSNR floor on a real encode->decode, audio
  gets PSNR plus a log-spectrogram distance. These are the numbers that would catch a
  faint vignette or a dull high end that PCC waves through.
* **Performance baselines.** One measured device time per component at the shipping shape,
  recorded against an ``expected_metrics`` dict in the style of
  ``tests/models/wan2_2/test_performance_wan.py``. Every component carries a number so
  optimisation has a before, and so a regression has something to trip over.

Because tiling fixes the work units, a baseline is a **per-invocation** time plus a count:
the encoder always runs ``(17,256,256)`` and the decoder always ``(7,16,16)``, so a full
clip is that time times the tile/chunk count. The counts for the four supported working
points are in the table below, which makes the projected wall time a multiplication rather
than another measurement.

The bars are set generously. They exist to catch a regression or a pathology, not to pin a
tuned number -- nothing here has been through ``bruteforce_conv3d_sweep.py`` yet.
"""

import json
import math
import os
import time

import pytest
import torch

import ttnn

from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from ....models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from ....models.audio_vae.minimax_h3.encoder_minimax_h3_audio import MiniMaxH3AudioEncoder
from ....models.vae.minimax_h3.decoder_minimax_h3 import MiniMaxH3ViTDecoder3d
from ....models.vae.minimax_h3.encoder_minimax_h3 import MiniMaxH3Encoder3d
from ....utils.check import assert_quality

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

TILE = 256
CLIP_FRAMES = 17
LATENT_TILE = 16
DECODE_LATENT_FRAMES = 7
HOP_LENGTH = 800

# Work-unit counts per supported working point, so a full-clip projection is a
# multiplication of the per-invocation baselines below rather than another measurement.
# 768P is 1344x768 -> 4x7 tiles; 1440P is 2560x1440 -> 8x13. 5 s is 124 frames (37 latent),
# 10 s is 243 frames (72 latent), both under the 17n+5 -> 5n+2 rule at 24 fps.
WORK_UNITS = {
    "768P_5s": {"tiles": 28, "encode_clips": 8, "decode_chunks": 7},
    "768P_10s": {"tiles": 28, "encode_clips": 15, "decode_chunks": 14},
    "768P_15s": {"tiles": 28, "encode_clips": 22, "decode_chunks": 21},
    "1440P_5s": {"tiles": 104, "encode_clips": 8, "decode_chunks": 7},
    "1440P_10s": {"tiles": 104, "encode_clips": 15, "decode_chunks": 14},
    "1440P_15s": {"tiles": 104, "encode_clips": 22, "decode_chunks": 21},
}
# 15 s is 362 frames (107 latent) -> n=21 under the 17n+5 -> 5n+2 rule, so 22 encode clips and 21 decode
# chunks. Cross-checked against the reference helpers rather than derived by hand:
# `align_num_frames(round(dur * MINIMAX_H3_FPS))` gives 124 / 243 / 362 frames for 5 / 10 / 15 s, i.e.
# n = 7 / 14 / 21, which reproduces the 5 s and 10 s rows above.

# The audio VAE runs at sampling_rate / hop_length = 32000 / 800 = 40 latent frames per second. The 5 s
# baseline below predates this and uses 207 frames rather than the exact 200, so durations scale that
# constant instead of recomputing it -- keeping the 5 s number comparable to the recorded baseline.
AUDIO_LATENT_FRAMES_5S = 207
AUDIO_DURATIONS_S = (5.0, 10.0, 15.0)

# Seconds per invocation. Generous: a regression bar, not a tuned target.
EXPECTED_METRICS = {
    "visual_encoder_clip_tile": 20.0,  # (1,3,17,256,256) -> (1,48,5,16,16)
    "visual_encoder_keyframe_tile": 5.0,  # (1,3,1,256,256)
    "visual_decoder_invocation": 20.0,  # (1,24,7,16,16), 1797 tokens, 36 layers
    "audio_encode_5s": 60.0,  # 207 latent frames, stereo as batch 2
    "audio_decode_5s": 60.0,
}


def _weights_dir(subfolder: str) -> str | None:
    base = os.environ.get("MINIMAX_H3_DIFFUSERS_DIR", "/data/cglagovich/MiniMax-H3-diffusers")
    candidate = os.path.join(base, subfolder)
    return candidate if os.path.isfile(os.path.join(candidate, "config.json")) else None


def _config(weights_dir: str) -> dict:
    return {
        k: v
        for k, v in json.loads(open(os.path.join(weights_dir, "config.json")).read()).items()
        if not k.startswith("_")
    }


def _reference_class(name: str):
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as ref

    return getattr(ref, name)


def _random_encoder_state(config: dict) -> dict:
    """State dict from a randomly-initialised reference encoder -- fast, and enough for timing."""
    cls = _reference_class("MiniMaxH3VideoEncoder3d")
    module = cls(
        in_channels=3,
        out_channels=2 * config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        layers_per_block=config["layers_per_block"],
        spatial_downsample_factors=tuple(config["spatial_downsample_factors"]),
        temporal_downsample_factors=tuple(config["temporal_downsample_factors"]),
        norm_num_groups=config["norm_num_groups"],
        norm_eps=config["norm_eps"],
        spatial_padding_mode=config["spatial_padding_mode"],
    )
    return dict(module.state_dict())


def _random_decoder_state(config: dict, *, num_layers: int | None = None) -> dict:
    """Likewise for the 36-layer decoder: 2.4 B random parameters beat a 10.4 GB read.

    ``num_layers`` overrides the config depth, for gates that only need the ops exercised
    rather than the full 2.4 B parameters materialised.
    """
    cls = _reference_class("MiniMaxH3VideoViTDecoder3d")
    module = cls(
        in_channels=config["latent_channels"],
        out_channels=config["out_channels"],
        patch_size=16,
        patch_size_t=4,
        num_layers=config["decoder_num_layers"] if num_layers is None else num_layers,
        num_attention_heads=config["decoder_num_attention_heads"],
        attention_head_dim=config["decoder_attention_head_dim"],
        num_register_tokens=config["decoder_num_register_tokens"],
        ffn_mult=config["decoder_ffn_mult"],
        rope_theta=config["decoder_rope_theta"],
        rope_dim_ratio=config["decoder_rope_dim_ratio"],
        norm_eps=config["decoder_norm_eps"],
    )
    return dict(module.state_dict())


def _psnr(reference: torch.Tensor, test: torch.Tensor) -> float:
    mse = torch.mean((reference.float() - test.float()) ** 2).item()
    if mse == 0.0:
        return float("inf")
    peak = reference.abs().max().item()
    return float("inf") if peak == 0.0 else 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def _time_it(fn, *, warmup: int = 1, iterations: int = 2, mesh_device=None) -> float:
    """Median-free minimum-of-N wall time, after an untimed warmup that absorbs JIT compile."""
    for _ in range(warmup):
        fn()
        if mesh_device is not None:
            ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(iterations):
        start = time.time()
        fn()
        if mesh_device is not None:
            ttnn.synchronize_device(mesh_device)
        best = min(best, time.time() - start)
    return best


def _report(measurements: dict[str, float]) -> None:
    """Collect-then-assert-once, as the wan2_2 perf test does."""
    from loguru import logger

    failures = []
    for key, seconds in measurements.items():
        budget = EXPECTED_METRICS[key]
        logger.info(f"PERF {key}: {seconds:.3f} s (budget {budget:.1f} s)")
        if seconds > budget:
            failures.append(f"{key} took {seconds:.3f} s, budget {budget:.1f} s")
    assert not failures, "\n".join(failures)


def _projected(per_invocation: float, units: int) -> float:
    return per_invocation * units


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_visual_encoder_baseline(mesh_device):
    """Per-invocation time for both encoder shapes, plus the full-clip projections."""
    from loguru import logger

    weights_dir = _weights_dir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = _config(weights_dir)
    torch.manual_seed(0)

    measurements = {}
    for label, num_frames, taps in (
        ("visual_encoder_clip_tile", CLIP_FRAMES, 3),
        ("visual_encoder_keyframe_tile", 1, 1),
    ):
        encoder = MiniMaxH3Encoder3d(
            num_frames=num_frames,
            height=TILE,
            width=TILE,
            in_channels=3,
            out_channels=2 * config["latent_channels"],
            block_out_channels=tuple(config["block_out_channels"]),
            layers_per_block=config["layers_per_block"],
            spatial_downsample_factors=tuple(config["spatial_downsample_factors"]),
            temporal_downsample_factors=tuple(config["temporal_downsample_factors"]),
            temporal_taps=taps,
            mesh_device=mesh_device,
        )
        # Random-init weights: timing does not depend on their values, and skipping the
        # 10.4 GB checkpoint read is what keeps this baseline quick enough to iterate on.
        encoder.load_torch_state_dict(_random_encoder_state(config))
        x = torch.randn(1, num_frames, TILE, TILE, encoder.conv_in.in_channels)
        x_device = ttnn.from_torch(x, dtype=ttnn.float32, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        measurements[label] = _time_it(lambda: encoder(x_device), mesh_device=mesh_device)

    per_tile = measurements["visual_encoder_clip_tile"]
    for name, units in WORK_UNITS.items():
        total = _projected(per_tile, units["tiles"] * units["encode_clips"])
        logger.info(f"PROJECTED encode {name}: {units['tiles'] * units['encode_clips']} invocations, {total:.1f} s")
    _report(measurements)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_visual_decoder_baseline(mesh_device):
    """Per-invocation time for the 36-layer decoder at the shipping latent tile."""
    from loguru import logger

    weights_dir = _weights_dir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = _config(weights_dir)
    torch.manual_seed(1)

    decoder = MiniMaxH3ViTDecoder3d(
        num_frames=DECODE_LATENT_FRAMES,
        height=LATENT_TILE,
        width=LATENT_TILE,
        in_channels=config["latent_channels"],
        out_channels=config["out_channels"],
        patch_size=16,
        patch_size_t=4,
        num_layers=config["decoder_num_layers"],
        num_heads=config["decoder_num_attention_heads"],
        head_dim=config["decoder_attention_head_dim"],
        num_register_tokens=config["decoder_num_register_tokens"],
        rope_theta=config["decoder_rope_theta"],
        rope_dim_ratio=config["decoder_rope_dim_ratio"],
        eps=config["decoder_norm_eps"],
        mesh_device=mesh_device,
    )
    decoder.load_torch_state_dict(_random_decoder_state(config))
    num_patches = DECODE_LATENT_FRAMES * LATENT_TILE * LATENT_TILE
    tokens = torch.randn(1, num_patches, config["latent_channels"])
    tokens_device = ttnn.from_torch(tokens, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)

    seconds = _time_it(lambda: decoder(tokens_device), mesh_device=mesh_device)
    # ~10.6 TFLOP per invocation at 1797 tokens x 36 layers x dim 2048.
    logger.info(f"PERF visual_decoder_invocation: {seconds:.3f} s -> {10.6 / seconds:.1f} TFLOP/s effective")
    for name, units in WORK_UNITS.items():
        count = units["tiles"] * units["decode_chunks"]
        logger.info(f"PROJECTED decode {name}: {count} invocations, {_projected(seconds, count):.1f} s")
    _report({"visual_decoder_invocation": seconds})


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_audio_baselines_and_roundtrip(mesh_device):
    """Audio encode/decode baselines at 5 s, plus a roundtrip quality gate.

    The roundtrip is compared against the **reference's own** encode->decode rather than
    against the input waveform: a VAE round trip is lossy by construction, so the reference
    is the contract.
    """
    weights_dir = _weights_dir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from safetensors.torch import load_file

    config = _config(weights_dir)
    reference = AutoencoderKLMiniMaxH3Audio(**config).eval()
    reference.load_state_dict(load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")))
    converted = convert_minimax_h3_audio_state_dict(dict(reference.state_dict()))

    torch.manual_seed(2)
    num_latent_frames = 207
    waveform = torch.randn(2, 1, num_latent_frames * HOP_LENGTH) * 0.1

    encoder = MiniMaxH3AudioEncoder(
        encoder_dim=config["encoder_dim"],
        encoder_rates=tuple(config["encoder_rates"]),
        latent_dim=config["latent_dim"],
        latent_channels=config["latent_channels"],
        num_attention_heads=config["num_attention_heads"],
        mesh_device=mesh_device,
    )
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
    encoder.load_torch_state_dict(converted, strict=False)
    decoder.load_torch_state_dict(converted, strict=False)

    with torch.no_grad():
        reference_latents = reference.encode(waveform).latent_dist.mode()
        reference_output = reference.decode(reference_latents).sample

    measurements = {
        "audio_encode_5s": _time_it(lambda: encoder(waveform), iterations=1, mesh_device=mesh_device),
        "audio_decode_5s": _time_it(lambda: decoder(reference_latents), iterations=1, mesh_device=mesh_device),
    }

    mean, _ = encoder(waveform)
    roundtrip = decoder(mean)
    assert roundtrip.shape == reference_output.shape
    psnr = _psnr(reference_output, roundtrip)
    from loguru import logger

    logger.info(f"ROUNDTRIP audio PSNR: {psnr:.2f} dB")
    assert psnr >= 20.0, f"audio roundtrip PSNR {psnr:.2f} dB < 20 dB"
    _report(measurements)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_audio_decoder_durations(mesh_device):
    """End-to-end audio decode time at 5 / 10 / 15 s.

    Separate from ``test_audio_baselines_and_roundtrip`` on purpose: that gate pays for a CPU
    reference encode->decode to check quality, which is not worth repeating per duration. The
    decoder takes ``(B, latent_channels, T)``, so latents can be synthesised directly at each
    length -- no encode needed, and timing does not depend on the values.

    Unlike the *visual* decoder there is no tiling here: the audio decoder consumes the whole
    clip in one call, so these are measured end-to-end times rather than a per-unit time to be
    multiplied. Stereo is carried as batch 2, matching the roundtrip gate.
    """
    from loguru import logger

    weights_dir = _weights_dir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = _config(weights_dir)
    torch.manual_seed(3)

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
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from safetensors.torch import load_file

    reference = AutoencoderKLMiniMaxH3Audio(**config).eval()
    reference.load_state_dict(load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")))
    decoder.load_torch_state_dict(convert_minimax_h3_audio_state_dict(dict(reference.state_dict())), strict=False)

    for duration in AUDIO_DURATIONS_S:
        frames = int(round(AUDIO_LATENT_FRAMES_5S * duration / 5.0))
        latents = torch.randn(2, config["latent_channels"], frames) * 0.1
        seconds = _time_it(lambda: decoder(latents), iterations=1, mesh_device=mesh_device)
        samples = frames * HOP_LENGTH
        logger.info(
            f"PERF audio_decode_{duration:.0f}s: {seconds:.3f} s  "
            f"({frames} latent frames -> {samples} samples, {samples / config['sampling_rate']:.2f} s audio, "
            f"{samples / config['sampling_rate'] / seconds:.2f}x realtime)"
        )


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_visual_roundtrip_quality(mesh_device):
    """Visual encode -> decode against the reference's own round trip, with a PSNR floor.

    Deliberately small (one 256x256 tile, 39 frames) and with a shallow reference decoder:
    the 36-layer numerics are gated per-tile elsewhere, and running 36 layers over multiple
    chunks on host would cost tens of TFLOP to prove something already proven. What this
    adds is that encode and decode compose -- that the latent one produces is the latent the
    other consumes, including the chunk geometry between them.
    """
    from loguru import logger

    from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig

    weights_dir = _weights_dir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3

    raw = _config(weights_dir)
    raw["decoder_num_layers"] = 2
    reference = AutoencoderKLMiniMaxH3(**raw).eval()
    with torch.no_grad():
        for block in reference.decoder.transformer_blocks:
            block.scale1.normal_(0, 0.1)
            block.scale2.normal_(0, 0.1)

    torch.manual_seed(3)
    x = torch.randn(1, 3, 39, TILE, TILE) * 0.5
    with torch.no_grad():
        # _encode emits 2 * latent_channels moments; decode consumes the mean.
        reference_latents = reference._encode(x).chunk(2, dim=1)[0]
        expected = reference.decode(reference_latents).sample

    config = MiniMaxH3VaeConfig(**raw)
    vae = MiniMaxH3Vae(config, mesh_device=mesh_device)
    state = dict(reference.state_dict())
    vae.load_encoder_state(state)
    vae.load_decoder_state(state)

    actual = vae.decode(vae.encode(x).chunk(2, dim=1)[0])

    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    psnr = _psnr(expected, actual)
    logger.info(f"ROUNDTRIP visual PSNR: {psnr:.2f} dB")
    assert_quality(expected, actual, pcc=0.99)
    assert psnr >= 25.0, f"visual roundtrip PSNR {psnr:.2f} dB < 25 dB"


MESH_4X8 = [
    pytest.param(
        (4, 8),
        {"fabric_config": None, "require_exact_physical_num_devices": True, "l1_small_size": 65536},
        id="mesh4x8",
    )
]


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_visual_data_parallel_throughput(mesh_device):
    """Per-wave time with one work unit per device, and the resulting full-video projections.

    The single-device baselines above are per *invocation*; this measures a whole mesh-sized
    **wave**, which is what the encode/decode paths now issue. Correctness of the
    decomposition is gated in ``test_vae_data_parallel_minimax_h3.py`` (bit-exact); this only
    times it.
    """
    from loguru import logger

    weights_dir = _weights_dir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = _config(weights_dir)
    devices = mesh_device.get_num_devices()
    torch.manual_seed(0)

    encoder = MiniMaxH3Encoder3d(
        num_frames=CLIP_FRAMES,
        height=TILE,
        width=TILE,
        in_channels=3,
        out_channels=2 * config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        layers_per_block=config["layers_per_block"],
        spatial_downsample_factors=tuple(config["spatial_downsample_factors"]),
        temporal_downsample_factors=tuple(config["temporal_downsample_factors"]),
        temporal_taps=3,
        mesh_device=mesh_device,
    )
    encoder.load_torch_state_dict(_random_encoder_state(config))
    x = torch.randn(devices, CLIP_FRAMES, TILE, TILE, encoder.conv_in.in_channels)
    x_device = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    encoder_wave = _time_it(lambda: encoder(x_device), iterations=8, mesh_device=mesh_device)
    logger.info(f"PERF encoder wave of {devices} units: {encoder_wave:.3f} s ({encoder_wave / devices:.4f} s/unit)")

    decoder = MiniMaxH3ViTDecoder3d(
        num_frames=DECODE_LATENT_FRAMES,
        height=LATENT_TILE,
        width=LATENT_TILE,
        in_channels=config["latent_channels"],
        out_channels=config["out_channels"],
        num_layers=config["decoder_num_layers"],
        mesh_device=mesh_device,
    )
    decoder.load_torch_state_dict(_random_decoder_state(config))
    tokens = torch.randn(devices, DECODE_LATENT_FRAMES * LATENT_TILE * LATENT_TILE, config["latent_channels"])
    tokens_device = ttnn.from_torch(
        tokens,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    decoder_wave = _time_it(lambda: decoder(tokens_device), iterations=8, mesh_device=mesh_device)
    logger.info(f"PERF decoder wave of {devices} units: {decoder_wave:.3f} s ({decoder_wave / devices:.4f} s/unit)")

    for name, units in WORK_UNITS.items():
        encode_units = units["tiles"] * units["encode_clips"]
        decode_units = units["tiles"] * units["decode_chunks"]
        encode_waves = -(-encode_units // devices)
        decode_waves = -(-decode_units // devices)
        encode_total = encode_waves * encoder_wave
        decode_total = decode_waves * decoder_wave
        logger.info(
            f"PROJECTED {name}: encode {encode_units} units / {encode_waves} waves = {encode_total:.1f} s, "
            f"decode {decode_units} units / {decode_waves} waves = {decode_total:.1f} s, "
            f"total {encode_total + decode_total:.1f} s"
        )


_HW_FACTORS = [
    pytest.param(1, 1, id="dp_only"),
    pytest.param(4, 1, id="h4"),
    pytest.param(1, 8, id="w8"),
    pytest.param(4, 8, id="h4w8"),
]

_HW_FABRIC = [
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


@pytest.mark.parametrize(("mesh_device", "device_params"), _HW_FABRIC, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(("h_factor", "w_factor"), _HW_FACTORS)
def test_visual_encoder_hw_vs_dp(mesh_device, h_factor, w_factor):
    """Latency of one encoder unit under H/W sharding, against the data-parallel per-unit cost.

    Two different quantities, deliberately reported side by side:

    * ``dp_only`` times a **whole 32-unit wave**, so its per-unit figure is throughput.
    * the sharded cases time **one unit** spread over ``h_factor * w_factor`` devices, so
      their figure is latency.

    H/W cannot win on throughput -- data-parallelism already runs at 95-97 % scaling
    efficiency with no communication, while sharding adds a full-activation all-gather and
    re-partition at every GroupNorm site plus a halo per conv, and its tiles shrink to 2 rows
    at the deepest, widest-channel blocks. It is measured here for the case where a single
    clip's latency matters more than aggregate throughput.
    """
    from loguru import logger

    from ....models.vae.minimax_h3.encoder_minimax_h3 import MiniMaxH3Encoder3d
    from ....parallel.config import ParallelFactor, VaeHWParallelConfig
    from ....parallel.manager import CCLManager

    weights_dir = _weights_dir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = _config(weights_dir)
    devices = mesh_device.get_num_devices()
    torch.manual_seed(0)

    sharded = h_factor > 1 or w_factor > 1
    ccl = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear) if sharded else None
    parallel_config = (
        VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=h_factor, mesh_axis=0),
            width_parallel=ParallelFactor(factor=w_factor, mesh_axis=1),
        )
        if sharded
        else None
    )

    encoder = MiniMaxH3Encoder3d(
        num_frames=CLIP_FRAMES,
        height=TILE,
        width=TILE,
        in_channels=3,
        out_channels=2 * config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        layers_per_block=config["layers_per_block"],
        spatial_downsample_factors=tuple(config["spatial_downsample_factors"]),
        temporal_downsample_factors=tuple(config["temporal_downsample_factors"]),
        temporal_taps=3,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl,
    )
    encoder.load_torch_state_dict(_random_encoder_state(config))
    in_channels = encoder.conv_in.in_channels

    if sharded:
        x = torch.randn(1, CLIP_FRAMES, TILE, TILE, in_channels)
        dims = [None, None]
        if h_factor > 1:
            dims[0] = 2
        if w_factor > 1:
            dims[1] = 3
        x_device = ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
        )
        units_in_flight = devices // (h_factor * w_factor)
    else:
        x = torch.randn(devices, CLIP_FRAMES, TILE, TILE, in_channels)
        x_device = ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        units_in_flight = devices

    seconds = _time_it(lambda: encoder(x_device), mesh_device=mesh_device)
    logger.info(
        f"PERF h{h_factor}w{w_factor}: {seconds:.3f} s per launch, {units_in_flight} unit(s) in flight, "
        f"{seconds / units_in_flight:.4f} s/unit throughput, {seconds:.3f} s/unit latency"
    )
