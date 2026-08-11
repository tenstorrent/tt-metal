# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gate M8f: MiniMax-H3 VAE performance baselines.

Every measurement is recorded against an ``EXPECTED_METRICS`` budget in the style of
``tests/models/wan2_2/test_performance_wan.py``, so optimisation has a before and a
regression has something to trip over. The bars are set generously: they exist to catch a
regression or a pathology, not to pin a tuned number -- nothing here has been through
``bruteforce_conv3d_sweep.py`` yet.

The visual shipping unit is a data-parallel **wave** -- one work unit per device -- which
is what the encode/decode paths issue, so the visual budgets gate the wave time directly.
H/W sharding of a single unit was measured and rejected for throughput: data parallelism
runs at 95-97 % scaling efficiency with no communication, while sharding adds a
full-activation all-gather and re-partition at every GroupNorm site plus a halo per conv,
and its tiles shrink to 2 rows at the deepest, widest-channel blocks -- it only pays when
one clip's latency matters more than aggregate throughput.

Because tiling fixes the work units, a baseline is a **per-wave** time plus a count: the
encoder always runs ``(17,256,256)`` tiles and the decoder always ``(7,16,16)`` chunks, so
a full clip is that time times the wave count. The counts for the supported working points
are in ``WORK_UNITS`` below, which makes the projected wall time a multiplication rather
than another measurement.

Quality gates live elsewhere: the visual encode -> decode roundtrip (PCC + PSNR floor) is
``test_vae_minimax_h3.py::test_visual_roundtrip_quality``, and the audio roundtrip is
``test_audio_minimax_h3.py::test_roundtrip``.

Tracy capture entry points for the visual/audio decoders live in ``tools/tracy_decode_harness.py``.
"""

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
from .common import (
    CLIP_FRAMES,
    DECODE_LATENT_FRAMES,
    LATENT_TILE,
    TILE,
    load_config,
    random_decoder_state,
    random_encoder_state,
    weights_subdir,
)

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

HOP_LENGTH = 800

# Work-unit counts per supported working point, so a full-clip projection is a
# multiplication of the per-wave baselines below rather than another measurement.
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
# baseline below predates this and uses 207 frames rather than the exact 200 -- keeping the number
# comparable to the recorded baseline matters more than the rounding.
AUDIO_LATENT_FRAMES_5S = 207

# Seconds per measurement. Generous: a regression bar, not a tuned target. The visual
# entries gate a whole 32-unit data-parallel wave; at the measured 95-97 % scaling a wave
# costs about one unit's single-device time, so these carry over the per-invocation
# budgets from the old single-device `test_visual_*_baseline` tests unchanged.
EXPECTED_METRICS = {
    "visual_encoder_clip_wave": 20.0,  # 32 x (1,3,17,256,256) -> (1,48,5,16,16)
    "visual_encoder_keyframe_wave": 5.0,  # 32 x (1,3,1,256,256)
    "visual_decoder_wave": 20.0,  # 32 x (1,24,7,16,16), 1797 tokens, 36 layers
    "audio_encode_5s": 60.0,  # 207 latent frames, stereo as batch 2
    "audio_decode_5s": 60.0,
}


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


MESH_4X8 = [
    pytest.param(
        (4, 8),
        {"fabric_config": None, "require_exact_physical_num_devices": True, "l1_small_size": 65536},
        id="mesh4x8",
    )
]


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_visual_data_parallel_throughput(mesh_device):
    """Per-wave time with one work unit per device, gated, plus full-video projections.

    The wave is the shipping unit, so the old single-device ``test_visual_encoder_baseline``
    and ``test_visual_decoder_baseline`` were folded in here and their ``EXPECTED_METRICS``
    budgets moved onto the waves -- this test fails when a wave blows its budget, not just
    logs. Correctness of the data-parallel decomposition is gated in
    ``test_vae_parallel_minimax_h3.py``; this only times it.
    """
    from loguru import logger

    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = load_config(weights_dir)
    devices = mesh_device.get_num_devices()
    torch.manual_seed(0)

    measurements = {}

    # Both encoder shapes: the 17-frame clip tile and the single-frame keyframe tile.
    for label, num_frames, taps in (
        ("visual_encoder_clip_wave", CLIP_FRAMES, 3),
        ("visual_encoder_keyframe_wave", 1, 1),
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
        encoder.load_torch_state_dict(random_encoder_state(config))
        x = torch.randn(devices, num_frames, TILE, TILE, encoder.conv_in.in_channels)
        x_device = ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        seconds = _time_it(lambda: encoder(x_device), iterations=8, mesh_device=mesh_device)
        logger.info(f"PERF {label} of {devices} units: {seconds:.3f} s ({seconds / devices:.4f} s/unit)")
        measurements[label] = seconds

    decoder = MiniMaxH3ViTDecoder3d(
        num_frames=DECODE_LATENT_FRAMES,
        height=LATENT_TILE,
        width=LATENT_TILE,
        in_channels=config["latent_channels"],
        out_channels=config["out_channels"],
        num_layers=config["decoder_num_layers"],
        mesh_device=mesh_device,
    )
    decoder.load_torch_state_dict(random_decoder_state(config))
    tokens = torch.randn(devices, DECODE_LATENT_FRAMES * LATENT_TILE * LATENT_TILE, config["latent_channels"])
    tokens_device = ttnn.from_torch(
        tokens,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    decoder_wave = _time_it(lambda: decoder(tokens_device), iterations=8, mesh_device=mesh_device)
    # ~10.6 TFLOP per invocation at 1797 tokens x 36 layers x dim 2048.
    logger.info(
        f"PERF visual_decoder_wave of {devices} units: {decoder_wave:.3f} s "
        f"({decoder_wave / devices:.4f} s/unit, {10.6 / (decoder_wave / devices):.1f} TFLOP/s effective per unit)"
    )
    measurements["visual_decoder_wave"] = decoder_wave

    encoder_wave = measurements["visual_encoder_clip_wave"]
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

    _report(measurements)


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_audio_baselines(mesh_device):
    """Audio encode/decode timing baselines at 5 s, against their budgets.

    Timing only. The roundtrip-quality half this test used to carry was dropped as a
    strictly weaker duplicate of ``test_audio_minimax_h3.py::test_roundtrip``, which gates
    the same encode -> decode against the reference with PSNR *and* a log-spectrogram
    distance. Real weights are still loaded -- the run doubles as a check that the
    state-dict conversion wires up -- but the timing inputs are synthesised, since timing
    does not depend on the values.
    """
    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    from safetensors.torch import load_file

    config = load_config(weights_dir)
    converted = convert_minimax_h3_audio_state_dict(
        load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
    )

    torch.manual_seed(2)
    waveform = torch.randn(2, 1, AUDIO_LATENT_FRAMES_5S * HOP_LENGTH) * 0.1
    latents = torch.randn(2, config["latent_channels"], AUDIO_LATENT_FRAMES_5S) * 0.1

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
    # `strict=False`: the converted dict carries both halves' tensors.
    encoder.load_torch_state_dict(converted, strict=False)
    decoder.load_torch_state_dict(converted, strict=False)

    measurements = {
        "audio_encode_5s": _time_it(lambda: encoder(waveform), iterations=1, mesh_device=mesh_device),
        "audio_decode_5s": _time_it(lambda: decoder(latents), iterations=1, mesh_device=mesh_device),
    }
    _report(measurements)
