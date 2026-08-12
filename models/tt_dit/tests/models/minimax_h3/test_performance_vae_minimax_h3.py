# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gate M8f: MiniMax-H3 VAE performance baselines, gated against ``EXPECTED_METRICS`` budgets
(style of ``tests/models/wan2_2/test_performance_wan.py``). Tracy capture entry points live
in ``tools/tracy_decode_harness.py``; quality gates live in the vae/audio test files."""

import os
import time

import pytest
import torch

import ttnn

from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from ....models.audio_vae.minimax_h3.encoder_minimax_h3_audio import MiniMaxH3AudioEncoder
from .common import (
    CLIP_FRAMES,
    DECODE_LATENT_FRAMES,
    LATENT_TILE,
    TILE,
    build_audio_decoder,
    build_visual_decoder,
    build_visual_encoder,
    load_config,
    random_decoder_state,
    random_encoder_state,
    weights_subdir,
)

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

HOP_LENGTH = 800

# Work units per working point: 768P = 4x7 tiles, 1440P = 8x13; clip/chunk counts per the 17n+5 -> 5n+2 rule.
WORK_UNITS = {
    "768P_5s": {"tiles": 28, "encode_clips": 8, "decode_chunks": 7},
    "768P_10s": {"tiles": 28, "encode_clips": 15, "decode_chunks": 14},
    "768P_15s": {"tiles": 28, "encode_clips": 22, "decode_chunks": 21},
    "1440P_5s": {"tiles": 104, "encode_clips": 8, "decode_chunks": 7},
    "1440P_10s": {"tiles": 104, "encode_clips": 15, "decode_chunks": 14},
    "1440P_15s": {"tiles": 104, "encode_clips": 22, "decode_chunks": 21},
}

AUDIO_LATENT_FRAMES_5S = 207  # ~5 s at 40 latent fps; budgets were calibrated at 207, not the exact 200

# Seconds per measurement. Generous regression bars, not tuned targets; visual entries gate a 32-unit wave.
EXPECTED_METRICS = {
    "visual_encoder_clip_wave": 20.0,
    "visual_encoder_keyframe_wave": 5.0,
    "visual_decoder_wave": 20.0,
    "audio_encode_5s": 60.0,
    "audio_decode_5s": 60.0,
}


def _time_it(fn, *, warmup: int = 1, iterations: int = 2, mesh_device=None) -> float:
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
    """Per-wave time with one work unit per device, gated, plus full-video projections."""
    from loguru import logger

    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    config = load_config(weights_dir)
    devices = mesh_device.get_num_devices()
    torch.manual_seed(0)

    measurements = {}

    for label, num_frames, taps in (
        ("visual_encoder_clip_wave", CLIP_FRAMES, 3),
        ("visual_encoder_keyframe_wave", 1, 1),
    ):
        encoder = build_visual_encoder(config, mesh_device, num_frames, temporal_taps=taps)
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

    decoder = build_visual_decoder(config, mesh_device)
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
    # 10.6 = TFLOP per invocation at 1797 tokens x 36 layers x dim 2048.
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
    """Audio encode/decode timing baselines at 5 s, against their budgets."""
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
    decoder = build_audio_decoder(config, mesh_device)
    # `strict=False`: the converted dict carries both halves' tensors.
    encoder.load_torch_state_dict(converted, strict=False)
    decoder.load_torch_state_dict(converted, strict=False)

    measurements = {
        "audio_encode_5s": _time_it(lambda: encoder(waveform), iterations=1, mesh_device=mesh_device),
        "audio_decode_5s": _time_it(lambda: decoder(latents), iterations=1, mesh_device=mesh_device),
    }
    _report(measurements)
