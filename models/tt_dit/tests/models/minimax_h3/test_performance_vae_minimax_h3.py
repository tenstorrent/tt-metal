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
from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig
from ....parallel.manager import CCLManager
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
# Audio decode runs the accurate-mode defaults (~13 s eager, ~3x the retired fast path); 60 s stays generous.
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
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")
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
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")
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


# ---- whole-stage `decode` timing, per output type -------------------------------------------
#
# `test_visual_data_parallel_throughput` times one decoder wave; this times the stage the pipeline
# bills to "VAE decode" -- tiling, upload, readback, unpatchify, stitch and the temporal assembly --
# and prints `_report_profile`'s breakdown, which is the point: a stage total alone cannot say which
# of device / readback / stitch moved.
#
# RING, not the `fabric_config=None` the throughput test uses: `_decode_clip_device_stitched`
# hardcodes `all_gather(topology=Ring)`, so `yuv420` cannot run on a line fabric. Built with a
# `CCLManager` because `_read_wave_units` picks its reader on `ccl_manager is None` -- without one a
# single-host run silently takes the `ConcatMeshToTensor` path and measures something production
# never runs.
#
#   export MINIMAX_H3_MODEL_PATH=/path/to/MiniMax-H3
#   pytest models/tt_dit/tests/models/minimax_h3/test_performance_vae_minimax_h3.py -k decode_stage -s

MESH_4X8_RING = [
    pytest.param(
        (4, 8),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "require_exact_physical_num_devices": True,
            "l1_small_size": 65536,
        },
        id="4x8ring",
    )
]

# 1344x768 at 16x spatial compression. Latent frame counts follow the 17n -> 5n rule less
# `token_drop`, so 124 frames (5 s) is 37 and 362 (15 s) is 107.
DECODE_STAGE_LATENT_HW = (48, 84)
DECODE_STAGE_FRAMES = {5: 37, 15: 107}
# 7 latent frames is one chunk, the cheapest warm-up that still exercises a whole wave.
DECODE_STAGE_WARM_FRAMES = 7

MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)

# Each mode is one lever, so a row of this table measures that lever alone.
DECODE_STAGE_MODES = {
    # Today's default: bf16 tiles in token space, blended and unpatchified on host.
    "float": {},
    # Same path, uint8 across PCIe. Halves the transfer.
    "uint8": {"pixel_denorm": (MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD), "readback_uint8": True},
    # Stitch, clamp and colour-convert on device; read one planar canvas at 1.5 bytes/pixel.
    "yuv420": {"pixel_denorm": (MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD), "device_stitch": True},
}


def _decode_stage_state(weights_dir: str) -> dict[str, torch.Tensor]:
    """Just the decoder-side tensors; over half the checkpoint is the encoder."""
    import json

    from safetensors.torch import load_file

    index_path = os.path.join(weights_dir, "diffusion_pytorch_model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return {
            k: v
            for k, v in load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")).items()
            if k.startswith(("decoder.", "post_quant_conv."))
        }
    weight_map = json.loads(open(index_path).read())["weight_map"]
    wanted = {k: f for k, f in weight_map.items() if k.startswith(("decoder.", "post_quant_conv."))}
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(wanted.values())):
        loaded = load_file(os.path.join(weights_dir, shard))
        state.update({k: loaded[k] for k in wanted if k in loaded})
    return state


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8_RING, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("seconds", [5, 15], ids=["5s", "15s"])
@pytest.mark.parametrize("mode", sorted(DECODE_STAGE_MODES), ids=sorted(DECODE_STAGE_MODES))
def test_decode_stage(mesh_device, seconds, mode):
    """Measurement driver, not a gate: reports the split, asserts nothing about wall time."""
    from loguru import logger

    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")

    config = MiniMaxH3VaeConfig.from_pretrained(weights_dir)
    ccl_manager = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    vae = MiniMaxH3Vae(config, mesh_device=mesh_device, ccl_manager=ccl_manager, **DECODE_STAGE_MODES[mode])
    output_type = "yuv420" if mode == "yuv420" else "float"

    vae.load_decoder_state(_decode_stage_state(weights_dir))
    # Build the per-shape decoder outside the timed region, as `_prepare_vae(decode_shape=...)` does:
    # its weight upload is seconds and would otherwise land inside the first `decode`.
    vae._decoder_for(config.tokens_chunk_size + config.token_overlap, 16, 16)

    torch.manual_seed(0)
    latent_h, latent_w = DECODE_STAGE_LATENT_HW
    warm = torch.randn(1, config.latent_channels, DECODE_STAGE_WARM_FRAMES, latent_h, latent_w)
    latents = torch.randn(1, config.latent_channels, DECODE_STAGE_FRAMES[seconds], latent_h, latent_w)

    vae.decode(warm, output_type=output_type)

    started = time.time()
    video = vae.decode(latents, output_type=output_type)
    elapsed = time.time() - started

    profile = dict(vae.last_decode_profile)
    logger.info(f"DECODE_STAGE mode={mode} seconds={seconds} total={elapsed:.3f}s video={tuple(video.shape)}")
    logger.info(
        f"DECODE_SPLIT mode={mode} seconds={seconds} "
        + " ".join(
            f"{k}={profile.get(k, 0.0):.3f}"
            for k in (
                "device",
                "readback",
                "stitch",
                "unpatchify",
                "blend",
                "concat",
                "tiling",
                "upload",
                "host_prep",
                "residual",
            )
        )
        + f" waves={int(profile.get('waves', 0))} units={int(profile.get('units', 0))}"
        + f" readback_gb={profile.get('readback_mb', 0.0) / 1000:.2f}"
    )


# ---- whole-stage `encode` timing, per conditioning case --------------------------------------
#
# `test_decode_stage`'s analog for the conditioning encoders: times the stage the pipeline bills
# to "vae_encode" -- host prep, upload, device forward, readback, stitch -- through the exact
# production entry points (`encode_keyframes` for fl2va, `encode_references` for ref2va), and
# prints the per-phase split the encoder counters in `_run_encoder_units` collect. Measurement
# driver, not a gate: asserts row counts, never wall time.
#
# The four cases map onto the tasks being optimized:
#   fl2va_1key         one 1344x768 keyframe   -> 28 tile units, one wave       (taps=1)
#   fl2va_2key         first + last keyframes  -> two sequential encode_clip calls
#   ref2va_video       124-frame 768P video    -> 8 clips x 28 tiles = 224 units, 7 waves (taps=3)
#   ref2va_video_audio the same video carrying a 5.17 s soundtrack through the audio encoder
#
#   export MINIMAX_H3_MODEL_PATH=/path/to/MiniMax-H3
#   pytest models/tt_dit/tests/models/minimax_h3/test_performance_vae_minimax_h3.py -k encode_stage -s

# 16384, not the decode stage's 65536: the taps=3 video encoder (only ref2va reaches it) clashes
# with L1 above it -- same override the ref2va e2e suite carries.
ENCODE_STAGE_MESH = [
    pytest.param(
        (4, 8),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "require_exact_physical_num_devices": True,
            "l1_small_size": 16384,
        },
        id="4x8ring",
    )
]

ENCODE_STAGE_WIDTH, ENCODE_STAGE_HEIGHT = 1344, 768
ENCODE_STAGE_FRAMES = 124  # 5 s at 24 fps; already 17n + 5, so the reference trim keeps all of it
ENCODE_STAGE_AUDIO_SAMPLES = int(ENCODE_STAGE_FRAMES / 24 * 32000)  # 165333, the docstring's 5.1667 s

ENCODE_STAGE_CASES = ("fl2va_1key", "fl2va_2key", "ref2va_video", "ref2va_video_audio")


def _encode_stage_state(weights_dir: str) -> dict[str, torch.Tensor]:
    """Just the encoder-side tensors, mirroring `_decode_stage_state`."""
    import json

    from safetensors.torch import load_file

    prefixes = ("encoder.", "quant_conv.")
    index_path = os.path.join(weights_dir, "diffusion_pytorch_model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return {
            k: v
            for k, v in load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")).items()
            if k.startswith(prefixes)
        }
    weight_map = json.loads(open(index_path).read())["weight_map"]
    wanted = {k: f for k, f in weight_map.items() if k.startswith(prefixes)}
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(wanted.values())):
        loaded = load_file(os.path.join(weights_dir, shard))
        state.update({k: loaded[k] for k in wanted if k in loaded})
    return state


def _encode_stage_audio_encoder(mesh_device) -> "MiniMaxH3AudioEncoder":
    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")
    from safetensors.torch import load_file

    config = load_config(weights_dir)
    converted = convert_minimax_h3_audio_state_dict(
        load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
    )
    encoder = MiniMaxH3AudioEncoder(
        encoder_dim=config["encoder_dim"],
        encoder_rates=tuple(config["encoder_rates"]),
        latent_dim=config["latent_dim"],
        latent_channels=config["latent_channels"],
        num_attention_heads=config["num_attention_heads"],
        mesh_device=mesh_device,
        split_mode="weight",  # the pipeline's production settings; see _prepare_audio_encoder
        stereo_split_axis=0,
    )
    # The encoder's four prefixes, which is what keeps the load strict (the converted dict
    # carries both halves' tensors) -- same filter `_prepare_audio_encoder` applies.
    encoder.load_torch_state_dict(
        {k: v for k, v in converted.items() if k.startswith(("encoder.", "pre_block.", "mean_proj.", "logs_proj."))}
    )
    return encoder


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), ENCODE_STAGE_MESH, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("case", ENCODE_STAGE_CASES, ids=ENCODE_STAGE_CASES)
def test_encode_stage(mesh_device, case):
    """Measurement driver, not a gate: reports the split, asserts nothing about wall time."""
    import numpy as np
    from loguru import logger
    from PIL import Image

    from ....pipelines.minimax_h3.conditioning import encode_keyframes
    from ....pipelines.minimax_h3.packing_ref2va import MiniMaxH3PreparedReference
    from ....pipelines.minimax_h3.references import encode_references
    from .common import create_fractal_image

    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")

    config = MiniMaxH3VaeConfig.from_pretrained(weights_dir)
    ccl_manager = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    # `profile` stays False: its per-forward sync would serialize the encode wave streaming and
    # measure a schedule production never runs. The cost is attribution -- `device` times the
    # enqueue and the wait lands in `readback` -- so flip it on only to chase a split, not a total.
    # `pixel_norm` and bf16 match the pipeline: conv_in carries the normalize, pixels upload
    # as uint8, and the encoders compute bf16 (4.2x the fp32 wave at PCC 99.998%).
    vae = MiniMaxH3Vae(
        config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        pixel_norm=(MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD),
    )
    vae.load_encoder_state(_encode_stage_state(weights_dir))

    ratio = config.spatial_compression_ratio
    latent_height, latent_width = ENCODE_STAGE_HEIGHT // ratio, ENCODE_STAGE_WIDTH // ratio
    rows_per_frame = (latent_height // 2) * (latent_width // 2)
    audio_seconds = {"audio_encode": 0.0}

    if case.startswith("fl2va"):
        # Build the per-shape encoder outside the timed region, as `_prepare_vae(encode_shape=...)`
        # does: its 0.72 GB weight upload is construction cost the stage does not bill.
        vae._encoder_for(1, vae.tile_size, vae.tile_size, 1)
        image = create_fractal_image(ENCODE_STAGE_WIDTH, ENCODE_STAGE_HEIGHT)
        keyframes = [image]
        if case == "fl2va_2key":
            keyframes.append(Image.fromarray(255 - np.asarray(image)))
        # One latent frame per keyframe; two keyframes are two sequential encode_clip calls,
        # exactly `encode_keyframes`' production loop.
        expected_rows = len(keyframes) * rows_per_frame

        def run():
            return encode_keyframes(
                keyframes, vae.encode_clip, config.latents_mean, config.latents_std, raw_pixels=True
            )

    else:
        vae._encoder_for(config.clip_length, vae.tile_size, vae.tile_size, 3)
        audio_config = load_config(weights_subdir("audio_vae")) if weights_subdir("audio_vae") else None
        if audio_config is None:
            pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")

        rng = np.random.default_rng(0)
        reference = MiniMaxH3PreparedReference(kind="video", has_audio=case == "ref2va_video_audio")
        reference.frames = rng.integers(
            0, 256, (ENCODE_STAGE_FRAMES, ENCODE_STAGE_HEIGHT, ENCODE_STAGE_WIDTH, 3), dtype=np.uint8
        )
        encode_audio = None
        if reference.has_audio:
            audio_encoder = _encode_stage_audio_encoder(mesh_device)
            torch.manual_seed(3)
            reference.waveform = torch.randn(2, ENCODE_STAGE_AUDIO_SAMPLES) * 0.1

            def encode_audio(waveform):
                mark = time.perf_counter()
                out = audio_encoder(waveform)[0]
                audio_seconds["audio_encode"] += time.perf_counter() - mark
                return out

        # 124 frames pad to 8 clips of 17; token_drop then leaves the 17n+5 -> 5n+2 count.
        expected_rows = (5 * 8 - config.token_drop) * rows_per_frame

        def run():
            return encode_references(
                [reference],
                encode_clip=vae.encode_clip,
                encode_video=vae.encode,
                encode_audio=encode_audio,
                latents_mean=config.latents_mean,
                latents_std=config.latents_std,
                audio_latents_mean=audio_config["latents_mean"],
                audio_latents_std=audio_config["latents_std"],
                audio_latent_channels=audio_config["latent_channels"],
                raw_pixels=True,
            )

    run()  # warm: compiles every program and fills lazy allocations, off the record
    vae._profile = vae._empty_profile()
    audio_seconds["audio_encode"] = 0.0

    started = time.perf_counter()
    rows = run()
    elapsed = time.perf_counter() - started

    if case.startswith("fl2va"):
        video_rows, audio_rows = rows, None
    else:
        video_rows, audio_rows = rows
    assert video_rows.shape[0] == expected_rows, f"{video_rows.shape[0]} video rows, expected {expected_rows}"
    if case == "ref2va_video_audio":
        assert audio_rows is not None and audio_rows.shape[0] == 2 * -(
            -ENCODE_STAGE_AUDIO_SAMPLES // 800
        ), f"audio rows {None if audio_rows is None else audio_rows.shape[0]}"

    profile = dict(vae._profile)
    split_keys = ("device", "readback", "upload", "host_prep", "unpatchify", "stitch", "tiling")
    accounted = sum(profile.get(k, 0.0) for k in split_keys) + audio_seconds["audio_encode"]
    logger.info(
        f"ENCODE_STAGE case={case} total={elapsed:.3f}s "
        f"video_rows={video_rows.shape[0]} audio_rows={0 if audio_rows is None else audio_rows.shape[0]}"
    )
    logger.info(
        f"ENCODE_SPLIT case={case} "
        + " ".join(f"{k}={profile.get(k, 0.0):.3f}" for k in split_keys)
        + f" audio_encode={audio_seconds['audio_encode']:.3f}"
        + f" residual={max(0.0, elapsed - accounted):.3f}"
        + f" waves={int(profile.get('waves', 0))} units={int(profile.get('units', 0))}"
        + f" upload_gb={profile.get('upload_mb', 0.0) / 1000:.2f}"
        + f" readback_gb={profile.get('readback_mb', 0.0) / 1000:.2f}"
    )
    for name, each in (("device", profile.get("device_each") or []), ("readback", profile.get("readback_each") or [])):
        if each:
            logger.info(
                f"    {name} per wave: min {min(each) * 1000:.0f} / median "
                f"{sorted(each)[len(each) // 2] * 1000:.0f} / max {max(each) * 1000:.0f} ms  "
                f"[{' '.join(f'{v * 1000:.0f}' for v in each)}]"
            )
