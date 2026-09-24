# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Speed/accuracy experiments for the H3 audio decoder on a 4x8 Galaxy at T-shard 8: each *recipe* builds the decoder
differently, is timed traced and scored (PSNR, log-mel) against the cached CPU reference decode (``sqz_reference``)."""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from ....parallel.config import ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.test import line_params_8k
from .common import build_audio_decoder, load_config, psnr, weights_subdir
from .sqz_reference import reference_clip

MESH = [
    pytest.param(
        (4, 8),
        {
            **line_params_8k,
            "require_exact_physical_num_devices": True,
            "l1_small_size": 65536,
            "trace_region_size": 1_200_000_000,
        },
        id="mesh4x8_8k",
    )
]

RECIPES = {
    "full": {},
    "weight": {"all": "weight"},
    "off": {"all": "off"},
    # time-packed late bands (layers/audio_pack.py): 2 steps/row at 16 ch, 4 steps/row at 8 ch
    "full_pack": {"pack": {5: 2, 6: 4}},
    # the full split done inside conv3d (Conv3dConfig.operand_split): same operands, one launch per conv
    "kernel": {"all": "kernel"},
    # the fused anti-alias SnakeBeta kernel (layers/audio_aa_snake.py) in place of every resampler/snake chain
    "fused_pack": {"pack": {5: 2, 6: 4}, "build_kwargs": {"act_mode": "fused"}},
    "kernel_fused_pack": {"all": "kernel", "pack": {5: 2, 6: 4}, "build_kwargs": {"act_mode": "fused"}},
    # one stereo channel per mesh row (batch_shard_axis=0); must equal kernel_fused_pack bit for bit
    "kernel_fused_pack_bshard": {
        "all": "kernel",
        "pack": {5: 2, 6: 4},
        "build_kwargs": {"act_mode": "fused", "batch_shard_axis": 0},
    },
}
RUN_ORDER = (
    "full_pack",
    "full",
    "kernel_fused_pack",
    "kernel_fused_pack_bshard",
    "fused_pack",
    "kernel",
    "weight",
    "off",
)
BUILD_KWARGS_KEY = "build_kwargs"  # extra constructor kwargs
PACK_KEY = "pack"


def _split_convs(decoder):
    """Yields every split-capable conv: dec_in_proj, conv_pre, the upsamplers, the AMP block convs and conv_post."""
    voc = decoder.decoder
    yield decoder.dec_in_proj
    yield voc.conv_pre
    for up in voc.ups:
        yield up.conv
    for block in voc.resblocks:
        yield from block.convs1
        yield from block.convs2
    yield voc.conv_post


def apply_recipe(decoder, recipe: dict) -> dict:
    """Set ``split_mode`` per conv after construction (forward reads it per call; the unused residual is harmless)."""
    counts = {}
    if PACK_KEY in recipe:
        counts["pack"] = dict(recipe[PACK_KEY])
    for conv in _split_convs(decoder):
        if "all" in recipe:
            conv.split_mode = recipe["all"]
        counts[conv.split_mode] = counts.get(conv.split_mode, 0) + 1
    return counts


def _load(mesh_device, **overrides):
    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")
    from safetensors.torch import load_file

    config = load_config(weights_dir)
    pc = ParallelFactor(factor=8, mesh_axis=1)
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    decoder = build_audio_decoder(config, mesh_device, parallel_config=pc, ccl_manager=ccl, **overrides)
    decoder.load_torch_state_dict(
        convert_minimax_h3_audio_state_dict(
            load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
        ),
        strict=False,
    )
    return decoder, config


def _log_mel_distance(a: torch.Tensor, b: torch.Tensor, *, n_fft: int = 1024, hop: int = 256) -> float:
    window = torch.hann_window(n_fft)
    spectra = []
    for signal in (a, b):
        flat = signal.reshape(-1, signal.shape[-1]).float()
        stft = torch.stft(flat, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        spectra.append(torch.log(stft.abs().clamp_min(1e-5)))
    return (spectra[0] - spectra[1]).abs().mean().item()


def _best(fn, mesh_device, n=3):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - t0)
    return best, out


# PSNR against the CPU reference and log-mel distance for the recipes whose fidelity is a contract (the packed
# kernel-split forms the pipeline ships measured 67.3 dB / 0.0034 at 600lat_b2); floors sit a little under that.
FIDELITY_FLOORS = {
    "full_pack": (66.0, 0.006),
    "fused_pack": (66.0, 0.006),
    "kernel_fused_pack": (66.0, 0.006),
    "kernel_fused_pack_bshard": (66.0, 0.006),
    "full": (66.0, 0.006),
}


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(
    ("num_latent_frames", "batch"),
    [(600, 1), (207, 2), (600, 2)],  # (600, 2) = the pipeline's shape: 15 s, stereo as two batch items
    ids=["600lat_b1", "207lat_b2", "600lat_b2"],
)
def test_audio_decode_squeeze(mesh_device, num_latent_frames, batch):
    latents, expected = reference_clip(num_latent_frames, batch)
    rows = []
    baseline_out = None
    for name in RUN_ORDER:
        recipe = RECIPES[name]
        build = dict(recipe.get(BUILD_KWARGS_KEY, {}))
        decoder, _ = _load(mesh_device, pack_bands=recipe.get(PACK_KEY), **build)
        counts = apply_recipe(decoder, recipe)
        eager, _ = _best(lambda: decoder(latents), mesh_device, n=1)
        try:
            traced, out = _best(lambda: decoder(latents, traced=True), mesh_device)
        finally:
            decoder.release_trace()
        assert out.shape == expected.shape, f"{name}: shape {tuple(out.shape)} != reference {tuple(expected.shape)}"
        db_ref = psnr(expected, out)
        mel = _log_mel_distance(expected, out)
        if baseline_out is None:
            baseline_out = out
        db_base = psnr(baseline_out, out)
        floors = FIDELITY_FLOORS.get(name)
        if floors is not None:
            db_floor, mel_ceiling = floors
            assert db_ref >= db_floor and mel <= mel_ceiling, (
                f"{name} is a shipping configuration and its fidelity moved: {db_ref:.2f} dB against a "
                f"{db_floor:.1f} dB floor, log-mel {mel:.4f} against a {mel_ceiling:.4f} ceiling"
            )
        rows.append((name, counts, eager, traced, db_ref, mel, db_base))
        logger.info(
            f"SQZ {name}: split {counts} eager {eager:.4f} s traced {traced:.4f} s | vs CPU ref {db_ref:.2f} dB "
            f"mel {mel:.4f} | vs {RUN_ORDER[0]} {db_base:.2f} dB"
        )
        del decoder
    logger.info(f"=== squeeze table: {num_latent_frames} latents x batch {batch}, 4x8 factor 8, traced best of 3 ===")
    logger.info(
        f"{'recipe':>12s} {'eager s':>8s} {'traced s':>9s} {'PSNR ref':>9s} {'mel':>7s} {'PSNR base':>10s}  split counts"
    )
    for name, counts, eager, traced, db_ref, mel, db_base in rows:
        logger.info(f"{name:>12s} {eager:8.4f} {traced:9.4f} {db_ref:9.2f} {mel:7.4f} {db_base:10.2f}  {counts}")
