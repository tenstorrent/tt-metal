# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Speed/accuracy experiments for the H3 audio decoder on a 4x8 Galaxy at T-shard factor 8.

Each *recipe* is a way of building the shipping decoder differently; every recipe is timed traced (best of 3
after a warm eager call) and scored against the pinned CPU reference decode of the same latents (PSNR, log-mel
distance), so speed and accuracy move together in one table. The CPU reference is computed once per clip and
cached under ``SQZ_REF_DIR`` by ``sqz_reference.py`` (needs the pinned diffusers reference importable).

    SQZ_RECIPES=full,weight,off pytest models/tt_dit/tests/models/minimax_h3/test_audio_decode_squeeze.py -s -k 600lat_b1
"""

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
HOP_LENGTH = 800

# Per-band conv split modes. Band i = ups[i] + its three AMP blocks; "pre" = dec_in_proj + conv_pre, "post" = conv_post.
# A recipe maps a band selector to a split mode; unspecified convs keep the constructed default ("full").
RECIPES = {
    "full": {},
    "weight": {"all": "weight"},
    "off": {"all": "off"},
    "off_ge3": {"bands_ge": (3, "off"), "post": "off"},
    "off_ge5": {"bands_ge": (5, "off"), "post": "off"},
    "weight_ge3": {"bands_ge": (3, "weight"), "post": "weight"},
    "act": {"all": "act"},
    "act_off_ge3": {"all": "act", "bands_ge": (3, "off"), "post": "off"},
    "act_off_ge5": {"all": "act", "bands_ge": (5, "off"), "post": "off"},
    # time-packed late bands (layers/audio_pack.py): 2 steps/row at 16 ch, 4 steps/row at 8 ch
    "full_pack": {"pack": {5: 2, 6: 4}},
    "act_pack": {"all": "act", "pack": {5: 2, 6: 4}},
    "off_pack": {"all": "off", "pack": {5: 2, 6: 4}},
    "act_off_ge3_pack": {"all": "act", "bands_ge": (3, "off"), "post": "off", "pack": {5: 2, 6: 4}},
}
PACK_KEY = "pack"


def _conv_modules_by_band(decoder):
    """Yields ``(band, conv)`` for every split-capable conv; band is an int, "pre" or "post"."""
    voc = decoder.decoder
    yield "pre", decoder.dec_in_proj
    yield "pre", voc.conv_pre
    for i, up in enumerate(voc.ups):
        yield i, up.conv
    nk = voc.num_kernels
    for idx, block in enumerate(voc.resblocks):
        band = idx // nk
        for conv in list(block.convs1) + list(block.convs2):
            yield band, conv
    yield "post", voc.conv_post


def apply_recipe(decoder, recipe: dict) -> dict:
    """Set ``split_mode`` per conv after construction (forward reads it per call; the unused residual is harmless)."""
    counts = {}
    if PACK_KEY in recipe:
        counts["pack"] = dict(recipe[PACK_KEY])
    for band, conv in _conv_modules_by_band(decoder):
        mode = None
        if "all" in recipe:
            mode = recipe["all"]
        if isinstance(band, int) and "bands_ge" in recipe and band >= recipe["bands_ge"][0]:
            mode = recipe["bands_ge"][1]
        if band in recipe:
            mode = recipe[band]
        if mode is not None:
            conv.split_mode = mode
        counts[conv.split_mode] = counts.get(conv.split_mode, 0) + 1
    return counts


def _load(mesh_device, *, factor: int = 8, axis: int = 1, **overrides):
    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")
    from safetensors.torch import load_file

    config = load_config(weights_dir)
    pc = None if factor <= 1 else ParallelFactor(factor=factor, mesh_axis=axis)
    ccl = None if pc is None else CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
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


def _selected_recipes():
    names = os.environ.get("SQZ_RECIPES", "full,weight,off,off_ge3,off_ge5").split(",")
    return [(n, RECIPES[n]) for n in names if n]


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(("num_latent_frames", "batch"), [(600, 1), (207, 2)], ids=["600lat_b1", "207lat_b2"])
def test_audio_decode_squeeze(mesh_device, num_latent_frames, batch):
    latents, expected = reference_clip(num_latent_frames, batch)
    rows = []
    baseline_out = None
    for name, recipe in _selected_recipes():
        decoder, _ = _load(mesh_device, pack_bands=recipe.get(PACK_KEY))
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
        rows.append((name, counts, eager, traced, db_ref, mel, db_base))
        logger.info(
            f"SQZ {name}: split {counts} eager {eager:.4f} s traced {traced:.4f} s | vs CPU ref {db_ref:.2f} dB "
            f"mel {mel:.4f} | vs {_selected_recipes()[0][0]} {db_base:.2f} dB"
        )
        del decoder
    logger.info(f"=== squeeze table: {num_latent_frames} latents x batch {batch}, 4x8 factor 8, traced best of 3 ===")
    logger.info(
        f"{'recipe':>12s} {'eager s':>8s} {'traced s':>9s} {'PSNR ref':>9s} {'mel':>7s} {'PSNR base':>10s}  split counts"
    )
    for name, counts, eager, traced, db_ref, mel, db_base in rows:
        logger.info(f"{name:>12s} {eager:8.4f} {traced:9.4f} {db_ref:9.2f} {mel:7.4f} {db_base:10.2f}  {counts}")
