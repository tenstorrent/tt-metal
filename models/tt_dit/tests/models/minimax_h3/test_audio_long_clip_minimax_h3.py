# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The T-sharded vocoder at the 15 s production length, against the reference, as the pipeline runs it.

Found 2026-09-07 on a 4x8 Galaxy: every 15 s t2va generation of this branch produced audio with 86-100 % of the
samples at -1.0. The latents were fine -- the reference decodes them to a normal waveform (std 0.031, the quad's
statistics) -- and the factor-8 decoder was exact on the 207-frame (5 s) clip that the pipeline's construction
warm-up decodes first (69 dB). `test_audio_decode_t_parallel` covers 207 and 192 frames only, and the perf test
asserts finiteness only, so nothing gated the 603-frame decode.

Two decoders are built the way `MiniMaxH3Pipeline._prepare_audio_decoder` builds the production one (factor 8 on
mesh axis 1, Linear CCL, two links, the shipping split mode). The first decodes 207 frames and then 603, like a
served process; the second decodes 603 first. Each 603-frame result is gated against the reference on its own, so
a cache poisoned by the shorter clip and a length-dependent bug are told apart by which case fails.

Set MINIMAX_H3_AUDIO_LATENTS_DIR to a directory holding `call0_latents.npy` (207 frames) and `call2_latents.npy`
(603 frames) from `dump_audio_latents_plugin` to use the pipeline's own latents; otherwise seeded random latents at
the pipeline's scale are used.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

from ....parallel.config import ParallelFactor
from ....parallel.manager import CCLManager
from .common import MESH_4X8_RING, build_audio_decoder, load_config, psnr, weights_subdir

SHORT_FRAMES = 207  # 5 s
LONG_FRAMES = 603  # 15 s
MIN_PSNR_DB = 40.0  # the reference-vs-device gate the T-parallel test uses
T_FACTOR, T_AXIS = 8, 1  # `_resolve_audio_t_shard` on a 4x8 with the default request of 8


def _latents(num_frames: int, config: dict) -> torch.Tensor:
    lat_dir = os.environ.get("MINIMAX_H3_AUDIO_LATENTS_DIR")
    name = {SHORT_FRAMES: "call0_latents.npy", LONG_FRAMES: "call2_latents.npy"}[num_frames]
    if lat_dir and os.path.exists(os.path.join(lat_dir, name)):
        latents = torch.from_numpy(np.load(os.path.join(lat_dir, name))).float()
        assert latents.shape == (2, config["latent_channels"], num_frames), latents.shape
        logger.info(f"{num_frames} frames: pipeline latents from {lat_dir}/{name}")
        return latents
    torch.manual_seed(num_frames)
    return torch.randn(2, config["latent_channels"], num_frames) * 0.9  # the pipeline's denormalized scale


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_audio_decode_15s_after_5s(mesh_device):
    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from safetensors.torch import load_file

    from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict

    config = load_config(weights_dir)
    reference = AutoencoderKLMiniMaxH3Audio(**config).eval()
    reference.load_state_dict(load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")))
    converted = convert_minimax_h3_audio_state_dict(dict(reference.state_dict()))

    short, long = _latents(SHORT_FRAMES, config), _latents(LONG_FRAMES, config)
    with torch.no_grad():
        ref_short = reference.decode(short)
        ref_long = reference.decode(long)
    ref_short = getattr(ref_short, "sample", ref_short).float()
    ref_long = getattr(ref_long, "sample", ref_long).float()
    logger.info(
        f"reference: {SHORT_FRAMES} frames std {ref_short.std():.4f}; {LONG_FRAMES} frames std {ref_long.std():.4f}"
    )

    def build():
        pc = ParallelFactor(factor=T_FACTOR, mesh_axis=T_AXIS)
        ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear)
        decoder = build_audio_decoder(config, mesh_device, parallel_config=pc, ccl_manager=ccl)
        decoder.load_torch_state_dict(converted, strict=False)
        return decoder

    def gate(label: str, out: torch.Tensor, ref: torch.Tensor) -> float:
        out = out.float().reshape(ref.shape)
        saturated = (out <= -0.999).float().mean().item()
        db = psnr(ref, out)
        logger.info(f"{label}: PSNR vs reference {db:.2f} dB, std {out.std():.4f}, {saturated:.1%} of samples at -1.0")
        return db

    # Served order: the construction warm-up's 5 s clip, then the 15 s request.
    served = build()
    db_short = gate(f"served decoder, {SHORT_FRAMES} frames", served(short), ref_short)
    db_long_after_short = gate(f"served decoder, {LONG_FRAMES} frames after {SHORT_FRAMES}", served(long), ref_long)
    del served

    # A fresh decoder that has never seen the short clip.
    fresh = build()
    db_long_fresh = gate(f"fresh decoder, {LONG_FRAMES} frames first", fresh(long), ref_long)
    del fresh

    assert db_short > MIN_PSNR_DB, f"{SHORT_FRAMES}-frame decode diverges from the reference: {db_short:.1f} dB"
    assert db_long_fresh > MIN_PSNR_DB, (
        f"{LONG_FRAMES}-frame decode is wrong even on a fresh decoder ({db_long_fresh:.1f} dB): a length-dependent "
        "bug in the factor-8 vocoder, not cache poisoning"
    )
    assert db_long_after_short > MIN_PSNR_DB, (
        f"{LONG_FRAMES}-frame decode is wrong after a {SHORT_FRAMES}-frame decode ({db_long_after_short:.1f} dB) but "
        f"right on a fresh decoder ({db_long_fresh:.1f} dB): a per-decoder cache keyed without the clip length"
    )
