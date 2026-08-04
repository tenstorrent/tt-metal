# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy target: the MiniMax-H3 audio decoder (BigVGAN) at the shipping 5 s shape.

Audio decode measures 1.273 s against a ~0.05 s target (STATE.md amendment 59) and has never
been profiled. Unlike the visual path it gets nothing from data-parallelism -- a 5 s clip is
one stream, not 224 independent work units -- so the answer has to come from the ops
themselves, and this is the first look at them.

    python -m tracy -p -r -m pytest \\
        models/tt_dit/tests/models/minimax_h3/profile_audio_decoder_minimax_h3.py

**Not yet run to completion.** A first attempt exceeded a 10 min cap while still JIT-building
instrumented kernels -- the decoder has many distinct conv1d shapes across the upsampling
stack, so there are a lot of them, and profiler instrumentation rebuilds every one. Give it a
longer budget on the next attempt; it is a build-time cost, not a hang.
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn

from ....models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from .test_performance_vae_minimax_h3 import HOP_LENGTH, _config, _weights_dir

SINGLE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
ITERATIONS = 2


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE, indirect=["mesh_device", "device_params"])
def test_profile_audio_decoder(mesh_device):
    weights_dir = _weights_dir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from safetensors.torch import load_file

    from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict

    config = _config(weights_dir)
    reference = AutoencoderKLMiniMaxH3Audio(**config).eval()
    reference.load_state_dict(load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")))
    converted = convert_minimax_h3_audio_state_dict(dict(reference.state_dict()))

    torch.manual_seed(2)
    num_latent_frames = 207
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

    latents = torch.randn(2, config["latent_channels"], num_latent_frames) * 0.1
    assert num_latent_frames * HOP_LENGTH > 0
    for _ in range(ITERATIONS):
        decoder(latents)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
