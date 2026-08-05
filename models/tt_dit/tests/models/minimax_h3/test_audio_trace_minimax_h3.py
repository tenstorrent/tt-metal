# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Traced vs untraced audio decode, with a correctness gate between them.

Audio decode measures 1.273 s against a ~0.05 s target (STATE.md amendment 59). It gets
nothing from the data-parallelism that carried the visual path -- a 5 s clip is one stream,
not 224 independent work units -- but it is the opposite kind of workload from the visual
halves: ~1 MB tensors over many ops, so **host dispatch**, not device time, is expected to
dominate. ``vocoder_ltx.Vocoder`` says so itself ("the vocoder is ~70% host-bound") and
already carries a `@traced_function` device region plus a ``forward_traced`` entry point --
H3's decoder simply called the untraced ``forward_BCT``.

Traced output must match untraced exactly-ish, so this asserts before it reports timing.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from ....models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from .test_performance_vae_minimax_h3 import _config, _psnr, _weights_dir

# 300 MB covers the default path but not the precision levers: with ``MINIMAX_H3_AUDIO_ACCURATE=1`` the
# graph grows (the depthwise MAC form does one pass per tap, and the shifted-matmul convs add
# ``3 * kernel_size`` matmuls each), and the trace needs 375463936 B. The failure names the requirement
# exactly -- ``mesh_trace.cpp:80``, "Creating trace buffers of size ... but only ... is allocated" -- so
# size for the larger of the two rather than making the region depend on an env var.
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
    weights_dir = _weights_dir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from loguru import logger
    from safetensors.torch import load_file

    from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict

    config = _config(weights_dir)
    reference = AutoencoderKLMiniMaxH3Audio(**config).eval()
    reference.load_state_dict(load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")))
    converted = convert_minimax_h3_audio_state_dict(dict(reference.state_dict()))

    torch.manual_seed(2)
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

    latents = torch.randn(2, config["latent_channels"], NUM_LATENT_FRAMES) * 0.1

    plain = decoder(latents)
    traced = decoder(latents, traced=True)
    assert traced.shape == plain.shape, f"{traced.shape} != {plain.shape}"

    # Trace replays the same program on the same data, so bit-identical (PSNR inf) is the
    # *expected* result rather than a suspicious one. It is a weak assertion on its own
    # though -- it would read inf just as happily if traced=True had silently fallen through
    # to the untraced path -- so check separately that a tracer was actually captured, and
    # that the output is not trivially zero (which would also give inf).
    tracers = type(decoder.decoder)._forward_device._tracers_keyed.get(decoder.decoder, {})
    assert tracers, "traced=True captured no trace; the PSNR below would be meaningless"
    assert plain.abs().max() > 1e-6, "decoder produced all-zero output; PSNR would be vacuous"

    psnr = _psnr(plain, traced)
    logger.info(f"traced vs untraced PSNR: {psnr:.2f} dB ({len(tracers)} trace(s) captured)")

    untraced_s = _best(lambda: decoder(latents))
    traced_s = _best(lambda: decoder(latents, traced=True))
    logger.info(
        f"PERF audio_decode_5s untraced {untraced_s:.4f} s | traced {traced_s:.4f} s "
        f"-> {untraced_s / traced_s:.2f}x"
    )

    # Where is the 1.2 s? The traced region is only the vocoder's `_forward_device`; the
    # latent projection round-trips through host in the middle of forward
    # (decoder_minimax_h3_audio.py: to_torch -> transpose/contiguous -> re-upload), and the
    # final readback is untraced too. Split them so the next step is not a guess.
    proj_s = _best(lambda: decoder._project_latents_device(latents))
    projected = decoder._project_latents_device(latents)
    voc_s = _best(lambda: decoder.decoder.forward_BCT(projected))
    voc_traced_s = _best(lambda: decoder.decoder.forward_BCT_traced(projected))
    logger.info(
        f"PERF split: dec_in_proj {proj_s:.4f} s | vocoder {voc_s:.4f} s | " f"vocoder traced {voc_traced_s:.4f} s"
    )
    decoder.release_trace()

    assert psnr > 60.0, f"traced output diverges from untraced: PSNR {psnr:.2f} dB"
