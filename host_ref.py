# SPDX-License-Identifier: Apache-2.0
"""Reconstructed host-reference loader for capture_stages.py.

Builds the upstream ltx_core DiffusionVideoDecoder from the shipped video-VAE
checkpoint, on CPU in fp32, with NO cutlass-FNA module op (that is a CUDA kernel;
TT hosts have neither CUDA nor NATTEN). Mirrors ltx-trainer's video-vae builder.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)


def load():
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae import (
        VideoDecoderConfigurator,
        is_diffusion_video_vae,
        video_decoder_sd_ops_for_checkpoint,
    )

    path = str(CHECKPOINT)
    assert is_diffusion_video_vae(path), f"{path} is not a diffusion video VAE"
    decoder = SingleGPUModelBuilder(
        model_path=path,
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=video_decoder_sd_ops_for_checkpoint(path, diffusion_vae=True),
        module_ops=(),  # host: no cutlass-FNA (CUDA)
    ).build(device=torch.device("cpu"), dtype=torch.float32)

    # Default NeighborhoodAttention3D needs NATTEN (CUDA). On a TT host, swap in
    # upstream's own eager tiled-SDPA fallback on every attention module — the same
    # backend the tt_dit stage-5 parity test uses, sharing the window geometry.
    from ltx_core.model.video_vae.transformer.fallback_na import EagerSdpaAttention

    eager = EagerSdpaAttention()
    swapped = 0
    for module in decoder.modules():
        if hasattr(module, "attention_function"):
            module.attention_function = eager
            swapped += 1
    print(f"host_ref: swapped {swapped} attention modules to EagerSdpaAttention")
    return decoder


if __name__ == "__main__":
    d = load()
    print("loaded:", type(d).__name__)
    for a in ("det_stages", "diff_blocks", "forward_stage_4", "model_output_type"):
        print(f"  has {a}:", hasattr(d, a))
