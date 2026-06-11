# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate real output videos with the FULLY-OPTIMIZED distill fast path.

Builds the distill pipeline ONCE with all image-encode optimizations enabled
(truncated 33-frame encode + swept conv3d blockings + T_out_block=1 + on-device
zero assembly), then generates one video per (image, prompt) pair and saves the
mp4. This is the ~4s shipping configuration — no baseline, no PCC, just outputs.

Enable the fast path via env before launching:
  WAN_DISTILL_FAST_VAE_ENCODER=1 WAN_DISTILL_ENCODER_T_OUT_1=1 WAN_DISTILL_ONDEVICE_COND=1

Env:
  GEN_IMAGES    comma-separated image paths
  GEN_PROMPTS   '|'-separated prompts (1:1 with GEN_IMAGES)
  GEN_SEED      seed (default 42)
  GEN_OUTDIR    output dir for mp4s (default cwd)
"""
import os
import time

import numpy as np
import PIL.Image
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.experimental.pipelines.pipeline_wan_distill import WanDistillPipelineI2V
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt
from models.tt_dit.utils.test import ring_params_8k

_DEFAULT_IMAGES = ",".join(
    [
        "/data_bh/videos/anime_girl.png",
        "/data_bh/videos/val_astronaut.jpg",
        "/data_bh/videos/val_landscape.jpg",
        "/data_bh/videos/val_person.jpg",
        "/data_bh/videos/val_cat.jpg",
    ]
)
_DEFAULT_PROMPTS = "|".join(
    [
        "The anime girl smiles gently as cherry blossom petals drift past her face, her purple hair flowing softly in a spring breeze, sparkling light, anime style",
        "Gentle waves roll onto the shore as the sun sets over the ocean, warm golden light reflecting on the water, cinematic",
        "A serene fantasy landscape, soft clouds drifting across the sky, sunlight glowing over the mountains, cinematic",
        "A young woman slowly turns her head and smiles softly, warm window light, painterly portrait",
        "A fluffy cat blinks slowly and looks around, whiskers twitching gently, soft natural light, photorealistic",
    ]
)


def _extract_frames(result) -> np.ndarray:
    arr = result.frames if hasattr(result, "frames") else (result[0] if isinstance(result, tuple) else result)
    return np.asarray(arr)[0]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(4, 32), (4, 32), 2, False, ring_params_8k, ttnn.Topology.Ring, False],
    ],
    ids=["bh_4x32sp1tp0_ring"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("width, height", [(1280, 720)], ids=["resolution_720p"])
def test_generate_fast_samples(mesh_device, mesh_shape, num_links, dynamic_load, topology, width, height, is_fsdp):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    images = os.environ.get("GEN_IMAGES", _DEFAULT_IMAGES).split(",")
    prompts = os.environ.get("GEN_PROMPTS", _DEFAULT_PROMPTS).split("|")
    assert len(images) == len(prompts), "GEN_IMAGES and GEN_PROMPTS must be 1:1"
    seed = int(os.environ.get("GEN_SEED", "42"))
    outdir = os.environ.get("GEN_OUTDIR", ".")
    num_frames = 81

    logger.info(
        f"[gen] fast flags: FAST_VAE_ENCODER={os.environ.get('WAN_DISTILL_FAST_VAE_ENCODER')} "
        f"ENCODER_T_OUT_1={os.environ.get('WAN_DISTILL_ENCODER_T_OUT_1')} "
        f"ONDEVICE_COND={os.environ.get('WAN_DISTILL_ONDEVICE_COND')}"
    )

    pipeline = WanDistillPipelineI2V.create_pipeline(
        mesh_device=mesh_device,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        height=height,
        width=width,
        num_frames=num_frames,
    )

    from models.tt_dit.utils.video import export_to_video

    summary = []
    for idx, (img_path, prompt) in enumerate(zip(images, prompts)):
        tag = os.path.splitext(os.path.basename(img_path))[0]
        image_prompt = [ImagePrompt(image=PIL.Image.open(img_path).convert("RGB"), frame_pos=0)]
        logger.info(f"[gen] === {idx+1}/{len(images)}: {tag} === prompt={prompt!r:.70}")

        t0 = time.perf_counter()
        with torch.no_grad():
            result = pipeline(
                prompts=[prompt],
                image_prompt=image_prompt,
                negative_prompts=[""],
                num_inference_steps=4,
                seed=seed,
                guidance_scale=1.0,
                guidance_scale_2=1.0,
                output_type="uint8",
            )
        dt = time.perf_counter() - t0
        frames = _extract_frames(result)
        out_fn = os.path.join(outdir, f"fast_{tag}.mp4")
        export_to_video(frames, out_fn, fps=16)
        summary.append((tag, dt, out_fn))
        logger.info(f"[gen] {tag}: total={dt:.2f}s -> {out_fn}")

    logger.info("=" * 80)
    logger.info("FAST-SAMPLES SUMMARY (quad 4x32, 720p, 81 frames, untraced wall-clock)")
    for tag, dt, out_fn in summary:
        logger.info(f"  {tag:18s} {dt:>7.2f}s  {out_fn}")
    logger.info("=" * 80)
