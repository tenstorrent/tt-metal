# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Quality sign-off for the Wan2.2 4-step distill fast image-encode path.

Builds the distill pipeline ONCE, then for each of several real, in-distribution
conditioning images generates two videos with the SAME seed:

  baseline   full 81-frame VAE encode, slow H/W=0 default blockings, host-built
             conditioning video  (the original shipped path).
  fast       truncated 33-frame encode + swept conv3d blockings + T_out_block=1
             + on-device zero assembly  (WAN_DISTILL_FAST_VAE_ENCODER +
             WAN_DISTILL_ENCODER_T_OUT_1 + WAN_DISTILL_ONDEVICE_COND).

Per image we save both mp4s and report a per-frame PCC of fast-vs-baseline as an
automated divergence proxy. The mp4s are the real sign-off — eyeball them for
artifacts (duplicate subject, frame-0 noise, late-frame collapse).

Env:
  COMPARE_IMAGES   comma-separated image paths
  COMPARE_PROMPTS  '|'-separated prompts (1:1 with COMPARE_IMAGES)
  COMPARE_SEED     single seed (default 42)
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

_DEFAULT_IMAGES = "/data_bh/videos/val_astronaut.jpg,/data_bh/videos/val_landscape.jpg,/data_bh/videos/val_person.jpg"
_DEFAULT_PROMPTS = "|".join(
    [
        "A rocket lifts off from the launch pad into a clear blue sky, billowing smoke and flames, cinematic",
        "A serene landscape, soft clouds drifting across the sky, gentle wind moving through the grass, cinematic",
        "A young woman slowly turns her head and smiles softly, warm window light, painterly portrait",
    ]
)


def _pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().to(torch.float64)
    y = y.flatten().to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else 1.0


def _extract_frames(result) -> np.ndarray:
    arr = result.frames if hasattr(result, "frames") else (result[0] if isinstance(result, tuple) else result)
    return np.asarray(arr)[0]  # [F, H, W, 3]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(4, 32), (4, 32), 2, False, ring_params_8k, ttnn.Topology.Ring, False],
    ],
    ids=["bh_4x32sp1tp0_ring"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("width, height", [(1280, 720)], ids=["resolution_720p"])
def test_validate_real_images(mesh_device, mesh_shape, num_links, dynamic_load, topology, width, height, is_fsdp):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    images = os.environ.get("COMPARE_IMAGES", _DEFAULT_IMAGES).split(",")
    prompts = os.environ.get("COMPARE_PROMPTS", _DEFAULT_PROMPTS).split("|")
    assert len(images) == len(prompts), "COMPARE_IMAGES and COMPARE_PROMPTS must be 1:1"
    seed = int(os.environ.get("COMPARE_SEED", "42"))
    num_frames = 81

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

    # Baseline state (as the base pipeline built it: slow encoder, full encode).
    enc_slow = pipeline.tt_vae_encoder
    base_chunk = pipeline._encoder_t_chunk_size
    orig_encode_frames = pipeline._encode_frames_for
    fast_chunk = pipeline._ENCODER_T_CHUNK_BY_MESH.get(tuple(mesh_device.shape), pipeline.DISTILL_ENCODE_FRAMES)

    logger.info("[validate] building fast swept encoder (T_out=1)...")
    enc_fast = pipeline._build_fast_vae_encoder(force_t_out_block_1=True)

    def full_frames(nf, mcp):
        return nf

    def gen(image_prompt, prompt, fast: bool):
        if fast:
            pipeline.tt_vae_encoder = enc_fast
            pipeline._encoder_t_chunk_size = fast_chunk
            pipeline._encode_frames_for = orig_encode_frames  # truncating (min 33)
            os.environ["WAN_DISTILL_ONDEVICE_COND"] = "1"
        else:
            pipeline.tt_vae_encoder = enc_slow
            pipeline._encoder_t_chunk_size = base_chunk
            pipeline._encode_frames_for = full_frames
            os.environ["WAN_DISTILL_ONDEVICE_COND"] = "0"
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
        return _extract_frames(result), time.perf_counter() - t0

    summary = []
    for idx, (img_path, prompt) in enumerate(zip(images, prompts)):
        tag = os.path.splitext(os.path.basename(img_path))[0]
        image_prompt = [ImagePrompt(image=PIL.Image.open(img_path).convert("RGB"), frame_pos=0)]
        logger.info(f"[validate] === image {idx+1}/{len(images)}: {tag} === prompt={prompt!r:.70}")

        base_frames, base_t = gen(image_prompt, prompt, fast=False)
        fast_frames, fast_t = gen(image_prompt, prompt, fast=True)

        pccs = [
            _pcc(
                torch.from_numpy(base_frames[f].astype(np.float64)), torch.from_numpy(fast_frames[f].astype(np.float64))
            )
            for f in range(base_frames.shape[0])
        ]
        min_pcc, min_f, mean_pcc = float(np.min(pccs)), int(np.argmin(pccs)), float(np.mean(pccs))

        try:
            from models.tt_dit.utils.video import export_to_video

            export_to_video(base_frames, f"validate_{tag}_baseline.mp4", fps=16)
            export_to_video(fast_frames, f"validate_{tag}_fast.mp4", fps=16)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[validate] export failed for {tag}: {e}")

        summary.append((tag, base_t, fast_t, min_pcc, min_f, mean_pcc))
        logger.info(
            f"[validate] {tag}: baseline_total={base_t:.2f}s fast_total={fast_t:.2f}s "
            f"minPCC={min_pcc:.4f}@f{min_f} meanPCC={mean_pcc:.4f}"
        )

    pipeline._encode_frames_for = orig_encode_frames

    logger.info("=" * 88)
    logger.info("REAL-IMAGE VALIDATION SUMMARY (quad 4x32, 720p, 81 frames, untraced)")
    logger.info(f"seed={seed}")
    logger.info("-" * 88)
    logger.info(f"{'image':18s} {'base total':>11s} {'fast total':>11s} {'minPCC':>9s} {'meanPCC':>9s}")
    for tag, base_t, fast_t, min_pcc, min_f, mean_pcc in summary:
        flag = "  <== inspect" if min_pcc < 0.85 else ""
        logger.info(f"{tag:18s} {base_t:>10.2f}s {fast_t:>10.2f}s {min_pcc:>9.4f} {mean_pcc:>9.4f}{flag}")
    logger.info("=" * 88)
