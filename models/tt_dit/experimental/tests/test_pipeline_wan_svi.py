# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for Wan2.2 I2V with Stable-Video-Infinity (SVI 2.0 Pro).

Requires ``SVI_HIGH_PATH`` and ``SVI_LOW_PATH`` env vars pointing at the SVI
LoRA safetensors (``vita-video-gen/svi-model`` on HuggingFace). All other
inference knobs (number of clips, num_motion_latent, num_overlap_frame) are
env-configurable to support experimentation.
"""
import os

import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.experimental.pipelines.pipeline_wan_svi import WanPipelineSVI
from models.tt_dit.utils.test import line_params, ring_params


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_params, ttnn.Topology.Linear, False],
        [(4, 8), (4, 8), 1, 0, 2, False, ring_params, ttnn.Topology.Ring, False],
    ],
    ids=["bh_2x4sp1tp0", "bh_4x8sp1tp0_ring"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "width, height",
    [
        (832, 480),
    ],
    ids=["resolution_480p"],
)
@pytest.mark.parametrize(
    "regime",
    ["python", "comfyui"],
)
def test_long_video(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    width,
    height,
    is_fsdp,
    regime,
):
    svi_high = os.environ.get("SVI_HIGH_PATH")
    svi_low = os.environ.get("SVI_LOW_PATH")
    if not svi_high or not svi_low:
        pytest.skip("SVI_HIGH_PATH and SVI_LOW_PATH env vars are required for this test.")

    lightx2v_high = os.environ.get("LIGHTX2V_HIGH_PATH")
    lightx2v_low = os.environ.get("LIGHTX2V_LOW_PATH")
    if regime == "comfyui" and (not lightx2v_high or not lightx2v_low):
        pytest.skip("regime='comfyui' requires LIGHTX2V_HIGH_PATH / LIGHTX2V_LOW_PATH env vars.")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_clips = int(os.environ.get("SVI_NUM_CLIPS", "2"))
    num_motion_latent = int(os.environ.get("SVI_NUM_MOTION_LATENT", "1"))
    num_overlap_frame = int(os.environ.get("SVI_NUM_OVERLAP_FRAME", "4"))
    base_seed = int(os.environ.get("SVI_SEED", "0"))
    regime_default_steps = "50" if regime == "python" else "6"
    regime_default_cfg = "5.0" if regime == "python" else "1.5"
    num_inference_steps = int(os.environ.get("NUM_STEPS", regime_default_steps))
    guidance_scale = float(os.environ.get("GUIDANCE_SCALE", regime_default_cfg))
    guidance_scale_2 = float(os.environ.get("GUIDANCE_SCALE_2", str(guidance_scale)))

    pil_image = PIL.Image.open(os.environ.get("PROMPT_IMAGE", "./prompt_image.png"))
    prompt_env = os.environ.get("PROMPT", "A golden retriever running on a sandy beach, waves in the background")
    if "||" in prompt_env:
        prompt = [p.strip() for p in prompt_env.split("||")]
        if len(prompt) != num_clips:
            pytest.fail(
                f"PROMPT contains '||' splitting into {len(prompt)} prompts but "
                f"SVI_NUM_CLIPS={num_clips}. Provide exactly {num_clips} prompts."
            )
    else:
        prompt = prompt_env
    num_frames = 81

    pipeline_kwargs = dict(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        height=height,
        width=width,
        num_frames=num_frames,
        svi_high=svi_high,
        svi_low=svi_low,
        regime=regime,
        num_motion_latent=num_motion_latent,
        num_overlap_frame=num_overlap_frame,
    )
    if regime == "comfyui":
        pipeline_kwargs["lightx2v_high"] = lightx2v_high
        pipeline_kwargs["lightx2v_low"] = lightx2v_low

    pipeline = WanPipelineSVI.create_pipeline(**pipeline_kwargs)

    logger.info(
        f"SVI run: {num_clips} clips × {num_frames} frames at {width}x{height}, "
        f"steps={num_inference_steps}, motion={num_motion_latent}, overlap={num_overlap_frame}"
    )

    output_basename = f"wan_svi_{regime}_{num_clips}clips_{width}x{height}"

    with torch.no_grad():
        video = pipeline.generate_long_video(
            prompt=prompt,
            image_prompt=pil_image,
            num_clips=num_clips,
            num_frames=num_frames,
            height=height,
            width=width,
            base_seed=base_seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            partial_output_path=f"{output_basename}.partial.pt",
        )

    expected_T = num_clips * num_frames - (num_clips - 1) * num_overlap_frame
    actual_T = video.shape[0]
    logger.info(f"Concat shape: {tuple(video.shape)} (expected T={expected_T})")
    assert actual_T == expected_T, f"Expected {expected_T} frames, got {actual_T}"

    assert not torch.isnan(video.float()).any(), "video contains NaN"

    torch.save(video.detach().cpu(), f"{output_basename}.pt")
    logger.info(f"Saved raw tensor to: {output_basename}.pt")

    try:
        from models.tt_dit.utils.video import export_to_video

        # output_type='pt' returns (T, C, H, W) float; export_to_video wants
        # (T, H, W, C) uint8.
        arr = video.detach().cpu()
        if arr.ndim == 4 and arr.shape[1] in (1, 3):
            arr = arr.permute(0, 2, 3, 1).contiguous()
        if arr.dtype != torch.uint8:
            arr = arr.float()
            if arr.min().item() < 0:
                arr = (arr + 1.0) * 127.5
            elif arr.max().item() <= 1.0 + 1e-3:
                arr = arr * 255.0
            arr = arr.clamp(0, 255).to(torch.uint8)
        export_to_video(arr.numpy(), f"{output_basename}.mp4", fps=16)
        logger.info(f"Saved video to: {output_basename}.mp4")
    except ImportError:
        logger.info("Could not export video - imageio_ffmpeg not available")


def test_comfyui_regime_requires_lightx2v():
    """regime='comfyui' must demand LightX2V LoRA paths."""
    with pytest.raises(ValueError, match="lightx2v_high"):
        WanPipelineSVI(
            mesh_device=None,
            parallel_config=None,
            vae_parallel_config=None,
            encoder_parallel_config=None,
            num_links=1,
            boundary_ratio=0.875,
            scheduler=None,
            dynamic_load=False,
            topology=None,
            is_fsdp=False,
            checkpoint_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            vae_t_chunk_size=None,
            sdpa_t_fracture_w_only=False,
            height=480,
            width=832,
            num_frames=81,
            svi_high="/tmp/does-not-exist.safetensors",
            svi_low="/tmp/does-not-exist.safetensors",
            regime="comfyui",
        )
