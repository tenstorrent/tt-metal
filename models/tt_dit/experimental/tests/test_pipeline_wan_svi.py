# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for Wan2.2 I2V with SVI 2.0 Pro.

Required env vars (or the test is skipped):
- ``SVI_HIGH_PATH`` / ``SVI_LOW_PATH``: SVI LoRA safetensors (high/low noise)
- ``LIGHTX2V_HIGH_PATH`` / ``LIGHTX2V_LOW_PATH``: required only when regime='comfyui'

Normal optional env vars:
- ``PROMPT_IMAGE``: anchor image for every clip (default: ``./prompt_image.png``)
- ``PROMPT``: text prompt for generation. Pass one string for all clips, or a
  semicolon-separated list with exactly ``SVI_NUM_CLIPS`` entries for per-clip
  prompts.
- ``SVI_NUM_CLIPS``: number of 81-frame clips to chain (default: ``2``)

Example per-clip prompt format:
``PROMPT="clip 1 prompt;clip 2 prompt;clip 3 prompt"`` requires
``SVI_NUM_CLIPS=3``. Example invocation::

    SVI_HIGH_PATH=/path/svi_high.safetensors \\
    SVI_LOW_PATH=/path/svi_low.safetensors \\
    LIGHTX2V_HIGH_PATH=/path/lx2v_high.safetensors \\
    LIGHTX2V_LOW_PATH=/path/lx2v_low.safetensors \\
    SVI_NUM_CLIPS=10 PROMPT_IMAGE=./example.png \\
    pytest models/tt_dit/experimental/tests/test_pipeline_wan_svi.py \\
      -v -k "comfyui and bh_2x4" -s

See ``experimental/models/Wan2_2_SVI.md`` for advanced knobs (steps, CFG, seed,
motion-latent handoff, overlap) and the upstream-workflow comparison.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.experimental.pipelines.pipeline_wan_svi import WanPipelineSVI
from models.tt_dit.utils.test import line_params, ring_params

PromptArg = Union[str, List[str]]
FloatArg = Union[float, List[float]]


_DEFAULT_PROMPT = "A golden retriever running on a sandy beach, waves in the background"


@dataclass(frozen=True)
class _SVIRunConfig:
    regime: str
    svi_high: str
    svi_low: str
    lightx2v_high: Optional[str]
    lightx2v_low: Optional[str]
    anchor_image: Image.Image
    prompt: PromptArg
    num_clips: int
    num_motion_latent: int
    num_overlap_frame: int
    base_seed: int
    num_inference_steps: int
    guidance_scale: FloatArg
    guidance_scale_2: FloatArg
    num_frames: int
    width: int
    height: int
    output_basename: str


def _parse_per_clip_env(env_val: str, num_clips: int, name: str, cast):
    """Parse a scalar or semicolon-separated per-clip env var."""
    parts = [p.strip() for p in env_val.split(";")]
    if len(parts) == 1:
        return cast(parts[0])

    if len(parts) != num_clips:
        pytest.fail(f"{name} has {len(parts)} entries but SVI_NUM_CLIPS={num_clips}")
    return [cast(p) for p in parts]


def _read_svi_config(regime: str, width: int, height: int) -> _SVIRunConfig:
    svi_high = os.environ.get("SVI_HIGH_PATH")
    svi_low = os.environ.get("SVI_LOW_PATH")
    if not svi_high or not svi_low:
        pytest.skip("SVI_HIGH_PATH and SVI_LOW_PATH env vars are required for this test.")

    lightx2v_high = os.environ.get("LIGHTX2V_HIGH_PATH")
    lightx2v_low = os.environ.get("LIGHTX2V_LOW_PATH")
    if regime == "comfyui" and (not lightx2v_high or not lightx2v_low):
        pytest.skip("regime='comfyui' requires LIGHTX2V_HIGH_PATH / LIGHTX2V_LOW_PATH env vars.")

    num_clips = int(os.environ.get("SVI_NUM_CLIPS", "2"))
    default_steps = "50" if regime == "python" else "6"
    default_cfg = "5.0" if regime == "python" else "1.5"
    guidance_scale_env = os.environ.get("GUIDANCE_SCALE", default_cfg)

    prompt = _parse_per_clip_env(os.environ.get("PROMPT", _DEFAULT_PROMPT), num_clips, "PROMPT", str)
    guidance_scale = _parse_per_clip_env(guidance_scale_env, num_clips, "GUIDANCE_SCALE", float)
    guidance_scale_2 = _parse_per_clip_env(
        os.environ.get("GUIDANCE_SCALE_2", guidance_scale_env),
        num_clips,
        "GUIDANCE_SCALE_2",
        float,
    )

    return _SVIRunConfig(
        regime=regime,
        svi_high=svi_high,
        svi_low=svi_low,
        lightx2v_high=lightx2v_high,
        lightx2v_low=lightx2v_low,
        anchor_image=Image.open(os.environ.get("PROMPT_IMAGE", "./prompt_image.png")),
        prompt=prompt,
        num_clips=num_clips,
        num_motion_latent=int(os.environ.get("SVI_NUM_MOTION_LATENT", "1")),
        num_overlap_frame=int(os.environ.get("SVI_NUM_OVERLAP_FRAME", "4")),
        base_seed=int(os.environ.get("SVI_SEED", "0")),
        num_inference_steps=int(os.environ.get("NUM_STEPS", default_steps)),
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        num_frames=81,
        width=width,
        height=height,
        output_basename=f"wan_svi_{regime}_{num_clips}clips_{width}x{height}",
    )


def _pipeline_kwargs(config: _SVIRunConfig, *, mesh_device, num_links, dynamic_load, topology, is_fsdp):
    kwargs = dict(
        mesh_device=mesh_device,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        svi_high=config.svi_high,
        svi_low=config.svi_low,
        regime=config.regime,
        num_motion_latent=config.num_motion_latent,
        num_overlap_frame=config.num_overlap_frame,
    )
    if config.regime == "comfyui":
        kwargs["lightx2v_high"] = config.lightx2v_high
        kwargs["lightx2v_low"] = config.lightx2v_low
    return kwargs


def _assert_video(video: torch.Tensor, *, num_clips: int, num_frames: int, num_overlap_frame: int) -> None:
    expected_t = num_clips * num_frames - (num_clips - 1) * num_overlap_frame
    actual_t = video.shape[0]
    logger.info(f"Concat shape: {tuple(video.shape)} (expected T={expected_t})")
    assert actual_t == expected_t, f"Expected {expected_t} frames, got {actual_t}"
    assert not torch.isnan(video.float()).any(), "video contains NaN"


def _save_outputs(video: torch.Tensor, output_basename: str) -> None:
    video = video.detach()
    torch.save(video, f"{output_basename}.pt")
    logger.info(f"Saved raw tensor to: {output_basename}.pt")

    try:
        from models.tt_dit.utils.video import export_to_video

        # output_type='pt_with_last_latent' decodes frames as (T, C, H, W) float;
        # export_to_video wants (T, H, W, C) uint8.
        arr = video
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


def _touch(path) -> str:
    path.write_bytes(b"")
    return str(path)


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 2, True, line_params, ttnn.Topology.Linear, False],
        [(4, 8), (4, 8), 2, False, ring_params, ttnn.Topology.Ring, False],
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
    num_links,
    dynamic_load,
    topology,
    width,
    height,
    is_fsdp,
    regime,
):
    config = _read_svi_config(regime, width, height)
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))
    pipeline = WanPipelineSVI.create_pipeline(
        **_pipeline_kwargs(
            config,
            mesh_device=mesh_device,
            num_links=num_links,
            dynamic_load=dynamic_load,
            topology=topology,
            is_fsdp=is_fsdp,
        )
    )

    logger.info(
        f"SVI run: {config.num_clips} clips × {config.num_frames} frames at {width}x{height}, "
        f"steps={config.num_inference_steps}, motion={config.num_motion_latent}, "
        f"overlap={config.num_overlap_frame}"
    )

    with torch.no_grad():
        video = pipeline.generate_long_video(
            prompt=config.prompt,
            anchor_image=config.anchor_image,
            num_clips=config.num_clips,
            base_seed=config.base_seed,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            guidance_scale_2=config.guidance_scale_2,
            partial_output_path=f"{config.output_basename}.partial.pt",
        )

    _assert_video(
        video,
        num_clips=config.num_clips,
        num_frames=config.num_frames,
        num_overlap_frame=config.num_overlap_frame,
    )
    _save_outputs(video, config.output_basename)


def test_comfyui_regime_requires_lightx2v(tmp_path):
    """regime='comfyui' must demand LightX2V LoRA paths."""
    with pytest.raises(ValueError, match="lightx2v_high"):
        WanPipelineSVI(
            device=None,
            config=None,
            svi_high=_touch(tmp_path / "svi_high.safetensors"),
            svi_low=_touch(tmp_path / "svi_low.safetensors"),
            regime="comfyui",
        )
