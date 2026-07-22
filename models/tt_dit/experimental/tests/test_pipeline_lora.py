# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for Wan2.2 I2V with LoRA adapters fused into the base.

Reads adapter paths via env vars (compatible with the previous experimental
test): ``LORA_HIGH_PATH``, ``LORA_LOW_PATH``, ``LORA_SCALE``. Set
``LORA_STACK_HIGH`` and/or ``LORA_STACK_LOW`` to a comma-separated list of
``path[:scale]`` entries to exercise multi-LoRA stacking. If both single-LoRA
and stack env vars are set, the stack form wins.
"""
import itertools
import os
from typing import List, Tuple

import numpy as np
import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.experimental.pipelines.pipeline_wan_lora import LoRASpec, WanPipelineI2VLora
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt
from models.tt_dit.utils.test import line_params, ring_params


def _parse_stack(env_val: str) -> List[LoRASpec]:
    """Parse ``path[:scale],path[:scale]`` into a LoRASpec list."""
    out: List[LoRASpec] = []
    for entry in env_val.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            path, scale_str = entry.rsplit(":", 1)
            out.append(LoRASpec(path.strip(), float(scale_str)))
        else:
            out.append(LoRASpec(entry))
    return out


def _resolve_lora_args() -> Tuple[List[LoRASpec], List[LoRASpec], float]:
    stack_high = os.environ.get("LORA_STACK_HIGH")
    stack_low = os.environ.get("LORA_STACK_LOW")
    single_high = os.environ.get("LORA_HIGH_PATH")
    single_low = os.environ.get("LORA_LOW_PATH")
    scale = float(os.environ.get("LORA_SCALE", "1.0"))

    if stack_high or stack_low:
        return (
            _parse_stack(stack_high) if stack_high else [],
            _parse_stack(stack_low) if stack_low else [],
            scale,
        )

    high = [LoRASpec(single_high, scale)] if single_high else []
    low = [LoRASpec(single_low, scale)] if single_low else []
    return high, low, scale


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
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
        (1280, 720),
    ],
    ids=[
        "resolution_480p",
        "resolution_720p",
    ],
)
def test_pipeline_inference(
    mesh_device,
    mesh_shape,
    num_links,
    dynamic_load,
    topology,
    width,
    height,
    is_fsdp,
    no_prompt,
):
    lora_high, lora_low, scale = _resolve_lora_args()
    if not lora_high and not lora_low:
        pytest.skip(
            "Set LORA_HIGH_PATH / LORA_LOW_PATH (single-LoRA) or "
            "LORA_STACK_HIGH / LORA_STACK_LOW (multi-LoRA) to run."
        )

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    pil_image = PIL.Image.open(os.environ.get("PROMPT_IMAGE", "./prompt_image.png"))
    image_prompt = [ImagePrompt(image=pil_image, frame_pos=0)]
    negative_prompt = ""

    num_frames = 81
    num_inference_steps = int(os.environ.get("NUM_STEPS", "40"))
    guidance_scale = float(os.environ.get("GUIDANCE_SCALE", "3.5"))
    guidance_scale_2 = float(os.environ.get("GUIDANCE_SCALE_2", str(guidance_scale)))
    boundary_ratio = float(os.environ.get("BOUNDARY_RATIO", "0.875"))

    pipeline = WanPipelineI2VLora.create_pipeline(
        mesh_device=mesh_device,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        height=height,
        width=width,
        num_frames=num_frames,
        boundary_ratio=boundary_ratio,
        lora_high=lora_high if lora_high else None,
        lora_low=lora_low if lora_low else None,
    )

    prompt = os.environ.get("PROMPT", "A golden retriever running on a sandy beach, waves in the background")

    def run(*, prompt, number, seed):
        logger.info(f"Running LoRA inference with prompt: '{prompt}'")
        logger.info(
            f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps, "
            f"scale={scale}, stack_high={len(lora_high)}, stack_low={len(lora_low)}"
        )

        with torch.no_grad():
            result = pipeline(
                prompts=[prompt],
                image_prompt=image_prompt,
                negative_prompts=[negative_prompt],
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                output_type="uint8",
            )

        if hasattr(result, "frames"):
            frames = result.frames
        else:
            frames = result[0] if isinstance(result, tuple) else result

        logger.info(f"  Output shape: {frames.shape if hasattr(frames, 'shape') else 'Unknown'}")
        if isinstance(frames, np.ndarray):
            logger.info(f"  Video data range: [{frames.min():.3f}, {frames.max():.3f}]")
        elif isinstance(frames, torch.Tensor):
            logger.info(f"  Video data range: [{frames.min().item():.3f}, {frames.max().item():.3f}]")

        frames = frames[0]
        output_filename = f"wan_lora_i2v_{width}x{height}_{number}.mp4"
        try:
            from models.tt_dit.utils.video import export_to_video

            export_to_video(frames, output_filename, fps=16)
            logger.info(f"Saved video to: {output_filename}")
        except ImportError:
            logger.info("Could not export video - imageio_ffmpeg not available")

    if no_prompt:
        run(prompt=prompt, number=0, seed=42)
    else:
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)
