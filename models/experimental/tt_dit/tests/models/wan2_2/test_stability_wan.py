# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from models.experimental.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.experimental.tt_dit.parallel.config import DiTParallelConfig, VaeHWParallelConfig, ParallelFactor
from diffusers.utils import export_to_video
import pytest
import ttnn
from ....utils.test import line_params, ring_params
import time
import os


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology",
    [
        [(2, 4), (2, 4), 0, 1, 1, True, line_params, ttnn.Topology.Linear],
        # WH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params, ttnn.Topology.Ring],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear],
    ],
    ids=[
        "2x4sp0tp1",
        "wh_4x8sp1tp0",
        "bh_4x8sp1tp0",
    ],
    indirect=["mesh_device", "device_params"],
)
def test_stability(mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    vae_parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    # Test parameters
    prompts = [
        "A massive, muscular sasquatch with long dark fur shreds an electric guitar on a forest stage at night, surrounded by glowing red and blue spotlights, smoke, and flickering firelight. The sasquatch headbangs with wild energy as the camera circles around, capturing sparks flying from the guitar strings. The scene has cinematic lighting, dynamic motion, and a sense of raw power — like a heavy-metal music video filmed in the wilderness. Ultra-detailed, realistic fur and lighting, cinematic camera movement, 4K, high contrast, volumetric fog, dramatic atmosphere.",
        "A massive, muscular sasquatch with long dark fur shreds an electric guitar on a forest stage at night, surrounded by glowing red and blue spotlights, smoke, and flickering firelight. The sasquatch headbangs with wild energy as the camera circles around, capturing sparks flying from the guitar strings. Ultra-realistic lighting, dynamic motion, detailed fur, cinematic depth of field, high-contrast 4K film look, volumetric fog and lens flares, intense heavy-metal atmosphere.",
        "A sasquatch dressed in a vintage band outfit plays heavy metal guitar in a perfectly symmetrical forest stage. Pastel lighting, retro props, and quirky audience of woodland animals nodding to the beat. Flat composition, symmetrical camera framing, warm color palette, meticulous set design, whimsical tone, shot on 35mm film — a Wes Anderson–style heavy metal performance.",
        "An anime-style sasquatch stands on a mountain stage, playing an electric guitar surrounded by streaks of lightning and flying embers. Dynamic camera sweeps, exaggerated hair motion, glowing guitar aura, dramatic speed lines, and stylized smoke. Ultra-fluid animation, high-saturation cel shading, glowing particle effects, shōnen anime intensity, cinematic anime style.",
        "A comic-book-style sasquatch rocks out on a flaming forest stage, electric guitar shooting neon energy waves through the trees. Thick black outlines, halftone shading, bold colors, stylized motion lines, and freeze-frame impacts synced to the beat. Looks like a page come to life from a graphic novel, high contrast, dynamic comic energy.",
        "A futuristic cyberpunk sasquatch performs heavy metal guitar on a neon-lit rooftop under pouring rain. Holographic lights pulse to the rhythm, rain sparks against the amplifier. Chrome textures, glowing tattoos, and electric fur reflections. Camera pans with cinematic speed ramps, high-tech dystopian atmosphere, rain, fog, and neon reflections on wet metal.",
        "A towering sasquatch bard wields an enchanted electric guitar made of dragon bone, playing heavy metal riffs that summon lightning through an ancient forest. Magic runes glow with each chord, camera pans through swirling mist and embers. Epic fantasy aesthetic, dynamic lighting, glowing magic effects, volumetric atmosphere, orchestral-metal fusion tone.",
        "A massive, muscular sasquatch with long dark fur shreds an electric guitar on a forest stage at night. Behind him, a bald eagle wearing aviator sunglasses and smoking a cigar pounds on a glowing drum kit with fierce precision. Red and blue spotlights sweep across the smoky clearing, sparks fly from the guitar strings, and the drums flash with each beat. The camera circles dynamically, capturing fur detail, reflections, and the eagle’s feathers ruffling in the wind. Ultra-realistic lighting, cinematic lens flares, volumetric fog, shallow depth of field, 4K film grain, high-contrast heavy-metal concert energy.",
    ]
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    height = 480
    width = 832
    num_frames = 81
    num_inference_steps = 40

    os.makedirs("wan_outputs", exist_ok=True)
    duration_seconds = 5 * 60 * 60
    start_time = time.time()
    iteration = 0
    print(f"Starting stability loop for up to {duration_seconds // 3600} hours with {len(prompts)} prompts")

    pipeline = WanPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        vae_parallel_config=vae_parallel_config,
        num_links=num_links,
        use_cache=True,
        boundary_ratio=0.875,
        dynamic_load=dynamic_load,
        topology=topology,
    )

    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration_seconds:
            print("Reached time limit. Ending stability loop.")
            break

        for prompt_idx, prompt in enumerate(prompts):
            if time.time() - start_time >= duration_seconds:
                print("Reached time limit. Ending stability loop.")
                break

            print(
                f"Iteration {iteration}, elapsed minutes: {elapsed // 60:.0f}, Prompt {prompt_idx + 1}/{len(prompts)}: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}'"
            )
            print(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

            # Run inference
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=3.0,
                    guidance_scale_2=4.0,
                )

            # Check output
            if hasattr(result, "frames"):
                frames = result.frames
            else:
                frames = result[0] if isinstance(result, tuple) else result

            print("✓ Inference completed successfully")
            print(f"  Output shape: {frames.shape if hasattr(frames, 'shape') else 'Unknown'}")

            # Save video using diffusers utility
            # Remove batch dimension
            frames_to_save = frames[0]
            out_path = os.path.join(
                "wan_outputs",
                f"wan_stability_prompt_{prompt_idx}_iter{iteration}.mp4",
            )
            export_to_video(frames_to_save, out_path, fps=16)
            print(f"✓ Saved video to: {out_path}")

            iteration += 1
