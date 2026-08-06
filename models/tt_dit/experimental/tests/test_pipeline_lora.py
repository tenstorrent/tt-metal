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
import statistics
from typing import List, Tuple

import numpy as np
import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.experimental.pipelines.pipeline_wan_lora import LoRASpec, WanPipelineI2VLora
from models.tt_dit.pipelines.events import profiler_event_callback
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt
from models.tt_dit.utils.test import line_params, ring_params_8k

# Only the 4x32 row traces, so only it needs the trace region carved out of DRAM.
DEVICE_PARAMS = {"trace_region_size": 150000000}


def _rank() -> int:
    return int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0


def create_fractal_image(width: int, height: int) -> PIL.Image.Image:
    c = np.linspace(-2.0, 1.0, width)[None, :] + 1j * np.linspace(-1.5, 1.5, height)[:, None]
    z = np.zeros_like(c)
    img = np.zeros(c.shape, dtype=np.uint8)
    for i in range(32):
        z = z * z + c
        img[(img == 0) & (np.abs(z) > 2)] = 255 - 8 * i
    return PIL.Image.fromarray(np.dstack((img, np.roll(img, width // 10, 1), np.roll(img, height // 10, 0))), "RGB")


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
        [(4, 8), (4, 8), 2, False, ring_params_8k, ttnn.Topology.Ring, False],
        # ring_params_8k adds the fabric router config test_performance_wan uses at 4x32.
        # [(4, 32), (4, 32), 2, False, {**DEVICE_PARAMS, **ring_params}, ttnn.Topology.Ring, False],
        [(4, 32), (4, 32), 2, False, {**DEVICE_PARAMS, **ring_params_8k}, ttnn.Topology.Ring, False],
    ],
    ids=["bh_2x4sp1tp0", "bh_4x8sp1tp0_ring", "bh_4x32sp1tp0_ring"],
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

    # Trace on 4x32 only, matching test_performance_wan; TEST_TRACED overrides either way.
    traced = True if mesh_shape == (4, 32) else False
    logger.info(f"traced={traced} (mesh_shape={mesh_shape})")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    prompt_image_path = os.environ.get("PROMPT_IMAGE")
    pil_image = PIL.Image.open(prompt_image_path) if prompt_image_path else create_fractal_image(width, height)
    image_prompt = [ImagePrompt(image=pil_image, frame_pos=0)]
    negative_prompt = ""

    num_frames = 81
    num_inference_steps = int(os.environ.get("NUM_STEPS", "4"))
    guidance_scale = float(os.environ.get("GUIDANCE_SCALE", "1"))
    guidance_scale_2 = float(os.environ.get("GUIDANCE_SCALE_2", str(guidance_scale)))
    boundary_ratio = float(os.environ.get("BOUNDARY_RATIO", "0.875"))

    # A scale of 1.0 is a no-op lerp, so the unconditional pass is pure waste: skip CFG
    # entirely and halve the denoising cost. Guidance-distilled LoRAs want exactly this.
    cfg_enabled = guidance_scale > 1 or guidance_scale_2 > 1
    logger.info(f"cfg_enabled={cfg_enabled} (guidance_scale={guidance_scale}, guidance_scale_2={guidance_scale_2})")

    pipeline = WanPipelineI2VLora(
        device=mesh_device,
        config=WanPipelineI2VLora.default_config(
            mesh_device=mesh_device,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_enabled=cfg_enabled,
            config_overrides={
                "num_links": num_links,
                "dynamic_load": dynamic_load,
                "topology": topology,
                "is_fsdp": is_fsdp,
                "boundary_ratio": boundary_ratio,
            },
        ),
        lora_high=lora_high if lora_high else None,
        lora_low=lora_low if lora_low else None,
    )

    # prompt = os.environ.get("PROMPT", "A golden retriever running on a sandy beach, waves in the background")
    prompt = os.environ.get("PROMPT", "A  cat running on a sandy beach, waves in the background")

    profiler = BenchmarkProfiler()

    def run(*, prompt, number, seed, profiler_iteration=None):
        logger.info(f"Running LoRA inference with prompt: '{prompt}'")
        logger.info(
            f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps, "
            f"scale={scale}, stack_high={len(lora_high)}, stack_low={len(lora_low)}"
        )

        call_kwargs = dict(
            prompts=[prompt],
            image_prompt=image_prompt,
            negative_prompts=[negative_prompt],
            num_inference_steps=num_inference_steps,
            seed=seed,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            output_type="uint8",
            traced=traced,
        )
        if profiler_iteration is not None:
            call_kwargs["on_event"] = profiler_event_callback(profiler, profiler_iteration)

        with torch.no_grad():
            result = pipeline(**call_kwargs)

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
        # Every rank holds the same gathered video, so only rank 0 writes it out.
        if _rank() == 0:
            output_filename = f"wan_lora_i2v_{width}x{height}_{number}.mp4"
            try:
                from models.tt_dit.utils.video import export_to_video

                export_to_video(frames, output_filename, fps=16)
                logger.info(f"Saved video to: {output_filename}")
            except ImportError:
                logger.info("Could not export video - imageio_ffmpeg not available")
        else:
            logger.info(f"Skipping video export on rank {_rank()}")

        return frames

    if no_prompt:
        # Warmup run (not included in stats)
        logger.info("Running warmup iteration...")
        with profiler("warmup", iteration=0):
            run(prompt=prompt, number=-1, seed=0)
        logger.info(f"Warmup completed in {profiler.get_duration('warmup', 0):.2f}s")

        # Timed performance run
        num_perf_runs = 1
        ttnn.synchronize_device(mesh_device)
        # Line up all hosts so the measured window is comparable across ranks.
        ttnn.distributed_context_barrier()

        for i in range(num_perf_runs):
            logger.info(f"Performance run {i+1}/{num_perf_runs}...")
            with profiler("run", iteration=i):
                run(prompt=prompt, number=i, seed=42, profiler_iteration=i)
            logger.info(f"  Run {i+1} completed in {profiler.get_duration('run', i):.2f}s")

        # Collect timing stats. "prepare_latents" is the I2V image-conditioning
        # section: host preprocess/upload of the conditioned frames plus the VAE encode.
        encoder_times = [profiler.get_duration("encoder", i) for i in range(num_perf_runs)]
        image_encode_times = [profiler.get_duration("prepare_latents", i) for i in range(num_perf_runs)]
        denoising_times = [profiler.get_duration("denoising", i) for i in range(num_perf_runs)]
        vae_times = [profiler.get_duration("vae", i) for i in range(num_perf_runs)]
        total_times = [profiler.get_duration("run", i) for i in range(num_perf_runs)]

        def fmt_stats(times):
            if not times:
                return "No data available"
            mean = statistics.mean(times)
            if len(times) > 1:
                std = statistics.stdev(times)
                return f"Mean: {mean:8.3f}s | Std: {std:7.3f}s | Min: {min(times):8.3f}s | Max: {max(times):8.3f}s"
            return f"{mean:8.3f}s"

        if _rank() == 0:
            sp_factor = tuple(mesh_device.shape)[1]
            tp_factor = tuple(mesh_device.shape)[0]
            print("\n" + "=" * 80)
            print("WAN 2.2 I2V LoRA — PERFORMANCE RESULTS")
            print("=" * 80)
            print(f"  Resolution:      {width}x{height}")
            print(f"  Frames:          {num_frames}")
            print(f"  Denoising steps: {num_inference_steps}")
            print(f"  Mesh shape:      {tuple(mesh_device.shape)}")
            print(f"  SP factor:       {sp_factor}  TP factor: {tp_factor}")
            print(f"  Topology:        {topology}")
            print(f"  FSDP:            {is_fsdp}")
            print(f"  Num links:       {num_links}")
            print(f"  LoRA high:       {len(lora_high)} adapter(s)")
            print(f"  LoRA low:        {len(lora_low)} adapter(s)")
            print(f"  Perf runs:       {num_perf_runs}")
            print("-" * 80)
            print(f"  {'Text Encoding':25s} | {fmt_stats(encoder_times)}")
            print(f"  {'Image Encoding (total)':25s} | {fmt_stats(image_encode_times)}")
            print(f"  {'Denoising':25s} | {fmt_stats(denoising_times)}")
            print(f"  {'VAE Decoding':25s} | {fmt_stats(vae_times)}")
            print(f"  {'Total Pipeline':25s} | {fmt_stats(total_times)}")
            print("=" * 80)
        else:
            logger.info(f"Skipping performance summary on rank {_rank()}")
    else:
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)

    if traced:
        # Release before the mesh fixture tears down, as test_performance_wan does.
        pipeline.release_traces()
