# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import statistics

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.events import profiler_event_callback
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V
from models.tt_dit.pipelines.wan.quant_config import QuantConfig, set_quant_config
from models.tt_dit.utils import vbench_bundle
from models.tt_dit.utils.video import export_to_video

from ....utils.test import (
    is_global_rank_zero,
    line_params_req_exact_devices,
    ring_params_8k_req_exact_devices,
    ring_params_req_exact_devices,
)
from .common import check_timings, create_fractal_image, timing_limits

DEVICE_PARAMS = {"trace_region_size": 150000000}

# VBench quality gate (RUN_VBENCH=1): every (prompt, seed) clip is rendered after the timed runs, and
# each prompt's mean over the seeds must clear that prompt's floors (models/tt_dit/utils/vbench_bundle.py).
# Indices into the test's prompt list: the boxing cats and the cyberpunk alley, both strongly dynamic.
VBENCH_PROMPTS = [0, 2]
VBENCH_SEEDS = [42, 43, 44, 45]


def vbench_thresholds(model_type: str, height: int) -> dict:
    # Loose "broken pipeline" floors carried over from the single-clip gate (#60488) until the
    # per-prompt spread over these seeds is measured on both bh_sc4 quads (#60481). dynamic_degree is
    # binary per clip, so 0.75 requires three of the four seeds to move.
    if model_type == "i2v":
        # The I2V conditioning image is a fractal: dynamic_degree and imaging_quality are meaningless
        # on it, so only intra-video consistency is gated (see test_pipeline_wan.py).
        return {"subject_consistency": 0.70, "background_consistency": 0.80 if height == 720 else 0.82}
    return {
        "subject_consistency": 0.8,
        "background_consistency": 0.8,
        "motion_smoothness": 0.9,
        "dynamic_degree": 0.75,
        "imaging_quality": 0.5 if height == 720 else 0.45,
    }


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp, quant_config_name",
    [
        # FSDP is needed for 2x2 with encoder now on device
        [(2, 2), (2, 2), 0, 1, 2, False, line_params_req_exact_devices, ttnn.Topology.Linear, True, None],
        [(2, 4), (2, 4), 0, 1, 1, True, line_params_req_exact_devices, ttnn.Topology.Linear, True, None],
        # BH on 2x4 with dynamic_load to avoid init-time DRAM OOM
        [(2, 4), (2, 4), 1, 0, 2, True, line_params_req_exact_devices, ttnn.Topology.Linear, False, None],
        # WH on 4x8
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params_req_exact_devices, ttnn.Topology.Ring, True, None],
        # BH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, ring_params_8k_req_exact_devices, ttnn.Topology.Ring, False, None],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, line_params_req_exact_devices, ttnn.Topology.Linear, False, None],
        [
            (4, 32),
            (4, 32),
            1,
            0,
            2,
            False,
            {**DEVICE_PARAMS, **ring_params_8k_req_exact_devices},
            ttnn.Topology.Ring,
            False,
            None,
        ],
        # FSDP on 2x4 with bf8 weights+activations, LoFi linear, bf8 HiFi2 SDPA
        [(2, 4), (2, 4), 0, 1, 1, True, line_params_req_exact_devices, ttnn.Topology.Linear, True, "all_bf8_lofi"],
    ],
    ids=[
        "2x2_sp0tp1",
        "2x4_sp0tp1",
        "bh_2x4_sp1tp0",
        "wh_4x8_sp1tp0",
        "ring_bh_4x8_sp1tp0",
        "line_bh_4x8_sp1tp0",
        "bh_4x32sp1tp0",
        "2x4_sp0tp1_bf8_lofi",
    ],
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
@pytest.mark.parametrize(
    "model_type",
    ["t2v", "i2v"],
    ids=[
        "t2v",
        "i2v",
    ],
)
def test_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple,
    model_type: str,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    dynamic_load: dict,
    topology: ttnn.Topology,
    width: int,
    height: int,
    is_ci_env: bool,
    galaxy_type: str,
    is_fsdp: bool,
    quant_config_name: str | None,
    tmp_path,
) -> None:
    """Performance test for Wan pipeline with detailed timing analysis."""

    benchmark_profiler = BenchmarkProfiler()
    traced = mesh_shape == (4, 32)  # trace only for quadx32
    # VBENCH_EXPORT_DIR leaves the bundle for a CI step that scores it after pytest has closed the
    # devices; without it rank 0 scores the bundle at the end of the test.
    run_vbench = os.environ.get("RUN_VBENCH", "0") == "1"
    vbench_export_dir = os.environ.get("VBENCH_EXPORT_DIR")
    vbench_dir = vbench_export_dir or tmp_path / "vbench"

    # Skip 4U.
    if galaxy_type == "4U":
        # NOTE: Pipelines fail if a performance test is skipped without providing a benchmark output.
        if is_ci_env and is_global_rank_zero():
            with benchmark_profiler("run", iteration=0):
                pass

            benchmark_data = BenchmarkData()
            benchmark_data.save_partial_run_json(
                benchmark_profiler,
                run_type="empty_run",
                ml_model_name="empty_run",
            )
        pytest.skip("4U is not supported for this test")

    if (not is_fsdp) and (not ttnn.device.is_blackhole()):
        pytest.skip("FSDP=False unsupported on non-blackhole systems due to memory constraints")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Test prompts
    prompts = [
        """Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.""",
        """A close-up of a beautiful butterfly landing on a flower, wings gently moving in the breeze.""",
        """A neon-lit alley in a sprawling cyberpunk metropolis at night, rain-slick streets reflecting glowing holograms, dense atmosphere, flying cars in the sky, people in high-tech streetwear — ultra-detailed, cinematic lighting, 4K""",
        """A colossal whale floating through a desert sky like a blimp, casting a long shadow over sand dunes, people in ancient robes watching in awe, golden hour lighting, dreamlike color palette — surrealism, concept art, Greg Rutkowski style""",
        """A Roman general standing on a battlefield at dawn, torn red cape blowing in the wind, distant soldiers forming ranks, painterly brushwork in the style of Caravaggio, chiaroscuro lighting, epic composition""",
        """An epic, high-definition cinematic shot of a rustic snowy cabin glowing warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle snow-covered pines and delicate falling snowflakes — captured in a rich, atmospheric, wide-angle scene with deep cinematic depth and warmth.""",
    ]

    num_frames = 81
    num_inference_steps = 40

    print(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

    limits = timing_limits(model_type, mesh_device.shape, height, topology)
    assert limits is not None, f"No timing limits for {model_type} on {mesh_device.shape} at {height}p"
    # Upper limits double as the CI report targets.
    expected_metrics = {k: upper for k, (_, upper) in limits.items()}
    if model_type == "i2v":
        pipeline_cls = WanPipelineI2V
        image_prompt = create_fractal_image(width, height)
    else:
        pipeline_cls = WanPipeline
        image_prompt = None

    h_factor = tuple(mesh_device.shape)[tp_axis]
    w_factor = tuple(mesh_device.shape)[sp_axis]
    parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(w_factor, sp_axis), tp=(h_factor, tp_axis))
    vae_parallel_config = VaeHWParallelConfig.from_tuples(height=(h_factor, tp_axis), width=(w_factor, sp_axis))
    encoder_parallel_config = EncoderParallelConfig.from_tuple((h_factor, tp_axis))

    pipeline = pipeline_cls(
        device=mesh_device,
        config=WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            dit_parallel_config=parallel_config,
            vae_parallel_config=vae_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            num_links=num_links,
            dynamic_load=dynamic_load,
            topology=topology,
            is_fsdp=is_fsdp,
            model_type=model_type,
            checkpoint_name=(
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers" if model_type == "i2v" else "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
            ),
            height=height,
            width=width,
            num_frames=num_frames,
        ),
    )

    if quant_config_name is not None:
        qc = getattr(QuantConfig, quant_config_name)()
        set_quant_config(pipeline, qc)

    # Warmup run (not timed)
    logger.info("Running warmup iteration...")

    with benchmark_profiler("run", iteration=0):
        if traced:
            with torch.no_grad():
                pipeline(
                    prompts=[prompts[0]],
                    image_prompt=image_prompt,
                    num_inference_steps=2,  # Small number of steps to reduce test time.
                    traced=traced,
                )

    logger.info(f"Warmup completed in {benchmark_profiler.get_duration('run', 0):.2f}s")

    # Performance measurement runs
    logger.info("Running performance measurement iterations...")
    num_perf_runs = 1  # For now use 1 prompt to minimize test time.

    ttnn.synchronize_device(mesh_device)
    ttnn.distributed_context_barrier()

    for i in range(num_perf_runs):
        logger.info(f"Performance run {i+1}/{num_perf_runs}...")

        # Run pipeline with different prompt
        prompt_idx = (i + 1) % len(prompts)
        with benchmark_profiler("run", iteration=i):
            with torch.no_grad():
                frames = pipeline(
                    prompts=[prompts[prompt_idx]],
                    image_prompt=image_prompt,
                    num_inference_steps=num_inference_steps,
                    on_event=profiler_event_callback(benchmark_profiler, i),
                    seed=42,
                    traced=traced,
                    output_type="uint8",
                )
                ttnn.synchronize_device(mesh_device)
        logger.info(f"  Run {i+1} completed in {benchmark_profiler.get_duration('run', i):.2f}s")
        # Check output

    # Gate clips are timed too: each prompt's mean over its seeds must also fall within that prompt's
    # timing limits (for now the same limits as the timed run).
    gate_prompts = []
    gate_timings = []
    if run_vbench:
        gate_prompts = [
            {"text": prompts[i], "thresholds": vbench_thresholds(model_type, height), "timing": limits}
            for i in VBENCH_PROMPTS
        ]
        gate_profiler = BenchmarkProfiler()
        if is_global_rank_zero():
            vbench_bundle.create_bundle(vbench_dir)
        for index, gate_prompt in enumerate(gate_prompts):
            iterations = range(index * len(VBENCH_SEEDS), (index + 1) * len(VBENCH_SEEDS))
            for iteration, seed in zip(iterations, VBENCH_SEEDS):
                logger.info(f"VBench clip: prompt {index}, seed {seed}")
                with gate_profiler("run", iteration=iteration):
                    with torch.no_grad():
                        clip = pipeline(
                            prompts=[gate_prompt["text"]],
                            image_prompt=image_prompt,
                            num_inference_steps=num_inference_steps,
                            on_event=profiler_event_callback(gate_profiler, iteration),
                            seed=seed,
                            traced=traced,
                            output_type="uint8",
                        )
                        ttnn.synchronize_device(mesh_device)
                logger.info(f"  completed in {gate_profiler.get_duration('run', iteration):.2f}s")
                if is_global_rank_zero():
                    export_to_video(clip[0], str(vbench_bundle.clip_path(vbench_dir, index, seed)), fps=16)
            gate_timings.append(
                {
                    name: statistics.mean(gate_profiler.get_duration(step, i) for i in iterations)
                    for name, step in (
                        ("encoder", "encoder"),
                        ("denoising", "denoising"),
                        ("vae", "vae"),
                        ("total", "run"),
                    )
                }
            )
        if is_global_rank_zero():
            vbench_bundle.write_manifest(
                vbench_dir,
                prompts=[{"text": p["text"], "thresholds": p["thresholds"]} for p in gate_prompts],
                seeds=VBENCH_SEEDS,
            )

    pipeline.release_traces()

    print(f"✓ Inference completed successfully")
    print(f"  Output shape: {frames.shape if hasattr(frames, 'shape') else 'Unknown'}")
    print(f"  Output type: {type(frames)}")

    # Basic validation
    if isinstance(frames, np.ndarray):
        print(f"  Video dtype:      {frames.dtype}")
        print(f"  Video data range: [{frames.min()}, {frames.max()}]")
    elif isinstance(frames, torch.Tensor):
        print(f"  Video dtype:      {frames.dtype}")
        print(f"  Video data range: [{frames.min().item()}, {frames.max().item()}]")

    # Save video
    # Remove batch dimension
    frames = frames[0]
    if not is_ci_env:
        if int(ttnn.distributed_context_get_rank()) == 0:
            output_path = f"wan_output_video_{model_type}{'_traced' if traced else ''}.mp4"
            try:
                export_to_video(frames, output_path, fps=16)
                print(f"✓ Saved video to: {output_path}")
            except ImportError:
                print("Could not export video - imageio_ffmpeg not available")
        else:
            print(f"Skipping video export on rank {ttnn.distributed_context_get_rank()}")

    # Calculate statistics
    text_encoder_times = [benchmark_profiler.get_duration("encoder", i) for i in range(num_perf_runs)]
    prepare_latents_times = [benchmark_profiler.get_duration("prepare_latents", i) for i in range(num_perf_runs)]
    vae_encode_times = (
        [benchmark_profiler.get_duration("vae_encode", i) for i in range(num_perf_runs)]
        if model_type == "i2v" and benchmark_profiler.contains_step("vae_encode")
        else []
    )
    denoising_times = [benchmark_profiler.get_duration("denoising", i) for i in range(num_perf_runs)]
    vae_times = [benchmark_profiler.get_duration("vae", i) for i in range(num_perf_runs)]
    total_times = [benchmark_profiler.get_duration("run", i) for i in range(num_perf_runs)]

    # Report results
    print("\n" + "=" * 80)
    print("WAN PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Image Size: {width}x{height}")
    print(f"Inference Steps: {num_inference_steps}")
    print(f"Num Frames: {num_frames}")
    print(f"DiT Configuration: sp={sp_factor}, tp={tp_factor}")
    print(f"Mesh Shape: {mesh_device.shape}")
    print(f"Topology: {topology}")
    print("-" * 80)

    def print_stats(name, times):
        if not times:
            print(f"{name:25} | No data available")
            return
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        print(
            f"{name:25} | Mean: {mean_time:8.4f}s | Std: {std_time:8.4f}s | Min: {min_time:8.4f}s | Max: {max_time:8.4f}s"
        )

    print_stats("Text Encoding", text_encoder_times)
    if model_type == "i2v":
        print_stats("Image Encoding (total)", prepare_latents_times)
        print_stats("  VAE Encoder only", vae_encode_times)
    print_stats("Denoising", denoising_times)
    print_stats("VAE Decoding", vae_times)
    print_stats("Total Pipeline", total_times)
    for index, timing in enumerate(gate_timings):
        print(
            f"VBench prompt {index} (mean of {len(VBENCH_SEEDS)} seeds) | "
            + " | ".join(f"{name}: {seconds:8.4f}s" for name, seconds in timing.items())
        )
    print("-" * 80)

    # Validate performance
    measurements = {
        "encoder": statistics.mean(text_encoder_times),
        "denoising": statistics.mean(denoising_times),
        "vae": statistics.mean(vae_times),
        "total": statistics.mean(total_times),
    }

    if is_ci_env and is_global_rank_zero():
        # In CI, dump a performance report from rank 0 only: all ranks time the same
        # collective run, and concurrent saves race on the shared output file.
        benchmark_data = BenchmarkData()
        for iteration in range(num_perf_runs):
            for step_name in ["encoder", "denoising", "vae", "run"]:
                benchmark_data.add_measurement(
                    profiler=benchmark_profiler,
                    iteration=iteration,
                    step_name=step_name,
                    name=step_name,
                    value=benchmark_profiler.get_duration(step_name, iteration),
                    target=expected_metrics["total" if step_name == "run" else step_name],
                )
        device_name_map = {
            (2, 2): "BH_QB",
            (2, 4): "BH_LB" if is_blackhole() else "WH_T3K",
            (4, 8): "BH_GLX" if is_blackhole() else "WH_GLX",
            (4, 32): "BH_QG",
        }
        benchmark_profiler.start("inference")
        benchmark_profiler.end("inference")
        benchmark_data.add_measurement(benchmark_profiler, 0, "inference", "fps", num_frames / total_times[-1])
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type=device_name_map[mesh_shape] + ("_quant" if quant_config_name else ""),
            ml_model_name="wan-2.2-i2v" if model_type == "i2v" else "wan-2.2-t2v",
            batch_size=1,
            config_params={
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_steps": num_inference_steps,
                "sp_factor": sp_factor,
                "tp_factor": tp_factor,
                "topology": str(topology),
                "num_links": num_links,
                "fsdp": is_fsdp,
            },
        )

    assert_msgs = check_timings(measurements, limits, "timed run")
    for index, (gate_prompt, timing) in enumerate(zip(gate_prompts, gate_timings)):
        assert_msgs += check_timings(timing, gate_prompt["timing"], f"VBench prompt {index} mean")

    assert not assert_msgs, "\n".join(assert_msgs)

    if run_vbench and not vbench_export_dir and is_global_rank_zero():
        vbench_bundle.evaluate(vbench_dir)

    logger.info("Performance test completed successfully!")
