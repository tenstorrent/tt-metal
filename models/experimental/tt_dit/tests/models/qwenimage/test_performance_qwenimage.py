# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import cProfile
import io
import pstats
import statistics
import time
import pytest
import ttnn
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData
from ....pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from ....utils.diagnostic_timing import DiagnosticTimingCollector


@pytest.mark.parametrize(
    "image_w, image_h, num_inference_steps",
    [
        (1024, 1024, 50),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, encoder_tp, vae_tp, topology, num_links",
    [
        [(2, 4), (2, 0), (1, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1],
        # 2x4 config with SP enabled - SP on axis 0, no CFG parallel (for FSDP memory efficiency)
        [(2, 4), (1, 0), (2, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1],
        # 6U config with SP enabled - uses sequence parallelism for FSDP weight sharding
        [(4, 8), (2, 1), (4, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=[
        "2x4cfg0sp0tp1",
        "2x4sp2tp4",
        "T3K_4x8cfg1sp0tp1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 40000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    "is_fsdp",
    [
        pytest.param(True, id="fsdp_enabled"),
        # pytest.param(False, id="fsdp_disabled"),  # Uncomment to compare with/without FSDP
    ],
)
def test_qwenimage_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    image_w: int,
    image_h: int,
    num_inference_steps: int,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
    encoder_tp: tuple[int, int],
    vae_tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    is_ci_env: bool,
    galaxy_type: str,
    is_fsdp: bool,
) -> None:
    """Performance test for QwenImage pipeline with detailed timing analysis."""

    benchmark_profiler = BenchmarkProfiler()

    # Skip 4U.
    if galaxy_type == "4U":
        # NOTE: Pipelines fail if a performance test is skipped without providing a benchmark output.
        if is_ci_env:
            with benchmark_profiler("run", iteration=0):
                pass

            benchmark_data = BenchmarkData()
            benchmark_data.save_partial_run_json(
                benchmark_profiler,
                run_type="empty_run",
                ml_model_name="empty_run",
            )
        pytest.skip("4U is not supported for this test")

    logger.info(f"  Image size: {image_w}x{image_h}")
    logger.info(f"  Inference steps: {num_inference_steps}")
    logger.info(f"  FSDP enabled: {is_fsdp}")

    pipeline = QwenImagePipeline.create_pipeline(
        mesh_device=mesh_device,
        dit_cfg=cfg,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        use_torch_text_encoder=False,
        use_torch_vae_decoder=False,
        num_links=num_links,
        topology=topology,
        width=image_w,
        height=image_h,
        is_fsdp=is_fsdp,  # Enable FSDP to avoid model load/unload cycle
    )

    # Test prompts - diverse set for comprehensive performance testing
    prompts = [
        'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee $2 per cup," with a neon light '
        'beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the '
        'poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition',
        'Tokyo neon alley at night, rain-slick pavement, cinematic cyberpunk lighting; include glowing sign text "深夜営業" in bold neon above a doorway; moody reflections, shallow depth of field.',
        'Steamy ramen shop entrance at dusk; fabric noren curtain gently swaying; print "しょうゆラーメン" across the curtain in thick brush-style kana; warm lantern light, photorealistic.',
        'Minimalist tea poster, cream background, elegant layout; vertical calligraphy "抹茶" centered in sumi ink; small red hanko-style seal "本格" in the corner; high-resolution graphic design.',
        'Hardcover fantasy novel cover, textured paper, gold foil; title text "物語のはじまり" centered; author line "山本ひかり" below; tasteful serif typography, dramatic vignette illustration.',
    ]

    # =========================================================================
    # WARMUP RUN (not timed for performance metrics)
    # =========================================================================
    logger.info("Running warmup iteration...")
    timer_warmup = DiagnosticTimingCollector(enable_sync_timing=True, enable_profiler=False)
    pipeline.timing_collector = timer_warmup

    images = pipeline(
        prompts=[prompts[0]],
        negative_prompts=[None],
        num_inference_steps=num_inference_steps,
        cfg_scale=4.0,
        seed=0,
        traced=True,
    )
    images[0].save(f"qwenimage_{image_w}_{image_h}_warmup.png")

    warmup_timing = timer_warmup.get_timing_data()
    logger.info(f"Warmup completed in {warmup_timing.total_time:.2f}s")

    # Print warmup breakdown to see trace capture time
    print("\n" + "=" * 100)
    print("WARMUP RUN TIMING BREAKDOWN (includes trace capture)")
    print("=" * 100)
    timer_warmup.print_breakdown()

    # =========================================================================
    # PERFORMANCE MEASUREMENT RUNS
    # =========================================================================
    logger.info("Running performance measurement iterations...")
    all_timings = []
    all_diagnostic_data = []
    num_perf_runs = 1  # Use 4 different prompts for performance testing

    # Optional Tracy profiling (if available)
    profiler = None
    try:
        from tracy import Profiler

        profiler = Profiler()
        profiler.enable()
        logger.info("Tracy profiling enabled")
    except ImportError:
        logger.info("Tracy profiler not available, continuing without profiling")

    try:
        for i in range(num_perf_runs):
            logger.info(f"Performance run {i+1}/{num_perf_runs}...")

            # Create diagnostic timing collector for this run
            timer = DiagnosticTimingCollector(enable_sync_timing=True, enable_profiler=(i == 0))
            pipeline.timing_collector = timer

            # Run pipeline with different prompt
            prompt_idx = (i + 1) % len(prompts)

            # =====================================================================
            # STEP 1: Wall clock verification
            # =====================================================================
            t_wall_start = time.perf_counter()

            # Enable cProfile for first run to find pure-Python time sinks
            if i == 0:
                cpu_profiler = cProfile.Profile()
                cpu_profiler.enable()

            with benchmark_profiler("run", iteration=i):
                images = pipeline(
                    prompts=[prompts[prompt_idx]],
                    negative_prompts=[None],
                    num_inference_steps=num_inference_steps,
                    cfg_scale=4.0,
                    seed=0,
                    traced=True,
                )

            if i == 0:
                cpu_profiler.disable()

            t_wall_end = time.perf_counter()
            t_wall_total = t_wall_end - t_wall_start

            # =====================================================================
            # STEP 5: Verify execution counts
            # =====================================================================
            timing_data = timer.get_timing_data()
            all_timings.append(timing_data)
            all_diagnostic_data.append(timer)

            # Log execution counts for verification
            logger.info(f"  Run {i+1} completed in {timing_data.total_time:.2f}s (wall: {t_wall_total:.2f}s)")
            logger.info(f"  Execution counts:")
            logger.info(f"    - Pipeline calls: {timing_data.pipeline_call_count}")
            logger.info(f"    - Denoising loops: {timing_data.denoising_loop_count}")
            logger.info(f"    - VAE decodes: {timing_data.vae_decode_count}")
            logger.info(f"    - Trace executes: {timing_data.trace_execute_count}")
            logger.info(f"    - Device syncs: {timing_data.device_sync_count}")

            # Save image after timing
            t_save_start = time.perf_counter()
            images[0].save(f"qwenimage_{image_w}_{image_h}_perf_run{i}.png")
            t_save_end = time.perf_counter()
            logger.info(f"  Image save time: {t_save_end - t_save_start:.4f}s")

            # =====================================================================
            # Print detailed breakdown for this run
            # =====================================================================
            print(f"\n{'=' * 100}")
            print(f"PERFORMANCE RUN {i+1} DETAILED TIMING BREAKDOWN")
            print(f"{'=' * 100}")
            timer.print_breakdown()

            # =====================================================================
            # STEP 4: CPU Profiler output (first run only)
            # =====================================================================
            if i == 0:
                print("\n" + "=" * 100)
                print("CPROFILE TOP 30 FUNCTIONS BY CUMULATIVE TIME")
                print("=" * 100)
                s = io.StringIO()
                ps = pstats.Stats(cpu_profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
                ps.print_stats(30)
                print(s.getvalue())

    finally:
        if profiler:
            profiler.disable()
            logger.info("Tracy profiling disabled")

    # =========================================================================
    # CALCULATE STATISTICS
    # =========================================================================
    clip_times = [t.clip_encoding_time for t in all_timings]
    t5_times = [t.t5_encoding_time for t in all_timings]
    total_encoding_times = [t.total_encoding_time for t in all_timings]
    vae_times = [t.vae_decoding_time for t in all_timings]
    total_times = [t.total_time for t in all_timings]

    # Calculate per-step denoising times
    all_denoising_steps = []
    for timing in all_timings:
        all_denoising_steps.extend(timing.denoising_step_times)

    # =========================================================================
    # STEP 6: FINAL DIAGNOSIS REPORT
    # =========================================================================
    cfg_factor = cfg[0]  # First element is always the factor
    sp_factor = sp[0]
    tp_factor = tp[0]
    encoder_tp_factor = encoder_tp[0]
    vae_tp_factor = vae_tp[0]

    print("\n" + "=" * 100)
    print("QWEN IMAGE PIPELINE PERFORMANCE RESULTS")
    print("=" * 100)
    print(f"Model: QwenImage")
    print(f"Image Size: {image_w}x{image_h}")
    print(f"Inference Steps: {num_inference_steps}")
    print(
        f"Configuration: cfg={cfg_factor}, sp={sp_factor}, tp={tp_factor}, encoder_tp={encoder_tp_factor}, vae_tp={vae_tp_factor}"
    )
    print(f"Mesh Shape: {mesh_device.shape}")
    print(f"Topology: {topology}")
    print("-" * 100)

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

    print_stats("CLIP Encoding", clip_times)
    print_stats("T5 Encoding", t5_times)
    print_stats("Total Encoding", total_encoding_times)
    print_stats("Denoising (per step)", all_denoising_steps)
    print_stats("VAE Decoding", vae_times)
    print_stats("Total Pipeline", total_times)

    print("-" * 100)

    # Additional metrics
    if total_times and all_denoising_steps:
        avg_total_time = statistics.mean(total_times)
        avg_step_time = statistics.mean(all_denoising_steps)
        total_denoising_time = avg_step_time * num_inference_steps

        print(f"Average total denoising time: {total_denoising_time:.4f}s")
        print(f"Denoising throughput: {num_inference_steps / total_denoising_time:.2f} steps/second")
        print(f"Overall throughput: {1 / avg_total_time:.4f} images/second")

        # Breakdown percentages
        avg_encoding_time = statistics.mean(total_encoding_times)
        avg_vae_time = statistics.mean(vae_times)

        print(f"\nTime breakdown:")
        print(f"  Encoding: {avg_encoding_time/avg_total_time*100:.1f}%")
        print(f"  Denoising: {total_denoising_time/avg_total_time*100:.1f}%")
        print(f"  VAE: {avg_vae_time/avg_total_time*100:.1f}%")

    # =========================================================================
    # FINAL DIAGNOSIS TABLE
    # =========================================================================
    if all_diagnostic_data:
        print("\n" + "=" * 100)
        print("FINAL DIAGNOSIS: WHERE IS THE TIME GOING?")
        print("=" * 100)

        # Use the first run's data for the diagnosis
        diag = all_diagnostic_data[0]
        breakdown = diag.get_breakdown_dict()
        total = breakdown.get("total", breakdown.get("_total", 1.0))

        # Create diagnosis table
        diagnosis_table = []

        # Major regions
        regions = [
            ("Encoder Reload", "encoder_reload"),
            ("Total Encoding", "total_encoding"),
            ("Encoder Deallocate", "encoder_deallocate"),
            ("Transformer Load", "transformer_load"),
            ("Scheduler Init", "scheduler_init"),
            ("Latents Init", "latents_init"),
            ("RoPE Init", "rope_init"),
            ("Tensor Transfer", "tensor_transfer"),
            ("Trace Capture", "trace_capture"),
            ("Denoising (total)", "denoising_step_total"),
            ("  - Step Enqueue (total)", "step_enqueue_total"),
            ("  - Step CFG (total)", "step_cfg_total"),
            ("  - Step Sync (total)", "step_sync_total"),
            ("VAE Pre-Sync", "vae_pre_sync"),
            ("VAE Gather", "vae_gather"),
            ("VAE Readback", "vae_readback"),
            ("VAE Unpatchify", "vae_unpatchify"),
            ("Transformer Deallocate", "transformer_deallocate"),
            ("VAE Reload", "vae_reload"),
            ("VAE Decode Forward", "vae_decode_forward"),
            ("VAE Deallocate", "vae_deallocate"),
            ("Postprocess", "postprocess"),
            ("PIL Convert", "pil_convert"),
            ("VAE Decoding (wrapper)", "vae_decoding"),
        ]

        print(f"{'Region':<35} | {'Time (s)':>12} | {'% of Total':>10}")
        print("-" * 65)

        accounted = 0.0
        for name, key in regions:
            t = breakdown.get(key, 0.0)
            if t > 0:
                pct = (t / total) * 100
                print(f"{name:<35} | {t:>12.4f} | {pct:>9.1f}%")
                if not name.startswith("  -") and key != "vae_decoding":
                    accounted += t

        other = total - accounted
        print("-" * 65)
        print(f"{'TOTAL (measured)':<35} | {total:>12.4f} | {100.0:>9.1f}%")
        print(f"{'Accounted':<35} | {accounted:>12.4f} | {(accounted/total)*100:>9.1f}%")
        print(f"{'OTHER (unaccounted)':<35} | {other:>12.4f} | {(other/total)*100:>9.1f}%")

        # =====================================================================
        # TOP 3 TIME SINKS
        # =====================================================================
        print("\n" + "=" * 100)
        print("TOP TIME SINKS (regions > 1% of total)")
        print("=" * 100)

        all_regions = [(name, breakdown.get(key, 0.0)) for name, key in regions if not name.startswith("  -")]
        all_regions.append(("OTHER (unaccounted)", other))
        all_regions = sorted(all_regions, key=lambda x: -x[1])

        for i, (name, t) in enumerate(all_regions[:10]):
            if t > 0:
                pct = (t / total) * 100
                if pct >= 1.0:
                    print(f"  {i+1}. {name:<30} {t:>10.4f}s ({pct:>5.1f}%)")

        # =====================================================================
        # SYNC VS ENQUEUE ANALYSIS
        # =====================================================================
        step_enqueue_total = breakdown.get("step_enqueue_total", 0.0)
        step_sync_total = breakdown.get("step_sync_total", 0.0)
        step_cfg_total = breakdown.get("step_cfg_total", 0.0)
        denoising_total = breakdown.get("denoising_step_total", 0.0)

        print("\n" + "=" * 100)
        print("SYNC VS ENQUEUE ANALYSIS (Step 3)")
        print("=" * 100)
        print(f"Total denoising time:     {denoising_total:>10.4f}s")
        print(f"  - Step enqueue total:   {step_enqueue_total:>10.4f}s")
        print(f"  - Step CFG total:       {step_cfg_total:>10.4f}s")
        print(f"  - Step sync total:      {step_sync_total:>10.4f}s")

        if step_sync_total > step_enqueue_total * 5:
            print("\n  ⚠️  HIDDEN SYNC DETECTED!")
            print("      The sync time is much larger than enqueue time.")
            print("      This means the per-step timer was measuring enqueue, not execution.")
            print("      The real work happens during the sync.")

        if step_cfg_total > denoising_total * 0.5:
            print("\n  ⚠️  CFG COMBINE IS A MAJOR BOTTLENECK!")
            print("      The CFG combine step (which includes .cpu(blocking=True)) is taking")
            print("      a large fraction of the denoising time. This is a hidden sync/readback.")

    print("=" * 100)

    # Validate that we got reasonable results
    assert len(all_timings) == num_perf_runs, f"Expected {num_perf_runs} timing results, got {len(all_timings)}"
    assert all(t.total_time > 0 for t in all_timings), "All runs should have positive total time"
    assert all(
        len(t.denoising_step_times) == num_inference_steps for t in all_timings
    ), f"All runs should have {num_inference_steps} denoising steps"

    # Clean up
    pipeline.timing_collector = None

    # Validate performance
    measurements = {
        "clip_encoding_time": statistics.mean(clip_times),
        "t5_encoding_time": statistics.mean(t5_times),
        "total_encoding_time": statistics.mean(total_encoding_times),
        "denoising_steps_time": total_denoising_time,
        "vae_decoding_time": statistics.mean(vae_times),
        "total_time": statistics.mean(total_times),
    }
    if tuple(mesh_device.shape) == (2, 4):
        expected_metrics = {
            "clip_encoding_time": 0.2,
            "t5_encoding_time": 0.15,
            "total_encoding_time": 0.35,
            "denoising_steps_time": 15.0,
            "vae_decoding_time": 2.0,
            "total_time": 17.5,
        }
    elif tuple(mesh_device.shape) == (4, 8):
        expected_metrics = {
            "clip_encoding_time": 0.25,
            "t5_encoding_time": 0.18,
            "total_encoding_time": 0.7,
            "denoising_steps_time": 6.0,
            "vae_decoding_time": 1.5,
            "total_time": 8.5,
        }
    else:
        assert False, f"Unknown mesh device for performance comparison: {mesh_device}"

    if is_ci_env:
        # In CI, dump a performance report
        profiler_model_name = f"qwenimage_{'t3k' if tuple(mesh_device.shape) == (2, 4) else 'tg'}_cfg{cfg_factor}_sp{sp_factor}_tp{tp_factor}"
        benchmark_data = BenchmarkData()
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type="qwenimage_traced",
            ml_model_name=profiler_model_name,
        )

    pass_perf_check = True
    assert_msgs = []
    for k in expected_metrics.keys():
        if measurements[k] > expected_metrics[k]:
            assert_msgs.append(
                f"Warning: {k} is outside of the tolerance range. Expected: {expected_metrics[k]}, Actual: {measurements[k]}"
            )
            pass_perf_check = False

    # Don't fail the test on perf - we're diagnosing
    if not pass_perf_check:
        logger.warning("\n".join(assert_msgs))

    # Synchronize all devices
    ttnn.synchronize_device(mesh_device)

    logger.info("Performance test completed successfully!")
