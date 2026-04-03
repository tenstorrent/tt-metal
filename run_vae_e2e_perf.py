#!/usr/bin/env python3
"""
Run VAE decoder timing via the full WAN pipeline with minimal denoising steps.
Uses 3 inference steps to get through denoising quickly, then measures VAE decode.
"""
import time
import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline

HEIGHT, WIDTH = 480, 832
NUM_FRAMES = 81
NUM_WARMUP_STEPS = 2
NUM_PERF_STEPS = 3
MESH_SHAPE = (2, 4)

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(
    ttnn.MeshShape(*MESH_SHAPE),
    num_command_queues=2,
)

pipeline = WanPipeline.create_pipeline(
    mesh_device=mesh_device,
    sp_axis=0,
    tp_axis=1,
    num_links=1,
    dynamic_load=True,
    topology=ttnn.Topology.Linear,
    is_fsdp=True,
    target_height=HEIGHT,
    target_width=WIDTH,
)

prompt = "A cat sitting on a windowsill watching rain fall outside."
profiler = BenchmarkProfiler()

# Warmup
print(f"Warmup ({NUM_WARMUP_STEPS} denoising steps)...")
t0 = time.time()
pipeline(
    prompt=prompt,
    height=HEIGHT,
    width=WIDTH,
    num_frames=NUM_FRAMES,
    num_inference_steps=NUM_WARMUP_STEPS,
    profiler=profiler,
    profiler_iteration=0,
)
print(f"  Warmup done in {time.time()-t0:.1f}s  (VAE: {profiler.get_duration('vae', 0):.3f}s)")

# Timed run
profiler2 = BenchmarkProfiler()
print(f"\nTimed run ({NUM_PERF_STEPS} denoising steps)...")
t0 = time.time()
pipeline(
    prompt=prompt,
    height=HEIGHT,
    width=WIDTH,
    num_frames=NUM_FRAMES,
    num_inference_steps=NUM_PERF_STEPS,
    profiler=profiler2,
    profiler_iteration=0,
)
total = time.time() - t0
vae_t = profiler2.get_duration("vae", 0)
deno_t = profiler2.get_duration("denoising", 0)
enc_t = profiler2.get_duration("encoder", 0)

print(f"\n{'='*60}")
print(f"WAN 2.2 VAE perf — {HEIGHT}x{WIDTH}, {NUM_FRAMES} frames, 2x4 BH LB")
print(f"  Encoder:   {enc_t:.3f}s")
print(f"  Denoising: {deno_t:.3f}s  ({NUM_PERF_STEPS} steps)")
print(f"  VAE:       {vae_t:.3f}s")
print(f"  Total:     {total:.3f}s")
print(f"{'='*60}")

ttnn.close_mesh_device(mesh_device)
