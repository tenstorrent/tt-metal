#!/usr/bin/env python3
"""Run Qwen-Image pipeline on P300x2 (2x2 mesh, 4 Blackhole chips)."""

import os
import time

os.environ.setdefault("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

import ttnn
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

MESH_SHAPE = (2, 2)
NUM_INFERENCE_STEPS = 50
WIDTH = 1024
HEIGHT = 1024


def main():
    print(f"Opening mesh device with shape {MESH_SHAPE}...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        trace_region_size=50000000,
    )
    print(f"Mesh device opened: {mesh_device.shape}")

    print("Creating Qwen-Image pipeline...")
    pipeline = QwenImagePipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name="Qwen/Qwen-Image",
        width=WIDTH,
        height=HEIGHT,
    )
    print("Pipeline created successfully!")

    prompt = (
        'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee $2 per cup," '
        "with a neon light beside it. Ultra HD, 4K, cinematic composition."
    )

    # Warmup run
    print("Running warmup...")
    t0 = time.time()
    images = pipeline(
        prompts=[prompt],
        negative_prompts=[None],
        num_inference_steps=2,
        cfg_scale=4.0,
        seed=0,
        traced=False,
    )
    t1 = time.time()
    print(f"Warmup done in {t1 - t0:.2f}s")

    # Benchmark run
    print(f"Running benchmark ({NUM_INFERENCE_STEPS} steps)...")
    t0 = time.time()
    images = pipeline(
        prompts=[prompt],
        negative_prompts=[None],
        num_inference_steps=NUM_INFERENCE_STEPS,
        cfg_scale=4.0,
        seed=42,
        traced=True,
    )
    t1 = time.time()

    total_time = t1 - t0
    gen_speed = 1.0 / total_time if total_time > 0 else 0

    images[0].save("qwenimage_p300x2_benchmark.png")

    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"generation_speed: {gen_speed:.4f} images/sec")
    print(f"total_time: {total_time:.2f}s")
    print(f"accuracy: baseline")
    print(f"peak_vram: N/A")
    print("=" * 80)

    ttnn.close_mesh_device(mesh_device)
    print("Done.")


if __name__ == "__main__":
    main()
