#!/usr/bin/env python3
"""Run Qwen-Image pipeline at 4K (3840x2160) on P300x2 (2x2 mesh, 4 Blackhole chips).

Strategy: Drop CFG parallelism, use all 4 chips for SP=2 + TP=2.
- CFG=1: conditional/unconditional run sequentially in same batch
- SP=2 (axis 0): spatial sequence split across 2 rows
- TP=2 (axis 1): weights split across 2 columns
- Dynamic encoder loading: free memory for the much larger activations
- Torch VAE decoder: TT VAE likely OOMs at 4K resolution
"""

import os
import time

os.environ.setdefault("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

import ttnn
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

MESH_SHAPE = (2, 2)
NUM_INFERENCE_STEPS = 20  # Fewer steps for faster iteration during optimization
# 4096x2304: 16:9 aspect, 9.4MP (larger than 4K UHD 8.3MP)
# Patches: 256x144 = 36864, cleanly divisible by SP=2, k_chunk=512, tile=32
WIDTH = 4096
HEIGHT = 2304


def main():
    print(f"Opening mesh device with shape {MESH_SHAPE}...")
    print(f"Target resolution: {WIDTH}x{HEIGHT}")
    print(f"Spatial sequence: {(WIDTH//16) * (HEIGHT//16)} patches")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        trace_region_size=200000000,  # 200MB trace region for 4K
    )
    print(f"Mesh device opened: {mesh_device.shape}")

    print("Creating Qwen-Image pipeline (4K config)...")
    pipeline = QwenImagePipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name="Qwen/Qwen-Image",
        width=WIDTH,
        height=HEIGHT,
        # Override parallelism for 4K: use all chips for one image
        dit_cfg=(1, 0),  # No CFG parallelism — run in single batch
        dit_sp=(2, 0),  # Sequence parallel across axis 0
        dit_tp=(2, 1),  # Tensor parallel across axis 1
        encoder_tp=(2, 1),  # Encoder TP
        vae_tp=(2, 1),  # VAE TP
        num_links=2,
        use_torch_vae_decoder=False,  # TT VAE with dynamic loading
        dynamic_load_encoder=True,  # Free memory for transformer
        dynamic_load_vae=True,  # Load/unload VAE to save memory
        is_fsdp=True,  # Required — OOM without FSDP at 4K
    )
    print("Pipeline created successfully!")

    prompt = (
        "A breathtaking panoramic view of a mountain valley at golden hour, "
        "crystal clear lake reflecting snow-capped peaks, wildflowers in the foreground. "
        "Ultra HD, 4K, cinematic composition, photorealistic."
    )

    # Warmup run (captures trace on first traced call)
    print("Running warmup (2 steps, traced — captures trace)...")
    t0 = time.time()
    images = pipeline(
        prompts=[prompt],
        negative_prompts=[None],
        num_inference_steps=2,
        cfg_scale=4.0,
        seed=0,
        traced=True,
    )
    t1 = time.time()
    print(f"Warmup done in {t1 - t0:.2f}s")

    # Benchmark run (replays trace)
    print(f"Running 4K benchmark ({NUM_INFERENCE_STEPS} steps, traced)...")
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

    images[0].save("qwenimage_4k_benchmark.png")

    print("=" * 80)
    print("4K BENCHMARK RESULTS")
    print("=" * 80)
    print(f"resolution: {WIDTH}x{HEIGHT}")
    print(f"generation_speed: {gen_speed:.4f} images/sec")
    print(f"total_time: {total_time:.2f}s")
    print(f"num_steps: {NUM_INFERENCE_STEPS}")
    print(f"per_step: {total_time/NUM_INFERENCE_STEPS:.2f}s")
    print("=" * 80)

    ttnn.close_mesh_device(mesh_device)
    print("Done.")


if __name__ == "__main__":
    main()
