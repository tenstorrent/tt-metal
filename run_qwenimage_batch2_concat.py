#!/usr/bin/env python3
"""Option 3: Batch=2 in single submesh — 2 images concatenated along batch dim.

Uses CFG=1 (all 4 chips in one submesh, TP=2 SP=1), concatenates 2 different
latents along the batch dimension. The transformer processes both simultaneously.
No CFG guidance.
"""

import os
import time

os.environ.setdefault("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

import torch
import tqdm

import ttnn
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import (
    PROMPT_DROP_IDX,
    PROMPT_TEMPLATE,
    QwenImagePipeline,
    _schedule,
)
from models.tt_dit.utils import tensor


def generate_batched(pipeline, prompts, num_inference_steps=50, seed=42):
    """Generate 2 images in one batched forward pass."""
    assert len(prompts) == 2, "Need exactly 2 prompts"

    sp_axis = pipeline._parallel_config.sequence_parallel.mesh_axis
    submesh = pipeline._submesh_devices[0]

    latents_height = pipeline._height // pipeline._vae_scale_factor
    latents_width = pipeline._width // pipeline._vae_scale_factor
    spatial_sequence_length = (latents_height // pipeline._patch_size) * (latents_width // pipeline._patch_size)

    # Encode both prompts
    pipeline.prepare_encoder()
    with pipeline.mesh_reshape(pipeline.encoder_device, pipeline.encoder_mesh_shape):
        formatted = [PROMPT_TEMPLATE.format(p) for p in prompts]
        embeds, mask = pipeline._text_encoder.encode(
            formatted,
            num_images_per_prompt=1,
            sequence_length=512 + PROMPT_DROP_IDX,
        )
        embeds[torch.logical_not(mask)] = 0.0
        prompt_embeds = embeds[:, PROMPT_DROP_IDX:]
    # prompt_embeds shape: [2, seq, dim] — batch of 2

    _, prompt_sequence_length, _ = prompt_embeds.shape

    pipeline.prepare_transformers()

    timesteps, sigmas = _schedule(
        pipeline._scheduler,
        step_count=num_inference_steps,
        spatial_sequence_length=spatial_sequence_length,
    )

    # Generate 2 different latents and concatenate along batch dim
    p = pipeline._patch_size
    shape = [1, pipeline._num_channels_latents, latents_height, latents_width]

    torch.manual_seed(seed)
    latents_A = pipeline.transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))
    torch.manual_seed(seed + 1)
    latents_B = pipeline.transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))

    # Batch=2: concatenate along dim 0
    latents_batch = torch.cat([latents_A, latents_B], dim=0)  # [2, seq, channels]

    # RoPE — same for both images, but needs batch=2
    img_shapes = [[(1, latents_height // p, latents_width // p)]] * 2
    txt_seq_lens = [prompt_sequence_length] * 2
    spatial_rope, prompt_rope = pipeline._pos_embed.forward(img_shapes, txt_seq_lens, "cpu")
    # spatial_rope is [2, seq, dim/2] complex — use first since both are identical
    spatial_rope_cos = spatial_rope[0:1].real.repeat_interleave(2, dim=-1)
    spatial_rope_sin = spatial_rope[0:1].imag.repeat_interleave(2, dim=-1)
    prompt_rope_cos = prompt_rope[0:1].real.repeat_interleave(2, dim=-1)
    prompt_rope_sin = prompt_rope[0:1].imag.repeat_interleave(2, dim=-1)

    # Send to device (on_host=True for tracing — data copied in via trace inputs)
    tt_latents = tensor.from_torch(latents_batch, device=submesh, on_host=True).to(submesh)
    tt_prompt = tensor.from_torch(prompt_embeds, device=submesh, on_host=True).to(submesh)
    tt_spatial_cos = tensor.from_torch(spatial_rope_cos, device=submesh, on_host=True).to(submesh)
    tt_spatial_sin = tensor.from_torch(spatial_rope_sin, device=submesh, on_host=True).to(submesh)
    tt_prompt_cos = tensor.from_torch(prompt_rope_cos, device=submesh, on_host=True).to(submesh)
    tt_prompt_sin = tensor.from_torch(prompt_rope_sin, device=submesh, on_host=True).to(submesh)

    tt_timestep = ttnn.full([1, 1], fill_value=0.0, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=submesh)
    tt_sigma_diff = ttnn.full([1, 1], fill_value=0.0, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=submesh)

    # Warmup forward pass (compiles kernels)
    noise_pred = pipeline.transformers[0].forward(
        spatial=tt_latents,
        prompt=tt_prompt,
        timestep=tt_timestep,
        spatial_rope=(tt_spatial_cos, tt_spatial_sin),
        prompt_rope=(tt_prompt_cos, tt_prompt_sin),
        spatial_sequence_length=spatial_sequence_length,
        prompt_sequence_length=prompt_sequence_length,
    )
    ttnn.synchronize_device(submesh)
    ttnn.multiply_(noise_pred, tt_sigma_diff)
    ttnn.add_(tt_latents, noise_pred)

    # Capture trace (no sync allowed inside trace region)
    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    noise_pred = pipeline.transformers[0].forward(
        spatial=tt_latents,
        prompt=tt_prompt,
        timestep=tt_timestep,
        spatial_rope=(tt_spatial_cos, tt_spatial_sin),
        prompt_rope=(tt_prompt_cos, tt_prompt_sin),
        spatial_sequence_length=spatial_sequence_length,
        prompt_sequence_length=prompt_sequence_length,
    )
    ttnn.multiply_(noise_pred, tt_sigma_diff)
    ttnn.add_(tt_latents, noise_pred)
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)

    # Re-load fresh latents (warmup + trace capture consumed 2 steps worth)
    tt_latents_fresh = tensor.from_torch(latents_batch, device=submesh, on_host=True)
    ttnn.copy_host_to_device_tensor(tt_latents_fresh, tt_latents)

    # Traced denoising loop — batch=2, no CFG
    for i, t in enumerate(tqdm.tqdm(timesteps)):
        sigma_difference = sigmas[i + 1] - sigmas[i]

        tt_ts_host = ttnn.full([1, 1], fill_value=t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        tt_sd_host = ttnn.full([1, 1], fill_value=sigma_difference, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.copy_host_to_device_tensor(tt_ts_host, tt_timestep)
        ttnn.copy_host_to_device_tensor(tt_sd_host, tt_sigma_diff)

        ttnn.execute_trace(submesh, trace_id, cq_id=0, blocking=True)

    # Decode both images — split batch and decode each via TT VAE
    torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
    # torch_latents shape: [2, seq, channels]

    pipeline.prepare_vae()

    images = []
    for b in range(2):
        lat = torch_latents[b : b + 1]
        lat = pipeline.transformers[0].unpatchify(lat, height=latents_height, width=latents_width)
        lat = lat / pipeline._latents_scaling + pipeline._latents_shift

        with pipeline.mesh_reshape(pipeline.vae_device, pipeline.vae_mesh_shape):
            tt_lat, logical_h = pipeline._vae_decoder.prepare_input(lat)
            tt_decoded, logical_h = pipeline._vae_decoder.forward(tt_lat, logical_h)
            decoded = pipeline._vae_decoder.postprocess_output(tt_decoded, logical_h)

        image = pipeline._image_processor.postprocess(decoded, output_type="pt")
        images.extend(pipeline._image_processor.numpy_to_pil(pipeline._image_processor.pt_to_numpy(image)))

    return images


def main():
    MESH_SHAPE = (2, 2)
    WIDTH, HEIGHT = 1024, 1024

    print("Opening mesh device...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE), trace_region_size=50000000)

    print("Creating pipeline (CFG=1, all 4 chips for batch=2)...")
    pipeline = QwenImagePipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name="Qwen/Qwen-Image",
        width=WIDTH,
        height=HEIGHT,
        dit_cfg=(1, 0),
        dit_sp=(1, 0),
        dit_tp=(2, 1),
        encoder_tp=(2, 1),
        vae_tp=(2, 1),
        use_torch_vae_decoder=False,
        num_links=2,
    )

    prompts = [
        "A cartoon white llama with sunglasses, playful, colorful background, 4K",
        "A cyberpunk cityscape at night with neon lights, rain-slick streets, 4K",
    ]

    # Warmup
    print("Warmup...")
    pipeline(prompts=[prompts[0]], negative_prompts=[None], num_inference_steps=2, cfg_scale=1.0, seed=0, traced=False)

    # Batched generation
    print(f"Generating 2 images in batch ({50} steps)...")
    t0 = time.time()
    images = generate_batched(pipeline, prompts, num_inference_steps=50, seed=42)
    t1 = time.time()

    total = t1 - t0
    print(f"\n{'='*80}")
    print(f"OPTION 3: BATCHED FORWARD RESULTS")
    print(f"{'='*80}")
    print(f"images: {len(images)}")
    print(f"total_time: {total:.2f}s")
    print(f"per_image: {total/2:.2f}s")
    print(f"throughput: {2/total:.4f} images/sec")
    print(f"{'='*80}")

    images[0].save("batch2_concat_img0.png")
    images[1].save("batch2_concat_img1.png")
    print("Saved batch2_concat_img0.png and batch2_concat_img1.png")

    ttnn.close_mesh_device(mesh_device)
    print("Done.")


if __name__ == "__main__":
    main()
