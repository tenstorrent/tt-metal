#!/usr/bin/env python3
"""4K batch=2: Generate 2 4K images in one batched forward pass.

Uses CFG=1, SP=2, TP=2, FSDP on (2,2) mesh. Batch=2 doubles activations
which may OOM — this is an experimental test.
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

MESH_SHAPE = (2, 2)
WIDTH = 4096
HEIGHT = 2304
NUM_STEPS = 50


def generate_batched_4k(pipeline, prompts, num_inference_steps=NUM_STEPS, seed=42):
    assert len(prompts) >= 2

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

    _, prompt_sequence_length, _ = prompt_embeds.shape

    pipeline.prepare_transformers()

    timesteps, sigmas = _schedule(
        pipeline._scheduler,
        step_count=num_inference_steps,
        spatial_sequence_length=spatial_sequence_length,
    )

    p = pipeline._patch_size
    shape = [1, pipeline._num_channels_latents, latents_height, latents_width]

    latent_list = []
    for i in range(len(prompts)):
        torch.manual_seed(seed + i)
        lat = pipeline.transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))
        latent_list.append(lat)
    latents_batch = torch.cat(latent_list, dim=0)

    # RoPE — same for both images. Match pipeline convention: [seq, dim] 2D tensors
    img_shapes = [[(1, latents_height // p, latents_width // p)]]
    txt_seq_lens = [prompt_sequence_length]
    spatial_rope, prompt_rope = pipeline._pos_embed.forward(img_shapes, txt_seq_lens, "cpu")
    # spatial_rope may be 2D [seq, dim/2] or 3D [batch, seq, dim/2] — ensure 2D
    if spatial_rope.dim() == 3:
        spatial_rope = spatial_rope[0]
    if prompt_rope.dim() == 3:
        prompt_rope = prompt_rope[0]
    spatial_rope_cos = spatial_rope.real.repeat_interleave(2, dim=-1)
    spatial_rope_sin = spatial_rope.imag.repeat_interleave(2, dim=-1)
    prompt_rope_cos = prompt_rope.real.repeat_interleave(2, dim=-1)
    prompt_rope_sin = prompt_rope.imag.repeat_interleave(2, dim=-1)
    print(f"Rope shapes: spatial={spatial_rope_cos.shape}, prompt={prompt_rope_cos.shape}")

    # Send to device
    tt_latents = tensor.from_torch(latents_batch, device=submesh, on_host=True, mesh_axes=[None, sp_axis, None]).to(
        submesh
    )
    tt_prompt = tensor.from_torch(prompt_embeds, device=submesh, on_host=True).to(submesh)
    tt_spatial_cos = tensor.from_torch(spatial_rope_cos, device=submesh, on_host=True, mesh_axes=[sp_axis, None]).to(
        submesh
    )
    tt_spatial_sin = tensor.from_torch(spatial_rope_sin, device=submesh, on_host=True, mesh_axes=[sp_axis, None]).to(
        submesh
    )
    tt_prompt_cos = tensor.from_torch(prompt_rope_cos, device=submesh, on_host=True).to(submesh)
    tt_prompt_sin = tensor.from_torch(prompt_rope_sin, device=submesh, on_host=True).to(submesh)

    # Denoising loop
    for i, t in enumerate(tqdm.tqdm(timesteps)):
        sigma_difference = sigmas[i + 1] - sigmas[i]
        tt_timestep = ttnn.full([1, 1], fill_value=t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=submesh)
        tt_sigma_diff = ttnn.full(
            [1, 1], fill_value=sigma_difference, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=submesh
        )

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

    # Decode via TT VAE
    sp_ccl = pipeline._ccl_managers[0]
    ttnn.synchronize_device(submesh)
    tt_latents_full = sp_ccl.all_gather_persistent_buffer(tt_latents, dim=1, mesh_axis=sp_axis, use_hyperparams=True)
    torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents_full)[0])

    pipeline.prepare_vae()

    images = []
    for b in range(len(prompts)):
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
    print(f"4K Batch=2: {WIDTH}x{HEIGHT}, {NUM_STEPS} steps")
    print(f"Spatial sequence: {(WIDTH//16) * (HEIGHT//16)} patches")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE), trace_region_size=200000000)

    pipeline = QwenImagePipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name="Qwen/Qwen-Image",
        width=WIDTH,
        height=HEIGHT,
        dit_cfg=(1, 0),
        dit_sp=(2, 0),
        dit_tp=(2, 1),
        encoder_tp=(2, 1),
        vae_tp=(2, 1),
        num_links=2,
        use_torch_vae_decoder=False,
        dynamic_load_encoder=True,
        dynamic_load_vae=True,
        is_fsdp=True,
    )

    prompts = [
        "A cartoon white llama with sunglasses on a colorful rainbow background, playful, fun, 4K ultra HD",
        "A cyberpunk cityscape at night with neon lights and rain-slick streets, cinematic, photorealistic, 4K",
        "A serene Japanese zen garden with cherry blossoms, koi pond, wooden bridge, ultra HD, 4K",
        "An astronaut riding a horse on Mars, Earth visible in the sky, photorealistic, 4K cinematic",
    ]

    # Warmup
    print("Warmup...")
    pipeline(prompts=[prompts[0]], negative_prompts=[None], num_inference_steps=2, cfg_scale=4.0, seed=0, traced=True)

    # Batch=2 generation
    print(f"Generating 2 4K images in batch ({NUM_STEPS} steps)...")
    t0 = time.time()
    images = generate_batched_4k(pipeline, prompts, num_inference_steps=NUM_STEPS, seed=42)
    t1 = time.time()

    n = len(images)
    total = t1 - t0
    print(f"\n{'='*80}")
    print(f"4K BATCH={n} RESULTS")
    print(f"{'='*80}")
    print(f"resolution: {WIDTH}x{HEIGHT}")
    print(f"images: {n}")
    print(f"total_time: {total:.2f}s")
    print(f"per_image: {total/n:.2f}s")
    print(f"throughput: {n/total:.4f} images/sec")
    print(f"per_step: {total/NUM_STEPS:.2f}s")
    print(f"{'='*80}")

    for i, img in enumerate(images):
        img.save(f"4k_batch{n}_img{i}.png")
    print(f"Saved {n} images as 4k_batch{n}_img*.png")

    ttnn.close_mesh_device(mesh_device)
    print("Done.")


if __name__ == "__main__":
    main()
