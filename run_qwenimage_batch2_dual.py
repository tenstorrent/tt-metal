#!/usr/bin/env python3
"""Option 2: Dual-submesh parallel generation — 2 images on 2 submeshes simultaneously.

Uses CFG=2 mesh structure (2 submeshes of TP=2), but instead of running
conditional/unconditional for one image, each submesh generates a different image.
No CFG guidance — each submesh independently denoises its own prompt.
"""

import os
import time

os.environ.setdefault("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

import torch

import ttnn
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import (
    PROMPT_DROP_IDX,
    PROMPT_TEMPLATE,
    QwenImagePipeline,
    _schedule,
)
from models.tt_dit.utils import tensor


def generate_parallel(pipeline, prompts, num_inference_steps=50, seed=42):
    """Generate 2 images in parallel on 2 submeshes."""
    assert len(prompts) == 2, "Need exactly 2 prompts"

    sp_axis = pipeline._parallel_config.sequence_parallel.mesh_axis
    cfg_factor = pipeline._parallel_config.cfg_parallel.factor
    assert cfg_factor == 2, "Need CFG=2 for dual submesh parallel"

    latents_height = pipeline._height // pipeline._vae_scale_factor
    latents_width = pipeline._width // pipeline._vae_scale_factor
    spatial_sequence_length = (latents_height // pipeline._patch_size) * (latents_width // pipeline._patch_size)

    # Encode both prompts independently (no CFG — no negative prompts)
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
    # prompt_embeds shape: [2, seq, dim] — one per submesh

    _, prompt_sequence_length, _ = prompt_embeds.shape

    pipeline.prepare_transformers()

    timesteps, sigmas = _schedule(
        pipeline._scheduler,
        step_count=num_inference_steps,
        spatial_sequence_length=spatial_sequence_length,
    )

    # Generate different latents for each submesh (different seeds)
    p = pipeline._patch_size
    shape = [1, pipeline._num_channels_latents, latents_height, latents_width]

    torch.manual_seed(seed)
    latents_A = pipeline.transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))
    torch.manual_seed(seed + 1)
    latents_B = pipeline.transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))

    img_shapes = [[(1, latents_height // p, latents_width // p)]]
    txt_seq_lens = [prompt_sequence_length]
    spatial_rope, prompt_rope = pipeline._pos_embed.forward(img_shapes, txt_seq_lens, "cpu")
    spatial_rope_cos = spatial_rope.real.repeat_interleave(2, dim=-1)
    spatial_rope_sin = spatial_rope.imag.repeat_interleave(2, dim=-1)
    prompt_rope_cos = prompt_rope.real.repeat_interleave(2, dim=-1)
    prompt_rope_sin = prompt_rope.imag.repeat_interleave(2, dim=-1)

    # Send to submeshes — each gets its own prompt and latents
    latents_list = [latents_A, latents_B]
    tt_latents_list = []
    tt_prompt_list = []

    for i, submesh_device in enumerate(pipeline._submesh_devices):
        tt_latent = tensor.from_torch(latents_list[i], device=submesh_device, on_host=False)
        tt_prompt = tensor.from_torch(prompt_embeds[i : i + 1], device=submesh_device, on_host=False)
        tt_latents_list.append(tt_latent)
        tt_prompt_list.append(tt_prompt)

    # Copy prompt embeddings (they don't change per step)
    # Rope tensors are the same for both submeshes
    tt_rope_data = []
    for submesh_device in pipeline._submesh_devices:
        tt_rope_data.append(
            {
                "spatial_cos": tensor.from_torch(spatial_rope_cos, device=submesh_device, on_host=False),
                "spatial_sin": tensor.from_torch(spatial_rope_sin, device=submesh_device, on_host=False),
                "prompt_cos": tensor.from_torch(prompt_rope_cos, device=submesh_device, on_host=False),
                "prompt_sin": tensor.from_torch(prompt_rope_sin, device=submesh_device, on_host=False),
            }
        )

    # Denoising loop — both submeshes run independently, NO CFG combine
    import tqdm

    for i, t in enumerate(tqdm.tqdm(timesteps)):
        sigma_difference = sigmas[i + 1] - sigmas[i]

        for submesh_id, submesh_device in enumerate(pipeline._submesh_devices):
            tt_timestep = ttnn.full(
                [1, 1], fill_value=t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=submesh_device
            )
            tt_sigma_diff = ttnn.full(
                [1, 1], fill_value=sigma_difference, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=submesh_device
            )

            noise_pred = pipeline.transformers[submesh_id].forward(
                spatial=tt_latents_list[submesh_id],
                prompt=tt_prompt_list[submesh_id],
                timestep=tt_timestep,
                spatial_rope=(tt_rope_data[submesh_id]["spatial_cos"], tt_rope_data[submesh_id]["spatial_sin"]),
                prompt_rope=(tt_rope_data[submesh_id]["prompt_cos"], tt_rope_data[submesh_id]["prompt_sin"]),
                spatial_sequence_length=spatial_sequence_length,
                prompt_sequence_length=prompt_sequence_length,
            )

            ttnn.synchronize_device(submesh_device)
            ttnn.multiply_(noise_pred, tt_sigma_diff)
            ttnn.add_(tt_latents_list[submesh_id], noise_pred)

    # Decode both images
    images = []
    for submesh_id in range(2):
        tt_latents = tt_latents_list[submesh_id]
        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
        torch_latents = pipeline.transformers[0].unpatchify(torch_latents, height=latents_height, width=latents_width)
        torch_latents = torch_latents / pipeline._latents_scaling + pipeline._latents_shift
        torch_latents = torch_latents.permute(0, 3, 1, 2).unsqueeze(2)
        with torch.no_grad():
            decoded = pipeline._torch_vae.decode(torch_latents).sample[:, :, 0]
        image = pipeline._image_processor.postprocess(decoded, output_type="pt")
        images.extend(pipeline._image_processor.numpy_to_pil(pipeline._image_processor.pt_to_numpy(image)))

    return images


def main():
    MESH_SHAPE = (2, 2)
    WIDTH, HEIGHT = 1024, 1024

    print("Opening mesh device...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE), trace_region_size=50000000)

    print("Creating pipeline (CFG=2 for dual submesh)...")
    pipeline = QwenImagePipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name="Qwen/Qwen-Image",
        width=WIDTH,
        height=HEIGHT,
        use_torch_vae_decoder=True,  # Use torch VAE for simplicity in parallel decode
    )

    prompts = [
        "A cartoon white llama with sunglasses, playful, colorful background, 4K",
        "A cyberpunk cityscape at night with neon lights, rain-slick streets, 4K",
    ]

    # Warmup
    print("Warmup...")
    pipeline(prompts=[prompts[0]], negative_prompts=[None], num_inference_steps=2, cfg_scale=4.0, seed=0, traced=False)

    # Parallel generation
    print(f"Generating 2 images in parallel ({50} steps)...")
    t0 = time.time()
    images = generate_parallel(pipeline, prompts, num_inference_steps=50, seed=42)
    t1 = time.time()

    total = t1 - t0
    print(f"\n{'='*80}")
    print(f"OPTION 2: DUAL SUBMESH RESULTS")
    print(f"{'='*80}")
    print(f"images: {len(images)}")
    print(f"total_time: {total:.2f}s")
    print(f"per_image: {total/2:.2f}s")
    print(f"throughput: {2/total:.4f} images/sec")
    print(f"{'='*80}")

    images[0].save("batch2_dual_img0.png")
    images[1].save("batch2_dual_img1.png")
    print("Saved batch2_dual_img0.png and batch2_dual_img1.png")

    ttnn.close_mesh_device(mesh_device)
    print("Done.")


if __name__ == "__main__":
    main()
