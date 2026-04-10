#!/usr/bin/env python3
"""Batch=2 with full CFG guidance — 2 high-quality images simultaneously.

Uses CFG=2 (2 submeshes), each processing batch=2:
- Submesh 0: [uncond_A, uncond_B] (negative prompts)
- Submesh 1: [cond_A, cond_B] (real prompts)
Both run in parallel. CFG combine applies per-image.

Also supports 4K mode with CFG=1 + sequential CFG.
"""

import os
import sys
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


def generate_batch2_cfg_1024(pipeline, prompts, negative_prompts=None, num_inference_steps=50, cfg_scale=4.0, seed=42):
    """1024x1024: batch=2 with CFG using dual submeshes (CFG=2)."""
    assert len(prompts) == 2
    if negative_prompts is None:
        negative_prompts = ["", ""]

    sp_axis = pipeline._parallel_config.sequence_parallel.mesh_axis
    submeshes = pipeline._submesh_devices

    latents_height = pipeline._height // pipeline._vae_scale_factor
    latents_width = pipeline._width // pipeline._vae_scale_factor
    spatial_sequence_length = (latents_height // pipeline._patch_size) * (latents_width // pipeline._patch_size)

    # Encode [neg_A, neg_B, pos_A, pos_B]
    pipeline.prepare_encoder()
    with pipeline.mesh_reshape(pipeline.encoder_device, pipeline.encoder_mesh_shape):
        all_prompts = [PROMPT_TEMPLATE.format(p) for p in negative_prompts + prompts]
        embeds, mask = pipeline._text_encoder.encode(
            all_prompts,
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
    torch.manual_seed(seed)
    lat_A = pipeline.transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))
    torch.manual_seed(seed + 1)
    lat_B = pipeline.transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))
    latents_batch = torch.cat([lat_A, lat_B], dim=0)

    img_shapes = [[(1, latents_height // p, latents_width // p)]]
    txt_seq_lens = [prompt_sequence_length]
    spatial_rope, prompt_rope = pipeline._pos_embed.forward(img_shapes, txt_seq_lens, "cpu")
    if spatial_rope.dim() == 3:
        spatial_rope = spatial_rope[0]
    if prompt_rope.dim() == 3:
        prompt_rope = prompt_rope[0]
    spatial_rope_cos = spatial_rope.real.repeat_interleave(2, dim=-1)
    spatial_rope_sin = spatial_rope.imag.repeat_interleave(2, dim=-1)
    prompt_rope_cos = prompt_rope.real.repeat_interleave(2, dim=-1)
    prompt_rope_sin = prompt_rope.imag.repeat_interleave(2, dim=-1)

    # Send to both submeshes
    tt_latents = []
    tt_prompts = []
    tt_rope = []
    for i, submesh in enumerate(submeshes):
        tt_lat = tensor.from_torch(latents_batch, device=submesh, on_host=True).to(submesh)
        tt_prompt = tensor.from_torch(prompt_embeds[i * 2 : (i + 1) * 2], device=submesh, on_host=True).to(submesh)
        tt_sc = tensor.from_torch(spatial_rope_cos, device=submesh, on_host=True).to(submesh)
        tt_ss = tensor.from_torch(spatial_rope_sin, device=submesh, on_host=True).to(submesh)
        tt_pc = tensor.from_torch(prompt_rope_cos, device=submesh, on_host=True).to(submesh)
        tt_ps = tensor.from_torch(prompt_rope_sin, device=submesh, on_host=True).to(submesh)
        tt_latents.append(tt_lat)
        tt_prompts.append(tt_prompt)
        tt_rope.append({"sc": tt_sc, "ss": tt_ss, "pc": tt_pc, "ps": tt_ps})

    # Denoising loop
    for step_i, t in enumerate(tqdm.tqdm(timesteps)):
        sigma_difference = sigmas[step_i + 1] - sigmas[step_i]

        noise_preds = []
        for i, submesh in enumerate(submeshes):
            tt_ts = ttnn.full([1, 1], fill_value=t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=submesh)
            pred = pipeline.transformers[i].forward(
                spatial=tt_latents[i],
                prompt=tt_prompts[i],
                timestep=tt_ts,
                spatial_rope=(tt_rope[i]["sc"], tt_rope[i]["ss"]),
                prompt_rope=(tt_rope[i]["pc"], tt_rope[i]["ps"]),
                spatial_sequence_length=spatial_sequence_length,
                prompt_sequence_length=prompt_sequence_length,
            )
            noise_preds.append(pred)

        # CFG combine (CPU, bfloat16 — no float32 conversion)
        uncond = ttnn.to_torch(ttnn.get_device_tensors(noise_preds[0])[0])
        cond = ttnn.to_torch(ttnn.get_device_tensors(noise_preds[1])[0])
        combined = (uncond.float() + cfg_scale * (cond.float() - uncond.float())).bfloat16()

        tt_sigma_diff = ttnn.full([1, 1], fill_value=sigma_difference, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        for i, submesh in enumerate(submeshes):
            tt_combined = tensor.from_torch(combined, device=submesh, on_host=True).to(submesh)
            ttnn.synchronize_device(submesh)
            ttnn.multiply_(tt_combined, tt_sigma_diff.to(submesh))
            ttnn.add_(tt_latents[i], tt_combined)

    # Decode
    torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents[0])[0])
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


def generate_batch2_cfg_4k(pipeline, prompts, negative_prompts=None, num_inference_steps=50, cfg_scale=4.0, seed=42):
    """4K: batch=2 with sequential CFG (CFG=1, SP=2). Each step runs cond then uncond."""
    assert len(prompts) == 2
    if negative_prompts is None:
        negative_prompts = ["", ""]

    sp_axis = pipeline._parallel_config.sequence_parallel.mesh_axis
    submesh = pipeline._submesh_devices[0]

    latents_height = pipeline._height // pipeline._vae_scale_factor
    latents_width = pipeline._width // pipeline._vae_scale_factor
    spatial_sequence_length = (latents_height // pipeline._patch_size) * (latents_width // pipeline._patch_size)

    # Encode all 4 prompts
    pipeline.prepare_encoder()
    with pipeline.mesh_reshape(pipeline.encoder_device, pipeline.encoder_mesh_shape):
        all_prompts = [PROMPT_TEMPLATE.format(p) for p in negative_prompts + prompts]
        embeds, mask = pipeline._text_encoder.encode(
            all_prompts,
            num_images_per_prompt=1,
            sequence_length=512 + PROMPT_DROP_IDX,
        )
        embeds[torch.logical_not(mask)] = 0.0
        prompt_embeds = embeds[:, PROMPT_DROP_IDX:]
    # [neg_A, neg_B, pos_A, pos_B]

    _, prompt_sequence_length, _ = prompt_embeds.shape
    pipeline.prepare_transformers()

    timesteps, sigmas = _schedule(
        pipeline._scheduler,
        step_count=num_inference_steps,
        spatial_sequence_length=spatial_sequence_length,
    )

    p = pipeline._patch_size
    shape = [1, pipeline._num_channels_latents, latents_height, latents_width]
    torch.manual_seed(seed)
    lat_A = pipeline.transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))
    torch.manual_seed(seed + 1)
    lat_B = pipeline.transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))
    latents_batch = torch.cat([lat_A, lat_B], dim=0)

    img_shapes = [[(1, latents_height // p, latents_width // p)]]
    txt_seq_lens = [prompt_sequence_length]
    spatial_rope, prompt_rope = pipeline._pos_embed.forward(img_shapes, txt_seq_lens, "cpu")
    if spatial_rope.dim() == 3:
        spatial_rope = spatial_rope[0]
    if prompt_rope.dim() == 3:
        prompt_rope = prompt_rope[0]
    spatial_rope_cos = spatial_rope.real.repeat_interleave(2, dim=-1)
    spatial_rope_sin = spatial_rope.imag.repeat_interleave(2, dim=-1)
    prompt_rope_cos = prompt_rope.real.repeat_interleave(2, dim=-1)
    prompt_rope_sin = prompt_rope.imag.repeat_interleave(2, dim=-1)

    # Send to device with SP sharding
    tt_latents = tensor.from_torch(latents_batch, device=submesh, on_host=True, mesh_axes=[None, sp_axis, None]).to(
        submesh
    )
    tt_neg_prompt = tensor.from_torch(prompt_embeds[0:2], device=submesh, on_host=True).to(submesh)
    tt_pos_prompt = tensor.from_torch(prompt_embeds[2:4], device=submesh, on_host=True).to(submesh)
    tt_spatial_cos = tensor.from_torch(spatial_rope_cos, device=submesh, on_host=True, mesh_axes=[sp_axis, None]).to(
        submesh
    )
    tt_spatial_sin = tensor.from_torch(spatial_rope_sin, device=submesh, on_host=True, mesh_axes=[sp_axis, None]).to(
        submesh
    )
    tt_prompt_cos = tensor.from_torch(prompt_rope_cos, device=submesh, on_host=True).to(submesh)
    tt_prompt_sin = tensor.from_torch(prompt_rope_sin, device=submesh, on_host=True).to(submesh)

    # Denoising: sequential CFG (uncond then cond forward per step)
    for step_i, t in enumerate(tqdm.tqdm(timesteps)):
        sigma_difference = sigmas[step_i + 1] - sigmas[step_i]
        tt_ts = ttnn.full([1, 1], fill_value=t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=submesh)

        # Unconditional forward
        uncond_pred = pipeline.transformers[0].forward(
            spatial=tt_latents,
            prompt=tt_neg_prompt,
            timestep=tt_ts,
            spatial_rope=(tt_spatial_cos, tt_spatial_sin),
            prompt_rope=(tt_prompt_cos, tt_prompt_sin),
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )
        # Conditional forward
        cond_pred = pipeline.transformers[0].forward(
            spatial=tt_latents,
            prompt=tt_pos_prompt,
            timestep=tt_ts,
            spatial_rope=(tt_spatial_cos, tt_spatial_sin),
            prompt_rope=(tt_prompt_cos, tt_prompt_sin),
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )

        # CFG combine on-device
        # combined = uncond + cfg_scale * (cond - uncond)
        diff = ttnn.subtract(cond_pred, uncond_pred)
        scaled = ttnn.multiply(diff, cfg_scale)
        combined = ttnn.add(uncond_pred, scaled)

        tt_sigma_diff = ttnn.full(
            [1, 1], fill_value=sigma_difference, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=submesh
        )
        ttnn.synchronize_device(submesh)
        ttnn.multiply_(combined, tt_sigma_diff)
        ttnn.add_(tt_latents, combined)

    # Decode
    sp_ccl = pipeline._ccl_managers[0]
    ttnn.synchronize_device(submesh)
    tt_latents_full = sp_ccl.all_gather_persistent_buffer(tt_latents, dim=1, mesh_axis=sp_axis, use_hyperparams=True)
    torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents_full)[0])

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
    mode = sys.argv[1] if len(sys.argv) > 1 else "1024"
    is_4k = mode == "4k"

    if is_4k:
        WIDTH, HEIGHT = 4096, 2304
    else:
        WIDTH, HEIGHT = 1024, 1024

    NUM_STEPS = 50
    print(f"Batch=2 + CFG: {WIDTH}x{HEIGHT}, {NUM_STEPS} steps, mode={mode}")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    trace_size = 200000000 if is_4k else 50000000
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 2), trace_region_size=trace_size)

    if is_4k:
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
    else:
        pipeline = QwenImagePipeline.create_pipeline(
            mesh_device=mesh_device,
            checkpoint_name="Qwen/Qwen-Image",
            width=WIDTH,
            height=HEIGHT,
        )

    prompts = [
        "A cartoon white llama with sunglasses, playful expression, colorful rainbow background, 4K",
        "A cyberpunk cityscape at night with neon lights, rain-slick streets, cinematic, 4K",
    ]

    # Warmup
    print("Warmup...")
    pipeline(prompts=[prompts[0]], negative_prompts=[None], num_inference_steps=2, cfg_scale=4.0, seed=0, traced=False)

    print(f"Generating 2 images with CFG ({NUM_STEPS} steps)...")
    t0 = time.time()
    if is_4k:
        images = generate_batch2_cfg_4k(pipeline, prompts, num_inference_steps=NUM_STEPS, cfg_scale=4.0, seed=42)
    else:
        images = generate_batch2_cfg_1024(pipeline, prompts, num_inference_steps=NUM_STEPS, cfg_scale=4.0, seed=42)
    t1 = time.time()

    n = len(images)
    total = t1 - t0
    print(f"\n{'='*80}")
    print(f"BATCH=2 + CFG RESULTS ({mode})")
    print(f"{'='*80}")
    print(f"resolution: {WIDTH}x{HEIGHT}")
    print(f"images: {n}")
    print(f"total_time: {total:.2f}s")
    print(f"per_image: {total/n:.2f}s")
    print(f"throughput: {n/total:.4f} images/sec")
    print(f"per_step: {total/NUM_STEPS:.2f}s")
    print(f"{'='*80}")

    for i, img in enumerate(images):
        img.save(f"batch2_cfg_{mode}_img{i}.png")
    print(f"Saved {n} images")

    ttnn.close_mesh_device(mesh_device)
    print("Done.")


if __name__ == "__main__":
    main()
