# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Flux1 image generation using distributed ttml transformer.

Text encoders, scheduler, and VAE run on torch/CPU.  Only the Flux
transformer (denoising backbone) runs on Tenstorrent devices via ttml.

Usage:
    cd tt-train/sources/examples/flux1

    python3 generate.py \\
        --checkpoint black-forest-labs/FLUX.1-dev \\
        --mesh_shape 1 2 \\
        --prompt "A cinematic neon-lit cyberpunk alley in the rain." \\
        --num_inference_steps 28 \\
        --output flux-dev.png

    python3 generate.py \\
        --checkpoint black-forest-labs/FLUX.1-schnell \\
        --mesh_shape 1 2 \\
        --prompt "A luxury sports car." \\
        --num_inference_steps 4 \\
        --output flux-schnell.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import tqdm as tqdm_mod

import ttnn
import ttml
from ttml.common.utils import no_grad

from model_flux_distributed import (
    DistributedFlux1Transformer,
    create_flux1_config_from_hf,
    empty_init,
    load_weights_from_hf_distributed,
)


# =====================================================================
# Torch-side helpers (latents, RoPE, encoders, VAE, scheduler)
# =====================================================================


def _sinusoidal_proj(timestep: torch.Tensor, num_channels: int = 256) -> torch.Tensor:
    """Project scalar timestep(s) into sinusoidal embedding (bfloat16).

    Matches HuggingFace get_timestep_embedding with flip_sin_to_cos=True: [cos, sin] order.
    """
    half = num_channels // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
    factor = torch.exp(exponent).to(timestep.device)
    emb = timestep.float().unsqueeze(-1) * factor.unsqueeze(0)
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1).to(torch.bfloat16)


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height, 2, width, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, height * width, num_channels_latents * 4)


def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = height // vae_scale_factor
    width = width // vae_scale_factor
    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, height * 2, width * 2)


def _latent_image_ids(height: int, width: int) -> torch.Tensor:
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(height)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(width)[None, :]
    return ids.reshape(height * width, 3)


def _calculate_shift(image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


# =====================================================================
# Encoding prompts (torch CLIP + T5)
# =====================================================================


def encode_prompts(prompt, checkpoint_name, joint_attention_dim):
    """Encode a text prompt using CLIP and T5 on CPU."""
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    print("  Loading CLIP tokenizer + model ...")
    clip_tokenizer = CLIPTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")
    clip_model = CLIPTextModel.from_pretrained(checkpoint_name, subfolder="text_encoder")
    clip_model.eval()

    print("  Loading T5 tokenizer + model ...")
    t5_tokenizer = T5TokenizerFast.from_pretrained(checkpoint_name, subfolder="tokenizer_2")
    t5_model = T5EncoderModel.from_pretrained(checkpoint_name, subfolder="text_encoder_2")
    t5_model.eval()

    clip_tokens = clip_tokenizer(
        [prompt], return_tensors="pt", padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True
    ).input_ids

    with torch.no_grad():
        clip_out = clip_model(clip_tokens, output_hidden_states=True)
    pooled_prompt_embeds = clip_out.pooler_output.to(torch.bfloat16)

    t5_tokens = t5_tokenizer(
        [prompt], return_tensors="pt", padding="max_length", max_length=512, truncation=True
    ).input_ids
    with torch.no_grad():
        t5_out = t5_model(t5_tokens)
    prompt_embeds = t5_out.last_hidden_state.to(torch.bfloat16)

    del clip_model, t5_model
    return prompt_embeds, pooled_prompt_embeds


# =====================================================================
# VAE decode (torch)
# =====================================================================


def decode_latents(latents_torch, checkpoint_name, vae_scale_factor, height, width):
    """Decode latent tensor to PIL image using HF AutoencoderKL on CPU."""
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor

    vae = AutoencoderKL.from_pretrained(checkpoint_name, subfolder="vae")
    vae.eval()

    scaling_factor = vae.config.scaling_factor
    shift_factor = vae.config.shift_factor

    latents_torch = (latents_torch.float() / scaling_factor) + shift_factor
    latents_torch = _unpack_latents(latents_torch, height, width, vae_scale_factor)

    with torch.no_grad():
        decoded = vae.decode(latents_torch).sample

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    image = image_processor.postprocess(decoded, output_type="pil")
    del vae
    return image


# =====================================================================
# Device setup
# =====================================================================


def setup_device(dp_size: int, tp_size: int, seed: int = 42):
    distributed = dp_size > 1 or tp_size > 1
    total_devices = dp_size * tp_size

    if distributed:
        if "TT_MESH_GRAPH_DESC_PATH" not in os.environ:
            print("WARNING: TT_MESH_GRAPH_DESC_PATH not set.", file=sys.stderr)
        print(f"  Enabling distributed mode: DP={dp_size}, TP={tp_size} ({total_devices} devices)")
        ttml.core.distributed.enable_fabric(total_devices)

    ctx = ttml.autograd.AutoContext.get_instance()
    if distributed:
        ctx.open_device([dp_size, tp_size])
        ctx.initialize_parallelism_context(
            ttml.autograd.DistributedConfig(enable_ddp=dp_size > 1, enable_tp=tp_size > 1)
        )
    else:
        ctx.open_device()
    ctx.set_seed(seed)
    return ctx, ctx.get_device()


# =====================================================================
# Main generation loop
# =====================================================================


def generate(
    *,
    checkpoint_name: str,
    prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    seed: int,
    dp_size: int,
    tp_size: int,
    output_path: str,
    guidance_scale: float,
):
    print(f"Checkpoint:     {checkpoint_name}")
    print(f"Prompt:         {prompt!r}")
    print(f"Size:           {width}×{height}")
    print(f"Steps:          {num_inference_steps}")
    print(f"Seed:           {seed}")
    print(f"Mesh:           [{dp_size}, {tp_size}]")

    # ---- 1. Load HF transformer config + weights ----
    from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel

    print("\nLoading HF FluxTransformer2DModel (for config + weights) ...")
    torch_transformer = FluxTransformer2DModel.from_pretrained(
        checkpoint_name, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    torch_transformer.eval()
    hf_state_dict = torch_transformer.state_dict()
    config = create_flux1_config_from_hf(torch_transformer.config)
    pos_embed_fn = torch_transformer.pos_embed

    # ---- 2. Encode prompts (torch CPU) ----
    print("\nEncoding prompts ...")
    prompt_embeds, pooled_prompt_embeds = encode_prompts(prompt, checkpoint_name, config.joint_attention_dim)
    _, prompt_seq_len, _ = prompt_embeds.shape

    # ---- 3. Setup device + create ttml model ----
    print("\nSetting up device ...")
    ctx, device = setup_device(dp_size, tp_size, seed=seed)

    shard_dim = 1 if tp_size > 1 else None
    print(f"\nCreating ttml Flux1 transformer (TP={tp_size}) ...")
    with empty_init():
        model = DistributedFlux1Transformer(config, shard_dim=shard_dim)

    print("Loading weights ...")
    load_weights_from_hf_distributed(model, hf_state_dict, config, shard_dim=shard_dim)
    del hf_state_dict, torch_transformer

    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    model.eval()

    # ---- 4. Prepare scheduler ----
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")
    from diffusers import AutoencoderKL

    tmp_vae = AutoencoderKL.from_pretrained(checkpoint_name, subfolder="vae")
    vae_scale_factor = 2 ** len(tmp_vae.config.block_out_channels)
    num_channels_latents = config.in_channels // 4
    del tmp_vae

    latents_height = height // vae_scale_factor
    latents_width = width // vae_scale_factor
    spatial_seq_len = latents_height * latents_width

    scheduler.set_timesteps(
        sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps),
        mu=_calculate_shift(
            spatial_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        ),
    )

    # ---- 5. Prepare latents, RoPE, inputs ----
    print("\nPreparing latents and RoPE ...")
    torch.manual_seed(seed)
    latents_shape = [1, num_channels_latents, latents_height * 2, latents_width * 2]
    latents = _pack_latents(
        torch.randn(latents_shape, dtype=torch.bfloat16), 1, num_channels_latents, latents_height, latents_width
    )

    text_ids = torch.zeros([prompt_seq_len, 3])
    image_ids = _latent_image_ids(height=latents_height, width=latents_width)
    ids = torch.cat((text_ids, image_ids), dim=0)
    rope_cos, rope_sin = pos_embed_fn.forward(ids)
    del pos_embed_fn

    spatial_rope_cos = rope_cos[prompt_seq_len:].to(torch.bfloat16)
    spatial_rope_sin = rope_sin[prompt_seq_len:].to(torch.bfloat16)
    prompt_rope_cos = rope_cos[:prompt_seq_len].to(torch.bfloat16)
    prompt_rope_sin = rope_sin[:prompt_seq_len].to(torch.bfloat16)

    spatial_rope_cos_np = spatial_rope_cos.float().numpy()
    spatial_rope_sin_np = spatial_rope_sin.float().numpy()
    prompt_rope_cos_np = prompt_rope_cos.float().numpy()
    prompt_rope_sin_np = prompt_rope_sin.float().numpy()

    rope_cos_s = ttml.autograd.Tensor.from_numpy(
        spatial_rope_cos_np.reshape(1, 1, *spatial_rope_cos_np.shape), ttnn.Layout.TILE, ttnn.bfloat16
    )
    rope_sin_s = ttml.autograd.Tensor.from_numpy(
        spatial_rope_sin_np.reshape(1, 1, *spatial_rope_sin_np.shape), ttnn.Layout.TILE, ttnn.bfloat16
    )
    rope_cos_p = ttml.autograd.Tensor.from_numpy(
        prompt_rope_cos_np.reshape(1, 1, *prompt_rope_cos_np.shape), ttnn.Layout.TILE, ttnn.bfloat16
    )
    rope_sin_p = ttml.autograd.Tensor.from_numpy(
        prompt_rope_sin_np.reshape(1, 1, *prompt_rope_sin_np.shape), ttnn.Layout.TILE, ttnn.bfloat16
    )

    # Prompt and pooled embeddings → ttml
    prompt_embeds_4d = prompt_embeds.unsqueeze(0).float().numpy()  # [1, 1, seq, dim]
    pooled_4d = pooled_prompt_embeds.unsqueeze(0).unsqueeze(0).float().numpy()  # [1, 1, 1, dim]

    tt_prompt = ttml.autograd.Tensor.from_numpy(prompt_embeds_4d, ttnn.Layout.TILE, ttnn.bfloat16)
    tt_pooled = ttml.autograd.Tensor.from_numpy(pooled_4d, ttnn.Layout.TILE, ttnn.bfloat16)

    guidance = (
        torch.full([1], fill_value=guidance_scale, dtype=torch.bfloat16) if config.guidance_embeds else None
    )

    # ---- 6. Denoising loop (latents stay on device) ----
    print(f"\nDenoising ({num_inference_steps} steps) ...")
    step_times = []

    latents_4d = latents.unsqueeze(0).float().numpy()  # [1, 1, spatial_seq, in_channels]
    tt_latents = ttml.autograd.Tensor.from_numpy(latents_4d, ttnn.Layout.TILE, ttnn.bfloat16)

    with no_grad():
      for i, t in enumerate(tqdm_mod.tqdm(scheduler.timesteps)):
        t0 = time.perf_counter()

        sigma_difference = float(scheduler.sigmas[i + 1] - scheduler.sigmas[i])

        timestep_proj = _sinusoidal_proj(torch.tensor([float(t)], dtype=torch.float32))
        tt_timestep_proj = ttml.autograd.Tensor.from_numpy(
            timestep_proj.float().numpy().reshape(1, 1, 1, -1), ttnn.Layout.TILE, ttnn.bfloat16
        )

        if guidance is not None:
            guidance_proj = _sinusoidal_proj(guidance * 1000.0)
            tt_guidance_proj = ttml.autograd.Tensor.from_numpy(
                guidance_proj.float().numpy().reshape(1, 1, 1, -1), ttnn.Layout.TILE, ttnn.bfloat16
            )
        else:
            tt_guidance_proj = None

        noise_pred = model(
            tt_latents,
            tt_prompt,
            tt_timestep_proj,
            tt_guidance_proj,
            tt_pooled,
            rope_cos_s,
            rope_sin_s,
            rope_cos_p,
            rope_sin_p,
        )

        pred_val = noise_pred.get_value()
        scaled_pred = ttnn.multiply(pred_val, sigma_difference)
        new_latent = ttnn.add(tt_latents.get_value(), scaled_pred)

        ttml.autograd.AutoContext.get_instance().reset_graph()

        tt_latents = ttml.autograd.create_tensor(new_latent)
        step_times.append(time.perf_counter() - t0)

    # ---- 7. Read latents back to CPU, then VAE decode ----
    print("\nDecoding with VAE ...")
    distributed = tp_size > 1
    if distributed:
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        latents_np = tt_latents.to_numpy(composer=composer).astype(np.float32)[:1]
    else:
        latents_np = tt_latents.to_numpy().astype(np.float32)

    latents = torch.from_numpy(latents_np).squeeze(0).to(torch.bfloat16)
    images = decode_latents(latents, checkpoint_name, vae_scale_factor, height, width)

    # ---- 8. Save ----
    images[0].save(output_path)
    print(f"\nSaved image to {output_path}")

    # ---- Timing summary ----
    print(f"\n{'Step':>5} {'Time (ms)':>10}")
    print("-" * 18)
    for idx, st in enumerate(step_times):
        print(f"{idx:>5} {st * 1000:>10.1f}")
    if step_times:
        avg = sum(step_times) / len(step_times)
        print(f"\n  Average: {avg * 1000:.1f} ms/step")

    ctx.close_device()


# =====================================================================
# CLI
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Flux1 image generation with ttml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="HF checkpoint (default: black-forest-labs/FLUX.1-dev)",
    )
    parser.add_argument("--prompt", type=str, default="A luxury sports car.")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument(
        "--mesh_shape",
        type=int,
        nargs=2,
        default=[1, 8],
        metavar=("DP", "TP"),
        help="Device mesh [dp, tp]. Default: 1 2.",
    )
    parser.add_argument("--output", type=str, default="flux-output.png")
    args = parser.parse_args()

    dp_size, tp_size = args.mesh_shape
    if tp_size < 1:
        parser.error("TP must be >= 1")

    if args.num_inference_steps is None:
        args.num_inference_steps = 4 if "schnell" in args.checkpoint.lower() else 28

    generate(
        checkpoint_name=args.checkpoint,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        dp_size=dp_size,
        tp_size=tp_size,
        output_path=args.output,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()
