# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Flux1 LoRA fine-tuning script using ttml.

Fine-tunes ONLY the transformer (DiT) part of the Flux pipeline using
LoRA adapters.  Text encoders (CLIP + T5) run frozen on device via ttnn
models from models/tt_dit.  VAE encoding uses torch (no ttnn VAE encoder).
The training objective is rectified flow matching loss.

LoRA scope: adapters are injected into modules whose attribute names
are listed in ``--lora_targets``. Both double (joint) and single
blocks are covered. All other parameters (embedders, final norm/
proj_out, etc.) stay frozen.

Note: target matching is by attribute name only, so e.g. ``ff1`` hits
both the spatial branch (``ff.ff1``) and the prompt branch
(``ff_context.ff1``) of every double block.

Uses the valhalla/pokemon-dataset for image-caption training data.

Usage:
    cd tt-train/sources/examples/flux1

    # Basic LoRA fine-tuning on Pokemon dataset:
    python train.py \\
        --checkpoint black-forest-labs/FLUX.1-dev \\
        --mesh_shape 1 8 \\
        --steps 10000 --lr 1e-4 --lora_rank 64 \\
        --save_dir ./output

    # Customise resolution, batch size, and LoRA scope:
    python train.py \\
        --checkpoint black-forest-labs/FLUX.1-dev \\
        --mesh_shape 1 8 \\
        --resolution 512 \\
        --steps 10000 --lr 5e-5 --lora_rank 16 \\
        --save_dir ./output

    # View TensorBoard logs:
    tensorboard --logdir ./output/logs
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import ttnn
import ttml

# Add repo root to path so `models.tt_dit` is importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model_flux_distributed import (
    DistributedFlux1Transformer,
    create_flux1_config_from_hf,
    empty_init,
    load_weights_from_hf_distributed,
)
from generate import (
    _sinusoidal_proj,
    _pack_latents,
    _latent_image_ids,
    _calculate_shift,
    setup_device,
    encode_prompts,
    decode_latents,
)
from lora import inject_lora, LORA_TARGETS_ALL


# =====================================================================
# LR schedules
# =====================================================================


def cosine_lr_schedule(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def constant_lr_schedule(step, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    return max_lr


# =====================================================================
# Dataset: valhalla/pokemon-dataset
# =====================================================================


class PokemonDataset:
    """Lazy-loading wrapper around valhalla/pokemon-dataset.

    Pre-encodes all captions with CLIP + T5 at init time using ttnn
    encoders on device (fast).  Images are loaded and VAE-encoded
    on-the-fly via torch (no ttnn VAE encoder exists).
    """

    T5_SEQUENCE_LENGTH = 512

    def __init__(
        self,
        checkpoint_name: str,
        joint_attention_dim: int,
        resolution: int,
        vae_scale_factor: int,
        mesh_device,
        cache_dir: str | None = None,
        eval_prompt: str | None = None,
    ):
        from datasets import load_dataset
        from torchvision import transforms

        self.resolution = resolution
        self.vae_scale_factor = vae_scale_factor
        self.checkpoint_name = checkpoint_name
        self.eval_prompt = eval_prompt

        print("Loading valhalla/pokemon-dataset ...")
        ds = load_dataset("valhalla/pokemon-dataset", split="train")
        self.dataset = ds
        self.num_samples = len(ds)
        print(f"  {self.num_samples} training samples")

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.eval_prompt_embeds = None
        self.eval_pooled_embeds = None

        self._load_vae(checkpoint_name)
        self._encode_all_prompts(checkpoint_name, joint_attention_dim, mesh_device, cache_dir)

    def _load_vae(self, checkpoint_name):
        from diffusers import AutoencoderKL

        print("  Loading VAE for encoding ...")
        self.vae = AutoencoderKL.from_pretrained(checkpoint_name, subfolder="vae")
        self.vae.eval()
        self.vae_scaling_factor = self.vae.config.scaling_factor
        self.vae_shift_factor = self.vae.config.shift_factor

    def _encode_all_prompts(self, checkpoint_name, joint_attention_dim, mesh_device, cache_dir):
        """Pre-encode all text prompts with ttnn CLIP + T5 encoders on device.

        Dataset prompts are cached to disk; the eval prompt is ALWAYS encoded
        fresh at startup (never cached). When the dataset cache is hit we
        still spin up the ttnn encoders briefly to encode the eval prompt,
        then tear them down before training begins.
        """
        cache_path = None
        dataset_cache_hit = False
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "pokemon_prompt_embeds.pt")
            if os.path.exists(cache_path):
                print(f"  Loading cached prompt embeddings from {cache_path}")
                cached = torch.load(cache_path, weights_only=True)
                self.all_prompt_embeds = cached["prompt_embeds"]
                self.all_pooled_embeds = cached["pooled_embeds"]
                dataset_cache_hit = True

        if dataset_cache_hit and self.eval_prompt is None:
            return

        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        from models.tt_dit.encoders.clip.model_clip import CLIPConfig, CLIPEncoder
        from models.tt_dit.encoders.t5.model_t5 import T5Config, T5Encoder
        from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
        from models.tt_dit.parallel.manager import CCLManager

        print("  Loading tokenizers ...")
        clip_tokenizer = CLIPTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")
        t5_tokenizer = T5TokenizerFast.from_pretrained(checkpoint_name, subfolder="tokenizer_2")

        print("  Loading torch CLIP + T5 (for config + weights) ...")
        torch_clip = CLIPTextModel.from_pretrained(checkpoint_name, subfolder="text_encoder")
        torch_clip.eval()
        torch_t5 = T5EncoderModel.from_pretrained(checkpoint_name, subfolder="text_encoder_2")
        torch_t5.eval()

        encoder_parallel = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=1, mesh_axis=0)
        )
        ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

        print("  Creating ttnn CLIP encoder ...")
        clip_config = CLIPConfig(
            vocab_size=torch_clip.config.vocab_size,
            embed_dim=torch_clip.config.hidden_size,
            ff_dim=torch_clip.config.intermediate_size,
            num_heads=torch_clip.config.num_attention_heads,
            num_hidden_layers=torch_clip.config.num_hidden_layers,
            max_prompt_length=77,
            layer_norm_eps=torch_clip.config.layer_norm_eps,
            attention_dropout=torch_clip.config.attention_dropout,
            hidden_act=torch_clip.config.hidden_act,
        )
        tt_clip = CLIPEncoder(
            config=clip_config,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=encoder_parallel,
            eos_token_id=2,
        )
        clip_sd = torch_clip.state_dict()
        del torch_clip
        tt_clip.load_torch_state_dict(clip_sd)
        del clip_sd

        print("  Creating ttnn T5 encoder ...")
        t5_config = T5Config(
            vocab_size=torch_t5.config.vocab_size,
            embed_dim=torch_t5.config.d_model,
            ff_dim=torch_t5.config.d_ff,
            kv_dim=torch_t5.config.d_kv,
            num_heads=torch_t5.config.num_heads,
            num_hidden_layers=torch_t5.config.num_layers,
            max_prompt_length=self.T5_SEQUENCE_LENGTH,
            layer_norm_eps=torch_t5.config.layer_norm_epsilon,
            relative_attention_num_buckets=torch_t5.config.relative_attention_num_buckets,
            relative_attention_max_distance=torch_t5.config.relative_attention_max_distance,
        )
        tt_t5 = T5Encoder(
            config=t5_config,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=encoder_parallel,
        )
        t5_sd = torch_t5.state_dict()
        del torch_t5
        tt_t5.load_torch_state_dict(t5_sd)
        del t5_sd

        ttnn.synchronize_device(mesh_device)

        def _encode_caption(caption):
            clip_tokens = clip_tokenizer(
                [caption],
                return_tensors="pt",
                padding="max_length",
                max_length=clip_tokenizer.model_max_length,
                truncation=True,
            ).input_ids
            tt_clip_tokens = ttnn.from_torch(
                clip_tokens,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
            )
            _, tt_pooled = tt_clip(prompt_tokenized=tt_clip_tokens, mesh_device=mesh_device)
            pooled = ttnn.to_torch(ttnn.get_device_tensors(tt_pooled)[0]).to(torch.bfloat16)

            t5_tokens = t5_tokenizer(
                [caption],
                return_tensors="pt",
                padding="max_length",
                max_length=self.T5_SEQUENCE_LENGTH,
                truncation=True,
            ).input_ids
            tt_t5_tokens = ttnn.from_torch(
                t5_tokens,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
            )
            tt_t5_out = tt_t5(prompt=tt_t5_tokens)
            tt_prompt_embeds = tt_t5_out[-1]
            prompt_emb = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0]).to(torch.bfloat16)
            return prompt_emb.cpu(), pooled.cpu()

        if not dataset_cache_hit:
            all_prompt_embeds = []
            all_pooled_embeds = []

            print("  Encoding all prompts with ttnn encoders ...")
            for i in tqdm(range(self.num_samples), desc="  Encoding prompts"):
                caption = self.dataset[i]['name']
                prompt_emb, pooled = _encode_caption(caption)
                all_prompt_embeds.append(prompt_emb)
                all_pooled_embeds.append(pooled)

            self.all_prompt_embeds = all_prompt_embeds
            self.all_pooled_embeds = all_pooled_embeds

            if cache_path:
                print(f"  Saving prompt embeddings cache to {cache_path}")
                torch.save(
                    {"prompt_embeds": all_prompt_embeds, "pooled_embeds": all_pooled_embeds},
                    cache_path,
                )

        if self.eval_prompt is not None:
            print(f"  Encoding eval prompt: {self.eval_prompt!r}")
            self.eval_prompt_embeds, self.eval_pooled_embeds = _encode_caption(self.eval_prompt)

        del tt_clip, tt_t5
        ttnn.synchronize_device(mesh_device)

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to VAE latent space, pack to Flux format."""
        pixel_values = self.transform(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            latent_dist = self.vae.encode(pixel_values.float())
            latents = latent_dist.latent_dist.sample()
        latents = (latents - self.vae_shift_factor) * self.vae_scaling_factor

        num_channels_latents = latents.shape[1]
        latents_h = self.resolution // self.vae_scale_factor
        latents_w = self.resolution // self.vae_scale_factor
        latents = _pack_latents(latents.to(torch.bfloat16), 1, num_channels_latents, latents_h, latents_w)
        return latents

    def get_batch(self, batch_size: int, rng: np.random.RandomState | None = None):
        """Sample a random batch of (latents, prompt_embeds, pooled_embeds)."""
        if rng is None:
            indices = np.random.randint(0, self.num_samples, size=batch_size)
        else:
            indices = rng.randint(0, self.num_samples, size=batch_size)

        batch_latents = []
        batch_prompt = []
        batch_pooled = []

        for idx in indices:
            sample = self.dataset[int(idx)]
            image = sample["image"]
            latents = self.encode_image(image)
            batch_latents.append(latents)
            batch_prompt.append(self.all_prompt_embeds[int(idx)])
            batch_pooled.append(self.all_pooled_embeds[int(idx)])

        latents = torch.cat(batch_latents, dim=0)
        prompt_embeds = torch.cat(batch_prompt, dim=0)
        pooled_embeds = torch.cat(batch_pooled, dim=0)
        return latents, prompt_embeds, pooled_embeds


# =====================================================================
# Rectified flow loss
# =====================================================================


def rectified_flow_loss(model, noisy_latents_tt, target_velocity_tt,
                        timestep_tt, prompt_tt, pooled_tt,
                        guidance_proj_tt, rope_cos_s, rope_sin_s, rope_cos_p, rope_sin_p):
    """Compute rectified flow matching loss: MSE(v_pred, v_target).

    v_target = noise - latents  (the velocity from data to noise)
    noisy_latents = (1 - sigma) * latents + sigma * noise
    v_pred = model(noisy_latents, ...)
    Loss = MSE(v_pred, v_target)
    """
    v_pred = model(
        noisy_latents_tt,
        prompt_tt,
        timestep_tt,
        guidance_proj_tt,
        pooled_tt,
        rope_cos_s,
        rope_sin_s,
        rope_cos_p,
        rope_sin_p,
    )
    loss = ttml.ops.loss.mse_loss(v_pred, target_velocity_tt)
    return loss


# =====================================================================
# Inference (generate a sample image with current weights)
# =====================================================================


def generate_sample(model, config, checkpoint_name, prompt, resolution,
                    num_inference_steps, seed, device, distributed, tp_size,
                    pos_embed_fn=None, prompt_embeds=None, pooled_prompt_embeds=None):
    """Generate an image using the current model weights (evaluation mode).

    If pos_embed_fn is provided, uses it directly instead of reloading the
    HuggingFace transformer model.

    If prompt_embeds and pooled_prompt_embeds are provided, they are used
    directly and text encoders are NOT loaded (fast path for training).
    """
    from ttml.common.utils import no_grad
    from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL

    model.eval()
    ctx = ttml.autograd.AutoContext.get_instance()

    if pos_embed_fn is None:
        from diffusers import FluxTransformer2DModel
        hf_transformer = FluxTransformer2DModel.from_pretrained(
            checkpoint_name, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pos_embed_fn = hf_transformer.pos_embed
        del hf_transformer

    if prompt_embeds is None or pooled_prompt_embeds is None:
        raise RuntimeError(
            "generate_sample requires pre-computed prompt_embeds and pooled_prompt_embeds; "
            "none were provided. The eval prompt must be encoded once at startup "
            "(see PokemonDataset._encode_all_prompts) and reused for every generation "
            "so the text encoders are never reloaded during training."
        )
    _, prompt_seq_len, _ = prompt_embeds.shape

    tmp_vae = AutoencoderKL.from_pretrained(checkpoint_name, subfolder="vae")
    vae_scale_factor = 2 ** len(tmp_vae.config.block_out_channels)
    num_channels_latents = config.in_channels // 4
    del tmp_vae

    height = width = resolution
    latents_height = height // vae_scale_factor
    latents_width = width // vae_scale_factor
    spatial_seq_len = latents_height * latents_width

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")
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

    torch.manual_seed(seed)
    latents_shape = [1, num_channels_latents, latents_height * 2, latents_width * 2]
    latents = _pack_latents(
        torch.randn(latents_shape, dtype=torch.bfloat16), 1, num_channels_latents, latents_height, latents_width
    )

    text_ids = torch.zeros([prompt_seq_len, 3])
    image_ids = _latent_image_ids(height=latents_height, width=latents_width)
    ids = torch.cat((text_ids, image_ids), dim=0)
    rope_cos, rope_sin = pos_embed_fn.forward(ids)

    spatial_rope_cos = rope_cos[prompt_seq_len:].to(torch.bfloat16)
    spatial_rope_sin = rope_sin[prompt_seq_len:].to(torch.bfloat16)
    prompt_rope_cos = rope_cos[:prompt_seq_len].to(torch.bfloat16)
    prompt_rope_sin = rope_sin[:prompt_seq_len].to(torch.bfloat16)

    def _to_ttml(arr):
        return ttml.autograd.Tensor.from_numpy(arr, ttnn.Layout.TILE, ttnn.bfloat16)

    rope_cos_s = _to_ttml(spatial_rope_cos.float().numpy().reshape(1, 1, *spatial_rope_cos.shape))
    rope_sin_s = _to_ttml(spatial_rope_sin.float().numpy().reshape(1, 1, *spatial_rope_sin.shape))
    rope_cos_p = _to_ttml(prompt_rope_cos.float().numpy().reshape(1, 1, *prompt_rope_cos.shape))
    rope_sin_p = _to_ttml(prompt_rope_sin.float().numpy().reshape(1, 1, *prompt_rope_sin.shape))

    prompt_embeds_4d = prompt_embeds.unsqueeze(0).float().numpy()
    pooled_4d = pooled_prompt_embeds.unsqueeze(0).unsqueeze(0).float().numpy()
    tt_prompt = _to_ttml(prompt_embeds_4d)
    tt_pooled = _to_ttml(pooled_4d)

    guidance = torch.full([1], fill_value=3.5, dtype=torch.bfloat16) if config.guidance_embeds else None

    latents_4d = latents.unsqueeze(0).float().numpy()
    tt_latents = _to_ttml(latents_4d)

    with no_grad():
        for i, t in enumerate(scheduler.timesteps):
            sigma_difference = float(scheduler.sigmas[i + 1] - scheduler.sigmas[i])

            timestep_proj = _sinusoidal_proj(torch.tensor([float(t)], dtype=torch.float32))
            tt_timestep_proj = _to_ttml(timestep_proj.float().numpy().reshape(1, 1, 1, -1))

            if guidance is not None:
                guidance_proj = _sinusoidal_proj(guidance * 1000.0)
                tt_guidance_proj = _to_ttml(guidance_proj.float().numpy().reshape(1, 1, 1, -1))
            else:
                tt_guidance_proj = None

            noise_pred = model(
                tt_latents, tt_prompt, tt_timestep_proj, tt_guidance_proj, tt_pooled,
                rope_cos_s, rope_sin_s, rope_cos_p, rope_sin_p,
            )

            pred_val = noise_pred.get_value()
            scaled_pred = ttnn.multiply(pred_val, sigma_difference)
            new_latent = ttnn.add(tt_latents.get_value(), scaled_pred)
            ctx.reset_graph()
            tt_latents = ttml.autograd.create_tensor(new_latent)

    if distributed:
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        latents_np = tt_latents.to_numpy(composer=composer).astype(np.float32)[:1]
    else:
        latents_np = tt_latents.to_numpy().astype(np.float32)

    latents_torch = torch.from_numpy(latents_np).squeeze(0).to(torch.bfloat16)
    images = decode_latents(latents_torch, checkpoint_name, vae_scale_factor, height, width)
    return images[0]


# =====================================================================
# Main training function
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Flux1 LoRA fine-tuning with ttml")
    parser.add_argument("--checkpoint", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="HF checkpoint (default: black-forest-labs/FLUX.1-dev)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Training image resolution (images are center-cropped)")
    parser.add_argument("--mesh_shape", type=int, nargs=2, default=[1, 8],
                        metavar=("DP", "TP"), help="Device mesh [dp, tp]")

    # Training
    parser.add_argument("--batch_size", type=int, default=1, help="Micro-batch size")
    parser.add_argument("--steps", type=int, default=10000, help="Total optimizer steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for cosine schedule")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Linear warmup steps")
    parser.add_argument("--lr_schedule", type=str, default="constant",
                        choices=["cosine", "constant"], help="LR schedule")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--clip_grad_norm", type=float, default=0,
                        help="Max gradient norm for clipping (0=disabled)")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="LoRA scaling alpha (default: same as rank)")
    parser.add_argument("--lora_targets", type=str, nargs="+",
                        default=[
                            "to_qkv",
                            "add_qkv_proj",
                            "to_out",
                            "to_add_out",
                            "ff1",
                            "ff2",
                            "norm1_linear",
                            "norm1_context_linear",
                            "proj_mlp",
                            "proj_out",
                            "time_embed",
                        ],
                        help="LoRA target modules. Valid: " + ", ".join(LORA_TARGETS_ALL))

    # Evaluation
    parser.add_argument("--gen_every", type=int, default=10,
                        help="Generate sample image every N steps")
    parser.add_argument("--gen_prompt", type=str, default="blaziken",
                        help="Prompt for periodic sample generation")

    # Output
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory for checkpoints, logs, and generated images")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save LoRA checkpoint every N steps (0=only at end)")
    parser.add_argument("--output_log", type=str, default="train_log.txt",
                        help="Path to training log file")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default="prompt_cache",
                        help="Directory for caching pre-encoded prompt embeddings")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Inference steps for generation (default: 4 for schnell, 28 for dev)")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.num_inference_steps is None:
        args.num_inference_steps = 4 if "schnell" in args.checkpoint.lower() else 28

    # ------------------------------------------------------------------
    # TensorBoard setup (graceful fallback)
    # ------------------------------------------------------------------
    tb_writer = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))
        except ImportError:
            print(
                "WARNING: tensorboard is not installed — TensorBoard logging disabled. "
                "Install with: pip install tensorboard"
            )

    # ------------------------------------------------------------------
    # 1. Load HF transformer config + weights
    # ------------------------------------------------------------------
    from diffusers import FluxTransformer2DModel, AutoencoderKL

    print(f"\nCheckpoint:  {args.checkpoint}")
    print(f"Resolution:  {args.resolution}")

    print("\nLoading HF FluxTransformer2DModel ...")
    torch_transformer = FluxTransformer2DModel.from_pretrained(
        args.checkpoint, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    torch_transformer.eval()
    hf_state_dict = torch_transformer.state_dict()
    config = create_flux1_config_from_hf(torch_transformer.config)
    pos_embed_fn = torch_transformer.pos_embed
    del torch_transformer

    # ------------------------------------------------------------------
    # 2. Get VAE scale factor
    # ------------------------------------------------------------------
    tmp_vae = AutoencoderKL.from_pretrained(args.checkpoint, subfolder="vae")
    vae_scale_factor = 2 ** len(tmp_vae.config.block_out_channels)
    del tmp_vae

    # ------------------------------------------------------------------
    # 3. Prepare RoPE embeddings for training resolution
    # ------------------------------------------------------------------
    height = width = args.resolution
    latents_height = height // vae_scale_factor
    latents_width = width // vae_scale_factor
    prompt_seq_len = 512

    text_ids = torch.zeros([prompt_seq_len, 3])
    image_ids = _latent_image_ids(height=latents_height, width=latents_width)
    ids = torch.cat((text_ids, image_ids), dim=0)
    rope_cos, rope_sin = pos_embed_fn.forward(ids)

    spatial_rope_cos = rope_cos[prompt_seq_len:].to(torch.bfloat16)
    spatial_rope_sin = rope_sin[prompt_seq_len:].to(torch.bfloat16)
    prompt_rope_cos = rope_cos[:prompt_seq_len].to(torch.bfloat16)
    prompt_rope_sin = rope_sin[:prompt_seq_len].to(torch.bfloat16)

    # ------------------------------------------------------------------
    # 4. Setup device (needed for ttnn encoders in dataset)
    # ------------------------------------------------------------------
    dp_size, tp_size = args.mesh_shape
    distributed = tp_size > 1 or dp_size > 1

    print("\nSetting up device ...")
    ctx, device = setup_device(dp_size, tp_size, seed=args.seed)

    # ------------------------------------------------------------------
    # 5. Setup dataset (uses ttnn CLIP + T5 encoders on device)
    # ------------------------------------------------------------------
    print("\nLoading dataset ...")
    dataset = PokemonDataset(
        checkpoint_name=args.checkpoint,
        joint_attention_dim=config.joint_attention_dim,
        resolution=args.resolution,
        vae_scale_factor=vae_scale_factor,
        mesh_device=device,
        cache_dir=args.cache_dir,
        eval_prompt=args.gen_prompt,
    )

    # ------------------------------------------------------------------
    # 6. Create ttml model and load weights
    # ------------------------------------------------------------------
    shard_dim = 1 if tp_size > 1 else None
    print(f"\nCreating ttml Flux1 transformer (TP={tp_size}) ...")
    with empty_init():
        model = DistributedFlux1Transformer(config, shard_dim=shard_dim, use_checkpoint=True)

    print("Loading pretrained weights ...")
    load_weights_from_hf_distributed(model, hf_state_dict, config, shard_dim=shard_dim)
    del hf_state_dict

    # ------------------------------------------------------------------
    # 7. Inject LoRA adapters
    #
    # Targets are driven by --lora_targets (matched by attribute name).
    # Both double (joint) and single blocks get LoRA injected.
    # NOTE: matching is by attribute name only, so listing ff1/ff2 hits
    # both spatial (ff) and prompt (ff_context) branches in double blocks.
    # ------------------------------------------------------------------
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else float(args.lora_rank)

    unknown_targets = [t for t in args.lora_targets if t not in LORA_TARGETS_ALL]
    if unknown_targets:
        parser.error(
            f"--lora_targets contains unknown names: {unknown_targets}. "
            f"Valid: {LORA_TARGETS_ALL}"
        )

    lora_config = {
        "targets": list(args.lora_targets),
        "rank": args.lora_rank,
        "alpha": lora_alpha,
    }
    print(
        f"\nInjecting LoRA: rank={args.lora_rank}, alpha={lora_alpha}, "
        f"targets={args.lora_targets}\n"
        f"  scope: all transformer_blocks + all single_transformer_blocks"
    )
    for block in model.transformer_blocks:
        inject_lora(block, lora_config)
    for block in model.single_transformer_blocks:
        inject_lora(block, lora_config)
    n_lora = sum(1 for n, _ in model.parameters().items() if "lora_A" in n)
    print(f"  Injected {n_lora} LoRA adapters")

    # ------------------------------------------------------------------
    # 7. Select trainable parameters (LoRA only)
    # ------------------------------------------------------------------
    all_params = model.parameters()
    trainable_params = {name: param for name, param in all_params.items() if "lora" in name}
    frozen_count = len(all_params) - len(trainable_params)
    print(f"\nLoRA fine-tuning: {len(trainable_params)} LoRA params trainable, {frozen_count} base params frozen")

    non_trainable_params = {name: param for name, param in all_params.items() if name not in trainable_params}
    for name, weight in non_trainable_params.items():
        weight.tensor.set_requires_grad(False)
    print(f"Set requires_grad=False for {len(non_trainable_params)} frozen params")

    # ------------------------------------------------------------------
    # 8. Setup optimizer
    # ------------------------------------------------------------------
    print("Setting up optimizer ...")
    adamw_config = ttml.optimizers.AdamWConfig.make(
        args.lr, args.beta1, args.beta2, args.eps, args.weight_decay,
    )
    optimizer = ttml.optimizers.AdamW(trainable_params, adamw_config)

    # ------------------------------------------------------------------
    # 9. Prepare static ttml tensors (RoPE)
    # ------------------------------------------------------------------
    def _to_ttml(arr):
        return ttml.autograd.Tensor.from_numpy(arr, ttnn.Layout.TILE, ttnn.bfloat16)

    rope_cos_s = _to_ttml(spatial_rope_cos.float().numpy().reshape(1, 1, *spatial_rope_cos.shape))
    rope_sin_s = _to_ttml(spatial_rope_sin.float().numpy().reshape(1, 1, *spatial_rope_sin.shape))
    rope_cos_p = _to_ttml(prompt_rope_cos.float().numpy().reshape(1, 1, *prompt_rope_cos.shape))
    rope_sin_p = _to_ttml(prompt_rope_sin.float().numpy().reshape(1, 1, *prompt_rope_sin.shape))

    # ------------------------------------------------------------------
    # 10. Training loop
    # ------------------------------------------------------------------
    ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    model.train()

    total_steps = args.steps
    accum_steps = args.gradient_accumulation_steps
    use_clip = args.clip_grad_norm > 0.0
    rng = np.random.RandomState(args.seed)

    print(f"\nTraining config:")
    print(f"  Steps: {total_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {accum_steps}")
    print(f"  Peak LR: {args.lr}")
    print(f"  LR schedule: {args.lr_schedule}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Gradient clipping: {'%.1f' % args.clip_grad_norm if use_clip else 'disabled'}")
    print(
        f"  LoRA: rank={args.lora_rank}, alpha={lora_alpha}, "
        f"targets={args.lora_targets}"
    )
    if args.save_dir:
        print(f"  Output dir: {args.save_dir}")
        if tb_writer is not None:
            tb_writer.add_text("config", f"```\n{vars(args)}\n```", 0)

    # ------------------------------------------------------------------
    # Generate sample at step 0 (before any training)
    # ------------------------------------------------------------------
    if args.save_dir and args.gen_every > 0:
        print("\n--- Generating sample at step 0 (before training) ---")
        try:
            img = generate_sample(
                model, config, args.checkpoint, args.gen_prompt, args.resolution,
                args.num_inference_steps, args.seed, device, distributed, tp_size,
                pos_embed_fn=pos_embed_fn,
                prompt_embeds=dataset.eval_prompt_embeds,
                pooled_prompt_embeds=dataset.eval_pooled_embeds,
            )
            img_dir = os.path.join(args.save_dir, "samples")
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, "step_0000.png")
            img.save(img_path)
            print(f"  Saved sample to {img_path}")
            if tb_writer is not None:
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                tb_writer.add_image("samples", img_tensor, 0)
        except Exception as e:
            print(f"  Sample generation failed: {e}")

        ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
        model.train()

    print(f"\n{'=' * 70}")
    print("Starting training...")
    print(f"{'=' * 70}\n")

    train_losses = []
    log_lines = []
    train_start = time.time()

    bar = tqdm(range(1, total_steps + 1), desc="Training")
    for step in bar:
        step_start = time.time()

        # LR schedule
        if args.lr_schedule == "constant":
            lr_now = constant_lr_schedule(step - 1, args.warmup_steps, args.lr)
        else:
            lr_now = cosine_lr_schedule(step - 1, args.warmup_steps, total_steps, args.lr, args.min_lr)
        optimizer.set_lr(lr_now)

        optimizer.zero_grad()

        accum_loss = 0.0
        for micro_step in range(accum_steps):
            latents_torch, prompt_embeds, pooled_embeds = dataset.get_batch(args.batch_size, rng)

            sigma = torch.rand(1).item()
            noise = torch.randn_like(latents_torch)

            # Rectified flow: interpolate between data and noise
            noisy_latents = ((1.0 - sigma) * latents_torch + sigma * noise).to(torch.bfloat16)
            # Velocity target: direction from data to noise
            velocity_target = (noise - latents_torch).to(torch.bfloat16)

            timestep = torch.tensor([sigma * 1000.0], dtype=torch.float32)
            timestep_proj = _sinusoidal_proj(timestep)

            tt_noisy = _to_ttml(noisy_latents.unsqueeze(0).float().numpy())
            tt_velocity = _to_ttml(velocity_target.unsqueeze(0).float().numpy())
            tt_prompt = _to_ttml(prompt_embeds.unsqueeze(0).float().numpy())
            tt_pooled = _to_ttml(pooled_embeds.unsqueeze(0).unsqueeze(0).float().numpy())
            tt_timestep = _to_ttml(timestep_proj.float().numpy().reshape(1, 1, 1, -1))

            # Flux-dev was distilled with classifier-free guidance, so during
            # LoRA fine-tuning we pass guidance=1.0 (no CFG); guidance=3.5 is
            # reserved for inference (see generate_sample).
            guidance = (
                torch.full([1], fill_value=1.0, dtype=torch.bfloat16) if config.guidance_embeds else None
            )
            if guidance is not None:
                guidance_proj = _sinusoidal_proj(guidance * 1000.0)
                tt_guidance = _to_ttml(guidance_proj.float().numpy().reshape(1, 1, 1, -1))
            else:
                tt_guidance = None

            loss = rectified_flow_loss(
                model, tt_noisy, tt_velocity, tt_timestep, tt_prompt, tt_pooled,
                tt_guidance, rope_cos_s, rope_sin_s, rope_cos_p, rope_sin_p,
            )

            if distributed:
                composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
                loss_np = loss.to_numpy(composer=composer)
            else:
                loss_np = loss.to_numpy()
            accum_loss += float(loss_np.mean())

            if accum_steps > 1:
                loss = loss * (1.0 / float(accum_steps))

            loss.backward(False)
            ctx.reset_graph()

        step_loss = accum_loss / accum_steps
        train_losses.append(step_loss)

        if use_clip:
            ttml.core.clip_grad_norm(
                model.parameters(), args.clip_grad_norm, 2.0, False,
            )

        optimizer.step()

        step_time = time.time() - step_start

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", step_loss, step)
            tb_writer.add_scalar("train/lr", lr_now, step)
            tb_writer.add_scalar("train/step_time_sec", step_time, step)

        bar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{lr_now:.2e}"}, refresh=False)

        # Periodic checkpoint
        if args.save_dir and args.save_every > 0 and step % args.save_every == 0:
            _save_lora_checkpoint(model, optimizer, step, args)

        # Periodic image generation
        if args.save_dir and args.gen_every > 0 and step % args.gen_every == 0:
            print(f"\n--- Generating sample at step {step} ---")
            try:
                img = generate_sample(
                    model, config, args.checkpoint, args.gen_prompt, args.resolution,
                    args.num_inference_steps, args.seed + step, device, distributed, tp_size,
                    pos_embed_fn=pos_embed_fn,
                    prompt_embeds=dataset.eval_prompt_embeds,
                    pooled_prompt_embeds=dataset.eval_pooled_embeds,
                )
                img_dir = os.path.join(args.save_dir, "samples")
                os.makedirs(img_dir, exist_ok=True)
                img_path = os.path.join(img_dir, f"step_{step:04d}.png")
                img.save(img_path)
                print(f"  Saved sample to {img_path}")
                if tb_writer is not None:
                    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    tb_writer.add_image("samples", img_tensor, step)
            except Exception as e:
                print(f"  Sample generation failed: {e}")

            ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
            model.train()

        log_line = f"step={step}, loss={step_loss:.6f}, lr={lr_now:.2e}, time={step_time:.2f}s"
        log_lines.append(log_line)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    train_time = time.time() - train_start
    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"{'=' * 70}")
    print(f"  Total steps: {total_steps}")
    print(f"  Total time: {train_time:.1f}s")
    if train_losses:
        print(f"  Final train loss: {train_losses[-1]:.6f}")

    # Final checkpoint
    if args.save_dir:
        _save_lora_checkpoint(model, optimizer, total_steps, args)

    # Final sample generation
    if args.save_dir:
        print(f"\n--- Final sample generation ---")
        try:
            img = generate_sample(
                model, config, args.checkpoint, args.gen_prompt, args.resolution,
                args.num_inference_steps, args.seed, device, distributed, tp_size,
                pos_embed_fn=pos_embed_fn,
                prompt_embeds=dataset.eval_prompt_embeds,
                pooled_prompt_embeds=dataset.eval_pooled_embeds,
            )
            img_path = os.path.join(args.save_dir, "samples", "final.png")
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            img.save(img_path)
            print(f"  Saved final sample to {img_path}")
            if tb_writer is not None:
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                tb_writer.add_image("samples", img_tensor, total_steps + 1)
        except Exception as e:
            print(f"  Final sample generation failed: {e}")

    # Save training log
    if args.output_log:
        log_path = args.output_log
        if args.save_dir:
            log_path = os.path.join(args.save_dir, args.output_log)
        with open(log_path, "w") as f:
            f.write(f"# Flux1 LoRA Training Log\n")
            f.write(f"# Checkpoint: {args.checkpoint}\n")
            f.write(f"# Resolution: {args.resolution}\n")
            f.write(
                f"# LoRA: rank={args.lora_rank}, alpha={lora_alpha}, "
                f"targets={args.lora_targets}\n"
            )
            f.write(f"# LR: {args.lr}, Schedule: {args.lr_schedule}, Warmup: {args.warmup_steps}\n")
            f.write(f"# Total time: {train_time:.1f}s\n\n")
            for line in log_lines:
                f.write(line + "\n")
        print(f"\nTraining log saved to: {log_path}")

    # Cleanup
    if tb_writer is not None:
        tb_writer.close()
    ctx.close_device()


def _save_lora_checkpoint(model, optimizer, step, args):
    """Save LoRA adapter weights to safetensors format."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("WARNING: safetensors not installed, skipping checkpoint save")
        return

    ckpt_dir = os.path.join(args.save_dir, "checkpoints", f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    all_params = model.parameters()
    lora_state = {}
    for name, param in all_params.items():
        if "lora" in name:
            lora_state[name] = torch.from_numpy(param.to_numpy().astype(np.float32))

    save_file(lora_state, os.path.join(ckpt_dir, "lora_weights.safetensors"))
    print(f"  LoRA checkpoint saved to {ckpt_dir} ({len(lora_state)} tensors)")


if __name__ == "__main__":
    main()
