# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 Video Generation Pipeline for tt_dit.

Implements the text-to-video inference pipeline:
1. Text encoding (Gemma, torch-only)
2. Sigma schedule computation (LTX2Scheduler)
3. Denoising loop (Euler first-order steps with CFG)
4. VAE decoding (future)

Reference: LTX-2/packages/ltx-pipelines/ + Wan pipeline_wan.py
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn

from ...encoders.gemma.encoder_pair import GemmaTokenizerEncoderPair
from ...models.transformers.ltx.ltx_transformer import LTXTransformerModel
from ...models.transformers.ltx.rope_ltx import precompute_freqs_cis
from ...parallel.config import DiTParallelConfig
from ...parallel.manager import CCLManager
from ...utils.mochi import get_rot_transformation_mat
from ...utils.tensor import bf16_tensor, bf16_tensor_2dshard

if TYPE_CHECKING:
    pass


# =============================================================================
# Scheduler
# =============================================================================

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


def compute_sigmas(
    steps: int,
    num_tokens: int = MAX_SHIFT_ANCHOR,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> torch.FloatTensor:
    """
    Compute the LTX-2 sigma schedule.

    Generates a sequence of noise levels (sigmas) from high noise (~1.0)
    to low noise (~terminal) with token-count-dependent shifting.

    Args:
        steps: Number of denoising steps
        num_tokens: Number of spatial tokens (affects sigma shift)
        max_shift: Maximum shift factor
        base_shift: Base shift factor
        stretch: Whether to stretch schedule to terminal value
        terminal: Final sigma value

    Returns:
        Tensor of shape (steps + 1,) with sigma values
    """
    sigmas = torch.linspace(1.0, 0.0, steps + 1)

    # Adaptive shift based on token count
    mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
    b = base_shift - mm * BASE_SHIFT_ANCHOR
    sigma_shift = num_tokens * mm + b

    # Exponential shift
    sigmas = torch.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1)),
        0,
    )

    # Stretch to terminal
    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero_mask] = stretched

    return sigmas.to(torch.float32)


# =============================================================================
# Diffusion step
# =============================================================================


def euler_step(
    sample: torch.Tensor,
    denoised: torch.Tensor,
    sigma: float,
    sigma_next: float,
) -> torch.Tensor:
    """
    First-order Euler diffusion step.

    x_{t+1} = x_t + velocity * dt
    where velocity = (x_t - denoised) / sigma, dt = sigma_next - sigma

    Args:
        sample: Current noisy latent
        denoised: Model's denoised prediction
        sigma: Current noise level
        sigma_next: Next noise level

    Returns:
        Updated latent at next noise level
    """
    dt = sigma_next - sigma
    velocity = (sample.float() - denoised.float()) / sigma
    return (sample.float() + velocity * dt).to(sample.dtype)


# =============================================================================
# Pipeline
# =============================================================================


class LTXPipeline:
    """
    LTX-2 text-to-video generation pipeline.

    Usage:
        pipeline = LTXPipeline(mesh_device, parallel_config, ccl_manager)
        pipeline.load_transformer(checkpoint_or_state_dict)
        pipeline.load_text_encoder(gemma_checkpoint)

        output = pipeline(
            prompt="A cat playing piano",
            num_frames=33,
            height=480,
            width=832,
            num_inference_steps=30,
        )
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig,
        ccl_manager: CCLManager,
        *,
        # Transformer config
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        # RoPE config
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: float = 1000.0,
    ):
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.cross_attention_dim = cross_attention_dim
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos or [20, 2048, 2048]
        self.timestep_scale_multiplier = timestep_scale_multiplier

        self.transformer: LTXTransformerModel | None = None
        self.text_encoder: GemmaTokenizerEncoderPair | None = None
        self.vae_decoder = None  # LTXVideoDecoder or LTXVideoDecoderTorch

        # Cached device tensors (computed once, reused across steps)
        self._cached_rope_cos: ttnn.Tensor | None = None
        self._cached_rope_sin: ttnn.Tensor | None = None
        self._cached_trans_mat: ttnn.Tensor | None = None
        self._cached_prompt: ttnn.Tensor | None = None
        self._cached_negative_prompt: ttnn.Tensor | None = None

    def load_transformer(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load transformer weights from a state dict."""
        self.transformer = LTXTransformerModel(
            num_attention_heads=self.num_attention_heads,
            attention_head_dim=self.attention_head_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            cross_attention_dim=self.cross_attention_dim,
            mesh_device=self.mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
        )
        self.transformer.load_torch_state_dict(state_dict)
        logger.info(f"Loaded LTX transformer with {self.num_layers} layers")

    def load_text_encoder(
        self,
        checkpoint: str = "google/gemma-3-12b-it",
        *,
        sequence_length: int = 256,
        hidden_layer_index: int = -1,
    ) -> None:
        """Load Gemma text encoder (torch-only)."""
        self.text_encoder = GemmaTokenizerEncoderPair(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            embedding_dim=self.cross_attention_dim,
            hidden_layer_index=hidden_layer_index,
        )
        logger.info(f"Loaded Gemma text encoder from {checkpoint}")

    def load_vae_decoder(
        self,
        state_dict: dict[str, torch.Tensor],
        decoder_blocks: list[tuple[str, dict]],
        *,
        use_ttnn: bool = True,
        patch_size: int = 4,
        base_channels: int = 128,
    ) -> None:
        """Load VAE decoder weights.

        Args:
            state_dict: PyTorch state dict for the decoder
            decoder_blocks: Block configuration list
            use_ttnn: If True, use TTNN decoder; if False, use torch-only wrapper
        """
        if use_ttnn:
            from ...models.vae.ltx.vae_ltx import LTXVideoDecoder

            self.vae_decoder = LTXVideoDecoder(
                decoder_blocks=decoder_blocks,
                in_channels=self.in_channels,
                out_channels=3,
                patch_size=patch_size,
                base_channels=base_channels,
                mesh_device=self.mesh_device,
            )
            self.vae_decoder.load_torch_state_dict(state_dict)
            logger.info("Loaded TTNN VAE decoder")
        else:
            from ...models.vae.ltx.vae_ltx import LTXVideoDecoderTorch

            self.vae_decoder = LTXVideoDecoderTorch.from_config(
                decoder_blocks, in_channels=self.in_channels, patch_size=patch_size, base_channels=base_channels
            )
            self.vae_decoder.load_state_dict(state_dict)
            logger.info("Loaded torch-only VAE decoder")

    def decode_latents(self, latent: torch.Tensor, latent_frames: int, latent_h: int, latent_w: int) -> torch.Tensor:
        """Decode latent tensor to video pixels.

        Args:
            latent: (B, num_tokens, C) flat latent from denoising loop
            latent_frames, latent_h, latent_w: Spatial dimensions

        Returns:
            (B, 3, F, H, W) decoded video
        """
        if self.vae_decoder is None:
            logger.warning("No VAE decoder loaded, returning raw latent")
            return latent

        B = latent.shape[0]
        # Reshape flat tokens to spatial: (B, num_tokens, C) -> (B, C, F', H', W')
        latent_spatial = latent.reshape(B, latent_frames, latent_h, latent_w, self.in_channels)
        latent_spatial = latent_spatial.permute(0, 4, 1, 2, 3)  # BCTHW

        from ...models.vae.ltx.vae_ltx import LTXVideoDecoder

        if isinstance(self.vae_decoder, LTXVideoDecoder):
            return self.vae_decoder(latent_spatial)
        else:
            return self.vae_decoder.decode(latent_spatial)

    def _prepare_rope(
        self, num_frames: int, latent_height: int, latent_width: int
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Compute and cache RoPE features for the given spatial dimensions."""
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        # Build position grid: center of each latent cell in pixel space
        t_ids = torch.arange(num_frames)
        h_ids = torch.arange(latent_height)
        w_ids = torch.arange(latent_width)
        grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
        fps = 24.0
        t_s = (grid_t.float() * 8 + 1 - 8).clamp(min=0) / fps
        t_e = ((grid_t.float() + 1) * 8 + 1 - 8).clamp(min=0) / fps
        t_mid = (t_s + t_e) / 2
        h_mid = (grid_h.float() + 0.5) * 32
        w_mid = (grid_w.float() + 0.5) * 32
        indices_grid = torch.stack([t_mid.flatten(), h_mid.flatten(), w_mid.flatten()], dim=-1).float().unsqueeze(0)

        cos_freq, sin_freq = precompute_freqs_cis(
            indices_grid,
            dim=self.inner_dim,
            out_dtype=torch.float32,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
        )

        seq_len = num_frames * latent_height * latent_width
        head_dim = self.attention_head_dim
        cos_heads = cos_freq.reshape(1, seq_len, self.num_attention_heads, head_dim).permute(0, 2, 1, 3)
        sin_heads = sin_freq.reshape(1, seq_len, self.num_attention_heads, head_dim).permute(0, 2, 1, 3)

        tt_cos = bf16_tensor_2dshard(cos_heads, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
        tt_sin = bf16_tensor_2dshard(sin_heads, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
        tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=self.mesh_device)

        self._cached_rope_cos = tt_cos
        self._cached_rope_sin = tt_sin
        self._cached_trans_mat = tt_trans_mat

        return tt_cos, tt_sin, tt_trans_mat

    def _prepare_prompt(self, prompt_embeds: torch.Tensor) -> ttnn.Tensor:
        """Push prompt embeddings to device, padding to cross_attention_dim if needed."""
        # (B, L, D) -> (1, B, L, D)
        prompt = prompt_embeds.unsqueeze(0)
        # Pad if encoder dim != cross_attention_dim (e.g., Gemma 3840 -> 4096)
        if prompt.shape[-1] < self.cross_attention_dim:
            pad_size = self.cross_attention_dim - prompt.shape[-1]
            prompt = torch.nn.functional.pad(prompt, (0, pad_size))
        elif prompt.shape[-1] > self.cross_attention_dim:
            prompt = prompt[..., : self.cross_attention_dim]
        tt_prompt = bf16_tensor(prompt, device=self.mesh_device)
        return tt_prompt

    def __call__(
        self,
        prompt: str | list[str],
        *,
        negative_prompt: str | list[str] | None = None,
        num_frames: int = 33,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        seed: int | None = None,
        # Scheduler params
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        # Latent space params (LTX uses 8x temporal, 32x spatial compression)
        temporal_compression: int = 8,
        spatial_compression: int = 32,
    ) -> torch.Tensor:
        """
        Run the full text-to-video generation pipeline.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s) for CFG
            num_frames: Number of output video frames
            height: Output video height in pixels
            width: Output video width in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance strength (1.0 = no guidance)
            seed: Random seed for reproducibility

        Returns:
            Denoised latent tensor of shape (B, num_tokens, out_channels)
        """
        assert self.transformer is not None, "Call load_transformer() first"

        if isinstance(prompt, str):
            prompt = [prompt]
        B = len(prompt)

        # Compute latent dimensions
        latent_frames = (num_frames - 1) // temporal_compression + 1
        latent_height = height // spatial_compression
        latent_width = width // spatial_compression
        num_tokens = latent_frames * latent_height * latent_width

        logger.info(
            f"Generating: {num_frames} frames @ {height}x{width}, "
            f"latent: {latent_frames}x{latent_height}x{latent_width} = {num_tokens} tokens"
        )

        # 1. Encode text
        do_cfg = guidance_scale > 1.0
        if self.text_encoder is not None:
            prompt_embeds = self.text_encoder.encode(prompt)
        else:
            # Fallback: zero embeddings
            prompt_embeds = torch.zeros(B, 256, self.cross_attention_dim)

        if do_cfg:
            if self.text_encoder is not None and negative_prompt is not None:
                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt] * B
                negative_embeds = self.text_encoder.encode(negative_prompt)
            else:
                negative_embeds = torch.zeros_like(prompt_embeds)

        # Push prompts to device
        tt_prompt = self._prepare_prompt(prompt_embeds)
        tt_negative_prompt = self._prepare_prompt(negative_embeds) if do_cfg else None

        # 2. Prepare RoPE
        rope_cos, rope_sin, trans_mat = self._prepare_rope(latent_frames, latent_height, latent_width)

        # 3. Compute sigma schedule
        sigmas = compute_sigmas(
            steps=num_inference_steps,
            num_tokens=num_tokens,
            max_shift=max_shift,
            base_shift=base_shift,
        )
        logger.info(f"Sigmas: {sigmas[0]:.4f} -> {sigmas[-1]:.4f} ({len(sigmas)} values)")

        # 4. Prepare initial noise
        if seed is not None:
            torch.manual_seed(seed)
        latent = torch.randn(B, num_tokens, self.in_channels, dtype=torch.float32)

        # Scale initial noise by first sigma
        latent = latent * sigmas[0]

        # 5. Denoising loop
        for step_idx in range(num_inference_steps):
            sigma = sigmas[step_idx].item()
            sigma_next = sigmas[step_idx + 1].item()

            # Prepare spatial input: (1, B, N, in_channels)
            spatial_torch = latent.unsqueeze(0)
            timestep_torch = torch.tensor([sigma])

            # Forward pass (conditioned)
            tt_denoised = self.transformer.inner_step(
                video_1BNI_torch=spatial_torch,
                video_prompt_1BLP=tt_prompt,
                video_rope_cos=rope_cos,
                video_rope_sin=rope_sin,
                trans_mat=trans_mat,
                video_N=num_tokens,
                timestep_torch=timestep_torch,
            )
            # Model output is velocity; convert to x0: denoised = sample - velocity * sigma
            velocity = LTXTransformerModel.device_to_host(tt_denoised).squeeze(0)
            denoised = latent.float() - velocity.float() * sigma

            # CFG
            if do_cfg:
                tt_uncond = self.transformer.inner_step(
                    video_1BNI_torch=spatial_torch,
                    video_prompt_1BLP=tt_negative_prompt,
                    video_rope_cos=rope_cos,
                    video_rope_sin=rope_sin,
                    trans_mat=trans_mat,
                    video_N=num_tokens,
                    timestep_torch=timestep_torch,
                )
                uncond_velocity = LTXTransformerModel.device_to_host(tt_uncond).squeeze(0)
                uncond = latent.float() - uncond_velocity.float() * sigma

                # guidance = uncond + scale * (cond - uncond)
                denoised = uncond + guidance_scale * (denoised - uncond)

            # Euler step
            latent = euler_step(latent, denoised, sigma, sigma_next)

            if (step_idx + 1) % 5 == 0 or step_idx == 0:
                logger.info(
                    f"Step {step_idx + 1}/{num_inference_steps}: "
                    f"sigma {sigma:.4f} -> {sigma_next:.4f}, "
                    f"latent range [{latent.min():.3f}, {latent.max():.3f}]"
                )

        logger.info(f"Denoising complete. Output latent shape: {latent.shape}")

        # Optionally decode latents to video
        if self.vae_decoder is not None:
            video = self.decode_latents(latent, latent_frames, latent_height, latent_width)
            logger.info(f"Decoded video shape: {video.shape}")
            return video

        return latent
