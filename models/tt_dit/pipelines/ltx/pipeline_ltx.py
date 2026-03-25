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
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
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
        # Mode: "video" or "av"
        mode: str = "video",
        # RoPE config
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: float = 1000.0,
    ):
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.mode = mode

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
        # trans_mat not needed: using SPLIT RoPE (not interleaved)
        self._cached_prompt: ttnn.Tensor | None = None
        self._cached_negative_prompt: ttnn.Tensor | None = None

    @staticmethod
    def create_pipeline(
        mesh_device: ttnn.MeshDevice,
        *,
        checkpoint_path: str | None = None,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
        num_links: int | None = None,
        dynamic_load: bool | None = None,
        topology: ttnn.Topology | None = None,
        is_fsdp: bool | None = None,
        mode: str = "av",
    ) -> "LTXPipeline":
        """Factory method matching Wan's create_pipeline pattern.

        Auto-configures parallel settings from mesh shape and loads checkpoint.
        """
        mesh_shape = tuple(mesh_device.shape)
        # Default configs per mesh shape
        device_configs = {
            (2, 4): {
                "sp_axis": 0,
                "tp_axis": 1,
                "num_links": 2,
                "dynamic_load": False,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
            },
            (1, 1): {
                "sp_axis": 0,
                "tp_axis": 1,
                "num_links": 1,
                "dynamic_load": False,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
            },
        }
        defaults = device_configs.get(mesh_shape, device_configs[(2, 4)])
        sp_axis = sp_axis if sp_axis is not None else defaults["sp_axis"]
        tp_axis = tp_axis if tp_axis is not None else defaults["tp_axis"]
        num_links = num_links if num_links is not None else defaults["num_links"]
        dynamic_load = dynamic_load if dynamic_load is not None else defaults["dynamic_load"]
        topology = topology if topology is not None else defaults["topology"]
        is_fsdp = is_fsdp if is_fsdp is not None else defaults["is_fsdp"]

        parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
            sequence_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
            tensor_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
        )
        ccl_manager = CCLManager(mesh_device, topology=topology)

        pipeline = LTXPipeline(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            mode=mode,
        )

        if checkpoint_path:
            pipeline.load_from_checkpoint(checkpoint_path)

        return pipeline

    def load_transformer(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load transformer weights from a state dict."""
        has_gate = any("to_gate_logits" in k for k in state_dict)
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
            has_audio=self.mode == "av",
            apply_gated_attention=has_gate,
        )
        self.transformer.load_torch_state_dict(state_dict)
        logger.info(f"Loaded LTX transformer ({self.mode} mode) with {self.num_layers} layers")

    def load_text_encoder(
        self,
        checkpoint: str = "google/gemma-3-12b-it",
        *,
        sequence_length: int = 256,
        hidden_layer_index: int = -1,
    ) -> None:
        """Load Gemma text encoder (torch-only). Fallback when reference encode_prompts is not available."""
        self.text_encoder = GemmaTokenizerEncoderPair(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            embedding_dim=self.cross_attention_dim,
            hidden_layer_index=hidden_layer_index,
        )
        logger.info(f"Loaded Gemma text encoder from {checkpoint}")

    def encode_prompts_reference(
        self,
        prompts: list[str],
        checkpoint_path: str,
        gemma_path: str,
    ) -> list:
        """Encode prompts using the official LTX-2 reference pipeline (recommended for AV mode).

        Returns list of encoding results with .video_encoding and .audio_encoding attributes.
        """
        try:
            import sys

            sys.path.insert(0, "LTX-2/packages/ltx-core/src")
            sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
            torch.cuda.synchronize = lambda *a, **kw: None  # No CUDA on TT host
            from ltx_pipelines.utils.helpers import encode_prompts
            from ltx_pipelines.utils.model_ledger import ModelLedger
        except ImportError as e:
            raise ImportError(
                "encode_prompts_reference() requires the LTX-2 reference package. "
                "Use load_text_encoder() + __call__() for standalone text encoding."
            ) from e

        # Check embedding cache to skip expensive Gemma encoding
        import hashlib
        import os

        cache_dir = os.environ.get("TT_DIT_CACHE_DIR", os.path.expanduser("~/.cache"))
        embed_cache_dir = os.path.join(cache_dir, "ltx-embeddings")
        os.makedirs(embed_cache_dir, exist_ok=True)

        cache_key = hashlib.md5("||".join(prompts).encode()).hexdigest()
        cache_path = os.path.join(embed_cache_dir, f"{cache_key}.pt")

        if os.path.exists(cache_path):
            logger.info(f"Loading cached embeddings from {cache_path}")
            return torch.load(cache_path, weights_only=False)

        ledger = ModelLedger(
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_path,
        )
        results = encode_prompts(prompts, ledger)
        del ledger

        # Cache for future use
        torch.save(results, cache_path)
        logger.info(f"Cached embeddings to {cache_path}")
        return results

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
            from ...models.vae.vae_ltx import LTXVideoDecoder

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
            from ...models.vae.vae_ltx import LTXVideoDecoderTorch

            self.vae_decoder = LTXVideoDecoderTorch.from_config(
                decoder_blocks, in_channels=self.in_channels, patch_size=patch_size, base_channels=base_channels
            )
            self.vae_decoder.load_state_dict(state_dict)
            logger.info("Loaded torch-only VAE decoder")

    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load transformer and VAE decoder from a single safetensors checkpoint file.

        This is the recommended way to load the 22B model. It handles:
        - Extracting diffusion model keys for the transformer
        - Extracting VAE decoder keys and config from checkpoint metadata
        - Setting up the 22B decoder_blocks config
        """
        import json

        from safetensors.torch import load_file

        raw = load_file(checkpoint_path)

        # Transformer
        prefix = "model.diffusion_model."
        transformer_sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
        self.load_transformer(transformer_sd)
        del transformer_sd

        # VAE decoder — extract config from metadata

        with open(checkpoint_path, "rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_size))
        metadata = header.get("__metadata__", {})
        config = json.loads(metadata.get("config", "{}"))
        vae_config = config.get("vae", {})
        decoder_blocks = vae_config.get("decoder_blocks", [])
        causal = vae_config.get("causal_decoder", False)
        base_channels = vae_config.get("decoder_base_channels", 128)

        vae_state = {}
        for k, v in raw.items():
            if k.startswith("vae.decoder."):
                vae_state[k[len("vae.decoder.") :]] = v
            elif k.startswith("vae.per_channel_statistics."):
                vae_state[k[len("vae.") :]] = v
        del raw

        # Store VAE config for lazy loading (avoids exceeding DRAM when both
        # transformer and VAE are loaded simultaneously)
        self._vae_checkpoint_path = checkpoint_path
        self._vae_decoder_blocks = decoder_blocks
        self._vae_causal = causal
        self._vae_base_channels = base_channels
        if decoder_blocks:
            logger.info(f"VAE config saved for lazy loading ({len(decoder_blocks)} blocks, causal={causal})")

    def load_vae_from_checkpoint(self) -> None:
        """Load VAE decoder from saved config. Call after transformer weights are freed if needed."""
        if not self._vae_decoder_blocks:
            return
        from safetensors.torch import load_file

        from ...models.vae.vae_ltx import LTXVideoDecoder

        raw = load_file(self._vae_checkpoint_path)
        vae_state = {}
        for k, v in raw.items():
            if k.startswith("vae.decoder."):
                vae_state[k[len("vae.decoder.") :]] = v
            elif k.startswith("vae.per_channel_statistics."):
                vae_state[k[len("vae.") :]] = v
        del raw
        self.vae_decoder = LTXVideoDecoder(
            decoder_blocks=self._vae_decoder_blocks,
            causal=self._vae_causal,
            base_channels=self._vae_base_channels,
            mesh_device=self.mesh_device,
        )
        self.vae_decoder.load_torch_state_dict(vae_state)
        logger.info(f"Loaded TTNN VAE decoder ({len(self._vae_decoder_blocks)} blocks)")

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

        from ...models.vae.vae_ltx import LTXVideoDecoder

        if isinstance(self.vae_decoder, LTXVideoDecoder):
            return self.vae_decoder(latent_spatial)
        else:
            return self.vae_decoder.decode(latent_spatial)

    def _prepare_rope(
        self, num_frames: int, latent_height: int, latent_width: int, fps: float = 24.0
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Compute video RoPE using reference SPLIT rotation with pixel-space coordinates.

        Uses the official LTX-2 VideoLatentPatchifier for positions, matching the reference
        pipeline's precompute_freqs_cis with SPLIT rotation. No trans_mat needed.
        """
        from ...models.transformers.ltx.rope_ltx import precompute_freqs_cis
        from ...utils.ltx import VideoLatentShape, get_pixel_coords, video_get_patch_grid_bounds

        v_shape = VideoLatentShape(batch=1, channels=128, frames=num_frames, height=latent_height, width=latent_width)
        v_coords = video_get_patch_grid_bounds(v_shape)
        v_positions = get_pixel_coords(v_coords, scale_factors=(8, 32, 32), causal_fix=True).float()
        v_positions[:, 0, ...] = v_positions[:, 0, ...] / fps

        from ...models.transformers.ltx.rope_ltx import LTXRopeType

        cos_freq, sin_freq = precompute_freqs_cis(
            v_positions.bfloat16(),
            dim=self.inner_dim,
            out_dtype=torch.float32,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            use_middle_indices_grid=True,
            num_attention_heads=self.num_attention_heads,
            rope_type=LTXRopeType.SPLIT,
        )  # (1, num_heads, N, D_half)

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        tt_cos = bf16_tensor_2dshard(cos_freq, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
        tt_sin = bf16_tensor_2dshard(sin_freq, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})

        self._cached_rope_cos = tt_cos
        self._cached_rope_sin = tt_sin

        return tt_cos, tt_sin

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
        rope_cos, rope_sin = self._prepare_rope(latent_frames, latent_height, latent_width)

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
                trans_mat=None,
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
                    trans_mat=None,
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

    def _prepare_audio_rope(self, audio_N: int, audio_N_real: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Compute audio RoPE using AudioPatchifier time-in-seconds positions with SPLIT rotation."""

        from ...models.transformers.ltx.rope_ltx import LTXRopeType, precompute_freqs_cis
        from ...utils.ltx import AudioLatentShape, audio_get_patch_grid_bounds

        a_shape = AudioLatentShape(batch=1, channels=8, frames=audio_N_real, mel_bins=16)
        a_positions = audio_get_patch_grid_bounds(a_shape).float()  # (1, 1, N, 2)

        a_cos, a_sin = precompute_freqs_cis(
            a_positions.bfloat16(),
            dim=2048,
            out_dtype=torch.float32,
            theta=self.positional_embedding_theta,
            max_pos=[20],  # 1D temporal only
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.SPLIT,
        )  # (1, 32, audio_N_real, D_half)

        # Pad to audio_N if needed
        if audio_N > audio_N_real:
            d_half = a_cos.shape[-1]
            a_cos_padded = torch.ones(1, 32, audio_N, d_half)
            a_cos_padded[:, :, :audio_N_real, :] = a_cos
            a_sin_padded = torch.zeros(1, 32, audio_N, d_half)
            a_sin_padded[:, :, :audio_N_real, :] = a_sin
            a_cos, a_sin = a_cos_padded, a_sin_padded

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        return (
            bf16_tensor_2dshard(a_cos, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1}),
            bf16_tensor_2dshard(a_sin, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1}),
        )

    def _prepare_audio_masks(self, audio_N: int, audio_N_real: int) -> tuple:
        """Create audio attention mask (for SDPA) and padding mask (for A-to-V)."""
        sp_factor = self.parallel_config.sequence_parallel.factor
        if audio_N <= audio_N_real:
            return None, None

        audio_N_local = audio_N // sp_factor
        mask = torch.zeros(1, 1, audio_N_local, audio_N)
        mask[:, :, :, audio_N_real:] = float("-inf")
        tt_attn_mask = bf16_tensor(mask, device=self.mesh_device)

        pad_mask = torch.ones(1, 1, audio_N, 1)
        pad_mask[:, :, audio_N_real:, :] = 0.0
        tt_pad_mask = bf16_tensor(pad_mask, device=self.mesh_device)
        return tt_attn_mask, tt_pad_mask

    @torch.no_grad()
    def call_av(
        self,
        video_prompt_embeds: torch.Tensor,
        audio_prompt_embeds: torch.Tensor,
        neg_video_prompt_embeds: torch.Tensor | None = None,
        neg_audio_prompt_embeds: torch.Tensor | None = None,
        num_frames: int = 33,
        height: int = 512,
        width: int = 768,
        num_inference_steps: int = 30,
        video_cfg_scale: float = 3.0,
        audio_cfg_scale: float = 7.0,
        video_stg_scale: float = 1.0,
        audio_stg_scale: float = 1.0,
        video_modality_scale: float = 3.0,
        audio_modality_scale: float = 3.0,
        rescale_scale: float = 0.7,
        stg_block: int = 28,
        seed: int | None = None,
        profiler=None,
        profiler_iteration: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run AV denoising with full MultiModalGuider guidance. Returns (video_latent, audio_latent)."""
        from ...utils.ltx import AudioLatentShape, VideoPixelShape

        B = 1
        latent_frames = (num_frames - 1) // 8 + 1
        latent_h, latent_w = height // 32, width // 32
        video_N = latent_frames * latent_h * latent_w

        vps = VideoPixelShape(batch=B, frames=num_frames, height=height, width=width, fps=24)
        als = AudioLatentShape.from_video_pixel_shape(vps)
        audio_N_real = als.frames
        sp_factor = self.parallel_config.sequence_parallel.factor
        audio_N = ((audio_N_real + 32 * sp_factor - 1) // (32 * sp_factor)) * (32 * sp_factor)

        logger.info(f"AV: {num_frames}f@{height}x{width}, vN={video_N}, aN={audio_N}(real={audio_N_real})")

        v_cos, v_sin = self._prepare_rope(latent_frames, latent_h, latent_w)
        a_cos, a_sin = self._prepare_audio_rope(audio_N, audio_N_real)
        tt_attn_mask, tt_pad_mask = self._prepare_audio_masks(audio_N, audio_N_real)

        tt_vp = self._prepare_prompt(video_prompt_embeds)
        tt_ap = bf16_tensor(audio_prompt_embeds.unsqueeze(0), device=self.mesh_device)
        tt_nv = self._prepare_prompt(neg_video_prompt_embeds) if neg_video_prompt_embeds is not None else None
        tt_na = (
            bf16_tensor(neg_audio_prompt_embeds.unsqueeze(0), device=self.mesh_device)
            if neg_audio_prompt_embeds is not None
            else None
        )

        sigmas = compute_sigmas(steps=num_inference_steps)
        if seed is not None:
            torch.manual_seed(seed)
        video_lat = torch.randn(B, video_N, self.in_channels, dtype=torch.bfloat16).float() * sigmas[0]
        audio_lat_real = torch.randn(B, audio_N_real, self.in_channels, dtype=torch.bfloat16).float() * sigmas[0]
        audio_lat = torch.zeros(B, audio_N, self.in_channels)
        audio_lat[:, :audio_N_real, :] = audio_lat_real

        do_cfg = video_cfg_scale > 1.0 or audio_cfg_scale > 1.0
        do_stg = video_stg_scale != 0.0 or audio_stg_scale != 0.0
        do_mod = video_modality_scale != 1.0 or audio_modality_scale != 1.0

        for step_idx in range(num_inference_steps):
            sigma = sigmas[step_idx].item()
            sigma_next = sigmas[step_idx + 1].item()

            def _run(vp, ap, skip_ca=False, skip_sa_blocks=None):
                v, a = self.transformer.inner_step(
                    video_1BNI_torch=video_lat.unsqueeze(0),
                    video_prompt_1BLP=vp,
                    video_rope_cos=v_cos,
                    video_rope_sin=v_sin,
                    video_N=video_N,
                    audio_1BNI_torch=audio_lat.unsqueeze(0),
                    audio_prompt_1BLP=ap,
                    audio_rope_cos=a_cos,
                    audio_rope_sin=a_sin,
                    audio_N=audio_N,
                    trans_mat=None,
                    timestep_torch=torch.tensor([sigma]),
                    skip_cross_attn=skip_ca,
                    skip_self_attn_blocks=skip_sa_blocks,
                    audio_attn_mask=tt_attn_mask,
                    audio_padding_mask=tt_pad_mask,
                )
                vv = LTXTransformerModel.device_to_host(v).squeeze(0)
                av = LTXTransformerModel.device_to_host(a).squeeze(0)
                vd = (video_lat.bfloat16().float() - vv.float() * sigma).bfloat16()
                ad = (audio_lat.bfloat16().float() - av.float() * sigma).bfloat16()
                return vd, ad

            v_den, a_den = _run(tt_vp, tt_ap)

            v_unc = a_unc = v_ptb = a_ptb = v_iso = a_iso = 0.0
            if do_cfg:
                v_unc, a_unc = _run(tt_nv, tt_na)
            if do_stg:
                v_ptb, a_ptb = _run(tt_vp, tt_ap, skip_sa_blocks=[stg_block])
            if do_mod:
                v_iso, a_iso = _run(tt_vp, tt_ap, skip_ca=True)

            if do_cfg or do_stg or do_mod:
                for label, den, unc, ptb, iso, cfg_s, stg_s, mod_s in [
                    ("v", v_den, v_unc, v_ptb, v_iso, video_cfg_scale, video_stg_scale, video_modality_scale),
                    ("a", a_den, a_unc, a_ptb, a_iso, audio_cfg_scale, audio_stg_scale, audio_modality_scale),
                ]:
                    c = den.float()
                    pred = c
                    if do_cfg and isinstance(unc, torch.Tensor):
                        pred = pred + (cfg_s - 1) * (c - unc.float())
                    if do_stg and isinstance(ptb, torch.Tensor):
                        pred = pred + stg_s * (c - ptb.float())
                    if do_mod and isinstance(iso, torch.Tensor):
                        pred = pred + (mod_s - 1) * (c - iso.float())
                    if rescale_scale != 0:
                        pred = pred * (rescale_scale * (c.std() / pred.std()) + (1 - rescale_scale))
                    if label == "v":
                        v_den = pred.bfloat16()
                    else:
                        a_den = pred.bfloat16()

            video_lat = euler_step(video_lat, v_den.float(), sigma, sigma_next).bfloat16().float()
            a_new = euler_step(audio_lat, a_den.float(), sigma, sigma_next).bfloat16().float()
            audio_lat = torch.zeros_like(audio_lat)
            audio_lat[:, :audio_N_real, :] = a_new[:, :audio_N_real, :]

            if (step_idx + 1) % 5 == 0 or step_idx == 0:
                logger.info(f"Step {step_idx+1}/{num_inference_steps}: σ {sigma:.4f}→{sigma_next:.4f}")

        logger.info(f"AV done. video: {video_lat.shape}, audio: ({B},{audio_N_real},{self.in_channels})")
        return video_lat, audio_lat[:, :audio_N_real, :]

    def decode_audio_reference(
        self, audio_latent: torch.Tensor, checkpoint_path: str, num_frames: int, fps: float = 24.0
    ):
        """Decode audio latent using reference audio VAE + vocoder (CPU torch).

        Matches the reference ti2vid pipeline: unpatchify → audio_decoder → vocoder → trim.

        Args:
            audio_latent: (1, audio_N, 128) raw audio latent from call_av()
            checkpoint_path: path to safetensors checkpoint
            num_frames: video frame count (for duration trimming)
            fps: video frame rate

        Returns:
            Audio object with .waveform and .sampling_rate, or None on failure
        """
        try:
            import sys

            sys.path.insert(0, "LTX-2/packages/ltx-core/src")
            sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
            torch.cuda.synchronize = lambda *a, **kw: None
            from ltx_core.model.audio_vae.audio_vae import decode_audio as vae_decode_audio
            from ltx_core.types import Audio
            from ltx_pipelines.utils.model_ledger import ModelLedger

            ledger = ModelLedger(dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=checkpoint_path)
            audio_decoder = ledger.audio_decoder()
            vocoder = ledger.vocoder()

            # Unpatchify: (1, N, 128) → (1, 8, N, 16)
            audio_N = audio_latent.shape[1]
            audio_spatial = audio_latent.reshape(1, audio_N, 8, 16).permute(0, 2, 1, 3).bfloat16()

            with torch.no_grad():
                audio_obj = vae_decode_audio(audio_spatial, audio_decoder, vocoder)

            # Trim to video duration
            video_duration = num_frames / fps
            target_samples = int(video_duration * audio_obj.sampling_rate)
            if audio_obj.waveform.shape[-1] > target_samples:
                audio_obj = Audio(
                    waveform=audio_obj.waveform[..., :target_samples], sampling_rate=audio_obj.sampling_rate
                )

            logger.info(
                f"Audio decoded: {audio_obj.waveform.shape} "
                f"({audio_obj.waveform.shape[-1]/audio_obj.sampling_rate:.2f}s @ {audio_obj.sampling_rate}Hz)"
            )
            return audio_obj
        except Exception as e:
            logger.warning(f"Audio decode failed: {e}")
            return None

    def export_video(self, video_pixels: torch.Tensor, output_path: str, fps: int = 24, audio=None) -> None:
        """Export decoded video (and optionally audio) to MP4.

        Matches reference encode_video: correct [-1,1] → uint8 conversion.

        Args:
            video_pixels: (B, C, F, H, W) from decode_latents(), range [-1, 1]
            output_path: output .mp4 path
            fps: frame rate
            audio: Audio object from decode_audio_reference(), or None
        """
        try:
            import sys

            sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
            from ltx_pipelines.utils.media_io import encode_video

            # Convert to reference format: (F, H, W, C) uint8
            frames = (((video_pixels[0] + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0)  # (F, H, W, C)

            encode_video(video=frames, fps=fps, audio=audio, output_path=output_path, video_chunks_number=1)
            logger.info(f"Saved: {output_path} ({frames.shape[0]}f @ {fps}fps)")
        except ImportError:
            import imageio

            frames = (((video_pixels[0] + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0).numpy()
            writer = imageio.get_writer(output_path, fps=fps)
            for f in frames:
                writer.append_data(f)
            writer.close()
            logger.info(f"Saved (no audio): {output_path} ({frames.shape[0]}f @ {fps}fps)")
