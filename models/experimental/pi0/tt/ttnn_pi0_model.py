# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Main PI0 model - TTNN Implementation (Inference Only)

This module assembles all PI0 components into a complete model:
    - PrefixEmbedding: Images + language → embeddings
    - SuffixEmbedding: State + actions + timestep → embeddings
    - PaliGemmaBackbone: VLM + Action Expert transformers

Architecture:
    1. Process images through SigLIP vision tower
    2. Embed language tokens through Gemma embeddings
    3. Concatenate to form prefix embeddings
    4. Prefill prefix, cache KV, denoise actions iteratively

Optimizations:
    1. Pre-computed timesteps
    2. Denoising loop stays entirely on device
    3. Single transfer at the end for final actions
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import ttnn

from models.experimental.pi0.common.configs import (
    PI0ModelConfig,
    PrefixConfig,
    SuffixConfig,
    PaliGemmaConfig,
    DenoiseConfig,
)
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from .ttnn_prefix import PrefixEmbeddingTTNN
from .ttnn_suffix import SuffixEmbeddingTTNN, convert_suffix_weights_to_ttnn
from .ttnn_paligemma import PaliGemmaBackboneTTNN


class PI0ModelTTNN:
    """
    Complete PI0 model implementation using TTNN.

    Maximizes execution on Tenstorrent hardware while keeping
    control flow and preprocessing on host.
    """

    def __init__(
        self,
        config: PI0ModelConfig,
        weight_loader: PI0WeightLoader,
        device: ttnn.Device,
    ):
        """
        Initialize PI0 model with TTNN.

        Args:
            config: Model configuration
            weight_loader: Loaded weights
            device: TTNN device
        """
        self.config = config
        self.weight_loader = weight_loader
        self.device = device

        # Initialize denoising config
        self.denoise_config = DenoiseConfig(
            num_steps=config.num_denoising_steps,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
        )

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all model components."""
        # Suffix embedding with TTNN weights
        suffix_config = SuffixConfig(
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            expert_width=self.config.expert_config.width,
            pi05=self.config.pi05,
        )
        pi0_weights = self.weight_loader.get_pi0_projections()

        # Convert weights to TTNN
        ttnn_weights = convert_suffix_weights_to_ttnn(pi0_weights, self.device)
        self.suffix_embedding = SuffixEmbeddingTTNN(suffix_config, ttnn_weights, self.device)

        # Backbone
        paligemma_config = PaliGemmaConfig(
            vlm_config=self.config.vlm_config,
            expert_config=self.config.expert_config,
            siglip_config=self.config.siglip_config,
            max_seq_len=self.config.max_seq_len,
        )
        weights = self.weight_loader.categorized_weights
        self.backbone = PaliGemmaBackboneTTNN(paligemma_config, weights, self.device)

        # Prefix embedding with backbone functions
        prefix_config = PrefixConfig(
            vlm_hidden_size=self.config.vlm_config.width,
            num_image_tokens=self.config.siglip_config.num_patches,
        )
        self.prefix_embedding = PrefixEmbeddingTTNN(
            prefix_config,
            self.device,
            embed_image_fn=self.backbone.embed_image,
            embed_language_fn=self.backbone.embed_language_tokens,
        )

    def embed_prefix(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Embed prefix (images + language) using TTNN.

        Args:
            images: List of input images (PyTorch)
            img_masks: Image validity masks (PyTorch)
            lang_tokens: Language token IDs (TTNN)
            lang_masks: Language masks (TTNN)

        Returns:
            Tuple of (embeddings, padding_mask, attention_mask) as TTNN tensors
        """
        return self.prefix_embedding.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    def embed_suffix(
        self,
        state: ttnn.Tensor,
        noisy_actions: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, Optional[ttnn.Tensor]]:
        """
        Embed suffix (state + noisy actions + timestep) using TTNN.

        Args:
            state: Robot state (TTNN)
            noisy_actions: Noisy actions (TTNN)
            timestep: Diffusion timestep (TTNN)

        Returns:
            Tuple of (embeddings, padding_mask, attention_mask, adarms_cond)
        """
        return self.suffix_embedding.embed_suffix(state, noisy_actions, timestep)

    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample actions via denoising (TTNN inference).

        This runs the full denoising loop:
        1. Compute prefix embeddings (images + language) once
        2. Forward prefix through VLM and cache KV
        3. For each denoising step: compute suffix, forward through expert with cached KV

        Args:
            images: Input images (PyTorch)
            img_masks: Image masks (PyTorch)
            lang_tokens: Language tokens (PyTorch)
            lang_masks: Language masks (PyTorch)
            state: Robot state (PyTorch)

        Returns:
            Sampled actions (PyTorch)
        """
        batch_size = lang_tokens.shape[0]

        # Convert inputs to TTNN
        lang_tokens_ttnn = ttnn.from_torch(
            lang_tokens,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        lang_masks_ttnn = ttnn.from_torch(
            lang_masks.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        state_ttnn = ttnn.from_torch(
            state,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Step 1: Embed prefix (images + language) using TTNN
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens_ttnn, lang_masks_ttnn)

        # Step 2: Forward prefix through VLM and cache KV
        _, prefix_kv_cache = self.backbone.forward_vlm(prefix_embs, use_cache=True)

        # Get timesteps using pure Python list (for control flow on host)
        num_steps = self.denoise_config.num_steps
        # Create timesteps as Python list: [1.0, 0.9, 0.8, ..., 0.0]
        timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

        # OPTIMIZATION: Pre-compute all timestep tensors on device using TTNN
        pad_steps = ((num_steps + 31) // 32) * 32

        # Create timestep indices on device using ttnn.arange
        timestep_indices = ttnn.arange(0, pad_steps, 1, device=self.device, dtype=ttnn.bfloat16)
        timestep_indices = ttnn.to_layout(timestep_indices, ttnn.TILE_LAYOUT)

        # Convert to timestep values: 1.0 - index / num_steps
        timestep_values = ttnn.multiply(timestep_indices, -1.0 / num_steps)
        timesteps_ttnn = ttnn.add(timestep_values, 1.0)
        timesteps_ttnn = ttnn.reshape(timesteps_ttnn, (1, pad_steps))

        # Cleanup
        ttnn.deallocate(timestep_indices)
        ttnn.deallocate(timestep_values)

        # Step 3: Sample initial noise (small tensor - host generation is fine)
        # Note: Using torch.randn ensures PCC compatibility with PyTorch reference
        # The tensor is small (batch * 50 * 32 = 1600 floats), so transfer is negligible
        x_t_torch = torch.randn(batch_size, self.config.action_horizon, self.config.action_dim)
        x_t_ttnn = ttnn.from_torch(
            x_t_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Step 4: Denoising loop (stays on device!)
        for i in range(num_steps):
            t = timesteps[i]  # Already Python float
            t_next = timesteps[i + 1]
            dt = t_next - t  # Negative since we go from 1.0 to 0.0

            # OPTIMIZATION: Slice timestep from pre-computed tensor (no transfer per step!)
            t_tensor = ttnn.slice(timesteps_ttnn, [0, i], [batch_size, i + 1])
            t_tensor = ttnn.reshape(t_tensor, (batch_size,))

            # Embed suffix (x_t_ttnn already on device - no transfer!)
            suffix_embs, suffix_pad, suffix_att, _ = self.embed_suffix(state_ttnn, x_t_ttnn, t_tensor)

            # Forward through expert with cached prefix KV
            expert_output, _ = self.backbone.forward_expert(
                suffix_embs,
                past_key_values=prefix_kv_cache,
            )

            # Extract action output (skip state token in PI0 mode)
            if not self.config.pi05:
                action_output = ttnn.slice(
                    expert_output, [0, 1, 0], [expert_output.shape[0], expert_output.shape[1], expert_output.shape[2]]
                )
            else:
                action_output = expert_output

            # Project to velocity
            velocity = self.suffix_embedding.project_output(action_output)

            # Euler step ON DEVICE (no transfer per step!)
            velocity_scaled = ttnn.mul(velocity, dt)
            x_t_ttnn = ttnn.add(x_t_ttnn, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Clear profiler buffer after each denoising step (~500 ops)
            ttnn.ReadDeviceProfiler(
                self.device
            )  # Clear device profiler buffer, this helps resolve a issue when building profiler perf sheets

        # Convert back to PyTorch only at the very end (1 transfer instead of 10!)
        return ttnn.to_torch(x_t_ttnn)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: ttnn.Device,
        config: Optional[PI0ModelConfig] = None,
    ) -> "PI0ModelTTNN":
        """
        Load pretrained PI0 model to TTNN device.

        Args:
            model_path: Path to model or HuggingFace model ID
            device: TTNN device
            config: Optional configuration override

        Returns:
            Loaded PI0 model
        """
        weight_loader = PI0WeightLoader(model_path)

        if config is None:
            config = PI0ModelConfig(
                action_dim=weight_loader.config.action_dim,
                action_horizon=weight_loader.config.action_horizon,
            )

        return cls(config, weight_loader, device)


# Default export
PI0Model = PI0ModelTTNN
