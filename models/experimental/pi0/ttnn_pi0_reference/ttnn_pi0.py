# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Main PI0 model orchestrator for TTNN implementation.

This module assembles all PI0 components into a complete model:
    - PrefixEmbedding: Images + language → embeddings
    - SuffixEmbedding: State + actions + timestep → embeddings
    - PaliGemmaBackbone: VLM + Action Expert transformers
    - DenoisingModule: Flow matching for action generation

Architecture:
    1. Process images through SigLIP vision tower
    2. Embed language tokens through Gemma embeddings
    3. Concatenate to form prefix embeddings
    4. (Training) Process prefix + suffix through shared attention
    5. (Inference) Prefill prefix, cache KV, denoise actions iteratively
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None

from .weight_loader import PI0WeightLoader
from .ttnn_attention import AttentionMaskUtils
from .ttnn_prefix import PrefixEmbeddingTorch, PrefixEmbeddingTTNN, PrefixConfig
from .ttnn_suffix import SuffixEmbeddingTorch, SuffixEmbeddingTTNN, SuffixConfig
from .ttnn_paligemma import PaliGemmaBackboneTorch, PaliGemmaBackboneTTNN, PaliGemmaConfig
from .ttnn_denoise import (
    DenoisingModuleTorch,
    DenoisingModuleTTNN,
    KVCacheManager,
    KVCacheManagerTTNN,
    DenoiseConfig,
)
from .ttnn_gemma import GemmaConfig
from .ttnn_siglip import SigLIPConfig


@dataclass
class PI0ModelConfig:
    """Complete configuration for PI0 model."""

    # Core dimensions
    action_dim: int = 32
    action_horizon: int = 50
    state_dim: int = 32

    # Model variants
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    # Processing
    precision: str = "bfloat16"
    num_denoising_steps: int = 10
    max_seq_len: int = 2048

    # PI05 mode (uses adaRMS instead of fused action-time)
    pi05: bool = False

    # Component configs (auto-populated)
    vlm_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_2b)
    expert_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_300m)
    siglip_config: SigLIPConfig = field(default_factory=SigLIPConfig)

    def __post_init__(self):
        self.vlm_config = GemmaConfig.gemma_2b()
        self.expert_config = GemmaConfig.gemma_300m()
        self.siglip_config = SigLIPConfig()


class PI0ModelTorch:
    """
    Complete PI0 model implementation (PyTorch).

    This class orchestrates all components for training and inference.
    """

    def __init__(
        self,
        config: PI0ModelConfig,
        weight_loader: PI0WeightLoader,
    ):
        """
        Initialize PI0 model.

        Args:
            config: Model configuration
            weight_loader: Loaded weights
        """
        self.config = config
        self.weight_loader = weight_loader

        # Initialize components
        self._init_suffix_embedding()
        self._init_prefix_embedding()
        self._init_backbone()
        self._init_denoising()

    def _init_suffix_embedding(self):
        """Initialize suffix embedding module."""
        suffix_config = SuffixConfig(
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            expert_width=self.config.expert_config.width,
            pi05=self.config.pi05,
        )
        pi0_weights = self.weight_loader.get_pi0_projections()
        self.suffix_embedding = SuffixEmbeddingTorch(suffix_config, pi0_weights)

    def _init_prefix_embedding(self):
        """Initialize prefix embedding module."""
        prefix_config = PrefixConfig(
            vlm_width=self.config.vlm_config.width,
            num_image_tokens=self.config.siglip_config.num_patches,
        )
        self.prefix_embedding = PrefixEmbeddingTorch(prefix_config)

    def _init_backbone(self):
        """Initialize PaliGemma backbone."""
        paligemma_config = PaliGemmaConfig(
            vlm_config=self.config.vlm_config,
            expert_config=self.config.expert_config,
            siglip_config=self.config.siglip_config,
            max_seq_len=self.config.max_seq_len,
        )
        weights = self.weight_loader.categorized_weights
        self.backbone = PaliGemmaBackboneTorch(paligemma_config, weights)

        # Set embedding functions for prefix
        self.prefix_embedding.embed_image_fn = self.backbone.embed_image
        self.prefix_embedding.embed_language_fn = self.backbone.embed_language_tokens

    def _init_denoising(self):
        """Initialize denoising module."""
        denoise_config = DenoiseConfig(
            num_steps=self.config.num_denoising_steps,
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
        )
        self.denoising = DenoisingModuleTorch(denoise_config, self._denoise_forward)

        # KV cache for inference
        self.kv_cache = KVCacheManager(
            num_layers=self.config.expert_config.depth,
            max_seq_len=self.config.max_seq_len,
            num_kv_heads=self.config.expert_config.num_kv_heads,
            head_dim=self.config.expert_config.head_dim,
        )

    def _denoise_forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for denoising (predicts velocity).

        Args:
            noisy_actions: Current noisy actions
            timestep: Current timestep
            kv_cache: Cached KV from prefix
            state: Robot state

        Returns:
            Predicted velocity for Euler step
        """
        # Embed suffix
        suffix_embs, _, _, _ = self.suffix_embedding.embed_suffix(
            state,
            noisy_actions,
            timestep,
        )

        # Forward through expert
        expert_output, _ = self.backbone.forward_expert(
            suffix_embs,
            past_key_values=kv_cache,
        )

        # Project back to action dimension
        # Extract action tokens (skip state token if present)
        if not self.config.pi05:
            action_output = expert_output[:, 1:, :]  # Skip state token
        else:
            action_output = expert_output

        velocity = self.suffix_embedding.project_output(action_output)

        return velocity

    def embed_prefix(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed images and language to form prefix.

        Args:
            images: List of image tensors
            img_masks: List of image validity masks
            lang_tokens: Language token IDs
            lang_masks: Language token masks

        Returns:
            Tuple of (prefix_embs, prefix_pad_masks, prefix_att_masks)
        """
        return self.prefix_embedding.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Embed state and actions to form suffix.

        Args:
            state: Robot state
            noisy_actions: Current noisy actions
            timestep: Current timestep

        Returns:
            Tuple of (suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond)
        """
        return self.suffix_embedding.embed_suffix(state, noisy_actions, timestep)

    def forward_training(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training (computes velocity for loss).

        Args:
            images: List of input images
            img_masks: Image validity masks
            lang_tokens: Language token IDs
            lang_masks: Language masks
            state: Robot state
            actions: Noisy actions
            timestep: Sampled timestep

        Returns:
            Predicted velocity (batch_size, action_horizon, action_dim)
        """
        # Embed prefix
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Embed suffix
        suffix_embs, suffix_pad, suffix_att, _ = self.embed_suffix(state, actions, timestep)

        # Combine masks
        pad_masks, att_masks = AttentionMaskUtils.combine_prefix_suffix_masks(
            prefix_pad, prefix_att, suffix_pad, suffix_att
        )

        # Create 2D attention mask
        att_2d = AttentionMaskUtils.make_att_2d_masks(pad_masks, att_masks)
        att_4d = AttentionMaskUtils.prepare_attention_masks_4d(att_2d)

        # Forward through backbone
        # For now, VLM and Expert process independently (not full shared attention)
        # VLM attends to prefix only, Expert attends to suffix only
        prefix_len = prefix_embs.shape[1]
        suffix_len = suffix_embs.shape[1]
        vlm_output, expert_output = self.backbone.forward_shared_attention(
            prefix_embs,
            suffix_embs,
            prefix_mask=att_4d[:, :, :prefix_len, :prefix_len],  # VLM: prefix→prefix
            suffix_mask=att_4d[
                :, :, prefix_len : prefix_len + suffix_len, prefix_len : prefix_len + suffix_len
            ],  # Expert: suffix→suffix
        )

        # Project expert output to velocity
        if not self.config.pi05:
            action_output = expert_output[:, 1:, :]  # Skip state token
        else:
            action_output = expert_output

        velocity = self.suffix_embedding.project_output(action_output)

        return velocity

    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample actions via denoising (inference).

        Args:
            images: Input images
            img_masks: Image validity masks
            lang_tokens: Language tokens
            lang_masks: Language masks
            state: Robot state

        Returns:
            Sampled actions (batch_size, action_horizon, action_dim)
        """
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        # Embed and cache prefix
        prefix_embs, _, _ = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Forward prefix through VLM and cache
        _, prefix_cache = self.backbone.forward_vlm(prefix_embs, use_cache=True)

        # Sample actions
        actions = self.denoising.sample_actions(
            batch_size=batch_size,
            prefix_kv_cache=prefix_cache,
            device=device,
            state=state,
        )

        return actions

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        config: Optional[PI0ModelConfig] = None,
    ) -> "PI0ModelTorch":
        """
        Load pretrained PI0 model.

        Args:
            model_path: Path to model or HuggingFace model ID
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

        return cls(config, weight_loader)


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
        device: "ttnn.Device",
    ):
        """
        Initialize PI0 model with TTNN.

        Args:
            config: Model configuration
            weight_loader: Loaded weights
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")

        self.config = config
        self.weight_loader = weight_loader
        self.device = device

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
        from .ttnn_suffix import convert_suffix_weights_to_ttnn

        ttnn_weights = convert_suffix_weights_to_ttnn(pi0_weights, self.device)
        self.suffix_embedding = SuffixEmbeddingTTNN(suffix_config, ttnn_weights, self.device)

        # Prefix embedding
        prefix_config = PrefixConfig(
            vlm_width=self.config.vlm_config.width,
            num_image_tokens=self.config.siglip_config.num_patches,
        )

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
        self.prefix_embedding = PrefixEmbeddingTTNN(
            prefix_config,
            self.device,
            embed_image_fn=self.backbone.embed_image,
            embed_language_fn=self.backbone.embed_language_tokens,
        )

        # Denoising with TTNN
        denoise_config = DenoiseConfig(
            num_steps=self.config.num_denoising_steps,
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
        )
        self.denoising = DenoisingModuleTTNN(denoise_config, self._denoise_forward, self.device)

        # KV cache manager
        self.kv_cache = KVCacheManagerTTNN(
            num_layers=self.config.expert_config.depth,
            max_seq_len=self.config.max_seq_len,
            num_kv_heads=self.config.expert_config.num_kv_heads,
            head_dim=self.config.expert_config.head_dim,
            device=self.device,
        )

    def _denoise_forward(
        self,
        noisy_actions: "ttnn.Tensor",
        timestep: "ttnn.Tensor",
        kv_cache: Optional[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = None,
        state: Optional["ttnn.Tensor"] = None,
        **kwargs,
    ) -> "ttnn.Tensor":
        """
        Forward pass for denoising (TTNN version).

        Args:
            noisy_actions: Current noisy actions
            timestep: Current timestep
            kv_cache: Cached KV from prefix
            state: Robot state

        Returns:
            Predicted velocity
        """
        # Embed suffix
        action_emb = self.suffix_embedding.embed_actions(noisy_actions)
        time_emb = self.suffix_embedding.embed_timestep(timestep)
        action_time_emb, _ = self.suffix_embedding.fuse_action_time(action_emb, time_emb)

        # Add state embedding if PI0 mode
        if not self.config.pi05 and state is not None:
            state_emb = self.suffix_embedding.embed_state(state)
            suffix_embs = ttnn.concat([state_emb, action_time_emb], dim=1)
        else:
            suffix_embs = action_time_emb

        # Forward through expert (simplified - full implementation would use backbone)
        # For now, convert to torch, process, convert back
        suffix_torch = ttnn.to_torch(suffix_embs)

        # Use torch backbone for now
        expert_output, _ = self.backbone.torch_weights  # This is a placeholder

        # Project back
        velocity = self.suffix_embedding.project_output(suffix_embs)

        return velocity

    def embed_prefix(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> Tuple["ttnn.Tensor", "ttnn.Tensor", "ttnn.Tensor"]:
        """
        Embed prefix (images + language) using TTNN.

        Args:
            images: List of input images (PyTorch)
            img_masks: Image validity masks (PyTorch)
            lang_tokens: Language token IDs (PyTorch)
            lang_masks: Language masks (PyTorch)

        Returns:
            Tuple of (embeddings, padding_mask, attention_mask) as TTNN tensors
        """
        return self.prefix_embedding.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple["ttnn.Tensor", "ttnn.Tensor", "ttnn.Tensor", Tuple["ttnn.Tensor", "ttnn.Tensor"]]:
        """
        Embed suffix (state + noisy actions + timestep) using TTNN.

        Args:
            state: Robot state (PyTorch)
            noisy_actions: Noisy actions (PyTorch)
            timestep: Diffusion timestep (PyTorch)

        Returns:
            Tuple of (embeddings, padding_mask, attention_mask, time_info)
        """
        return self.suffix_embedding.embed_suffix(state, noisy_actions, timestep)

    def forward_training(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training using TTNN (computes velocity for loss).

        Args:
            images: List of input images
            img_masks: Image validity masks
            lang_tokens: Language token IDs
            lang_masks: Language masks
            state: Robot state
            actions: Noisy actions
            timestep: Sampled timestep

        Returns:
            Predicted velocity (batch_size, action_horizon, action_dim) as PyTorch tensor
        """
        # Convert inputs to TTNN where needed
        state_ttnn = ttnn.from_torch(
            state.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        actions_ttnn = ttnn.from_torch(
            actions.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        timestep_ttnn = ttnn.from_torch(
            timestep.float().unsqueeze(-1) if timestep.dim() == 1 else timestep.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Convert language tokens to TTNN (uint32 for embedding lookup)
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

        # Embed prefix using TTNN
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens_ttnn, lang_masks_ttnn)

        # Embed suffix using TTNN (pass TTNN tensors)
        suffix_embs, suffix_pad, suffix_att, _ = self.embed_suffix(state_ttnn, actions_ttnn, timestep_ttnn)

        # Get sequence lengths
        prefix_len = prefix_embs.shape[1]
        suffix_len = suffix_embs.shape[1]

        # Forward through backbone using TTNN
        vlm_output, expert_output = self.backbone.forward_shared_attention(
            prefix_embs,
            suffix_embs,
            prefix_mask=None,  # For now, no causal masking in TTNN
            suffix_mask=None,
        )

        # Project expert output to velocity
        if not self.config.pi05:
            action_output = ttnn.slice(
                expert_output, [0, 1, 0], [expert_output.shape[0], expert_output.shape[1], expert_output.shape[2]]
            )
        else:
            action_output = expert_output

        velocity = self.suffix_embedding.project_output(action_output)

        # Convert back to PyTorch for loss computation
        return ttnn.to_torch(velocity)

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

        # Get timesteps (on host for control flow)
        timesteps = torch.linspace(1.0, 0.0, self.denoising.config.num_steps + 1)

        # Step 3: Sample initial noise
        x_t = torch.randn(batch_size, self.config.action_horizon, self.config.action_dim)

        # Step 4: Denoising loop
        for i in range(self.denoising.config.num_steps):
            t = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t

            # Convert noisy actions to TTNN
            x_t_ttnn = ttnn.from_torch(
                x_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

            # Create timestep tensor
            t_tensor = ttnn.from_torch(
                torch.full((batch_size,), t, dtype=torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

            # Embed suffix
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
            velocity_torch = ttnn.to_torch(velocity)

            # Euler step
            x_t = x_t + velocity_torch * dt
            # Clear profiler buffer after each denoising step (~500 ops)
            ttnn.ReadDeviceProfiler(self.device)  # Clear device profiler buffer

        return x_t

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: "ttnn.Device",
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


# Default export - Use TTNN when available for performance
if TTNN_AVAILABLE:
    PI0Model = PI0ModelTTNN
else:
    PI0Model = PI0ModelTorch
