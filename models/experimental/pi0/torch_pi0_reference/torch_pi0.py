"""
Main PI-Zero PyTorch model orchestrator.

This module assembles all components into the complete PI-Zero model.
It provides the main interface for training and inference.

Use Case:
    - Main entry point for using the PI-Zero model
    - Users instantiate this class and call forward() for training or
      sample_actions() for inference
    - Orchestrates all sub-modules (prefix, suffix, attention, denoising, etc.)
"""

import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import sys
import pathlib

# Add the openpi source to path if needed
openpi_src = pathlib.Path(__file__).parent.parent.parent.parent / "src"
if str(openpi_src) not in sys.path:
    sys.path.insert(0, str(openpi_src))

import openpi.models.gemma as _gemma
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

try:
    from .torch_paligemma import PaliGemmaBackbone
    from .torch_prefix import PrefixEmbedding
    from .torch_suffix import SuffixEmbedding
    from .torch_denoise import DenoisingModule
    from .torch_attention import AttentionMaskUtils
    from .common import (
        sample_noise,
        sample_time,
        compute_position_ids,
        safe_cat,
    )
except ImportError:
    # Handle direct script execution
    from torch_paligemma import PaliGemmaBackbone
    from torch_prefix import PrefixEmbedding
    from torch_suffix import SuffixEmbedding
    from torch_denoise import DenoisingModule
    from torch_attention import AttentionMaskUtils
    from common import (
        sample_noise,
        sample_time,
        compute_position_ids,
        safe_cat,
    )


class PI0Pytorch(nn.Module):
    """
    Main PI-Zero PyTorch model.
    
    This class assembles all components (vision, language, expert, prefix,
    suffix, denoising) into a complete model that can be used for training
    and inference.
    
    Use Case:
        Main entry point for using the PI-Zero model. Users instantiate this
        class and call forward() for training or sample_actions() for inference.
        This orchestrates all sub-modules to perform vision-language-action
        learning and prediction.
    """
    
    def __init__(self, config):
        """
        Initialize PI-Zero model.
        
        Args:
            config: Pi0Config object with model configuration
        """
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        
        # Get model configurations
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        
        # Initialize PaliGemma backbone
        self.paligemma_backbone = PaliGemmaBackbone(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )
        
        # Initialize projection layers
        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)
        
        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.state_proj = None
            self.action_time_mlp_in = None
            self.action_time_mlp_out = None
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_in = None
            self.time_mlp_out = None
        
        # Initialize sub-modules
        self.prefix_embedding = PrefixEmbedding(self.paligemma_backbone)
        self.suffix_embedding = SuffixEmbedding(
            config=self.config,
            action_in_proj=self.action_in_proj,
            action_out_proj=self.action_out_proj,
            state_proj=self.state_proj,
            action_time_mlp_in=self.action_time_mlp_in,
            action_time_mlp_out=self.action_time_mlp_out,
            time_mlp_in=self.time_mlp_in,
            time_mlp_out=self.time_mlp_out,
            pi05=self.pi05,
        )
        self.denoising_module = DenoisingModule(
            model=self,
            config=self.config,
            suffix_embedding=self.suffix_embedding,
            paligemma_backbone=self.paligemma_backbone,
            prefix_embedding=self.prefix_embedding,
            action_out_proj=self.action_out_proj,
        )
        self.attention_utils = AttentionMaskUtils()
        
        # Set float32 matmul precision for performance
        torch.set_float32_matmul_precision("high")
        
        # Compile sample_actions for performance
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
        
        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        
        # Verify transformers_replace is installed correctly
        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check
            
            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_backbone.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_backbone.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_backbone.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        
        logging.info("Enabled gradient checkpointing for PI0Pytorch model")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_backbone.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_backbone.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_backbone.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        
        logging.info("Disabled gradient checkpointing for PI0Pytorch model")
    
    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled
    
    def _apply_checkpoint(self, func, *args, **kwargs):
        """
        Helper method to apply gradient checkpointing if enabled.
        
        Args:
            func: Function to checkpoint
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
        
        Returns:
            Result of function call, with checkpointing applied if enabled
        """
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)
    
    def _preprocess_observation(self, observation, *, train=True):
        """
        Helper method to preprocess observation.
        
        Args:
            observation: Observation object with images, state, language tokens
            train: Whether in training mode (affects data augmentation)
        
        Returns:
            Tuple of (images, img_masks, lang_tokens, lang_masks, state)
        """
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )
    
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """
        Training forward pass.
        
        Args:
            observation: Observation object with images, state, language tokens
            actions: Target actions of shape (batch_size, action_horizon, action_dim)
            noise: Optional noise tensor. If None, will be sampled.
            time: Optional timestep tensor. If None, will be sampled.
        
        Returns:
            Tensor of shape (batch_size, action_horizon, action_dim) with
            per-token MSE loss values
        
        Use Case:
            Main training method. Computes the loss between predicted and
            target actions. This is called during training to compute gradients.
        """
        # Preprocess observation
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=True
        )
        
        # Sample noise and time if not provided
        if noise is None:
            noise = sample_noise(actions.shape, actions.device)
        
        if time is None:
            time = sample_time(actions.shape[0], actions.device)
        
        # Create noisy actions: x_t = t * noise + (1 - t) * actions
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # Target velocity field
        
        # Embed prefix (images + language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedding.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        # Embed suffix (state + noisy actions + timestep)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.suffix_embedding.embed_suffix(
            state, x_t, time, apply_checkpoint=self._apply_checkpoint
        )
        
        # Convert to bfloat16 if needed
        if (
            self.paligemma_backbone.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        # Combine prefix and suffix masks
        pad_masks = safe_cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = safe_cat([prefix_att_masks, suffix_att_masks], dim=1)
        
        # Create 2D attention masks
        att_2d_masks = self.attention_utils.make_att_2d_masks(pad_masks, att_masks)
        position_ids = compute_position_ids(pad_masks)
        
        # Prepare 4D attention masks for transformer
        att_2d_masks_4d = self.attention_utils.prepare_attention_masks_4d(att_2d_masks)
        
        # Forward pass through PaliGemma and expert
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_backbone.forward(
                prefix_embs=prefix_embs,
                suffix_embs=suffix_embs,
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out
        
        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        
        # Extract action predictions
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        
        # Project to action space
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)
        
        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        
        # Compute MSE loss
        return F.mse_loss(u_t, v_t, reduction="none")
    
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """
        Inference: generate actions from observation.
        
        Args:
            device: Device to run on
            observation: Observation object with images, state, language tokens
            noise: Optional initial noise tensor. If None, will be sampled.
            num_steps: Number of denoising steps (default: 10)
        
        Returns:
            Tensor of shape (batch_size, action_horizon, action_dim) with
            generated actions
        
        Use Case:
            Main inference method. Generates actions by iteratively denoising
            noise. This is called during inference to produce robot actions
            from observations.
        """
        return self.denoising_module.sample_actions(device, observation, noise, num_steps)
    
    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """
        Apply one denoising step (used by denoising module).
        
        Args:
            state: Robot state
            prefix_pad_masks: Prefix padding masks
            past_key_values: Cached key-value pairs
            x_t: Current noisy actions
            timestep: Current timestep
        
        Returns:
            Velocity field v_t
        
        Use Case:
            Helper method for denoising. Called by DenoisingModule during
            the denoising loop.
        """
        return self.denoising_module.denoise_step(
            state, prefix_pad_masks, past_key_values, x_t, timestep
        )

