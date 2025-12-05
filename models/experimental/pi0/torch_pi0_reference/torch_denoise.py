"""
Denoising module for PI-Zero PyTorch model.

This module implements the flow matching denoising process used during inference.
It iteratively denoises noise to produce final actions.

Use Case:
    - Implements the flow matching denoising process
    - During inference, starts with noise and iteratively denoises it to produce
      final actions over multiple steps
    - Uses Euler integration to solve the ODE
"""

import torch
from torch import Tensor

try:
    from .torch_attention import AttentionMaskUtils
    from .common import compute_position_ids, sample_noise
except ImportError:
    from torch_attention import AttentionMaskUtils
    from common import compute_position_ids, sample_noise


class DenoisingModule:
    """
    Denoising process for inference.
    
    This class implements the flow matching denoising process, which starts
    with random noise and iteratively denoises it to produce final actions.
    The process uses Euler integration to solve the ODE defined by the velocity field.
    
    Use Case:
        Implements the flow matching denoising process. During inference, starts
        with noise and iteratively denoises it to produce final actions over
        multiple steps. This is the core of the action generation process.
    """
    
    def __init__(self, model, config, suffix_embedding, paligemma_backbone, prefix_embedding, action_out_proj):
        """
        Initialize denoising module.
        
        Args:
            model: The main PI0Pytorch model (for accessing components)
            config: Model configuration (Pi0Config)
            suffix_embedding: SuffixEmbedding instance
            paligemma_backbone: PaliGemmaBackbone instance
            prefix_embedding: PrefixEmbedding instance
            action_out_proj: Action output projection layer
        """
        self.model = model
        self.config = config
        self.suffix_embedding = suffix_embedding
        self.paligemma_backbone = paligemma_backbone
        self.prefix_embedding = prefix_embedding
        self.action_out_proj = action_out_proj
        self.attention_utils = AttentionMaskUtils()
    
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        """
        Full denoising loop to generate actions.
        
        Args:
            device: Device to run on
            observation: Observation object with images, state, language tokens
            noise: Optional initial noise tensor. If None, will be sampled.
            num_steps: Number of denoising steps (default: 10)
        
        Returns:
            Tensor of shape (batch_size, action_horizon, action_dim) with
            final denoised actions
        
        Use Case:
            Main inference method. Generates actions by iteratively denoising
            noise. This is called during inference to produce robot actions
            from observations.
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = sample_noise(actions_shape, device)
        
        # Preprocess observation and create prefix embeddings
        images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(
            observation, train=False
        )
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedding.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        # Create prefix attention masks and position IDs
        prefix_att_2d_masks = self.attention_utils.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = compute_position_ids(prefix_pad_masks)
        
        # Compute image and language key value cache (for efficient inference)
        prefix_att_2d_masks_4d = self.attention_utils.prepare_attention_masks_4d(prefix_att_2d_masks)
        
        # Set attention implementation to eager for inference
        self.paligemma_backbone.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        
        # Forward pass through prefix to get cached key-values
        _, past_key_values = self.paligemma_backbone.forward(
            prefix_embs=prefix_embs,
            suffix_embs=None,
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
            adarms_cond=[None, None],
        )
        
        # Initialize denoising loop
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        # Denoising loop: iterate from t=1.0 to t=0.0
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            
            # Euler step: x_{t+dt} = x_t + dt * v_t
            x_t = x_t + dt * v_t
            time += dt
        
        return x_t
    
    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """
        Apply one denoising step.
        
        This computes the velocity field v_t at the current timestep and
        returns it for Euler integration.
        
        Args:
            state: Tensor of shape (batch_size, state_dim) with robot state
            prefix_pad_masks: Padding masks for prefix of shape
                            (batch_size, prefix_len)
            past_key_values: Cached key-value pairs from prefix processing
            x_t: Current noisy actions of shape
                (batch_size, action_horizon, action_dim)
            timestep: Current timestep tensor of shape (batch_size,)
        
        Returns:
            Tensor of shape (batch_size, action_horizon, action_dim) with
            velocity field v_t
        
        Use Case:
            Computes one step of the denoising process. Called repeatedly
            during the denoising loop to iteratively refine the actions.
        """
        # Embed suffix (state + noisy actions + timestep)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.suffix_embedding.embed_suffix(
            state, x_t, timestep, apply_checkpoint=self.model._apply_checkpoint
        )
        
        # Create combined attention masks for prefix + suffix
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        
        # Prefix can attend to itself, suffix can attend to prefix and itself
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = self.attention_utils.make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        
        # Compute position IDs (prefix positions + suffix positions)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + compute_position_ids(suffix_pad_masks)
        
        # Prepare attention masks for transformer
        full_att_2d_masks_4d = self.attention_utils.prepare_attention_masks_4d(full_att_2d_masks)
        
        # Set attention implementation to eager for inference
        self.paligemma_backbone.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        
        # Forward pass through expert (with cached prefix key-values)
        outputs_embeds, _ = self.paligemma_backbone.forward(
            prefix_embs=None,
            suffix_embs=suffix_embs,
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        
        # Extract expert output (suffix part)
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        
        # Project to action space
        return self.action_out_proj(suffix_out)
    
    def _euler_step(self, x_t, v_t, dt):
        """
        Euler integration step.
        
        Args:
            x_t: Current state tensor
            v_t: Velocity field tensor
            dt: Time step size
        
        Returns:
            Updated state tensor
        
        Use Case:
            Performs one step of Euler integration: x_{t+dt} = x_t + dt * v_t.
            This is used in the denoising loop to update the actions.
        """
        return x_t + dt * v_t
    
    def _compute_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """
        Compute velocity field at current timestep.
        
        This is an alias for denoise_step that emphasizes the velocity
        field interpretation.
        
        Args:
            state: Robot state
            prefix_pad_masks: Prefix padding masks
            past_key_values: Cached key-value pairs
            x_t: Current noisy actions
            timestep: Current timestep
        
        Returns:
            Velocity field v_t
        
        Use Case:
            Computes the velocity field that drives the denoising process.
            This is the core of flow matching: the model learns to predict
            the velocity field that transforms noise into actions.
        """
        return self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)


def create_denoising_schedule(num_steps):
    """
    Create timestep schedule for denoising.
    
    Args:
        num_steps: Number of denoising steps
    
    Returns:
        List of timestep values from 1.0 to 0.0
    
    Use Case:
        Creates a schedule of timesteps for the denoising process.
        The schedule goes from t=1.0 (pure noise) to t=0.0 (denoised actions).
    """
    dt = -1.0 / num_steps
    times = []
    time = 1.0
    while time >= -dt / 2:
        times.append(time)
        time += dt
    return times

