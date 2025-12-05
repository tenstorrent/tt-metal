"""
Suffix Embedding module for PI-Zero PyTorch model.

This module handles embedding of state, noisy actions, and timestep to create
the suffix part of the sequence (output side) for expert transformer processing.

Use Case:
    - Prepares the "suffix" part of the sequence (state + noisy actions + timestep)
      for expert processing
    - This is what gets denoised to produce final actions
    - Handles both PI0 and PI05 variants (different state and time handling)
"""

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

try:
    from .common import create_sinusoidal_pos_embedding
except ImportError:
    from common import create_sinusoidal_pos_embedding


class SuffixEmbedding:
    """
    Embeds state + actions + timestep (output side).
    
    This class handles the embedding of the suffix sequence, which consists
    of robot state (for PI0), noisy actions, and timestep. These are combined
    and processed to create embeddings that the expert transformer can process.
    
    Use Case:
        Prepares the "suffix" part of the sequence (state + noisy actions + timestep)
        for expert processing. This is what gets denoised to produce final actions.
        The suffix tokens have causal attention (each action token can attend to
        previous action tokens and state).
    """
    
    def __init__(
        self,
        config,
        action_in_proj,
        action_out_proj,
        state_proj=None,
        action_time_mlp_in=None,
        action_time_mlp_out=None,
        time_mlp_in=None,
        time_mlp_out=None,
        pi05=False,
    ):
        """
        Initialize suffix embedding module.
        
        Args:
            config: Model configuration (Pi0Config)
            action_in_proj: Linear layer to project actions to hidden dimension
            action_out_proj: Linear layer to project expert output back to action dimension
            state_proj: Linear layer to project state (None for PI05)
            action_time_mlp_in: MLP input layer for fusing action+time (None for PI05)
            action_time_mlp_out: MLP output layer for fusing action+time (None for PI05)
            time_mlp_in: MLP input layer for time embedding (None for PI0)
            time_mlp_out: MLP output layer for time embedding (None for PI0)
            pi05: Whether this is PI05 variant (affects state and time handling)
        """
        self.config = config
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj
        self.state_proj = state_proj
        self.action_time_mlp_in = action_time_mlp_in
        self.action_time_mlp_out = action_time_mlp_out
        self.time_mlp_in = time_mlp_in
        self.time_mlp_out = time_mlp_out
        self.pi05 = pi05
    
    def embed_suffix(self, state, noisy_actions, timestep, apply_checkpoint=None):
        """
        Main embedding function for suffix (state + actions + timestep).
        
        Args:
            state: Tensor of shape (batch_size, state_dim) with robot state
                  (ignored for PI05)
            noisy_actions: Tensor of shape (batch_size, action_horizon, action_dim)
                          with noisy actions to denoise
            timestep: Tensor of shape (batch_size,) with timestep values [0, 1]
            apply_checkpoint: Optional function to apply gradient checkpointing
        
        Returns:
            Tuple of (suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond):
            - suffix_embs: Concatenated embeddings of shape
                          (batch_size, total_suffix_len, hidden_dim)
            - suffix_pad_masks: Padding masks of shape
                              (batch_size, total_suffix_len)
            - suffix_att_masks: Attention masks of shape
                              (batch_size, total_suffix_len)
                              where 0 means can attend, 1 means cannot attend
            - adarms_cond: Adaptive RMS normalization condition (for PI05)
                          or None (for PI0)
        
        Use Case:
            Main method for creating suffix embeddings. Called during forward
            pass and denoising to prepare the output side of the sequence.
            Returns embeddings and masks that will be used in the expert transformer.
        """
        if apply_checkpoint is None:
            apply_checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        embs = []
        pad_masks = []
        att_masks = []
        
        # Embed state (PI0 only)
        if not self.pi05:
            state_emb = self._embed_state(state, apply_checkpoint)
            if state_emb is not None:
                embs.append(state_emb)
                bsize = state_emb.shape[0]
                device = state_emb.device
                state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
                pad_masks.append(state_mask)
                # Set attention masks so that image and language inputs do not attend to state or actions
                att_masks += [1]
        else:
            bsize = noisy_actions.shape[0]
            device = noisy_actions.device
        
        # Embed timestep and actions
        time_emb = self._embed_timestep(timestep, apply_checkpoint)
        action_emb = self._embed_actions(noisy_actions, apply_checkpoint)
        
        # Fuse action and time embeddings
        action_time_emb, adarms_cond = self._fuse_action_time(
            action_emb, time_emb, apply_checkpoint
        )
        
        # Add action-time embeddings
        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)
        
        # Set attention masks so that image, language and state inputs do not attend to action tokens
        # First action token has mask=1 (cannot be attended to by prefix), rest have mask=0 (causal)
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))
        
        # Concatenate all embeddings
        suffix_embs = torch.cat(embs, dim=1)
        suffix_pad_masks = torch.cat(pad_masks, dim=1)
        suffix_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        suffix_att_masks = suffix_att_masks[None, :].expand(bsize, len(att_masks))
        
        return suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond
    
    def _embed_state(self, state, apply_checkpoint):
        """
        Embed robot state (PI0 only).
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
            apply_checkpoint: Function to apply gradient checkpointing
        
        Returns:
            Tensor of shape (batch_size, 1, hidden_dim) with state embedding
            or None if PI05
        
        Use Case:
            Projects robot state into the hidden dimension. Only used for PI0,
            where state is part of the suffix. For PI05, state is handled
            differently (as discrete tokens in the prefix).
        """
        if self.pi05 or self.state_proj is None:
            return None
        
        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)
        
        def state_proj_func(state):
            return self.state_proj(state)
        
        state_emb = apply_checkpoint(state_proj_func, state)
        return state_emb[:, None, :]
    
    def _embed_timestep(self, timestep, apply_checkpoint):
        """
        Create timestep embedding using sinusoidal positional encoding.
        
        Args:
            timestep: Tensor of shape (batch_size,) with timestep values [0, 1]
            apply_checkpoint: Function to apply gradient checkpointing
        
        Returns:
            Tensor of shape (batch_size, hidden_dim) with timestep embedding
        
        Use Case:
            Encodes continuous timestep values into high-dimensional embeddings.
            Used to condition the denoising process on the current timestep.
            The embedding dimension matches the action projection output dimension.
        """
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        return time_emb.type(dtype=timestep.dtype)
    
    def _embed_actions(self, noisy_actions, apply_checkpoint):
        """
        Embed noisy actions.
        
        Args:
            noisy_actions: Tensor of shape (batch_size, action_horizon, action_dim)
            apply_checkpoint: Function to apply gradient checkpointing
        
        Returns:
            Tensor of shape (batch_size, action_horizon, hidden_dim) with
            action embeddings
        
        Use Case:
            Projects noisy actions into the hidden dimension. These embeddings
            will be fused with timestep embeddings and processed by the expert.
        """
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)
        
        return apply_checkpoint(action_proj_func, noisy_actions)
    
    def _fuse_action_time(self, action_emb, time_emb, apply_checkpoint):
        """
        Fuse action and time embeddings.
        
        For PI0: Concatenates action and time embeddings, then applies MLP.
        For PI05: Processes time through MLP separately for adaRMS conditioning.
        
        Args:
            action_emb: Tensor of shape (batch_size, action_horizon, hidden_dim)
            time_emb: Tensor of shape (batch_size, hidden_dim)
            apply_checkpoint: Function to apply gradient checkpointing
        
        Returns:
            Tuple of (action_time_emb, adarms_cond):
            - action_time_emb: Fused embeddings of shape
                             (batch_size, action_horizon, hidden_dim)
            - adarms_cond: Adaptive RMS condition (for PI05) or None (for PI0)
        
        Use Case:
            Combines action and timestep information. For PI0, this is done via
            concatenation and MLP. For PI05, time is processed separately for
            adaRMS normalization conditioning.
        """
        if not self.pi05:
            # PI0: Concatenate action and time, then apply MLP
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)
            
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)
            
            action_time_emb = apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # PI05: Process time separately for adaRMS
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)
            
            time_emb = apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb
        
        return action_time_emb, adarms_cond
    
    def _create_suffix_masks(self, bsize, action_horizon, has_state, device):
        """
        Create attention masks for suffix.
        
        Args:
            bsize: Batch size
            action_horizon: Number of action tokens
            has_state: Whether state is included (PI0)
            device: Device to create masks on
        
        Returns:
            Tuple of (pad_masks, att_masks) with appropriate shapes
        
        Use Case:
            Creates padding and attention masks for the suffix sequence.
            Ensures proper causal attention for action tokens.
        """
        pad_masks = []
        att_masks = []
        
        if has_state:
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]  # State cannot be attended to by prefix
        
        pad_masks.append(torch.ones(bsize, action_horizon, dtype=torch.bool, device=device))
        att_masks += [1] + ([0] * (action_horizon - 1))  # First action token masked, rest causal
        
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        
        return pad_masks, att_masks

