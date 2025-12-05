"""
Action Expert interface for PI-Zero PyTorch model.

This module provides an interface to the Gemma expert model that processes
action embeddings (from suffix) to predict denoised actions.

Use Case:
    - Processes action embeddings through the expert transformer
    - Specializes in action prediction (the "expert" component)
    - Provides a clean interface to the action expert model
"""

import torch
from torch import Tensor


class ActionExpert:
    """
    Interface to the Gemma expert model for action prediction.
    
    This class wraps the GemmaForCausalLM expert model, providing a clean
    interface for processing action embeddings.
    
    Use Case:
        Processes action embeddings (from suffix) to predict denoised actions.
        This is the "expert" that specializes in action prediction, separate
        from the vision-language backbone.
    """
    
    def __init__(self, expert_model):
        """
        Initialize with GemmaForCausalLM expert model.
        
        Args:
            expert_model: The Gemma expert model
                         (typically gemma_expert.model)
        """
        self.expert_model = expert_model
    
    def forward(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        past_key_values=None,
        use_cache=False,
        adarms_cond=None,
    ):
        """
        Forward pass through the expert model.
        
        Args:
            inputs_embeds: Input embeddings of shape (batch_size, seq_len, hidden_dim)
                         These are the suffix embeddings (state + actions + time)
            attention_mask: Attention mask of shape (batch_size, seq_len, seq_len)
            position_ids: Position IDs of shape (batch_size, seq_len)
            past_key_values: Cached key-value pairs for efficient inference
            use_cache: Whether to return cached key-value pairs
            adarms_cond: Adaptive RMS normalization condition (for PI05)
        
        Returns:
            Model outputs including last_hidden_state and optionally past_key_values
        
        Use Case:
            Processes suffix embeddings (state + noisy actions + timestep) through
            the expert transformer to predict the velocity field for denoising.
            This is the core of the action prediction.
        """
        return self.expert_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            adarms_cond=adarms_cond,
        )
    
    def process_actions(self, action_embeddings, attention_mask, position_ids, **kwargs):
        """
        Action-specific processing through the expert.
        
        Args:
            action_embeddings: Action embeddings from suffix
            attention_mask: Attention mask for actions
            position_ids: Position IDs for actions
            **kwargs: Additional arguments passed to forward()
        
        Returns:
            Processed action embeddings
        
        Use Case:
            Convenience method that emphasizes action processing.
            Calls forward() with action-specific parameters.
        """
        return self.forward(
            inputs_embeds=action_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

