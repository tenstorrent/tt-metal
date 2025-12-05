"""
Language Model interface for PI-Zero PyTorch model.

This module provides an interface to the Gemma language model that processes
natural language prompts/instructions into language tokens.

Use Case:
    - Embeds tokenized language prompts into embeddings
    - Processes language tokens through the transformer
    - Provides a clean interface to the language component of PaliGemma
"""

import torch
from torch import Tensor


class LanguageModel:
    """
    Interface to the Gemma language model.
    
    This class wraps the language model from PaliGemma, providing a clean
    interface for processing language tokens.
    
    Use Case:
        Processes natural language prompts/instructions into language tokens
        that are concatenated with image tokens to form the prefix.
        Extracted from paligemma.language_model.
    """
    
    def __init__(self, language_model):
        """
        Initialize with Gemma model.
        
        Args:
            language_model: The language model from PaliGemma
                           (typically paligemma.language_model)
        """
        self.language_model = language_model
    
    def embed_tokens(self, tokens):
        """
        Embed language tokens into embeddings.
        
        Args:
            tokens: Tensor of shape (batch_size, seq_len) with token IDs
        
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_dim) with
            language token embeddings
        
        Use Case:
            Converts discrete token IDs into continuous embeddings that can
            be processed by the transformer. Called during prefix embedding.
        """
        return self.language_model.embed_tokens(tokens)
    
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
        Forward pass through the language model.
        
        Args:
            inputs_embeds: Input embeddings of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask of shape (batch_size, seq_len, seq_len)
            position_ids: Position IDs of shape (batch_size, seq_len)
            past_key_values: Cached key-value pairs for efficient inference
            use_cache: Whether to return cached key-value pairs
            adarms_cond: Adaptive RMS normalization condition (for PI05)
        
        Returns:
            Model outputs including last_hidden_state and optionally past_key_values
        
        Use Case:
            Processes language embeddings through the transformer layers.
            Used when we need full language model processing, not just embedding.
        """
        return self.language_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            adarms_cond=adarms_cond,
        )
    
    def embed_language_tokens(self, tokens):
        """
        Alias for embed_tokens.
        
        Args:
            tokens: Tensor of shape (batch_size, seq_len) with token IDs
        
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_dim)
        
        Use Case:
            Provides a consistent naming convention with embed_image.
            Makes it clear that this is an embedding operation.
        """
        return self.embed_tokens(tokens)

