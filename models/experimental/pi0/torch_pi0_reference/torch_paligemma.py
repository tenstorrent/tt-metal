"""
PaliGemma Backbone wrapper for PI-Zero PyTorch model.

This module provides a wrapper around PaliGemmaWithExpertModel, providing a
unified interface for the vision-language backbone and action expert.

Use Case:
    - Abstracts the complex interaction between vision, language, and expert models
    - Handles the dual-model forward pass where prefix (vision+language) and
      suffix (expert) are processed together
    - Manages precision (bfloat16/float32) for different components
"""

import torch
from torch import Tensor

# Import the actual PaliGemmaWithExpertModel from the existing codebase
import sys
import pathlib

# Add the openpi source to path if needed
openpi_src = pathlib.Path(__file__).parent.parent.parent.parent / "src"
if str(openpi_src) not in sys.path:
    sys.path.insert(0, str(openpi_src))

from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models.gemma as _gemma


class PaliGemmaBackbone:
    """
    Wrapper for PaliGemmaWithExpertModel.
    
    This class provides a clean interface to the complex PaliGemmaWithExpertModel,
    which combines vision (SigLIP), language (Gemma), and action expert (Gemma)
    into a unified model.
    
    Use Case:
        Abstracts the complex interaction between vision, language, and expert models.
        Handles the dual-model forward pass where prefix (vision+language) and
        suffix (expert) are processed together. This is the core backbone of PI-Zero.
    """
    
    def __init__(self, paligemma_config, action_expert_config, use_adarms, precision):
        """
        Initialize the PaliGemma backbone.
        
        Args:
            paligemma_config: Configuration for PaliGemma (vision-language model)
            action_expert_config: Configuration for action expert (Gemma)
            use_adarms: List of [paligemma_use_adarms, expert_use_adarms] booleans
                       For PI05, typically [False, True]
            precision: Precision to use ("bfloat16" or "float32")
        """
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=use_adarms,
            precision=precision,
        )
        self.precision = precision
    
    def embed_image(self, image):
        """
        Embed image using the vision tower (SigLIP).
        
        Args:
            image: Tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Tensor of shape (batch_size, num_patches, hidden_dim) with
            visual token embeddings
        
        Use Case:
            Extracts visual features from images using SigLIP. Called during
            prefix embedding to convert raw images into embeddings.
        """
        return self.paligemma_with_expert.embed_image(image)
    
    def embed_language_tokens(self, tokens):
        """
        Embed language tokens using the language model (Gemma).
        
        Args:
            tokens: Tensor of shape (batch_size, seq_len) with token IDs
        
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_dim) with
            language token embeddings
        
        Use Case:
            Converts discrete token IDs into continuous embeddings. Called
            during prefix embedding to process language prompts.
        """
        return self.paligemma_with_expert.embed_language_tokens(tokens)
    
    def forward(
        self,
        prefix_embs,
        suffix_embs,
        attention_mask,
        position_ids,
        past_key_values=None,
        use_cache=False,
        adarms_cond=None,
    ):
        """
        Joint forward pass through PaliGemma and expert models.
        
        This processes prefix (vision+language) and suffix (state+actions+time)
        together, with proper attention masking to ensure prefix tokens can
        attend to each other but suffix has causal attention.
        
        Args:
            prefix_embs: Prefix embeddings (images + language) of shape
                        (batch_size, prefix_len, hidden_dim)
            suffix_embs: Suffix embeddings (state + actions + time) of shape
                        (batch_size, suffix_len, hidden_dim)
            attention_mask: 4D attention mask of shape (batch_size, 1, seq_len, seq_len)
            position_ids: Position IDs of shape (batch_size, seq_len)
            past_key_values: Cached key-value pairs for efficient inference
            use_cache: Whether to return cached key-value pairs
            adarms_cond: List of [paligemma_adarms_cond, expert_adarms_cond]
                        For PI05, typically [None, time_emb]
        
        Returns:
            Tuple of (prefix_output, suffix_output) and past_key_values
            - prefix_output: Output from PaliGemma language model
            - suffix_output: Output from expert model (used for action prediction)
            - past_key_values: Cached key-value pairs (if use_cache=True)
        
        Use Case:
            Core forward pass that processes both prefix and suffix together.
            The models share attention computation but have separate transformer
            layers. This is the main computation in both training and inference.
        """
        return self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=use_cache,
            adarms_cond=adarms_cond,
        )
    
    def to_bfloat16_for_selected_params(self, precision):
        """
        Manage precision for different model components.
        
        Some components (like vision embeddings) need to stay in float32
        for numerical stability, while others can use bfloat16 for efficiency.
        
        Args:
            precision: Target precision ("bfloat16" or "float32")
        
        Use Case:
            Optimizes memory usage by using bfloat16 where possible while
            maintaining numerical stability for sensitive operations.
        """
        self.paligemma_with_expert.to_bfloat16_for_selected_params(precision)
        self.precision = precision

