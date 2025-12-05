"""
Vision Tower interface for PI-Zero PyTorch model.

This module provides an interface to the SigLIP vision model that processes
RGB images from cameras (base, left_wrist, right_wrist) into visual tokens.

Use Case:
    - Extracts visual features from camera images
    - Converts images to embeddings that can be processed by the transformer
    - Provides a clean interface to the vision component of PaliGemma
"""

import torch
from torch import Tensor


class VisionTower:
    """
    Interface to the SigLIP vision model.
    
    This class wraps the vision tower from PaliGemma, providing a clean
    interface for extracting image features.
    
    Use Case:
        Processes RGB images from multiple camera views into visual tokens
        that are concatenated with language tokens to form the prefix.
        Extracted from paligemma.vision_tower.
    """
    
    def __init__(self, vision_model):
        """
        Initialize with SigLIP model.
        
        Args:
            vision_model: The vision tower model from PaliGemma
                         (typically paligemma.vision_tower)
        """
        self.vision_model = vision_model
    
    def forward(self, images):
        """
        Extract image features from RGB images.
        
        Args:
            images: Tensor of shape (batch_size, channels, height, width)
                   Images should be in [-1, 1] range
        
        Returns:
            Tensor of shape (batch_size, num_patches, hidden_dim) with
            visual token embeddings
        
        Use Case:
            Main method for processing images. Called during prefix embedding
            to convert raw images into transformer-compatible embeddings.
        """
        return self.vision_model(images)
    
    def get_image_features(self, image):
        """
        Get vision embeddings for a single image.
        
        Args:
            image: Tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Tensor of shape (batch_size, num_patches, hidden_dim)
        
        Use Case:
            Alias for forward() that provides a more descriptive name.
            Used when we want to emphasize that we're extracting features.
        """
        return self.forward(image)
    
    def embed_image(self, image):
        """
        Alias for get_image_features.
        
        Args:
            image: Tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Tensor of shape (batch_size, num_patches, hidden_dim)
        
        Use Case:
            Provides a consistent naming convention with embed_language_tokens.
            Makes it clear that this is an embedding operation.
        """
        return self.get_image_features(image)

