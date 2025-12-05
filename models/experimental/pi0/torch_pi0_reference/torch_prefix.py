"""
Prefix Embedding module for PI-Zero PyTorch model.

This module handles embedding of images and language tokens to create the
prefix part of the sequence (input side) for transformer processing.

Use Case:
    - Prepares the "prefix" part of the sequence (images + language) for
      transformer processing
    - This is the multimodal input that conditions the model
    - Combines visual and language embeddings with proper masking
"""

import math

import torch
from torch import Tensor


class PrefixEmbedding:
    """
    Embeds images + language tokens (input side).
    
    This class handles the embedding of the prefix sequence, which consists
    of images (from multiple camera views) and language tokens. These are
    concatenated together to form the input to the transformer.
    
    Use Case:
        Prepares the "prefix" part of the sequence (images + language) for
        transformer processing. This is the multimodal input that conditions
        the model. The prefix tokens can attend to each other (full attention
        between images and language).
    """
    
    def __init__(self, paligemma_backbone):
        """
        Initialize with PaliGemma backbone.
        
        Args:
            paligemma_backbone: PaliGemmaBackbone instance that provides
                               embed_image and embed_language_tokens methods
        """
        self.paligemma_backbone = paligemma_backbone
    
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """
        Main embedding function for prefix (images + language).
        
        Args:
            images: List of image tensors, each of shape
                   (batch_size, channels, height, width)
            img_masks: List of boolean masks, each of shape (batch_size,)
                      indicating which images are valid
            lang_tokens: Tensor of shape (batch_size, seq_len) with token IDs
            lang_masks: Boolean tensor of shape (batch_size, seq_len)
                       indicating which tokens are valid
        
        Returns:
            Tuple of (prefix_embs, prefix_pad_masks, prefix_att_masks):
            - prefix_embs: Concatenated embeddings of shape
                          (batch_size, total_prefix_len, hidden_dim)
            - prefix_pad_masks: Padding masks of shape
                               (batch_size, total_prefix_len)
            - prefix_att_masks: Attention masks of shape
                              (batch_size, total_prefix_len)
                              where 0 means can attend, 1 means cannot attend
        
        Use Case:
            Main method for creating prefix embeddings. Called during forward
            pass to prepare the input side of the sequence. Returns embeddings
            and masks that will be used in the transformer.
        """
        embs = []
        pad_masks = []
        att_masks = []
        
        # Process images
        image_embs = self._embed_images(images, img_masks)
        for img_emb, img_mask in zip(image_embs, img_masks, strict=True):
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs
        
        # Process language tokens
        lang_emb = self._embed_language(lang_tokens, lang_masks)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        
        # Full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        
        # Concatenate all embeddings
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._concatenate_embeddings(
            embs, pad_masks, att_masks, img_masks, lang_masks
        )
        
        return prefix_embs, prefix_pad_masks, prefix_att_masks
    
    def _embed_images(self, images, img_masks):
        """
        Embed multiple images.
        
        Args:
            images: List of image tensors
            img_masks: List of image masks
        
        Returns:
            List of image embeddings, each of shape
            (batch_size, num_patches, hidden_dim)
        
        Use Case:
            Processes each image through the vision tower to extract visual
            features. Called internally by embed_prefix.
        """
        image_embs = []
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.paligemma_backbone.embed_image(img)
            image_embs.append(img_emb)
        return image_embs
    
    def _embed_language(self, lang_tokens, lang_masks):
        """
        Embed language tokens.
        
        Args:
            lang_tokens: Tensor of shape (batch_size, seq_len) with token IDs
            lang_masks: Boolean tensor of shape (batch_size, seq_len)
        
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_dim) with language
            embeddings, scaled by sqrt(hidden_dim)
        
        Use Case:
            Processes language tokens through the embedding layer and scales
            them. Called internally by embed_prefix.
        """
        lang_emb = self.paligemma_backbone.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        # Scale embeddings by sqrt of dimension (standard practice)
        return lang_emb * math.sqrt(lang_emb_dim)
    
    def _concatenate_embeddings(self, embs, pad_masks, att_masks, img_masks, lang_masks):
        """
        Combine embeddings from images and language.
        
        Args:
            embs: List of embedding tensors
            pad_masks: List of padding mask tensors
            att_masks: List of attention mask values (integers)
            img_masks: List of image masks (for batch size)
            lang_masks: Language masks (for batch size)
        
        Returns:
            Tuple of (concatenated_embs, concatenated_pad_masks, concatenated_att_masks)
        
        Use Case:
            Concatenates all embeddings and masks into single tensors.
            Called internally by embed_prefix.
        """
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        
        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        
        return embs, pad_masks, att_masks

