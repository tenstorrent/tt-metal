# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefix Embedding module - TTNN Implementation

This module handles embedding of images and language tokens to create the
prefix part of the sequence for transformer processing.

Components:
    - Image embedding via SigLIP vision tower
    - Language token embedding via Gemma embeddings
    - Concatenation of image and language embeddings with proper masking

Attention Pattern:
    - All prefix tokens can attend to each other (bidirectional)
    - Suffix tokens can attend to prefix (cross-attention)
"""

import math
from typing import List, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PrefixConfig


class PrefixEmbeddingTTNN:
    """
    TTNN implementation of prefix embedding.

    Uses TTNN operations for efficient execution on Tenstorrent hardware.
    """

    def __init__(
        self,
        config: PrefixConfig,
        device: ttnn.Device,
        embed_image_fn=None,
        embed_language_fn=None,
    ):
        """
        Initialize prefix embedding with TTNN.

        Args:
            config: Prefix configuration
            device: TTNN device
            embed_image_fn: Function to embed images
            embed_language_fn: Function to embed language tokens
        """
        self.config = config
        self.device = device
        self.embed_image_fn = embed_image_fn
        self.embed_language_fn = embed_language_fn

        self.prefix_att_masks = ttnn.zeros(
            (1, 544),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def embed_images(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
    ) -> Tuple[List[ttnn.Tensor], List[ttnn.Tensor]]:
        """
        Embed multiple images using TTNN.

        Args:
            images: List of PyTorch image tensors (vision tower handles TTNN conversion)
            img_masks: List of PyTorch mask tensors

        Returns:
            Tuple of (image_embeddings, expanded_masks) as TTNN tensors
        """
        if self.embed_image_fn is None:
            raise RuntimeError("embed_image_fn not set")

        image_embs: List[ttnn.Tensor] = []
        expanded_masks: List[ttnn.Tensor] = []

        # PI0_SIGLIP_SKIP_MASKED=1 opt-in: skip the SigLIP forward pass for images
        # whose mask is all-False, then inject a zero embedding for their slot so
        # downstream prefix layout (and prefix_len) is unchanged. Math-equivalent
        # because the prefix attention mask zeros out the contribution of these
        # tokens at every attention layer (VLM self-attn + expert joint attn).
        # Saves SigLIP+projector cost on the masked images (~33% at bs=3 single-arm
        # LIBERO, where img3 is always a zero/-1 placeholder).
        #
        # Two ways to specify which images to skip:
        # 1. Host-side: img_masks are torch.Tensor (non-trace path) — auto-detect.
        # 2. Explicit env: PI0_SIGLIP_KEEP_MASK="1,1,0" (1=keep, 0=skip) — required for
        #    trace path where img_masks reach embed_images as ttnn.Tensor (can't read
        #    device → host without a sync, which trace forbids).
        import os as _os

        _skip_enabled = _os.environ.get("PI0_SIGLIP_SKIP_MASKED", "").lower() in ("1", "true", "yes", "on")
        keep_indices: List[int] = list(range(len(images)))
        skip_indices: List[int] = []
        if _skip_enabled and len(images) > 1:
            _explicit = _os.environ.get("PI0_SIGLIP_KEEP_MASK", "").strip()
            if _explicit:
                try:
                    pattern = [bool(int(x)) for x in _explicit.split(",")]
                    if len(pattern) == len(images):
                        keep_indices = [i for i, k in enumerate(pattern) if k]
                        skip_indices = [i for i in range(len(images)) if i not in keep_indices]
                except (ValueError, IndexError):
                    pass  # malformed env var → fall through to host-side check
            if not skip_indices:
                # Host-side mask read; only safe for torch.Tensor or pre-staged host views.
                _readable = all(isinstance(m, torch.Tensor) for m in img_masks)
                if _readable:
                    keep_indices = [i for i, m in enumerate(img_masks) if bool(m.any().item())]
                    skip_indices = [i for i in range(len(images)) if i not in keep_indices]
                # else: masks are device tensors (trace path) — keep all (no-op).

        # OPTIMIZATION: when multiple images share the same shape, run SigLIP
        # in a single bs=N pass instead of N sequential bs=1 calls. Halves
        # the kernel dispatch count for the vision tower.
        images_for_siglip = [images[i] for i in keep_indices]
        # PI0_SIGLIP_USE_FOLD=1 path: caller pre-stacks all cameras on host into a single
        # (N, H, W, 3) NHWC ROW_MAJOR tensor and passes a list of length 1. Detect by
        # (a) only one tensor in the list AND (b) batch dim > 1 in its shape.
        _use_fold = _os.environ.get("PI0_SIGLIP_USE_FOLD", "").lower() in ("1", "true", "yes", "on")
        pre_stacked = (
            _use_fold
            and len(images_for_siglip) == 1
            and isinstance(images_for_siglip[0], ttnn.Tensor)
            and int(images_for_siglip[0].shape[0]) > 1
        )
        same_shape = (
            len(images_for_siglip) > 1
            and isinstance(images_for_siglip[0], ttnn.Tensor)
            and all(
                isinstance(im, ttnn.Tensor) and tuple(im.shape) == tuple(images_for_siglip[0].shape)
                for im in images_for_siglip
            )
        )
        kept_embs: List[ttnn.Tensor] = []
        if pre_stacked:
            # Caller already concatenated all cameras on host — skip the device concat
            # entirely (saves ~0.5 ms on ROW_MAJOR concat at bs=3).
            stacked = images_for_siglip[0]
            n_kept = int(stacked.shape[0])
            all_embs = self.embed_image_fn(stacked)  # (N, num_tokens, vlm_hidden)
            num_tokens = all_embs.shape[1]
            hidden = all_embs.shape[2]
            for i in range(n_kept):
                img_emb = ttnn.slice(all_embs, [i, 0, 0], [i + 1, num_tokens, hidden])
                if img_emb.layout != ttnn.TILE_LAYOUT:
                    img_emb = ttnn.to_layout(img_emb, ttnn.TILE_LAYOUT)
                kept_embs.append(img_emb)
            ttnn.deallocate(all_embs)
            # Override keep/skip to reflect n_kept cameras (one per slice) so the
            # downstream reconstruction loops don't try to use keep_indices=[0]
            # against kept_embs of length N.
            keep_indices = list(range(n_kept))
            skip_indices = []
            # Override `images` length-wise so downstream "for img_emb, mask in
            # zip(image_embs, img_masks)" matches the right number of cameras.
            images = [stacked] * n_kept
        elif same_shape:
            if _use_fold and images_for_siglip[0].layout == ttnn.ROW_MAJOR_LAYOUT:
                stacked = ttnn.concat(images_for_siglip, dim=0)  # (N_kept, H, W, C) ROW_MAJOR
            else:
                # Original BCHW TILE path
                imgs_tiled = [
                    im if im.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(im, ttnn.TILE_LAYOUT)
                    for im in images_for_siglip
                ]
                stacked = ttnn.concat(imgs_tiled, dim=0)  # (N_kept, C, H, W)
            all_embs = self.embed_image_fn(stacked)  # (N_kept, num_tokens, vlm_hidden)
            ttnn.deallocate(stacked)
            num_tokens = all_embs.shape[1]
            hidden = all_embs.shape[2]
            for i in range(len(images_for_siglip)):
                img_emb = ttnn.slice(all_embs, [i, 0, 0], [i + 1, num_tokens, hidden])
                if img_emb.layout != ttnn.TILE_LAYOUT:
                    img_emb = ttnn.to_layout(img_emb, ttnn.TILE_LAYOUT)
                kept_embs.append(img_emb)
            ttnn.deallocate(all_embs)
        else:
            for img in images_for_siglip:
                emb = self.embed_image_fn(img)
                if emb.layout != ttnn.TILE_LAYOUT:
                    emb = ttnn.to_layout(emb, ttnn.TILE_LAYOUT)
                kept_embs.append(emb)

        # Reconstruct full N-length image_embs: real embeddings at keep_indices,
        # zero embeddings at skip_indices (preserves downstream prefix layout +
        # position IDs). The zero contribution is masked out at attention time.
        if skip_indices and kept_embs:
            template = kept_embs[0]
            zero_emb = ttnn.zeros(
                template.shape,
                dtype=template.dtype,
                layout=template.layout,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            image_embs = [None] * len(images)
            for slot, emb in zip(keep_indices, kept_embs):
                image_embs[slot] = emb
            for slot in skip_indices:
                # Each skipped slot gets its own zero tensor (avoid aliasing
                # across slots since downstream may deallocate independently).
                image_embs[slot] = (
                    zero_emb
                    if slot == skip_indices[0]
                    else ttnn.zeros(
                        template.shape,
                        dtype=template.dtype,
                        layout=template.layout,
                        device=self.device,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                )
        else:
            image_embs = kept_embs

        # Build expanded masks (mask handling unchanged — masks may be per-image)
        for img_emb, mask in zip(image_embs, img_masks):
            shape = img_emb.shape
            batch_size, num_tokens = shape[0], shape[1]

            if isinstance(mask, torch.Tensor):
                mask_ttnn = ttnn.from_torch(
                    mask.float(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                mask_ttnn = ttnn.reshape(mask_ttnn, (batch_size, 1))
                mask_ttnn = ttnn.to_layout(mask_ttnn, ttnn.TILE_LAYOUT)
                expanded_mask = ttnn.repeat(mask_ttnn, (1, num_tokens), memory_config=ttnn.L1_MEMORY_CONFIG)
            else:
                mask_reshaped = ttnn.reshape(mask, (batch_size, 1))
                expanded_mask = ttnn.repeat(mask_reshaped, (1, num_tokens), memory_config=ttnn.L1_MEMORY_CONFIG)

            expanded_masks.append(expanded_mask)

        return image_embs, expanded_masks

    def embed_language(
        self,
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Embed language tokens using TTNN.

        Args:
            lang_tokens: TTNN tensor of token IDs
            lang_masks: TTNN tensor of validity masks

        Returns:
            TTNN tensor of scaled embeddings
        """
        if self.embed_language_fn is None:
            raise RuntimeError("embed_language_fn not set")

        lang_emb = self.embed_language_fn(lang_tokens)

        # ttnn.embedding returns ROW_MAJOR; convert to TILE so downstream
        # concat with TILE image embeddings works for any token length.
        if lang_emb.layout != ttnn.TILE_LAYOUT:
            lang_emb = ttnn.to_layout(lang_emb, ttnn.TILE_LAYOUT)

        # Scale by sqrt(hidden_dim) - use scalar multiply
        hidden_dim = lang_emb.shape[-1]
        scale = math.sqrt(hidden_dim)

        return ttnn.mul(lang_emb, scale)

    def embed_prefix(
        self,
        images: List[ttnn.Tensor],
        img_masks: List[ttnn.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Main embedding function for prefix (TTNN version).

        Args:
            images: List of TTNN image tensors
            img_masks: List of TTNN mask tensors
            lang_tokens: TTNN tensor of language tokens
            lang_masks: TTNN tensor of language masks

        Returns:
            Tuple of (prefix_embs, prefix_pad_masks, prefix_att_masks)
        """
        embs = []
        pad_masks = []
        num_tokens_list = []

        # Process images
        if images and self.embed_image_fn is not None:
            image_embs, img_pad_masks = self.embed_images(images, img_masks)
            for img_emb, img_pad_mask in zip(image_embs, img_pad_masks):
                embs.append(img_emb)
                pad_masks.append(img_pad_mask)
                num_tokens_list.append(img_emb.shape[1])

        # Process language
        if self.embed_language_fn is not None:
            lang_emb = self.embed_language(lang_tokens, lang_masks)
            embs.append(lang_emb)
            pad_masks.append(lang_masks)
            num_tokens_list.append(lang_emb.shape[1])

        # Defensive: TTNN concat requires all inputs in the same layout.
        embs = [e if e.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(e, ttnn.TILE_LAYOUT) for e in embs]
        pad_masks = [m if m.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(m, ttnn.TILE_LAYOUT) for m in pad_masks]
        prefix_embs = ttnn.concat(embs, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        prefix_pad_masks = ttnn.concat(pad_masks, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Create attention mask (all zeros for bidirectional prefix attention)
        total_tokens = sum(num_tokens_list)
        batch_size = prefix_embs.shape[0]

        # Create zeros mask directly on device (no host transfer needed)
        prefix_att_masks = self.prefix_att_masks

        return prefix_embs, prefix_pad_masks, prefix_att_masks


# Default export
PrefixEmbedding = PrefixEmbeddingTTNN
