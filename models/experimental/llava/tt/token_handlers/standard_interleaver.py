# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Standard Token Interleaver - TTNN Implementation

Interleaves text embeddings with image features at <image> placeholder
positions, matching LLaVA's prepare_inputs_labels_for_multimodal().

LLaVA 1.5 uses a single IMAGE_TOKEN_INDEX (-200) as a placeholder for
each image. This token is replaced by the projected image features
(576 tokens for CLIP ViT-L/14-336), making the output sequence longer
than the input.

The interleaving is a host-side operation (variable-length split/concat),
so this implementation operates on torch tensors and returns a ttnn tensor
on device. This follows the qwen25_vl pattern where merge_vision_tokens
runs on host before the result is moved to device.

Example:
    Input IDs:  [<image>, "USER", ":", "What", "?"]  (5 tokens)
    Text embeds: [E_dummy, E_USER, E_colon, E_what, E_?]  [1, 5, 4096]
    Image feats: [F0, F1, ..., F575]  [1, 576, 4096]

    Output: [F0, ..., F575, E_USER, E_colon, E_what, E_?]  [1, 580, 4096]

Reference:
    - LLaVA/llava/model/llava_arch.py:prepare_inputs_labels_for_multimodal
    - models/demos/qwen25_vl/tt/common.py:merge_vision_tokens (host-side pattern)
    - LLaVA/llava/constants.py (IMAGE_TOKEN_INDEX=-200, IGNORE_INDEX=-100)
"""

from typing import Any, Dict, Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule

# LLaVA constants
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


class StandardInterleaver(LightweightModule):
    """
    Standard Token Interleaver - TTNN Implementation

    Interleaves text embeddings with image features at IMAGE_TOKEN_INDEX positions.
    The core interleaving runs on host (torch) because it involves variable-length
    split and concat operations. The result is converted to a ttnn tensor on device.

    Args:
        args: VisionModelArgs with device configuration
        image_token_index: Token ID marking image positions (default: -200)
        ignore_index: Label value for ignored positions (default: -100)

    Input:
        input_ids: torch.Tensor [B, seq_len] with IMAGE_TOKEN_INDEX at image positions
        text_embeds: torch.Tensor [B, seq_len, hidden_size] embeddings of all tokens
        image_features: torch.Tensor [B, num_patches, hidden_size] projected vision features

    Output (dict):
        input_embeds: ttnn.Tensor [B, new_seq_len, hidden_size] on device
        attention_mask: torch.Tensor [B, new_seq_len] on host
        position_ids: torch.Tensor [B, new_seq_len] on host
        labels: torch.Tensor [B, new_seq_len] or None on host
    """

    def __init__(self, args, image_token_index=IMAGE_TOKEN_INDEX, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.args = args
        self.mesh_device = args.mesh_device
        self.image_token_index = image_token_index
        self.ignore_index = ignore_index

        logger.info("StandardInterleaver initialized (host-side interleaving)")

    def _interleave_host(self, input_ids, text_embeds, image_features, labels=None, attention_mask=None):
        """
        Host-side interleaving using torch tensors.

        This is the core logic matching LLaVA's prepare_inputs_labels_for_multimodal().
        """
        batch_size = input_ids.shape[0]
        hidden_size = text_embeds.shape[-1]

        # Strip padding using attention_mask (matching LLaVA reference)
        if attention_mask is not None:
            stripped_input_ids = []
            stripped_text_embeds = []
            stripped_labels = [] if labels is not None else None
            for i in range(batch_size):
                mask = attention_mask[i].bool()
                stripped_input_ids.append(input_ids[i][mask])
                stripped_text_embeds.append(text_embeds[i][mask])
                if labels is not None:
                    stripped_labels.append(labels[i][mask])
            input_ids = stripped_input_ids
            text_embeds = stripped_text_embeds
            labels = stripped_labels
        else:
            input_ids = [input_ids[i] for i in range(batch_size)]
            text_embeds = [text_embeds[i] for i in range(batch_size)]
            labels = [labels[i] for i in range(batch_size)] if labels is not None else None

        # Handle image_features as [B, num_patches, hidden_size] → list per batch
        if isinstance(image_features, torch.Tensor) and image_features.dim() == 3:
            image_features_list = [image_features[i] for i in range(batch_size)]
        elif isinstance(image_features, list):
            image_features_list = image_features
        else:
            image_features_list = [image_features]

        # Store label dtype for creating IGNORE_INDEX tensors
        label_dtype = labels[0].dtype if labels is not None else torch.long

        new_input_embeds = []
        new_labels = []

        for batch_idx in range(batch_size):
            cur_input_ids = input_ids[batch_idx]
            cur_text_embeds = text_embeds[batch_idx]
            # Find image token positions
            image_token_indices = torch.where(cur_input_ids == self.image_token_index)[0]
            num_images = len(image_token_indices)

            if num_images == 0:
                new_input_embeds.append(cur_text_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue

            # Build split points: [-1, img_pos_0, ..., seq_len]
            split_points = [-1] + image_token_indices.tolist() + [cur_input_ids.shape[0]]

            # Extract text segments between image tokens (excluding image tokens)
            text_segments = []
            for i in range(len(split_points) - 1):
                start = split_points[i] + 1
                end = split_points[i + 1]
                if end > start:
                    text_segments.append(cur_text_embeds[start:end])
                else:
                    text_segments.append(cur_text_embeds[0:0])

            # Interleave: text[0] + image[0] + text[1] + ... + text[N]
            cur_new_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_embeds.append(text_segments[i])
                if labels is not None:
                    seg_start = split_points[i] + 1
                    seg_end = split_points[i + 1]
                    if seg_end > seg_start:
                        cur_new_labels.append(labels[batch_idx][seg_start:seg_end])
                    else:
                        cur_new_labels.append(torch.empty(0, dtype=label_dtype))

                if i < num_images:
                    img_idx = min(i, len(image_features_list) - 1)
                    img_feat = image_features_list[img_idx]
                    cur_new_embeds.append(img_feat)
                    if labels is not None:
                        cur_new_labels.append(
                            torch.full(
                                (img_feat.shape[0],),
                                self.ignore_index,
                                dtype=label_dtype,
                            )
                        )

            cur_new_embeds = torch.cat(cur_new_embeds, dim=0)
            new_input_embeds.append(cur_new_embeds)

            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # Pad to max length (right-padding, LLaVA default)
        max_len = max(x.shape[0] for x in new_input_embeds)
        padded_embeds = torch.zeros(batch_size, max_len, hidden_size, dtype=new_input_embeds[0].dtype)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

        if labels is not None:
            padded_labels = torch.full((batch_size, max_len), self.ignore_index, dtype=label_dtype)
        else:
            padded_labels = None

        for i, cur_embed in enumerate(new_input_embeds):
            cur_len = cur_embed.shape[0]
            padded_embeds[i, :cur_len] = cur_embed
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long)
            if padded_labels is not None and cur_len > 0:
                padded_labels[i, :cur_len] = new_labels[i]

        return padded_embeds, attention_mask, position_ids, padded_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        image_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Interleave text embeddings with image features.

        Args:
            input_ids: [B, seq_len] token IDs with IMAGE_TOKEN_INDEX at image positions
            text_embeds: [B, seq_len, hidden_size] embeddings of all tokens
            image_features: [B, num_patches, hidden_size] projected vision features
            labels: Optional [B, seq_len] labels
            attention_mask: Optional [B, seq_len] mask to strip padding before interleaving
                           (1 for real tokens, 0 for padding). If None, all tokens are processed.

        Returns:
            dict with:
                input_embeds: ttnn.Tensor [B, new_seq_len, hidden_size] on device
                attention_mask: torch.Tensor [B, new_seq_len] on host
                position_ids: torch.Tensor [B, new_seq_len] on host
                labels: torch.Tensor [B, new_seq_len] or None on host
        """
        # Step 1: Host-side interleaving
        padded_embeds, out_attention_mask, position_ids, padded_labels = self._interleave_host(
            input_ids, text_embeds, image_features, labels, attention_mask
        )

        logger.debug(f"Interleaved: input_ids {input_ids.shape} -> input_embeds {padded_embeds.shape}")

        # Step 2: Convert input_embeds to ttnn tensor on device
        input_embeds_ttnn = ttnn.from_torch(
            padded_embeds,
            dtype=self.args.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return {
            "input_embeds": input_embeds_ttnn,
            "attention_mask": out_attention_mask,
            "position_ids": position_ids,
            "labels": padded_labels,
        }


def create_standard_interleaver(args):
    """Factory function to create StandardInterleaver"""
    return StandardInterleaver(args)
