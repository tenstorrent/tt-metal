# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation for LLaVA vision encoder

This module provides baseline implementations for PCC validation.
Used as golden outputs for comparing against TTNN implementations.

Reference:
    - LLaVA/llava/model/multimodal_encoder/clip_encoder.py
    - transformers.CLIPVisionModel
    - openai/clip-vit-large-patch14-336

Architecture (CLIP ViT-L/14 @ 336px):
    - Input: [B, 3, 336, 336]
    - Patch embedding: 14×14 patches → [B, 577, 1024]  (576 patches + 1 CLS token)
    - Transformer: 24 layers, 16 heads, hidden=1024, intermediate=4096
    - Output: [B, 576, 1024]  (drop CLS token)
"""

import torch
import torch.nn as nn
from loguru import logger
from transformers import CLIPImageProcessor, CLIPVisionModel


class CLIPVisionEncoder(nn.Module):
    """
    CLIP Vision Encoder - PyTorch Reference Implementation

    This wraps HuggingFace CLIPVisionModel to match LLaVA behavior:
    1. Load pre-trained CLIP ViT-L/14-336
    2. Extract features from specified layer (default: -2, second-to-last)
    3. Drop CLS token (keep only patch tokens)

    Args:
        model_name: HuggingFace model name (default: openai/clip-vit-large-patch14-336)
        select_layer: Which hidden layer to extract (-2 = second-to-last)
        select_feature: 'patch' (drop CLS) or 'cls_patch' (keep CLS)

    Input:
        images: torch.Tensor [B, 3, 336, 336]

    Output:
        features: torch.Tensor [B, 576, 1024]

    Reference:
        - LLaVA uses select_layer=-2 (not final layer!)
        - LLaVA uses select_feature='patch' (drops CLS token)
    """

    def __init__(
        self,
        model_name="openai/clip-vit-large-patch14-336",
        select_layer=-2,
        select_feature="patch",
    ):
        super().__init__()

        self.model_name = model_name
        self.select_layer = select_layer
        self.select_feature = select_feature

        logger.info(f"Loading CLIP vision encoder: {model_name}")
        logger.info(f"  Select layer: {select_layer}")
        logger.info(f"  Select feature: {select_feature}")

        # Load pre-trained CLIP vision model
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.vision_model.eval()  # Set to eval mode
        self.vision_model.requires_grad_(False)  # Freeze weights

        # Load image processor (for preprocessing)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)

        # Cache config
        self.config = self.vision_model.config

        logger.info(f"  Hidden size: {self.config.hidden_size}")
        logger.info(f"  Num layers: {self.config.num_hidden_layers}")
        logger.info(f"  Num heads: {self.config.num_attention_heads}")
        logger.info(f"  Image size: {self.config.image_size}")
        logger.info(f"  Patch size: {self.config.patch_size}")
        logger.info(f"  Num patches: {self.num_patches}")

    def feature_select(self, image_forward_outs):
        """
        Select features from CLIP outputs

        Args:
            image_forward_outs: CLIPVisionModelOutput with hidden_states

        Returns:
            features: Selected features [B, num_patches, hidden_size]
        """
        # Extract features from specified layer
        image_features = image_forward_outs.hidden_states[self.select_layer]

        # Select patch or cls_patch
        if self.select_feature == "patch":
            # Drop CLS token (index 0), keep only patch tokens
            image_features = image_features[:, 1:]  # [B, 576, 1024]
        elif self.select_feature == "cls_patch":
            # Keep both CLS and patch tokens
            image_features = image_features  # [B, 577, 1024]
        else:
            raise ValueError(f"Invalid select_feature: {self.select_feature}")

        return image_features

    @torch.no_grad()
    def forward(self, images):
        """
        Forward pass through CLIP vision encoder

        Args:
            images: torch.Tensor [B, 3, H, W] or list of tensors

        Returns:
            features: torch.Tensor [B, num_patches, hidden_size]
        """
        # Handle list of images
        if isinstance(images, list):
            image_features = []
            for image in images:
                # Add batch dimension if needed
                if image.dim() == 3:
                    image = image.unsqueeze(0)

                # Forward through vision model
                image_forward_out = self.vision_model(
                    image.to(device=self.device, dtype=self.dtype), output_hidden_states=True
                )

                # Select features
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)

            return image_features
        else:
            # Single batch
            image_forward_outs = self.vision_model(
                images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

            return image_features

    @property
    def device(self):
        return self.vision_model.device

    @property
    def dtype(self):
        return self.vision_model.dtype

    @property
    def hidden_size(self):
        """Output feature dimension"""
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        """Number of patches per side (e.g., 24 for 336/14)"""
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        """Total number of patches (e.g., 576 for 24×24)"""
        return (self.config.image_size // self.config.patch_size) ** 2


def create_clip_encoder(model_name="openai/clip-vit-large-patch14-336", device="cuda"):
    """
    Factory function to create CLIP encoder

    Args:
        model_name: HuggingFace model name
        device: Device to load model on

    Returns:
        CLIPVisionEncoder
    """
    encoder = CLIPVisionEncoder(
        model_name=model_name,
        select_layer=-2,  # LLaVA default
        select_feature="patch",  # LLaVA default (drop CLS)
    )
    encoder = encoder.to(device)
    return encoder


class SigLIPVisionEncoder(nn.Module):
    """
    SigLIP Vision Encoder - PyTorch Reference Implementation

    Wraps HuggingFace `SiglipVisionModel` to match LLaVA-MORE behavior:
    1. Load pre-trained `google/siglip-so400m-patch14-384`
    2. Extract features from specified layer (default: -2, second-to-last)
    3. No CLS token to drop (SigLIP has only patch tokens)

    Args:
        model_name: HuggingFace model name (default: google/siglip-so400m-patch14-384)
        select_layer: Which hidden layer to extract (-2 = second-to-last)
        select_feature: 'patch' (no-op for SigLIP) or 'cls_patch' (no-op)

    Input:
        images: torch.Tensor [B, 3, 384, 384]

    Output:
        features: torch.Tensor [B, 729, 1152]

    Reference:
        - LLaVA-MORE/src/llava/model/multimodal_encoder/siglip_encoder.py
        - transformers.SiglipVisionModel
    """

    def __init__(
        self,
        model_name="google/siglip-so400m-patch14-384",
        select_layer=-2,
        select_feature="patch",
    ):
        super().__init__()

        from transformers import SiglipImageProcessor, SiglipVisionModel

        self.model_name = model_name
        self.select_layer = select_layer
        self.select_feature = select_feature

        logger.info(f"Loading SigLIP vision encoder: {model_name}")
        logger.info(f"  Select layer: {select_layer}")

        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        self.vision_model.eval()
        self.vision_model.requires_grad_(False)

        self.image_processor = SiglipImageProcessor.from_pretrained(model_name)
        self.config = self.vision_model.config

        logger.info(f"  Hidden size: {self.config.hidden_size}")
        logger.info(f"  Num layers: {self.config.num_hidden_layers}")
        logger.info(f"  Num heads: {self.config.num_attention_heads}")
        logger.info(f"  Image size: {self.config.image_size}")
        logger.info(f"  Patch size: {self.config.patch_size}")
        logger.info(f"  Num patches: {self.num_patches}")

    def feature_select(self, image_forward_outs):
        """
        Select features from SigLIP outputs.

        SigLIP has no CLS token, so 'patch' vs 'cls_patch' are no-ops here;
        the implementation mirrors LLaVA-MORE's `siglip_encoder.py:45`
        ("not remove the first token").
        """
        image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == "patch":
            pass
        elif self.select_feature == "cls_patch":
            pass
        else:
            raise ValueError(f"Invalid select_feature: {self.select_feature}")

        return image_features

    @torch.no_grad()
    def forward(self, images):
        """
        Forward pass through SigLIP vision encoder

        Args:
            images: torch.Tensor [B, 3, 384, 384] or list of tensors

        Returns:
            features: torch.Tensor [B, 729, 1152]
        """
        if isinstance(images, list):
            image_features = []
            for image in images:
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                image_forward_out = self.vision_model(
                    image.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
            return image_features
        else:
            image_forward_outs = self.vision_model(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            return image_features

    @property
    def device(self):
        return self.vision_model.device

    @property
    def dtype(self):
        return self.vision_model.dtype

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


def create_siglip_encoder(model_name="google/siglip-so400m-patch14-384", device="cuda"):
    """
    Factory function to create SigLIP encoder

    Args:
        model_name: HuggingFace model name
        device: Device to load model on

    Returns:
        SigLIPVisionEncoder
    """
    encoder = SigLIPVisionEncoder(
        model_name=model_name,
        select_layer=-2,
        select_feature="patch",
    )
    encoder = encoder.to(device)
    return encoder


class MLPProjector(nn.Module):
    """
    MLP Projector - PyTorch Reference Implementation

    LLaVA 1.5 uses a 2-layer MLP with GELU activation to project
    vision encoder features into the LLM embedding space.

    Architecture (mlp2x_gelu):
        Linear(vision_hidden_size → llm_hidden_size) → GELU → Linear(llm_hidden_size → llm_hidden_size)

    For LLaVA 1.5 7B:
        Linear(1024 → 4096) → GELU → Linear(4096 → 4096)

    Args:
        vision_hidden_size: Input dimension (CLIP output = 1024)
        llm_hidden_size: Output dimension (LLaMA hidden = 4096)
        state_dict: Optional pretrained weights (keys: mm_projector.0/2.weight/bias)
        projector_type: "mlp2x_gelu" (default, only supported type)

    Input:
        features: torch.Tensor [B, num_patches, vision_hidden_size]

    Output:
        projected: torch.Tensor [B, num_patches, llm_hidden_size]

    Reference:
        - LLaVA/llava/model/multimodal_projector/builder.py
        - config.mm_projector_type = "mlp2x_gelu"
    """

    def __init__(
        self,
        vision_hidden_size=1024,
        llm_hidden_size=4096,
        state_dict=None,
        projector_type="mlp2x_gelu",
    ):
        super().__init__()

        assert projector_type == "mlp2x_gelu", f"Only mlp2x_gelu supported, got {projector_type}"

        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size

        # Build Sequential: Linear → GELU → Linear
        # This matches LLaVA's build_vision_projector() for mlp2x_gelu
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),  # [0]
            nn.GELU(),  # [1]
            nn.Linear(llm_hidden_size, llm_hidden_size),  # [2]
        )

        # Load pretrained weights if provided
        if state_dict is not None:
            self._load_state_dict(state_dict)

        self.eval()
        self.requires_grad_(False)

    def _load_state_dict(self, state_dict):
        """Load weights from state_dict, handling key prefix variations"""
        projector_sd = {}

        for key, val in state_dict.items():
            # Strip "mm_projector." prefix if present
            if key.startswith("mm_projector."):
                stripped = key[len("mm_projector.") :]
                projector_sd[stripped] = val
            elif key.startswith("0.") or key.startswith("2."):
                # Already without prefix
                projector_sd[key] = val

        if projector_sd:
            self.projector.load_state_dict(projector_sd, strict=True)

    @torch.no_grad()
    def forward(self, features):
        """
        Forward pass through MLP projector

        Args:
            features: torch.Tensor [B, num_patches, vision_hidden_size]

        Returns:
            projected: torch.Tensor [B, num_patches, llm_hidden_size]
        """
        return self.projector(features)

    @property
    def hidden_size(self):
        """Output dimension"""
        return self.llm_hidden_size


def create_mlp_projector(
    vision_hidden_size=1024,
    llm_hidden_size=4096,
    state_dict=None,
):
    """
    Factory function to create MLP projector

    Args:
        vision_hidden_size: Input dimension (CLIP output = 1024)
        llm_hidden_size: Output dimension (LLaMA hidden = 4096)
        state_dict: Optional pretrained weights

    Returns:
        MLPProjector
    """
    return MLPProjector(
        vision_hidden_size=vision_hidden_size,
        llm_hidden_size=llm_hidden_size,
        state_dict=state_dict,
    )


# Constants matching LLaVA
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


class StandardInterleaver(nn.Module):
    """
    Token Interleaver - PyTorch Reference Implementation

    Interleaves text embeddings with image features at <image> placeholder
    positions, matching LLaVA's prepare_inputs_labels_for_multimodal().

    LLaVA 1.5 uses a single IMAGE_TOKEN_INDEX (-200) as a placeholder for
    each image. This token is replaced by the projected image features
    (576 tokens for CLIP ViT-L/14-336), making the output sequence longer
    than the input.

    Example:
        Input IDs:  [<image>, "USER", ":", "What", "'s", "in", "this", "image", "?"]
        Text embeds: [E_img_dummy, E_USER, E_colon, E_what, ...]
        Image feats: [F0, F1, ..., F575]  (576 features)

        Output:  [F0, F1, ..., F575, E_USER, E_colon, E_what, ...]
        (image placeholder removed, image features inserted)

    Args:
        image_token_index: Token ID marking image positions (default: -200)
        ignore_index: Label value for ignored positions (default: -100)

    Reference:
        - LLaVA/llava/model/llava_arch.py:prepare_inputs_labels_for_multimodal
        - LLaVA/llava/constants.py (IMAGE_TOKEN_INDEX, IGNORE_INDEX)
    """

    def __init__(self, image_token_index=IMAGE_TOKEN_INDEX, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.image_token_index = image_token_index
        self.ignore_index = ignore_index

    @torch.no_grad()
    def forward(self, input_ids, text_embeds, image_features, labels=None, attention_mask=None):
        """
        Interleave text embeddings with image features.

        Args:
            input_ids: [B, seq_len] token IDs with IMAGE_TOKEN_INDEX at image positions
            text_embeds: [B, seq_len, hidden_size] embeddings of all tokens
                         (image positions can have any value, they are removed)
            image_features: [B, num_patches, hidden_size] or list of [num_patches_i, hidden_size]
                           projected vision features per image
            labels: Optional [B, seq_len] labels (image positions get IGNORE_INDEX)
            attention_mask: Optional [B, seq_len] mask to strip padding before interleaving

        Returns:
            dict with:
                input_embeds: [B, new_seq_len, hidden_size] interleaved embeddings
                attention_mask: [B, new_seq_len] mask (1 for real tokens, 0 for padding)
                position_ids: [B, new_seq_len] position IDs
                labels: [B, new_seq_len] or None (IGNORE_INDEX for image positions)
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
            input_ids_list = stripped_input_ids
            text_embeds_list = stripped_text_embeds
            labels_list = stripped_labels
        else:
            input_ids_list = [input_ids[i] for i in range(batch_size)]
            text_embeds_list = [text_embeds[i] for i in range(batch_size)]
            labels_list = [labels[i] for i in range(batch_size)] if labels is not None else None

        label_dtype = labels_list[0].dtype if labels_list is not None else torch.long

        # Handle single image_features tensor [B, num_patches, hidden_size]
        # Convert to list of per-batch features
        if isinstance(image_features, torch.Tensor) and image_features.dim() == 3:
            image_features_list = [image_features[i] for i in range(batch_size)]
        elif isinstance(image_features, list):
            image_features_list = image_features
        else:
            image_features_list = [image_features]

        new_input_embeds = []
        new_labels = []

        for batch_idx in range(batch_size):
            cur_input_ids = input_ids_list[batch_idx]  # [seq_len]
            cur_text_embeds = text_embeds_list[batch_idx]  # [seq_len, hidden_size]

            # Find image token positions
            image_token_indices = torch.where(cur_input_ids == self.image_token_index)[0]
            num_images = len(image_token_indices)

            if num_images == 0:
                # No image tokens — just use text embeds as-is
                new_input_embeds.append(cur_text_embeds)
                if labels_list is not None:
                    new_labels.append(labels_list[batch_idx])
                continue

            # Build split points: [-1, img_pos_0, img_pos_1, ..., seq_len]
            split_points = [-1] + image_token_indices.tolist() + [cur_input_ids.shape[0]]

            # Extract text segments between image tokens (excluding the image tokens themselves)
            text_segments = []
            for i in range(len(split_points) - 1):
                start = split_points[i] + 1  # +1 to skip previous image token (or -1+1=0)
                end = split_points[i + 1]
                if end > start:
                    text_segments.append(cur_text_embeds[start:end])
                else:
                    text_segments.append(cur_text_embeds[0:0])  # empty segment

            # Interleave: text[0] + image[0] + text[1] + image[1] + ... + text[N]
            cur_new_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                # Add text segment
                cur_new_embeds.append(text_segments[i])
                if labels_list is not None:
                    seg_start = split_points[i] + 1
                    seg_end = split_points[i + 1]
                    if seg_end > seg_start:
                        cur_new_labels.append(labels_list[batch_idx][seg_start:seg_end])
                    else:
                        cur_new_labels.append(torch.empty(0, dtype=label_dtype))

                # Add image features (if not last segment)
                if i < num_images:
                    img_idx = i  # Assume images are in order
                    if img_idx < len(image_features_list):
                        img_feat = image_features_list[img_idx]
                    else:
                        img_feat = image_features_list[0]  # fallback
                    cur_new_embeds.append(img_feat)
                    if labels_list is not None:
                        # Image positions get IGNORE_INDEX
                        cur_new_labels.append(
                            torch.full(
                                (img_feat.shape[0],),
                                self.ignore_index,
                                dtype=label_dtype,
                            )
                        )

            # Concatenate all segments
            cur_new_embeds = torch.cat(cur_new_embeds, dim=0)  # [new_seq_len, hidden_size]
            new_input_embeds.append(cur_new_embeds)

            if labels_list is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # Pad sequences to max length in batch
        max_len = max(x.shape[0] for x in new_input_embeds)
        padded_embeds = torch.zeros(batch_size, max_len, hidden_size, dtype=new_input_embeds[0].dtype)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

        if labels_list is not None:
            padded_labels = torch.full((batch_size, max_len), self.ignore_index, dtype=label_dtype)
        else:
            padded_labels = None

        for i, (cur_embed, cur_len) in enumerate(zip(new_input_embeds, [x.shape[0] for x in new_input_embeds])):
            # Right-padding (LLaVA default)
            padded_embeds[i, :cur_len] = cur_embed
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long)
            if padded_labels is not None and cur_len > 0:
                padded_labels[i, :cur_len] = new_labels[i]

        return {
            "input_embeds": padded_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": padded_labels,
        }


def create_standard_interleaver():
    """Factory function to create StandardInterleaver"""
    return StandardInterleaver()


# Example usage
if __name__ == "__main__":
    logger.info("Testing CLIP vision encoder reference implementation")

    # Create encoder
    encoder = create_clip_encoder(device="cpu")

    # Create dummy input [B=1, C=3, H=336, W=336]
    images = torch.randn(1, 3, 336, 336)

    # Forward pass
    features = encoder(images)

    logger.info(f"Input shape: {images.shape}")
    logger.info(f"Output shape: {features.shape}")
    logger.info(f"Expected: [1, 576, 1024]")

    assert features.shape == (1, 576, 1024), f"Expected [1, 576, 1024], got {features.shape}"
    logger.info("✅ CLIP reference validated")

    # Test MLP projector
    logger.info("\nTesting MLP projector reference implementation")

    projector = create_mlp_projector(vision_hidden_size=1024, llm_hidden_size=4096)
    projected = projector(features)

    logger.info(f"Input shape: {features.shape}")
    logger.info(f"Output shape: {projected.shape}")
    logger.info(f"Expected: [1, 576, 4096]")

    assert projected.shape == (1, 576, 4096), f"Expected [1, 576, 4096], got {projected.shape}"
    logger.info("✅ MLP projector reference validated")

    # Test token interleaver
    logger.info("\nTesting token interleaver reference implementation")

    interleaver = create_standard_interleaver()

    # Simulate: [<image>, "USER", ":", "What", "?"]  (5 tokens, 1 image placeholder)
    input_ids = torch.tensor([[-200, 100, 101, 102, 103]])
    text_embeds = torch.randn(1, 5, 4096)  # Embeddings for all 5 tokens
    image_features = torch.randn(1, 576, 4096)  # 576 image features

    result = interleaver(input_ids, text_embeds, image_features)
    logger.info(f"Input seq len: {input_ids.shape[1]}")
    logger.info(f"Output seq len: {result['input_embeds'].shape[1]}")
    logger.info(f"Expected: {5 - 1 + 576} = 580")

    assert result["input_embeds"].shape == (1, 580, 4096)
    logger.info("✅ Token interleaver reference validated")
