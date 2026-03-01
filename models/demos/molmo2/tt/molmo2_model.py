# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Molmo2-8B Full Model Implementation.

Combines the Vision Backbone (ViT + Adapter) and Text Model (LM) into a
unified multimodal model for visual question answering and image captioning.

Architecture:
    1. Image Processing: Preprocess images and compute pooled_patches_idx
    2. Vision Backbone: ViT encoder -> multi-scale features -> pooling -> projection
    3. Embedding Fusion: Insert visual embeddings into text sequence
    4. Text Model: Decoder-only transformer for autoregressive generation
"""

from typing import Dict, List, Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.molmo2.tt.text_model import TextModel
from models.demos.molmo2.tt.vision_backbone import VisionBackbone


class Molmo2Model(LightweightModule):
    """
    Full Molmo2-8B multimodal model.

    Combines vision encoder, adapter, and language model for
    image-conditioned text generation.
    """

    def __init__(
        self,
        mesh_device,
        state_dict: Dict[str, torch.Tensor],
        # Vision config
        vit_num_layers: int = 25,
        vit_hidden_dim: int = 1152,
        vit_intermediate_dim: int = 4304,
        vit_num_heads: int = 16,
        vit_head_dim: int = 72,
        patch_size: int = 14,
        image_size: int = 378,
        feature_layers: Tuple[int, int] = (24, 18),  # HF order: [-3, -9]
        # Adapter config
        adapter_hidden_dim: int = 1152,
        adapter_intermediate_dim: int = 12288,
        adapter_num_heads: int = 16,
        adapter_head_dim: int = 72,
        # Text config
        text_num_layers: int = 36,
        text_hidden_dim: int = 4096,
        text_intermediate_dim: int = 12288,
        text_num_heads: int = 32,
        text_num_kv_heads: int = 8,
        text_head_dim: int = 128,
        vocab_size: int = 152064,
        max_seq_len: int = 8192,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-5,
        # Common config
        layer_norm_eps: float = 1e-6,
        weight_cache_path=None,
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize Molmo2Model.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Complete model state dict
            vit_*: Vision transformer configuration
            adapter_*: Vision adapter configuration
            text_*: Language model configuration
            weight_cache_path: Path to cache weights
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.text_hidden_dim = text_hidden_dim
        self.dtype = dtype

        # Special token IDs
        self.image_patch_id = 151938
        self.bos_token_id = 151643
        self.eos_token_id = 151645

        # Vision backbone (ViT + Adapter)
        self.vision_backbone = VisionBackbone(
            mesh_device=mesh_device,
            state_dict=state_dict,
            vit_num_layers=vit_num_layers,
            vit_hidden_dim=vit_hidden_dim,
            vit_intermediate_dim=vit_intermediate_dim,
            vit_num_heads=vit_num_heads,
            vit_head_dim=vit_head_dim,
            patch_size=patch_size,
            image_size=image_size,
            feature_layers=feature_layers,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_intermediate_dim=adapter_intermediate_dim,
            adapter_num_heads=adapter_num_heads,
            adapter_head_dim=adapter_head_dim,
            output_dim=text_hidden_dim,
            layer_norm_eps=layer_norm_eps,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

        # Text model (Language Model)
        self.text_model = TextModel(
            mesh_device=mesh_device,
            state_dict=state_dict,
            num_layers=text_num_layers,
            hidden_dim=text_hidden_dim,
            intermediate_dim=text_intermediate_dim,
            num_heads=text_num_heads,
            num_kv_heads=text_num_kv_heads,
            head_dim=text_head_dim,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

    def embed_image(
        self,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process image through vision backbone.

        Args:
            pixel_values: Preprocessed image tensor [B, C, H, W] or [B, T, C, H, W]
            pooled_patches_idx: Patch indices for pooling [B, N_out, K_pool]

        Returns:
            Visual embeddings [num_valid_tokens, hidden_dim]
        """
        # This is a simplified version - full implementation would handle
        # patch embedding and positional encoding here

        # For now, assume pixel_values are already embedded patches
        # In production, this would call patch_embed + pos_embed

        # Forward through vision backbone
        visual_embeddings = self.vision_backbone(
            images_embedded=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
        )

        return visual_embeddings

    def prepare_inputs_for_multimodal(
        self,
        input_ids: torch.Tensor,
        visual_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare input embeddings by fusing text and visual tokens.

        Replaces image_patch_id tokens with corresponding visual embeddings.

        Args:
            input_ids: Token IDs with image_patch_id placeholders [batch, seq_len]
            visual_embeddings: Visual embeddings from vision backbone [num_visual_tokens, hidden_dim]

        Returns:
            Fused embeddings [seq_len, hidden_dim]
        """
        batch_size, seq_len = input_ids.shape
        assert batch_size == 1, "Only batch_size=1 is currently supported"

        # Check if mesh device
        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        # Get text embeddings
        input_ids_ttnn = ttnn.from_torch(
            input_ids,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        text_embeddings = self.text_model.embed_tokens(input_ids_ttnn)
        # Shape: [1, 1, seq_len, hidden_dim] -> [seq_len, hidden_dim]
        if is_mesh_device:
            text_embeddings_torch = (
                ttnn.to_torch(text_embeddings, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0]
                .squeeze(0)
                .squeeze(0)
            )
        else:
            text_embeddings_torch = ttnn.to_torch(text_embeddings).squeeze(0).squeeze(0)

        # Find image patch positions (flatten for batch_size=1)
        image_positions = (input_ids[0] == self.image_patch_id).nonzero(as_tuple=True)[0]

        # Add visual embeddings to image patch token positions
        # This matches HuggingFace: x[is_image_patch] += image_features
        # The image_patch_id tokens have learned embeddings that are added to visual features
        if len(image_positions) > 0:
            num_visual_tokens = visual_embeddings.shape[0]
            assert (
                len(image_positions) == num_visual_tokens
            ), f"Mismatch: {len(image_positions)} placeholders vs {num_visual_tokens} visual tokens"

            for i, seq_idx in enumerate(image_positions):
                text_embeddings_torch[seq_idx] += visual_embeddings[i]

        return text_embeddings_torch

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        start_pos: int = 0,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """
        Forward pass through the full Molmo2 model.

        Args:
            input_ids: Token IDs [batch, seq_len]
            pixel_values: Optional image tensor for visual input
            pooled_patches_idx: Optional patch indices for pooling
            attention_mask: Optional attention mask
            kv_caches: Optional KV cache for incremental decoding
            start_pos: Starting position for KV cache

        Returns:
            Tuple of (logits, new_kv_caches)
        """
        # Process images if provided
        if pixel_values is not None and pooled_patches_idx is not None:
            visual_embeddings = self.embed_image(pixel_values, pooled_patches_idx)
            hidden_states = self.prepare_inputs_for_multimodal(input_ids, visual_embeddings)
        else:
            # Text-only forward
            is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

            input_ids_ttnn = ttnn.from_torch(
                input_ids,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
            hidden_states = self.text_model.embed_tokens(input_ids_ttnn)

            # Convert to torch for shape manipulation
            if is_mesh_device:
                # Take first device's output (they're all replicated)
                hidden_states = ttnn.to_torch(
                    hidden_states, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                )[0]
            else:
                hidden_states = ttnn.to_torch(hidden_states)
            hidden_states = hidden_states.squeeze(0).squeeze(0)

        # Convert to TTNN
        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        hidden_states_ttnn = ttnn.from_torch(
            hidden_states.unsqueeze(0).unsqueeze(0),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # Forward through text model (handles both prefill and decode via KV cache)
        logits, new_kv_caches = self.text_model(
            hidden_states=hidden_states_ttnn,
            start_pos=start_pos,
            attn_mask=None,
            kv_caches=kv_caches,
        )

        return logits, new_kv_caches

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Initial token IDs [batch, seq_len]
            pixel_values: Optional image input
            pooled_patches_idx: Optional patch indices
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()

        # Initial forward pass (prefill)
        logits, kv_caches = self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
        )

        # Get next token
        logits_torch = ttnn.to_torch(logits).squeeze()
        # Handle different output shapes
        if logits_torch.dim() == 2:
            # Shape: [seq_len, vocab_size] - take last position
            next_token_logits = logits_torch[-1:, :]
        else:
            # Shape: [batch, seq_len, vocab_size]
            next_token_logits = logits_torch[:, -1, :]

        for _ in range(max_new_tokens):
            # Sample or greedy decode
            if do_sample:
                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if (next_token == self.eos_token_id).all():
                break

            # Decode step (single token)
            logits, kv_caches = self.forward(
                input_ids=next_token,
                kv_caches=kv_caches,
                start_pos=generated_ids.shape[1] - 1,
            )

            logits_torch = ttnn.to_torch(logits).squeeze(0).squeeze(0)
            next_token_logits = logits_torch[:, -1, :]

        return generated_ids
