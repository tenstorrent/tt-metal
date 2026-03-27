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
from loguru import logger

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

        from loguru import logger

        self.mesh_device = mesh_device
        self.text_hidden_dim = text_hidden_dim
        self.dtype = dtype

        # Special token IDs
        self.image_patch_id = 151938
        self.bos_token_id = 151643
        self.eos_token_id = 151645

        logger.info("Creating VisionBackbone...")
        # Vision backbone (ViT + Adapter)
        # Use bfloat16 for vision backbone to avoid numerical precision issues
        # that cause overflow with bfloat8_b (visual embeddings with values ~37000-61000)
        vision_dtype = ttnn.bfloat16  # Higher precision for vision computations
        logger.info(f"Using vision_dtype={vision_dtype} (model dtype={dtype})")
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
            dtype=vision_dtype,
        )
        logger.info("VisionBackbone created")

        logger.info("Creating TextModel...")
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
    ) -> Tuple[ttnn.Tensor, torch.Tensor]:
        """
        Process image through vision backbone (fully on TTNN).

        Args:
            pixel_values: Preprocessed image tensor [B, C, H, W]
            pooled_patches_idx: Patch indices for pooling [B, N_out, K_pool]

        Returns:
            Tuple of:
              - visual_embeddings: [1, 1, N_out, hidden_dim] on device (unfiltered)
              - valid_token: [B, N_out] bool tensor on CPU
        """
        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        batch_size = pooled_patches_idx.shape[0]
        n_out = pooled_patches_idx.shape[1]
        k_pool = pooled_patches_idx.shape[2]

        # Patch embedding on TTNN: CPU unfold only, linear+pos_embed on device
        vit = self.vision_backbone.image_vit
        patch_features = vit.patch_size * vit.patch_size * 3  # 14*14*3 = 588

        # Detect input format:
        # - Pre-unfolded from vLLM: [num_crops, num_patches, 588] - 3D with last dim == 588
        # - Raw image format: [B, C, H, W] - 4D
        if pixel_values.dim() == 3 and pixel_values.shape[-1] == patch_features:
            # Pre-unfolded patch format from vLLM [num_crops, num_patches, 588]
            embedded_ttnn = vit.patch_embed_from_patches_ttnn(pixel_values)
        else:
            embedded_ttnn = vit.patch_embed_ttnn(pixel_values)

        # Prepare gather indices and masks (CPU, fast)
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, dim=-1)  # [B, N_out] bool
        clipped_idx = torch.clip(pooled_patches_idx, min=0)
        flat_idx = clipped_idx.reshape(1, -1).to(torch.int32)
        valid_mask = valid.reshape(1, 1, -1, 1).float()

        idx_ttnn = ttnn.from_torch(
            flat_idx,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        valid_mask_ttnn = ttnn.from_torch(
            valid_mask,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        valid_token_ttnn = ttnn.from_torch(
            valid_token.flatten().float(),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        visual_embeddings = self.vision_backbone.forward_ttnn(
            images_embedded=embedded_ttnn,
            pooled_patches_idx_ttnn=idx_ttnn,
            valid_mask_ttnn=valid_mask_ttnn,
            valid_token_ttnn=valid_token_ttnn,
            n_out=n_out,
            k_pool=k_pool,
            batch_size=batch_size,
        )

        # Debug: check visual embeddings values
        from loguru import logger

        try:
            is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
            if is_mesh_device:
                mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                vis_torch = ttnn.to_torch(visual_embeddings, mesh_composer=mesh_composer)[0]
            else:
                vis_torch = ttnn.to_torch(visual_embeddings)
            logger.info(f"embed_image DEBUG: visual_embeddings shape={vis_torch.shape}")
            logger.info(
                f"embed_image DEBUG: visual_embeddings stats: min={vis_torch.min().item():.4f}, max={vis_torch.max().item():.4f}, mean={vis_torch.mean().item():.4f}, std={vis_torch.std().item():.4f}"
            )
            logger.info(f"embed_image DEBUG: visual_embeddings first 5 values: {vis_torch.flatten()[:5].tolist()}")
        except Exception as e:
            logger.warning(f"embed_image DEBUG: Failed to inspect visual_embeddings: {e}")

        ttnn.deallocate(embedded_ttnn)
        ttnn.deallocate(idx_ttnn)
        ttnn.deallocate(valid_mask_ttnn)
        ttnn.deallocate(valid_token_ttnn)

        return visual_embeddings, valid_token

    def prepare_inputs_for_multimodal(
        self,
        input_ids: torch.Tensor,
        visual_embeddings_ttnn: ttnn.Tensor,
        valid_token: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Fuse text and visual embeddings on device using selector matmul.

        No CPU roundtrip: text embed + selector matmul + add all on device.

        Args:
            input_ids: Token IDs with image_patch_id placeholders [batch, seq_len]
            visual_embeddings_ttnn: Visual embeddings [1, 1, N_out, hidden_dim] on device
            valid_token: [B, N_out] bool mask for which visual tokens are valid (CPU)

        Returns:
            Fused embeddings [1, 1, seq_len, hidden_dim] on device
        """
        batch_size, seq_len = input_ids.shape
        assert batch_size == 1, "Only batch_size=1 is currently supported"
        hidden_dim = self.text_hidden_dim

        # DEBUG: Track request count
        if not hasattr(self, "_multimodal_request_count"):
            self._multimodal_request_count = 0
        self._multimodal_request_count += 1
        logger.info(f"prepare_inputs_for_multimodal: REQUEST #{self._multimodal_request_count}")
        logger.info(f"  input_ids.shape={input_ids.shape}, seq_len={seq_len}")
        logger.info(f"  visual_embeddings_ttnn shape={visual_embeddings_ttnn.shape}")
        logger.info(f"  valid_token.shape={valid_token.shape}, sum={valid_token.sum().item()}")

        # FIX: vLLM places image tokens BEFORE the chat template (wrong position)
        # We need to reorder: move image tokens to after <|im_start|>user\n
        im_start_token = 151644  # <|im_start|>
        user_token = 872  # user
        newline_token = 198  # \n

        # Check if image tokens are at start (wrong) and chat template comes after (also wrong)
        image_positions = (input_ids[0] == self.image_patch_id).nonzero(as_tuple=True)[0]
        if len(image_positions) > 0 and image_positions[0] == 0:
            # Image tokens at start - find where chat template is
            im_start_pos = (input_ids[0] == im_start_token).nonzero(as_tuple=True)[0]
            if len(im_start_pos) > 0:
                chat_start = im_start_pos[0].item()
                # Check if this is the user message (followed by user token and newline)
                if (
                    chat_start + 2 < seq_len
                    and input_ids[0, chat_start + 1] == user_token
                    and input_ids[0, chat_start + 2] == newline_token
                ):
                    # Find how many image tokens are at the start
                    num_leading_images = 0
                    for i in range(chat_start):
                        if input_ids[0, i] == self.image_patch_id:
                            num_leading_images += 1
                        else:
                            break  # Stop at first non-image token

                    if num_leading_images > 0 and num_leading_images == chat_start:
                        # All tokens before chat template are image tokens - reorder!
                        logger.info(
                            f"  FIX: Reordering {num_leading_images} image tokens from start to after chat template"
                        )
                        # New order: [<|im_start|>user\n, IMAGE_TOKENS, rest...]
                        image_tokens = input_ids[0, :num_leading_images]
                        chat_and_rest = input_ids[0, num_leading_images:]
                        # Insert image tokens after <|im_start|>user\n (3 tokens)
                        new_input_ids = torch.cat(
                            [
                                chat_and_rest[:3],  # <|im_start|>user\n
                                image_tokens,  # image tokens
                                chat_and_rest[3:],  # rest of the message
                            ]
                        ).unsqueeze(0)
                        input_ids = new_input_ids
                        logger.info(f"  FIX: Reordered input_ids, new shape={input_ids.shape}")
                        logger.info(f"  FIX: New input_ids first 20: {input_ids[0][:20].tolist()}")

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        # Get text embeddings on device
        input_ids_ttnn = ttnn.from_torch(
            input_ids,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        text_embeddings_ttnn = self.text_model.embed_tokens(input_ids_ttnn)
        ttnn.deallocate(input_ids_ttnn)

        # Filter valid visual embeddings on device via ttnn.embedding (gather)
        valid_indices = valid_token.flatten().nonzero(as_tuple=True)[0].to(torch.int32)
        num_valid = len(valid_indices)

        if num_valid == 0:
            return text_embeddings_ttnn

        valid_indices_ttnn = ttnn.from_torch(
            valid_indices.unsqueeze(0),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        # visual_embeddings_ttnn: [1, 1, N_out, hidden_dim] -> [1, N_out, hidden_dim] for gather
        visual_for_gather = ttnn.reshape(visual_embeddings_ttnn, [1, -1, hidden_dim])
        valid_visual_ttnn = ttnn.embedding(valid_indices_ttnn, visual_for_gather)
        ttnn.deallocate(valid_indices_ttnn)
        # NOTE: Do NOT deallocate visual_for_gather - reshape may return a view

        # valid_visual_ttnn: [1, num_valid, hidden_dim] -> [1, 1, num_valid, hidden_dim]
        valid_visual_ttnn = ttnn.reshape(valid_visual_ttnn, [1, 1, num_valid, hidden_dim])
        # NOTE: Do NOT deallocate original - reshape may return a view

        # Build selector matrix on CPU (fast, input_ids-sized sparse matrix)
        image_positions = (input_ids[0] == self.image_patch_id).nonzero(as_tuple=True)[0]
        logger.info(
            f"prepare_inputs_for_multimodal: image_patch_id={self.image_patch_id}, found {len(image_positions)} image positions, num_valid={num_valid}"
        )
        logger.info(f"  input_ids first 20: {input_ids[0][:20].tolist()}")
        logger.info(f"  input_ids LAST 20: {input_ids[0][-20:].tolist()}")
        logger.info(f"  input_ids ALL: {input_ids[0].tolist()}")
        logger.info(f"  input_ids unique tokens: {input_ids[0].unique().tolist()[:20]}...")
        if len(image_positions) != num_valid:
            logger.warning(
                f"  MISMATCH! len(image_positions)={len(image_positions)} != num_valid={num_valid}, returning text-only embeddings!"
            )
            ttnn.deallocate(valid_visual_ttnn)
            return text_embeddings_ttnn
        logger.info(f"  image_positions (first 10): {image_positions[:10].tolist()}")

        selector = torch.zeros(seq_len, num_valid, dtype=torch.bfloat16)
        for i, pos in enumerate(image_positions):
            selector[pos, i] = 1.0

        # DEBUG: Synchronize device before creating new tensor
        logger.info(f"  DEBUG: About to create selector_ttnn, syncing device first...")
        try:
            ttnn.synchronize_device(self.mesh_device)
            logger.info(f"  DEBUG: Device sync completed successfully")
        except Exception as e:
            logger.error(f"  DEBUG: Device sync FAILED: {e}")
            raise

        selector_ttnn = ttnn.from_torch(
            selector.unsqueeze(0).unsqueeze(0),  # [1, 1, seq_len, num_valid]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # visual_contribution = selector @ valid_visual: [1, 1, seq_len, hidden_dim]
        visual_contribution = ttnn.matmul(selector_ttnn, valid_visual_ttnn)
        ttnn.deallocate(selector_ttnn)
        ttnn.deallocate(valid_visual_ttnn)

        # Debug: check visual_contribution values before fusion
        try:
            is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
            if is_mesh_device:
                mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                vis_contrib = ttnn.to_torch(visual_contribution, mesh_composer=mesh_composer)[0]
                text_emb = ttnn.to_torch(text_embeddings_ttnn, mesh_composer=mesh_composer)[0]
            else:
                vis_contrib = ttnn.to_torch(visual_contribution)
                text_emb = ttnn.to_torch(text_embeddings_ttnn)
            logger.info(f"prepare_inputs_for_multimodal DEBUG: visual_contribution shape={vis_contrib.shape}")
            logger.info(
                f"prepare_inputs_for_multimodal DEBUG: visual_contribution stats: min={vis_contrib.min().item():.4f}, max={vis_contrib.max().item():.4f}, mean={vis_contrib.mean().item():.4f}"
            )
            logger.info(f"prepare_inputs_for_multimodal DEBUG: text_embeddings shape={text_emb.shape}")
            logger.info(
                f"prepare_inputs_for_multimodal DEBUG: text_embeddings stats: min={text_emb.min().item():.4f}, max={text_emb.max().item():.4f}, mean={text_emb.mean().item():.4f}"
            )
            # Check visual contrib at image positions
            logger.info(
                f"prepare_inputs_for_multimodal DEBUG: visual_contrib at pos 0: first 5 = {vis_contrib[0,0,0,:5].tolist()}"
            )
            logger.info(
                f"prepare_inputs_for_multimodal DEBUG: text_emb at pos 0: first 5 = {text_emb[0,0,0,:5].tolist()}"
            )
        except Exception as e:
            logger.warning(f"prepare_inputs_for_multimodal DEBUG: Failed to inspect: {e}")

        # Fuse: ADD visual to text at image positions (matching HuggingFace reference)
        # Reference: x.view(-1, x.shape[-1])[is_image_patch] += image_features
        fused_ttnn = ttnn.add(text_embeddings_ttnn, visual_contribution)
        ttnn.deallocate(text_embeddings_ttnn)
        ttnn.deallocate(visual_contribution)

        return fused_ttnn

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        start_pos: int = 0,
        page_table: Optional[ttnn.Tensor] = None,
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
            page_table: Optional page table for paged attention (vLLM)

        Returns:
            Tuple of (logits, new_kv_caches)
        """
        # Process images if provided -- fully on TTNN, no CPU roundtrip
        if pixel_values is not None and pooled_patches_idx is not None:
            visual_embeddings_ttnn, valid_token = self.embed_image(pixel_values, pooled_patches_idx)
            hidden_states_ttnn = self.prepare_inputs_for_multimodal(input_ids, visual_embeddings_ttnn, valid_token)
            ttnn.deallocate(visual_embeddings_ttnn)
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
            hidden_states_ttnn = self.text_model.embed_tokens(input_ids_ttnn)
            ttnn.deallocate(input_ids_ttnn)

        # Forward through text model (handles both prefill and decode via KV cache)
        logits, new_kv_caches = self.text_model(
            hidden_states=hidden_states_ttnn,
            start_pos=start_pos,
            attn_mask=None,
            kv_caches=kv_caches,
            page_table=page_table,
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
