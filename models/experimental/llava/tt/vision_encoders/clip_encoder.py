# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CLIP Vision Encoder - TTNN Implementation

Implements CLIP ViT-L/14-336 for Tenstorrent hardware.

Architecture:
    - Input: [B, 3, 336, 336]
    - Patch embedding: Conv 14×14 → [B, 577, 1024] (576 patches + 1 CLS)
    - Position embeddings: Learnable [1, 577, 1024]
    - Transformer blocks: 24 layers
        - Multi-head self-attention (16 heads, head_dim=64)
        - LayerNorm
        - MLP (1024 → 4096 → 1024, GELU activation)
    - Output: [B, 576, 1024] (drop CLS token for LLaVA)

Sharding Strategy (n300 single-device):
    - Tensors replicated across both chips
    - Attention: HEIGHT_SHARDED across sequence dimension
    - MLP: BLOCK_SHARDED for parallel FFN computation

Reference:
    - models/demos/vision/classification/vit/wormhole/tt/ttnn_optimized_sharded_vit_wh.py
    - models/demos/qwen25_vl/tt/model.py (VisionTransformer)
    - transformers.CLIPVisionModel
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn
from models.demos.llava.tt.vision_encoders.base_encoder import BaseVisionEncoder


class CLIPVisionTransformerBlock(torch.nn.Module):
    """
    Single CLIP Vision Transformer Block

    Components:
        1. LayerNorm
        2. Multi-head Self-Attention
        3. Residual connection
        4. LayerNorm
        5. MLP (Linear → GELU → Linear)
        6. Residual connection

    Args:
        args: VisionModelArgs with device config
        state_dict: Pre-trained weights
        layer_num: Block index (0-23)
        dtype: ttnn.bfloat16
    """

    def __init__(self, args, state_dict, layer_num, dtype):
        super().__init__()

        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype

        # Extract config
        self.hidden_size = 1024
        self.num_heads = 16
        self.head_dim = self.hidden_size // self.num_heads  # 64
        self.intermediate_size = 4096

        # Weight prefix in state_dict
        prefix = f"encoder.layers.{layer_num}"

        # Load and preprocess weights
        self._load_weights(state_dict, prefix)

    def _load_weights(self, state_dict, prefix):
        """Load and preprocess weights from PyTorch state_dict"""

        # Layer norm 1 (before attention)
        self.ln1_weight = ttnn.from_torch(
            state_dict[f"{prefix}.layer_norm1.weight"],
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        self.ln1_bias = ttnn.from_torch(
            state_dict[f"{prefix}.layer_norm1.bias"],
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )

        # QKV projection weights (combined)
        qkv_weight = torch.cat(
            [
                state_dict[f"{prefix}.self_attn.q_proj.weight"],
                state_dict[f"{prefix}.self_attn.k_proj.weight"],
                state_dict[f"{prefix}.self_attn.v_proj.weight"],
            ],
            dim=0,
        )  # [3072, 1024]

        qkv_bias = torch.cat(
            [
                state_dict[f"{prefix}.self_attn.q_proj.bias"],
                state_dict[f"{prefix}.self_attn.k_proj.bias"],
                state_dict[f"{prefix}.self_attn.v_proj.bias"],
            ],
            dim=0,
        )  # [3072]

        self.qkv_weight = preprocess_linear_weight(
            qkv_weight, dtype=self.dtype, weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device)
        )
        self.qkv_weight = ttnn.to_device(self.qkv_weight, self.args.mesh_device)

        self.qkv_bias = preprocess_linear_bias(
            qkv_bias, dtype=self.dtype, weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device)
        )
        self.qkv_bias = ttnn.to_device(self.qkv_bias, self.args.mesh_device)

        # Output projection (after attention)
        self.attn_out_weight = preprocess_linear_weight(
            state_dict[f"{prefix}.self_attn.out_proj.weight"],
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        self.attn_out_weight = ttnn.to_device(self.attn_out_weight, self.args.mesh_device)

        self.attn_out_bias = preprocess_linear_bias(
            state_dict[f"{prefix}.self_attn.out_proj.bias"],
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        self.attn_out_bias = ttnn.to_device(self.attn_out_bias, self.args.mesh_device)

        # Layer norm 2 (before MLP)
        self.ln2_weight = ttnn.from_torch(
            state_dict[f"{prefix}.layer_norm2.weight"],
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        self.ln2_bias = ttnn.from_torch(
            state_dict[f"{prefix}.layer_norm2.bias"],
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )

        # MLP weights
        self.mlp_fc1_weight = preprocess_linear_weight(
            state_dict[f"{prefix}.mlp.fc1.weight"],
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        self.mlp_fc1_weight = ttnn.to_device(self.mlp_fc1_weight, self.args.mesh_device)

        self.mlp_fc1_bias = preprocess_linear_bias(
            state_dict[f"{prefix}.mlp.fc1.bias"],
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        self.mlp_fc1_bias = ttnn.to_device(self.mlp_fc1_bias, self.args.mesh_device)

        self.mlp_fc2_weight = preprocess_linear_weight(
            state_dict[f"{prefix}.mlp.fc2.weight"],
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        self.mlp_fc2_weight = ttnn.to_device(self.mlp_fc2_weight, self.args.mesh_device)

        self.mlp_fc2_bias = preprocess_linear_bias(
            state_dict[f"{prefix}.mlp.fc2.bias"],
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        self.mlp_fc2_bias = ttnn.to_device(self.mlp_fc2_bias, self.args.mesh_device)

    def forward(self, x, attention_mask=None):
        """
        Forward pass through transformer block

        Args:
            x: Input tensor [B, seq_len, hidden_size]
            attention_mask: Optional attention mask

        Returns:
            x: Output tensor [B, seq_len, hidden_size]
        """
        # Store residual for first residual connection
        residual = x

        # Layer norm 1 (pre-attention)
        x = ttnn.layer_norm(
            x,
            weight=self.ln1_weight,
            bias=self.ln1_bias,
            epsilon=1e-5,
        )
        ln1_output = x

        # Multi-head self-attention with HEIGHT_SHARDED memory
        # QKV projection: [B, seq_len, hidden_size] → [B, seq_len, 3*hidden_size]
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=self.qkv_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # Split into Q, K, V and split heads
        # [B, seq_len, 3*hidden_size] → Q, K, V each [B, num_heads, seq_len, head_dim]
        # Note: Using L1_MEMORY_CONFIG instead of HEIGHT_SHARDED because seq_len=577
        # doesn't fit sharding requirements (19 tiles > 16 cores)
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            num_heads=self.num_heads,
        )
        ttnn.deallocate(qkv)

        # Scaled dot-product attention: Q @ K^T
        # [B, num_heads, seq_len, head_dim] @ [B, num_heads, head_dim, seq_len]
        # → [B, num_heads, seq_len, seq_len]
        attention_scores = ttnn.matmul(
            query,
            key,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(query)
        ttnn.deallocate(key)

        # Scale by 1/sqrt(head_dim)
        scale = 1.0 / (self.head_dim**0.5)
        attention_scores = ttnn.mul(attention_scores, scale)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = ttnn.add(attention_scores, attention_mask)

        # Softmax over last dimension
        attention_probs = ttnn.softmax(attention_scores, dim=-1)
        ttnn.deallocate(attention_scores)

        # Apply attention to values: attention_probs @ V
        # [B, num_heads, seq_len, seq_len] @ [B, num_heads, seq_len, head_dim]
        # → [B, num_heads, seq_len, head_dim]
        context_layer = ttnn.matmul(
            attention_probs,
            value,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attention_probs)
        ttnn.deallocate(value)

        # Concatenate heads: [B, num_heads, seq_len, head_dim] → [B, seq_len, hidden_size]
        attention_output = ttnn.transformer.concatenate_heads(
            context_layer,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(context_layer)

        # Output projection
        x = ttnn.linear(
            attention_output,
            self.attn_out_weight,
            bias=self.attn_out_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attention_output)

        # First residual connection
        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)

        # Store residual for second residual connection
        residual = x

        # Layer norm 2 (pre-MLP)
        x = ttnn.layer_norm(
            x,
            weight=self.ln2_weight,
            bias=self.ln2_bias,
            epsilon=1e-5,
        )
        ln2_output = x

        # MLP with QuickGELU activation
        # fc1: [B, seq_len, 1024] → [B, seq_len, 4096]
        x = ttnn.linear(
            x,
            self.mlp_fc1_weight,
            bias=self.mlp_fc1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(ln2_output)

        # QuickGELU activation: x * sigmoid(1.702 * x)
        # This is CLIP-specific (not standard GELU)
        x_scaled = ttnn.mul(x, 1.702)
        x_sigmoid = ttnn.sigmoid(x_scaled)
        ttnn.deallocate(x_scaled)
        x = ttnn.mul(x, x_sigmoid)
        ttnn.deallocate(x_sigmoid)

        # fc2: [B, seq_len, 4096] → [B, seq_len, 1024]
        x = ttnn.linear(
            x,
            self.mlp_fc2_weight,
            bias=self.mlp_fc2_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Second residual connection
        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)

        return x


class CLIPVisionEncoder(BaseVisionEncoder):
    """
    CLIP Vision Encoder for Tenstorrent Hardware

    Implements openai/clip-vit-large-patch14-336 in TTNN.

    Args:
        args: VisionModelArgs with device configuration
        state_dict: Pre-trained weights from HuggingFace
        weight_cache_path: Path to cache preprocessed weights
        select_layer: Which hidden layer to extract (default: -2, LLaVA default)
            hidden_states has num_layers+1 entries (embedding output + each layer output)
            -2 = output of second-to-last transformer block (before post_layernorm)
            -1 = output of last transformer block (before post_layernorm)
    """

    def __init__(self, args, state_dict, weight_cache_path: Optional[Path] = None, select_layer: int = -2):
        super().__init__()

        self.args = args
        self.dtype = ttnn.bfloat16
        self.weight_cache_path = weight_cache_path
        self.select_layer = select_layer

        # CLIP ViT-L/14-336 config
        self._hidden_size = 1024
        self.num_layers = 24
        self.num_heads = 16
        self.image_size = 336
        self._patch_size = 14
        self._num_patches = (self.image_size // self._patch_size) ** 2  # 576

        # Calculate how many transformer blocks to run
        # hidden_states has num_layers+1 entries: [embedding, layer_0, ..., layer_{num_layers-1}]
        # select_layer=-2 → hidden_states[-2] = output of layer (num_layers-2) = layer 22
        # Formula: num_blocks_to_run = num_layers + 1 + select_layer
        self.num_blocks_to_run = self.num_layers + 1 + self.select_layer  # 23 for select_layer=-2

        logger.info(f"Initializing CLIP Vision Encoder")
        logger.info(f"  Hidden size: {self._hidden_size}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  Num heads: {self.num_heads}")
        logger.info(f"  Image size: {self.image_size}")
        logger.info(f"  Num patches: {self._num_patches}")
        logger.info(f"  Select layer: {self.select_layer} → running {self.num_blocks_to_run} transformer blocks")

        # Load embeddings
        self._load_embeddings(state_dict)

        # Create transformer blocks (only the ones we need)
        self.blocks = []
        for layer_num in range(self.num_blocks_to_run):
            block = CLIPVisionTransformerBlock(
                args=args,
                state_dict=state_dict,
                layer_num=layer_num,
                dtype=self.dtype,
            )
            self.blocks.append(block)

        logger.info(f"✅ CLIP encoder initialized with {self.num_blocks_to_run} layers")

    def _load_embeddings(self, state_dict):
        """Load patch embeddings, position embeddings, and CLS token"""

        # Patch embedding: Convert conv2d weights to linear for fold+linear approach
        # Original weight shape: [1024, 3, 14, 14] = [out_channels, in_channels, kernel_h, kernel_w]
        # Pad channels from 3 to 4 (required by ttnn.fold for tile alignment)
        # Then convert to HWIO format and reshape to 2D for ttnn.linear
        # Note: CLIP patch embedding has no bias
        # Pattern: models/demos/vision/classification/vit/wormhole/tests/test_ttnn_optimized_sharded_vit_wh.py
        patch_embed_weight = state_dict["embeddings.patch_embedding.weight"]

        # Step 1: Pad channels from 3 to 4
        # [1024, 3, 14, 14] → [1024, 4, 14, 14]
        preprocessed_weight = torch.nn.functional.pad(patch_embed_weight, (0, 0, 0, 0, 0, 1))

        # Step 2: Permute to HWIO format (height, width, in_channels, out_channels)
        # [1024, 4, 14, 14] → [14, 14, 4, 1024]
        preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))

        # Step 3: Reshape to 2D matrix for linear layer
        # [14, 14, 4, 1024] → [14*14*4, 1024] = [784, 1024]
        preprocessed_weight = torch.reshape(
            preprocessed_weight, (self._patch_size * self._patch_size * 4, self._hidden_size)
        )

        # Convert to ttnn tensors
        self.patch_embed_weight = ttnn.from_torch(
            preprocessed_weight,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )

        # CLS token embedding [1, 1, 1024]
        # Store as torch tensor, convert to ttnn in forward() after expanding to batch size
        # This is a tiny tensor (2KB) so CPU→device transfer is negligible
        # Pattern from models/demos/vision/classification/vit/wormhole/tests/test_ttnn_optimized_sharded_vit_wh.py
        self.cls_token_torch = state_dict["embeddings.class_embedding"].unsqueeze(0).unsqueeze(0)

        # Position embeddings [1, 577, 1024] (576 patches + 1 CLS)
        # Option C: Use ttnn.as_tensor with DRAM + TILE (Qwen25 pattern)
        # DRAM memory config avoids sharding conflicts with small tensors
        position_embeddings = state_dict["embeddings.position_embedding.weight"]
        self.position_embeddings = ttnn.as_tensor(
            position_embeddings.unsqueeze(0),
            dtype=self.dtype,
            device=self.args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Post-LayerNorm (NOT USED - skip to match LLaVA select_layer=-2)
        # LLaVA uses hidden_states[-2] which is before post_layernorm
        # Weights kept here for reference, but not loaded to device to save memory
        # self.post_layernorm_weight = state_dict["post_layernorm.weight"]
        # self.post_layernorm_bias = state_dict["post_layernorm.bias"]

        # Pre-LayerNorm (applied BEFORE transformer blocks, after patch+pos embeddings)
        # Note: HuggingFace uses "pre_layrnorm" (typo in transformers library)
        self.pre_layrnorm_weight = ttnn.from_torch(
            state_dict["pre_layrnorm.weight"],
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        self.pre_layrnorm_bias = ttnn.from_torch(
            state_dict["pre_layrnorm.bias"],
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )

    def patchify(self, images):
        """
        Convert images to patch embeddings using ttnn.fold + ttnn.linear

        This is the recommended approach for Tenstorrent hardware as it's faster
        than conv2d. The convolutional patch embedding is converted to a linear
        projection after folding the image into patches.

        Args:
            images: torch.Tensor [B, 3, 336, 336]

        Returns:
            patches: ttnn.Tensor [B, 576, 1024]
        """
        batch_size = images.shape[0]

        # Step 1: Permute to NHWC format (required by ttnn.fold)
        # [B, 3, 336, 336] → [B, 336, 336, 3]
        images = torch.permute(images, (0, 2, 3, 1))

        # Step 2: Pad channels from 3 to 4 (required by ttnn.fold for tile alignment)
        # [B, 336, 336, 3] → [B, 336, 336, 4]
        images = torch.nn.functional.pad(images, (0, 1, 0, 0, 0, 0, 0, 0))

        # Step 3: Reshape for folding
        # [B, 336, 336, 4] → [B, 336, 24, 56] where 24 = 336/14, 56 = 14*4
        img_h, img_w = 336, 336
        images = images.reshape(batch_size, img_h, img_w // self._patch_size, 4 * self._patch_size)

        # Step 4: Convert to ttnn tensor (ROW_MAJOR for ttnn.fold)
        images_ttnn = ttnn.from_torch(
            images,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.args.mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )

        # Step 5: Fold patches (stride_h=patch_size, stride_w=1)
        # This rearranges the image into a sequence of flattened patches
        folded = ttnn.fold(images_ttnn, stride_h=self._patch_size, stride_w=1)
        ttnn.deallocate(images_ttnn)

        # Step 6: Convert to TILE layout for linear projection
        folded = ttnn.to_memory_config(folded, memory_config=ttnn.L1_MEMORY_CONFIG)
        folded = ttnn.to_layout(folded, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        # Step 7: Apply linear projection (folded shape is automatically compatible)
        # Note: CLIP patch embedding has no bias
        patches = ttnn.linear(
            folded,
            self.patch_embed_weight,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(folded)

        # Step 8: Reshape to [B, num_patches, hidden_size]
        patches = ttnn.to_layout(patches, layout=ttnn.ROW_MAJOR_LAYOUT)
        patches = ttnn.reshape(patches, (batch_size, self._num_patches, self._hidden_size))
        patches = ttnn.to_layout(patches, layout=ttnn.TILE_LAYOUT)

        return patches

    def forward(self, images):
        """
        Forward pass through CLIP vision encoder

        Args:
            images: torch.Tensor [B, 3, 336, 336]

        Returns:
            features: ttnn.Tensor [B, 576, 1024] (CLS token dropped)
        """
        batch_size = images.shape[0]

        # 1. Patchify: [B, 3, 336, 336] → [B, 576, 1024]
        patch_embeddings = self.patchify(images)

        # 2. Prepend CLS token: [B, 576, 1024] → [B, 577, 1024]
        # Expand CLS token to batch size in torch, then convert to ttnn
        # Pattern from ViT: models/demos/vision/classification/vit/wormhole/tests/test_ttnn_optimized_sharded_vit_wh.py
        torch_cls_token = self.cls_token_torch.expand(batch_size, -1, -1)
        cls_tokens = ttnn.from_torch(
            torch_cls_token,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )

        # Concat CLS with patches, then convert to TILE for subsequent ops
        embeddings = ttnn.concat([cls_tokens, patch_embeddings], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(cls_tokens)
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.TILE_LAYOUT)

        # 3. Add position embeddings (already in TILE layout from init via ttnn.as_tensor)
        embeddings = ttnn.add(embeddings, self.position_embeddings, memory_config=ttnn.L1_MEMORY_CONFIG)

        # 4. Apply pre_layrnorm (CLIP-specific: layer norm before transformer blocks)
        embeddings = ttnn.layer_norm(
            embeddings,
            weight=self.pre_layrnorm_weight,
            bias=self.pre_layrnorm_bias,
            epsilon=1e-5,
        )

        # 5. Forward through transformer blocks
        hidden_states = embeddings
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # 5. SKIP post-layernorm to match LLaVA select_layer=-2 behavior
        # LLaVA uses hidden_states[-2] which is the output of the last transformer block
        # WITHOUT post_layernorm applied. This matches the reference implementation.
        # NOTE: If you need the final CLIP output with post_layernorm, uncomment below:
        # hidden_states = ttnn.layer_norm(
        #     hidden_states,
        #     weight=self.post_layernorm_weight,
        #     bias=self.post_layernorm_bias,
        #     epsilon=1e-5,
        # )

        # 6. Drop CLS token (LLaVA uses only patch features)
        # [B, 577, 1024] → [B, 576, 1024]
        # NOTE: Using temporary host-side slicing (Option 3) for Week 1
        # Week 2 TODO: Replace with on-device ttnn.slice for better performance

        # Get batch size dynamically
        batch_size = hidden_states.shape[0]

        # Convert to torch, slice, convert back
        # Use get_device_tensors for mesh device compatibility (n300 dual chip)
        hidden_states_single = ttnn.get_device_tensors(hidden_states)[0]
        hidden_states_torch = ttnn.to_torch(hidden_states_single)
        ttnn.deallocate(hidden_states)  # Free device tensor after torch conversion

        # Slice to remove CLS token (Python slicing on CPU)
        patch_features_torch = hidden_states_torch[:, 1:, :]  # Drop first token (CLS)

        # Convert back to ttnn device tensor
        patch_features = ttnn.from_torch(
            patch_features_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.args.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )

        # Debug logging and validation
        logger.debug(f"CLIP encoder: Dropped CLS token. Output shape: {patch_features.shape}")

        # Shape assertion
        expected_shape = (batch_size, 576, 1024)
        actual_shape = tuple(patch_features.shape)
        assert (
            actual_shape == expected_shape
        ), f"Shape mismatch after CLS token slice: expected {expected_shape}, got {actual_shape}"

        return patch_features

    @property
    def hidden_size(self) -> int:
        """Return output hidden dimension"""
        return self._hidden_size

    @property
    def num_patches(self) -> int:
        """Return number of output patches"""
        return self._num_patches

    def get_output_dim(self):
        """Return output feature dimension (deprecated, use hidden_size property)"""
        return self._hidden_size

    def get_num_patches(self):
        """Return number of output patches (deprecated, use num_patches property)"""
        return self._num_patches


def create_clip_encoder(args, state_dict, weight_cache_path=None):
    """
    Factory function to create CLIP encoder

    Args:
        args: VisionModelArgs
        state_dict: Pre-trained weights
        weight_cache_path: Optional weight cache path

    Returns:
        CLIPVisionEncoder
    """
    return CLIPVisionEncoder(args, state_dict, weight_cache_path)
