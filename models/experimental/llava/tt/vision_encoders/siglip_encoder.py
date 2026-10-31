# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
SigLIP Vision Encoder - TTNN Implementation

Implements `google/siglip-so400m-patch14-384` for Tenstorrent hardware.

Architecture:
    - Input: [B, 3, 384, 384]
    - Patch embedding: Conv 14x14 -> [B, 729, 1152]  (27x27 patches, no CLS token)
    - Position embeddings: Learnable [1, 729, 1152]
    - Transformer blocks: 27 layers (LLaVA select_layer=-2 -> run 26)
        - Multi-head self-attention (16 heads, head_dim=72)
        - LayerNorm (eps=1e-6)
        - MLP (1152 -> 4304 -> 1152, gelu_pytorch_tanh activation)
    - Output: [B, 729, 1152]  (no CLS token to drop)

Key differences from CLIP (`clip_encoder.py`):
    1. No CLS token (SigLIP has only patch tokens)
    2. Patch embedding has a bias (CLIP does not)
    3. No pre_layernorm (CLIP has `pre_layrnorm` before blocks)
    4. Activation is `gelu_pytorch_tanh` (CLIP uses QuickGELU)
    5. Dims: hidden 1152, layers 27, intermediate 4304, image 384,
       patches 729, head_dim 72, LN eps 1e-6

Reference:
    - LLaVA-MORE/src/llava/model/multimodal_encoder/siglip_encoder.py
    - transformers.SiglipVisionModel
    - models/demos/llava/tt/vision_encoders/clip_encoder.py (template)
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn
from models.demos.llava.tt.vision_encoders.base_encoder import BaseVisionEncoder


class SigLIPVisionTransformerBlock(torch.nn.Module):
    """
    Single SigLIP Vision Transformer Block

    Components:
        1. LayerNorm (eps=1e-6)
        2. Multi-head Self-Attention
        3. Residual connection
        4. LayerNorm (eps=1e-6)
        5. MLP (Linear -> gelu_pytorch_tanh -> Linear)
        6. Residual connection

    Args:
        args: VisionModelArgs with device config
        state_dict: Pre-trained weights
        layer_num: Block index (0-25)
        dtype: ttnn.bfloat16
    """

    def __init__(self, args, state_dict, layer_num, dtype):
        super().__init__()

        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype

        self.hidden_size = 1152
        self.num_heads = 16
        self.head_dim = self.hidden_size // self.num_heads  # 72
        self.intermediate_size = 4304
        self.ln_eps = 1e-6

        prefix = f"encoder.layers.{layer_num}"

        self._load_weights(state_dict, prefix)

    def _load_weights(self, state_dict, prefix):
        """Load and preprocess weights from PyTorch state_dict"""

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

        # SigLIP uses separate q_proj / k_proj / v_proj (with bias)
        qkv_weight = torch.cat(
            [
                state_dict[f"{prefix}.self_attn.q_proj.weight"],
                state_dict[f"{prefix}.self_attn.k_proj.weight"],
                state_dict[f"{prefix}.self_attn.v_proj.weight"],
            ],
            dim=0,
        )  # [3456, 1152]
        qkv_bias = torch.cat(
            [
                state_dict[f"{prefix}.self_attn.q_proj.bias"],
                state_dict[f"{prefix}.self_attn.k_proj.bias"],
                state_dict[f"{prefix}.self_attn.v_proj.bias"],
            ],
            dim=0,
        )  # [3456]

        self.qkv_weight = preprocess_linear_weight(
            qkv_weight,
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        self.qkv_weight = ttnn.to_device(self.qkv_weight, self.args.mesh_device)

        self.qkv_bias = preprocess_linear_bias(
            qkv_bias,
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        self.qkv_bias = ttnn.to_device(self.qkv_bias, self.args.mesh_device)

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
        residual = x

        x = ttnn.layer_norm(
            x,
            weight=self.ln1_weight,
            bias=self.ln1_bias,
            epsilon=self.ln_eps,
        )

        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=self.qkv_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # Manual QKV split + head reshape. SigLIP head_dim=72 is NOT a multiple of
        # the tile width (32), so the fused `ttnn.transformer.split_query_key_value_and_split_heads`
        # kernel rejects it (`TT_FATAL: head_size % TILE_WIDTH == 0`). The CLIP
        # path uses that fused op because CLIP head_dim=64 is tile-aligned.
        # For SigLIP we instead reshape into [B, S, 3, n_heads, head_dim], slice
        # along the Q/K/V axis, permute into [B, n_heads, S, head_dim] for matmul.
        # The downstream matmuls + softmax + manual concat-heads (below) work
        # unchanged on these "natural" shapes — only the data-movement ops differ.
        B = qkv.shape[0]
        S = qkv.shape[-2]
        H = self.num_heads
        DH = self.head_dim  # 72

        qkv_5d = ttnn.reshape(qkv, (B, S, 3, H, DH))
        ttnn.deallocate(qkv)
        # Split along axis=2 (size 3) -> [q, k, v], each [B, S, 1, H, DH]
        q_5d, k_5d, v_5d = ttnn.split(qkv_5d, 1, dim=2)
        ttnn.deallocate(qkv_5d)
        # Squeeze the size-1 axis 2 and permute for matmul:
        #   query, value -> [B, H, S, DH]  (permute (0, 2, 1, 3))
        #   key           -> [B, H, DH, S] (permute (0, 2, 3, 1) — pre-transposed for Q @ K^T)
        # The fused `ttnn.transformer.split_query_key_value_and_split_heads` returns K in
        # this DH-S ordering; the manual path must do the same so `matmul(q, k)`
        # has K @ N = DH @ S, giving [B, H, S, S] attention scores.
        query = ttnn.permute(ttnn.reshape(q_5d, (B, S, H, DH)), (0, 2, 1, 3))
        key = ttnn.permute(ttnn.reshape(k_5d, (B, S, H, DH)), (0, 2, 3, 1))
        value = ttnn.permute(ttnn.reshape(v_5d, (B, S, H, DH)), (0, 2, 1, 3))
        q_5d.deallocate(True)
        k_5d.deallocate(True)
        v_5d.deallocate(True)

        attention_scores = ttnn.matmul(
            query,
            key,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(query)
        ttnn.deallocate(key)

        scale = 1.0 / (self.head_dim**0.5)
        attention_scores = ttnn.mul(attention_scores, scale)

        if attention_mask is not None:
            attention_scores = ttnn.add(attention_scores, attention_mask)

        attention_probs = ttnn.softmax(attention_scores, dim=-1)
        ttnn.deallocate(attention_scores)

        context_layer = ttnn.matmul(
            attention_probs,
            value,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attention_probs)
        ttnn.deallocate(value)

        # Manual concatenate heads (counterpart of manual split above).
        # context_layer is [B, H, S, DH]; permute to [B, S, H, DH] and reshape
        # to [B, S, H*DH] = [B, S, hidden_size]. Avoids the same head_dim%-32
        # constraint that the fused `ttnn.transformer.concatenate_heads` enforces.
        context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
        attention_output = ttnn.reshape(context_layer, (B, S, H * DH))
        ttnn.deallocate(context_layer)

        x = ttnn.linear(
            attention_output,
            self.attn_out_weight,
            bias=self.attn_out_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attention_output)

        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)

        residual = x

        x = ttnn.layer_norm(
            x,
            weight=self.ln2_weight,
            bias=self.ln2_bias,
            epsilon=self.ln_eps,
        )
        ln2_output = x

        # fc1: [B, seq_len, 1152] -> [B, seq_len, 4304]
        x = ttnn.linear(
            x,
            self.mlp_fc1_weight,
            bias=self.mlp_fc1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(ln2_output)

        # gelu_pytorch_tanh: tanh-approximation of GELU
        # Per diffusion_gemma/tt/self_conditioning.py:80 this is the canonical mapping
        x = ttnn.gelu(x, fast_and_approximate_mode=True)

        # fc2: [B, seq_len, 4304] -> [B, seq_len, 1152]
        x = ttnn.linear(
            x,
            self.mlp_fc2_weight,
            bias=self.mlp_fc2_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)

        return x


class SigLIPVisionEncoder(BaseVisionEncoder):
    """
    SigLIP Vision Encoder for Tenstorrent Hardware

    Implements `google/siglip-so400m-patch14-384` in TTNN.

    Args:
        args: VisionModelArgs with device configuration
        state_dict: Pre-trained weights from HuggingFace SiglipVisionModel
        weight_cache_path: Path to cache preprocessed weights
        select_layer: Which hidden layer to extract (default: -2, LLaVA-MORE default)
            hidden_states has num_layers+1 entries (embedding output + each layer output)
            -2 = output of second-to-last transformer block (before post_layernorm)
            -1 = output of last transformer block (before post_layernorm)
    """

    def __init__(
        self,
        args,
        state_dict,
        weight_cache_path: Optional[Path] = None,
        select_layer: int = -2,
    ):
        super().__init__()

        self.args = args
        self.dtype = ttnn.bfloat16
        self.weight_cache_path = weight_cache_path
        self.select_layer = select_layer

        # SigLIP SO400M @ 384 config
        self._hidden_size = 1152
        self.num_layers = 27
        self.num_heads = 16
        self.image_size = 384
        self._patch_size = 14
        self._num_patches = (self.image_size // self._patch_size) ** 2  # 729
        self.ln_eps = 1e-6

        # num_blocks_to_run = num_layers + 1 + select_layer = 27 + 1 - 2 = 26
        self.num_blocks_to_run = self.num_layers + 1 + self.select_layer

        logger.info(f"Initializing SigLIP Vision Encoder")
        logger.info(f"  Hidden size: {self._hidden_size}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  Num heads: {self.num_heads}")
        logger.info(f"  Image size: {self.image_size}")
        logger.info(f"  Num patches: {self._num_patches}")
        logger.info(f"  Select layer: {self.select_layer} -> running {self.num_blocks_to_run} transformer blocks")

        self._load_embeddings(state_dict)

        self.blocks = []
        for layer_num in range(self.num_blocks_to_run):
            block = SigLIPVisionTransformerBlock(
                args=args,
                state_dict=state_dict,
                layer_num=layer_num,
                dtype=self.dtype,
            )
            self.blocks.append(block)

        logger.info(f"SigLIP encoder initialized with {self.num_blocks_to_run} layers")

    def _load_embeddings(self, state_dict):
        """Load patch embeddings + position embeddings.

        Differences from CLIP:
        - Patch embedding HAS a bias (CLIP does not)
        - No CLS token (SigLIP has only patch tokens)
        - No pre_layernorm (CLIP has `pre_layrnorm`)
        - Position embeddings shape [729, 1152] (vs CLIP's [577, 1024])
        """

        # Patch embedding: conv2d weight [1152, 3, 14, 14] + bias [1152]
        # Same channel-pad-to-4 + fold + ttnn.linear pattern as CLIP, but WITH bias
        patch_embed_weight = state_dict["embeddings.patch_embedding.weight"]
        patch_embed_bias = state_dict["embeddings.patch_embedding.bias"]  # [1152]

        # Step 1: Pad channels from 3 to 4 (required by ttnn.fold for tile alignment)
        preprocessed_weight = torch.nn.functional.pad(patch_embed_weight, (0, 0, 0, 0, 0, 1))

        # Step 2: Permute to HWIO (height, width, in_channels, out_channels)
        # [1152, 4, 14, 14] -> [14, 14, 4, 1152]
        preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))

        # Step 3: Reshape to 2D matrix for linear layer
        # [14, 14, 4, 1152] -> [14*14*4, 1152] = [784, 1152]
        preprocessed_weight = torch.reshape(
            preprocessed_weight,
            (self._patch_size * self._patch_size * 4, self._hidden_size),
        )

        self.patch_embed_weight = ttnn.from_torch(
            preprocessed_weight,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )

        # Tile-align bias to [1, 32] (tile width) so ttnn.linear can use it
        # Pattern: preprocess_linear_bias pads bias to a tile-friendly shape
        self.patch_embed_bias = preprocess_linear_bias(
            patch_embed_bias,
            dtype=self.dtype,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        self.patch_embed_bias = ttnn.to_device(self.patch_embed_bias, self.args.mesh_device)

        # Position embeddings [729, 1152] -> [1, 729, 1152]
        # Use ttnn.as_tensor with DRAM + TILE (Qwen25 pattern, same as CLIP)
        position_embeddings = state_dict["embeddings.position_embedding.weight"]
        self.position_embeddings = ttnn.as_tensor(
            position_embeddings.unsqueeze(0),
            dtype=self.dtype,
            device=self.args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # SigLIP has NO pre_layernorm and NO class_embedding
        # (post_layernorm is skipped at select_layer=-2, same as CLIP)

    def patchify(self, images):
        """
        Convert images to patch embeddings using ttnn.fold + ttnn.linear

        Args:
            images: torch.Tensor [B, 3, 384, 384]

        Returns:
            patches: ttnn.Tensor [B, 729, 1152]
        """
        batch_size = images.shape[0]

        # Step 1: Permute to NHWC (required by ttnn.fold)
        # [B, 3, 384, 384] -> [B, 384, 384, 3]
        images = torch.permute(images, (0, 2, 3, 1))

        # Step 1b: SigLIP image_size (384) is NOT divisible by patch_size (14).
        # HF SiglipVisionEmbeddings uses nn.Conv2d(stride=14, padding="valid").
        # On a 384x384 input that produces 27x27=729 patches by floor-cropping
        # the last 6 rows/cols (covers pixels [0..377]). To match HF semantics
        # with our fold-based patchify (which requires H % patch_size == 0),
        # crop H and W to the nearest multiple of patch_size <= image_size:
        # 378 = 27 * 14. The dropped 6 rows/cols correspond to exactly the
        # bottom/right pixels HF conv doesn't reach, so output is bit-identical.
        grid_per_side = self.image_size // self._patch_size  # 27
        cropped = grid_per_side * self._patch_size  # 378
        images = images[:, :cropped, :cropped, :]  # [B, 378, 378, 3]

        # Step 2: Pad channels from 3 to 4 (required by ttnn.fold for tile alignment)
        # [B, 378, 378, 3] -> [B, 378, 378, 4]
        images = torch.nn.functional.pad(images, (0, 1, 0, 0, 0, 0, 0, 0))

        # Step 3: Reshape for folding
        # [B, 378, 378, 4] -> [B, 378, 27, 56] where 27 = 378/14, 56 = 14*4
        img_h, img_w = cropped, cropped
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

        # Step 5: Fold patches
        folded = ttnn.fold(images_ttnn, stride_h=self._patch_size, stride_w=1)
        ttnn.deallocate(images_ttnn)

        # Step 6: Convert to TILE layout for linear projection
        folded = ttnn.to_memory_config(folded, memory_config=ttnn.L1_MEMORY_CONFIG)
        folded = ttnn.to_layout(folded, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        # Step 7: Apply linear projection WITH bias (SigLIP-specific)
        patches = ttnn.linear(
            folded,
            self.patch_embed_weight,
            bias=self.patch_embed_bias,
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
        Forward pass through SigLIP vision encoder

        Args:
            images: torch.Tensor [B, 3, 384, 384]

        Returns:
            features: ttnn.Tensor [B, 729, 1152]  (no CLS token to drop)
        """
        batch_size = images.shape[0]

        # 1. Patchify: [B, 3, 384, 384] -> [B, 729, 1152]
        embeddings = self.patchify(images)

        # 2. Add position embeddings (no CLS token, no concat needed)
        embeddings = ttnn.add(embeddings, self.position_embeddings, memory_config=ttnn.L1_MEMORY_CONFIG)

        # 3. No pre_layernorm (SigLIP-specific) - embeddings go straight into blocks

        # 4. Forward through transformer blocks
        hidden_states = embeddings
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # 5. SKIP post-layernorm to match LLaVA-MORE select_layer=-2 behavior
        # (same rationale as CLIP: hidden_states[-2] is before post_layernorm)

        # 6. SigLIP has no CLS token to drop; output is already [B, 729, 1152]
        # Convert to DRAM for downstream projector
        hidden_states_single = ttnn.get_device_tensors(hidden_states)[0]
        patch_features_torch = ttnn.to_torch(hidden_states_single)
        ttnn.deallocate(hidden_states)

        patch_features = ttnn.from_torch(
            patch_features_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.args.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )

        logger.debug(f"SigLIP encoder output shape: {patch_features.shape}")

        expected_shape = (batch_size, 729, 1152)
        actual_shape = tuple(patch_features.shape)
        assert actual_shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {actual_shape}"

        return patch_features

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def resolution(self) -> int:
        return self.image_size

    @property
    def patch_size(self) -> int:
        return self._patch_size


def create_siglip_encoder(args, state_dict, weight_cache_path=None):
    """
    Factory function to create SigLIP encoder

    Args:
        args: VisionModelArgs
        state_dict: Pre-trained weights (from SiglipVisionModel.state_dict())
        weight_cache_path: Optional weight cache path

    Returns:
        SigLIPVisionEncoder
    """
    return SigLIPVisionEncoder(args, state_dict, weight_cache_path)
