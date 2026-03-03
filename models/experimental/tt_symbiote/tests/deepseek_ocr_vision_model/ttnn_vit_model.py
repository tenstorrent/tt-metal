# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Implementation of VitModel from DeepSeek-OCR

This module implements the VitModel using TTNN operations, converting all PyTorch
operations to TTNN equivalents. All tensors are TTNN tensors.

Architecture:
    - CLIPVisionEmbeddings: Patch embedding, class token, position embedding
    - Pre-layer norm
    - NoTPTransformer: Stack of transformer blocks
    - NoTPTransformerBlock: Layer norm, attention, feedforward
    - NoTPAttention: QKV projection, scaled dot product attention, output projection
    - NoTPFeedForward: Linear layers with quick_gelu activation
"""

import math
from typing import Optional, Dict
import torch
import ttnn
from easydict import EasyDict as adict


# ============================================================================
# Helper Functions
# ============================================================================


def tensor_1d_to_2d_ttnn(
    tensor_1d: torch.Tensor, device: ttnn.Device, dtype: ttnn.DataType = ttnn.bfloat16
) -> ttnn.Tensor:
    """
    Convert 1D PyTorch tensor to 2D TTNN tensor (1, N) for bias operations.

    Args:
        tensor_1d: 1D PyTorch tensor
        device: TTNN device
        dtype: TTNN data type

    Returns:
        2D TTNN tensor of shape (1, N)
    """
    tensor_2d = tensor_1d.unsqueeze(0)
    return ttnn.from_torch(
        tensor_2d,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def quick_gelu_ttnn(x: ttnn.Tensor) -> ttnn.Tensor:
    """
    Quick GELU activation: x * sigmoid(1.702 * x)

    Args:
        x: TTNN tensor

    Returns:
        TTNN tensor with quick_gelu applied
    """
    # Compute 1.702 * x
    scaled = ttnn.multiply(x, 1.702)
    # Compute sigmoid(1.702 * x)
    sigmoid_output = ttnn.sigmoid(scaled)
    # Compute x * sigmoid(1.702 * x)
    result = ttnn.multiply(x, sigmoid_output)

    ttnn.deallocate(scaled)
    ttnn.deallocate(sigmoid_output)

    return result


def get_abs_pos_ttnn(
    abs_pos: ttnn.Tensor,
    tgt_size: int,
    device: ttnn.Device,
) -> ttnn.Tensor:
    """
    Get absolute positional embeddings, interpolating if needed.

    Args:
        abs_pos: TTNN tensor of shape (1, L, C) with positional embeddings
        tgt_size: Target sequence size (excluding CLS token)
        device: TTNN device

    Returns:
        TTNN tensor of shape (1, tgt_size + 1, C) with interpolated positional embeddings
    """
    # Convert to torch for interpolation (TTNN doesn't have bicubic interpolation)
    abs_pos_torch = ttnn.to_torch(abs_pos)

    # Extract CLS token and position embeddings
    cls_token = abs_pos_torch[:, :1, :]  # (1, 1, C)
    old_pos_embed = abs_pos_torch[:, 1:, :]  # (1, L-1, C)

    src_size = int(math.sqrt(old_pos_embed.shape[1]))
    tgt_size_sqrt = int(math.sqrt(tgt_size))

    if src_size != tgt_size_sqrt:
        # Reshape for interpolation: (1, L-1, C) -> (1, C, src_size, src_size)
        old_pos_embed_2d = old_pos_embed.view(1, src_size, src_size, -1).permute(0, 3, 1, 2).contiguous()
        old_pos_embed_2d = old_pos_embed_2d.to(torch.float32)

        # Interpolate using PyTorch
        new_pos_embed_2d = torch.nn.functional.interpolate(
            old_pos_embed_2d,
            size=(tgt_size_sqrt, tgt_size_sqrt),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(old_pos_embed.dtype)

        # Reshape back: (1, C, tgt_size, tgt_size) -> (1, tgt_size, C)
        new_pos_embed = new_pos_embed_2d.permute(0, 2, 3, 1).contiguous()
        new_pos_embed = new_pos_embed.view(1, tgt_size, -1)

        # Concatenate CLS token
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=1)  # (1, tgt_size + 1, C)
    else:
        vision_pos_embed = abs_pos_torch

    # Convert back to TTNN
    return ttnn.from_torch(
        vision_pos_embed,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ============================================================================
# CLIPVisionEmbeddings (TTNN)
# ============================================================================


class CLIPVisionEmbeddingsTTNN:
    """
    CLIP Vision Embeddings using TTNN operations.

    Converts image patches to embeddings with class token and positional embeddings.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        image_size: int = 224,
        patch_size: int = 14,
        num_channels: int = 3,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        device: ttnn.Device = None,
    ):
        """
        Initialize CLIP vision embeddings.

        Args:
            hidden_size: Embedding dimension
            image_size: Input image size
            patch_size: Patch size
            num_channels: Number of input channels
            weights: PyTorch weights dict (optional, for loading pretrained)
            device: TTNN device
        """
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.device = device

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        # Initialize or load weights
        if weights is not None:
            # Load from pretrained weights
            self.class_embedding = ttnn.from_torch(
                weights["class_embedding"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Patch embedding: Conv2d weight (out_channels, in_channels, kernel_h, kernel_w)
            conv_weight = weights["patch_embedding.weight"]  # (hidden_size, 3, patch_size, patch_size)
            conv_bias = weights.get("patch_embedding.bias")

            # Convert Conv2d to linear format for TTNN
            # Flatten kernel: (hidden_size, 3, patch_size, patch_size) -> (hidden_size, 3*patch_size*patch_size)
            linear_weight = conv_weight.view(self.embed_dim, -1)  # (hidden_size, 3*patch_size*patch_size)
            linear_weight = linear_weight.T  # (3*patch_size*patch_size, hidden_size) for TTNN linear

            self.patch_embedding_weight = ttnn.from_torch(
                linear_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            if conv_bias is not None:
                self.patch_embedding_bias = tensor_1d_to_2d_ttnn(conv_bias, device)
            else:
                self.patch_embedding_bias = None

            # Position embedding - shape (num_positions, embed_dim)
            position_embedding_weight = weights["position_embedding.weight"]
            # Reshape to (1, num_positions, embed_dim) for get_abs_pos_ttnn
            position_embedding_reshaped = position_embedding_weight.unsqueeze(0)
            self.position_embedding = ttnn.from_torch(
                position_embedding_reshaped,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            # Initialize randomly
            class_embedding = torch.randn(self.embed_dim)
            self.class_embedding = ttnn.from_torch(
                class_embedding,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Initialize patch embedding weights
            linear_weight = torch.randn(3 * patch_size * patch_size, self.embed_dim)
            self.patch_embedding_weight = ttnn.from_torch(
                linear_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.patch_embedding_bias = None

            # Initialize position embedding - shape (1, num_positions, embed_dim)
            position_embedding = torch.randn(1, self.num_positions, self.embed_dim)
            self.position_embedding = ttnn.from_torch(
                position_embedding,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def _unfold_patches(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        """
        Extract patches from image using TTNN operations.

        Args:
            pixel_values: TTNN tensor (batch_size, channels, height, width)

        Returns:
            TTNN tensor (batch_size, num_patches, patch_size * patch_size * channels)
        """
        batch_size = pixel_values.shape[0]
        img_h = pixel_values.shape[2]
        img_w = pixel_values.shape[3]

        patches_h = img_h // self.patch_size
        patches_w = img_w // self.patch_size

        # Reshape to extract patches: (B, C, H, W) -> (B, C, patches_h, patch_size, patches_w, patch_size)
        pixel_values = ttnn.reshape(
            pixel_values, (batch_size, self.num_channels, patches_h, self.patch_size, patches_w, self.patch_size)
        )

        # Permute to group patches: (B, patches_h, patches_w, patch_size, patch_size, C)
        pixel_values = ttnn.permute(pixel_values, (0, 2, 4, 1, 3, 5))

        # Flatten patches: (B, patches_h, patches_w, patch_size * patch_size * C)
        pixel_values = ttnn.reshape(
            pixel_values, (batch_size, patches_h * patches_w, self.patch_size * self.patch_size * self.num_channels)
        )

        return pixel_values

    def forward(self, pixel_values: ttnn.Tensor, patch_embeds: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """
        Forward pass of CLIP vision embeddings.

        Args:
            pixel_values: TTNN tensor (batch_size, channels, height, width)
            patch_embeds: Optional pre-computed patch embeddings (batch_size, num_patches, embed_dim)

        Returns:
            TTNN tensor (batch_size, num_patches + 1, embed_dim)
        """
        batch_size = pixel_values.shape[0]

        # Get patch embeddings
        if patch_embeds is not None:
            patch_embeds = patch_embeds
        else:
            # Extract patches
            patches = self._unfold_patches(pixel_values)

            # Apply linear projection
            patch_embeds = ttnn.linear(
                patches,
                self.patch_embedding_weight,
                bias=self.patch_embedding_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(patches)

        # Expand class embedding: (embed_dim) -> (batch_size, 1, embed_dim)
        class_embeds = ttnn.reshape(self.class_embedding, (1, 1, self.embed_dim))
        class_embeds = ttnn.repeat(class_embeds, (batch_size, 1, 1))

        # Concatenate class token and patch embeddings
        embeddings = ttnn.concat([class_embeds, patch_embeds], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(class_embeds)
        # Note: Don't deallocate patch_embeds here - it's either passed in (user's responsibility)
        # or was just created and will be used in the concat, so it's part of embeddings now

        # Get position embeddings (with interpolation if needed)
        # Position embedding is already in shape (1, num_positions, embed_dim)
        # We need to interpolate if sequence length doesn't match
        # Note: embeddings.size(1) is the actual sequence length (num_patches + 1)
        # but get_abs_pos_ttnn expects the number of patches (excluding CLS token)
        actual_seq_len = embeddings.shape[1]  # This is num_patches + 1
        num_patches_actual = actual_seq_len - 1  # Exclude CLS token
        pos_embeds = get_abs_pos_ttnn(
            self.position_embedding,
            num_patches_actual,
            self.device,
        )

        # Add position embeddings
        embeddings = ttnn.add(embeddings, pos_embeds, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pos_embeds)

        return embeddings


# ============================================================================
# NoTPAttention (TTNN)
# ============================================================================


class NoTPAttentionTTNN:
    """
    No Tensor Parallelism Attention using TTNN operations.

    Implements multi-head self-attention with QKV projection and scaled dot product attention.
    """

    def __init__(
        self,
        cfg,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        device: ttnn.Device = None,
    ):
        """
        Initialize attention layer.

        Args:
            cfg: Configuration dict with num_attention_heads, hidden_size, etc.
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.device = device
        self.use_flash_attention = cfg.get("use_flash_attn", False)

        if weights is not None:
            # Load QKV projection weights
            qkv_weight = weights["qkv_proj.weight"]  # (hidden_size * 3, hidden_size)
            qkv_bias = weights.get("qkv_proj.bias")

            # Split into Q, K, V
            q_weight = qkv_weight[: self.hidden_size, :].T  # (hidden_size, hidden_size)
            k_weight = qkv_weight[self.hidden_size : 2 * self.hidden_size, :].T
            v_weight = qkv_weight[2 * self.hidden_size :, :].T

            # Convert to TTNN
            self.q_weight = ttnn.from_torch(
                q_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.k_weight = ttnn.from_torch(
                k_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.v_weight = ttnn.from_torch(
                v_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            if qkv_bias is not None:
                q_bias = qkv_bias[: self.hidden_size]
                k_bias = qkv_bias[self.hidden_size : 2 * self.hidden_size]
                v_bias = qkv_bias[2 * self.hidden_size :]

                self.q_bias = tensor_1d_to_2d_ttnn(q_bias, device)
                self.k_bias = tensor_1d_to_2d_ttnn(k_bias, device)
                self.v_bias = tensor_1d_to_2d_ttnn(v_bias, device)
            else:
                self.q_bias = None
                self.k_bias = None
                self.v_bias = None

            # Output projection
            out_weight = weights["out_proj.weight"].T  # (hidden_size, hidden_size)
            self.out_weight = ttnn.from_torch(
                out_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            out_bias = weights.get("out_proj.bias")
            if out_bias is not None:
                self.out_bias = tensor_1d_to_2d_ttnn(out_bias, device)
            else:
                self.out_bias = None
        else:
            # Initialize randomly
            q_weight = torch.randn(self.hidden_size, self.hidden_size)
            k_weight = torch.randn(self.hidden_size, self.hidden_size)
            v_weight = torch.randn(self.hidden_size, self.hidden_size)

            self.q_weight = ttnn.from_torch(
                q_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.k_weight = ttnn.from_torch(
                k_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.v_weight = ttnn.from_torch(
                v_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

            out_weight = torch.randn(self.hidden_size, self.hidden_size)
            self.out_weight = ttnn.from_torch(
                out_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.out_bias = None

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of attention.

        Args:
            x: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        bsz, seqlen, _ = x.shape

        # QKV projections
        query = ttnn.linear(
            x,
            self.q_weight,
            bias=self.q_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        key = ttnn.linear(
            x,
            self.k_weight,
            bias=self.k_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        value = ttnn.linear(
            x,
            self.v_weight,
            bias=self.v_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Reshape to (batch_size, seqlen, num_heads, head_dim)
        query = ttnn.reshape(query, (bsz, seqlen, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (bsz, seqlen, self.num_heads, self.head_dim))
        value = ttnn.reshape(value, (bsz, seqlen, self.num_heads, self.head_dim))

        # Permute to (batch_size, num_heads, seqlen, head_dim)
        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.permute(value, (0, 2, 1, 3))

        # Scaled dot product attention
        scale = 1.0 / math.sqrt(self.head_dim)

        # Configure SDPA
        device_grid = self.device.compute_with_storage_grid_size()
        grid_x = min(8, device_grid.x)
        grid_y = min(8, device_grid.y)

        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            q_chunk_size=min(256, seqlen),
            k_chunk_size=min(256, seqlen),
            exp_approx_mode=False,
        )

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,  # SDPA needs this off
        )

        attention_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=scale,
            is_causal=False,
            program_config=sdpa_cfg,
            compute_kernel_config=compute_kernel_config,
        )

        # Permute back to (batch_size, seqlen, num_heads, head_dim)
        attention_output = ttnn.permute(attention_output, (0, 2, 1, 3))

        # Reshape to (batch_size, seqlen, hidden_size)
        attention_output = ttnn.reshape(attention_output, (bsz, seqlen, self.hidden_size))

        # Output projection
        output = ttnn.linear(
            attention_output,
            self.out_weight,
            bias=self.out_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Cleanup
        ttnn.deallocate(query)
        ttnn.deallocate(key)
        ttnn.deallocate(value)
        ttnn.deallocate(attention_output)

        return output


# ============================================================================
# NoTPFeedForward (TTNN)
# ============================================================================


class NoTPFeedForwardTTNN:
    """
    No Tensor Parallelism Feed Forward using TTNN operations.

    Implements two linear layers with quick_gelu activation.
    """

    def __init__(
        self,
        cfg,
        dim: int,
        hidden_dim: int,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        device: ttnn.Device = None,
    ):
        """
        Initialize feedforward layer.

        Args:
            cfg: Configuration dict
            dim: Input/output dimension
            hidden_dim: Hidden dimension
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.device = device

        if weights is not None:
            # FC1 weights
            fc1_weight = weights["fc1.weight"].T  # (hidden_dim, dim)
            self.fc1_weight = ttnn.from_torch(
                fc1_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            fc1_bias = weights.get("fc1.bias")
            if fc1_bias is not None:
                self.fc1_bias = tensor_1d_to_2d_ttnn(fc1_bias, device)
            else:
                self.fc1_bias = None

            # FC2 weights
            fc2_weight = weights["fc2.weight"].T  # (dim, hidden_dim)
            self.fc2_weight = ttnn.from_torch(
                fc2_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            fc2_bias = weights.get("fc2.bias")
            if fc2_bias is not None:
                self.fc2_bias = tensor_1d_to_2d_ttnn(fc2_bias, device)
            else:
                self.fc2_bias = None
        else:
            # Initialize randomly
            fc1_weight = torch.randn(hidden_dim, dim)
            self.fc1_weight = ttnn.from_torch(
                fc1_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.fc1_bias = None

            fc2_weight = torch.randn(dim, hidden_dim)
            self.fc2_weight = ttnn.from_torch(
                fc2_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.fc2_bias = None

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of feedforward layer.

        Args:
            x: TTNN tensor (batch_size, seq_len, dim)

        Returns:
            TTNN tensor (batch_size, seq_len, dim)
        """
        # FC1
        output = ttnn.linear(
            x,
            self.fc1_weight,
            bias=self.fc1_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Quick GELU
        output = quick_gelu_ttnn(output)

        # FC2
        output = ttnn.linear(
            output,
            self.fc2_weight,
            bias=self.fc2_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return output


# ============================================================================
# NoTPTransformerBlock (TTNN)
# ============================================================================


class NoTPTransformerBlockTTNN:
    """
    No Tensor Parallelism Transformer Block using TTNN operations.

    Implements pre-norm transformer block with attention and feedforward.
    """

    def __init__(
        self,
        cfg,
        layer_id: int,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        device: ttnn.Device = None,
    ):
        """
        Initialize transformer block.

        Args:
            cfg: Configuration dict
            layer_id: Layer index
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        self.layer_id = layer_id
        self.device = device
        self.hidden_size = cfg.hidden_size
        self.layernorm_epsilon = cfg.layernorm_epsilon

        # Layer norms
        if weights is not None:
            ln1_weight = weights["layer_norm1.weight"]
            ln1_bias = weights.get("layer_norm1.bias")
            ln2_weight = weights["layer_norm2.weight"]
            ln2_bias = weights.get("layer_norm2.bias")
        else:
            ln1_weight = torch.ones(self.hidden_size)
            ln1_bias = torch.zeros(self.hidden_size)
            ln2_weight = torch.ones(self.hidden_size)
            ln2_bias = torch.zeros(self.hidden_size)

        self.layer_norm1_weight = ttnn.from_torch(
            ln1_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layer_norm1_bias = ttnn.from_torch(
            ln1_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layer_norm2_weight = ttnn.from_torch(
            ln2_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layer_norm2_bias = ttnn.from_torch(
            ln2_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Attention
        attn_weights = weights.get("self_attn", {}) if weights is not None else None
        self.self_attn = NoTPAttentionTTNN(cfg, weights=attn_weights, device=device)

        # Feedforward
        mlp_weights = weights.get("mlp", {}) if weights is not None else None
        self.mlp = NoTPFeedForwardTTNN(
            cfg,
            dim=cfg.hidden_size,
            hidden_dim=cfg.ffn_hidden_size,
            weights=mlp_weights,
            device=device,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of transformer block.

        Args:
            x: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        # Pre-norm attention
        residual = ttnn.layer_norm(
            x,
            weight=self.layer_norm1_weight,
            bias=self.layer_norm1_bias,
            epsilon=self.layernorm_epsilon,
        )
        residual = self.self_attn.forward(residual)
        h = ttnn.add(x, residual, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(residual)

        # Pre-norm feedforward
        out = ttnn.layer_norm(
            h,
            weight=self.layer_norm2_weight,
            bias=self.layer_norm2_bias,
            epsilon=self.layernorm_epsilon,
        )
        out = self.mlp.forward(out)
        out = ttnn.add(h, out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)

        return out


# ============================================================================
# NoTPTransformer (TTNN)
# ============================================================================


class NoTPTransformerTTNN:
    """
    No Tensor Parallelism Transformer using TTNN operations.

    Stack of transformer blocks.
    """

    def __init__(
        self,
        cfg,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        device: ttnn.Device = None,
    ):
        """
        Initialize transformer.

        Args:
            cfg: Configuration dict
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        self.cfg = cfg
        self.num_layers = cfg.num_layers
        self.device = device

        self.layers = []
        for layer_id in range(self.num_layers):
            layer_weights = weights.get(f"layers.{layer_id}", {}) if weights is not None else None
            layer = NoTPTransformerBlockTTNN(
                cfg,
                layer_id=layer_id + 1,
                weights=layer_weights,
                device=device,
            )
            self.layers.append(layer)

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of transformer.

        Args:
            hidden_states: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states)
        return hidden_states


# ============================================================================
# VitModel (TTNN)
# ============================================================================


class VitModelTTNN:
    """
    Vision Transformer Model using TTNN operations.

    Complete ViT model with embeddings, pre-norm, and transformer encoder.
    """

    def __init__(
        self,
        cfg,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        device: ttnn.Device = None,
        freeze_embed: bool = False,
        freeze_pre_norm: bool = False,
    ):
        """
        Initialize ViT model.

        Args:
            cfg: Configuration dict
            weights: PyTorch weights dict (optional)
            device: TTNN device
            freeze_embed: Whether to freeze embedding weights
            freeze_pre_norm: Whether to freeze pre-norm weights
        """
        self.cfg = cfg
        self.device = device

        # Embeddings
        embed_weights = weights.get("embeddings", {}) if weights is not None else None
        self.embeddings = CLIPVisionEmbeddingsTTNN(
            hidden_size=cfg.hidden_size,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            num_channels=3,
            weights=embed_weights,
            device=device,
        )

        # Transformer
        transformer_weights = weights.get("transformer", {}) if weights is not None else None
        self.transformer = NoTPTransformerTTNN(
            cfg=cfg,
            weights=transformer_weights,
            device=device,
        )

        # Pre-layer norm
        if weights is not None:
            pre_norm_weight = weights.get("pre_layrnorm.weight")
            pre_norm_bias = weights.get("pre_layrnorm.bias")
        else:
            pre_norm_weight = torch.ones(cfg.hidden_size)
            pre_norm_bias = torch.zeros(cfg.hidden_size)

        self.pre_layrnorm_weight = ttnn.from_torch(
            pre_norm_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.pre_layrnorm_bias = ttnn.from_torch(
            pre_norm_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.pre_layernorm_epsilon = cfg.get("pre_layernorm_epsilon", 1e-5)

    def forward(
        self,
        x: ttnn.Tensor,
        patch_embeds: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass of ViT model.

        Args:
            x: TTNN tensor (batch_size, channels, height, width) - input image
            patch_embeds: Optional pre-computed patch embeddings

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        # Embeddings
        x = self.embeddings.forward(x, patch_embeds)

        # Pre-layer norm
        hidden_states = ttnn.layer_norm(
            x,
            weight=self.pre_layrnorm_weight,
            bias=self.pre_layrnorm_bias,
            epsilon=self.pre_layernorm_epsilon,
        )
        ttnn.deallocate(x)

        # Transformer
        output = self.transformer.forward(hidden_states)
        ttnn.deallocate(hidden_states)

        return output


# ============================================================================
# Configuration and Builder
# ============================================================================


vit_model_cfg = adict(
    num_layers=24,
    hidden_size=1024,
    num_heads=16,
    num_attention_heads=16,
    ffn_hidden_size=4096,
    seq_length=256,
    max_position_embeddings=256,
    use_flash_attn=False,
    understand_projector_stride=2,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    no_persist_layer_norm=False,
    layernorm_epsilon=1e-5,
    pre_layernorm_epsilon=1e-5,
    image_size=224,
    patch_size=14,
    recompute_list=[],
)


def build_clip_l_ttnn(
    weights: Optional[Dict[str, torch.Tensor]] = None,
    device: ttnn.Device = None,
    freeze_embed: bool = False,
    freeze_pre_norm: bool = False,
) -> VitModelTTNN:
    """
    Build CLIP-L ViT model using TTNN.

    Args:
        weights: PyTorch weights dict (optional)
        device: TTNN device
        freeze_embed: Whether to freeze embedding weights
        freeze_pre_norm: Whether to freeze pre-norm weights

    Returns:
        VitModelTTNN instance
    """
    return VitModelTTNN(
        cfg=vit_model_cfg,
        weights=weights,
        device=device,
        freeze_embed=freeze_embed,
        freeze_pre_norm=freeze_pre_norm,
    )
