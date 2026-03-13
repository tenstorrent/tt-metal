# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import math

from models.experimental.tt_symbiote.core.module import TTNNModule
from typing import Optional
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor


########## CLIP Vision Embeddings ############
class TTNNClipVisionEmbeddings(TTNNModule):
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

        super().__init__()

        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.torch_layer_cp = None

    @classmethod
    def from_torch(cls, visionEmbedding):
        """Create TTNN module from PyTorch equivalent."""
        new_clip = cls(
            hidden_size=visionEmbedding.embed_dim,
            image_size=visionEmbedding.image_size,
            patch_size=visionEmbedding.patch_size,
            num_channels=3,
        )

        new_clip.torch_layer_cp = visionEmbedding
        new_clip._fallback_torch_layer = visionEmbedding
        return new_clip

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        # Load from pretrained weights
        self.class_embedding = ttnn.from_torch(
            self.torch_layer_cp.class_embedding.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Patch embedding: Conv2d weight (out_channels, in_channels, kernel_h, kernel_w)
        conv_weight = self.torch_layer_cp.patch_embedding.weight.data  # (hidden_size, 3, patch_size, patch_size)

        conv_bias = None
        if self.torch_layer_cp.patch_embedding.bias is not None:
            conv_bias = self.torch_layer_cp.patch_embedding.bias.data

        # Convert Conv2d to linear format for TTNN
        # Flatten kernel: (hidden_size, 3, patch_size, patch_size) -> (hidden_size, 3*patch_size*patch_size)
        linear_weight = conv_weight.view(self.embed_dim, -1)  # (hidden_size, 3*patch_size*patch_size)
        linear_weight = linear_weight.T  # (3*patch_size*patch_size, hidden_size) for TTNN linear

        self.patch_embedding_weight = ttnn.from_torch(
            linear_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if conv_bias is not None:
            self.patch_embedding_bias = self.tensor_1d_to_2d_ttnn(conv_bias)
        else:
            self.patch_embedding_bias = None

        # Position embedding - shape (num_positions, embed_dim)
        position_embedding_weight = self.torch_layer_cp.position_embedding.weight.data
        # Reshape to (1, num_positions, embed_dim) for get_abs_pos_ttnn
        position_embedding_reshaped = position_embedding_weight.unsqueeze(0)
        self.position_embedding = ttnn.from_torch(
            position_embedding_reshaped,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        self.class_embedding = ttnn.to_device(self.class_embedding, self.device)
        self.patch_embedding_weight = ttnn.to_device(self.patch_embedding_weight, self.device)
        self.position_embedding = ttnn.to_device(self.position_embedding, self.device)
        if self.patch_embedding_bias is not None:
            self.patch_embedding_bias = ttnn.to_device(self.patch_embedding_bias, self.device)

    def deallocate_weights_impl(self):
        """Deallocate device memory."""
        ttnn.deallocate(self.class_embedding)
        ttnn.deallocate(self.patch_embedding_weight)
        ttnn.deallocate(self.position_embedding)
        if self.patch_embedding_bias is not None:
            ttnn.deallocate(self.patch_embedding_bias)

    def tensor_1d_to_2d_ttnn(self, tensor_1d: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
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

    def get_abs_pos_ttnn(
        self,
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

    def forward(self, pixel_values: ttnn.Tensor, patch_embeds: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """
        Forward pass of CLIP vision embeddings.

        Args:
            pixel_values: TTNN tensor (batch_size, channels, height, width)
            patch_embeds: Optional pre-computed patch embeddings (batch_size, num_patches, embed_dim)

        Returns:
            TTNN tensor (batch_size, num_patches + 1, embed_dim)
        """

        if pixel_values.layout != ttnn.TILE_LAYOUT:
            pixel_values = ttnn.to_layout(pixel_values, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if patch_embeds is not None and patch_embeds.layout != ttnn.TILE_LAYOUT:
            patch_embeds = ttnn.to_layout(patch_embeds, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

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
        pos_embeds = self.get_abs_pos_ttnn(
            self.position_embedding,
            num_patches_actual,
            self.device,
        )

        # Add position embeddings
        embeddings = ttnn.add(embeddings, pos_embeds, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pos_embeds)

        return embeddings


########## No Tensor Parallelism Attention ############
class TTNNNoTPAttention(TTNNModule):
    """
    No Tensor Parallelism Attention using TTNN operations.

    Implements multi-head self-attention with QKV projection and scaled dot product attention.
    """

    def __init__(self, num_heads, n_local_heads, head_dim, max_seq_len, use_flash_attention):
        """
        Initialize attention layer.

        Args:
            cfg: Configuration dict with num_attention_heads, hidden_size, etc.
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        super().__init__()

        self.num_heads = num_heads
        self.n_local_heads = n_local_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.use_flash_attention = use_flash_attention
        self.hidden_size = self.head_dim * self.num_heads
        self.torch_layer_cp = None

    @classmethod
    def from_torch(cls, NoTPAttention):
        """Create TTNN module from PyTorch equivalent."""
        new_Attn = cls(
            NoTPAttention.num_heads,
            NoTPAttention.n_local_heads,
            NoTPAttention.head_dim,
            NoTPAttention.max_seq_len,
            NoTPAttention.use_flash_attention,
        )
        new_Attn._fallback_torch_layer = NoTPAttention
        new_Attn.torch_layer_cp = NoTPAttention
        return new_Attn

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        # Load QKV projection weights
        qkv_weight = self.torch_layer_cp.qkv_proj.weight.data  # (hidden_size * 3, hidden_size)
        qkv_bias = None
        if self.torch_layer_cp.qkv_proj.bias is not None:
            qkv_bias = self.torch_layer_cp.qkv_proj.bias.data

        # Split into Q, K, V
        q_weight = qkv_weight[: self.hidden_size, :].T  # (hidden_size, hidden_size)
        k_weight = qkv_weight[self.hidden_size : 2 * self.hidden_size, :].T
        v_weight = qkv_weight[2 * self.hidden_size :, :].T

        # Convert to TTNN
        self.q_weight = ttnn.from_torch(
            q_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.k_weight = ttnn.from_torch(
            k_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_weight = ttnn.from_torch(
            v_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if qkv_bias is not None:
            q_bias = qkv_bias[: self.hidden_size]
            k_bias = qkv_bias[self.hidden_size : 2 * self.hidden_size]
            v_bias = qkv_bias[2 * self.hidden_size :]

            self.q_bias = self.tensor_1d_to_2d_ttnn(q_bias)
            self.k_bias = self.tensor_1d_to_2d_ttnn(k_bias)
            self.v_bias = self.tensor_1d_to_2d_ttnn(v_bias)
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        # Output projection
        out_weight = self.torch_layer_cp.out_proj.weight.data.T  # (hidden_size, hidden_size)
        self.out_weight = ttnn.from_torch(
            out_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        out_bias = self.torch_layer_cp.out_proj.bias.data
        if out_bias is not None:
            self.out_bias = self.tensor_1d_to_2d_ttnn(out_bias)
        else:
            self.out_bias = None

    def tensor_1d_to_2d_ttnn(self, tensor_1d: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        self.q_weight = ttnn.to_device(self.q_weight, self.device)
        self.k_weight = ttnn.to_device(self.k_weight, self.device)
        self.v_weight = ttnn.to_device(self.v_weight, self.device)
        self.out_weight = ttnn.to_device(self.out_weight, self.device)

        if self.q_bias is not None and self.k_bias is not None and self.v_bias is not None:
            self.q_bias = ttnn.to_device(self.q_bias, self.device)
            self.k_bias = ttnn.to_device(self.k_bias, self.device)
            self.v_bias = ttnn.to_device(self.v_bias, self.device)
        if self.out_bias is not None:
            self.out_bias = ttnn.to_device(self.out_bias, self.device)

    def deallocate_weights_impl(self):
        """Deallocate device memory."""

        ttnn.deallocate(self.q_weight)
        ttnn.deallocate(self.k_weight)
        ttnn.deallocate(self.v_weight)
        ttnn.deallocate(self.out_weight)
        if self.q_bias is not None and self.k_bias is not None and self.v_bias is not None:
            ttnn.deallocate(self.q_bias)
            ttnn.deallocate(self.k_bias)
            ttnn.deallocate(self.v_bias)
        if self.out_bias is not None:
            ttnn.deallocate(self.out_bias)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of attention.

        Args:
            x: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        if isinstance(x, TorchTTNNTensor):
            x = x.to_ttnn
            x = ttnn.to_device(x, device=self.device)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

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

        try:
            # Configure SDPA (chunk sizes must be multiples of TILE_SIZE=32)
            TILE_SIZE = 32
            chunk_size = max(128, ((seqlen + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE)

            device_grid = self.device.compute_with_storage_grid_size()
            grid_x = min(8, device_grid.x)
            grid_y = min(8, device_grid.y)

            sdpa_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(grid_x, grid_y),
                q_chunk_size=chunk_size,
                k_chunk_size=chunk_size,
                exp_approx_mode=False,
            )

            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
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
        except RuntimeError:
            q_torch = ttnn.to_torch(query).float()
            k_torch = ttnn.to_torch(key).float()
            v_torch = ttnn.to_torch(value).float()
            attn_out_torch = torch.nn.functional.scaled_dot_product_attention(
                q_torch, k_torch, v_torch, scale=scale
            ).to(torch.bfloat16)
            attn_out_torch = attn_out_torch.permute(0, 2, 1, 3).reshape(bsz, seqlen, self.hidden_size)
            attention_output = ttnn.from_torch(
                attn_out_torch,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

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

        output = ttnn.to_torch(output)

        return output


########## No Tensor Parallelism Feed Forward ############
class TTNNNoTPFeedForward(TTNNModule):
    """
    No Tensor Parallelism Feed Forward using TTNN operations.

    Implements two linear layers with quick_gelu activation.
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden_dim: int = 4096,
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
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.torch_layer_cp = None

    @classmethod
    def from_torch(cls, NoTPFeedForward):
        """Create TTNN module from PyTorch equivalent."""
        new_TPFeedForward = cls(NoTPFeedForward.fc1.in_features, NoTPFeedForward.fc1.out_features)
        new_TPFeedForward.torch_layer_cp = NoTPFeedForward
        new_TPFeedForward._fallback_torch_layer = NoTPFeedForward
        return new_TPFeedForward

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        # FC1 weights
        fc1_weight = self.torch_layer_cp.fc1.weight.data.T  # (hidden_dim, dim)
        self.fc1_weight = ttnn.from_torch(
            fc1_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if self.torch_layer_cp.fc1.bias is not None:
            fc1_bias = self.torch_layer_cp.fc1.bias.data
            self.fc1_bias = self.tensor_1d_to_2d_ttnn(fc1_bias)
        else:
            self.fc1_bias = None

        # FC2 weights
        fc2_weight = self.torch_layer_cp.fc2.weight.data.T  # (dim, hidden_dim)
        self.fc2_weight = ttnn.from_torch(
            fc2_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if self.torch_layer_cp.fc2.bias is not None:
            fc2_bias = self.torch_layer_cp.fc2.bias.data
            self.fc2_bias = self.tensor_1d_to_2d_ttnn(fc2_bias)
        else:
            self.fc2_bias = None

    def tensor_1d_to_2d_ttnn(self, tensor_1d: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""

        self.fc1_weight = ttnn.to_device(self.fc1_weight, self.device)
        self.fc2_weight = ttnn.to_device(self.fc2_weight, self.device)
        if self.fc1_bias is not None:
            self.fc1_bias = ttnn.to_device(self.fc1_bias, self.device)
        if self.fc2_bias is not None:
            self.fc2_bias = ttnn.to_device(self.fc2_bias, self.device)

    def deallocate_weights_impl(self):
        """Deallocate device memory."""

        ttnn.deallocate(self.fc1_weight)
        ttnn.deallocate(self.fc2_weight)
        if self.fc1_bias is not None:
            ttnn.deallocate(self.fc1_bias)
        if self.fc2_bias is not None:
            ttnn.deallocate(self.fc2_bias)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of feedforward layer.

        Args:
            x: TTNN tensor (batch_size, seq_len, dim)

        Returns:
            TTNN tensor (batch_size, seq_len, dim)
        """
        if isinstance(x, TorchTTNNTensor):
            x = x.to_ttnn
            x = ttnn.to_device(x, device=self.device)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # FC1
        output = ttnn.linear(
            x,
            self.fc1_weight,
            bias=self.fc1_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Quick GELU
        output = self.quick_gelu_ttnn(output)

        # FC2
        output = ttnn.linear(
            output,
            self.fc2_weight,
            bias=self.fc2_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        output = ttnn.to_torch(output)
        return output

    def quick_gelu_ttnn(self, x: ttnn.Tensor) -> ttnn.Tensor:
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


########## No Tensor Parallelism Transformer Block ############
class TTNNNoTPTransformerBlock(TTNNModule):
    """
    No Tensor Parallelism Transformer Block using TTNN operations.

    Implements pre-norm transformer block with attention and feedforward.
    """

    def __init__(
        self,
        n_heads,
        dim,
        head_dim,
        layer_id: int,
    ):
        """
        Initialize transformer block.

        Args:
            cfg: Configuration dict
            layer_id: Layer index
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = head_dim
        self.layer_id = layer_id
        self.layernorm_epsilon = 1e-5
        self.torch_layer_cp = None
        self.self_attn = None
        self.mlp = None

    @classmethod
    def from_torch(cls, NoTPTransformerBlock):
        """Create TTNN module from PyTorch equivalent."""
        new_NoTPTransformerBlock = cls(
            NoTPTransformerBlock.n_heads,
            NoTPTransformerBlock.dim,
            NoTPTransformerBlock.head_dim,
            NoTPTransformerBlock.layer_id,
        )
        new_NoTPTransformerBlock.self_attn = TTNNNoTPAttention.from_torch(NoTPTransformerBlock.self_attn)
        new_NoTPTransformerBlock.mlp = TTNNNoTPFeedForward.from_torch(NoTPTransformerBlock.mlp)

        new_NoTPTransformerBlock.torch_layer_cp = NoTPTransformerBlock
        new_NoTPTransformerBlock._fallback_torch_layer = NoTPTransformerBlock
        return new_NoTPTransformerBlock

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        ln1_weight = self.torch_layer_cp.layer_norm1.weight.data
        ln1_bias = self.torch_layer_cp.layer_norm1.bias.data
        ln2_weight = self.torch_layer_cp.layer_norm2.weight.data
        ln2_bias = self.torch_layer_cp.layer_norm2.bias.data

        self.layer_norm1_weight = ttnn.from_torch(
            ln1_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layer_norm1_bias = ttnn.from_torch(
            ln1_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layer_norm2_weight = ttnn.from_torch(
            ln2_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layer_norm2_bias = ttnn.from_torch(
            ln2_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.self_attn.preprocess_weights_impl()
        self.mlp.preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        self.layer_norm1_weight = ttnn.to_device(self.layer_norm1_weight, self.device)
        self.layer_norm1_bias = ttnn.to_device(self.layer_norm1_bias, self.device)
        self.layer_norm2_weight = ttnn.to_device(self.layer_norm2_weight, self.device)
        self.layer_norm2_bias = ttnn.to_device(self.layer_norm2_bias, self.device)
        self.self_attn.move_weights_to_device_impl()
        self.mlp.move_weights_to_device_impl()

    def deallocate_weights_impl(self):
        """Deallocate device memory."""
        ttnn.deallocate(self.layer_norm1_weight)
        ttnn.deallocate(self.layer_norm1_bias)
        ttnn.deallocate(self.layer_norm2_weight)
        ttnn.deallocate(self.layer_norm2_bias)
        self.self_attn.deallocate_weights_impl()
        self.mlp.deallocate_weights_impl()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of transformer block.

        Args:
            x: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        if isinstance(x, TorchTTNNTensor):
            x = x.to_ttnn
            x = ttnn.to_device(x, device=self.device)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        if isinstance(x, torch.Tensor):
            x = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Pre-norm attention
        residual = ttnn.layer_norm(
            x,
            weight=self.layer_norm1_weight,
            bias=self.layer_norm1_bias,
            epsilon=self.layernorm_epsilon,
        )

        residual = self.self_attn.forward(residual)
        if isinstance(residual, torch.Tensor):
            residual = ttnn.from_torch(
                residual,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

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
        if isinstance(out, torch.Tensor):
            out = ttnn.from_torch(
                out,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        out = ttnn.add(h, out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)

        out = ttnn.to_torch(out)
        return out


########## No Tensor Parallelism Transformer ############
class TTNNNoTPTransformer(TTNNModule):
    """
    No Tensor Parallelism Transformer using TTNN operations.

    Stack of transformer blocks.
    """

    def __init__(
        self,
        cfg,
        num_layers,
    ):
        """
        Initialize transformer.

        Args:
            cfg: Configuration dict
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        super().__init__()

        self.cfg = cfg
        self.num_layers = num_layers
        self.torch_layer_cp = None
        self.layers = []

    @classmethod
    def from_torch(cls, NoTPTransformer):
        """Create TTNN module from PyTorch equivalent."""
        new_NoTPTransformer = cls(NoTPTransformer.cfg, NoTPTransformer.num_layers)

        for layer_id in range(new_NoTPTransformer.num_layers):
            layer = TTNNNoTPTransformerBlock.from_torch(NoTPTransformer.layers[layer_id])
            new_NoTPTransformer.layers.append(layer)

        new_NoTPTransformer.torch_layer_cp = NoTPTransformer
        new_NoTPTransformer._fallback_torch_layer = NoTPTransformer
        return new_NoTPTransformer

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        for layer in self.layers:
            layer.preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        for layer in self.layers:
            layer.move_weights_to_device_impl()

    def deallocate_weights_impl(self):
        """Deallocate device memory."""
        for layer in self.layers:
            layer.deallocate_weights_impl()

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of transformer.

        Args:
            hidden_states: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        if isinstance(hidden_states, TorchTTNNTensor):
            hidden_states = hidden_states.to_ttnn
            hidden_states = ttnn.to_device(hidden_states, device=self.device)
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        for layer in self.layers:
            hidden_states = layer.forward(hidden_states)

        return hidden_states


########## VitModel (VISION MODEL) ############
class TTNNVitModel(TTNNModule):
    """
    Vision Transformer Model using TTNN operations.

    Complete ViT model with embeddings, pre-norm, and transformer encoder.
    """

    def __init__(
        self,
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
        super().__init__()
        self.torch_layer_cp = None
        self.embeddings = None
        self.transformer = None
        self.pre_layernorm_epsilon = 1e-5

    @classmethod
    def from_torch(cls, VitModel):
        """Create TTNN module from PyTorch equivalent."""
        new_VitModel = cls()

        new_VitModel.embeddings = TTNNClipVisionEmbeddings.from_torch(VitModel.embeddings)
        new_VitModel.transformer = TTNNNoTPTransformer.from_torch(VitModel.transformer)

        new_VitModel.torch_layer_cp = VitModel
        new_VitModel._fallback_torch_layer = VitModel
        return new_VitModel

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        pre_norm_weight = self.torch_layer_cp.pre_layrnorm.weight.data
        pre_norm_bias = self.torch_layer_cp.pre_layrnorm.bias.data

        self.pre_layrnorm_weight = ttnn.from_torch(
            pre_norm_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.pre_layrnorm_bias = ttnn.from_torch(
            pre_norm_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.embeddings.preprocess_weights_impl()
        self.transformer.preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        self.pre_layrnorm_weight = ttnn.to_device(self.pre_layrnorm_weight, self.device)
        self.pre_layrnorm_bias = ttnn.to_device(self.pre_layrnorm_bias, self.device)
        self.embeddings.move_weights_to_device_impl()
        self.transformer.move_weights_to_device_impl()

    def deallocate_weights_impl(self):
        """Deallocate device memory."""
        ttnn.deallocate(self.pre_layrnorm_weight)
        ttnn.deallocate(self.pre_layrnorm_bias)
        self.embeddings.deallocate_weights_impl()
        self.transformer.deallocate_weights_impl()

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
        if isinstance(x, torch.Tensor):
            x = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        if isinstance(patch_embeds, torch.Tensor):
            patch_embeds = ttnn.from_torch(
                patch_embeds,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if patch_embeds is not None and patch_embeds.layout != ttnn.TILE_LAYOUT:
            patch_embeds = ttnn.to_layout(patch_embeds, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if patch_embeds is not None and len(patch_embeds.shape) == 4:
            patch_embeds = ttnn.reshape(patch_embeds, shape=[patch_embeds.shape[0], patch_embeds.shape[1], -1])
            patch_embeds = ttnn.transpose(patch_embeds, 1, 2)

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
