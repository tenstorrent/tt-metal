# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SigLIP Vision Tower - TTNN Implementation (Optimized).

This module implements the SigLIP vision encoder using TTNN operations.

SigLIP Architecture:
    - Patch embedding (Unfold + TTNN linear - optimized)
    - Positional embedding (learned)
    - Transformer encoder blocks (fused QKV, native head operations)
    - Multi-modal projector (linear to match language model dimension)

Optimizations over baseline:
    1. Unfold + TTNN linear for patch embedding (from Gemma3)
    2. Fused QKV projection (single linear instead of 3)
    3. Native ttnn.experimental.nlp_create_qkv_heads (no PyTorch transfers)
    4. Native ttnn.experimental.nlp_concat_heads for output
"""

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
import ttnn

from models.experimental.pi0.common.configs import SigLIPConfig


# ============================================================================
# Helper Functions
# ============================================================================


def nearest_32(x: int) -> int:
    """Round up to nearest multiple of 32 for TTNN tile alignment."""
    return ((x + 31) // 32) * 32


# ============================================================================
# Patch Embedding (TTNN - Optimized)
# ============================================================================


class PatchEmbeddingTTNN:
    """
    Convert image patches to embeddings using Unfold + TTNN linear.

    OPTIMIZED: Uses torch.nn.Unfold for patch extraction, then TTNN linear.
    This approach (from Gemma3) uses optimized TTNN linear instead of PyTorch conv2d.
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize patch embedding with TTNN weights.

        Args:
            config: SigLIP configuration
            weights: PyTorch weights for conv2d (will be converted to linear format)
            device: TTNN device
        """
        self.config = config
        self.device = device
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size

        # Create unfold operation (runs on host, very fast)
        self._unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        # Handle both formats: vision_model.embeddings.patch_embedding (checkpoint) and patch_embedding (legacy)
        conv_weight = weights.get("patch_embedding.weight") or weights.get(
            "vision_model.embeddings.patch_embedding.weight"
        )
        conv_bias = weights.get("patch_embedding.bias") or weights.get("vision_model.embeddings.patch_embedding.bias")

        # Convert conv2d weight to linear format
        # Conv weight: (out_channels, in_channels, kernel_h, kernel_w) = (hidden_size, 3, patch_size, patch_size)
        # Linear weight: (in_features, out_features) where in_features = 3 * patch_size * patch_size
        out_channels = conv_weight.shape[0]  # hidden_size
        in_features = conv_weight.shape[1] * conv_weight.shape[2] * conv_weight.shape[3]  # 3 * 14 * 14 = 588

        # Reshape: (out, in, h, w) -> (out, in*h*w) -> transpose -> (in*h*w, out)
        linear_weight = conv_weight.view(out_channels, -1)  # (hidden_size, 588)

        # Pad input dimension to tile-aligned (588 -> 608)
        self.in_features_padded = nearest_32(in_features)
        pad_len = self.in_features_padded - in_features

        if pad_len > 0:
            padding = torch.zeros(out_channels, pad_len, dtype=linear_weight.dtype)
            linear_weight = torch.cat([linear_weight, padding], dim=-1)

        # Transpose for TTNN linear: (hidden_size, padded_in) -> (padded_in, hidden_size)
        linear_weight = linear_weight.T.contiguous()

        # Store as TTNN tensor on device
        self._linear_weight = ttnn.from_torch(
            linear_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Bias (if present)
        if conv_bias is not None:
            self._linear_bias = ttnn.from_torch(
                conv_bias.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self._linear_bias = None

        # Compute kernel config
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, pixel_values) -> ttnn.Tensor:
        """
        OPTIMIZED: Extract patch embeddings using Unfold + TTNN linear.

        Args:
            pixel_values: PyTorch tensor (batch_size, channels, height, width)

        Returns:
            TTNN tensor (batch_size, num_patches, hidden_size)
        """
        # Convert to PyTorch if needed (shouldn't happen in normal flow)
        if isinstance(pixel_values, ttnn.Tensor):
            pixel_values = ttnn.to_torch(pixel_values)

        batch_size = pixel_values.shape[0]

        # Step 1: Unfold on host (very fast - just memory reorganization)
        # Input: (B, C, H, W) -> Output: (B, C*patch_size*patch_size, num_patches)
        x = self._unfold(pixel_values)

        # Step 2: Permute to (B, num_patches, C*patch_size*patch_size)
        x = x.permute(0, 2, 1)

        # Step 3: Pad to tile-aligned dimension
        in_features = x.shape[-1]
        pad_len = self.in_features_padded - in_features

        if pad_len > 0:
            padding = torch.zeros((batch_size, x.shape[1], pad_len), dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=-1)

        # Step 4: Transfer to device
        x_ttnn = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Step 5: TTNN linear (optimized on device)
        out = ttnn.linear(
            x_ttnn,
            self._linear_weight,
            bias=self._linear_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        ttnn.deallocate(x_ttnn)

        return out


# ============================================================================
# Vision Transformer Block (TTNN - Optimized)
# ============================================================================


class SigLIPAttentionTTNN:
    """
    SigLIP self-attention using TTNN operations.

    OPTIMIZED: Uses fused QKV projection (single linear) and native TTNN head operations.
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize attention with TTNN weights.

        Args:
            config: SigLIP configuration
            weights: PyTorch weights to convert
            device: TTNN device
        """
        self.config = config
        self.device = device
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Pad head_dim to multiple of 32 for TTNN tile alignment
        self.padded_head_dim = ((self.head_dim + 31) // 32) * 32  # 72 -> 96

        # Helper function to pad weights to tile-aligned head_dim
        def pad_head_dim_weight(weight, heads_out=True):
            """Pad weight tensor's head dimension to multiple of 32."""
            dim = weight.shape[0]  # hidden_size
            padded_head_dim = self.padded_head_dim
            padding_size = padded_head_dim - self.head_dim

            if padding_size > 0:
                if heads_out:
                    weight = weight.T  # (hidden, hidden) -> transpose for reshape
                weight = weight.reshape(dim, self.num_heads, self.head_dim)
                padding = torch.zeros(dim, self.num_heads, padding_size, dtype=weight.dtype)
                weight = torch.cat([weight, padding], dim=-1)
                weight = weight.reshape(dim, self.num_heads * padded_head_dim)
                if heads_out:
                    weight = weight.T  # Transpose back
            return weight

        def pad_head_dim_bias(bias):
            """Pad 1D bias to match padded head dim."""
            padded_head_dim = self.padded_head_dim
            padding_size = padded_head_dim - self.head_dim

            if padding_size > 0:
                bias = bias.view(self.num_heads, self.head_dim)
                padding = torch.zeros(self.num_heads, padding_size, dtype=bias.dtype)
                bias = torch.cat([bias, padding], dim=-1)
                bias = bias.view(self.num_heads * padded_head_dim)
            return bias

        # OPTIMIZATION: Fused QKV weights - single linear instead of 3
        # Pad each weight, transpose for TTNN linear, then concatenate
        wq_padded = pad_head_dim_weight(weights["self_attn.q_proj.weight"])
        wk_padded = pad_head_dim_weight(weights["self_attn.k_proj.weight"])
        wv_padded = pad_head_dim_weight(weights["self_attn.v_proj.weight"])

        # Concatenate Q, K, V weights: [hidden, 3 * num_heads * padded_head_dim]
        wqkv_combined = torch.cat([wq_padded.T, wk_padded.T, wv_padded.T], dim=-1)

        self.wqkv = ttnn.from_torch(
            wqkv_combined.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Fused QKV biases
        if "self_attn.q_proj.bias" in weights:
            bq_padded = pad_head_dim_bias(weights["self_attn.q_proj.bias"])
            bk_padded = pad_head_dim_bias(weights["self_attn.k_proj.bias"])
            bv_padded = pad_head_dim_bias(weights["self_attn.v_proj.bias"])

            # Concatenate biases
            bqkv_combined = torch.cat([bq_padded, bk_padded, bv_padded], dim=-1)

            self.bqkv = ttnn.from_torch(
                bqkv_combined.unsqueeze(0).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.bqkv = None

        # Output projection - pad input head dim, output is hidden_size
        wo_padded = pad_head_dim_weight(weights["self_attn.out_proj.weight"], heads_out=False)
        self.wo = ttnn.from_torch(
            wo_padded.T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if "self_attn.out_proj.bias" in weights:
            self.bo = ttnn.from_torch(
                weights["self_attn.out_proj.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.bo = None

        # Compute kernel configs
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_sdpa = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,  # SDPA needs this off
        )

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        OPTIMIZED forward pass using fused QKV and native TTNN head operations.

        Key optimizations:
        1. Single fused QKV linear (3x fewer linear ops)
        2. Native ttnn.experimental.nlp_create_qkv_heads (no PyTorch transfers)
        3. Native ttnn.experimental.nlp_concat_heads (no PyTorch transfers)

        Args:
            hidden_states: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Reshape to 4D for nlp_create_qkv_heads: [batch, 1, seq, hidden]
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, seq_len, -1))

        # OPTIMIZATION 1: Single fused QKV linear (instead of 3 separate)
        # Output: [batch, 1, seq, 3 * num_heads * padded_head_dim]
        xqkv_fused = ttnn.linear(
            hidden_states,
            self.wqkv,
            bias=self.bqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        # OPTIMIZATION 2: Native TTNN head splitting (no PyTorch transfers!)
        # This splits the fused QKV into separate Q, K, V with proper head layout
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,  # SigLIP uses MHA, not MQA
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)

        # SDPA configuration
        device_grid = self.device.compute_with_storage_grid_size()
        grid_x = min(8, device_grid.x)
        grid_y = min(8, device_grid.y)

        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            q_chunk_size=min(256, seq_len),
            k_chunk_size=min(256, seq_len),
            exp_approx_mode=False,
        )

        # SDPA - stays entirely on device
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            scale=self.scale,
            program_config=sdpa_cfg,
            compute_kernel_config=self.compute_kernel_config_sdpa,
        )

        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # OPTIMIZATION 3: Native TTNN head concatenation (no PyTorch transfers!)
        # This concatenates heads back to [batch, 1, seq, num_heads * padded_head_dim]
        attn_concat = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output)

        # Output projection
        output = ttnn.linear(
            attn_concat,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        ttnn.deallocate(attn_concat)

        # Add bias if present
        if self.bo is not None:
            output = ttnn.add(output, self.bo)

        # Reshape back to 3D: [batch, 1, seq, hidden] -> [batch, seq, hidden]
        output = ttnn.reshape(output, (batch_size, seq_len, self.hidden_size))

        return output


class SigLIPMLPTTNN:
    """
    SigLIP MLP with GELU activation using TTNN.
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize MLP with TTNN weights.

        Args:
            config: SigLIP configuration
            weights: PyTorch weights to convert
            device: TTNN device
        """
        self.config = config
        self.device = device

        # FC1 (input -> intermediate)
        fc1_weight = weights["mlp.fc1.weight"].T.contiguous()
        self.fc1_weight = ttnn.from_torch(
            fc1_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if "mlp.fc1.bias" in weights:
            self.fc1_bias = ttnn.from_torch(
                weights["mlp.fc1.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.fc1_bias = None

        # FC2 (intermediate -> output)
        fc2_weight = weights["mlp.fc2.weight"].T.contiguous()
        self.fc2_weight = ttnn.from_torch(
            fc2_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if "mlp.fc2.bias" in weights:
            self.fc2_bias = ttnn.from_torch(
                weights["mlp.fc2.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.fc2_bias = None

        # Compute kernel config
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass using TTNN operations.

        Args:
            hidden_states: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        # FC1 with GELU activation
        x = ttnn.linear(
            hidden_states,
            self.fc1_weight,
            bias=self.fc1_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            activation="gelu",
        )

        # FC2
        output = ttnn.linear(
            x,
            self.fc2_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(x)

        # Add bias if present
        if self.fc2_bias is not None:
            output = ttnn.add(output, self.fc2_bias)

        return output


class SigLIPBlockTTNN:
    """
    Complete SigLIP transformer block using TTNN.
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize block with TTNN weights.

        Args:
            config: SigLIP configuration
            weights: PyTorch weights to convert
            device: TTNN device
        """
        self.config = config
        self.device = device

        # Layer norms
        self.ln1_weight = ttnn.from_torch(
            weights["layer_norm1.weight"].reshape(1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if "layer_norm1.bias" in weights:
            self.ln1_bias = ttnn.from_torch(
                weights["layer_norm1.bias"].reshape(1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.ln1_bias = None

        self.ln2_weight = ttnn.from_torch(
            weights["layer_norm2.weight"].reshape(1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if "layer_norm2.bias" in weights:
            self.ln2_bias = ttnn.from_torch(
                weights["layer_norm2.bias"].reshape(1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.ln2_bias = None

        # Attention and MLP - using native TTNN with padded head dim workaround
        self.attention = SigLIPAttentionTTNN(config, weights, device)
        self.mlp = SigLIPMLPTTNN(config, weights, device)

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass using native TTNN operations.

        Args:
            hidden_states: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        # Pre-attention LayerNorm
        normed = ttnn.layer_norm(
            hidden_states,
            weight=self.ln1_weight,
            bias=self.ln1_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Native TTNN attention with padded head dim workaround
        attn_output = self.attention.forward(normed)
        ttnn.deallocate(normed)

        # Residual connection
        hidden_states = ttnn.add(hidden_states, attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_output)

        # Pre-MLP LayerNorm
        normed = ttnn.layer_norm(
            hidden_states,
            weight=self.ln2_weight,
            bias=self.ln2_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # MLP with residual
        mlp_output = self.mlp.forward(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, mlp_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_output)

        return hidden_states


# ============================================================================
# Full Vision Tower (TTNN)
# ============================================================================


class SigLIPVisionTowerTTNN:
    """
    Complete SigLIP vision tower using TTNN operations.

    Fully implemented in TTNN:
        - Patch embedding on host (Unfold) + device (TTNN linear)
        - Position embedding addition on device
        - All transformer blocks on device (TTNN)
        - Final layer norm on device
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize vision tower.

        Args:
            config: SigLIP configuration
            weights: PyTorch weights (will be converted)
            device: TTNN device
        """
        self.config = config
        self.device = device

        # Patch embedding (Unfold + TTNN linear)
        self.patch_embed = PatchEmbeddingTTNN(config, weights, device)

        # Position embedding on device (handle both formats)
        pos_emb = weights.get("position_embedding.weight") or weights.get(
            "vision_model.embeddings.position_embedding.weight"
        )

        if pos_emb is not None:
            # Calculate target number of patches based on config
            num_patches = (config.image_size // config.patch_size) ** 2

            # Check if we need to interpolate position embeddings
            if pos_emb.shape[0] != num_patches:
                original_num_patches = pos_emb.shape[0]
                original_size = int(math.sqrt(original_num_patches))
                target_size = int(math.sqrt(num_patches))

                # Reshape to 2D grid: (num_patches, hidden_size) -> (1, H, W, hidden_size)
                pos_emb_2d = pos_emb.view(1, original_size, original_size, -1)
                pos_emb_2d = pos_emb_2d.permute(0, 3, 1, 2)  # (1, hidden_size, H, W)

                # Interpolate using bicubic
                pos_emb_interpolated = F.interpolate(
                    pos_emb_2d,
                    size=(target_size, target_size),
                    mode="bicubic",
                    align_corners=False,
                )

                # Reshape back: (1, hidden_size, H, W) -> (num_patches, hidden_size)
                pos_emb = pos_emb_interpolated.permute(0, 2, 3, 1).flatten(0, 2)

            # Create position IDs
            self.position_ids = ttnn.arange(0, num_patches, 1, dtype=ttnn.uint32, device=device)
            self.position_ids = ttnn.reshape(self.position_ids, (1, -1))

            # Load position embedding weights
            self.pos_emb_weights = ttnn.as_tensor(
                pos_emb,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.position_ids = None
            self.pos_emb_weights = None

        # Initialize TTNN transformer blocks
        self.blocks = []
        for i in range(config.num_hidden_layers):
            block_weights = self._get_layer_weights(weights, i)
            self.blocks.append(SigLIPBlockTTNN(config, block_weights, device))

        # Final layer norm weights (handle both formats)
        post_ln_weight = weights.get("post_layernorm.weight") or weights.get("vision_model.post_layernorm.weight")
        post_ln_bias = weights.get("post_layernorm.bias") or weights.get("vision_model.post_layernorm.bias")

        if post_ln_weight is not None:
            self.post_ln_weight = ttnn.from_torch(
                post_ln_weight.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.post_ln_bias = (
                ttnn.from_torch(
                    post_ln_bias.unsqueeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if post_ln_bias is not None
                else None
            )
        else:
            self.post_ln_weight = None
            self.post_ln_bias = None

    def _get_layer_weights(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Extract weights for a specific layer."""
        # Handle both formats: vision_model.encoder.layers.X (checkpoint) and encoder.layers.X (legacy)
        prefixes = [f"vision_model.encoder.layers.{layer_idx}.", f"encoder.layers.{layer_idx}."]
        layer_weights = {}
        for prefix in prefixes:
            for key, value in weights.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    layer_weights[new_key] = value
        return layer_weights

    def forward(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """
        Process images to embeddings (TTNN).

        Args:
            pixel_values: PyTorch tensor (batch_size, channels, height, width)

        Returns:
            TTNN tensor (batch_size, num_patches, hidden_size)
        """
        # Patch embedding (hybrid - Unfold on host, linear on device)
        hidden_states = self.patch_embed.forward(pixel_values)

        # Add position embeddings (on device)
        if self.pos_emb_weights is not None:
            num_patches_actual = hidden_states.shape[1]
            num_patches_expected = self.position_ids.shape[1]

            # Check if we need to interpolate position embeddings dynamically
            if num_patches_actual != num_patches_expected:
                # Convert position embeddings back to torch for interpolation
                pos_emb_torch = ttnn.to_torch(self.pos_emb_weights)

                original_size = int(math.sqrt(num_patches_expected))
                target_size = int(math.sqrt(num_patches_actual))
                hidden_size = pos_emb_torch.shape[1]

                # Reshape and interpolate
                pos_emb_2d = pos_emb_torch.view(1, original_size, original_size, hidden_size)
                pos_emb_2d = pos_emb_2d.permute(0, 3, 1, 2)

                pos_emb_interpolated = F.interpolate(
                    pos_emb_2d,
                    size=(target_size, target_size),
                    mode="bicubic",
                    align_corners=False,
                )

                pos_emb_resized = pos_emb_interpolated.permute(0, 2, 3, 1).flatten(0, 2)

                # Create new position IDs for actual number of patches
                position_ids_new = ttnn.arange(0, num_patches_actual, 1, dtype=ttnn.uint32, device=self.device)
                position_ids_new = ttnn.reshape(position_ids_new, (1, -1))

                # Convert resized embeddings to TTNN
                pos_emb_weights_new = ttnn.as_tensor(
                    pos_emb_resized,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                # Use ttnn.embedding with resized weights
                positional_embeddings = ttnn.embedding(
                    position_ids_new,
                    pos_emb_weights_new,
                    layout=ttnn.TILE_LAYOUT,
                )
            else:
                # Use pre-loaded position embeddings
                positional_embeddings = ttnn.embedding(
                    self.position_ids,
                    self.pos_emb_weights,
                    layout=ttnn.TILE_LAYOUT,
                )

            hidden_states = ttnn.add(hidden_states, positional_embeddings)

        # Run through TTNN transformer blocks
        for block in self.blocks:
            hidden_states = block.forward(hidden_states)
            ttnn.ReadDeviceProfiler(self.device)  # Clear device profiler buffer

        # Final layer norm (on device)
        if self.post_ln_weight is not None:
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=self.post_ln_weight,
                bias=self.post_ln_bias,
                epsilon=self.config.layer_norm_eps,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        return hidden_states


# ============================================================================
# Multi-modal Projector (TTNN)
# ============================================================================


class MultiModalProjectorTTNN:
    """
    Projects vision features to language model dimension using TTNN.
    """

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize projector with TTNN weights.

        Args:
            weights: PyTorch weights to convert
            device: TTNN device
        """
        self.device = device

        # Convert weight to TTNN format (transposed)
        self.weight = ttnn.from_torch(
            weights["linear.weight"].T.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if "linear.bias" in weights:
            self.bias = ttnn.from_torch(
                weights["linear.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.bias = None

    def forward(self, vision_features: ttnn.Tensor) -> ttnn.Tensor:
        """
        Project vision features using TTNN linear.

        Args:
            vision_features: TTNN tensor (batch_size, num_patches, vision_hidden_size)

        Returns:
            TTNN tensor (batch_size, num_patches, language_hidden_size)
        """
        return ttnn.linear(
            vision_features,
            self.weight,
            bias=self.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


# Default exports
PatchEmbedding = PatchEmbeddingTTNN
SigLIPAttention = SigLIPAttentionTTNN
SigLIPMLP = SigLIPMLPTTNN
SigLIPBlock = SigLIPBlockTTNN
SigLIPVisionTower = SigLIPVisionTowerTTNN
MultiModalProjector = MultiModalProjectorTTNN
