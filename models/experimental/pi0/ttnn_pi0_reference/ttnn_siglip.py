# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SigLIP Vision Tower for TTNN PI0 implementation.

This module implements the SigLIP vision encoder that processes images
into feature embeddings for the VLM backbone.

SigLIP Architecture:
    - Patch embedding (conv2d to extract patches)
    - Positional embedding (learned)
    - Transformer encoder blocks
    - Multi-modal projector (linear to match language model dimension)

The TTNN implementation leverages existing optimized kernels from:
    - models/demos/siglip/tt/attention.py (TtLlamaImageAttention)
    - models/demos/grayskull/vit/tt/ttnn_optimized_vit_highres_gs.py
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None


@dataclass
class SigLIPConfig:
    """Configuration for SigLIP vision encoder."""
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-6
    
    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# ============================================================================
# Patch Embedding
# ============================================================================

class PatchEmbeddingTorch:
    """
    Convert image patches to embeddings (PyTorch).
    
    Uses Conv2d with kernel_size = patch_size to extract non-overlapping patches.
    """
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize patch embedding.
        
        Args:
            config: SigLIP configuration
            weights: Dictionary with:
                - patch_embedding.weight: (hidden_size, channels, patch_size, patch_size)
                - patch_embedding.bias: (hidden_size,)
        """
        self.config = config
        self.conv_weight = weights.get("patch_embedding.weight")
        self.conv_bias = weights.get("patch_embedding.bias")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract patch embeddings from images.
        
        Args:
            pixel_values: (batch_size, channels, height, width)
        
        Returns:
            (batch_size, num_patches, hidden_size)
        """
        batch_size = pixel_values.shape[0]
        
        # Apply convolution
        x = F.conv2d(
            pixel_values,
            self.conv_weight,
            self.conv_bias,
            stride=self.config.patch_size,
        )
        
        # Reshape: (B, C, H, W) -> (B, num_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)
        
        return x


class PatchEmbeddingTTNN:
    """
    Convert image patches to embeddings using TTNN.
    
    Note: Conv2d is performed on host (PyTorch) as TTNN conv2d
    may not be optimal for this use case. The result is then
    transferred to device for subsequent operations.
    """
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: "ttnn.Device",
    ):
        """
        Initialize patch embedding.
        
        Args:
            config: SigLIP configuration
            weights: PyTorch weights (kept on host for conv2d)
            device: TTNN device for output
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
        self.config = config
        self.device = device
        self.conv_weight = weights.get("patch_embedding.weight")
        self.conv_bias = weights.get("patch_embedding.bias")
    
    def forward(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        """
        Extract patch embeddings (hybrid CPU + device).
        
        Args:
            pixel_values: PyTorch tensor (batch_size, channels, height, width)
        
        Returns:
            TTNN tensor (batch_size, num_patches, hidden_size)
        """
        # Conv2d on host
        x = F.conv2d(
            pixel_values,
            self.conv_weight,
            self.conv_bias,
            stride=self.config.patch_size,
        )
        x = x.flatten(2).transpose(1, 2)
        
        # Transfer to device
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


# ============================================================================
# Vision Transformer Block
# ============================================================================

class SigLIPAttentionTorch:
    """
    SigLIP self-attention (PyTorch).
    
    Standard multi-head attention without rotary embeddings.
    """
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize attention.
        
        Args:
            config: SigLIP configuration
            weights: Attention weights
        """
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = weights["self_attn.q_proj.weight"]
        self.k_proj = weights["self_attn.k_proj.weight"]
        self.v_proj = weights["self_attn.v_proj.weight"]
        self.out_proj = weights["self_attn.out_proj.weight"]
        
        self.q_bias = weights.get("self_attn.q_proj.bias")
        self.k_bias = weights.get("self_attn.k_proj.bias")
        self.v_bias = weights.get("self_attn.v_proj.bias")
        self.out_bias = weights.get("self_attn.out_proj.bias")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        
        Returns:
            (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projections
        q = F.linear(hidden_states, self.q_proj, self.q_bias)
        k = F.linear(hidden_states, self.k_proj, self.k_bias)
        v = F.linear(hidden_states, self.v_proj, self.v_bias)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return F.linear(attn_output, self.out_proj, self.out_bias)


class SigLIPMLPTorch:
    """
    SigLIP MLP with GELU activation (PyTorch).
    """
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize MLP.
        
        Args:
            config: SigLIP configuration
            weights: MLP weights
        """
        self.fc1_weight = weights["mlp.fc1.weight"]
        self.fc1_bias = weights.get("mlp.fc1.bias")
        self.fc2_weight = weights["mlp.fc2.weight"]
        self.fc2_bias = weights.get("mlp.fc2.bias")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        
        Returns:
            (batch_size, seq_len, hidden_size)
        """
        x = F.linear(hidden_states, self.fc1_weight, self.fc1_bias)
        x = F.gelu(x, approximate="tanh")
        return F.linear(x, self.fc2_weight, self.fc2_bias)


class SigLIPBlockTorch:
    """
    Complete SigLIP transformer block (PyTorch).
    
    Architecture: Pre-LN
        x -> LayerNorm -> Attention -> + -> LayerNorm -> MLP -> +
    """
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize block.
        
        Args:
            config: SigLIP configuration
            weights: Block weights
        """
        self.config = config
        
        self.ln1_weight = weights["layer_norm1.weight"]
        self.ln1_bias = weights.get("layer_norm1.bias")
        self.ln2_weight = weights["layer_norm2.weight"]
        self.ln2_bias = weights.get("layer_norm2.bias")
        
        self.attention = SigLIPAttentionTorch(config, weights)
        self.mlp = SigLIPMLPTorch(config, weights)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        
        Returns:
            (batch_size, seq_len, hidden_size)
        """
        # Pre-attention norm
        normed = F.layer_norm(
            hidden_states,
            (self.config.hidden_size,),
            self.ln1_weight,
            self.ln1_bias,
            self.config.layer_norm_eps,
        )
        
        # Attention with residual
        hidden_states = hidden_states + self.attention.forward(normed)
        
        # Pre-MLP norm
        normed = F.layer_norm(
            hidden_states,
            (self.config.hidden_size,),
            self.ln2_weight,
            self.ln2_bias,
            self.config.layer_norm_eps,
        )
        
        # MLP with residual
        hidden_states = hidden_states + self.mlp.forward(normed)
        
        return hidden_states


# ============================================================================
# Full Vision Tower
# ============================================================================

class SigLIPVisionTowerTorch:
    """
    Complete SigLIP vision tower (PyTorch).
    
    Processes images into embeddings for the VLM backbone.
    """
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize vision tower.
        
        Args:
            config: SigLIP configuration
            weights: All vision tower weights
        """
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbeddingTorch(config, weights)
        
        # Position embedding
        self.position_embedding = weights["position_embedding.weight"]
        
        # Encoder blocks
        self.blocks = []
        for i in range(config.num_hidden_layers):
            block_weights = self._get_layer_weights(weights, i)
            self.blocks.append(SigLIPBlockTorch(config, block_weights))
        
        # Final layer norm
        self.post_layernorm_weight = weights.get("post_layernorm.weight")
        self.post_layernorm_bias = weights.get("post_layernorm.bias")
    
    def _get_layer_weights(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Extract weights for a specific layer."""
        prefix = f"encoder.layers.{layer_idx}."
        layer_weights = {}
        for key, value in weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                layer_weights[new_key] = value
        return layer_weights
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Process images to embeddings.
        
        Args:
            pixel_values: (batch_size, channels, height, width)
        
        Returns:
            (batch_size, num_patches, hidden_size)
        """
        # Patch embedding
        hidden_states = self.patch_embed.forward(pixel_values)
        
        # Add position embeddings
        hidden_states = hidden_states + self.position_embedding
        
        # Encoder blocks
        for block in self.blocks:
            hidden_states = block.forward(hidden_states)
        
        # Final layer norm
        if self.post_layernorm_weight is not None:
            hidden_states = F.layer_norm(
                hidden_states,
                (self.config.hidden_size,),
                self.post_layernorm_weight,
                self.post_layernorm_bias,
                self.config.layer_norm_eps,
            )
        
        return hidden_states


class SigLIPVisionTowerTTNN:
    """
    SigLIP vision tower using TTNN operations.
    
    Uses hybrid computation:
        - Patch embedding on host (conv2d)
        - Position embedding addition on device
        - Transformer blocks on device
        - Leverages TtLlamaImageAttention for attention
    """
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: "ttnn.Device",
    ):
        """
        Initialize vision tower.
        
        Args:
            config: SigLIP configuration
            weights: PyTorch weights (will be converted)
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
        self.config = config
        self.device = device
        
        # Keep patch embedding on host
        self.patch_embed = PatchEmbeddingTTNN(config, weights, device)
        
        # Position embedding on device
        pos_emb = weights["position_embedding.weight"]
        self.position_embedding = ttnn.from_torch(
            pos_emb.unsqueeze(0),  # Add batch dim
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # Store weights for blocks (converted lazily)
        self.torch_weights = weights
        
        # Final layer norm weights
        if "post_layernorm.weight" in weights:
            self.post_ln_weight = ttnn.from_torch(
                weights["post_layernorm.weight"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.post_ln_bias = ttnn.from_torch(
                weights["post_layernorm.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ) if "post_layernorm.bias" in weights else None
        else:
            self.post_ln_weight = None
            self.post_ln_bias = None
    
    def forward(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        """
        Process images to embeddings (TTNN).
        
        Args:
            pixel_values: PyTorch tensor (batch_size, channels, height, width)
        
        Returns:
            TTNN tensor (batch_size, num_patches, hidden_size)
        """
        # Patch embedding (hybrid)
        hidden_states = self.patch_embed.forward(pixel_values)
        
        # Add position embeddings
        hidden_states = ttnn.add(hidden_states, self.position_embedding)
        
        # For now, use PyTorch for transformer blocks
        # TODO: Implement TTNN blocks using TtLlamaImageAttention
        hidden_states_torch = ttnn.to_torch(hidden_states)
        
        torch_tower = SigLIPVisionTowerTorch(self.config, self.torch_weights)
        torch_tower.patch_embed = None  # Skip patch embedding
        
        # Run through blocks
        for block in torch_tower.blocks:
            hidden_states_torch = block.forward(hidden_states_torch)
        
        # Final layer norm
        if torch_tower.post_layernorm_weight is not None:
            hidden_states_torch = F.layer_norm(
                hidden_states_torch,
                (self.config.hidden_size,),
                torch_tower.post_layernorm_weight,
                torch_tower.post_layernorm_bias,
                self.config.layer_norm_eps,
            )
        
        # Transfer back to device
        return ttnn.from_torch(
            hidden_states_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


# ============================================================================
# Multi-modal Projector
# ============================================================================

class MultiModalProjectorTorch:
    """
    Projects vision features to language model dimension (PyTorch).
    """
    
    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize projector.
        
        Args:
            weights: Dictionary with linear.weight and optional linear.bias
        """
        self.weight = weights["linear.weight"]
        self.bias = weights.get("linear.bias")
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features.
        
        Args:
            vision_features: (batch_size, num_patches, vision_hidden_size)
        
        Returns:
            (batch_size, num_patches, language_hidden_size)
        """
        return F.linear(vision_features, self.weight, self.bias)


class MultiModalProjectorTTNN:
    """
    Projects vision features to language model dimension using TTNN.
    """
    
    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        device: "ttnn.Device",
    ):
        """
        Initialize projector with TTNN weights.
        
        Args:
            weights: PyTorch weights to convert
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
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
    
    def forward(self, vision_features: "ttnn.Tensor") -> "ttnn.Tensor":
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
PatchEmbedding = PatchEmbeddingTorch
SigLIPAttention = SigLIPAttentionTorch
SigLIPMLP = SigLIPMLPTorch
SigLIPBlock = SigLIPBlockTorch
SigLIPVisionTower = SigLIPVisionTowerTorch
MultiModalProjector = MultiModalProjectorTorch

