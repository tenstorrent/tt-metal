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

    @property
    def padded_head_dim(self) -> int:
        """Head dim padded to next multiple of 32 for TTNN tile operations."""
        return ((self.head_dim + 31) // 32) * 32


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
        # Handle both formats: vision_model.embeddings.patch_embedding (checkpoint) and patch_embedding (legacy)
        self.conv_weight = (weights.get("patch_embedding.weight") or 
                           weights.get("vision_model.embeddings.patch_embedding.weight"))
        self.conv_bias = (weights.get("patch_embedding.bias") or 
                         weights.get("vision_model.embeddings.patch_embedding.bias"))
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract patch embeddings from images.
        
        Args:
            pixel_values: (batch_size, channels, height, width)
        
        Returns:
            (batch_size, num_patches, hidden_size)
        """
        batch_size = pixel_values.shape[0]
        
        # Apply convolution (ensure dtype compatibility)
        conv_weight = self.conv_weight.to(pixel_values.dtype)
        conv_bias = self.conv_bias.to(pixel_values.dtype) if self.conv_bias is not None else None
        x = F.conv2d(
            pixel_values,
            conv_weight,
            conv_bias,
            stride=self.config.patch_size,
        )
        
        # Reshape: (B, C, H, W) -> (B, num_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)
        
        return x


class PatchEmbeddingTTNN:
    """
    Convert image patches to embeddings using hybrid approach.
    
    Uses PyTorch conv2d for patch extraction, then converts to TTNN.
    This hybrid approach is more reliable than pure TTNN fold for non-standard patch sizes.
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
            weights: PyTorch weights for conv2d
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
        self.config = config
        self.device = device
        
        # Handle both formats: vision_model.embeddings.patch_embedding (checkpoint) and patch_embedding (legacy)
        conv_weight = (weights.get("patch_embedding.weight") or 
                      weights.get("vision_model.embeddings.patch_embedding.weight"))
        conv_bias = (weights.get("patch_embedding.bias") or 
                    weights.get("vision_model.embeddings.patch_embedding.bias"))
        
        # Store PyTorch weights for hybrid conv2d approach
        self._torch_weight = conv_weight
        self._torch_bias = conv_bias
    
    def forward(self, pixel_values) -> "ttnn.Tensor":
        """
        Extract patch embeddings - use PyTorch for convolution, then convert to TTNN.
        
        Hybrid approach for reliability (PyTorch conv → TTNN tensor).
        
        Args:
            pixel_values: PyTorch or TTNN tensor (batch_size, channels, height, width)
        
        Returns:
            TTNN tensor (batch_size, num_patches, hidden_size)
        """
        # Convert to PyTorch if needed
        if isinstance(pixel_values, ttnn.Tensor):
            pixel_values = ttnn.to_torch(pixel_values)
        
        batch_size = pixel_values.shape[0]
        patch_size = self.config.patch_size
        
        # Use PyTorch convolution for reliable patch extraction
        conv_weight = self._torch_weight.to(pixel_values.dtype)
        conv_bias = self._torch_bias.to(pixel_values.dtype) if self._torch_bias is not None else None
        
        # Apply convolution
        x = torch.nn.functional.conv2d(
            pixel_values,
            conv_weight,
            conv_bias,
            stride=patch_size,
        )
        
        # Reshape: (B, C, H_out, W_out) -> (B, num_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)
        
        # Convert to TTNN
        x_ttnn = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        return x_ttnn


# ============================================================================
# Vision Transformer Block
# ============================================================================

class SigLIPAttentionTTNN:
    """
    SigLIP self-attention using TTNN operations.
    
    Uses separate Q/K/V projections with manual reshaping (like CLIP) to avoid
    nlp_create_qkv_heads padding issues with non-tile-aligned head dimensions.
    """
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: "ttnn.Device",
    ):
        """
        Initialize attention with TTNN weights.
        
        Args:
            config: SigLIP configuration
            weights: PyTorch weights to convert
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
        self.config = config
        self.device = device
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Separate Q, K, V weight matrices (transposed for TTNN linear)
        self.wq = ttnn.from_torch(
            weights["self_attn.q_proj.weight"].T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.wk = ttnn.from_torch(
            weights["self_attn.k_proj.weight"].T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.wv = ttnn.from_torch(
            weights["self_attn.v_proj.weight"].T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # Q, K, V biases
        if "self_attn.q_proj.bias" in weights:
            self.bq = ttnn.from_torch(
                weights["self_attn.q_proj.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.bk = ttnn.from_torch(
                weights["self_attn.k_proj.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.bv = ttnn.from_torch(
                weights["self_attn.v_proj.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.bq = self.bk = self.bv = None
        
        # Output projection (no padding needed with manual approach)
        self.wo = ttnn.from_torch(
            weights["self_attn.out_proj.weight"].T.contiguous(),
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
        
        # Compute kernel config
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    
    def forward(self, hidden_states: "ttnn.Tensor") -> "ttnn.Tensor":
        """
        Forward pass using native TTNN SDPA with head_dim padding.
        
        Since head_dim=72 is not tile-aligned (need multiple of 32), we:
        1. Pad Q/K/V to padded_head_dim=96
        2. Run TTNN SDPA
        3. Slice back to original head_dim=72
        
        Args:
            hidden_states: TTNN tensor (batch_size, seq_len, hidden_size)
        
        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        padded_head_dim = ((self.head_dim + 31) // 32) * 32  # 72 -> 96
        
        # Separate Q, K, V projections on TTNN -> each [batch, seq, hidden_size]
        q = ttnn.linear(
            hidden_states, self.wq, bias=self.bq,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        k = ttnn.linear(
            hidden_states, self.wk, bias=self.bk,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        v = ttnn.linear(
            hidden_states, self.wv, bias=self.bv,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        
        # Convert to PyTorch for reshape/pad (TTNN reshape doesn't support non-tile dims)
        q_torch = ttnn.to_torch(q)
        k_torch = ttnn.to_torch(k)
        v_torch = ttnn.to_torch(v)
        
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        
        # Reshape: [batch, seq, hidden] -> [batch, seq, heads, head_dim]
        q_torch = q_torch.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_torch = k_torch.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_torch = v_torch.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        q_torch = q_torch.transpose(1, 2).contiguous()
        k_torch = k_torch.transpose(1, 2).contiguous()
        v_torch = v_torch.transpose(1, 2).contiguous()
        
        # Pad head_dim: [batch, heads, seq, 72] -> [batch, heads, seq, 96]
        pad_size = padded_head_dim - self.head_dim  # 96 - 72 = 24
        q_padded = torch.nn.functional.pad(q_torch, (0, pad_size), value=0.0)
        k_padded = torch.nn.functional.pad(k_torch, (0, pad_size), value=0.0)
        v_padded = torch.nn.functional.pad(v_torch, (0, pad_size), value=0.0)
        
        # Convert padded tensors to TTNN
        q_ttnn = ttnn.from_torch(
            q_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k_ttnn = ttnn.from_torch(
            k_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_ttnn = ttnn.from_torch(
            v_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # TTNN SDPA with tile-aligned head_dim
        device_grid = self.device.compute_with_storage_grid_size()
        grid_x = min(8, device_grid.x)
        grid_y = min(8, device_grid.y)
        
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            q_chunk_size=min(256, seq_len),
            k_chunk_size=min(256, seq_len),
            exp_approx_mode=False,
        )
        
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_ttnn, k_ttnn, v_ttnn,
            is_causal=False,
            scale=self.scale,
            program_config=sdpa_cfg,
            compute_kernel_config=self.compute_kernel_config,
        )
        
        ttnn.deallocate(q_ttnn)
        ttnn.deallocate(k_ttnn)
        ttnn.deallocate(v_ttnn)
        
        # Convert back to PyTorch, slice to original head_dim, reshape
        attn_torch = ttnn.to_torch(attn_output)  # [batch, heads, seq, padded_head_dim]
        ttnn.deallocate(attn_output)
        
        # Slice: [batch, heads, seq, 96] -> [batch, heads, seq, 72]
        attn_torch = attn_torch[..., :self.head_dim]
        
        # Transpose: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        attn_torch = attn_torch.transpose(1, 2).contiguous()
        
        # Reshape: [batch, seq, heads, head_dim] -> [batch, seq, hidden]
        attn_torch = attn_torch.reshape(batch_size, seq_len, self.hidden_size)
        
        # Convert back to TTNN for output projection
        attn_ttnn = ttnn.from_torch(
            attn_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # Output projection on TTNN
        output = ttnn.linear(
            attn_ttnn, self.wo,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn_ttnn)
        
        # Add bias if present
        if self.bo is not None:
            output = ttnn.add(output, self.bo)
        
        return output


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
        
        # QKV projections (ensure dtype compatibility)
        q_proj = self.q_proj.to(hidden_states.dtype)
        k_proj = self.k_proj.to(hidden_states.dtype)
        v_proj = self.v_proj.to(hidden_states.dtype)
        q_bias = self.q_bias.to(hidden_states.dtype) if self.q_bias is not None else None
        k_bias = self.k_bias.to(hidden_states.dtype) if self.k_bias is not None else None
        v_bias = self.v_bias.to(hidden_states.dtype) if self.v_bias is not None else None
        
        q = F.linear(hidden_states, q_proj, q_bias)
        k = F.linear(hidden_states, k_proj, k_bias)
        v = F.linear(hidden_states, v_proj, v_bias)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project (ensure dtype compatibility)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out_proj = self.out_proj.to(hidden_states.dtype)
        out_bias = self.out_bias.to(hidden_states.dtype) if self.out_bias is not None else None
        return F.linear(attn_output, out_proj, out_bias)


class SigLIPMLPTTNN:
    """
    SigLIP MLP with GELU activation using TTNN.
    
    Based on TtGemmaImageFeedForward from models/demos/gemma3/tt/gemma_image_mlp.py
    """
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: "ttnn.Device",
    ):
        """
        Initialize MLP with TTNN weights.
        
        Args:
            config: SigLIP configuration
            weights: PyTorch weights to convert
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
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
    
    def forward(self, hidden_states: "ttnn.Tensor") -> "ttnn.Tensor":
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
        # Ensure dtype compatibility
        fc1_weight = self.fc1_weight.to(hidden_states.dtype)
        fc1_bias = self.fc1_bias.to(hidden_states.dtype) if self.fc1_bias is not None else None
        fc2_weight = self.fc2_weight.to(hidden_states.dtype)
        fc2_bias = self.fc2_bias.to(hidden_states.dtype) if self.fc2_bias is not None else None
        
        x = F.linear(hidden_states, fc1_weight, fc1_bias)
        x = F.gelu(x, approximate="tanh")
        return F.linear(x, fc2_weight, fc2_bias)


class SigLIPBlockTTNN:
    """
    Complete SigLIP transformer block using TTNN.
    
    Based on TtGemmaImageTransformerBlock from models/demos/gemma3/tt/gemma_image_block.py
    """
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: "ttnn.Device",
    ):
        """
        Initialize block with TTNN weights.
        
        Args:
            config: SigLIP configuration
            weights: PyTorch weights to convert
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
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
    
    def forward(self, hidden_states: "ttnn.Tensor") -> "ttnn.Tensor":
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
        # Pre-attention norm (ensure dtype compatibility)
        ln1_weight = self.ln1_weight.to(hidden_states.dtype) if self.ln1_weight is not None else None
        ln1_bias = self.ln1_bias.to(hidden_states.dtype) if self.ln1_bias is not None else None
        normed = F.layer_norm(
            hidden_states,
            (self.config.hidden_size,),
            ln1_weight,
            ln1_bias,
            self.config.layer_norm_eps,
        )
        
        # Attention with residual
        hidden_states = hidden_states + self.attention.forward(normed)
        
        # Pre-MLP norm (ensure dtype compatibility)
        ln2_weight = self.ln2_weight.to(hidden_states.dtype) if self.ln2_weight is not None else None
        ln2_bias = self.ln2_bias.to(hidden_states.dtype) if self.ln2_bias is not None else None
        normed = F.layer_norm(
            hidden_states,
            (self.config.hidden_size,),
            ln2_weight,
            ln2_bias,
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
        
        # Position embedding (handle both formats)
        self.position_embedding = (weights.get("position_embedding.weight") or 
                                   weights.get("vision_model.embeddings.position_embedding.weight"))
        
        # Encoder blocks
        self.blocks = []
        for i in range(config.num_hidden_layers):
            block_weights = self._get_layer_weights(weights, i)
            self.blocks.append(SigLIPBlockTorch(config, block_weights))
        
        # Final layer norm
        self.post_layernorm_weight = weights.get("post_layernorm.weight") or weights.get("vision_model.post_layernorm.weight")
        self.post_layernorm_bias = weights.get("post_layernorm.bias") or weights.get("vision_model.post_layernorm.bias")
    
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
        
        # Add position embeddings (with interpolation if needed)
        if self.position_embedding is not None:
            num_patches = hidden_states.shape[1]
            num_positions = self.position_embedding.shape[0]
            
            if num_patches != num_positions:
                # Interpolate position embeddings to match the number of patches
                pos_embed = self.position_embedding.unsqueeze(0).permute(0, 2, 1)  # (1, hidden_size, num_positions)
                
                # Calculate original grid size (assume square)
                orig_size = int(num_positions ** 0.5)
                new_size = int(num_patches ** 0.5)
                
                pos_embed = pos_embed.reshape(1, self.config.hidden_size, orig_size, orig_size)
                pos_embed = torch.nn.functional.interpolate(
                    pos_embed,
                    size=(new_size, new_size),
                    mode='bicubic',
                    align_corners=False
                )
                pos_embed = pos_embed.reshape(1, self.config.hidden_size, -1).permute(0, 2, 1)  # (1, num_patches, hidden_size)
                pos_embed = pos_embed.squeeze(0)  # (num_patches, hidden_size)
            else:
                pos_embed = self.position_embedding
            
            hidden_states = hidden_states + pos_embed
        
        # Encoder blocks
        for block in self.blocks:
            hidden_states = block.forward(hidden_states)
        
        # Final layer norm (ensure dtype compatibility)
        if self.post_layernorm_weight is not None:
            post_ln_weight = self.post_layernorm_weight.to(hidden_states.dtype)
            post_ln_bias = self.post_layernorm_bias.to(hidden_states.dtype) if self.post_layernorm_bias is not None else None
            hidden_states = F.layer_norm(
                hidden_states,
                (self.config.hidden_size,),
                post_ln_weight,
                post_ln_bias,
                self.config.layer_norm_eps,
            )
        
        return hidden_states


class SigLIPVisionTowerTTNN:
    """
    SigLIP vision tower using TTNN operations.
    
    Now fully implemented in TTNN:
        - Patch embedding on host (conv2d unfold)
        - Position embedding addition on device
        - All transformer blocks on device (TTNN)
        - Final layer norm on device
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
        
        # Position embedding on device (handle both formats) - Following Gemma3 pattern
        pos_emb = (weights.get("position_embedding.weight") or 
                  weights.get("vision_model.embeddings.position_embedding.weight"))
        
        if pos_emb is not None:
            # Calculate target number of patches based on config
            num_patches = (config.image_size // config.patch_size) ** 2
            print(f"[INIT DEBUG] pos_emb.shape: {pos_emb.shape}, num_patches: {num_patches}, config.image_size: {config.image_size}, config.patch_size: {config.patch_size}")
            
            # Check if we need to interpolate position embeddings
            if pos_emb.shape[0] != num_patches:
                import math
                import torch.nn.functional as F
                
                print(f"[INFO] Interpolating position embeddings from {pos_emb.shape[0]} to {num_patches} patches")
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
            
            # Create position IDs (like Gemma3)
            self.position_ids = ttnn.arange(0, num_patches, 1, dtype=ttnn.uint32, device=device)
            self.position_ids = ttnn.reshape(self.position_ids, (1, -1))
            
            # Load position embedding weights (like Gemma3)
            self.pos_emb_weights = ttnn.as_tensor(
                pos_emb,  # Now correctly sized: (num_patches, hidden_size)
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
        post_ln_weight = (weights.get("post_layernorm.weight") or 
                         weights.get("vision_model.post_layernorm.weight"))
        post_ln_bias = (weights.get("post_layernorm.bias") or 
                       weights.get("vision_model.post_layernorm.bias"))
        
        if post_ln_weight is not None:
            self.post_ln_weight = ttnn.from_torch(
                post_ln_weight.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.post_ln_bias = ttnn.from_torch(
                post_ln_bias.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ) if post_ln_bias is not None else None
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
    
    def forward(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        """
        Process images to embeddings (TTNN).
        
        Args:
            pixel_values: PyTorch tensor (batch_size, channels, height, width)
        
        Returns:
            TTNN tensor (batch_size, num_patches, hidden_size)
        """
        # Patch embedding (hybrid - conv2d on host, then transfer to device)
        hidden_states = self.patch_embed.forward(pixel_values)
        print(f"[DEBUG] After patch embed, hidden_states shape: {hidden_states.shape}")
        
        # Add position embeddings (on device) - Handle dynamic image sizes
        if self.pos_emb_weights is not None:
            num_patches_actual = hidden_states.shape[1]
            num_patches_expected = self.position_ids.shape[1]
            
            print(f"[DEBUG] Actual patches: {num_patches_actual}, Expected patches: {num_patches_expected}")
            
            # Check if we need to interpolate position embeddings dynamically
            if num_patches_actual != num_patches_expected:
                import math
                import torch.nn.functional as F
                
                print(f"[INFO] Dynamic position embedding interpolation needed: {num_patches_expected} → {num_patches_actual}")
                
                # Convert position embeddings back to torch for interpolation
                pos_emb_torch = ttnn.to_torch(self.pos_emb_weights)  # (num_patches_expected, hidden_size)
                
                original_size = int(math.sqrt(num_patches_expected))
                target_size = int(math.sqrt(num_patches_actual))
                hidden_size = pos_emb_torch.shape[1]
                
                # Reshape and interpolate
                pos_emb_2d = pos_emb_torch.view(1, original_size, original_size, hidden_size)
                pos_emb_2d = pos_emb_2d.permute(0, 3, 1, 2)  # (1, hidden_size, H, W)
                
                pos_emb_interpolated = F.interpolate(
                    pos_emb_2d,
                    size=(target_size, target_size),
                    mode="bicubic",
                    align_corners=False,
                )
                
                pos_emb_resized = pos_emb_interpolated.permute(0, 2, 3, 1).flatten(0, 2)  # (num_patches_actual, hidden_size)
                
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
            
            print(f"[DEBUG] Positional embeddings shape: {positional_embeddings.shape}")
            # Use explicit ttnn.add (like Gemma3)
            print("[DEBUG] About to add hidden_states + positional_embeddings...")
            hidden_states = ttnn.add(hidden_states, positional_embeddings)
            print("[DEBUG] Addition successful!")
        
        # Run through TTNN transformer blocks
        for block in self.blocks:
            hidden_states = block.forward(hidden_states)
        
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
        # Ensure dtype compatibility
        weight = self.weight.to(vision_features.dtype)
        bias = self.bias.to(vision_features.dtype) if self.bias is not None else None
        return F.linear(vision_features, weight, bias)


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

