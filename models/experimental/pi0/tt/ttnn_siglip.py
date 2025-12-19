# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SigLIP Vision Tower - TTNN Implementation.

This module implements the SigLIP vision encoder using TTNN operations
for efficient execution on Tenstorrent hardware.

SigLIP Architecture:
    - Patch embedding (hybrid: conv2d on host)
    - Positional embedding (on device)
    - Transformer encoder blocks (TTNN)
    - Multi-modal projector (TTNN linear)
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import ttnn

from models.experimental.pi0.common.configs import SigLIPConfig
from models.experimental.pi0.reference.torch_siglip import SigLIPAttention as SigLIPAttentionTorch


# ============================================================================
# Patch Embedding (Hybrid)
# ============================================================================

class TtPatchEmbedding:
    """
    Convert image patches to embeddings (Hybrid approach).
    
    Uses PyTorch conv2d for reliability, then converts to TTNN.
    """
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        self.config = config
        self.device = device
        
        # Store PyTorch weights for conv2d
        self._torch_weight = (weights.get("patch_embedding.weight") or 
                             weights.get("vision_model.embeddings.patch_embedding.weight"))
        self._torch_bias = (weights.get("patch_embedding.bias") or 
                           weights.get("vision_model.embeddings.patch_embedding.bias"))
    
    def forward(self, pixel_values) -> ttnn.Tensor:
        """Extract patch embeddings using hybrid approach."""
        # Convert to PyTorch if needed
        if isinstance(pixel_values, ttnn.Tensor):
            pixel_values = ttnn.to_torch(pixel_values)
        
        patch_size = self.config.patch_size
        
        # Use PyTorch convolution
        conv_weight = self._torch_weight.to(pixel_values.dtype)
        conv_bias = self._torch_bias.to(pixel_values.dtype) if self._torch_bias is not None else None
        
        x = torch.nn.functional.conv2d(
            pixel_values,
            conv_weight,
            conv_bias,
            stride=patch_size,
        )
        
        # Reshape: (B, C, H_out, W_out) -> (B, num_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)
        
        # Convert to TTNN
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


# ============================================================================
# SigLIP Attention (TTNN)
# ============================================================================

class TtSigLIPAttention:
    """SigLIP self-attention using TTNN operations."""
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        self.config = config
        self.device = device
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Concatenate QKV weights
        q_weight = weights["self_attn.q_proj.weight"]
        k_weight = weights["self_attn.k_proj.weight"]
        v_weight = weights["self_attn.v_proj.weight"]
        
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0).T.contiguous()
        
        self.wqkv = ttnn.from_torch(
            qkv_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # Biases
        if "self_attn.q_proj.bias" in weights:
            qkv_bias = torch.cat([
                weights["self_attn.q_proj.bias"],
                weights["self_attn.k_proj.bias"],
                weights["self_attn.v_proj.bias"]
            ], dim=0)
            self.bqkv = ttnn.from_torch(
                qkv_bias.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        else:
            self.bqkv = None
        
        # Output projection
        out_weight = weights["self_attn.out_proj.weight"].T.contiguous()
        self.wo = ttnn.from_torch(
            out_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        
        if "self_attn.out_proj.bias" in weights:
            self.bo = ttnn.from_torch(
                weights["self_attn.out_proj.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        else:
            self.bo = None
        
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    
    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass using TTNN operations."""
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        
        # Ensure 4D shape
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, seq_len, -1))
        
        # Fused QKV projection
        xqkv_fused = ttnn.linear(
            hidden_states,
            self.wqkv,
            bias=self.bqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        
        # Split into Q, K, V heads
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)
        
        # Get device grid size
        device_grid = self.device.compute_with_storage_grid_size()
        grid_x = min(8, device_grid.x)
        grid_y = min(8, device_grid.y)
        
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,
        )
        
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            scale=self.scale,
            program_config=sdpa_cfg,
            compute_kernel_config=self.compute_kernel_config,
        )
        
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)
        
        # Concatenate heads
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # Output projection
        output = ttnn.linear(
            attn_output,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn_output)
        
        if self.bo is not None:
            output = ttnn.add(output, self.bo)
        
        if seq_len > 0:
            output = ttnn.reshape(output, (batch_size, seq_len, -1))
        
        return output


# ============================================================================
# SigLIP MLP (TTNN)
# ============================================================================

class TtSigLIPMLP:
    """SigLIP MLP with GELU activation using TTNN."""
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        self.config = config
        self.device = device
        
        # FC1
        self.fc1_weight = ttnn.from_torch(
            weights["mlp.fc1.weight"].T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        
        self.fc1_bias = None
        if "mlp.fc1.bias" in weights:
            self.fc1_bias = ttnn.from_torch(
                weights["mlp.fc1.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        
        # FC2
        self.fc2_weight = ttnn.from_torch(
            weights["mlp.fc2.weight"].T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        
        self.fc2_bias = None
        if "mlp.fc2.bias" in weights:
            self.fc2_bias = ttnn.from_torch(
                weights["mlp.fc2.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    
    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass using TTNN operations."""
        x = ttnn.linear(
            hidden_states,
            self.fc1_weight,
            bias=self.fc1_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            activation="gelu",
        )
        
        output = ttnn.linear(
            x,
            self.fc2_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(x)
        
        if self.fc2_bias is not None:
            output = ttnn.add(output, self.fc2_bias)
        
        return output


# ============================================================================
# SigLIP Block (TTNN with Hybrid Attention)
# ============================================================================

class TtSigLIPBlock:
    """SigLIP transformer block using TTNN with hybrid attention fallback."""
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        self.config = config
        self.device = device
        
        # Layer norms
        self.ln1_weight = ttnn.from_torch(
            weights["layer_norm1.weight"].reshape(1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        
        self.ln1_bias = None
        if "layer_norm1.bias" in weights:
            self.ln1_bias = ttnn.from_torch(
                weights["layer_norm1.bias"].reshape(1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        
        self.ln2_weight = ttnn.from_torch(
            weights["layer_norm2.weight"].reshape(1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        
        self.ln2_bias = None
        if "layer_norm2.bias" in weights:
            self.ln2_bias = ttnn.from_torch(
                weights["layer_norm2.bias"].reshape(1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        
        # Use PyTorch attention (hybrid) due to nlp_concat_heads dimension issues
        self.attention = SigLIPAttentionTorch(config, weights)
        self.attention_is_torch = True
        self.mlp = TtSigLIPMLP(config, weights, device)
    
    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with hybrid attention fallback."""
        # Pre-attention LayerNorm
        normed = ttnn.layer_norm(
            hidden_states,
            weight=self.ln1_weight,
            bias=self.ln1_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        
        # Hybrid attention
        if self.attention_is_torch:
            normed_torch = ttnn.to_torch(normed)
            attn_output_torch = self.attention.forward(normed_torch)
            attn_output = ttnn.from_torch(
                attn_output_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        else:
            attn_output = self.attention.forward(normed)
        
        hidden_states = ttnn.add(hidden_states, attn_output)
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
        hidden_states = ttnn.add(hidden_states, mlp_output)
        ttnn.deallocate(mlp_output)
        
        return hidden_states


# ============================================================================
# Full Vision Tower (TTNN)
# ============================================================================

class TtSigLIPVisionTower:
    """SigLIP vision tower using TTNN operations."""
    
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        self.config = config
        self.device = device
        
        # Patch embedding (hybrid)
        self.patch_embed = TtPatchEmbedding(config, weights, device)
        
        # Position embedding
        pos_emb = (weights.get("position_embedding.weight") or 
                  weights.get("vision_model.embeddings.position_embedding.weight"))
        
        if pos_emb is not None:
            num_patches = (config.image_size // config.patch_size) ** 2
            
            # Interpolate if needed
            if pos_emb.shape[0] != num_patches:
                original_size = int(math.sqrt(pos_emb.shape[0]))
                target_size = int(math.sqrt(num_patches))
                
                pos_emb_2d = pos_emb.view(1, original_size, original_size, -1)
                pos_emb_2d = pos_emb_2d.permute(0, 3, 1, 2)
                
                pos_emb_interpolated = F.interpolate(
                    pos_emb_2d,
                    size=(target_size, target_size),
                    mode="bicubic",
                    align_corners=False,
                )
                
                pos_emb = pos_emb_interpolated.permute(0, 2, 3, 1).flatten(0, 2)
            
            self.position_ids = ttnn.arange(0, num_patches, 1, dtype=ttnn.uint32, device=device)
            self.position_ids = ttnn.reshape(self.position_ids, (1, -1))
            
            self.pos_emb_weights = ttnn.as_tensor(
                pos_emb,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        else:
            self.position_ids = None
            self.pos_emb_weights = None
        
        # Transformer blocks
        self.blocks = []
        for i in range(config.num_hidden_layers):
            block_weights = self._get_layer_weights(weights, i)
            self.blocks.append(TtSigLIPBlock(config, block_weights, device))
        
        # Final layer norm
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
            )
            self.post_ln_bias = ttnn.from_torch(
                post_ln_bias.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
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
        prefixes = [f"vision_model.encoder.layers.{layer_idx}.", f"encoder.layers.{layer_idx}."]
        layer_weights = {}
        for prefix in prefixes:
            for key, value in weights.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    layer_weights[new_key] = value
        return layer_weights
    
    def forward(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """Process images to embeddings."""
        hidden_states = self.patch_embed.forward(pixel_values)
        
        # Add position embeddings
        if self.pos_emb_weights is not None:
            num_patches_actual = hidden_states.shape[1]
            num_patches_expected = self.position_ids.shape[1]
            
            if num_patches_actual != num_patches_expected:
                # Dynamic interpolation
                pos_emb_torch = ttnn.to_torch(self.pos_emb_weights)
                
                original_size = int(math.sqrt(num_patches_expected))
                target_size = int(math.sqrt(num_patches_actual))
                hidden_size = pos_emb_torch.shape[1]
                
                pos_emb_2d = pos_emb_torch.view(1, original_size, original_size, hidden_size)
                pos_emb_2d = pos_emb_2d.permute(0, 3, 1, 2)
                
                pos_emb_interpolated = F.interpolate(
                    pos_emb_2d,
                    size=(target_size, target_size),
                    mode="bicubic",
                    align_corners=False,
                )
                
                pos_emb_resized = pos_emb_interpolated.permute(0, 2, 3, 1).flatten(0, 2)
                
                position_ids_new = ttnn.arange(0, num_patches_actual, 1, dtype=ttnn.uint32, device=self.device)
                position_ids_new = ttnn.reshape(position_ids_new, (1, -1))
                
                pos_emb_weights_new = ttnn.as_tensor(
                    pos_emb_resized,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
                
                positional_embeddings = ttnn.embedding(
                    position_ids_new,
                    pos_emb_weights_new,
                    layout=ttnn.TILE_LAYOUT,
                )
            else:
                positional_embeddings = ttnn.embedding(
                    self.position_ids,
                    self.pos_emb_weights,
                    layout=ttnn.TILE_LAYOUT,
                )
            
            hidden_states = ttnn.add(hidden_states, positional_embeddings)
        
        # Transformer blocks
        for block in self.blocks:
            hidden_states = block.forward(hidden_states)
        
        # Final layer norm
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

class TtMultiModalProjector:
    """Projects vision features to language model dimension using TTNN."""
    
    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        self.device = device
        
        self.weight = ttnn.from_torch(
            weights["linear.weight"].T.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        
        self.bias = None
        if "linear.bias" in weights:
            self.bias = ttnn.from_torch(
                weights["linear.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
    
    def forward(self, vision_features: ttnn.Tensor) -> ttnn.Tensor:
        """Project vision features using TTNN linear."""
        return ttnn.linear(
            vision_features,
            self.weight,
            bias=self.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

