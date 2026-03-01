"""
Vision Transformer (ViT) image encoder for CLIP using TTNN APIs.
"""

import ttnn
import torch
from typing import Optional, Tuple


class PatchEmbedding:
    """Patch embedding layer for ViT."""
    
    def __init__(self, device, in_channels: int = 3, embed_dim: int = 768, 
                 patch_size: int = 16, image_size: int = 224):
        self.device = device
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Create convolution for patch embedding
        self.conv = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            device=device,
            dtype=ttnn.bfloat16,
            activation=None
        )
        
        # Class token and position embeddings
        self.cls_token = ttnn.create_parameter(
            shape=(1, 1, embed_dim),
            dtype=ttnn.bfloat16,
            device=device,
            requires_grad=True
        )
        
        self.pos_embed = ttnn.create_parameter(
            shape=(1, self.num_patches + 1, embed_dim),
            dtype=ttnn.bfloat16,
            device=device,
            requires_grad=True
        )
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x shape: [batch, channels, height, width]
        x = self.conv(x)  # [batch, embed_dim, grid_h, grid_w]
        x = ttnn.permute(x, (0, 2, 3, 1))  # [batch, grid_h, grid_w, embed_dim]
        batch_size = x.shape[0]
        x = ttnn.reshape(x, (batch_size, self.num_patches, self.embed_dim))
        
        # Add class token
        cls_tokens = ttnn.repeat(self.cls_token, (batch_size, 1, 1))
        x = ttnn.concat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = ttnn.add(x, self.pos_embed)
        return x


class MultiHeadAttention:
    """Multi-head attention layer."""
    
    def __init__(self, device, embed_dim: int = 768, num_heads: int = 12):
        self.device = device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = ttnn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        self.k_proj = ttnn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        self.v_proj = ttnn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        
        # Output projection
        self.out_proj = ttnn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = ttnn.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ttnn.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        
        # Transpose for attention computation
        q = ttnn.permute(q, (0, 2, 1, 3))  # [batch, heads, seq_len, head_dim]
        k = ttnn.permute(k, (0, 2, 3, 1))  # [batch, heads, head_dim, seq_len]
        v = ttnn.permute(v, (0, 2, 1, 3))  # [batch, heads, seq_len, head_dim]
        
        # Attention scores
        attn_scores = ttnn.matmul(q, k)  # [batch, heads, seq_len, seq_len]
        attn_scores = ttnn.multiply(attn_scores, 1.0 / (self.head_dim ** 0.5))
        attn_probs = ttnn.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = ttnn.matmul(attn_probs, v)  # [batch, heads, seq_len, head_dim]
        
        # Reshape back
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.embed_dim))
        
        # Output projection
        output = self.out_proj(attn_output)
        return output


class MLP:
    """Multi-layer perceptron block."""
    
    def __init__(self, device, embed_dim: int = 768, mlp_ratio: int = 4):
        self.device = device
        hidden_dim = embed_dim * mlp_ratio
        
        self.fc1 = ttnn.Linear(
            in_features=embed_dim,
            out_features=hidden_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        self.fc2 = ttnn.Linear(
            in_features=hidden_dim,
            out_features=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        self.activation = ttnn.gelu
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer:
    """Single transformer encoder layer."""
    
    def __init__(self, device, embed_dim: int = 768, num_heads: int = 12, 
                 mlp_ratio: int = 4):
        self.device = device
        
        # Self-attention with layer norm
        self.norm1 = ttnn.LayerNorm(
            normalized_shape=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        self.attn = MultiHeadAttention(device, embed_dim, num_heads)
        
        # MLP with layer norm
        self.norm2 = ttnn.LayerNorm(
            normalized_shape=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        self.mlp = MLP(device, embed_dim, mlp_ratio)
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = ttnn.add(x, residual)
        
        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = ttnn.add(x, residual)
        
        return x


class CLIPVisionTransformer:
    """CLIP Vision Transformer encoder."""
    
    def __init__(self, device, embed_dim: int = 768, num_layers: int = 12, 
                 num_heads: int = 12, mlp_ratio: int = 4, patch_size: int = 16,
                 image_size: int = 224):
        self.device = device
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            device=device,
            embed_dim=embed_dim,
            patch_size=patch_size,
            image_size=image_size
        )
        
        # Transformer encoder layers
        self.layers = [
            TransformerEncoderLayer(
                device=device,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.norm = ttnn.LayerNorm(
            normalized_shape=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        
        # Projection to CLIP embedding space
        self.proj = ttnn.Linear(
            in_features=embed_dim,
            out_features=512,  # CLIP embedding dimension
            device=device,
            dtype=ttnn.bfloat16
        )
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Convert input to TTNN tensor if needed
        if isinstance(x, torch.Tensor):
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=self.device)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer encoder
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm (on class token)
        x = self.norm(x)
        cls_token = ttnn.slice(x, (0, 0, 0), (x.shape[0], 1, x.shape[2]))
        
        # Project to CLIP embedding space
        output = self.proj(cls_token)
        output = ttnn.squeeze(output, dim=1)  # Remove sequence dimension
        
        return output
