"""
Transformer text encoder for CLIP using TTNN APIs.
"""

import ttnn
import torch
from typing import Optional, Tuple


class TextEmbedding:
    """Text embedding layer (token + position embeddings)."""
    
    def __init__(self, device, vocab_size: int = 49408, embed_dim: int = 512, 
                 max_length: int = 77):
        self.device = device
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Token embeddings
        self.token_embedding = ttnn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        
        # Position embeddings
        self.position_embedding = ttnn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        
        # Create position indices
        self.position_ids = ttnn.arange(
            start=0,
            end=max_length,
            device=device,
            dtype=ttnn.int32
        )
        self.position_ids = ttnn.reshape(self.position_ids, (1, max_length))
    
    def __call__(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        seq_len = input_ids.shape[1]
        pos_ids = ttnn.slice(self.position_ids, (0, 0), (1, seq_len))
        pos_embeds = self.position_embedding(pos_ids)
        
        # Add embeddings
        embeddings = ttnn.add(token_embeds, pos_embeds)
        return embeddings


class TextTransformerEncoderLayer:
    """Single transformer encoder layer for text."""
    
    def __init__(self, device, embed_dim: int = 512, num_heads: int = 8):
        self.device = device
        
        # Self-attention with layer norm
        self.norm1 = ttnn.LayerNorm(
            normalized_shape=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        
        # Multi-head attention
        self.attn = ttnn.MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device,
            dtype=ttnn.bfloat16
        )
        
        # MLP with layer norm
        self.norm2 = ttnn.LayerNorm(
            normalized_shape=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        
        # MLP (feed-forward network)
        self.mlp = ttnn.Sequential(
            ttnn.Linear(embed_dim, embed_dim * 4, device=device, dtype=ttnn.bfloat16),
            ttnn.gelu,
            ttnn.Linear(embed_dim * 4, embed_dim, device=device, dtype=ttnn.bfloat16)
        )
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x)  # Self-attention
        x = ttnn.add(x, residual)
        
        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = ttnn.add(x, residual)
        
        return x


class CLIPTextTransformer:
    """CLIP Text Transformer encoder."""
    
    def __init__(self, device, embed_dim: int = 512, num_layers: int = 12, 
                 num_heads: int = 8, vocab_size: int = 49408, 
                 max_length: int = 77):
        self.device = device
        self.embed_dim = embed_dim
        
        # Text embedding
        self.embedding = TextEmbedding(
            device=device,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_length=max_length
        )
        
        # Transformer encoder layers
        self.layers = [
            TextTransformerEncoderLayer(
                device=device,
                embed_dim=embed_dim,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.norm = ttnn.LayerNorm(
            normalized_shape=embed_dim,
            device=device,
            dtype=ttnn.bfloat16
        )
        
        # EOS token projection
        self.eos_proj = ttnn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,  # Same as input for CLIP
            device=device,
            dtype=ttnn.bfloat16
        )
    
    def __call__(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        # Convert input to TTNN tensor if needed
        if isinstance(input_ids, torch.Tensor):
            input_ids = ttnn.from_torch(input_ids, dtype=ttnn.int32, device=self.device)
        
        # Get embeddings
        x = self.embedding(input_ids)
        
        # Transformer encoder
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Extract EOS token (last token)
        batch_size = x.shape[0]
        eos_token = ttnn.slice(x, (0, -1, 0), (batch_size, 1, self.embed_dim))
        eos_token = ttnn.squeeze(eos_token, dim=1)
        
        # Project EOS token
        output = self.eos_proj(eos_token)
        
        return output
