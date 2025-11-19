# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pure PyTorch Reference Implementation of MiniCPM-o-2_6 Components

Based on the MiniCPM-o-2_6 model structure. All components implemented from scratch
to avoid std::bad_alloc errors when loading full HuggingFace models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class WhisperConv1d(nn.Module):
    """
    Whisper Conv1d Preprocessing Layers

    Architecture (from MiniCPM-o-2_6 WhisperEncoder):
        - conv1: Conv1d(80 -> 1024, kernel_size=3, stride=1, padding=1)
        - conv2: Conv1d(1024 -> 1024, kernel_size=3, stride=2, padding=1)
        - Reduces sequence length by 2x (stride=2 in conv2)
    """

    def __init__(
        self,
        in_channels: int = 80,
        hidden_size: int = 1024,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, seq_len)
        Returns:
            output: (batch_size, hidden_size, seq_len//2)
        """
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        return x


class WhisperAttention(nn.Module):
    """Whisper Multi-Head Self Attention"""

    def __init__(self, hidden_size: int = 1024, num_heads: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.out_proj(attn_output)


class WhisperEncoderLayer(nn.Module):
    """Single Whisper Encoder Layer"""

    def __init__(self, hidden_size: int = 1024, num_heads: int = 16, ffn_dim: int = 4096):
        super().__init__()
        self.self_attn = WhisperAttention(hidden_size, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention with residual
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        # Feed forward with residual
        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = residual + x

        return x


class WhisperEncoder(nn.Module):
    """
    Complete Whisper Encoder Implementation

    Based on MiniCPM-o-2_6 audio encoder specifications.
    """

    def __init__(
        self,
        input_channels: int = 80,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        ffn_dim: int = 4096,
        max_source_positions: int = 1500,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Convolutional preprocessing
        self.conv_layers = WhisperConv1d(input_channels, hidden_size)

        # Positional embeddings
        self.embed_positions = nn.Embedding(max_source_positions, hidden_size)

        # Encoder layers
        self.layers = nn.ModuleList([WhisperEncoderLayer(hidden_size, num_heads, ffn_dim) for _ in range(num_layers)])

        # Final layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: (batch_size, input_channels, seq_len)
        Returns:
            last_hidden_state: (batch_size, seq_len//2, hidden_size)
        """
        # Convolutional preprocessing
        hidden_states = self.conv_layers(input_features)

        # Transpose for sequence processing: (batch, hidden_size, seq_len//2) -> (batch, seq_len//2, hidden_size)
        hidden_states = hidden_states.transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape

        # Add positional embeddings
        positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        hidden_states = hidden_states + self.embed_positions(positions)

        # Create attention mask (causal for encoder)
        attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1).bool()
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
        attention_mask = attention_mask.masked_fill(attention_mask, float("-inf"))

        # Encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class SiglipVisionEmbeddings(nn.Module):
    """
    SigLIP Vision Embeddings - Converts images to patch embeddings

    Architecture (from MiniCPM-o-2_6 SiglipVisionConfig):
        - patch_embedding: Conv2d(3 -> 1152, kernel_size=14, stride=14, padding=0)
        - position_embedding: Embedding(4900, 1152)
        - For 980x980 images: (980/14)^2 = 4900 patches
    """

    def __init__(
        self,
        image_size: int = 980,
        patch_size: int = 14,
        num_channels: int = 3,
        hidden_size: int = 1152,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Position embedding (includes CLS token)
        self.position_embedding = nn.Embedding(self.num_patches + 1, hidden_size)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch_size, num_channels, image_size, image_size)
        Returns:
            embeddings: (batch_size, num_patches + 1, hidden_size)
        """
        batch_size = pixel_values.shape[0]

        # Patch embedding: (batch, hidden_size, num_patches_sqrt, num_patches_sqrt)
        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten patches: (batch, hidden_size, num_patches)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeds], dim=1)

        # Add position embeddings
        positions = torch.arange(self.num_patches + 1, device=embeddings.device).unsqueeze(0)
        embeddings = embeddings + self.position_embedding(positions)

        return embeddings


class SiglipAttention(nn.Module):
    """SigLIP Multi-Head Self Attention"""

    def __init__(self, hidden_size: int = 1152, num_heads: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.out_proj(attn_output)


class SiglipMLP(nn.Module):
    """SigLIP MLP Block"""

    def __init__(self, hidden_size: int = 1152, intermediate_size: int = 4304):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class SiglipEncoderLayer(nn.Module):
    """Single SigLIP Encoder Layer"""

    def __init__(self, hidden_size: int = 1152, num_heads: int = 16, intermediate_size: int = 4304):
        super().__init__()
        self.self_attn = SiglipAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.mlp = SiglipMLP(hidden_size, intermediate_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention with residual
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class SiglipVisionTransformer(nn.Module):
    """
    Complete SigLIP Vision Transformer

    Based on MiniCPM-o-2_6 vision encoder specifications.
    """

    def __init__(
        self,
        image_size: int = 980,
        patch_size: int = 14,
        num_channels: int = 3,
        hidden_size: int = 1152,
        num_layers: int = 27,
        num_heads: int = 16,
        intermediate_size: int = 4304,
    ):
        super().__init__()

        # Embeddings
        self.embeddings = SiglipVisionEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
        )

        # Encoder layers
        self.encoder = nn.ModuleList(
            [SiglipEncoderLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)]
        )

        # Final layer norm
        self.post_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch_size, num_channels, image_size, image_size)
        Returns:
            last_hidden_state: (batch_size, num_patches + 1, hidden_size)
        """
        # Embed patches
        hidden_states = self.embeddings(pixel_values)

        # Encoder layers
        for layer in self.encoder:
            hidden_states = layer(hidden_states)

        # Final layer norm
        hidden_states = self.post_layernorm(hidden_states)

        return hidden_states


class MiniCPMResampler(nn.Module):
    """
    MiniCPM Resampler for modality alignment

    Based on MiniCPM-o-2_6 resampler architecture.
    """

    def __init__(
        self,
        num_queries: int = 32,
        embed_dim: int = 3584,
        num_heads: int = 16,
        kv_dim: int = 1152,  # SigLip hidden size
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim

        # Learnable queries
        self.query = nn.Parameter(torch.randn(1, num_queries, embed_dim))

        # KV projection (project to embed_dim first)
        self.kv_proj = nn.Linear(kv_dim, embed_dim)

        # Cross attention (now kv_dim == embed_dim after projection)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Layer norms
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(kv_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, kv_dim) - e.g., SigLip embeddings
        Returns:
            output: (batch_size, num_queries, embed_dim)
        """
        batch_size = x.shape[0]

        # Prepare queries
        q = self.query.repeat(batch_size, 1, 1)
        q = self.ln_q(q)

        # Prepare keys and values
        kv = self.ln_kv(x)
        kv = self.kv_proj(kv)

        # Cross attention
        output, _ = self.cross_attn(q, kv, kv)

        return output


# Test functions
def test_whisper_encoder():
    """Test Whisper encoder implementation"""
    print("🧪 Testing Whisper Encoder...")

    # Create model
    model = WhisperEncoder()

    # Test input (batch_size=1, channels=80, seq_len=3000)
    x = torch.randn(1, 80, 3000)

    with torch.no_grad():
        output = model(x)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print("✅ Whisper encoder test passed!")


def test_siglip_vision():
    """Test SigLip vision transformer"""
    print("🧪 Testing SigLip Vision Transformer...")

    # Create model
    model = SiglipVisionTransformer()

    # Test input (batch_size=1, channels=3, height=980, width=980)
    x = torch.randn(1, 3, 980, 980)

    with torch.no_grad():
        output = model(x)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print("✅ SigLip vision test passed!")


def test_resampler():
    """Test MiniCPM resampler"""
    print("🧪 Testing MiniCPM Resampler...")

    # Create resampler
    resampler = MiniCPMResampler()

    # Test input (batch_size=1, seq_len=4901, kv_dim=1152) - SigLip output
    x = torch.randn(1, 4901, 1152)

    with torch.no_grad():
        output = resampler(x)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print("✅ Resampler test passed!")


if __name__ == "__main__":
    print("🚀 Testing MiniCPM Components Implementation")
    test_whisper_encoder()
    print()
    test_siglip_vision()
    print()
    test_resampler()
    print("\n🎉 All component tests passed!")
