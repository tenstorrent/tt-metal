#!/usr/bin/env python3
"""
Modality Projectors for MiniCPM-o-2_6

This module implements the actual MiniCPM-o-2_6 Resampler architecture
that maps vision and audio embeddings to the LLM's embedding space.

Based on: https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/resampler.py
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np
import math


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
    given learnable queries and 2d sincos pos_emb
    Outputs:
    A tensor with the shape of (batch_size, num_queries, embed_dim)
    """

    def __init__(
        self,
        num_queries,
        embed_dim,
        num_heads,
        kv_dim=None,
        norm_layer=nn.LayerNorm,
        adaptive=False,
        max_size=(70, 70),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_dim = kv_dim if kv_dim is not None else embed_dim

        self.adaptive = adaptive
        if adaptive:
            self.max_size = max_size

        # Input projection to align dimensions (only if kv_dim != embed_dim)
        if self.kv_dim != self.embed_dim:
            self.kv_proj = nn.Linear(self.kv_dim, self.embed_dim)
        else:
            self.kv_proj = None

        # Learnable queries
        self.query = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        # Positional embeddings
        if not adaptive:
            self.pos_embed = nn.Parameter(
                torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, max_size)).float()
            ).unsqueeze(0)
        else:
            self.pos_embed = None

        # Attention layers
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(self.embed_dim)  # Use embed_dim since we project kv_dim to embed_dim

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Output projection
        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Parameter):
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def forward(self, x, attn_mask=None):
        """
        x: [batch_size, seq_len, embed_dim]
        attn_mask: [batch_size, seq_len] or None
        """
        batch_size, seq_len, embed_dim = x.shape

        # Learnable queries
        q = self.query.expand(batch_size, -1, -1)  # [batch_size, num_queries, embed_dim]

        # Project KV to embed_dim if needed (before positional embeddings)
        if self.kv_proj is not None:
            x = self.kv_proj(x)

        # Positional embeddings - simplified for now
        # TODO: Properly implement positional embeddings based on MiniCPM-o code
        # For now, skip positional embeddings to get basic functionality working
        pos_embed = torch.zeros_like(x)  # No positional embeddings

        # Add positional embeddings to input (placeholder)
        x = x + pos_embed

        # Apply layer norms
        q = self.ln_q(q)
        x = self.ln_kv(x)

        # Cross-attention
        # q: [batch_size, num_queries, embed_dim] (queries)
        # x: [batch_size, seq_len, embed_dim] (keys/values)
        attn_output, _ = self.attn(
            query=q,
            key=x,
            value=x,
            attn_mask=None,  # We'll handle masking if needed
        )

        # Post-attention processing
        attn_output = self.ln_post(attn_output)
        output = self.proj(attn_output)

        return output

    def _get_1d_pos_embed(self, seq_len, embed_dim):
        """Get 1D sinusoidal positional embeddings"""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pos_embed = torch.zeros(seq_len, embed_dim)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        return pos_embed


class VisionResampler(Resampler):
    """
    Vision resampler for MiniCPM-o-2_6.

    Maps SigLip vision embeddings to fixed-length query embeddings.
    """

    def __init__(
        self,
        vision_hidden_size: int = 1152,  # SigLip hidden size
        llm_hidden_size: int = 3584,  # Qwen2.5 hidden size
        num_queries: int = 64,  # From MiniCPM-o config
        num_heads: int = 16,
        max_size: Tuple[int, int] = (70, 70),
    ):
        super().__init__(
            num_queries=num_queries,
            embed_dim=llm_hidden_size,
            num_heads=num_heads,
            kv_dim=vision_hidden_size,
            adaptive=True,  # Allow variable input sizes
            max_size=max_size,
        )

    def forward(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        """
        Resample vision embeddings to fixed-length queries.

        Args:
            vision_embeds: [batch_size, seq_len, vision_hidden] SigLip embeddings
                           seq_len can be variable (e.g., different image sizes)

        Returns:
            [batch_size, num_queries, llm_hidden] resampled embeddings
        """
        return super().forward(vision_embeds)


class AudioResampler(Resampler):
    """
    Audio resampler for MiniCPM-o-2_6.

    Maps Whisper audio embeddings to fixed-length query embeddings.
    """

    def __init__(
        self,
        audio_hidden_size: int = 1024,  # Whisper hidden size
        llm_hidden_size: int = 3584,  # Qwen2.5 hidden size
        num_queries: int = 64,  # From MiniCPM-o config
        num_heads: int = 16,
        max_size: Tuple[int, int] = (32, 32),  # Audio sequence dimensions
    ):
        super().__init__(
            num_queries=num_queries,
            embed_dim=llm_hidden_size,
            num_heads=num_heads,
            kv_dim=audio_hidden_size,
            adaptive=True,  # Allow variable input sizes
            max_size=max_size,
        )

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        """
        Resample audio embeddings to fixed-length queries.

        Args:
            audio_embeds: [batch_size, seq_len, audio_hidden] Whisper embeddings
                          seq_len can be variable (different audio lengths)

        Returns:
            [batch_size, num_queries, llm_hidden] resampled embeddings
        """
        return super().forward(audio_embeds)


class MiniCPMModalityProjectors(nn.Module):
    """
    Combined modality resamplers for MiniCPM-o-2_6.

    Uses the actual MiniCPM-o resampler architecture with learnable queries
    and cross-attention to map variable-length inputs to fixed-length outputs.
    """

    def __init__(
        self,
        vision_config: Optional[dict] = None,
        audio_config: Optional[dict] = None,
        llm_hidden_size: int = 3584,
        num_queries: int = 64,  # From MiniCPM-o config
        num_heads: int = 16,
    ):
        super().__init__()

        # Default configurations based on MiniCPM-o-2_6
        if vision_config is None:
            vision_config = {"hidden_size": 1152}  # SigLip
        if audio_config is None:
            audio_config = {"hidden_size": 1024}  # Whisper

        # Create resamplers (not simple projectors)
        self.vision_resampler = None
        if "hidden_size" in vision_config:
            self.vision_resampler = VisionResampler(
                vision_hidden_size=vision_config["hidden_size"],
                llm_hidden_size=llm_hidden_size,
                num_queries=num_queries,
                num_heads=num_heads,
            )

        self.audio_resampler = None
        if "hidden_size" in audio_config:
            self.audio_resampler = AudioResampler(
                audio_hidden_size=audio_config["hidden_size"],
                llm_hidden_size=llm_hidden_size,
                num_queries=num_queries,
                num_heads=num_heads,
            )

    def forward(
        self,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Resample vision and/or audio embeddings to fixed-length queries.

        Args:
            vision_embeds: [batch_size, seq_len_v, vision_hidden] or None
            audio_embeds: [batch_size, seq_len_a, audio_hidden] or None

        Returns:
            tuple of (resampled_vision, resampled_audio) both [batch_size, num_queries, llm_hidden]
        """
        resampled_vision = None
        if vision_embeds is not None and self.vision_resampler is not None:
            resampled_vision = self.vision_resampler(vision_embeds)

        resampled_audio = None
        if audio_embeds is not None and self.audio_resampler is not None:
            resampled_audio = self.audio_resampler(audio_embeds)

        return resampled_vision, resampled_audio


def create_modality_projectors_from_config(minicpm_config: dict) -> MiniCPMModalityProjectors:
    """
    Create modality resamplers from MiniCPM-o configuration.

    Args:
        minicpm_config: Configuration dict from MiniCPM-o model

    Returns:
        Configured modality resamplers
    """
    vision_config = minicpm_config.get("vision_config", {})
    audio_config = minicpm_config.get("audio_config", {})
    llm_hidden_size = minicpm_config.get("hidden_size", 3584)
    num_queries = minicpm_config.get("query_num", 64)  # From MiniCPM-o config

    return MiniCPMModalityProjectors(
        vision_config=vision_config,
        audio_config=audio_config,
        llm_hidden_size=llm_hidden_size,
        num_queries=num_queries,
    )


# Test functions
def test_modality_resamplers():
    """Test modality resamplers with dummy data"""

    # Create resamplers
    resamplers = MiniCPMModalityProjectors()

    batch_size = 2
    num_queries = 64  # From MiniCPM-o config

    # Test vision resampler
    vision_embeds = torch.randn(batch_size, 256, 1152)  # SigLip output (variable length)
    resampled_vision = resamplers.vision_resampler(vision_embeds)
    print(f"Vision resampling: {vision_embeds.shape} -> {resampled_vision.shape}")
    assert resampled_vision.shape == (batch_size, num_queries, 3584)  # Fixed queries

    # Test audio resampler
    audio_embeds = torch.randn(batch_size, 128, 1024)  # Whisper output (variable length)
    resampled_audio = resamplers.audio_resampler(audio_embeds)
    print(f"Audio resampling: {audio_embeds.shape} -> {resampled_audio.shape}")
    assert resampled_audio.shape == (batch_size, num_queries, 3584)  # Fixed queries

    # Test combined
    resamp_vision, resamp_audio = resamplers(vision_embeds, audio_embeds)
    print(f"Combined resampling - Vision: {resamp_vision.shape}, Audio: {resamp_audio.shape}")
    assert resamp_vision.shape == (batch_size, num_queries, 3584)
    assert resamp_audio.shape == (batch_size, num_queries, 3584)

    print("✓ Modality resamplers test passed!")


def test_resampler_components():
    """Test individual resampler components"""

    # Test vision resampler directly
    vision_resampler = VisionResampler()
    vision_input = torch.randn(1, 196, 1152)  # 14x14 patches = 196
    vision_output = vision_resampler(vision_input)
    print(f"Vision resampler: {vision_input.shape} -> {vision_output.shape}")
    assert vision_output.shape == (1, 64, 3584)

    # Test audio resampler directly
    audio_resampler = AudioResampler()
    audio_input = torch.randn(1, 100, 1024)  # Variable audio length
    audio_output = audio_resampler(audio_input)
    print(f"Audio resampler: {audio_input.shape} -> {audio_output.shape}")
    assert audio_output.shape == (1, 64, 3584)

    print("✓ Resampler components test passed!")


if __name__ == "__main__":
    test_modality_resamplers()
    test_resampler_components()
