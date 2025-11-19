# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of MiniCPM-o-2_6 Resampler (Perceiver-style cross-attention).

Translates the PyTorch reference from reference_pytorch/minicpm_official/resampler.py.
"""

import torch
import ttnn
import numpy as np
from typing import Optional, Tuple, Dict
from loguru import logger

try:
    from .common import (
        get_weights_memory_config,
        get_activations_memory_config,
        torch_to_ttnn,
        ttnn_to_torch,
    )
except ImportError:
    from common import (
        get_weights_memory_config,
        get_activations_memory_config,
        torch_to_ttnn,
        ttnn_to_torch,
    )


def get_2d_sincos_pos_embed(embed_dim: int, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension
        image_size: (height, width) of the image grid

    Returns:
        np.ndarray: Positional embeddings of shape [height, width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    Generate 2D positional embeddings from grid coordinates.

    Args:
        embed_dim: Embedding dimension (must be even)
        grid: Grid coordinates of shape [2, H, W]

    Returns:
        np.ndarray: Positional embeddings of shape [H, W, embed_dim]
    """
    assert embed_dim % 2 == 0

    # Use half dimensions for height, half for width
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H, W, D/2]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H, W, D/2]

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # [H, W, D]
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate 1D sinusoidal positional embeddings.

    Args:
        embed_dim: Output dimension for each position (must be even)
        pos: Grid positions of shape [H, W]

    Returns:
        np.ndarray: Positional embeddings of shape [H, W, embed_dim]
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # [D/2,]

    out = np.einsum("hw,d->hwd", pos, omega)  # [H, W, D/2]

    emb_sin = np.sin(out)  # [H, W, D/2]
    emb_cos = np.cos(out)  # [H, W, D/2]

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # [H, W, D]
    return emb


class TtnnResampler:
    """
    TTNN implementation of Perceiver Resampler with cross-attention.

    Resamples input features (e.g., from vision encoder) to a fixed number of queries
    using learnable queries and 2D sinusoidal positional embeddings.

    Architecture:
        - Learnable queries: [num_queries, embed_dim]
        - Optional KV projection: [kv_dim, embed_dim]
        - Cross-attention: queries attend to input features
        - LayerNorm for queries and key/values
        - Final LayerNorm + projection

    Args:
        device: TTNN device
        num_queries: Number of learnable query tokens (default 64)
        embed_dim: Output embedding dimension (default 3584)
        num_heads: Number of attention heads (default 28)
        kv_dim: Input key/value dimension (default None, same as embed_dim)
        max_size: Maximum image size for positional embeddings (default [70, 70])
    """

    def __init__(
        self,
        device: ttnn.Device,
        num_queries: int = 64,
        embed_dim: int = 3584,
        num_heads: int = 28,
        kv_dim: Optional[int] = None,
        max_size: Tuple[int, int] = (70, 70),
    ):
        self.device = device
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kv_dim = kv_dim if kv_dim is not None else embed_dim
        self.max_size = max_size

        # Weights will be loaded later
        self.query = None  # Learnable queries [num_queries, embed_dim]
        self.kv_proj_weight = None  # Optional KV projection

        # Attention projections
        self.q_proj_weight = None
        self.q_proj_bias = None
        self.k_proj_weight = None
        self.k_proj_bias = None
        self.v_proj_weight = None
        self.v_proj_bias = None
        self.o_proj_weight = None
        self.o_proj_bias = None

        # Layer norms
        self.ln_q_weight = None
        self.ln_q_bias = None
        self.ln_kv_weight = None
        self.ln_kv_bias = None
        self.ln_post_weight = None
        self.ln_post_bias = None

        # Final projection
        self.proj_weight = None

        # Positional embeddings (cached as PyTorch tensor initially)
        # Positional embeddings should match key/value dimension (kv_dim)
        pos_embed = get_2d_sincos_pos_embed(self.kv_dim, max_size)
        self.pos_embed = torch.from_numpy(pos_embed).float()  # [H, W, D]

        logger.info(
            f"TtnnResampler initialized: num_queries={num_queries}, embed_dim={embed_dim}, "
            f"num_heads={num_heads}, kv_dim={self.kv_dim}"
        )

    def load_weights(self, weights_dict: dict):
        """
        Load weights from PyTorch state dict.

        Args:
            weights_dict: Dictionary containing weight tensors with keys:
                - 'query': Learnable queries
                - 'attn.q_proj.weight', 'attn.q_proj.bias': Query projection
                - 'attn.k_proj.weight', 'attn.k_proj.bias': Key projection
                - 'attn.v_proj.weight', 'attn.v_proj.bias': Value projection
                - 'attn.o_proj.weight', 'attn.o_proj.bias': Output projection
                - 'ln_q.weight', 'ln_q.bias': Query LayerNorm
                - 'ln_kv.weight', 'ln_kv.bias': Key/Value LayerNorm
                - 'ln_post.weight', 'ln_post.bias': Post-attention LayerNorm
                - 'proj': Final projection matrix
                - 'kv_proj.weight' (optional): KV projection if kv_dim != embed_dim
        """
        logger.info("Loading Resampler weights...")

        # Learnable queries
        self.query = torch_to_ttnn(
            weights_dict["query"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        # KV projection (optional)
        if "kv_proj.weight" in weights_dict:
            self.kv_proj_weight = torch_to_ttnn(
                weights_dict["kv_proj.weight"].transpose(-1, -2),  # Transpose for matmul
                self.device,
                memory_config=get_weights_memory_config(),
            )

        # Handle combined attention projection (official MiniCPM-o-2_6 format)
        if "attn.in_proj_weight" in weights_dict:
            # Split combined weight [3*embed_dim, embed_dim] into Q, K, V
            in_proj = weights_dict["attn.in_proj_weight"]
            weights_dict["attn.q_proj.weight"] = in_proj[: self.embed_dim, :]
            weights_dict["attn.k_proj.weight"] = in_proj[self.embed_dim : 2 * self.embed_dim, :]
            weights_dict["attn.v_proj.weight"] = in_proj[2 * self.embed_dim :, :]
            logger.debug(f"Split combined attn.in_proj_weight into Q/K/V projections")

        if "attn.in_proj_bias" in weights_dict:
            # Split combined bias [3*embed_dim] into Q, K, V
            in_proj_bias = weights_dict["attn.in_proj_bias"]
            weights_dict["attn.q_proj.bias"] = in_proj_bias[: self.embed_dim]
            weights_dict["attn.k_proj.bias"] = in_proj_bias[self.embed_dim : 2 * self.embed_dim]
            weights_dict["attn.v_proj.bias"] = in_proj_bias[2 * self.embed_dim :]
            logger.debug(f"Split combined attn.in_proj_bias into Q/K/V biases")

        # Attention projections
        self.q_proj_weight = torch_to_ttnn(
            weights_dict["attn.q_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        self.q_proj_bias = torch_to_ttnn(
            weights_dict["attn.q_proj.bias"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        self.k_proj_weight = torch_to_ttnn(
            weights_dict["attn.k_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        self.k_proj_bias = torch_to_ttnn(
            weights_dict["attn.k_proj.bias"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        self.v_proj_weight = torch_to_ttnn(
            weights_dict["attn.v_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        self.v_proj_bias = torch_to_ttnn(
            weights_dict["attn.v_proj.bias"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        self.o_proj_weight = torch_to_ttnn(
            weights_dict.get("attn.o_proj.weight", weights_dict.get("attn.out_proj.weight")).transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        if "attn.o_proj.bias" in weights_dict:
            self.o_proj_bias = torch_to_ttnn(
                weights_dict.get("attn.o_proj.bias", weights_dict.get("attn.out_proj.bias")),
                self.device,
                memory_config=get_weights_memory_config(),
            )

        # Layer norms
        self.ln_q_weight = torch_to_ttnn(
            weights_dict["ln_q.weight"],
            self.device,
            memory_config=get_weights_memory_config(),
        )
        self.ln_q_bias = torch_to_ttnn(
            weights_dict["ln_q.bias"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        self.ln_kv_weight = torch_to_ttnn(
            weights_dict["ln_kv.weight"],
            self.device,
            memory_config=get_weights_memory_config(),
        )
        self.ln_kv_bias = torch_to_ttnn(
            weights_dict["ln_kv.bias"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        self.ln_post_weight = torch_to_ttnn(
            weights_dict["ln_post.weight"],
            self.device,
            memory_config=get_weights_memory_config(),
        )
        self.ln_post_bias = torch_to_ttnn(
            weights_dict["ln_post.bias"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        # Final projection
        self.proj_weight = torch_to_ttnn(
            weights_dict["proj"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        logger.info("✅ Resampler weights loaded")

    def __call__(
        self,
        x: ttnn.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Forward pass of the resampler.

        Args:
            x: Input features of shape [batch_size, seq_len, kv_dim]
            tgt_sizes: Target sizes for each batch element, shape [batch_size, 2] (height, width)

        Returns:
            ttnn.Tensor: Resampled features of shape [batch_size, num_queries, embed_dim]
        """
        batch_size = x.shape[0]

        # Apply KV projection if needed
        if self.kv_proj_weight is not None:
            x = ttnn.matmul(x, self.kv_proj_weight)

        # Apply LayerNorm to keys/values
        x = ttnn.layer_norm(
            x,
            weight=self.ln_kv_weight,
            bias=self.ln_kv_bias,
        )

        # Prepare positional embeddings
        # Convert tgt_sizes to CPU for numpy operations
        tgt_sizes_cpu = tgt_sizes.cpu().numpy() if isinstance(tgt_sizes, torch.Tensor) else tgt_sizes

        # Create positional embeddings for each batch element
        pos_embeds = []
        max_patch_len = 0
        for i in range(batch_size):
            tgt_h, tgt_w = int(tgt_sizes_cpu[i, 0]), int(tgt_sizes_cpu[i, 1])
            patch_len = tgt_h * tgt_w
            max_patch_len = max(max_patch_len, patch_len)

            # Extract positional embeddings for this size
            pos_embed_i = self.pos_embed[:tgt_h, :tgt_w, :].reshape(patch_len, -1)
            pos_embeds.append(pos_embed_i)

        # Pad positional embeddings to max_patch_len
        pos_embeds_padded = []
        for pos_embed in pos_embeds:
            if pos_embed.shape[0] < max_patch_len:
                pad_size = max_patch_len - pos_embed.shape[0]
                padding = torch.zeros(pad_size, pos_embed.shape[1], dtype=pos_embed.dtype)
                pos_embed = torch.cat([pos_embed, padding], dim=0)
            pos_embeds_padded.append(pos_embed)

        # Stack and prepare positional embeddings: [batch_size, max_patch_len, kv_dim]
        pos_embeds_tensor = torch.stack(pos_embeds_padded, dim=0)

        # Prepend CLS token positional (zeros) to match SigLip outputs which include CLS at index 0
        cls_padding = torch.zeros((batch_size, 1, pos_embeds_tensor.shape[-1]), dtype=pos_embeds_tensor.dtype)
        pos_with_cls = torch.cat([cls_padding, pos_embeds_tensor], dim=1)  # [B, 1 + max_patch_len, kv_dim]

        # Trim or pad to match actual seq_len if necessary
        if pos_with_cls.shape[1] > max_patch_len + 1:
            pos_with_cls = pos_with_cls[:, : (max_patch_len + 1), :]

        # If keys have different seq_len, we'll slice later after conversion when shapes are known.
        tt_pos_embed = torch_to_ttnn(
            pos_with_cls,
            self.device,
            memory_config=get_activations_memory_config(),
        )

        # Add positional embeddings to keys (slice tt_pos_embed to match x.seq_len if needed)
        try:
            seq_len = x.shape[1]
            # tt_pos_embed may be [B, 1+max_patch_len, kv_dim]; slice to seq_len
            tt_pos = tt_pos_embed[:, :seq_len, :]
        except Exception:
            tt_pos = tt_pos_embed

        x_with_pos = ttnn.add(x, tt_pos)

        # Prepare queries: [num_queries, embed_dim] -> [batch_size, num_queries, embed_dim]
        # Apply LayerNorm to queries
        queries_normalized = ttnn.layer_norm(
            self.query,
            weight=self.ln_q_weight,
            bias=self.ln_q_bias,
        )

        # Repeat queries for batch
        # queries_normalized shape: [num_queries, embed_dim]
        # Need to expand to [batch_size, num_queries, embed_dim]
        queries_normalized_torch = ttnn_to_torch(queries_normalized)
        queries_batched = queries_normalized_torch.unsqueeze(0).repeat(batch_size, 1, 1)
        tt_queries_batched = torch_to_ttnn(
            queries_batched,
            self.device,
            memory_config=get_activations_memory_config(),
        )

        # Cross-attention: queries attend to keys/values
        attention_output = self._cross_attention(
            queries=tt_queries_batched,  # [B, num_queries, embed_dim]
            keys_values=x_with_pos,  # [B, seq_len, embed_dim]
            values=x,  # [B, seq_len, embed_dim] (without pos embed)
        )

        # Apply post-attention LayerNorm
        attention_output = ttnn.layer_norm(
            attention_output,
            weight=self.ln_post_weight,
            bias=self.ln_post_bias,
        )

        # Final projection
        output = ttnn.matmul(attention_output, self.proj_weight)

        return output

    def _cross_attention(
        self,
        queries: ttnn.Tensor,
        keys_values: ttnn.Tensor,
        values: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Perform multi-head cross-attention.

        Args:
            queries: Query tensor [batch_size, num_queries, embed_dim]
            keys_values: Key tensor with positional embeddings [batch_size, seq_len, embed_dim]
            values: Value tensor without positional embeddings [batch_size, seq_len, embed_dim]

        Returns:
            ttnn.Tensor: Attention output [batch_size, num_queries, embed_dim]
        """
        batch_size, num_queries, _ = queries.shape
        _, seq_len, _ = keys_values.shape

        # Project queries, keys, values
        q = ttnn.matmul(queries, self.q_proj_weight)
        q = ttnn.add(q, self.q_proj_bias)

        k = ttnn.matmul(keys_values, self.k_proj_weight)
        k = ttnn.add(k, self.k_proj_bias)

        v = ttnn.matmul(values, self.v_proj_weight)
        v = ttnn.add(v, self.v_proj_bias)

        # Reshape for multi-head attention: [B, seq, embed_dim] -> [B, seq, num_heads, head_dim]
        # Then permute to [B, num_heads, seq, head_dim]
        q = ttnn.to_layout(q, layout=ttnn.ROW_MAJOR_LAYOUT)
        q = ttnn.reshape(q, (batch_size, num_queries, self.num_heads, self.head_dim))
        q = ttnn.to_layout(q, layout=ttnn.TILE_LAYOUT)
        q = ttnn.permute(q, (0, 2, 1, 3))  # [B, num_heads, num_queries, head_dim]

        k = ttnn.to_layout(k, layout=ttnn.ROW_MAJOR_LAYOUT)
        k = ttnn.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ttnn.to_layout(k, layout=ttnn.TILE_LAYOUT)
        k = ttnn.permute(k, (0, 2, 3, 1))  # [B, num_heads, head_dim, seq_len] for matmul

        v = ttnn.to_layout(v, layout=ttnn.ROW_MAJOR_LAYOUT)
        v = ttnn.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ttnn.to_layout(v, layout=ttnn.TILE_LAYOUT)
        v = ttnn.permute(v, (0, 2, 1, 3))  # [B, num_heads, seq_len, head_dim]

        # Compute attention scores: Q @ K^T
        # [B, num_heads, num_queries, head_dim] @ [B, num_heads, head_dim, seq_len]
        # -> [B, num_heads, num_queries, seq_len]
        attention_scores = ttnn.matmul(q, k)

        # Scale by sqrt(head_dim)
        scale = 1.0 / (self.head_dim**0.5)
        attention_scores = ttnn.multiply(attention_scores, scale)

        # Apply softmax
        attention_probs = ttnn.softmax(attention_scores, dim=-1)

        # Apply attention to values: attention_probs @ V
        # [B, num_heads, num_queries, seq_len] @ [B, num_heads, seq_len, head_dim]
        # -> [B, num_heads, num_queries, head_dim]
        context = ttnn.matmul(attention_probs, v)

        # Reshape back: [B, num_heads, num_queries, head_dim] -> [B, num_queries, embed_dim]
        context = ttnn.permute(context, (0, 2, 1, 3))  # [B, num_queries, num_heads, head_dim]
        context = ttnn.to_layout(context, layout=ttnn.ROW_MAJOR_LAYOUT)
        context = ttnn.reshape(context, (batch_size, num_queries, self.embed_dim))
        context = ttnn.to_layout(context, layout=ttnn.TILE_LAYOUT)

        # Output projection
        output = ttnn.matmul(context, self.o_proj_weight)
        if self.o_proj_bias is not None:
            output = ttnn.add(output, self.o_proj_bias)

        return output


"""
TTNN implementation of MiniCPM-o-2_6 Resampler (Perceiver-style cross-attention).

Translates the PyTorch reference from reference_pytorch/minicpm_official/resampler.py.
"""

import torch
import ttnn
import numpy as np
from typing import Optional, Tuple, Dict
from loguru import logger

try:
    from .common import (
        get_weights_memory_config,
        get_activations_memory_config,
        torch_to_ttnn,
        ttnn_to_torch,
    )
except ImportError:
    from common import (
        get_weights_memory_config,
        get_activations_memory_config,
        torch_to_ttnn,
        ttnn_to_torch,
    )


def get_2d_sincos_pos_embed(embed_dim: int, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension
        image_size: (height, width) of the image grid

    Returns:
        np.ndarray: Positional embeddings of shape [height, width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    Generate 2D positional embeddings from grid coordinates.

    Args:
        embed_dim: Embedding dimension (must be even)
        grid: Grid coordinates of shape [2, H, W]

    Returns:
        np.ndarray: Positional embeddings of shape [H, W, embed_dim]
    """
    assert embed_dim % 2 == 0

    # Use half dimensions for height, half for width
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H, W, D/2]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H, W, D/2]

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # [H, W, D]
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate 1D sinusoidal positional embeddings.

    Args:
        embed_dim: Output dimension for each position (must be even)
        pos: Grid positions of shape [H, W]

    Returns:
        np.ndarray: Positional embeddings of shape [H, W, embed_dim]
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # [D/2,]

    out = np.einsum("hw,d->hwd", pos, omega)  # [H, W, D/2]

    emb_sin = np.sin(out)  # [H, W, D/2]
    emb_cos = np.cos(out)  # [H, W, D/2]

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # [H, W, D]
    return emb


class TtnnResampler:
    """
    TTNN implementation of Perceiver Resampler with cross-attention.

    Resamples input features (e.g., from vision encoder) to a fixed number of queries
    using learnable queries and 2D sinusoidal positional embeddings.

    Architecture:
        - Learnable queries: [num_queries, embed_dim]
        - Optional KV projection: [kv_dim, embed_dim]
        - Cross-attention: queries attend to input features
        - LayerNorm for queries and key/values
        - Final LayerNorm + projection

    Args:
        device: TTNN device
        num_queries: Number of learnable query tokens (default 64)
        embed_dim: Output embedding dimension (default 3584)
        num_heads: Number of attention heads (default 28)
        kv_dim: Input key/value dimension (default None, same as embed_dim)
        max_size: Maximum image size for positional embeddings (default [70, 70])
    """

    def __init__(
        self,
        device: ttnn.Device,
        num_queries: int = 64,
        embed_dim: int = 3584,
        num_heads: int = 28,
        kv_dim: Optional[int] = None,
        max_size: Tuple[int, int] = (70, 70),
    ):
        self.device = device
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kv_dim = kv_dim if kv_dim is not None else embed_dim
        self.max_size = max_size

        # Weights will be loaded later
        self.query = None  # Learnable queries [num_queries, embed_dim]
        self.kv_proj_weight = None  # Optional KV projection

        # Attention projections
        self.q_proj_weight = None
        self.q_proj_bias = None
        self.k_proj_weight = None
        self.k_proj_bias = None
        self.v_proj_weight = None
        self.v_proj_bias = None
        self.o_proj_weight = None
        self.o_proj_bias = None

        # Layer norms
        self.ln_q_weight = None
        self.ln_q_bias = None
        self.ln_kv_weight = None
        self.ln_kv_bias = None
        self.ln_post_weight = None
        self.ln_post_bias = None

        # Final projection
        self.proj_weight = None

        # Positional embeddings (cached as PyTorch tensor initially)
        # Positional embeddings should match key/value dimension (kv_dim)
        pos_embed = get_2d_sincos_pos_embed(self.kv_dim, max_size)
        self.pos_embed = torch.from_numpy(pos_embed).float()  # [H, W, D]

        logger.info(
            f"TtnnResampler initialized: num_queries={num_queries}, embed_dim={embed_dim}, "
            f"num_heads={num_heads}, kv_dim={self.kv_dim}"
        )

    def load_weights(self, weights_dict: dict):
        """
        Load weights from PyTorch state dict.

        Args:
            weights_dict: Dictionary containing weight tensors with keys:
                - 'query': Learnable queries
                - 'attn.q_proj.weight', 'attn.q_proj.bias': Query projection
                - 'attn.k_proj.weight', 'attn.k_proj.bias': Key projection
                - 'attn.v_proj.weight', 'attn.v_proj.bias': Value projection
                - 'attn.o_proj.weight', 'attn.o_proj.bias': Output projection
                - 'ln_q.weight', 'ln_q.bias': Query LayerNorm
                - 'ln_kv.weight', 'ln_kv.bias': Key/Value LayerNorm
                - 'ln_post.weight', 'ln_post.bias': Post-attention LayerNorm
                - 'proj': Final projection matrix
                - 'kv_proj.weight' (optional): KV projection if kv_dim != embed_dim
        """
        logger.info("Loading Resampler weights...")

        # Learnable queries
        self.query = torch_to_ttnn(
            weights_dict["query"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        # KV projection (optional)
        if "kv_proj.weight" in weights_dict:
            self.kv_proj_weight = torch_to_ttnn(
                weights_dict["kv_proj.weight"].transpose(-1, -2),  # Transpose for matmul
                self.device,
                memory_config=get_weights_memory_config(),
            )

        # Handle combined attention projection (official MiniCPM-o-2_6 format)
        if "attn.in_proj_weight" in weights_dict:
            # Split combined weight [3*embed_dim, embed_dim] into Q, K, V
            in_proj = weights_dict["attn.in_proj_weight"]
            weights_dict["attn.q_proj.weight"] = in_proj[: self.embed_dim, :]
            weights_dict["attn.k_proj.weight"] = in_proj[self.embed_dim : 2 * self.embed_dim, :]
            weights_dict["attn.v_proj.weight"] = in_proj[2 * self.embed_dim :, :]
            logger.debug(f"Split combined attn.in_proj_weight into Q/K/V projections")

        if "attn.in_proj_bias" in weights_dict:
            # Split combined bias [3*embed_dim] into Q, K, V
            in_proj_bias = weights_dict["attn.in_proj_bias"]
            weights_dict["attn.q_proj.bias"] = in_proj_bias[: self.embed_dim]
            weights_dict["attn.k_proj.bias"] = in_proj_bias[self.embed_dim : 2 * self.embed_dim]
            weights_dict["attn.v_proj.bias"] = in_proj_bias[2 * self.embed_dim :]
            logger.debug(f"Split combined attn.in_proj_bias into Q/K/V biases")

        # Attention projections
        self.q_proj_weight = torch_to_ttnn(
            weights_dict["attn.q_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        self.q_proj_bias = torch_to_ttnn(
            weights_dict["attn.q_proj.bias"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        self.k_proj_weight = torch_to_ttnn(
            weights_dict["attn.k_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        self.k_proj_bias = torch_to_ttnn(
            weights_dict["attn.k_proj.bias"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        self.v_proj_weight = torch_to_ttnn(
            weights_dict["attn.v_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        self.v_proj_bias = torch_to_ttnn(
            weights_dict["attn.v_proj.bias"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        self.o_proj_weight = torch_to_ttnn(
            weights_dict["attn.o_proj.weight"].transpose(-1, -2),
            self.device,
            memory_config=get_weights_memory_config(),
        )
        if "attn.o_proj.bias" in weights_dict:
            self.o_proj_bias = torch_to_ttnn(
                weights_dict["attn.o_proj.bias"],
                self.device,
                memory_config=get_weights_memory_config(),
            )

        # Layer norms
        self.ln_q_weight = torch_to_ttnn(
            weights_dict["ln_q.weight"],
            self.device,
            memory_config=get_weights_memory_config(),
        )
        self.ln_q_bias = torch_to_ttnn(
            weights_dict["ln_q.bias"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        self.ln_kv_weight = torch_to_ttnn(
            weights_dict["ln_kv.weight"],
            self.device,
            memory_config=get_weights_memory_config(),
        )
        self.ln_kv_bias = torch_to_ttnn(
            weights_dict["ln_kv.bias"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        self.ln_post_weight = torch_to_ttnn(
            weights_dict["ln_post.weight"],
            self.device,
            memory_config=get_weights_memory_config(),
        )
        self.ln_post_bias = torch_to_ttnn(
            weights_dict["ln_post.bias"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        # Final projection
        self.proj_weight = torch_to_ttnn(
            weights_dict["proj"],
            self.device,
            memory_config=get_weights_memory_config(),
        )

        logger.info("✅ Resampler weights loaded")

    def __call__(
        self,
        x: ttnn.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Forward pass of the resampler.

        Args:
            x: Input features of shape [batch_size, seq_len, kv_dim]
            tgt_sizes: Target sizes for each batch element, shape [batch_size, 2] (height, width)

        Returns:
            ttnn.Tensor: Resampled features of shape [batch_size, num_queries, embed_dim]
        """
        batch_size = x.shape[0]

        # Apply KV projection if needed
        if self.kv_proj_weight is not None:
            x = ttnn.matmul(x, self.kv_proj_weight)

        # Apply LayerNorm to keys/values
        x = ttnn.layer_norm(
            x,
            weight=self.ln_kv_weight,
            bias=self.ln_kv_bias,
        )

        # Prepare positional embeddings
        # Convert tgt_sizes to CPU for numpy operations
        tgt_sizes_cpu = tgt_sizes.cpu().numpy() if isinstance(tgt_sizes, torch.Tensor) else tgt_sizes

        # Create positional embeddings for each batch element
        pos_embeds = []
        max_patch_len = 0
        for i in range(batch_size):
            tgt_h, tgt_w = int(tgt_sizes_cpu[i, 0]), int(tgt_sizes_cpu[i, 1])
            patch_len = tgt_h * tgt_w
            max_patch_len = max(max_patch_len, patch_len)

            # Extract positional embeddings for this size
            pos_embed_i = self.pos_embed[:tgt_h, :tgt_w, :].reshape(patch_len, -1)
            pos_embeds.append(pos_embed_i)

        # Pad positional embeddings to max_patch_len
        pos_embeds_padded = []
        for pos_embed in pos_embeds:
            if pos_embed.shape[0] < max_patch_len:
                pad_size = max_patch_len - pos_embed.shape[0]
                padding = torch.zeros(pad_size, pos_embed.shape[1], dtype=pos_embed.dtype)
                pos_embed = torch.cat([pos_embed, padding], dim=0)
            pos_embeds_padded.append(pos_embed)

        # Stack and prepare positional embeddings: [batch_size, max_patch_len, kv_dim]
        pos_embeds_tensor = torch.stack(pos_embeds_padded, dim=0)

        # Prepend CLS token positional (zeros) to match SigLip outputs which include CLS at index 0
        cls_padding = torch.zeros((batch_size, 1, pos_embeds_tensor.shape[-1]), dtype=pos_embeds_tensor.dtype)
        pos_with_cls = torch.cat([cls_padding, pos_embeds_tensor], dim=1)  # [B, 1 + max_patch_len, kv_dim]

        # Trim or pad to match actual seq_len if necessary
        if pos_with_cls.shape[1] > max_patch_len + 1:
            pos_with_cls = pos_with_cls[:, : (max_patch_len + 1), :]

        # If keys have different seq_len, we'll slice later after conversion when shapes are known.
        tt_pos_embed = torch_to_ttnn(
            pos_with_cls,
            self.device,
            memory_config=get_activations_memory_config(),
        )

        # Add positional embeddings to keys (slice tt_pos_embed to match x.seq_len if needed)
        try:
            seq_len = x.shape[1]
            # tt_pos_embed may be [B, 1+max_patch_len, kv_dim]; slice to seq_len
            tt_pos = tt_pos_embed[:, :seq_len, :]
        except Exception:
            tt_pos = tt_pos_embed

        x_with_pos = ttnn.add(x, tt_pos)

        # Prepare queries: [num_queries, embed_dim] -> [batch_size, num_queries, embed_dim]
        # Apply LayerNorm to queries
        queries_normalized = ttnn.layer_norm(
            self.query,
            weight=self.ln_q_weight,
            bias=self.ln_q_bias,
        )

        # Repeat queries for batch
        # queries_normalized shape: [num_queries, embed_dim]
        # Need to expand to [batch_size, num_queries, embed_dim]
        queries_normalized_torch = ttnn_to_torch(queries_normalized)
        queries_batched = queries_normalized_torch.unsqueeze(0).repeat(batch_size, 1, 1)
        tt_queries_batched = torch_to_ttnn(
            queries_batched,
            self.device,
            memory_config=get_activations_memory_config(),
        )

        # Cross-attention: queries attend to keys/values
        attention_output = self._cross_attention(
            queries=tt_queries_batched,  # [B, num_queries, embed_dim]
            keys_values=x_with_pos,  # [B, seq_len, embed_dim]
            values=x,  # [B, seq_len, embed_dim] (without pos embed)
        )

        # Apply post-attention LayerNorm
        attention_output = ttnn.layer_norm(
            attention_output,
            weight=self.ln_post_weight,
            bias=self.ln_post_bias,
        )

        # Final projection
        output = ttnn.matmul(attention_output, self.proj_weight)

        return output

    def _cross_attention(
        self,
        queries: ttnn.Tensor,
        keys_values: ttnn.Tensor,
        values: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Perform multi-head cross-attention.

        Args:
            queries: Query tensor [batch_size, num_queries, embed_dim]
            keys_values: Key tensor with positional embeddings [batch_size, seq_len, embed_dim]
            values: Value tensor without positional embeddings [batch_size, seq_len, embed_dim]

        Returns:
            ttnn.Tensor: Attention output [batch_size, num_queries, embed_dim]
        """
        batch_size, num_queries, _ = queries.shape
        _, seq_len, _ = keys_values.shape

        # Project queries, keys, values
        q = ttnn.matmul(queries, self.q_proj_weight)
        q = ttnn.add(q, self.q_proj_bias)

        k = ttnn.matmul(keys_values, self.k_proj_weight)
        k = ttnn.add(k, self.k_proj_bias)

        v = ttnn.matmul(values, self.v_proj_weight)
        v = ttnn.add(v, self.v_proj_bias)

        # Reshape for multi-head attention: [B, seq, embed_dim] -> [B, seq, num_heads, head_dim]
        # Then permute to [B, num_heads, seq, head_dim]
        q = ttnn.to_layout(q, layout=ttnn.ROW_MAJOR_LAYOUT)
        q = ttnn.reshape(q, (batch_size, num_queries, self.num_heads, self.head_dim))
        q = ttnn.to_layout(q, layout=ttnn.TILE_LAYOUT)
        q = ttnn.permute(q, (0, 2, 1, 3))  # [B, num_heads, num_queries, head_dim]

        k = ttnn.to_layout(k, layout=ttnn.ROW_MAJOR_LAYOUT)
        k = ttnn.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ttnn.to_layout(k, layout=ttnn.TILE_LAYOUT)
        k = ttnn.permute(k, (0, 2, 3, 1))  # [B, num_heads, head_dim, seq_len] for matmul

        v = ttnn.to_layout(v, layout=ttnn.ROW_MAJOR_LAYOUT)
        v = ttnn.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ttnn.to_layout(v, layout=ttnn.TILE_LAYOUT)
        v = ttnn.permute(v, (0, 2, 1, 3))  # [B, num_heads, seq_len, head_dim]

        # Compute attention scores: Q @ K^T
        # [B, num_heads, num_queries, head_dim] @ [B, num_heads, head_dim, seq_len]
        # -> [B, num_heads, num_queries, seq_len]
        attention_scores = ttnn.matmul(q, k)

        # Scale by sqrt(head_dim)
        scale = 1.0 / (self.head_dim**0.5)
        attention_scores = ttnn.multiply(attention_scores, scale)

        # Apply softmax
        attention_probs = ttnn.softmax(attention_scores, dim=-1)

        # Apply attention to values: attention_probs @ V
        # [B, num_heads, num_queries, seq_len] @ [B, num_heads, seq_len, head_dim]
        # -> [B, num_heads, num_queries, head_dim]
        context = ttnn.matmul(attention_probs, v)

        # Reshape back: [B, num_heads, num_queries, head_dim] -> [B, num_queries, embed_dim]
        context = ttnn.permute(context, (0, 2, 1, 3))  # [B, num_queries, num_heads, head_dim]
        context = ttnn.to_layout(context, layout=ttnn.ROW_MAJOR_LAYOUT)
        context = ttnn.reshape(context, (batch_size, num_queries, self.embed_dim))
        context = ttnn.to_layout(context, layout=ttnn.TILE_LAYOUT)

        # Output projection
        output = ttnn.matmul(context, self.o_proj_weight)
        if self.o_proj_bias is not None:
            output = ttnn.add(output, self.o_proj_bias)

        return output


class TtnnVisionResampler:
    """
    TTNN Vision Resampler with Weight Loading

    Wrapper around TtnnResampler that handles weight loading
    from MiniCPM safetensors format for vision processing.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        weights: Dict[str, torch.Tensor],
        num_queries: int = 32,
        embed_dim: int = 3584,
        kv_dim: int = 1152,
    ):
        """
        Initialize vision resampler with loaded weights.

        Args:
            mesh_device: TTNN mesh device
            weights: Pre-loaded weights from MiniCPM checkpoint
            num_queries: Number of output query tokens (default 32 for vision)
            embed_dim: Output embedding dimension (default 3584, Qwen hidden size)
            kv_dim: Input key/value dimension (default 1152, SigLIP hidden size)
        """
        self.mesh_device = mesh_device
        self.weights = weights
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim

        # Create the underlying resampler
        self.resampler = TtnnResampler(
            device=mesh_device,
            num_queries=num_queries,
            embed_dim=embed_dim,
            kv_dim=kv_dim,
        )

        # Load weights into TTNN format
        self._load_weights(weights)

    def _load_weights(self, weights: Dict[str, torch.Tensor]):
        """
        Load PyTorch weights into TTNN tensors and move to device.

        This converts the safetensors weights to TTNN format and loads them
        into the resampler components.
        """
        logger.info("Loading vision resampler weights into TTNN format...")

        # Load query embeddings (learnable queries)
        if "resampler.query" in weights:
            query_tensor = weights["resampler.query"]
            query_tensor = ttnn.from_torch(
                query_tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            )
            query_tensor = ttnn.to_device(query_tensor, self.mesh_device)
            self.resampler.query = query_tensor

        # Load KV projection weights (if present)
        if "resampler.kv_proj.weight" in weights:
            kv_proj_weight = weights["resampler.kv_proj.weight"]
            kv_proj_weight = ttnn.from_torch(
                kv_proj_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            )
            kv_proj_weight = ttnn.to_device(kv_proj_weight, self.mesh_device)
            self.resampler.kv_proj_weight = kv_proj_weight

        if "resampler.kv_proj.bias" in weights:
            kv_proj_bias = weights["resampler.kv_proj.bias"]
            kv_proj_bias = ttnn.from_torch(
                kv_proj_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            )
            kv_proj_bias = ttnn.to_device(kv_proj_bias, self.mesh_device)
            self.resampler.kv_proj_bias = kv_proj_bias

        # Load attention projection weights
        attention_projs = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for proj_name in attention_projs:
            weight_key = f"resampler.attn.{proj_name}.weight"
            bias_key = f"resampler.attn.{proj_name}.bias"

            if weight_key in weights:
                weight_tensor = weights[weight_key]
                weight_tensor = ttnn.from_torch(
                    weight_tensor,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                )
                weight_tensor = ttnn.to_device(weight_tensor, self.mesh_device)
                setattr(self.resampler, f"{proj_name}_weight", weight_tensor)

            if bias_key in weights:
                bias_tensor = weights[bias_key]
                bias_tensor = ttnn.from_torch(
                    bias_tensor,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                )
                bias_tensor = ttnn.to_device(bias_tensor, self.mesh_device)
                setattr(self.resampler, f"{proj_name}_bias", bias_tensor)

        # Load layer norms
        layer_norms = ["ln_q", "ln_kv", "ln_post"]
        for ln_name in layer_norms:
            weight_key = f"resampler.{ln_name}.weight"
            bias_key = f"resampler.{ln_name}.bias"

            if weight_key in weights:
                weight_tensor = weights[weight_key]
                weight_tensor = ttnn.from_torch(
                    weight_tensor,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                )
                weight_tensor = ttnn.to_device(weight_tensor, self.mesh_device)
                setattr(self.resampler, f"{ln_name}_weight", weight_tensor)

            if bias_key in weights:
                bias_tensor = weights[bias_key]
                bias_tensor = ttnn.from_torch(
                    bias_tensor,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                )
                bias_tensor = ttnn.to_device(bias_tensor, self.mesh_device)
                setattr(self.resampler, f"{ln_name}_bias", bias_tensor)

        # Load final projection weights
        if "resampler.proj.weight" in weights:
            proj_weight = weights["resampler.proj.weight"]
            proj_weight = ttnn.from_torch(
                proj_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            )
            proj_weight = ttnn.to_device(proj_weight, self.mesh_device)
            self.resampler.proj_weight = proj_weight

        if "resampler.proj.bias" in weights:
            proj_bias = weights["resampler.proj.bias"]
            proj_bias = ttnn.from_torch(
                proj_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            )
            proj_bias = ttnn.to_device(proj_bias, self.mesh_device)
            self.resampler.proj_bias = proj_bias

        logger.info("✅ Vision resampler weights loaded into TTNN format")

    def __call__(self, vision_features: torch.Tensor, tgt_sizes: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward pass through vision resampler.

        Args:
            vision_features: Vision embeddings [batch, seq_len, kv_dim]
            tgt_sizes: Optional target sizes for positional embeddings

        Returns:
            Resampled tokens [batch, num_queries, embed_dim]
        """
        return self.resampler(vision_features, tgt_sizes)
