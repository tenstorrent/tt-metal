# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Text Encoder for TTS pipeline.

Encodes phoneme sequences into hidden representations using
embedding + transformer encoder.
"""

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

from models.demos.openvoice.tt.modules.conv1d import ttnn_conv1d


def _ensure_conv1d_weight(w):
    """Ensure weight tensor has correct shape for F.conv1d [out, in, kernel]."""
    if w is None:
        return None
    if w.dim() == 2:
        return w.unsqueeze(2)
    return w


class MultiHeadAttention:
    """
    Multi-head attention with relative positional encoding.

    Used in the TextEncoder's transformer blocks.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        window_size: Optional[int] = None,
        conv_q_weight: Any = None,
        conv_q_bias: Any = None,
        conv_k_weight: Any = None,
        conv_k_bias: Any = None,
        conv_v_weight: Any = None,
        conv_v_bias: Any = None,
        conv_o_weight: Any = None,
        conv_o_bias: Any = None,
        emb_rel_k: Optional[Any] = None,
        emb_rel_v: Optional[Any] = None,
        device: Optional[Any] = None,
    ):
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.device = device

        # Projection weights
        self.conv_q_weight = conv_q_weight
        self.conv_q_bias = conv_q_bias
        self.conv_k_weight = conv_k_weight
        self.conv_k_bias = conv_k_bias
        self.conv_v_weight = conv_v_weight
        self.conv_v_bias = conv_v_bias
        self.conv_o_weight = conv_o_weight
        self.conv_o_bias = conv_o_bias

        # Relative position embeddings
        self.emb_rel_k = emb_rel_k
        self.emb_rel_v = emb_rel_v

    def __call__(self, x: Any, c: Any, attn_mask: Optional[Any] = None) -> Any:
        """
        Forward pass.

        Args:
            x: Query input [B, C, T]
            c: Key/Value input [B, C, T]
            attn_mask: Attention mask [B, 1, T, T]

        Returns:
            Output [B, C, T]
        """
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, c, attn_mask)
        return self._forward_ttnn(x, c, attn_mask)

    def _forward_pytorch(self, x, c, attn_mask):
        # Project Q, K, V using 1x1 convolutions
        q = F.conv1d(x, _ensure_conv1d_weight(self.conv_q_weight), self.conv_q_bias)
        k = F.conv1d(c, _ensure_conv1d_weight(self.conv_k_weight), self.conv_k_bias)
        v = F.conv1d(c, _ensure_conv1d_weight(self.conv_v_weight), self.conv_v_bias)

        # Compute attention
        x, _ = self._attention(q, k, v, attn_mask)

        # Output projection
        x = F.conv1d(x, _ensure_conv1d_weight(self.conv_o_weight), self.conv_o_bias)
        return x

    def _forward_ttnn(self, x, c, attn_mask):
        # Project Q, K, V
        q = ttnn_conv1d(x, self.conv_q_weight, self.conv_q_bias, device=self.device)
        k = ttnn_conv1d(c, self.conv_k_weight, self.conv_k_bias, device=self.device)
        v = ttnn_conv1d(c, self.conv_v_weight, self.conv_v_bias, device=self.device)

        # Compute attention
        x, _ = self._attention_ttnn(q, k, v, attn_mask)

        # Output projection
        x = ttnn_conv1d(x, self.conv_o_weight, self.conv_o_bias, device=self.device)
        return x

    def _attention(self, query, key, value, mask=None):
        """Attention computation (PyTorch)."""
        b, d, t_s = key.size()
        t_t = query.size(2)

        # Reshape to [B, n_heads, T, k_channels]
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        # Add relative position bias if available
        if self.window_size is not None and self.emb_rel_k is not None:
            # Simplified: skip relative attention for now
            pass

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, value)

        # Reshape back to [B, C, T]
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)

        return output, attn

    def _attention_ttnn(self, query, key, value, mask=None):
        """Attention computation (TTNN) - uses fused attention when available."""
        b = query.shape[0]
        d = query.shape[1]
        t_t = query.shape[2]
        t_s = key.shape[2]

        # Reshape to [B, n_heads, T, k_channels]
        query = ttnn.reshape(query, (b, self.n_heads, self.k_channels, t_t))
        query = ttnn.permute(query, (0, 1, 3, 2))  # [B, n_heads, T, k_channels]

        key = ttnn.reshape(key, (b, self.n_heads, self.k_channels, t_s))
        key = ttnn.permute(key, (0, 1, 3, 2))

        value = ttnn.reshape(value, (b, self.n_heads, self.k_channels, t_s))
        value = ttnn.permute(value, (0, 1, 3, 2))

        # Try to use fused scaled_dot_product_attention for better performance
        use_fused = hasattr(ttnn, "transformer") and hasattr(ttnn.transformer, "scaled_dot_product_attention")

        if use_fused:
            try:
                # Use FlashAttention-2 via TTNN
                scale = 1.0 / math.sqrt(self.k_channels)
                output = ttnn.transformer.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=mask,
                    is_causal=False,
                    scale=scale,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                # Reshape back
                output = ttnn.permute(output, (0, 1, 3, 2))  # [B, n_heads, k_channels, T]
                output = ttnn.reshape(output, (b, d, t_t))
                return output, None
            except Exception:
                # Fall back to manual implementation if fused attention fails
                pass

        # Manual attention computation (fallback)
        # Use L1 memory config for efficient computation
        scale = 1.0 / math.sqrt(self.k_channels)
        key_t = ttnn.permute(key, (0, 1, 3, 2))  # Transpose last two dims
        scores = ttnn.matmul(query, key_t, memory_config=ttnn.L1_MEMORY_CONFIG)
        scores = ttnn.multiply(scores, scale)

        if mask is not None:
            # Apply mask
            scores = ttnn.where(mask == 0, -1e4, scores)

        attn = ttnn.softmax(scores, dim=-1)
        output = ttnn.matmul(attn, value, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Reshape back
        output = ttnn.permute(output, (0, 1, 3, 2))  # [B, n_heads, k_channels, T]
        output = ttnn.reshape(output, (b, d, t_t))

        return output, attn


class FFN:
    """
    Feed-forward network with Conv1d layers.

    Used in transformer blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        conv_1_weight: Any = None,
        conv_1_bias: Any = None,
        conv_2_weight: Any = None,
        conv_2_bias: Any = None,
        causal: bool = False,
        device: Optional[Any] = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.device = device

        self.conv_1_weight = conv_1_weight
        self.conv_1_bias = conv_1_bias
        self.conv_2_weight = conv_2_weight
        self.conv_2_bias = conv_2_bias

    def __call__(self, x: Any, x_mask: Any) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask)
        return self._forward_ttnn(x, x_mask)

    def _get_padding(self):
        if self.causal:
            return (self.kernel_size - 1, 0)
        else:
            pad_l = (self.kernel_size - 1) // 2
            pad_r = self.kernel_size // 2
            return (pad_l, pad_r)

    def _forward_pytorch(self, x, x_mask):
        pad_l, pad_r = self._get_padding()

        # First conv
        x_padded = F.pad(x * x_mask, (pad_l, pad_r))
        x = F.conv1d(x_padded, _ensure_conv1d_weight(self.conv_1_weight), self.conv_1_bias)
        x = F.relu(x)

        # Second conv
        x_padded = F.pad(x * x_mask, (pad_l, pad_r))
        x = F.conv1d(x_padded, _ensure_conv1d_weight(self.conv_2_weight), self.conv_2_bias)

        return x * x_mask

    def _forward_ttnn(self, x, x_mask):
        pad_l, pad_r = self._get_padding()

        # First conv with fused relu activation
        x = ttnn.multiply(x, x_mask)
        x = ttnn.pad(x, ((0, 0), (0, 0), (pad_l, pad_r)))
        x = ttnn_conv1d(x, self.conv_1_weight, self.conv_1_bias, device=self.device, activation="relu")

        # Second conv
        x = ttnn.multiply(x, x_mask)
        x = ttnn.pad(x, ((0, 0), (0, 0), (pad_l, pad_r)))
        x = ttnn_conv1d(x, self.conv_2_weight, self.conv_2_bias, device=self.device)

        return ttnn.multiply(x, x_mask)


class LayerNorm1d:
    """Layer normalization for 1D sequences (channel-first)."""

    def __init__(self, channels: int, weight: Any = None, bias: Any = None, eps: float = 1e-5):
        self.channels = channels
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x: Any) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            # Transpose, apply layer norm, transpose back
            x = x.transpose(1, -1)
            x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
            return x.transpose(1, -1)

        # TTNN
        x = ttnn.permute(x, (0, 2, 1))  # [B, C, T] -> [B, T, C]
        x = ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps)
        x = ttnn.permute(x, (0, 2, 1))  # [B, T, C] -> [B, C, T]
        return x


class TTNNTextEncoder:
    """
    Text Encoder for TTS.

    Architecture:
        1. Embedding lookup
        2. N transformer blocks (attention + FFN)
        3. Projection to mean/log-variance

    Used for encoding phoneme sequences in the TTS pipeline.
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        emb_weight: Any = None,
        attn_layers: Optional[List[MultiHeadAttention]] = None,
        norm_layers_1: Optional[List[LayerNorm1d]] = None,
        ffn_layers: Optional[List[FFN]] = None,
        norm_layers_2: Optional[List[LayerNorm1d]] = None,
        proj_weight: Any = None,
        proj_bias: Any = None,
        device: Optional[Any] = None,
    ):
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.device = device

        self.emb_weight = emb_weight
        self.attn_layers = attn_layers or []
        self.norm_layers_1 = norm_layers_1 or []
        self.ffn_layers = ffn_layers or []
        self.norm_layers_2 = norm_layers_2 or []
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias

    def __call__(
        self,
        x: Any,
        x_lengths: Any,
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Encode text sequence.

        Args:
            x: Token indices [B, T]
            x_lengths: Sequence lengths [B]

        Returns:
            Tuple of (hidden, mean, log_variance, mask)
        """
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_lengths)
        return self._forward_ttnn(x, x_lengths)

    def _forward_pytorch(self, x, x_lengths):
        # Embedding
        x = F.embedding(x, self.emb_weight)  # [B, T, hidden]
        x = x * math.sqrt(self.hidden_channels)
        x = x.transpose(1, 2)  # [B, hidden, T]

        # Create mask
        max_len = x.size(2)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < x_lengths.unsqueeze(1)
        x_mask = mask.unsqueeze(1).to(x.dtype)  # [B, 1, T]

        # Create attention mask [B, 1, T, T]
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        # Transformer blocks
        x = x * x_mask
        for i in range(self.n_layers):
            # Self-attention
            y = self.attn_layers[i](x, x, attn_mask)
            x = self.norm_layers_1[i](x + y)

            # FFN
            y = self.ffn_layers[i](x, x_mask)
            x = self.norm_layers_2[i](x + y)

        x = x * x_mask

        # Project to stats
        stats = F.conv1d(x, _ensure_conv1d_weight(self.proj_weight), self.proj_bias)
        stats = stats * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)

        return x, m, logs, x_mask

    def _forward_ttnn(self, x, x_lengths):
        # Embedding
        x = ttnn.embedding(x, self.emb_weight)  # [B, T, hidden]
        x = ttnn.multiply(x, math.sqrt(self.hidden_channels))
        x = ttnn.permute(x, (0, 2, 1))  # [B, hidden, T]

        # Create mask
        max_len = x.shape[2]
        mask = ttnn.arange(0, max_len)
        mask = ttnn.unsqueeze(mask, 0)
        x_lengths_expanded = ttnn.unsqueeze(x_lengths, 1)
        mask = ttnn.lt(mask, x_lengths_expanded)
        x_mask = ttnn.unsqueeze(mask, 1)  # [B, 1, T]
        x_mask = ttnn.to_dtype(x_mask, x.dtype)

        # Attention mask
        attn_mask = ttnn.multiply(ttnn.unsqueeze(x_mask, 2), ttnn.unsqueeze(x_mask, -1))

        # Transformer blocks
        x = ttnn.multiply(x, x_mask)
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            x = self.norm_layers_1[i](ttnn.add(x, y))

            y = self.ffn_layers[i](x, x_mask)
            x = self.norm_layers_2[i](ttnn.add(x, y))

        x = ttnn.multiply(x, x_mask)

        # Project
        stats = ttnn_conv1d(x, self.proj_weight, self.proj_bias, device=self.device)
        stats = ttnn.multiply(stats, x_mask)

        m = stats[:, : self.out_channels, :]
        logs = stats[:, self.out_channels :, :]

        return x, m, logs, x_mask

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        prefix: str,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        device: Optional[Any] = None,
    ) -> "TTNNTextEncoder":
        """Create TextEncoder from state dict."""

        emb_weight = state_dict.get(f"{prefix}.emb.weight")

        attn_layers = []
        norm_layers_1 = []
        ffn_layers = []
        norm_layers_2 = []

        for i in range(n_layers):
            # Attention
            attn = MultiHeadAttention(
                channels=hidden_channels,
                out_channels=hidden_channels,
                n_heads=n_heads,
                conv_q_weight=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_q.weight"),
                conv_q_bias=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_q.bias"),
                conv_k_weight=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_k.weight"),
                conv_k_bias=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_k.bias"),
                conv_v_weight=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_v.weight"),
                conv_v_bias=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_v.bias"),
                conv_o_weight=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_o.weight"),
                conv_o_bias=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_o.bias"),
                device=device,
            )
            attn_layers.append(attn)

            # Norm 1
            norm1 = LayerNorm1d(
                hidden_channels,
                weight=state_dict.get(f"{prefix}.encoder.norm_layers_1.{i}.gamma"),
                bias=state_dict.get(f"{prefix}.encoder.norm_layers_1.{i}.beta"),
            )
            norm_layers_1.append(norm1)

            # FFN
            ffn = FFN(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                filter_channels=filter_channels,
                kernel_size=kernel_size,
                conv_1_weight=state_dict.get(f"{prefix}.encoder.ffn_layers.{i}.conv_1.weight"),
                conv_1_bias=state_dict.get(f"{prefix}.encoder.ffn_layers.{i}.conv_1.bias"),
                conv_2_weight=state_dict.get(f"{prefix}.encoder.ffn_layers.{i}.conv_2.weight"),
                conv_2_bias=state_dict.get(f"{prefix}.encoder.ffn_layers.{i}.conv_2.bias"),
                device=device,
            )
            ffn_layers.append(ffn)

            # Norm 2
            norm2 = LayerNorm1d(
                hidden_channels,
                weight=state_dict.get(f"{prefix}.encoder.norm_layers_2.{i}.gamma"),
                bias=state_dict.get(f"{prefix}.encoder.norm_layers_2.{i}.beta"),
            )
            norm_layers_2.append(norm2)

        proj_weight = state_dict.get(f"{prefix}.proj.weight")
        proj_bias = state_dict.get(f"{prefix}.proj.bias")

        return cls(
            n_vocab=n_vocab,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            emb_weight=emb_weight,
            attn_layers=attn_layers,
            norm_layers_1=norm_layers_1,
            ffn_layers=ffn_layers,
            norm_layers_2=norm_layers_2,
            proj_weight=proj_weight,
            proj_bias=proj_bias,
            device=device,
        )
