# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Transformer-based Coupling Flow for MeloTTS.

Uses attention-based flow layers instead of WaveNet-based coupling.
"""

import math
from typing import Optional, Any, List

import torch
import torch.nn as nn
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


class LayerNorm1d:
    """Layer normalization for 1D sequences."""

    def __init__(self, channels: int, weight: Any = None, bias: Any = None, eps: float = 1e-5):
        self.channels = channels
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x: Any) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            x = x.transpose(1, -1)
            x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
            return x.transpose(1, -1)

        x = ttnn.permute(x, (0, 2, 1))
        x = ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps)
        x = ttnn.permute(x, (0, 2, 1))
        return x


class MultiHeadAttentionFlow:
    """Multi-head attention for flow layers."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        conv_q_weight: Any = None,
        conv_q_bias: Any = None,
        conv_k_weight: Any = None,
        conv_k_bias: Any = None,
        conv_v_weight: Any = None,
        conv_v_bias: Any = None,
        conv_o_weight: Any = None,
        conv_o_bias: Any = None,
        device: Optional[Any] = None,
    ):
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.device = device

        self.conv_q_weight = conv_q_weight
        self.conv_q_bias = conv_q_bias
        self.conv_k_weight = conv_k_weight
        self.conv_k_bias = conv_k_bias
        self.conv_v_weight = conv_v_weight
        self.conv_v_bias = conv_v_bias
        self.conv_o_weight = conv_o_weight
        self.conv_o_bias = conv_o_bias

    def __call__(self, x: Any, c: Any, attn_mask: Optional[Any] = None) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, c, attn_mask)
        return self._forward_ttnn(x, c, attn_mask)

    def _forward_pytorch(self, x, c, attn_mask):
        q = F.conv1d(x, _ensure_conv1d_weight(self.conv_q_weight), self.conv_q_bias)
        k = F.conv1d(c, _ensure_conv1d_weight(self.conv_k_weight), self.conv_k_bias)
        v = F.conv1d(c, _ensure_conv1d_weight(self.conv_v_weight), self.conv_v_bias)

        b, d, t_s = k.size()
        t_t = q.size(2)

        q = q.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        k = k.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        v = v.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        output = F.conv1d(output, _ensure_conv1d_weight(self.conv_o_weight), self.conv_o_bias)

        return output

    def _forward_ttnn(self, x, c, attn_mask):
        """Attention computation (TTNN) - uses fused attention when available."""
        q = ttnn_conv1d(x, self.conv_q_weight, self.conv_q_bias, device=self.device)
        k = ttnn_conv1d(c, self.conv_k_weight, self.conv_k_bias, device=self.device)
        v = ttnn_conv1d(c, self.conv_v_weight, self.conv_v_bias, device=self.device)

        b, d, t_s = k.shape[0], k.shape[1], k.shape[2]
        t_t = q.shape[2]

        q = ttnn.reshape(q, (b, self.n_heads, self.k_channels, t_t))
        q = ttnn.permute(q, (0, 1, 3, 2))  # [B, n_heads, T, k_channels]
        k = ttnn.reshape(k, (b, self.n_heads, self.k_channels, t_s))
        k = ttnn.permute(k, (0, 1, 3, 2))  # [B, n_heads, T, k_channels]
        v = ttnn.reshape(v, (b, self.n_heads, self.k_channels, t_s))
        v = ttnn.permute(v, (0, 1, 3, 2))  # [B, n_heads, T, k_channels]

        # Try to use fused scaled_dot_product_attention for better performance
        use_fused = hasattr(ttnn, 'transformer') and hasattr(ttnn.transformer, 'scaled_dot_product_attention')

        if use_fused:
            try:
                # Use FlashAttention-2 via TTNN fused SDPA
                scale = 1.0 / math.sqrt(self.k_channels)
                output = ttnn.transformer.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    is_causal=False,
                    scale=scale,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                # Reshape back
                output = ttnn.permute(output, (0, 1, 3, 2))  # [B, n_heads, k_channels, T]
                output = ttnn.reshape(output, (b, d, t_t))
                output = ttnn_conv1d(output, self.conv_o_weight, self.conv_o_bias, device=self.device)
                return output
            except Exception:
                # Fall back to manual implementation if fused attention fails
                pass

        # Manual attention computation (fallback)
        scale = 1.0 / math.sqrt(self.k_channels)
        k_t = ttnn.permute(k, (0, 1, 3, 2))

        # Attention with L1 memory config for efficient computation
        scores = ttnn.matmul(q, k_t, memory_config=ttnn.L1_MEMORY_CONFIG)
        scores = ttnn.multiply(scores, scale)

        if attn_mask is not None:
            scores = ttnn.where(attn_mask == 0, -1e4, scores)

        attn = ttnn.softmax(scores, dim=-1)
        output = ttnn.matmul(attn, v, memory_config=ttnn.L1_MEMORY_CONFIG)

        output = ttnn.permute(output, (0, 1, 3, 2))
        output = ttnn.reshape(output, (b, d, t_t))
        output = ttnn_conv1d(output, self.conv_o_weight, self.conv_o_bias, device=self.device)

        return output


class FFTBlock:
    """
    Feed-Forward Transformer block for flow.

    Architecture: Self-Attention -> FFN with residual connections.
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        attn_layers: Optional[List] = None,
        norm1_layers: Optional[List] = None,
        ffn_conv1_weights: Optional[List] = None,
        ffn_conv1_biases: Optional[List] = None,
        ffn_conv2_weights: Optional[List] = None,
        ffn_conv2_biases: Optional[List] = None,
        norm2_layers: Optional[List] = None,
        cond_weight: Any = None,
        cond_bias: Any = None,
        device: Optional[Any] = None,
    ):
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.gin_channels = gin_channels
        self.device = device

        self.attn_layers = attn_layers or []
        self.norm1_layers = norm1_layers or []
        self.ffn_conv1_weights = ffn_conv1_weights or []
        self.ffn_conv1_biases = ffn_conv1_biases or []
        self.ffn_conv2_weights = ffn_conv2_weights or []
        self.ffn_conv2_biases = ffn_conv2_biases or []
        self.norm2_layers = norm2_layers or []
        self.cond_weight = cond_weight
        self.cond_bias = cond_bias

    def __call__(self, x: Any, x_mask: Any, g: Optional[Any] = None) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask, g)
        return self._forward_ttnn(x, x_mask, g)

    def _forward_pytorch(self, x, x_mask, g):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        if g is not None and self.cond_weight is not None:
            g_cond = F.conv1d(g, _ensure_conv1d_weight(self.cond_weight), self.cond_bias)
            x = x + g_cond

        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            x = self.norm1_layers[i](x + y)

            # FFN with padding to preserve sequence length
            w1 = _ensure_conv1d_weight(self.ffn_conv1_weights[i])
            k1 = w1.shape[2] if w1.dim() == 3 else 1
            y = F.conv1d(x * x_mask, w1, self.ffn_conv1_biases[i], padding=k1 // 2)
            y = F.gelu(y)
            w2 = _ensure_conv1d_weight(self.ffn_conv2_weights[i])
            k2 = w2.shape[2] if w2.dim() == 3 else 1
            y = F.conv1d(y * x_mask, w2, self.ffn_conv2_biases[i], padding=k2 // 2)
            x = self.norm2_layers[i](x + y)

        return x * x_mask

    def _forward_ttnn(self, x, x_mask, g):
        attn_mask = ttnn.multiply(
            ttnn.unsqueeze(x_mask, 2),
            ttnn.unsqueeze(x_mask, -1)
        )

        if g is not None and self.cond_weight is not None:
            g_cond = ttnn_conv1d(g, self.cond_weight, self.cond_bias, device=self.device)
            x = ttnn.add(x, g_cond)

        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            x = self.norm1_layers[i](ttnn.add(x, y))

            y = ttnn.multiply(x, x_mask)
            y = ttnn_conv1d(y, self.ffn_conv1_weights[i], self.ffn_conv1_biases[i], device=self.device)
            y = ttnn.gelu(y)
            y = ttnn.multiply(y, x_mask)
            y = ttnn_conv1d(y, self.ffn_conv2_weights[i], self.ffn_conv2_biases[i], device=self.device)
            x = self.norm2_layers[i](ttnn.add(x, y))

        return ttnn.multiply(x, x_mask)


class TransformerCouplingLayer:
    """
    Single transformer coupling layer for normalizing flow.

    Splits input, applies transformer to one half to predict mean shift.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        n_layers: int,
        n_heads: int,
        p_dropout: float = 0.0,
        filter_channels: int = 0,
        mean_only: bool = True,
        gin_channels: int = 0,
        pre_weight: Any = None,
        pre_bias: Any = None,
        post_weight: Any = None,
        post_bias: Any = None,
        fft_block: Optional[FFTBlock] = None,
        device: Optional[Any] = None,
    ):
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.device = device

        self.pre_weight = pre_weight
        self.pre_bias = pre_bias
        self.post_weight = post_weight
        self.post_bias = post_bias
        self.fft = fft_block

    def __call__(self, x: Any, x_mask: Any, g: Optional[Any] = None, reverse: bool = False):
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask, g, reverse)
        return self._forward_ttnn(x, x_mask, g, reverse)

    def _forward_pytorch(self, x, x_mask, g, reverse):
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], 1)

        h = F.conv1d(x0, _ensure_conv1d_weight(self.pre_weight), self.pre_bias)
        h = self.fft(h, x_mask, g=g)
        stats = F.conv1d(h, _ensure_conv1d_weight(self.post_weight), self.post_bias) * x_mask

        if self.mean_only:
            m = stats
            logs = torch.zeros_like(m)
        else:
            m, logs = torch.split(stats, [self.half_channels, self.half_channels], 1)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs * x_mask, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

    def _forward_ttnn(self, x, x_mask, g, reverse):
        x0 = x[:, :self.half_channels, :]
        x1 = x[:, self.half_channels:, :]

        h = ttnn_conv1d(x0, self.pre_weight, self.pre_bias, device=self.device)
        h = self.fft(h, x_mask, g=g)
        stats = ttnn_conv1d(h, self.post_weight, self.post_bias, device=self.device)
        stats = ttnn.multiply(stats, x_mask)

        if self.mean_only:
            m = stats
            logs = ttnn.zeros_like(m)
        else:
            m = stats[:, :self.half_channels, :]
            logs = stats[:, self.half_channels:, :]

        if not reverse:
            x1 = ttnn.add(m, ttnn.multiply(ttnn.multiply(x1, ttnn.exp(logs)), x_mask))
            x = ttnn.concat([x0, x1], dim=1)
            logdet = ttnn.sum(ttnn.multiply(logs, x_mask), dim=[1, 2])
            return x, logdet
        else:
            x1 = ttnn.multiply(ttnn.multiply(ttnn.subtract(x1, m), ttnn.exp(ttnn.neg(logs))), x_mask)
            x = ttnn.concat([x0, x1], dim=1)
            return x


class Flip:
    """
    Flip operation for normalizing flows.

    Note: Uses CPU roundtrip - TTNN lacks native flip operation.
    Impact is minimal (~0.01ms per flip).
    """

    def __call__(self, x: Any, *args, reverse: bool = False, **kwargs):
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            x = torch.flip(x, [1])
            if not reverse:
                return x, torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x

        # CPU roundtrip required - TTNN has no native flip operation
        was_on_device = ttnn.is_tensor_storage_on_device(x)
        device = x.device() if was_on_device else None
        orig_layout = x.get_layout()

        x_torch = ttnn.to_torch(x)
        x_flipped = torch.flip(x_torch, [1])
        x = ttnn.from_torch(x_flipped, dtype=ttnn.bfloat16, layout=orig_layout)

        if was_on_device and device is not None:
            x = ttnn.to_device(x, device)
        if not reverse:
            return x, ttnn.zeros((x.shape[0],), dtype=x.dtype)
        return x


class TTNNTransformerCouplingBlock:
    """
    Transformer-based Coupling Block for normalizing flow.

    Uses attention instead of WaveNet for the coupling layers.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        n_flows: int = 4,
        gin_channels: int = 0,
        flows: Optional[List] = None,
        device: Optional[Any] = None,
    ):
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.device = device

        self.flows = flows or []

    def __call__(self, x: Any, x_mask: Any, g: Optional[Any] = None, reverse: bool = False) -> Any:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        prefix: str,
        channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        n_flows: int = 4,
        gin_channels: int = 0,
        device: Optional[Any] = None,
    ) -> "TTNNTransformerCouplingBlock":
        """Create TransformerCouplingBlock from state dict."""

        flows = []
        for i in range(n_flows):
            flow_prefix = f"{prefix}.flows.{i * 2}"

            # Build FFT block for this flow
            attn_layers = []
            norm1_layers = []
            ffn_conv1_weights = []
            ffn_conv1_biases = []
            ffn_conv2_weights = []
            ffn_conv2_biases = []
            norm2_layers = []

            for j in range(n_layers):
                attn = MultiHeadAttentionFlow(
                    channels=hidden_channels,
                    out_channels=hidden_channels,
                    n_heads=n_heads,
                    conv_q_weight=state_dict.get(f"{flow_prefix}.enc.attn_layers.{j}.conv_q.weight"),
                    conv_q_bias=state_dict.get(f"{flow_prefix}.enc.attn_layers.{j}.conv_q.bias"),
                    conv_k_weight=state_dict.get(f"{flow_prefix}.enc.attn_layers.{j}.conv_k.weight"),
                    conv_k_bias=state_dict.get(f"{flow_prefix}.enc.attn_layers.{j}.conv_k.bias"),
                    conv_v_weight=state_dict.get(f"{flow_prefix}.enc.attn_layers.{j}.conv_v.weight"),
                    conv_v_bias=state_dict.get(f"{flow_prefix}.enc.attn_layers.{j}.conv_v.bias"),
                    conv_o_weight=state_dict.get(f"{flow_prefix}.enc.attn_layers.{j}.conv_o.weight"),
                    conv_o_bias=state_dict.get(f"{flow_prefix}.enc.attn_layers.{j}.conv_o.bias"),
                    device=device,
                )
                attn_layers.append(attn)

                norm1 = LayerNorm1d(
                    hidden_channels,
                    weight=state_dict.get(f"{flow_prefix}.enc.norm_layers_1.{j}.gamma"),
                    bias=state_dict.get(f"{flow_prefix}.enc.norm_layers_1.{j}.beta"),
                )
                norm1_layers.append(norm1)

                ffn_conv1_weights.append(state_dict.get(f"{flow_prefix}.enc.ffn_layers.{j}.conv_1.weight"))
                ffn_conv1_biases.append(state_dict.get(f"{flow_prefix}.enc.ffn_layers.{j}.conv_1.bias"))
                ffn_conv2_weights.append(state_dict.get(f"{flow_prefix}.enc.ffn_layers.{j}.conv_2.weight"))
                ffn_conv2_biases.append(state_dict.get(f"{flow_prefix}.enc.ffn_layers.{j}.conv_2.bias"))

                norm2 = LayerNorm1d(
                    hidden_channels,
                    weight=state_dict.get(f"{flow_prefix}.enc.norm_layers_2.{j}.gamma"),
                    bias=state_dict.get(f"{flow_prefix}.enc.norm_layers_2.{j}.beta"),
                )
                norm2_layers.append(norm2)

            fft = FFTBlock(
                hidden_channels=hidden_channels,
                filter_channels=filter_channels,
                n_heads=n_heads,
                n_layers=n_layers,
                gin_channels=gin_channels,
                attn_layers=attn_layers,
                norm1_layers=norm1_layers,
                ffn_conv1_weights=ffn_conv1_weights,
                ffn_conv1_biases=ffn_conv1_biases,
                ffn_conv2_weights=ffn_conv2_weights,
                ffn_conv2_biases=ffn_conv2_biases,
                norm2_layers=norm2_layers,
                cond_weight=state_dict.get(f"{flow_prefix}.enc.cond_layer.weight") if gin_channels > 0 else None,
                cond_bias=state_dict.get(f"{flow_prefix}.enc.cond_layer.bias") if gin_channels > 0 else None,
                device=device,
            )

            layer = TransformerCouplingLayer(
                channels=channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                n_layers=n_layers,
                n_heads=n_heads,
                filter_channels=filter_channels,
                gin_channels=gin_channels,
                pre_weight=state_dict.get(f"{flow_prefix}.pre.weight"),
                pre_bias=state_dict.get(f"{flow_prefix}.pre.bias"),
                post_weight=state_dict.get(f"{flow_prefix}.post.weight"),
                post_bias=state_dict.get(f"{flow_prefix}.post.bias"),
                fft_block=fft,
                device=device,
            )
            flows.append(layer)
            flows.append(Flip())

        return cls(
            channels=channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            n_flows=n_flows,
            gin_channels=gin_channels,
            flows=flows,
            device=device,
        )
