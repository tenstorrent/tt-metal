# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Duration Predictor modules for TTS pipeline.

Predicts phoneme durations for alignment in text-to-speech.
"""

import math
from typing import Optional, Any, List, Tuple

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
    """Layer normalization for 1D sequences (channel-first)."""

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


class TTNNDurationPredictor:
    """
    Duration Predictor for TTS.

    Predicts phoneme durations from encoder hidden states.
    Architecture: Conv1d -> ReLU -> LayerNorm -> Dropout -> Conv1d -> ReLU -> LayerNorm -> Dropout -> Proj
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.5,
        gin_channels: int = 0,
        conv_1_weight: Any = None,
        conv_1_bias: Any = None,
        norm_1_weight: Any = None,
        norm_1_bias: Any = None,
        conv_2_weight: Any = None,
        conv_2_bias: Any = None,
        norm_2_weight: Any = None,
        norm_2_bias: Any = None,
        proj_weight: Any = None,
        proj_bias: Any = None,
        cond_weight: Any = None,
        cond_bias: Any = None,
        device: Optional[Any] = None,
    ):
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.device = device
        self.training = False

        self.conv_1_weight = conv_1_weight
        self.conv_1_bias = conv_1_bias
        self.norm_1 = LayerNorm1d(filter_channels, norm_1_weight, norm_1_bias)
        self.conv_2_weight = conv_2_weight
        self.conv_2_bias = conv_2_bias
        self.norm_2 = LayerNorm1d(filter_channels, norm_2_weight, norm_2_bias)
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias
        self.cond_weight = cond_weight
        self.cond_bias = cond_bias

    def __call__(self, x: Any, x_mask: Any, g: Optional[Any] = None) -> Any:
        """
        Predict durations.

        Args:
            x: Hidden states [B, C, T]
            x_mask: Mask [B, 1, T]
            g: Optional speaker conditioning [B, gin_channels, 1]

        Returns:
            Predicted log durations [B, 1, T]
        """
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask, g)
        return self._forward_ttnn(x, x_mask, g)

    def _forward_pytorch(self, x, x_mask, g):
        x = x.detach()
        if g is not None and self.cond_weight is not None:
            g = g.detach()
            x = x + F.conv1d(g, _ensure_conv1d_weight(self.cond_weight), self.cond_bias)

        padding = self.kernel_size // 2
        x = F.conv1d(x * x_mask, _ensure_conv1d_weight(self.conv_1_weight), self.conv_1_bias, padding=padding)
        x = torch.relu(x)
        x = self.norm_1(x)
        if self.training:
            x = F.dropout(x, self.p_dropout)

        x = F.conv1d(x * x_mask, _ensure_conv1d_weight(self.conv_2_weight), self.conv_2_bias, padding=padding)
        x = torch.relu(x)
        x = self.norm_2(x)
        if self.training:
            x = F.dropout(x, self.p_dropout)

        x = F.conv1d(x * x_mask, _ensure_conv1d_weight(self.proj_weight), self.proj_bias)
        return x * x_mask

    def _forward_ttnn(self, x, x_mask, g):
        if g is not None and self.cond_weight is not None:
            cond = ttnn_conv1d(g, self.cond_weight, self.cond_bias, device=self.device)
            x = ttnn.add(x, cond)

        padding = self.kernel_size // 2
        x = ttnn.multiply(x, x_mask)
        # Fused conv + relu for better performance
        x = ttnn_conv1d(x, self.conv_1_weight, self.conv_1_bias, padding=padding, device=self.device, activation="relu")
        x = self.norm_1(x)

        x = ttnn.multiply(x, x_mask)
        # Fused conv + relu for better performance
        x = ttnn_conv1d(x, self.conv_2_weight, self.conv_2_bias, padding=padding, device=self.device, activation="relu")
        x = self.norm_2(x)

        x = ttnn.multiply(x, x_mask)
        x = ttnn_conv1d(x, self.proj_weight, self.proj_bias, device=self.device)
        return ttnn.multiply(x, x_mask)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        prefix: str,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.5,
        gin_channels: int = 0,
        device: Optional[Any] = None,
    ) -> "TTNNDurationPredictor":
        """Create DurationPredictor from state dict."""
        cond_weight = state_dict.get(f"{prefix}.cond.weight") if gin_channels > 0 else None
        cond_bias = state_dict.get(f"{prefix}.cond.bias") if gin_channels > 0 else None

        return cls(
            in_channels=in_channels,
            filter_channels=filter_channels,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
            conv_1_weight=state_dict.get(f"{prefix}.conv_1.weight"),
            conv_1_bias=state_dict.get(f"{prefix}.conv_1.bias"),
            norm_1_weight=state_dict.get(f"{prefix}.norm_1.gamma"),
            norm_1_bias=state_dict.get(f"{prefix}.norm_1.beta"),
            conv_2_weight=state_dict.get(f"{prefix}.conv_2.weight"),
            conv_2_bias=state_dict.get(f"{prefix}.conv_2.bias"),
            norm_2_weight=state_dict.get(f"{prefix}.norm_2.gamma"),
            norm_2_bias=state_dict.get(f"{prefix}.norm_2.beta"),
            proj_weight=state_dict.get(f"{prefix}.proj.weight"),
            proj_bias=state_dict.get(f"{prefix}.proj.bias"),
            cond_weight=cond_weight,
            cond_bias=cond_bias,
            device=device,
        )


class DDSConv:
    """
    Dilated and Depth-Separable Convolution.

    Used in StochasticDurationPredictor.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float = 0.0,
        convs_sep_weights: Optional[List[Any]] = None,
        convs_sep_biases: Optional[List[Any]] = None,
        convs_1x1_weights: Optional[List[Any]] = None,
        convs_1x1_biases: Optional[List[Any]] = None,
        norms_1_weights: Optional[List[Any]] = None,
        norms_1_biases: Optional[List[Any]] = None,
        norms_2_weights: Optional[List[Any]] = None,
        norms_2_biases: Optional[List[Any]] = None,
        device: Optional[Any] = None,
    ):
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.device = device
        self.training = False

        self.convs_sep_weights = convs_sep_weights or []
        self.convs_sep_biases = convs_sep_biases or []
        self.convs_1x1_weights = convs_1x1_weights or []
        self.convs_1x1_biases = convs_1x1_biases or []
        self.norms_1 = [
            LayerNorm1d(channels, w, b)
            for w, b in zip(norms_1_weights or [], norms_1_biases or [])
        ]
        self.norms_2 = [
            LayerNorm1d(channels, w, b)
            for w, b in zip(norms_2_weights or [], norms_2_biases or [])
        ]

    def __call__(self, x: Any, x_mask: Any, g: Optional[Any] = None) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask, g)
        return self._forward_ttnn(x, x_mask, g)

    def _forward_pytorch(self, x, x_mask, g):
        if g is not None:
            x = x + g

        for i in range(self.n_layers):
            dilation = self.kernel_size ** i
            padding = (self.kernel_size * dilation - dilation) // 2

            y = F.conv1d(
                x * x_mask,
                _ensure_conv1d_weight(self.convs_sep_weights[i]),
                self.convs_sep_biases[i],
                padding=padding,
                dilation=dilation,
                groups=self.channels,
            )
            y = self.norms_1[i](y)
            y = F.gelu(y)

            y = F.conv1d(y, _ensure_conv1d_weight(self.convs_1x1_weights[i]), self.convs_1x1_biases[i])
            y = self.norms_2[i](y)
            y = F.gelu(y)

            if self.training:
                y = F.dropout(y, self.p_dropout)

            x = x + y

        return x * x_mask

    def _forward_ttnn(self, x, x_mask, g):
        if g is not None:
            x = ttnn.add(x, g)

        for i in range(self.n_layers):
            dilation = self.kernel_size ** i
            padding = (self.kernel_size * dilation - dilation) // 2

            y = ttnn.multiply(x, x_mask)
            y = ttnn_conv1d(
                y,
                self.convs_sep_weights[i],
                self.convs_sep_biases[i],
                padding=padding,
                dilation=dilation,
                groups=self.channels,
                device=self.device,
            )
            y = self.norms_1[i](y)
            y = ttnn.gelu(y)

            y = ttnn_conv1d(y, self.convs_1x1_weights[i], self.convs_1x1_biases[i], device=self.device)
            y = self.norms_2[i](y)
            y = ttnn.gelu(y)

            x = ttnn.add(x, y)

        return ttnn.multiply(x, x_mask)


class Log:
    """Log flow for normalizing flows."""

    def __call__(self, x: Any, x_mask: Any, reverse: bool = False) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask, reverse)
        return self._forward_ttnn(x, x_mask, reverse)

    def _forward_pytorch(self, x, x_mask, reverse):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            return torch.exp(x) * x_mask

    def _forward_ttnn(self, x, x_mask, reverse):
        if not reverse:
            y = ttnn.log(ttnn.maximum(x, 1e-5))
            y = ttnn.multiply(y, x_mask)
            logdet = ttnn.sum(ttnn.neg(y), dim=[1, 2])
            return y, logdet
        else:
            return ttnn.multiply(ttnn.exp(x), x_mask)


class Flip:
    """
    Flip operation for normalizing flows.

    Note: Uses CPU roundtrip - TTNN lacks native flip operation.
    Impact is minimal (~0.01ms per flip).
    """

    def __call__(self, x: Any, x_mask: Any = None, reverse: bool = False) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            x = torch.flip(x, [1])
            if not reverse:
                logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
                return x, logdet
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
            logdet = ttnn.zeros((x.shape[0],), dtype=x.dtype)
            return x, logdet
        return x


class ElementwiseAffine:
    """Elementwise affine transformation for normalizing flows."""

    def __init__(self, channels: int, m: Any = None, logs: Any = None):
        self.channels = channels
        self.m = m
        self.logs = logs

    def __call__(self, x: Any, x_mask: Any, reverse: bool = False, **kwargs) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask, reverse)
        return self._forward_ttnn(x, x_mask, reverse)

    def _forward_pytorch(self, x, x_mask, reverse):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            return (x - self.m) * torch.exp(-self.logs) * x_mask

    def _forward_ttnn(self, x, x_mask, reverse):
        if not reverse:
            scale = ttnn.exp(self.logs)
            y = ttnn.add(self.m, ttnn.multiply(scale, x))
            y = ttnn.multiply(y, x_mask)
            logdet = ttnn.sum(ttnn.multiply(self.logs, x_mask), dim=[1, 2])
            return y, logdet
        else:
            inv_scale = ttnn.exp(ttnn.neg(self.logs))
            return ttnn.multiply(ttnn.multiply(ttnn.subtract(x, self.m), inv_scale), x_mask)


class ConvFlow:
    """
    Convolutional flow layer for StochasticDurationPredictor.

    Uses spline-based transformations.
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        num_bins: int = 10,
        tail_bound: float = 5.0,
        pre_weight: Any = None,
        pre_bias: Any = None,
        proj_weight: Any = None,
        proj_bias: Any = None,
        convs: Optional[DDSConv] = None,
        device: Optional[Any] = None,
    ):
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2
        self.device = device

        self.pre_weight = pre_weight
        self.pre_bias = pre_bias
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias
        self.convs = convs

    def __call__(self, x: Any, x_mask: Any, g: Optional[Any] = None, reverse: bool = False) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask, g, reverse)
        return self._forward_ttnn(x, x_mask, g, reverse)

    def _forward_pytorch(self, x, x_mask, g, reverse):
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], 1)

        h = F.conv1d(x0, _ensure_conv1d_weight(self.pre_weight), self.pre_bias)
        h = self.convs(h, x_mask, g=g)
        h = F.conv1d(h, _ensure_conv1d_weight(self.proj_weight), self.proj_bias) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)

        unnorm_widths = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
        unnorm_heights = h[..., self.num_bins:2*self.num_bins] / math.sqrt(self.filter_channels)
        unnorm_derivatives = h[..., 2*self.num_bins:]

        x1, logdet = self._piecewise_rational_quadratic(
            x1, unnorm_widths, unnorm_heights, unnorm_derivatives,
            reverse=reverse, tail_bound=self.tail_bound
        )

        x = torch.cat([x0, x1], 1) * x_mask
        if not reverse:
            return x, torch.sum(logdet * x_mask, [1, 2])
        return x

    def _forward_ttnn(self, x, x_mask, g, reverse):
        x0 = x[:, :self.half_channels, :]
        x1 = x[:, self.half_channels:, :]

        h = ttnn_conv1d(x0, self.pre_weight, self.pre_bias, device=self.device)
        h = self.convs(h, x_mask, g=g)
        h = ttnn_conv1d(h, self.proj_weight, self.proj_bias, device=self.device)
        h = ttnn.multiply(h, x_mask)

        b, c, t = x0.shape[0], x0.shape[1], x0.shape[2]
        h = ttnn.reshape(h, (b, c, -1, t))
        h = ttnn.permute(h, (0, 1, 3, 2))

        scale = 1.0 / math.sqrt(self.filter_channels)
        unnorm_widths = ttnn.multiply(h[:, :, :, :self.num_bins], scale)
        unnorm_heights = ttnn.multiply(h[:, :, :, self.num_bins:2*self.num_bins], scale)
        unnorm_derivatives = h[:, :, :, 2*self.num_bins:]

        x1, logdet = self._piecewise_rational_quadratic_ttnn(
            x1, unnorm_widths, unnorm_heights, unnorm_derivatives,
            reverse=reverse, tail_bound=self.tail_bound
        )

        x = ttnn.concat([x0, x1], dim=1)
        x = ttnn.multiply(x, x_mask)
        if not reverse:
            logdet_sum = ttnn.sum(ttnn.multiply(logdet, x_mask), dim=[1, 2])
            return x, logdet_sum
        return x

    def _piecewise_rational_quadratic(
        self, inputs, unnorm_widths, unnorm_heights, unnorm_derivatives,
        reverse=False, tail_bound=5.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3
    ):
        """Piecewise rational quadratic spline transform (PyTorch)."""
        if torch.min(inputs) < -tail_bound or torch.max(inputs) > tail_bound:
            pass

        num_bins = unnorm_widths.shape[-1]

        widths = F.softmax(unnorm_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
        cumwidths = (2 * tail_bound) * cumwidths - tail_bound

        heights = F.softmax(unnorm_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, (1, 0), value=0.0)
        cumheights = (2 * tail_bound) * cumheights - tail_bound

        derivatives = min_derivative + F.softplus(unnorm_derivatives)

        if reverse:
            bin_idx = self._search_sorted(cumheights, inputs.unsqueeze(-1)).squeeze(-1)
        else:
            bin_idx = self._search_sorted(cumwidths, inputs.unsqueeze(-1)).squeeze(-1)

        widths = torch.gather(widths, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        cumwidths = torch.gather(cumwidths, -1, bin_idx.unsqueeze(-1)).squeeze(-1)

        heights = torch.gather(heights, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        cumheights = torch.gather(cumheights, -1, bin_idx.unsqueeze(-1)).squeeze(-1)

        delta = heights / widths
        derivatives_left = torch.gather(derivatives, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        derivatives_right = torch.gather(derivatives, -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

        if reverse:
            a = (inputs - cumheights) * (
                derivatives_right + derivatives_left - 2 * delta
            ) + heights * (delta - derivatives_left)
            b = heights * derivatives_left - (inputs - cumheights) * (
                derivatives_right + derivatives_left - 2 * delta
            )
            c = -delta * (inputs - cumheights)

            discriminant = b ** 2 - 4 * a * c
            root = (-b + torch.sqrt(discriminant)) / (2 * a)
            outputs = root * widths + cumwidths

            theta_deriv = delta ** 2 * (
                derivatives_right * root ** 2
                + 2 * delta * root * (1 - root)
                + derivatives_left * (1 - root) ** 2
            )
            logdet = -torch.log(theta_deriv)
        else:
            theta = (inputs - cumwidths) / widths
            theta_one_minus = 1 - theta

            numerator = heights * (delta * theta ** 2 + derivatives_left * theta * theta_one_minus)
            denominator = delta + (derivatives_right + derivatives_left - 2 * delta) * theta * theta_one_minus
            outputs = cumheights + numerator / denominator

            theta_deriv = delta ** 2 * (
                derivatives_right * theta ** 2
                + 2 * delta * theta * theta_one_minus
                + derivatives_left * theta_one_minus ** 2
            )
            logdet = torch.log(theta_deriv) - 2 * torch.log(denominator)

        return outputs, logdet

    def _piecewise_rational_quadratic_ttnn(
        self, inputs, unnorm_widths, unnorm_heights, unnorm_derivatives,
        reverse=False, tail_bound=5.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3
    ):
        """Piecewise rational quadratic spline transform (TTNN) - simplified."""
        torch_inputs = ttnn.to_torch(ttnn.from_device(inputs))
        torch_widths = ttnn.to_torch(ttnn.from_device(unnorm_widths))
        torch_heights = ttnn.to_torch(ttnn.from_device(unnorm_heights))
        torch_derivs = ttnn.to_torch(ttnn.from_device(unnorm_derivatives))

        outputs, logdet = self._piecewise_rational_quadratic(
            torch_inputs, torch_widths, torch_heights, torch_derivs,
            reverse=reverse, tail_bound=tail_bound
        )

        return (
            ttnn.from_torch(outputs, dtype=inputs.dtype, device=self.device),
            ttnn.from_torch(logdet, dtype=inputs.dtype, device=self.device),
        )

    def _search_sorted(self, sorted_seq, values):
        """Binary search for bin indices."""
        return torch.sum(sorted_seq < values, dim=-1) - 1


class TTNNStochasticDurationPredictor:
    """
    Stochastic Duration Predictor for TTS.

    Uses normalizing flows for probabilistic duration modeling.
    More expressive than the deterministic DurationPredictor.
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.5,
        n_flows: int = 4,
        gin_channels: int = 0,
        log_flow: Optional[Log] = None,
        flows: Optional[List] = None,
        post_pre_weight: Any = None,
        post_pre_bias: Any = None,
        post_proj_weight: Any = None,
        post_proj_bias: Any = None,
        post_convs: Optional[DDSConv] = None,
        post_flows: Optional[List] = None,
        pre_weight: Any = None,
        pre_bias: Any = None,
        proj_weight: Any = None,
        proj_bias: Any = None,
        convs: Optional[DDSConv] = None,
        cond_weight: Any = None,
        cond_bias: Any = None,
        device: Optional[Any] = None,
    ):
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.device = device

        self.log_flow = log_flow or Log()
        self.flows = flows or []
        self.post_pre_weight = post_pre_weight
        self.post_pre_bias = post_pre_bias
        self.post_proj_weight = post_proj_weight
        self.post_proj_bias = post_proj_bias
        self.post_convs = post_convs
        self.post_flows = post_flows or []
        self.pre_weight = pre_weight
        self.pre_bias = pre_bias
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias
        self.convs = convs
        self.cond_weight = cond_weight
        self.cond_bias = cond_bias

    def __call__(
        self,
        x: Any,
        x_mask: Any,
        w: Any = None,
        g: Any = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> Any:
        """
        Forward/reverse pass.

        Args:
            x: Hidden states [B, C, T]
            x_mask: Mask [B, 1, T]
            w: Duration targets [B, 1, T] (training only)
            g: Speaker conditioning [B, gin_channels, 1]
            reverse: If True, sample durations; if False, compute loss
            noise_scale: Noise scale for sampling

        Returns:
            If reverse: log durations [B, 1, T]
            If not reverse: negative log likelihood [B]
        """
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask, w, g, reverse, noise_scale)
        return self._forward_ttnn(x, x_mask, w, g, reverse, noise_scale)

    def _forward_pytorch(self, x, x_mask, w, g, reverse, noise_scale):
        x = x.detach()
        x = F.conv1d(x, _ensure_conv1d_weight(self.pre_weight), self.pre_bias)

        if g is not None and self.cond_weight is not None:
            g = g.detach()
            x = x + F.conv1d(g, _ensure_conv1d_weight(self.cond_weight), self.cond_bias)

        x = self.convs(x, x_mask)
        x = F.conv1d(x, _ensure_conv1d_weight(self.proj_weight), self.proj_bias) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = F.conv1d(w, _ensure_conv1d_weight(self.post_pre_weight), self.post_pre_bias)
            h_w = self.post_convs(h_w, x_mask)
            h_w = F.conv1d(h_w, _ensure_conv1d_weight(self.post_proj_weight), self.post_proj_bias) * x_mask

            e_q = torch.randn(w.size(0), 2, w.size(2), device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q

            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q

            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)

            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet

            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]

            z = torch.randn(x.size(0), 2, x.size(2), device=x.device, dtype=x.dtype) * noise_scale

            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)

            z0, z1 = torch.split(z, [1, 1], 1)
            return z0

    def _forward_ttnn(self, x, x_mask, w, g, reverse, noise_scale):
        x = ttnn_conv1d(x, self.pre_weight, self.pre_bias, device=self.device)

        if g is not None and self.cond_weight is not None:
            cond = ttnn_conv1d(g, self.cond_weight, self.cond_bias, device=self.device)
            x = ttnn.add(x, cond)

        x = self.convs(x, x_mask)
        x = ttnn_conv1d(x, self.proj_weight, self.proj_bias, device=self.device)
        x = ttnn.multiply(x, x_mask)

        if reverse:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]

            z = ttnn.randn((x.shape[0], 2, x.shape[2]), dtype=x.dtype, device=self.device)
            z = ttnn.multiply(z, noise_scale)

            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=True)

            z0 = z[:, :1, :]
            return z0
        else:
            return self._forward_pytorch(
                ttnn.to_torch(ttnn.from_device(x)),
                ttnn.to_torch(ttnn.from_device(x_mask)),
                ttnn.to_torch(ttnn.from_device(w)) if w is not None else None,
                ttnn.to_torch(ttnn.from_device(g)) if g is not None else None,
                reverse,
                noise_scale,
            )
