# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
HiFi-GAN Generator (Vocoder) for TTNN.

Converts latent representations to raw audio waveforms using
transposed convolutions and residual blocks.
"""

from typing import Optional, Any, List, Tuple

import torch
import torch.nn.functional as F

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

from models.demos.openvoice.tt.modules.conv1d import ttnn_conv1d, ttnn_conv_transpose1d

# Leaky ReLU slope used in HiFi-GAN
LRELU_SLOPE = 0.1


def get_weight_from_state_dict(state_dict: dict, prefix_key: str):
    """
    Get weight from state dict, handling weight normalization.

    Weight normalization stores weight_g and weight_v instead of weight.
    The actual weight is: w = g * (v / ||v||)

    Args:
        state_dict: Model state dict
        prefix_key: Key prefix (without .weight suffix)

    Returns:
        Weight tensor or None if not found
    """
    # Try direct weight first
    w = state_dict.get(f"{prefix_key}.weight")
    if w is not None:
        return w

    # Try weight normalization format
    g = state_dict.get(f"{prefix_key}.weight_g")
    v = state_dict.get(f"{prefix_key}.weight_v")
    if g is not None and v is not None:
        # Compute normalized weight: w = g * (v / ||v||)
        # Normalize across all dims except the first (output channels)
        dims = tuple(range(1, v.dim()))
        v_norm = v / (torch.norm(v, dim=dims, keepdim=True) + 1e-7)
        return g * v_norm

    return None


class ResBlock:
    """
    Residual block with dilated convolutions.

    Two variants:
        ResBlock1: 3 pairs of (dilated_conv, conv)
        ResBlock2: 2 single dilated convs
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: Tuple[int, ...],
        conv_weights: List[Any],
        conv_biases: List[Any],
        block_type: str = "1",  # "1" or "2"
        device: Optional[Any] = None,
    ):
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.conv_weights = conv_weights
        self.conv_biases = conv_biases
        self.block_type = block_type
        self.device = device

    def __call__(self, x: Any, x_mask: Optional[Any] = None) -> Any:
        # Check if input is PyTorch tensor
        is_torch = isinstance(x, torch.Tensor)

        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask)
        return self._forward_ttnn(x, x_mask)

    def _forward_pytorch(self, x, x_mask):
        # Helper to convert TTNN tensors to PyTorch (and match dtype)
        def to_torch(t, dtype=torch.float32):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.to(dtype) if t.dtype != dtype else t
            if TTNN_AVAILABLE:
                return ttnn.to_torch(t).to(dtype)
            return t

        if self.block_type == "1":
            # ResBlock1: pairs of (dilated, non-dilated)
            for i in range(len(self.dilations)):
                xt = F.leaky_relu(x, LRELU_SLOPE)
                if x_mask is not None:
                    xt = xt * x_mask

                # Dilated conv
                dilation = self.dilations[i]
                padding = (self.kernel_size * dilation - dilation) // 2
                w1 = to_torch(self.conv_weights[i * 2])
                w1 = w1.squeeze(2) if w1.dim() == 4 else w1
                xt = F.conv1d(xt, w1, to_torch(self.conv_biases[i * 2]), padding=padding, dilation=dilation)

                xt = F.leaky_relu(xt, LRELU_SLOPE)
                if x_mask is not None:
                    xt = xt * x_mask

                # Non-dilated conv
                padding = (self.kernel_size - 1) // 2
                w2 = to_torch(self.conv_weights[i * 2 + 1])
                w2 = w2.squeeze(2) if w2.dim() == 4 else w2
                xt = F.conv1d(xt, w2, to_torch(self.conv_biases[i * 2 + 1]), padding=padding)

                x = xt + x
        else:
            # ResBlock2: single dilated convs
            for i, dilation in enumerate(self.dilations):
                xt = F.leaky_relu(x, LRELU_SLOPE)
                if x_mask is not None:
                    xt = xt * x_mask

                padding = (self.kernel_size * dilation - dilation) // 2
                w = to_torch(self.conv_weights[i])
                w = w.squeeze(2) if w.dim() == 4 else w
                xt = F.conv1d(xt, w, to_torch(self.conv_biases[i]), padding=padding, dilation=dilation)

                x = xt + x

        if x_mask is not None:
            x = x * x_mask
        return x

    def _forward_ttnn(self, x, x_mask):
        if self.block_type == "1":
            for i in range(len(self.dilations)):
                xt = ttnn.leaky_relu(x, LRELU_SLOPE)
                if x_mask is not None:
                    xt = ttnn.multiply(xt, x_mask)

                # Dilated conv
                dilation = self.dilations[i]
                padding = (self.kernel_size * dilation - dilation) // 2
                xt = ttnn_conv1d(xt, self.conv_weights[i * 2], self.conv_biases[i * 2],
                                padding=padding, dilation=dilation, device=self.device)

                xt = ttnn.leaky_relu(xt, LRELU_SLOPE)
                if x_mask is not None:
                    xt = ttnn.multiply(xt, x_mask)

                # Non-dilated conv
                padding = (self.kernel_size - 1) // 2
                xt = ttnn_conv1d(xt, self.conv_weights[i * 2 + 1], self.conv_biases[i * 2 + 1],
                                padding=padding, device=self.device)

                x = ttnn.add(xt, x)
        else:
            for i, dilation in enumerate(self.dilations):
                xt = ttnn.leaky_relu(x, LRELU_SLOPE)
                if x_mask is not None:
                    xt = ttnn.multiply(xt, x_mask)

                padding = (self.kernel_size * dilation - dilation) // 2
                xt = ttnn_conv1d(xt, self.conv_weights[i], self.conv_biases[i],
                                padding=padding, dilation=dilation, device=self.device)

                x = ttnn.add(xt, x)

        if x_mask is not None:
            x = ttnn.multiply(x, x_mask)
        return x


class TTNNGenerator:
    """
    HiFi-GAN Generator (Vocoder).

    Architecture:
        1. Pre-conv: Conv1d to expand channels
        2. Upsampling blocks:
            - ConvTranspose1d for upsampling
            - Multiple ResBlocks with different kernel sizes
        3. Post-conv: Conv1d to single channel audio

    Output: Raw audio waveform
    """

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[Tuple[int, ...]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        gin_channels: int,
        conv_pre_weight: Any,
        conv_pre_bias: Any,
        ups_weights: List[Any],
        ups_biases: List[Any],
        resblock_weights: List[List[Any]],
        resblock_biases: List[List[Any]],
        conv_post_weight: Any,
        conv_post_bias: Optional[Any],
        cond_weight: Optional[Any] = None,
        cond_bias: Optional[Any] = None,
        device: Optional[Any] = None,
    ):
        self.initial_channel = initial_channel
        self.resblock_type = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.gin_channels = gin_channels
        self.device = device

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Pre-conv
        self.conv_pre_weight = conv_pre_weight
        self.conv_pre_bias = conv_pre_bias

        # Upsampling convolutions
        self.ups_weights = ups_weights
        self.ups_biases = ups_biases

        # ResBlocks - organized as [upsample_idx][kernel_idx]
        self.resblocks = []
        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2 ** (i + 1))
            blocks_for_upsample = []
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                rb_idx = i * self.num_kernels + j
                block = ResBlock(
                    channels=ch,
                    kernel_size=k,
                    dilations=d,
                    conv_weights=resblock_weights[rb_idx],
                    conv_biases=resblock_biases[rb_idx],
                    block_type=resblock,
                    device=device,
                )
                blocks_for_upsample.append(block)
            self.resblocks.append(blocks_for_upsample)

        # Post-conv
        self.conv_post_weight = conv_post_weight
        self.conv_post_bias = conv_post_bias

        # Conditioning (speaker embedding)
        self.cond_weight = cond_weight
        self.cond_bias = cond_bias

    def __call__(self, x: Any, g: Optional[Any] = None) -> Any:
        """
        Generate audio from latent representation.

        Args:
            x: Latent tensor [B, inter_channels, T]
            g: Speaker conditioning [B, gin_channels, 1] (optional)

        Returns:
            Audio waveform [B, 1, T_audio]
        """
        # Check if input is PyTorch tensor
        is_torch = isinstance(x, torch.Tensor)

        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, g)
        return self._forward_ttnn(x, g)

    def _forward_pytorch(self, x, g):
        # Helper to convert TTNN tensors to PyTorch (and match dtype)
        def to_torch(t, dtype=torch.float32):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.to(dtype) if t.dtype != dtype else t
            if TTNN_AVAILABLE:
                return ttnn.to_torch(t).to(dtype)
            return t

        # Pre-conv
        pre_w = to_torch(self.conv_pre_weight)
        w = pre_w.squeeze(2) if pre_w.dim() == 4 else pre_w
        x = F.conv1d(x, w, to_torch(self.conv_pre_bias), padding=3)

        # Add conditioning
        if g is not None and self.cond_weight is not None:
            cond_w = to_torch(self.cond_weight)
            cw = cond_w.squeeze(2) if cond_w.dim() == 4 else cond_w
            x = x + F.conv1d(g, cw, to_torch(self.cond_bias))

        # Upsampling blocks
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)

            # Transposed convolution
            stride = self.upsample_rates[i]
            kernel_size = self.upsample_kernel_sizes[i]
            padding = (kernel_size - stride) // 2
            ups_w = to_torch(self.ups_weights[i])
            w = ups_w.squeeze(2) if ups_w.dim() == 4 else ups_w
            x = F.conv_transpose1d(x, w, to_torch(self.ups_biases[i]), stride=stride, padding=padding)

            # Sum of ResBlocks
            xs = None
            for j in range(self.num_kernels):
                rb_out = self.resblocks[i][j](x)
                xs = rb_out if xs is None else xs + rb_out
            x = xs / self.num_kernels

        # Final activation + post-conv
        x = F.leaky_relu(x, LRELU_SLOPE)
        post_w = to_torch(self.conv_post_weight)
        w = post_w.squeeze(2) if post_w.dim() == 4 else post_w
        x = F.conv1d(x, w, to_torch(self.conv_post_bias), padding=3)
        x = torch.tanh(x)

        return x

    def _forward_ttnn(self, x, g):
        # Pre-conv
        x = ttnn_conv1d(x, self.conv_pre_weight, self.conv_pre_bias,
                       padding=3, device=self.device)

        # Add conditioning
        if g is not None and self.cond_weight is not None:
            cond = ttnn_conv1d(g, self.cond_weight, self.cond_bias, device=self.device)
            x = ttnn.add(x, cond)

        # Upsampling blocks
        for i in range(self.num_upsamples):
            x = ttnn.leaky_relu(x, LRELU_SLOPE)

            # Transposed convolution for upsampling
            stride = self.upsample_rates[i]
            kernel_size = self.upsample_kernel_sizes[i]
            padding = (kernel_size - stride) // 2
            x = ttnn_conv_transpose1d(
                x, self.ups_weights[i], self.ups_biases[i],
                stride=stride, padding=padding, device=self.device
            )

            # Sum of ResBlocks
            xs = None
            for j in range(self.num_kernels):
                rb_out = self.resblocks[i][j](x)
                if xs is None:
                    xs = rb_out
                else:
                    xs = ttnn.add(xs, rb_out)

            # Average
            x = ttnn.multiply(xs, 1.0 / self.num_kernels)

        # Final activation + post-conv + tanh
        x = ttnn.leaky_relu(x, LRELU_SLOPE)
        x = ttnn_conv1d(x, self.conv_post_weight, self.conv_post_bias,
                       padding=3, device=self.device)
        x = ttnn.tanh(x)

        return x

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        prefix: str,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[Tuple[int, ...]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        gin_channels: int = 0,
        device: Optional[Any] = None,
    ) -> "TTNNGenerator":
        """Create Generator from state dict."""

        num_kernels = len(resblock_kernel_sizes)
        num_upsamples = len(upsample_rates)

        # Pre-conv (with weight normalization support)
        conv_pre_weight = get_weight_from_state_dict(state_dict, f"{prefix}.conv_pre")
        conv_pre_bias = state_dict.get(f"{prefix}.conv_pre.bias")

        # Upsampling weights (with weight normalization support)
        ups_weights = []
        ups_biases = []
        for i in range(num_upsamples):
            w = get_weight_from_state_dict(state_dict, f"{prefix}.ups.{i}")
            b = state_dict.get(f"{prefix}.ups.{i}.bias")
            ups_weights.append(w)
            ups_biases.append(b)

        # ResBlock weights (with weight normalization support)
        resblock_weights = []
        resblock_biases = []

        for rb_idx in range(num_upsamples * num_kernels):
            if resblock == "1":
                # ResBlock1 has convs1 and convs2
                w1 = [get_weight_from_state_dict(state_dict, f"{prefix}.resblocks.{rb_idx}.convs1.{j}") for j in range(3)]
                b1 = [state_dict.get(f"{prefix}.resblocks.{rb_idx}.convs1.{j}.bias") for j in range(3)]
                w2 = [get_weight_from_state_dict(state_dict, f"{prefix}.resblocks.{rb_idx}.convs2.{j}") for j in range(3)]
                b2 = [state_dict.get(f"{prefix}.resblocks.{rb_idx}.convs2.{j}.bias") for j in range(3)]
                weights = []
                biases = []
                for j in range(3):
                    weights.extend([w1[j], w2[j]])
                    biases.extend([b1[j], b2[j]])
                resblock_weights.append(weights)
                resblock_biases.append(biases)
            else:
                # ResBlock2 has single convs list
                weights = []
                biases = []
                for j in range(2):  # 2 convs
                    w = get_weight_from_state_dict(state_dict, f"{prefix}.resblocks.{rb_idx}.convs.{j}")
                    b = state_dict.get(f"{prefix}.resblocks.{rb_idx}.convs.{j}.bias")
                    weights.append(w)
                    biases.append(b)
                resblock_weights.append(weights)
                resblock_biases.append(biases)

        # Post-conv (with weight normalization support)
        conv_post_weight = get_weight_from_state_dict(state_dict, f"{prefix}.conv_post")
        conv_post_bias = state_dict.get(f"{prefix}.conv_post.bias")

        # Conditioning (with weight normalization support)
        cond_weight = get_weight_from_state_dict(state_dict, f"{prefix}.cond") if gin_channels > 0 else None
        cond_bias = state_dict.get(f"{prefix}.cond.bias") if gin_channels > 0 else None

        return cls(
            initial_channel=initial_channel,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gin_channels=gin_channels,
            conv_pre_weight=conv_pre_weight,
            conv_pre_bias=conv_pre_bias,
            ups_weights=ups_weights,
            ups_biases=ups_biases,
            resblock_weights=resblock_weights,
            resblock_biases=resblock_biases,
            conv_post_weight=conv_post_weight,
            conv_post_bias=conv_post_bias,
            cond_weight=cond_weight,
            cond_bias=cond_bias,
            device=device,
        )
