# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Conv1d wrapper using native ttnn.conv1d API.

Uses the official ttnn.conv1d op which handles 1D→2D mapping internally.
Follows the Whisper implementation pattern for audio model conv layers.

Input format: [batch, seq_len, channels] (channels-last)
Weight format: [out_channels, in_channels // groups, kernel_size] (PyTorch format)

References:
    - Whisper: models/demos/audio/whisper/tt/ttnn_optimized_functional_whisper.py (L856-873)
    - ttnn.conv1d docs: "Applies a 1D convolution over an input signal"
"""

import torch
import ttnn
from typing import Optional, Tuple


class TTNNConv1d:
    """
    Conv1d wrapper using native ttnn.conv1d.

    Key behaviors:
        - Accepts [B, L, C] channels-last input (same as ttnn.conv1d expects)
        - Weights in PyTorch format [out_ch, in_ch/groups, kernel_size]
        - Supports weight caching via return_weights_and_bias
        - Output is TTNN tensor on device in TILE_LAYOUT

    Tensor layout notes:
        - Input: ROW_MAJOR on host → conv1d handles internal layout
        - Weights: ROW_MAJOR on host (Whisper pattern, NOT TILE)
        - Output: TILE_LAYOUT on device (height-sharded by default)
        - After conv: call ttnn.sharded_to_interleaved() if needed for downstream ops

    Memory config:
        - Stage 1: default (interleaved DRAM) — no explicit sharding
        - Conv1d may internally shard to height-sharded L1
    """

    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_config: Optional[ttnn.Conv2dConfig] = None,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        # Weight tensors (host-side until first call, then device-cached)
        self._weight_host = None
        self._bias_host = None
        self._weight_device = None  # Cached after first call

        # Conv config — Stage 1: default, correctness-first
        if conv_config is not None:
            self.conv_config = conv_config
        else:
            self.conv_config = ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat16,
                deallocate_activation=False,
                reallocate_halo_output=True,
            )

    def load_weights(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Load PyTorch Conv1d weights.

        Weights are stored in ROW_MAJOR layout on host (Whisper pattern).
        They will be preprocessed and cached on device during the first forward call.

        Args:
            weight: [out_ch, in_ch/groups, kernel_size] — standard PyTorch format.
            bias: [out_ch] or None.
        """
        # Store as TTNN host tensors in ROW_MAJOR (Whisper pattern)
        self._weight_host = ttnn.from_torch(
            weight.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        if bias is not None:
            self._bias_host = ttnn.from_torch(
                bias.reshape(1, 1, bias.shape[0]).float(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        else:
            self._bias_host = None

        # Clear device cache (weights need re-upload)
        self._weight_device = None

    def __call__(
        self,
        x: ttnn.Tensor,
        batch_size: int,
        input_length: int,
    ) -> Tuple[ttnn.Tensor, int]:
        """
        Run Conv1d using native ttnn.conv1d.

        Args:
            x: Input tensor [B, L, C] channels-last. Can be host or device tensor.
            batch_size: Explicit batch size.
            input_length: Sequence length of input.

        Returns:
            Tuple of (output_tensor, output_length).
            output_tensor is on device, may be height-sharded.
        """
        weight = self._weight_device if self._weight_device is not None else self._weight_host

        result = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=weight,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_length=input_length,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias_tensor=self._bias_host,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        # Parse return: (output, out_len, (weight_device, bias_device))
        output_tensor = result[0]
        output_length = result[1]
        weights_and_bias = result[2]

        # Cache preprocessed weights on device (Whisper pattern, line 874)
        self._weight_device = weights_and_bias[0]

        return output_tensor, output_length

    @staticmethod
    def postprocess_output(
        output_tensor: ttnn.Tensor,
        batch_size: int,
        output_length: int,
        out_channels: int,
    ) -> torch.Tensor:
        """
        Convert TTNN conv1d output to torch [B, C_out, L_out] format.

        Conv1d output is [B, 1, L_out, C_out] in NHWC. We convert to
        PyTorch's [B, C_out, L_out] for comparison.

        Args:
            output_tensor: TTNN tensor from conv1d (may be sharded).
            batch_size: Batch size.
            output_length: Output sequence length from conv1d.
            out_channels: Number of output channels.

        Returns:
            torch.Tensor in [B, C_out, L_out] format.
        """
        # If sharded, convert to interleaved first
        try:
            output_tensor = ttnn.sharded_to_interleaved(output_tensor)
        except RuntimeError:
            pass  # Already interleaved

        output = ttnn.from_device(output_tensor)
        out_torch = ttnn.to_torch(output).float()

        # Output shape from conv1d is [1, 1, L_out, C_out]
        out_torch = out_torch.reshape(batch_size, 1, output_length, -1)
        out_torch = out_torch[:, :, :, :out_channels]  # Remove channel padding
        out_torch = out_torch.squeeze(1)  # [B, L_out, C_out]
        out_torch = out_torch.permute(0, 2, 1)  # [B, C_out, L_out] — PyTorch format

        return out_torch
