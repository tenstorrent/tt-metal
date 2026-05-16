# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN ConvTranspose1d wrapper via ttnn.conv_transpose2d.

No native ttnn.conv_transpose1d exists, so we map 1D→2D with H=1.

This is the HIGHEST RISK operator for RVC — HiFi-GAN uses large strides (10, 6, 2, 2)
and large kernels (16, 16, 4, 4) for upsampling. All configs validated on N300.

Input format: [B, 1, L, C_in] (NHWC with H=1)
Weight format: [in_ch, out_ch/groups, 1, kernel_size]
Output: [B, 1, L_out, C_out] (NHWC with H=1)

Output size formula:
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
"""

import torch
import ttnn
from typing import Optional, Tuple


class TTNNConvTranspose1d:
    """
    ConvTranspose1d wrapper using ttnn.conv_transpose2d internally.

    Tensor layout notes:
        - Input must be NHWC: [B, 1, L, C] (H=1 for 1D)
        - Weights: [in_ch, out_ch/groups, 1, kernel_size]
        - Output: NHWC on device

    Known limitations:
        - No native 1D API — manual H=1 mapping required
        - output_padding only tested with (0, 0)
        - Large strides (10, 6) consume significant L1 memory
    """

    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias

        self._weight_host = None
        self._bias_host = None

    def load_weights(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Load PyTorch ConvTranspose1d weights.

        Args:
            weight: [in_ch, out_ch/groups, kernel_size] — PyTorch transposed conv format.
            bias: [out_ch] or None.
        """
        # conv_transpose2d expects: [in_ch, out_ch/groups, kH, kW]
        w_4d = weight.unsqueeze(2).float()  # [in_ch, out_ch/groups, 1, kernel_size]
        self._weight_host = ttnn.from_torch(w_4d, dtype=ttnn.bfloat16)

        if bias is not None:
            b_4d = bias.reshape(1, 1, 1, -1).float()
            self._bias_host = ttnn.from_torch(b_4d, dtype=ttnn.bfloat16)
        else:
            self._bias_host = None

    def __call__(
        self,
        x: ttnn.Tensor,
        batch_size: int,
        input_length: int,
    ) -> Tuple[ttnn.Tensor, int]:
        """
        Run ConvTranspose1d via conv_transpose2d with H=1.

        Args:
            x: Input tensor in NHWC [B, 1, L, C_in]. Can be host tensor.
            batch_size: Explicit batch size.
            input_length: Input sequence length.

        Returns:
            Tuple of (output_tensor, output_length).
        """
        result = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self._weight_host,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self._bias_host,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, self.padding),
            output_padding=(0, self.output_padding),
            groups=self.groups,
            batch_size=batch_size,
            input_height=1,
            input_width=input_length,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
        )

        # Parse return value
        if isinstance(result, tuple) and len(result) >= 2:
            output_tensor = result[0]
            dims = result[1]
            if isinstance(dims, (list, tuple)):
                out_w = dims[1] if len(dims) == 2 else dims[0]
            else:
                out_w = dims
        else:
            output_tensor = result
            # Compute manually
            out_w = ((input_length - 1) * self.stride
                     - 2 * self.padding
                     + self.kernel_size
                     + self.output_padding)

        return output_tensor, out_w

    @staticmethod
    def postprocess_output(
        output_tensor: ttnn.Tensor,
        batch_size: int,
        output_length: int,
        out_channels: int,
    ) -> torch.Tensor:
        """
        Convert TTNN conv_transpose2d output to torch [B, C_out, L_out].

        Args:
            output_tensor: TTNN tensor from conv_transpose2d.
            batch_size: Batch size.
            output_length: Output sequence length.
            out_channels: Number of output channels.

        Returns:
            torch.Tensor in [B, C_out, L_out] format.
        """
        try:
            output_tensor = ttnn.sharded_to_interleaved(output_tensor)
        except RuntimeError:
            pass

        output = ttnn.from_device(output_tensor)
        out_torch = ttnn.to_torch(output).float()
        out_torch = out_torch.reshape(batch_size, 1, output_length, -1)
        out_torch = out_torch[:, :, :, :out_channels]
        out_torch = out_torch.squeeze(1)  # [B, L_out, C_out]
        out_torch = out_torch.permute(0, 2, 1)  # [B, C_out, L_out]

        return out_torch
