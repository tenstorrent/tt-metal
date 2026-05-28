# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 ``HifiGanResidualBlock``.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::hifigan_residual_block_forward``,
which reproduces the forward of HuggingFace
``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.HifiGanResidualBlock``.

For each of ``N = len(dilation)`` (default ``(1, 3, 5)``) inner stages::

    residual = h
    h = leaky_relu(h, slope)
    h = convs1[i](h)            # Conv1d k, dilation=dilation[i], same-pad
    h = leaky_relu(h, slope)
    h = convs2[i](h)            # Conv1d k, dilation=1,           same-pad
    h = h + residual

All Conv1d layers use ``stride=1`` and "same" padding
``(kernel_size * dilation - dilation) // 2`` so the time dim is preserved.

Implementation notes (TTNN port):

* Input is ``[B, C, T]`` (channels first) per the HF API. ``ttnn.conv1d``
  expects NHWC with ``H=1`` -> ``[B, 1, T, C]`` row-major. We permute /
  reshape between the two on every conv.
* ``ttnn.conv1d`` supports ``dilation > 1`` (see
  ``tests/ttnn/unit_tests/operations/conv/test_conv1d.py::test_conv1d_dilation``).
* ``ttnn.leaky_relu(slope=0.1)`` is the channel-wise activation; it works
  in TILE layout. We move conv outputs from row-major back to TILE layout
  via ``ttnn.to_layout`` before the activation / residual add.
* The residual is captured BEFORE the first ``leaky_relu`` of each stage,
  in NHWC ``[B, 1, T, C]`` ROW_MAJOR layout, so we can add it to the
  conv2 output (which we reshape back to the same layout/shape) without
  another permute.
* Compute config: HiFi4 + fp32 dest-acc, matching the existing
  ``variance_predictor`` / ``conformer_convolution_module`` patterns.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class HifiGanResidualBlock(LightweightModule):
    """HiFi-GAN MRF residual block.

    Args:
        device: ttnn device.
        convs1_weights: sequence of ``len(dilation)`` torch tensors of shape
            ``(C, C, K)`` for the dilated conv in each stage.
        convs1_biases: sequence of torch tensors of shape ``(C,)``.
        convs2_weights: sequence of ``len(dilation)`` torch tensors of shape
            ``(C, C, K)`` for the dilation=1 conv in each stage.
        convs2_biases: sequence of torch tensors of shape ``(C,)``.
        kernel_size: conv kernel size (default 3).
        dilation: tuple of dilation factors (default ``(1, 3, 5)``).
        leaky_relu_slope: negative slope (default 0.1).
        weight_dtype: storage dtype for conv weights on device.
    """

    def __init__(
        self,
        device,
        convs1_weights: Sequence[torch.Tensor],
        convs1_biases: Sequence[torch.Tensor],
        convs2_weights: Sequence[torch.Tensor],
        convs2_biases: Sequence[torch.Tensor],
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
        leaky_relu_slope: float = 0.1,
        weight_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.kernel_size = int(kernel_size)
        self.dilation = tuple(int(d) for d in dilation)
        self.leaky_relu_slope = float(leaky_relu_slope)
        n = len(self.dilation)
        assert (
            len(convs1_weights) == len(convs1_biases) == len(convs2_weights) == len(convs2_biases) == n
        ), "convs1/convs2/dilation length mismatch"

        # All convs share the same channel count.
        channels = int(convs1_weights[0].shape[0])
        for i in range(n):
            assert int(convs1_weights[i].shape[0]) == channels
            assert int(convs1_weights[i].shape[1]) == channels
            assert int(convs1_weights[i].shape[2]) == self.kernel_size
            assert tuple(convs1_biases[i].shape) == (channels,)
            assert int(convs2_weights[i].shape[0]) == channels
            assert int(convs2_weights[i].shape[1]) == channels
            assert int(convs2_weights[i].shape[2]) == self.kernel_size
            assert tuple(convs2_biases[i].shape) == (channels,)
        self.channels = channels

        # Conv weights stay in row-major; ttnn.conv1d preprocesses them on the
        # first call. We cache the prepared tensors in-place so subsequent
        # forwards skip the preprocessing.
        self.convs1_weight = [
            ttnn.from_torch(w, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT) for w in convs1_weights
        ]
        self.convs1_bias = [
            ttnn.from_torch(b.reshape(1, 1, 1, channels), dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            for b in convs1_biases
        ]
        self.convs2_weight = [
            ttnn.from_torch(w, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT) for w in convs2_weights
        ]
        self.convs2_bias = [
            ttnn.from_torch(b.reshape(1, 1, 1, channels), dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            for b in convs2_biases
        ]

        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weight_dtype,
            shard_layout=None,  # auto-pick
            deallocate_activation=False,
        )
        self.conv_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            # packer_l1_acc=True is the SKILL.md "standard recipe" -- enables
            # in-tile packer accumulation. Matches the conv config used by the
            # newer tt-metal conv1d kernels. Tracy validated: PCC unchanged.
            packer_l1_acc=True,
        )

    @staticmethod
    def _same_padding(k: int, d: int) -> int:
        return (k * d - d) // 2

    def _conv1d_stage(
        self,
        x_nhwc_rm: ttnn.Tensor,
        weight_list_attr: str,
        bias_list_attr: str,
        idx: int,
        dilation: int,
        batch: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        """Run one Conv1d on ``[B, 1, T, C]`` row-major NHWC input.

        Returns the conv output reshaped to ``[B, 1, T, C]`` row-major
        (suitable for layout conversion + activation).
        """
        weight_list = getattr(self, weight_list_attr)
        bias_list = getattr(self, bias_list_attr)
        weight_tt = weight_list[idx]
        bias_tt = bias_list[idx]

        pad = self._same_padding(self.kernel_size, dilation)

        out, _out_len, [new_w, new_b] = ttnn.conv1d(
            input_tensor=x_nhwc_rm,
            weight_tensor=weight_tt,
            device=self.device,
            in_channels=self.channels,
            out_channels=self.channels,
            batch_size=batch,
            input_length=seq_len,
            kernel_size=self.kernel_size,
            stride=1,
            padding=[pad, pad],
            dilation=dilation,
            groups=1,
            bias_tensor=bias_tt,
            conv_config=self.conv_config,
            compute_config=self.conv_compute_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        # Cache prepared weights/bias for subsequent calls.
        weight_list[idx] = new_w
        bias_list[idx] = new_b
        return out

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Run the HiFi-GAN residual block.

        Args:
            x: ttnn tensor of shape ``[B, C, T]`` (channels first), TILE_LAYOUT.

        Returns:
            ttnn tensor of shape ``[B, C, T]`` (channels first), TILE_LAYOUT.
        """
        mc_dram = ttnn.DRAM_MEMORY_CONFIG
        batch = int(x.shape[0])
        channels = int(x.shape[1])
        seq_len = int(x.shape[2])
        assert channels == self.channels, f"channels mismatch: got {channels} vs expected {self.channels}"

        # Permute [B, C, T] -> [B, T, C] (NLC). ttnn.permute works on TILE.
        h_tile = ttnn.permute(x, (0, 2, 1), memory_config=mc_dram)
        # We carry ``h_tile`` as a ``[B, T, C]`` TILE tensor between iterations
        # (this is the residual sum buffer). It is converted to row-major NHWC
        # ``[B, 1, T, C]`` each time we need to feed ``ttnn.conv1d``.

        for i, d in enumerate(self.dilation):
            # Residual (TILE [B, T, C]).
            residual_t = h_tile

            # 1. LeakyReLU on TILE-layout tensor.
            y_tile = ttnn.leaky_relu(residual_t, self.leaky_relu_slope, memory_config=mc_dram)

            # 2. Row-major NHWC for conv1d: [B, T, C] -> [B, 1, T, C].
            y_rm = ttnn.to_layout(y_tile, ttnn.ROW_MAJOR_LAYOUT)
            y_rm = ttnn.reshape(y_rm, (batch, 1, seq_len, channels))

            # 3. convs1[i]: dilated Conv1d.
            h_conv = self._conv1d_stage(
                y_rm,
                weight_list_attr="convs1_weight",
                bias_list_attr="convs1_bias",
                idx=i,
                dilation=d,
                batch=batch,
                seq_len=seq_len,
            )
            # conv1d returns [1, 1, B*T, C] flattened NHWC.

            # 4. Move to TILE, reshape to [B, T, C].
            if h_conv.layout != ttnn.TILE_LAYOUT:
                h_conv = ttnn.to_layout(h_conv, ttnn.TILE_LAYOUT)
            h_conv = ttnn.reshape(h_conv, (batch, seq_len, channels))

            # 5. LeakyReLU.
            y_tile = ttnn.leaky_relu(h_conv, self.leaky_relu_slope, memory_config=mc_dram)

            # 6. Row-major NHWC for conv2.
            y_rm = ttnn.to_layout(y_tile, ttnn.ROW_MAJOR_LAYOUT)
            y_rm = ttnn.reshape(y_rm, (batch, 1, seq_len, channels))

            # 7. convs2[i]: Conv1d, dilation=1.
            h_conv = self._conv1d_stage(
                y_rm,
                weight_list_attr="convs2_weight",
                bias_list_attr="convs2_bias",
                idx=i,
                dilation=1,
                batch=batch,
                seq_len=seq_len,
            )

            # 8. Move to TILE, reshape to [B, T, C].
            if h_conv.layout != ttnn.TILE_LAYOUT:
                h_conv = ttnn.to_layout(h_conv, ttnn.TILE_LAYOUT)
            h_conv = ttnn.reshape(h_conv, (batch, seq_len, channels))

            # 9. Residual add (both TILE [B, T, C]).
            h_tile = ttnn.add(h_conv, residual_t, memory_config=mc_dram)

        # h_tile is TILE ``[B, T, C]``. Permute back to ``[B, C, T]``.
        out = ttnn.permute(h_tile, (0, 2, 1), memory_config=mc_dram)
        return out
