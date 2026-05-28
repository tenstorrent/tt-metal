# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 Conformer convolution module.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::conformer_convolution_module_forward``,
which reproduces the forward of HuggingFace
``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2ConformerConvolutionModule``::

    x = LayerNorm(x)                              # (B, T, C)
    # optional mask zeroing (False/0 positions -> 0)
    x = x.transpose(1, 2)                         # (B, C, T)
    x = pointwise_conv1(x)                        # (B, 2C, T), no bias, k=1
    x = GLU(dim=1)(x)                             # (B, C, T)
    x = F.pad(x, (kernel_size - 1, 0))            # causal left padding
    x = depthwise_conv(x)                         # (B, C, T), groups=C, k=31
    x = LayerNorm(x.transpose(1, 2)).transpose(1, 2)  # (B, C, T)
    x = swish(x)
    x = pointwise_conv2(x)                        # (B, C, T), no bias, k=1
    return x.transpose(1, 2)                      # (B, T, C)

All three Conv1d layers have ``bias=False``. The depthwise Conv1d has
``groups = hidden_size`` and is CAUSAL (left padding ``kernel_size - 1``,
right padding 0). Both LayerNorms operate on the channel dim with eps 1e-5.

Implementation notes (TTNN port):

* The two pointwise convolutions (``k=1``) are realised as ``ttnn.linear``
  in NLC layout (input is ``[B, T, C]``), which is mathematically identical
  but maps onto the matmul kernel rather than the conv path.
* GLU is run with ``ttnn.glu(dim=-1)`` after pointwise_conv1, on the
  ``[B, T, 2C]`` activation -- same semantics as ``F.glu(dim=1)`` on the
  channel-first reference.
* The depthwise convolution is a 1D causal conv (``groups=C``, ``kernel=31``).
  ``ttnn.conv1d`` is used with explicit ``padding=[kernel_size - 1, 0]`` to
  realise the causal left-padding without zero-padding the activation on
  host. Input is laid out as ``[B, 1, T, C]`` (NHWC, as required by the
  conv1d op), and the output is reshaped back to ``[B, T, C]`` before the
  subsequent LayerNorm.
* The optional padding mask is converted to a ``[B, T, 1]`` broadcast
  multiplier and applied before the depthwise convolution (matches the HF
  ``masked_fill`` step).

For SeamlessM4T-v2-Large: ``hidden_size = 1024``, ``kernel_size = 31``.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class ConformerConvolutionModule(LightweightModule):
    """Causal depthwise-separable convolution branch of the Conformer block."""

    def __init__(
        self,
        device,
        layer_norm_weight: torch.Tensor,
        layer_norm_bias: torch.Tensor,
        pointwise_conv1_weight: torch.Tensor,  # (2C, C, 1)
        depthwise_conv_weight: torch.Tensor,  # (C, 1, K)
        depthwise_layer_norm_weight: torch.Tensor,
        depthwise_layer_norm_bias: torch.Tensor,
        pointwise_conv2_weight: torch.Tensor,  # (C, C, 1)
        kernel_size: int = 31,
        eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        self.eps = float(eps)
        self.kernel_size = int(kernel_size)

        hidden = int(layer_norm_weight.shape[-1])
        self.hidden = hidden
        two_hidden = int(pointwise_conv1_weight.shape[0])
        assert two_hidden == 2 * hidden, f"pointwise_conv1 out_channels={two_hidden} must equal 2*hidden={2*hidden}"
        assert int(pointwise_conv1_weight.shape[1]) == hidden
        assert int(pointwise_conv1_weight.shape[2]) == 1
        assert int(depthwise_conv_weight.shape[0]) == hidden
        assert int(depthwise_conv_weight.shape[1]) == 1
        assert int(depthwise_conv_weight.shape[2]) == self.kernel_size
        assert int(pointwise_conv2_weight.shape[0]) == hidden
        assert int(pointwise_conv2_weight.shape[1]) == hidden
        assert int(pointwise_conv2_weight.shape[2]) == 1

        # --- LayerNorm 1 (pre) weights ---------------------------------------
        self.ln1_weight = ttnn.from_torch(
            layer_norm_weight.reshape(1, 1, 1, hidden),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.ln1_bias = ttnn.from_torch(
            layer_norm_bias.reshape(1, 1, 1, hidden),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        # --- Pointwise conv 1 as linear (in_features=C, out_features=2C) -----
        # pointwise_conv1_weight: (2C, C, 1) -> squeeze last dim and transpose
        # to (C, 2C) for the ttnn.linear weight convention.
        pw1_w = pointwise_conv1_weight.squeeze(-1).transpose(0, 1).contiguous()
        self.pointwise_conv1_weight = ttnn.from_torch(
            pw1_w,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        # --- Depthwise conv1d weight ----------------------------------------
        # Conv1d weight is already in PyTorch [out=C, in_per_group=1, K] shape.
        # ttnn.conv1d accepts the same convention; store in row-major in DRAM
        # so the kernel can preprocess/shard it the first time the conv runs.
        self.depthwise_conv_weight = ttnn.from_torch(
            depthwise_conv_weight,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # --- LayerNorm 2 (depthwise post-conv) weights ----------------------
        self.ln2_weight = ttnn.from_torch(
            depthwise_layer_norm_weight.reshape(1, 1, 1, hidden),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.ln2_bias = ttnn.from_torch(
            depthwise_layer_norm_bias.reshape(1, 1, 1, hidden),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        # --- Pointwise conv 2 as linear (in=C, out=C) -----------------------
        pw2_w = pointwise_conv2_weight.squeeze(-1).transpose(0, 1).contiguous()
        self.pointwise_conv2_weight = ttnn.from_torch(
            pw2_w,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        # Precision / compute config: HiFi4 + fp32 dest accum -- matches the
        # other Conformer block kernels in this model.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Conv1dConfig for the depthwise pass.
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weight_dtype,
            shard_layout=None,  # let ttnn pick
            deallocate_activation=False,
        )
        self.conv_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # State for the conv weight after first call: ttnn returns a
        # preprocessed (sharded/tiled) weight which we cache and reuse.
        self._dw_weight_prepared = False

    def forward(
        self,
        x: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run the conformer convolution module.

        Args:
            x: input ttnn tensor of shape ``[B, T, C]`` in TILE_LAYOUT.
            attention_mask: optional ttnn tensor broadcastable to ``[B, T, 1]``
                where 0.0 marks padded positions to zero out before the
                depthwise convolution (1.0 means "keep"). The caller is
                responsible for converting the boolean HF mask to this float
                form.

        Returns:
            ttnn tensor of shape ``[B, T, C]`` in TILE_LAYOUT.
        """
        mc_dram = ttnn.DRAM_MEMORY_CONFIG
        batch = int(x.shape[0])
        seq_len = int(x.shape[1])
        hidden = self.hidden

        # 1. LayerNorm over the channel (last) dim.
        y = ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=self.ln1_weight,
            bias=self.ln1_bias,
            compute_kernel_config=self.compute_kernel_config,
        )

        # 2. Optional mask: zero padded positions before the depthwise conv.
        if attention_mask is not None:
            y = ttnn.multiply(y, attention_mask, memory_config=mc_dram)

        # 3. Pointwise conv1 (k=1) implemented as a linear over the channel dim.
        #    y: [B, T, C] -> [B, T, 2C]
        y_pw1 = ttnn.linear(
            y,
            self.pointwise_conv1_weight,
            memory_config=mc_dram,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(y)

        # 4. GLU along the last (channel) dim. ttnn.glu only supports the last
        #    dim of a rank-4 tensor, so reshape ``[B, T, 2C] -> [B, 1, T, 2C]``
        #    first; the output is then ``[B, 1, T, C]`` which is also the
        #    NHWC layout ttnn.conv1d expects (H=1).
        y_pw1_4d = ttnn.reshape(y_pw1, (batch, 1, seq_len, 2 * hidden), memory_config=mc_dram)
        ttnn.deallocate(y_pw1)
        y_dw_in = ttnn.glu(y_pw1_4d, dim=-1)
        ttnn.deallocate(y_pw1_4d)

        # 5. Depthwise conv1d (k=kernel_size, groups=C, causal).
        #    ttnn.conv1d input format is NHWC with H=1, i.e. [B, 1, T, C].
        # ttnn.conv1d wants ROW_MAJOR input; convert if needed.
        if y_dw_in.layout != ttnn.ROW_MAJOR_LAYOUT:
            y_dw_in = ttnn.to_layout(y_dw_in, ttnn.ROW_MAJOR_LAYOUT)

        dw_out, _y_out_len, [self.depthwise_conv_weight, _bias_tt] = ttnn.conv1d(
            input_tensor=y_dw_in,
            weight_tensor=self.depthwise_conv_weight,
            device=self.device,
            in_channels=hidden,
            out_channels=hidden,
            batch_size=batch,
            input_length=seq_len,
            kernel_size=self.kernel_size,
            stride=1,
            padding=[self.kernel_size - 1, 0],  # causal: left-only padding
            dilation=1,
            groups=hidden,
            bias_tensor=None,
            conv_config=self.conv_config,
            compute_config=self.conv_compute_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        ttnn.deallocate(y_dw_in)
        self._dw_weight_prepared = True

        # Reshape conv output back to [B, T, C]. ttnn.conv1d returns
        # [1, 1, B*T, C] (height-flattened NHWC) in TILE_LAYOUT typically.
        # We move to TILE_LAYOUT (for layer_norm) and reshape.
        if dw_out.layout != ttnn.TILE_LAYOUT:
            dw_out = ttnn.to_layout(dw_out, ttnn.TILE_LAYOUT)
        dw_out = ttnn.reshape(dw_out, (batch, seq_len, hidden), memory_config=mc_dram)

        # 6. LayerNorm over the channel dim.
        z = ttnn.layer_norm(
            dw_out,
            epsilon=self.eps,
            weight=self.ln2_weight,
            bias=self.ln2_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(dw_out)

        # 7. Swish (SiLU) activation.
        z = ttnn.silu(z, memory_config=mc_dram)

        # 8. Pointwise conv2 (k=1) as a linear projection back to hidden.
        out = ttnn.linear(
            z,
            self.pointwise_conv2_weight,
            memory_config=mc_dram,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(z)
        return out
