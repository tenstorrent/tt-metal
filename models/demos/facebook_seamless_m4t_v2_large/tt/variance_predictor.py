# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 ``VariancePredictor`` block.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::variance_predictor_forward``,
which reproduces the forward of HuggingFace
``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2VariancePredictor``::

    if padding_mask is not None:
        x = x.masked_fill(~padding_mask.unsqueeze(-1), 0)        # (B, T, C)
    x = conv1(x.transpose(1, 2))                                 # (B, H, T)
    x = relu(x).transpose(1, 2)                                  # (B, T, H)
    x = ln1(x)                                                   # (B, T, H)
    if padding_mask is not None:
        x = x.masked_fill(~padding_mask.unsqueeze(-1), 0)        # (B, T, H)
    x = conv2(x.transpose(1, 2))                                 # (B, H, T)
    x = relu(x).transpose(1, 2)                                  # (B, T, H)
    x = ln2(x)                                                   # (B, T, H)
    return proj(x).squeeze(-1)                                   # (B, T)

For SeamlessM4T-v2-Large T2U duration predictor:
``embed_dim = 1024``, ``hidden_dim = 256``, ``kernel_size = 3``.

Implementation notes (TTNN port):

* Both Conv1d layers use ``kernel_size=3`` with ``padding="same"`` -> symmetric
  padding of ``(kernel_size - 1) // 2 = 1`` on each side. Both have bias.
* ``ttnn.conv1d`` is used in NLC/NHWC layout (``[B, 1, T, C]`` row-major) to
  realise the convolutions. After the conv we move back to ``[B, T, H]`` tile
  layout for the subsequent ReLU + LayerNorm.
* Both LayerNorms are over the channel (hidden) dim, eps 1e-5, with their own
  weight/bias.
* The final projection (``Linear(H, 1)``) is run via ``ttnn.linear`` and the
  squeeze of the trailing scalar dim is realised via ``ttnn.reshape`` from
  ``[B, T, 1]`` to ``[B, T]``.
* The optional padding mask is broadcast-multiplied as a ``[B, T, 1]`` float
  tensor BEFORE each Conv1d (mirroring the HF ``masked_fill`` step).
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class VariancePredictor(LightweightModule):
    """Duration variance predictor used by the T2U decoder.

    Args:
        device: ttnn device.
        conv1_weight: torch.Tensor shape ``(hidden, embed, kernel)``.
        conv1_bias:   torch.Tensor shape ``(hidden,)``.
        ln1_weight:   torch.Tensor shape ``(hidden,)``.
        ln1_bias:     torch.Tensor shape ``(hidden,)``.
        conv2_weight: torch.Tensor shape ``(hidden, hidden, kernel)``.
        conv2_bias:   torch.Tensor shape ``(hidden,)``.
        ln2_weight:   torch.Tensor shape ``(hidden,)``.
        ln2_bias:     torch.Tensor shape ``(hidden,)``.
        proj_weight:  torch.Tensor shape ``(1, hidden)``.
        proj_bias:    torch.Tensor shape ``(1,)``.
        kernel_size:  Conv1d kernel size (must be odd; default 3).
        eps:          LayerNorm epsilon (default 1e-5).
        weight_dtype: storage dtype for non-conv weights on device.
        weight_memory_config: where to store non-conv weights (default DRAM).
    """

    def __init__(
        self,
        device,
        conv1_weight: torch.Tensor,
        conv1_bias: torch.Tensor,
        ln1_weight: torch.Tensor,
        ln1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        conv2_bias: torch.Tensor,
        ln2_weight: torch.Tensor,
        ln2_bias: torch.Tensor,
        proj_weight: torch.Tensor,
        proj_bias: torch.Tensor,
        kernel_size: int = 3,
        eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        self.eps = float(eps)
        self.kernel_size = int(kernel_size)
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for padding='same', got {self.kernel_size}")
        self.pad = self.kernel_size // 2

        hidden = int(conv1_weight.shape[0])
        embed = int(conv1_weight.shape[1])
        assert int(conv1_weight.shape[2]) == self.kernel_size, conv1_weight.shape
        assert tuple(conv1_bias.shape) == (hidden,), conv1_bias.shape
        assert tuple(ln1_weight.shape) == (hidden,), ln1_weight.shape
        assert tuple(ln1_bias.shape) == (hidden,), ln1_bias.shape
        assert int(conv2_weight.shape[0]) == hidden
        assert int(conv2_weight.shape[1]) == hidden
        assert int(conv2_weight.shape[2]) == self.kernel_size
        assert tuple(conv2_bias.shape) == (hidden,)
        assert tuple(ln2_weight.shape) == (hidden,)
        assert tuple(ln2_bias.shape) == (hidden,)
        assert tuple(proj_weight.shape) == (1, hidden)
        assert tuple(proj_bias.shape) == (1,)

        self.hidden = hidden
        self.embed = embed

        # ---- Conv1d weights (kept in row-major DRAM; ttnn.conv1d pre-processes
        #      them on the first call and we cache the prepared tensor) --------
        # ttnn.conv1d follows the PyTorch ``[out_C, in_C/groups, K]`` weight
        # convention, so no reshape needed here.
        self.conv1_weight = ttnn.from_torch(
            conv1_weight,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        # Bias must be [1, 1, 1, out_C] in row-major for ttnn.conv1d.
        self.conv1_bias = ttnn.from_torch(
            conv1_bias.reshape(1, 1, 1, hidden),
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.conv2_weight = ttnn.from_torch(
            conv2_weight,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.conv2_bias = ttnn.from_torch(
            conv2_bias.reshape(1, 1, 1, hidden),
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # ---- LayerNorm 1 / 2 (over the channel dim) -----------------------
        self.ln1_weight = ttnn.from_torch(
            ln1_weight.reshape(1, 1, 1, hidden),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.ln1_bias = ttnn.from_torch(
            ln1_bias.reshape(1, 1, 1, hidden),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.ln2_weight = ttnn.from_torch(
            ln2_weight.reshape(1, 1, 1, hidden),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.ln2_bias = ttnn.from_torch(
            ln2_bias.reshape(1, 1, 1, hidden),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        # ---- Final projection (hidden -> 1) -------------------------------
        # ``ttnn.linear`` expects the weight transposed relative to torch
        # ``nn.Linear.weight`` (i.e. ``[in, out]``).
        self.proj_weight = ttnn.from_torch(
            proj_weight.transpose(0, 1).contiguous(),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.proj_bias = ttnn.from_torch(
            proj_bias.reshape(1, -1),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        # Precision / compute config: HiFi4 + fp32 dest accum, matches the
        # other Conformer / Seamless block kernels in this model.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weight_dtype,
            shard_layout=None,  # auto-pick
            deallocate_activation=False,
        )
        self.conv_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _conv1d_block(
        self,
        x: ttnn.Tensor,
        weight_attr: str,
        bias_attr: str,
        in_channels: int,
        out_channels: int,
        batch: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        """Run a single Conv1d (kernel=K, symmetric pad) on a ``[B, T, C]`` input.

        Internally moves to ROW_MAJOR ``[B, 1, T, C]`` NHWC for ``ttnn.conv1d``,
        and returns the conv output reshaped back to ``[B, T, out_C]`` TILE_LAYOUT.
        """
        x_rm = x
        if x_rm.layout != ttnn.ROW_MAJOR_LAYOUT:
            x_rm = ttnn.to_layout(x_rm, ttnn.ROW_MAJOR_LAYOUT)
        # [B, T, C] -> [B, 1, T, C] (NHWC with H=1) via reshape.
        x_rm = ttnn.reshape(x_rm, (batch, 1, seq_len, in_channels))

        weight_tt = getattr(self, weight_attr)
        bias_tt = getattr(self, bias_attr)
        out, _out_len, [new_w, new_b] = ttnn.conv1d(
            input_tensor=x_rm,
            weight_tensor=weight_tt,
            device=self.device,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch,
            input_length=seq_len,
            kernel_size=self.kernel_size,
            stride=1,
            padding=[self.pad, self.pad],
            dilation=1,
            groups=1,
            bias_tensor=bias_tt,
            conv_config=self.conv_config,
            compute_config=self.conv_compute_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        # Cache the prepared weight + bias for subsequent calls.
        setattr(self, weight_attr, new_w)
        setattr(self, bias_attr, new_b)
        ttnn.deallocate(x_rm)

        # ttnn.conv1d returns ``[1, 1, B*T, out_C]`` flattened NHWC. Move to
        # TILE_LAYOUT and reshape back to ``[B, T, out_C]`` for the downstream
        # ReLU/LayerNorm/linear.
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        out = ttnn.reshape(out, (batch, seq_len, out_channels))
        return out

    def forward(
        self,
        x: ttnn.Tensor,
        padding_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run the variance predictor.

        Args:
            x: input ttnn tensor of shape ``[B, T, embed]`` in TILE_LAYOUT.
            padding_mask: optional ttnn tensor broadcastable to ``[B, T, 1]``
                where 0.0 marks padded positions to zero out before each
                Conv1d (1.0 means "keep"). The caller is responsible for
                converting the boolean HF mask to this float form.

        Returns:
            ttnn tensor of shape ``[B, T]`` (channel dim squeezed) in
            TILE_LAYOUT.
        """
        mc_dram = ttnn.DRAM_MEMORY_CONFIG
        batch = int(x.shape[0])
        seq_len = int(x.shape[1])
        hidden = self.hidden
        embed = self.embed

        # 1. Optional padding-mask zeroing before conv1.
        y = x
        if padding_mask is not None:
            y = ttnn.multiply(y, padding_mask, memory_config=mc_dram)

        # 2. Conv1 (embed -> hidden).
        y = self._conv1d_block(
            y,
            weight_attr="conv1_weight",
            bias_attr="conv1_bias",
            in_channels=embed,
            out_channels=hidden,
            batch=batch,
            seq_len=seq_len,
        )

        # 3. ReLU.
        y = ttnn.relu(y, memory_config=mc_dram)

        # 4. LayerNorm 1.
        y = ttnn.layer_norm(
            y,
            epsilon=self.eps,
            weight=self.ln1_weight,
            bias=self.ln1_bias,
            compute_kernel_config=self.compute_kernel_config,
        )

        # 5. Optional padding-mask zeroing before conv2.
        if padding_mask is not None:
            y = ttnn.multiply(y, padding_mask, memory_config=mc_dram)

        # 6. Conv2 (hidden -> hidden).
        y = self._conv1d_block(
            y,
            weight_attr="conv2_weight",
            bias_attr="conv2_bias",
            in_channels=hidden,
            out_channels=hidden,
            batch=batch,
            seq_len=seq_len,
        )

        # 7. ReLU.
        y = ttnn.relu(y, memory_config=mc_dram)

        # 8. LayerNorm 2.
        y = ttnn.layer_norm(
            y,
            epsilon=self.eps,
            weight=self.ln2_weight,
            bias=self.ln2_bias,
            compute_kernel_config=self.compute_kernel_config,
        )

        # 9. Final projection (hidden -> 1) and squeeze the trailing dim.
        out = ttnn.linear(
            y,
            self.proj_weight,
            bias=self.proj_bias,
            memory_config=mc_dram,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(y)

        # Squeeze [B, T, 1] -> [B, T].
        out = ttnn.reshape(out, (batch, seq_len))
        return out
