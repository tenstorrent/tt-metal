# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT port of `SeamlessM4Tv2ConformerConvolutionModule` (Phase 1).

Reference forward (modeling_seamless_m4t_v2.py:347-375), operating on (B, T, C):
    h = layer_norm(h)                       # LN over C
    h = pointwise_conv1(h)                  # C -> 2C, k=1, no bias   (matmul)
    h = glu(h)                              # split 2C -> C: a * sigmoid(b)
    h = causal_pad(h, left=k-1)             # left pad in time
    h = depthwise_conv(h)                   # C -> C, k=31, groups=C, no bias
    h = depthwise_layer_norm(h)             # LN over C
    h = silu(h)
    h = pointwise_conv2(h)                  # C -> C, k=1, no bias    (matmul)

The two pointwise (k=1) convs are matmuls. GLU / LN / SiLU run on device. The
depthwise conv (k=31, groups=1024) is the critical-path risk: when
`config.conv_cpu_fallback` is set it runs on host (ttnn -> torch -> ttnn);
Phase 7 replaces it with an on-device `ttnn.conv1d`.

Note: the reference also zeroes padded positions before the depthwise conv using
`conv_attention_mask`. For batch=1 / unpadded single clips (Phase 1-6 scope) there
is no padding, so the mask is a no-op and is omitted here.
"""

from __future__ import annotations

import torch

import ttnn


class TtConformerConvModule:
    def __init__(self, weights: dict[str, torch.Tensor], config, device, dtype=ttnn.bfloat16):
        """
        weights: HF state-dict slice for one `conv_module.` (keys stripped of that prefix):
            layer_norm.{weight,bias}
            pointwise_conv1.weight        (2C, C, 1)
            depthwise_conv.weight         (C, 1, k)
            depthwise_layer_norm.{weight,bias}
            pointwise_conv2.weight        (C, C, 1)
        """
        self.config = config
        self.device = device
        self.dtype = dtype
        self.C = config.hidden_size
        self.kernel = config.conv_depthwise_kernel_size
        self.eps = config.layer_norm_eps
        self.cpu_fallback = config.conv_cpu_fallback

        def to_tt(t, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)

        # LayerNorm params (over C)
        self.ln_w = to_tt(weights["layer_norm.weight"])
        self.ln_b = to_tt(weights["layer_norm.bias"])
        self.dw_ln_w = to_tt(weights["depthwise_layer_norm.weight"])
        self.dw_ln_b = to_tt(weights["depthwise_layer_norm.bias"])

        # Pointwise convs as matmuls: Conv1d weight (out, in, 1) -> linear weight (in, out)
        pw1 = weights["pointwise_conv1.weight"].squeeze(-1)  # (2C, C)
        pw2 = weights["pointwise_conv2.weight"].squeeze(-1)  # (C, C)
        self.pw1_w = to_tt(pw1.t().contiguous())  # (C, 2C)
        self.pw2_w = to_tt(pw2.t().contiguous())  # (C, C)

        # Depthwise conv weight: (C, 1, k). Keep on host for the fallback path.
        self.dw_weight_torch = weights["depthwise_conv.weight"].float()
        # On-device path (Phase 7): prepare the ttnn weight + conv/compute configs.
        if not self.cpu_fallback:
            self.dw_weight_tt = ttnn.from_torch(weights["depthwise_conv.weight"].float(), dtype=ttnn.float32)
            self.conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16)
            # Cap activation block height so the conv circular buffers fit L1 at
            # long sequence lengths (depthwise, 1024 channels).
            self.conv_config.act_block_h_override = 32
            self.compute_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    def _depthwise_conv_host(self, x_tt):
        """Causal depthwise conv over time. ttnn (B,T,C) -> torch -> ttnn (B,T,C)."""
        x = ttnn.to_torch(x_tt).float()  # (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        x = torch.nn.functional.pad(x, (self.kernel - 1, 0))  # causal left pad
        x = torch.nn.functional.conv1d(x, self.dw_weight_torch, groups=self.C)  # (B, C, T)
        x = x.transpose(1, 2).contiguous()  # (B, T, C)
        return ttnn.from_torch(x, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _depthwise_conv_device(self, x_tt):
        """On-device causal depthwise conv1d (k=31, groups=C) via ttnn.conv1d.

        Input/output (1, T, C). Causal: left-pad the time dim by k-1 (row-major),
        then conv with padding=0 so out_length == T.
        """
        T = x_tt.shape[1]
        x = ttnn.to_layout(x_tt, ttnn.ROW_MAJOR_LAYOUT)
        # pad time dim (dim 1) on the left by kernel-1; channels/ batch unchanged
        x = ttnn.pad(x, [(0, 0), (self.kernel - 1, 0), (0, 0)], value=0.0)
        out, _out_len = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.dw_weight_tt,
            in_channels=self.C,
            out_channels=self.C,
            device=self.device,
            kernel_size=self.kernel,
            stride=1,
            padding=0,
            groups=self.C,
            batch_size=1,
            input_length=T + self.kernel - 1,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            dtype=self.dtype,
            return_output_dim=True,
        )
        # conv1d returns a height-sharded tensor; move to interleaved so the
        # downstream (non-sharded) layer_norm gets a standard core grid.
        out = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)
        out = ttnn.reshape(out, [1, T, self.C])
        return ttnn.to_layout(out, ttnn.TILE_LAYOUT)

    def __call__(self, hidden):
        """hidden: ttnn tensor (B, T, C) in TILE layout. Returns (B, T, C)."""
        # 1. LayerNorm over C
        h = ttnn.layer_norm(hidden, weight=self.ln_w, bias=self.ln_b, epsilon=self.eps)

        # 2. pointwise_conv1 (C -> 2C) as matmul
        h = ttnn.linear(h, self.pw1_w)  # (B, T, 2C)

        # 3. GLU along channel (last dim): a * sigmoid(b)
        a = h[..., : self.C]
        b = h[..., self.C :]
        h = ttnn.multiply(a, ttnn.sigmoid(b))  # (B, T, C)

        # 4. causal depthwise conv — host fallback or on-device (config.conv_cpu_fallback)
        if self.cpu_fallback:
            h = self._depthwise_conv_host(h)
        else:
            h = self._depthwise_conv_device(h)

        # 5. depthwise LayerNorm over C
        h = ttnn.layer_norm(h, weight=self.dw_ln_w, bias=self.dw_ln_b, epsilon=self.eps)

        # 6. SiLU
        h = ttnn.silu(h)

        # 7. pointwise_conv2 (C -> C) as matmul
        h = ttnn.linear(h, self.pw2_w)  # (B, T, C)
        return h
