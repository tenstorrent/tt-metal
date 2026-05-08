# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import ttnn


def _compute_cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class _TtConv1d:
    def __init__(self, device, weight_rm: ttnn.Tensor, bias_rm: Optional[ttnn.Tensor]):
        self.device = device
        self.weight = weight_rm
        self.bias = bias_rm
        self.out_channels = int(weight_rm.shape[0])
        self.in_channels = int(weight_rm.shape[1])
        self.kernel_size = int(weight_rm.shape[2])
        self.padding = (self.kernel_size - 1) // 2
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.float32,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=False,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            config_tensors_in_dram=True,
            reshard_if_not_optimal=False,
            enable_kernel_stride_folding=False,
            force_split_reader=False,
            transpose_shards=False,
            enable_activation_reuse=False,
            full_inner_dim=False,
        )

    def __call__(self, x: ttnn.Tensor, batch_size: int, input_length: int) -> ttnn.Tensor:
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        x = ttnn.permute(x, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, [batch_size, 1, input_length, self.in_channels], memory_config=ttnn.L1_MEMORY_CONFIG)
        result, _, [self.weight, self.bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_length,
            input_width=1,
            kernel_size=(self.kernel_size, 1),
            stride=(1, 1),
            padding=(self.padding, 0),
            bias_tensor=self.bias,
            conv_config=self.conv_config,
            return_weights_and_bias=True,
            return_output_dim=True,
        )
        result = ttnn.reshape(
            result, [batch_size, input_length, self.out_channels], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        result = ttnn.permute(result, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.to_memory_config(result, ttnn.L1_MEMORY_CONFIG)


def _adain_instance_norm(
    x: ttnn.Tensor,
    inst_weight_1c1: ttnn.Tensor,
    inst_bias_1c1: ttnn.Tensor,
    style_lin: ttnn.Tensor,
    num_features: int,
    eps: float,
) -> ttnn.Tensor:
    mean = ttnn.mean(x, dim=2, keepdim=True)
    xc = ttnn.subtract(x, mean, memory_config=ttnn.L1_MEMORY_CONFIG)
    var = ttnn.mean(ttnn.multiply(xc, xc, memory_config=ttnn.L1_MEMORY_CONFIG), dim=2, keepdim=True)
    denom = ttnn.sqrt(ttnn.add(var, eps, memory_config=ttnn.L1_MEMORY_CONFIG))
    inv = ttnn.reciprocal(denom)
    x_norm = ttnn.multiply(xc, inv, memory_config=ttnn.L1_MEMORY_CONFIG)
    x_norm = ttnn.multiply(x_norm, inst_weight_1c1, memory_config=ttnn.L1_MEMORY_CONFIG)
    x_norm = ttnn.add(x_norm, inst_bias_1c1, memory_config=ttnn.L1_MEMORY_CONFIG)
    bsz = int(x.shape[0])
    gamma = style_lin[:, :num_features]
    beta = style_lin[:, num_features:]
    gamma = ttnn.reshape(gamma, [bsz, num_features, 1])
    beta = ttnn.reshape(beta, [bsz, num_features, 1])
    one = ttnn.ones_like(gamma)
    g1 = ttnn.add(one, gamma, memory_config=ttnn.L1_MEMORY_CONFIG)
    x_norm = ttnn.multiply(g1, x_norm, memory_config=ttnn.L1_MEMORY_CONFIG)
    return ttnn.add(x_norm, beta, memory_config=ttnn.L1_MEMORY_CONFIG)


class AdainResBlk1d:
    """TTNN port of `decoder.encode` AdainResBlk1d (Kokoro ISTFTNet)."""

    def __init__(self, device, parameters: dict, dim_in: int, dim_out: int, style_dim: int):
        self.device = device
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.style_dim = style_dim
        self.eps = float(parameters.get("eps", 1e-5))
        self.compute_cfg = _compute_cfg(device)
        self.conv1 = _TtConv1d(device, parameters["conv1"]["weight"], parameters["conv1"]["bias"])
        self.conv2 = _TtConv1d(device, parameters["conv2"]["weight"], parameters["conv2"]["bias"])
        self.conv1x1 = _TtConv1d(device, parameters["conv1x1"]["weight"], parameters["conv1x1"]["bias"])
        self.norm1_fc_w = parameters["norm1"]["fc_weight"]
        self.norm1_fc_b = parameters["norm1"]["fc_bias"]
        self.norm1_inst_w = parameters["norm1"]["inst_weight"]
        self.norm1_inst_b = parameters["norm1"]["inst_bias"]
        self.norm2_fc_w = parameters["norm2"]["fc_weight"]
        self.norm2_fc_b = parameters["norm2"]["fc_bias"]
        self.norm2_inst_w = parameters["norm2"]["inst_weight"]
        self.norm2_inst_b = parameters["norm2"]["inst_bias"]
        self.inv_sqrt2 = ttnn.from_torch(
            torch.tensor([[[1.0 / math.sqrt(2.0)]]], dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        s = ttnn.to_memory_config(s, ttnn.L1_MEMORY_CONFIG)
        if s.dtype != ttnn.float32:
            s = ttnn.typecast(s, ttnn.float32)
        batch_size = x.shape[0]
        input_length = x.shape[2]

        def residual(x_in: ttnn.Tensor) -> ttnn.Tensor:
            h1 = ttnn.linear(
                s,
                self.norm1_fc_w,
                bias=self.norm1_fc_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_cfg,
            )
            h2 = ttnn.linear(
                s,
                self.norm2_fc_w,
                bias=self.norm2_fc_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_cfg,
            )
            x_in = _adain_instance_norm(
                x_in,
                self.norm1_inst_w,
                self.norm1_inst_b,
                h1,
                self.dim_in,
                self.eps,
            )
            x_in = ttnn.leaky_relu(x_in, negative_slope=0.2)
            x_in = self.conv1(x_in, batch_size, input_length)
            x_in = _adain_instance_norm(
                x_in,
                self.norm2_inst_w,
                self.norm2_inst_b,
                h2,
                self.dim_out,
                self.eps,
            )
            x_in = ttnn.leaky_relu(x_in, negative_slope=0.2)
            return self.conv2(x_in, batch_size, input_length)

        def shortcut(x_in: ttnn.Tensor) -> ttnn.Tensor:
            return self.conv1x1(x_in, batch_size, input_length)

        out = ttnn.add(residual(x), shortcut(x), memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.multiply(out, self.inv_sqrt2, memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)


def preprocess_encode_parameters(torch_encode: nn.Module, device) -> dict:
    for name in ("conv1", "conv2", "conv1x1"):
        nn.utils.remove_weight_norm(getattr(torch_encode, name))

    DRAM = ttnn.DRAM_MEMORY_CONFIG

    def conv_rm(conv: nn.Conv1d):
        w = conv.weight.data.unsqueeze(-1)
        oc = w.shape[0]
        b = conv.bias.data if conv.bias is not None else None
        bias_t = torch.reshape(b, (1, 1, 1, oc)) if b is not None else None
        return {
            "weight": ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT),
            "bias": None
            if bias_t is None
            else ttnn.from_torch(bias_t, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT),
        }

    def lin_params(linear: nn.Linear):
        return {
            "fc_weight": ttnn.from_torch(
                linear.weight.data.T.contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM,
            ),
            "fc_bias": ttnn.from_torch(
                linear.bias.data,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM,
            ),
        }

    def inst_params(norm: nn.InstanceNorm1d):
        c = norm.num_features
        w = norm.weight.data.reshape(1, c, 1)
        b = norm.bias.data.reshape(1, c, 1)
        return {
            "inst_weight": ttnn.from_torch(
                w, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
            ),
            "inst_bias": ttnn.from_torch(
                b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
            ),
        }

    p: dict = {}
    p["conv1"] = conv_rm(torch_encode.conv1)
    p["conv2"] = conv_rm(torch_encode.conv2)
    p["conv1x1"] = conv_rm(torch_encode.conv1x1)
    p["norm1"] = {**lin_params(torch_encode.norm1.fc), **inst_params(torch_encode.norm1.norm)}
    p["norm2"] = {**lin_params(torch_encode.norm2.fc), **inst_params(torch_encode.norm2.norm)}
    p["eps"] = float(torch_encode.norm1.norm.eps)
    return p


def infer_encode_dims(torch_encode: nn.Module) -> tuple[int, int, int]:
    dim_in = int(torch_encode.conv1.weight.shape[1])
    dim_out = int(torch_encode.conv1.weight.shape[0])
    style_dim = int(torch_encode.norm1.fc.weight.shape[1])
    return dim_in, dim_out, style_dim


infer_adain_resblk1d_dims = infer_encode_dims
preprocess_adain_resblk1d_parameters = preprocess_encode_parameters
