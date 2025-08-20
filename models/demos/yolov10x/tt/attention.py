# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.demos.yolov10x.tt.common import Conv


class TtnnAttention:
    def __init__(self, dim, num_heads=8, attn_ratio=0.5, device=None, parameters=None, conv_pt=None):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2

        self.qkv = Conv(device, parameters.qkv, self.conv_pt.qkv, enable_identity=True, activation_dtype=ttnn.bfloat16)
        self.proj = Conv(
            device, parameters.proj, self.conv_pt.proj, enable_identity=True, activation_dtype=ttnn.bfloat16
        )

        self.pe = Conv(
            device,
            parameters.pe,
            self.conv_pt.pe,
            enable_identity=True,
            use_1d_systolic_array=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )

    def __call__(self, input_tensor):
        B, C, H, W = (
            1,
            input_tensor.shape[-1],
            int(math.sqrt(input_tensor.shape[2])),
            int(math.sqrt(input_tensor.shape[2])),
        )
        N = H * W

        qkv = self.qkv(input_tensor)
        if qkv.is_sharded():
            qkv = ttnn.sharded_to_interleaved(qkv, ttnn.L1_MEMORY_CONFIG)
        qkv = ttnn.permute(qkv, (0, 3, 1, 2))

        qkv = ttnn.reshape(qkv, (B, 5, 128, qkv.shape[3]))

        q, k, v = qkv[:, :, :32, :], qkv[:, :, 32:64, :], qkv[:, :, 64:, :]

        q = ttnn.permute(q, (0, 1, 3, 2))

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        attn = ttnn.matmul(q, k, compute_kernel_config=compute_kernel_config, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn = ttnn.multiply(attn, self.scale)

        attn = ttnn.softmax(attn, dim=-1)

        attn = ttnn.permute(attn, (0, 1, 3, 2))
        attn = ttnn.matmul(v, attn, compute_kernel_config=compute_kernel_config, memory_config=ttnn.L1_MEMORY_CONFIG)

        attn = ttnn.reshape(attn, (1, 320, 1, 400))
        attn = ttnn.permute(attn, (0, 2, 3, 1))

        v = ttnn.reshape(v, (1, 320, 1, 400))
        v = ttnn.permute(v, (0, 2, 3, 1))

        v = self.pe(v)

        x = ttnn.add(attn, v, memory_config=ttnn.L1_MEMORY_CONFIG)

        output = self.proj(x)
        return output
