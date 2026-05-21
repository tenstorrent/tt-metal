# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN MLP for Swin-L blocks.
Adapted from models/experimental/swin_s/tt/tt_mlp.py.
Initial version: generic (no hardcoded sharding configs).
"""

import ttnn


class TtSwinMLP:
    """Two-layer MLP with GELU activation."""

    def __init__(self, device, parameters, dim, mlp_ratio=4.0):
        self.device = device
        self.parameters = parameters
        self.hidden_dim = int(dim * mlp_ratio)
        self.dim = dim

    def __call__(self, input_tensor):
        # fc1 + GELU. Output dtype=bfloat8_b halves the (B*nW, S, 4*C) write/read
        # bandwidth for fc2 — same trick ViT uses on every linear in the encoder.
        output = ttnn.linear(
            input_tensor,
            self.parameters["fc1"]["weight"],
            bias=self.parameters["fc1"]["bias"],
            activation="gelu",
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        # fc2
        return ttnn.linear(
            output,
            self.parameters["fc2"]["weight"],
            bias=self.parameters["fc2"]["bias"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
