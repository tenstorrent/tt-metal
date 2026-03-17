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
        self._compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            math_approx_mode=False,
        )
        self._compute_config_fp32 = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            math_approx_mode=False,
        )

    def __call__(self, input_tensor):
        output = ttnn.linear(
            input_tensor,
            self.parameters["fc1"]["weight"],
            bias=self.parameters["fc1"]["bias"],
            activation="gelu",
            compute_kernel_config=self._compute_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.linear(
            output,
            self.parameters["fc2"]["weight"],
            bias=self.parameters["fc2"]["bias"],
            compute_kernel_config=self._compute_config_fp32,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
