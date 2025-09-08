# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn.torch_tracer
import ttnn

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

from models.experimental.vovnet.tt.common import get_nested


class TtClassifierHead:
    def __init__(self, device=None, parameters=None, base_address=None, lay_idx=1000):
        self.device = device
        self.lay_idx = lay_idx
        self.base_address = base_address
        path_parts = self.base_address.split(".")
        self.weight = get_nested(parameters, path_parts + ["fc", "weight"])
        self.bias = get_nested(parameters, path_parts + ["fc", "bias"])

    def forward(self, x):
        if use_signpost:
            signpost(header="classifier_head")

        x = ttnn.global_avg_pool2d(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.permute(x, (0, 3, 1, 2))

        x = ttnn.reshape(x, [x.shape[0], 1, 1, x.shape[1] * x.shape[2] * x.shape[3]])
        x = ttnn.linear(
            x,
            self.weight,
            bias=self.bias,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=x.shape[0], x=8),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                packer_l1_acc=False,
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
            ),
        )

        x = ttnn.reshape(x, [x.shape[0], 1000])
        return x
