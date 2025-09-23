# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.vovnet.tt.common import Conv

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TtEffectiveSEModule:
    def __init__(
        self,
        stride: int = 1,
        padding: int = 1,
        parameters=None,
        device=None,
        base_address=None,
        **_,
    ):
        self.device = device
        self.base_address = base_address

        self.fc = Conv(
            device=device,
            parameters=parameters,
            path=base_address,
            conv_params=[stride, stride, padding, padding],
            fused_op=False,
            effective_se=True,
        )

        self.activation = ttnn.hardsigmoid

    def forward(self, input: ttnn.Tensor) -> ttnn.Tensor:
        if use_signpost:
            signpost(header="effective_se_module")

        out = ttnn.mean(input, dim=[2, 3], keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
        out, _out_height, _out_width = self.fc(out)
        out = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)
        out = ttnn.reshape(out, (out.shape[0], _out_height, _out_width, out.shape[-1]))
        out = ttnn.permute(out, (0, 3, 1, 2))

        out = self.activation(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        out = ttnn.multiply(input, out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(input)
        return out
