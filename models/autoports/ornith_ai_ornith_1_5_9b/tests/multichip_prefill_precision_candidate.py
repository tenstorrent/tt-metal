# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase-specific BFP8 activation/communication candidate on current TP4 code."""

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_pre_fusion_reference import MultichipDecoder


class PrefillPrecisionCandidate(MultichipDecoder):
    def norm(self, x, name):
        result = super().norm(x, name)
        if x.shape[2] > 1 and name in ("input_layernorm", "post_attention_layernorm"):
            result = ttnn.typecast(result, ttnn.bfloat8_b)
        return result

    def linear(self, x, name):
        if x.shape[2] == 1:
            return super().linear(x, name)
        return ttnn.experimental.minimal_matmul(
            ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), self.weights[name + ".weight"],
            config=self.prefill_program, compute_kernel_config=self.matmul_compute(name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
        )

    def reduce(self, x):
        if x.shape[2] == 1:
            return super().reduce(x)
        return ttnn.typecast(super().reduce(ttnn.typecast(x, ttnn.bfloat8_b)), ttnn.bfloat16)

    def finish(self, x, mixed):
        if x.shape[2] == 1:
            return super().finish(x, mixed)
        residual = ttnn.add(x, mixed)
        h = self.norm(residual, "post_attention_layernorm")
        h = ttnn.experimental.minimal_matmul(
            h, self.weights["packed_mlp.weight"], fuse_swiglu=True,
            config=self.prefill_program, compute_kernel_config=self.mlp_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
        )
        return ttnn.add(residual, self.reduce(self.linear(h, "mlp.down_proj")))


if __name__ == "__main__":
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_probe
    multichip_probe.MultichipDecoder = PrefillPrecisionCandidate
    multichip_probe.main()
