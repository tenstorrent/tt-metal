# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tuned separate BF16 router with LoFi, direct L1 input/output and BF4 shared MLP."""

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.optimized_decoder import OptimizedDecoder as Base


class OptimizedDecoder(Base):
    def _linear(self, x, name, activation=None):
        if name == "mlp.gate" and self.options.get("router_interleaved", True):
            return ttnn.linear(
                x,
                self.weights[name + ".weight"],
                compute_kernel_config=self.matmul_compute["mlp"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG if x.shape[-2] == 32 else ttnn.DRAM_MEMORY_CONFIG,
            )
        return super()._linear(x, name, activation=activation)
