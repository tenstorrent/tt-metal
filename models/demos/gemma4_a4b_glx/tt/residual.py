# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtResidualAdd:
    """Residual add on device: out = a + b. Both inputs share one layout (replicated within each row)."""

    def __init__(self, mesh):
        self.mesh = mesh

    def __call__(self, a, b):
        return ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class TtScaledResidualAdd:
    """Scaled residual on device: out = (a + b) * scalar. The scalar (layer_scalar) is read on the host at load; the
    sum stays fp32 until the multiply so only the output is rounded to bf16."""

    def __init__(self, mesh, scalar):
        self.mesh = mesh
        self.scalar = float(scalar)

    def __call__(self, a, b):
        s = ttnn.add(a, b, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.multiply(s, self.scalar, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s)
        return out
