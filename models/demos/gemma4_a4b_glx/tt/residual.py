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
