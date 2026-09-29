# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""moe_add on device: mlp_out = experts_out + shared_out, replicated [1, 1, S, H] bf16, no CCL, no weights.

From models/demos/mimo_v2_6_d_p/tt/residual.py (TtResidualAdd).
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMoeAdd(LightweightModule):
    def __init__(self, out_dtype=ttnn.bfloat16):
        self.out_dtype = out_dtype

    def __call__(self, experts: ttnn.Tensor, shared: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.add(experts, shared, dtype=self.out_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def build_moe_add(cfg) -> TtMoeAdd:
    return TtMoeAdd()
