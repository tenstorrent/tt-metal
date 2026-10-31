# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""moe_add on device: mlp_out = experts_out + shared_out (the reference's `a + b`, fp32).

Both inputs are [1, 1, S/SP, H/TP] fp32 (rows split over mesh axis 0, hidden columns over axis 1), as TtExperts and
TtSharedExpert return them; the sum keeps that layout. Local elementwise add (fp32 operands take the SFPU path, so the
sum equals torch's fp32 sum, K-30), no CCL, no weights, nothing that depends on the box. In the assembled model the add
happens on the fp32 partials before the one MoE exchange (components.yaml notes); this module is the standalone step.
From models/demos/xing40_a4b_d_p/tt/moe_add.py:TtMoeAdd.
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule

from .cores import sub_cores


class TtMoeAdd(LightweightModule):
    def __init__(self, out_dtype=ttnn.float32):
        self.out_dtype = out_dtype

    def __call__(self, experts: ttnn.Tensor, shared: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.add(
            experts,
            shared,
            dtype=self.out_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            sub_core_grids=sub_cores(experts),
        )


def build_moe_add(cfg=None) -> TtMoeAdd:
    return TtMoeAdd()
