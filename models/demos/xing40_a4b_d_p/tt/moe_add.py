# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""moe_add on device: mlp_out = experts_out + shared_out.

Both inputs are column-split [1, 1, S/4, H/2] fp32 (row split over axis 0, hidden split over axis 1), as TtExperts and
the shared TtDenseMLP return them; the sum keeps that layout. Local elementwise add, no CCL, no weights.
From models/demos/glm53_flash_d_p/tt/moe_add.py (TtMoeAdd), with an fp32 output (the residual mix consumes fp32).
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMoeAdd(LightweightModule):
    def __init__(self, out_dtype=ttnn.float32):
        self.out_dtype = out_dtype

    def __call__(self, experts: ttnn.Tensor, shared: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.add(experts, shared, dtype=self.out_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def build_moe_add(cfg=None) -> TtMoeAdd:
    return TtMoeAdd()
