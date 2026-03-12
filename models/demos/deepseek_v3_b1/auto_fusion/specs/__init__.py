# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Micro-op specifications for auto-fusion.

Each spec captures the full hardware contract of an atomic operation:
which headers to include, per-RISC behavior, CB ports, and C++ Op template.
"""

from models.demos.deepseek_v3_b1.auto_fusion.specs.rmsnorm import RMSNORM
from models.demos.deepseek_v3_b1.auto_fusion.specs.matmul import MATMUL
from models.demos.deepseek_v3_b1.auto_fusion.specs.mcast import MCAST
from models.demos.deepseek_v3_b1.auto_fusion.specs.gather import GATHER
from models.demos.deepseek_v3_b1.auto_fusion.specs.local_reduce import LOCAL_REDUCE
from models.demos.deepseek_v3_b1.auto_fusion.specs.eltwise_mul import ELTWISE_MUL
from models.demos.deepseek_v3_b1.auto_fusion.specs.residual_add import RESIDUAL_ADD

__all__ = ["RMSNORM", "MATMUL", "MCAST", "GATHER", "LOCAL_REDUCE", "ELTWISE_MUL", "RESIDUAL_ADD"]
