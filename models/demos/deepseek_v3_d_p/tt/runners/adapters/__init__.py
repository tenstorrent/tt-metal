# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""DeepSeek-V3-family prefill adapters.

Concrete ``PrefillModelAdapter`` implementations for this model package. The
adapter CONTRACT and the model registry (``get_adapter`` / ``ADAPTER_PATHS``) live
in the common package at ``models.demos.common.prefill.adapter``; these classes are
registered there by dotted path and resolved lazily, so the common engine never
imports this package at module load.
"""

from models.demos.deepseek_v3_d_p.tt.runners.adapters.deepseek_v3 import DeepSeekV3Adapter
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k2_6 import KimiK26Adapter
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter
from models.demos.deepseek_v3_d_p.tt.runners.adapters.sparse_mla import (
    DeepSeekV32Adapter,
    GLM51Adapter,
    GLM52Adapter,
    SparseMLAPrefillAdapter,
)

__all__ = [
    "MLAPrefillAdapter",
    "DeepSeekV3Adapter",
    "KimiK26Adapter",
    "SparseMLAPrefillAdapter",
    "DeepSeekV32Adapter",
    "GLM51Adapter",
    "GLM52Adapter",
]
