# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Centralized runtime configuration for GLM-4.7-Flash hybrid.

All GLM4_MOE_LITE_* environment variables are parsed once into a frozen
dataclass at model init time. Imported directly from the agentic implementation
to maintain a single source of truth for runtime knobs.
"""

from models.demos.glm4_moe_lite.tt.runtime_config import (
    Glm4RuntimeConfig,
    mesh_shape,
    parse_math_fidelity,
    tp_cluster_axis,
)

__all__ = [
    "Glm4RuntimeConfig",
    "mesh_shape",
    "tp_cluster_axis",
    "parse_math_fidelity",
]
