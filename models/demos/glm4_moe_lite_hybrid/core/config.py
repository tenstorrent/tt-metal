# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model hyperparameters for GLM-4.7-Flash hybrid.

Re-exports the agentic config dataclass which provides from_hf_config() and
validate() methods, plus all MLA, MoE, RoPE, and MTP parameters.
"""

from models.demos.glm4_moe_lite.tt.config import Glm4MoeLiteHParams

__all__ = ["Glm4MoeLiteHParams"]
