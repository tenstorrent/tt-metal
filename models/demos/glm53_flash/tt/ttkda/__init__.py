# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's copy of the ttKDA prefill driver (``models/demos/deepseek_v3_d_p/tt/kda`` at tt-metal f8fabeb9b9c,
vendored 2026-10-29 at the user's direction so its grouping geometry, L1 budgets and chronology contract are tuned
here, inside the model folder; the ttnn ``experimental.kda`` ops stay shared). The HF-side reference helpers
(``deepseek_v3_d_p/reference/kda``) are still imported from the sibling.

Local tweaks (keep this list current):
- config.py KDA_PREPARATION_L1_BYTES_PER_CORE 320 -> 256 KiB: the 1024-1088-local-row sequence-parallel geometry kept
  its chunk terms in L1 and clashed with the exclusive scan's circular buffers (2026-10-29).
"""
"""Public Kimi Delta Attention layer API."""

from models.demos.glm53_flash.tt.ttkda.kda import KdaState, ttKDA

__all__ = ["ttKDA", "KdaState"]
