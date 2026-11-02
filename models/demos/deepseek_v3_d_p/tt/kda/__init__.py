# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Public Kimi Delta Attention layer API."""

from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA

__all__ = ["ttKDA", "KdaState"]
