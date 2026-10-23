# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Supported pure-Torch reference API for Kimi Delta Attention."""

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState, kda_forward_reference

__all__ = ["KDAConfig", "KDAReferenceState", "kda_forward_reference"]
