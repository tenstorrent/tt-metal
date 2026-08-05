# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Supported pure-Torch reference API for Kimi Delta Attention."""

from models.experimental.kimi_delta_attention.reference.layer import KDAReferenceState, kda_forward_reference

__all__ = [
    "KDAReferenceState",
    "kda_forward_reference",
]
