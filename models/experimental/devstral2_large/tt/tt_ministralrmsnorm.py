# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Tenstorrent RMSNorm for Devstral-2 (Ministral3 text stack).

Implementation is shared with ``models.experimental.devstarl2_small``; this module re-exports it for
``models.experimental.devstral2_large`` bring-up (same ``Ministral3RMSNorm`` mapping).
"""

from __future__ import annotations

from models.experimental.devstarl2_small.tt.tt_ministralrmsnorm import TtMinistralRMSNorm

TtDevstral2LargeRMSNorm = TtMinistralRMSNorm

__all__ = ["TtDevstral2LargeRMSNorm", "TtMinistralRMSNorm"]
