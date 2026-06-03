# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-side weight helpers for Phi-4 TTTv2.

Construction lives in :func:`models.common.models.phi4.model.Phi4Transformer.from_pretrained`;
low-level fused-projection splits and HF tensor layout live in ``weight_utils``.
"""

from models.common.models.phi4 import weight_utils

__all__ = ["weight_utils"]
