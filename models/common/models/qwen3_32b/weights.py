# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-side weight helpers for Qwen3-32B TTTv2.

Construction lives in
:func:`models.common.models.qwen3_32b.model.Qwen3_32B.from_pretrained`;
low-level permutes and HF tensor layout live in ``weight_utils``.
"""

from models.common.models.qwen3_32b import weight_utils

__all__ = ["weight_utils"]
