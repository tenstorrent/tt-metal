# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-side weight helpers for Qwen2.5-72B-Instruct TTTv2.

Construction lives in
:func:`models.common.models.qwen25_72b.model.Qwen25_72B.from_pretrained`;
low-level permutes and HF tensor layout live in ``weight_utils``.
"""

from models.common.models.qwen25_72b import weight_utils

__all__ = ["weight_utils"]
