# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-side weight helpers for Qwen2.5-72B-Instruct TTTv2.

Construction lives in ``models.common.models.qwen25_72b.hf_adaptor``;
low-level permutes and HF tensor layout live in ``weight_utils``.
"""

from models.common.models.qwen25_72b import weight_utils

__all__ = ["weight_utils"]
