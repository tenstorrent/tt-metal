# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-side weight helpers for Qwen2.5-Coder-32B-Instruct TTTv2.

Construction lives in
:func:`models.common.models.qwen25_coder_32b.model.Qwen25Coder32B.from_pretrained`;
low-level permutes and HF tensor layout live in ``weight_utils``.
"""

from models.common.models.qwen25_coder_32b import weight_utils

__all__ = ["weight_utils"]
