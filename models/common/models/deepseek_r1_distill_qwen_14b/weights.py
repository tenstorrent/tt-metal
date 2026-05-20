# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-side weight helpers for DeepSeek-R1-Distill-Qwen-14B TTTv2.

Construction lives in
:func:`models.common.models.deepseek_r1_distill_qwen_14b.model.DeepSeekR1Qwen14B.from_pretrained`;
low-level permutes and HF tensor layout live in ``weight_utils``.
"""

from models.common.models.deepseek_r1_distill_qwen_14b import weight_utils

__all__ = ["weight_utils"]
