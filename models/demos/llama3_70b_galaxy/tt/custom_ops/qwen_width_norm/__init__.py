# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Width-split local RMSNorm of the Qwen context-parallel residual as one generic_op program per device."""

from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_width_norm.op import qwen_width_norm

__all__ = ["qwen_width_norm"]
