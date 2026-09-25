# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CCL manager: the gpt_oss_d_p one is model-agnostic (semaphores, ring-gather buffers, CCL column)."""

from models.demos.gpt_oss_d_p.tt.ccl import CCLManager

__all__ = ["CCLManager"]
