# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CCL manager (the gpt_oss_d_p one is model-agnostic: semaphores, ring-gather buffers, CCL column). The fabric-link
default shared by the ring SDPA and the MoE dispatch / combine / reduce is ``MiMoRuntimeOptions.num_links``."""

from models.demos.gpt_oss_d_p.tt.ccl import CCLManager

__all__ = ["CCLManager"]
