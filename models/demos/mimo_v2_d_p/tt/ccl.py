# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CCL manager (the gpt_oss_d_p one is model-agnostic: semaphores, ring-gather buffers, CCL column) and the
fabric-link default shared by the ring SDPA and the MoE dispatch / combine / reduce."""

import os

from models.demos.gpt_oss_d_p.tt.ccl import CCLManager

__all__ = ["CCLManager", "default_num_links"]


def default_num_links() -> int:
    """Fabric links per CCL (``MIMO_NUM_LINKS``). 3 measured best on the BH QuietBox 2x2: MoE dispatch
    2.59 / 1.31 / 0.88 ms for 1 / 2 / 3 links at 640 tokens/chip; 4 does not fit the dispatch core layout."""
    return int(os.environ.get("MIMO_NUM_LINKS", "3"))
