# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-bounce transport for the BH-Galaxy pipeline.

One function: ttnn.to_torch on the source tensor, ttnn.from_torch onto the
destination submesh. No P2P, no sockets, no fabric — by design.
"""

from __future__ import annotations

import ttnn


def send_via_host(src_tensor, dst_submesh, memory_config=None):
    """Move src_tensor → host → dst_submesh. Preserves dtype + layout.

    `dst_submesh` is a 1x1 MeshDevice (a per-chip submesh).
    """
    host = ttnn.to_torch(src_tensor)
    return ttnn.from_torch(
        host,
        dtype=src_tensor.dtype,
        layout=src_tensor.layout,
        device=dst_submesh,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )
