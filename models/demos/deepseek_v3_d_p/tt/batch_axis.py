# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Batch-axis prefill: one user per coordinate along the mesh axis TP would otherwise use.

Attention runs per user (``ttMLA(batch_axis=...)``): every chip holds its user's rows at full hidden
width, [1, 1, S/sp, H]. The modules that stay tensor-parallel -- the embedding, the MoE and the dense
FFN, whose weights are split across that axis -- instead see all users' rows with a hidden slice,
[1, 1, U*S/sp, H/U]: today's TP activation layout with U times the tokens. These two all-to-alls move
between the two layouts; nothing else changes in those modules.

Row order after interleave_users is source-coordinate order, [u0 | u1 | ... | u_{U-1}], each block
this chip's SP rows of that user. deinterleave_users is the exact inverse.
"""

import ttnn


def interleave_users(x: ttnn.Tensor, batch_axis: int, num_links: int) -> ttnn.Tensor:
    """[1, 1, S/sp, H] (own user, full hidden) -> [1, 1, U*S/sp, H/U] (all users, hidden slice).

    Each device splits its hidden into U slices and sends slice h to coordinate h, which concatenates
    the U row blocks it receives in source order."""
    return ttnn.experimental.all_to_all_async_generic(
        x,
        in_dim=2,  # grows: every user's rows
        out_dim=3,  # splits: hidden slices
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=batch_axis,
    )


def deinterleave_users(x: ttnn.Tensor, batch_axis: int, num_links: int) -> ttnn.Tensor:
    """Inverse of interleave_users: [1, 1, U*S/sp, H/U] -> [1, 1, S/sp, H].

    Each device splits its rows into the U user blocks and sends block u to coordinate u, which
    concatenates the U hidden slices it receives back into the full hidden dim."""
    return ttnn.experimental.all_to_all_async_generic(
        x,
        in_dim=3,  # grows: hidden slices back to full width
        out_dim=2,  # splits: per-user row blocks
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=batch_axis,
    )
