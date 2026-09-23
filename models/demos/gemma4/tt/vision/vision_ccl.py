# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TP collectives for the Gemma-4 vision tower.

The vision tower is tensor-parallel across the whole mesh: attention is sharded by
head, the MLP by intermediate dim, and each row-parallel matmul leaves every device
holding a *partial sum* of the full hidden dim. ``tp_all_reduce`` sums those partials
so the block output is replicated again, which is the I/O contract every other vision
module (norms, residual adds, pooler, projector) assumes.
"""

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce


def tp_all_reduce(x, args):
    """Sum row-parallel partial sums across the TP devices.

    Args:
        x: ttnn.Tensor ``[..., width]`` partial sum (4D, TILE layout, DRAM).
        args: ``VisionModelArgs`` (carries ``tp`` / ``mesh_config`` / ``ccl_manager``).

    Returns:
        ttnn.Tensor of the same shape, replicated across the mesh.
    """
    if args.tp <= 1:
        return x

    # The all-reduce reduce-scatters along dim 3, so the width must split into
    # tile-aligned per-device chunks (e.g. dim=1152 over TP=8 does not). Zero-pad
    # up to a multiple of ``tile_size * tp`` — summing zeros is a no-op — and slice
    # the padding back off afterwards.
    width = int(x.shape[-1])
    pad = -width % (args.tile_size * args.tp)
    if pad:
        padded = ttnn.pad(x, [(0, 0), (0, 0), (0, 0), (0, pad)], value=0.0)
        ttnn.deallocate(x)
        x = padded

    out = ccl_allreduce(x, args.mesh_config, args.ccl_manager)

    if pad:
        sliced = out[:, :, :, :width]
        ttnn.deallocate(out)
        out = sliced
    return out
