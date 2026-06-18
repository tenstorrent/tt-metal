# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Sequence-parallel (SP) reshard helpers for the HunyuanImage-3.0 backbone.
#
# SP splits the (long, ~22.8k) token sequence across a mesh axis so per-token work
# (norms, projections, MoE, residuals) and the per-query attention output are
# divided across the row devices. Hunyuan attention uses a dense mixed mask, so
# the proven DiT ring-SDPA does NOT apply — instead we keep Q sequence-sharded and
# all-gather K/V (see attention.py). These helpers bridge the replicated <-> shard
# representations:
#
#   sp_shard:  replicated [.., D, ..] -> per-device [.., D/n, ..]  (scatter on mesh_axis)
#   sp_gather: per-device [.., D/n, ..] -> replicated [.., D, ..]   (all-gather on mesh_axis)
#
# sp_shard is built from reduce_scatter: feeding a tensor that is IDENTICAL across
# the axis makes reduce_scatter return n * (this device's chunk), so scaling by 1/n
# recovers the plain scatter. (No native "scatter replicated" collective is exposed
# by CCLManager; this is the standard construction.)
#
# NOTE (on-box bring-up): the sharded extent D/n must be tile-aligned (multiple of
# 32). The image-gen sequence is padded to a tile multiple upstream; confirm S/n is
# also tile-aligned on the box, else pad the sequence to n*TILE before sharding.

import ttnn

TILE = 32


def sp_shard(ccl, t: ttnn.Tensor, *, dim: int, mesh_axis: int, n: int) -> ttnn.Tensor:
    """Replicated -> sequence-sharded along `dim` over `mesh_axis` (n = axis size)."""
    if n == 1:
        return t
    scattered = ccl.reduce_scatter(t, dim=dim, mesh_axis=mesh_axis)  # [.., D/n, ..], = n * chunk
    out = ttnn.multiply(scattered, 1.0 / n)  # undo the n-fold sum from replicated input
    ttnn.deallocate(scattered)
    return out


def sp_gather(ccl, t: ttnn.Tensor, *, dim: int, mesh_axis: int, n: int) -> ttnn.Tensor:
    """Sequence-sharded -> replicated (full) along `dim` over `mesh_axis`."""
    if n == 1:
        return t
    return ccl.all_gather(t, dim=dim, mesh_axis=mesh_axis, use_hyperparams=False)
