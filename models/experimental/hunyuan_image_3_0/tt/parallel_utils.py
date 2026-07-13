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


def sp_shard(ccl, t: ttnn.Tensor, *, dim: int, mesh_axis: int, n: int, out_memory_config=None) -> ttnn.Tensor:
    """Replicated -> sequence-sharded along `dim` over `mesh_axis` (n = axis size).

    out_memory_config: placement for the rescaled output. Defaults to DRAM (the SDPA
    mask MUST stay DRAM). Pass L1 only for the `hidden` reshard, whose consumer is the
    input_layernorm — landing it L1-resident there avoids a separate DRAM->L1 copy.
    """
    if n == 1:
        return t
    scattered = ccl.reduce_scatter(t, dim=dim, mesh_axis=mesh_axis)  # [.., D/n, ..], = n * chunk
    out = ttnn.multiply(
        scattered, 1.0 / n, memory_config=out_memory_config or ttnn.DRAM_MEMORY_CONFIG
    )  # undo the n-fold sum from replicated input
    ttnn.deallocate(scattered)
    return out


def sp_gather(ccl, t, *, dim: int, mesh_axis: int, n: int, out_memory_config=None):
    """Sequence-sharded -> replicated (full) along `dim` over `mesh_axis`.

    out_memory_config: placement for the gathered output (default follows the
    all_gather default, i.e. the input's placement). Pass L1 to land the result
    L1-resident for its consumer.
    """
    if n == 1:
        return t
    gathered = ccl.all_gather(t, dim=dim, mesh_axis=mesh_axis, use_hyperparams=False)
    if out_memory_config is not None:
        gathered = ttnn.to_memory_config(gathered, out_memory_config)
    return gathered
