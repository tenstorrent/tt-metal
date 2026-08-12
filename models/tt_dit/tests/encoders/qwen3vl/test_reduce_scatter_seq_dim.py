# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Head-to-head: reduce-scatter on the SEQUENCE dim (dim=2) vs the HIDDEN dim
# (dim=3), across the TP axis, on the same (1, 1, S, 1152) tensor.
#
# This probes the primitive the Qwen3-VL vision tower's "Megatron sequence
# parallelism" reroute (the pad-dance fix) rests on. Today `RowParallelLinear`
# reduce-scatters on dim=3 (hidden): at hidden 1152 / TP 8 = 144 = 4.5 tiles the
# scatter boundary lands mid-tile, so the op falls back to a
# slice/untilize/pad/concat/permute composite -- the pad dance, ~46% of the
# block. Megatron SP moves that reduce-scatter to dim=2 (sequence), where the
# shard boundary is a whole number of tile rows.
#
# Both dims are exercised on an identical tensor so the two are directly
# comparable:
#   * scatter_dim=2  -> (1, 1, S/8, 1152), a tile-aligned row shard    (fast path)
#   * scatter_dim=3  -> (1, 1, S, 144),    a 4.5-tile column shard     (pad dance)
# Reduce-scatter is numerically correct either way -- the pad dance is a *perf*
# fallback, not a wrong answer -- so both assert PCC. The point of the pair is
# what `python -m tracy` shows: dim=2 lowers to a single ReduceScatter, dim=3
# additionally emits the slice/untilize/pad/concat/permute chain.
#
# ALIGNMENT CONTRACT (dim=2). The reduce-scatter splits the local (already
# SP-sharded) sequence `S` across TP(8); the `S/8`-row shard must be tile-aligned,
# so `S % (TP * 32) == 0` (i.e. `% 256`). Compounded with the SP-axis sharding
# this is the global `% (tp * sp * 32) == 1024` precondition. Sequences that miss
# it (e.g. two_refs' local 1192 rows) must be tail-padded up to the next multiple,
# as `transformer_minimax_h3.py:434-457` pads the packed sequence to
# `sp_factor * TILE` before `mesh_partition`. This test covers aligned shards only.
# =============================================================================

import pytest
import torch

import ttnn

from ....parallel.manager import CCLManager
from ....utils.check import assert_quality

HIDDEN_SIZE = 1152  # the vision tower's residual-stream width; 1152 / TP(8) = 144 = 4.5 tiles
_TILE = 32
_L1_SMALL = 32768
_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": _L1_SMALL}

# TP is mesh axis 0 (factor 8), matching the vision block's tp8_sp4 placement. We reduce-scatter over
# it. SP (axis 1, factor 4) is a bystander: the partials are replicated across it, so a single SP
# replica reconstructs the result.
_TP_AXIS = 0


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param((8, 4), _FABRIC, id="8x4")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_links", [2])
# Local (post-SP) sequence lengths that keep the S/8 TP shard tile-aligned (all `% 256 == 0`). 4096 is
# the local shard of a 16384-patch `ref_1to1` input at SP=4; 1024 is a small fast case.
@pytest.mark.parametrize("seq_len", [1024, 4096], ids=["S1024", "S4096"])
# dim=2 is the sequence reroute (fast path); dim=3 is today's hidden scatter (the pad dance) on the
# SAME tensor, for a direct comparison under tracy.
@pytest.mark.parametrize("scatter_dim", [2, 3], ids=["seq_dim2", "hidden_dim3"])
def test_reduce_scatter_axis(mesh_device, device_params, num_links, seq_len, scatter_dim):
    """`reduce_scatter` over TP on dim 2 vs 3: gather-back must equal the sum of the per-device partials.

    Both are correct; the difference is device-side -- dim=3's 144-wide shard triggers the pad-dance
    composite, dim=2's tile-aligned row shard does not. Run under `python -m tracy` to see it.
    """
    torch.manual_seed(0)

    tp = tuple(mesh_device.shape)[_TP_AXIS]
    if scatter_dim == 2:
        assert seq_len % (tp * _TILE) == 0, f"S={seq_len} would give a non-tile-aligned TP row shard"
    # scatter_dim == 3 is deliberately unaligned: 1152 / 8 = 144 = 4.5 tiles. That is the pad dance.

    ccl = CCLManager(mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)

    # Each TP device holds a distinct partial sum of the full (1, 1, S, H) activation. Stack the `tp`
    # partials on dim 0 and shard that dim across the TP axis (replicate across SP) so device d on the
    # TP axis gets `partials[d]`. Small values keep the bf16 reduction well-conditioned.
    partials = torch.randn(tp, 1, seq_len, HIDDEN_SIZE) * 0.1
    x = ttnn.from_torch(
        partials,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[0, None], mesh_shape=tuple(mesh_device.shape)),
    )

    # The operation under test: sum across the 8 TP devices, scatter the result on `scatter_dim`.
    out = ccl.reduce_scatter(x, dim=scatter_dim, mesh_axis=_TP_AXIS, use_persistent_buffer=True)
    ttnn.synchronize_device(mesh_device)

    # A wrong scatter axis shows up in the shape first.
    expected_shape = [1, 1, seq_len, HIDDEN_SIZE]
    expected_shape[scatter_dim] //= tp
    assert tuple(out.shape) == tuple(expected_shape), f"unexpected RS output shape {out.shape}"

    # Gather the TP shards back along `scatter_dim` (one SP replica) -> the full reduced tensor. RS + AG
    # == an all-reduce, so this must equal the plain sum of the partials.
    concat_dims = [scatter_dim, 0]  # TP axis -> concat on scatter_dim; SP axis -> pick a single replica
    gathered = ttnn.to_torch(
        out,
        mesh_composer=ttnn.create_mesh_composer(
            mesh_device, ttnn.MeshComposerConfig(concat_dims, ttnn.MeshShape([tp, 1]))
        ),
    )

    expected = partials.sum(dim=0, keepdim=True)  # (1, 1, S, H) -- what an all-reduce would produce
    assert tuple(gathered.shape) == tuple(expected.shape), f"{gathered.shape} != {expected.shape}"
    assert_quality(expected, gathered, pcc=0.999)
