# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh smoke test — the D3 prerequisite, run before any module work.

Opens the target 8x4 Blackhole Galaxy mesh, builds ``MeshConfig`` + ``CCLManager``, and checks that
an all-gather and an all-reduce on the TP axis both produce the right answer. Per the recipe, no
module is written until this passes: it is what turns "the collectives are misconfigured" from a
mystery PCC failure inside attention into a two-line failure here.

Run (from the tt-metal checkout, with the prepared env exported):

    scripts/run_safe_pytest.sh models/demos/mistral_medium_3_5_128b/tests/galaxy_mesh_smoke.py -s

Topology comes from ``MISTRAL_FABRIC`` / ``MISTRAL_CCL_TOPOLOGY`` (default ``1d`` / ``linear``,
which maps on any galaxy). Run it again with ``MISTRAL_FABRIC=1d_ring MISTRAL_CCL_TOPOLOGY=ring``
where the pod is torus-wired; a failure there is an ``env`` log line and a fall back to linear, not
a bring-up blocker.
"""


import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_3_5_128b.tt.ccl import CCLManager
from models.demos.mistral_medium_3_5_128b.tt.config import MeshConfig
from models.demos.mistral_medium_3_5_128b.utils.fabric_env import (
    ccl_topology_from_env,
    fabric_config_from_env,
    topology_name,
)

L1_SMALL_SIZE = 1152
MESH_SHAPE = (8, 4)  # SP=8 rows, TP=4 cols — the spec's parallelism
PCC_LOWER_BOUND = 0.85  # spec acceptance.pcc_lower_bound


@pytest.fixture(scope="module")
def galaxy():
    rows, cols = MESH_SHAPE
    ndev = ttnn.get_num_devices()
    if ndev < rows * cols:
        pytest.skip(f"target mesh {rows}x{cols} needs {rows * cols} devices, found {ndev}")

    fabric = fabric_config_from_env()
    ttnn.set_fabric_config(fabric)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), l1_small_size=L1_SMALL_SIZE)
    logger.info(
        f"[smoke] mesh opened {tuple(mesh.shape)} ndev={mesh.get_num_devices()} "
        f"fabric={ttnn.get_fabric_config()} topology={topology_name()}"
    )
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def mesh_config():
    rows, cols = MESH_SHAPE
    return MeshConfig(MESH_SHAPE, tp=cols)


@pytest.fixture(scope="module")
def ccl(galaxy):
    # Blackhole exposes 2 fabric links per device on a multi-row mesh.
    return CCLManager(galaxy, num_links=2, topology=ccl_topology_from_env())


def test_mesh_opens(galaxy, mesh_config):
    """The target mesh opens at the spec's shape and MeshConfig agrees with it."""
    assert tuple(galaxy.shape) == MESH_SHAPE
    assert galaxy.get_num_devices() == MESH_SHAPE[0] * MESH_SHAPE[1] == 32
    assert (mesh_config.sp, mesh_config.tp) == (8, 4), f"spec wants sp=8 tp=4, got {mesh_config}"
    assert mesh_config.sp_axis == 0 and mesh_config.tp_axis == 1


def test_all_gather_tp(galaxy, mesh_config, ccl):
    """All-gather on the TP axis reassembles a column-sharded tensor.

    Shapes are the real per-chunk ones: [1, 1, seq_local, hidden] with hidden sharded over TP=4.
    """
    seq_local, hidden = 1280, 12288  # 10240 / sp=8, and the model's hidden_size
    torch.manual_seed(0)
    ref = torch.randn(1, 1, seq_local, hidden, dtype=torch.bfloat16)

    # Shard hidden over the TP cols, replicate across the SP rows.
    tt = ttnn.from_torch(
        ref,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=galaxy,
        mesh_mapper=ttnn.ShardTensor2dMesh(galaxy, mesh_shape=MESH_SHAPE, dims=[None, -1]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gathered = mesh_config.allgather(tt, ccl, axis=mesh_config.tp_axis, dim=3)

    # Every device should now hold the full tensor; check the first one.
    out = ttnn.to_torch(gathered, mesh_composer=ttnn.ConcatMesh2dToTensor(galaxy, mesh_shape=MESH_SHAPE, dims=(0, 1)))[
        :1, :1
    ]
    passing, pcc = comp_pcc(ref.float(), out.float(), PCC_LOWER_BOUND)
    logger.info(f"[smoke] all_gather(TP) PCC={pcc}")
    assert passing, f"all-gather on TP axis: PCC {pcc} < {PCC_LOWER_BOUND}"


def test_all_reduce_tp(galaxy, mesh_config, ccl):
    """All-reduce on the TP axis sums the per-column partials.

    This is the exact tail of a row-parallel matmul (o_proj / down_proj), so it is the collective
    every layer leans on. Feeding each TP column the same tensor makes the expected result
    ``tp * ref``, which catches a partial reduction as well as a wrong axis.
    """
    seq_local, hidden = 1280, 12288
    torch.manual_seed(1)
    ref = torch.randn(1, 1, seq_local, hidden, dtype=torch.bfloat16)

    # Replicate across the whole mesh: every TP column contributes the same partial.
    tt = ttnn.from_torch(
        ref,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=galaxy,
        mesh_mapper=ttnn.ReplicateTensorToMesh(galaxy),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    reduced = mesh_config.allreduce(tt, ccl, axis=mesh_config.tp_axis)

    out = ttnn.to_torch(reduced, mesh_composer=ttnn.ConcatMesh2dToTensor(galaxy, mesh_shape=MESH_SHAPE, dims=(0, 1)))[
        :1, :1
    ]
    expected = ref.float() * mesh_config.tp
    passing, pcc = comp_pcc(expected, out.float(), PCC_LOWER_BOUND)
    logger.info(f"[smoke] all_reduce(TP) PCC={pcc}  (expected {mesh_config.tp}x the input)")
    assert passing, f"all-reduce on TP axis: PCC {pcc} < {PCC_LOWER_BOUND}"
    # PCC alone would pass a uniformly-scaled result, so pin the magnitude too.
    scale = out.float().abs().mean() / ref.float().abs().mean()
    assert abs(scale - mesh_config.tp) < 0.1, f"all-reduce scaled by {scale:.3f}, expected {mesh_config.tp}"


def test_all_gather_sp(galaxy, mesh_config, ccl):
    """All-gather on the SP axis reassembles the sequence — the collective chunked prefill needs
    when a stage has to see the whole sequence rather than its SP shard."""
    seq_total, hidden = 10240, 12288
    torch.manual_seed(2)
    ref = torch.randn(1, 1, seq_total, hidden, dtype=torch.bfloat16)

    # Shard seq over the SP rows, replicate across the TP cols.
    tt = ttnn.from_torch(
        ref,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=galaxy,
        mesh_mapper=ttnn.ShardTensor2dMesh(galaxy, mesh_shape=MESH_SHAPE, dims=[-2, None]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gathered = mesh_config.allgather(tt, ccl, axis=mesh_config.sp_axis, dim=2)

    out = ttnn.to_torch(gathered, mesh_composer=ttnn.ConcatMesh2dToTensor(galaxy, mesh_shape=MESH_SHAPE, dims=(0, 1)))[
        :1, :1
    ]
    passing, pcc = comp_pcc(ref.float(), out.float(), PCC_LOWER_BOUND)
    logger.info(f"[smoke] all_gather(SP) PCC={pcc}")
    assert passing, f"all-gather on SP axis: PCC {pcc} < {PCC_LOWER_BOUND}"


def test_dram_banks_and_alignment(galaxy):
    """The KV allocator's two hard preconditions, checked before any cache is allocated."""
    from models.demos.common.prefill.runners.migration import get_num_dram_banks

    banks = get_num_dram_banks(galaxy)
    logger.info(f"[smoke] dram banks per device: {banks}")
    assert banks > 0

    max_seq_len, sp, chunk_size = 262144, 8, 5120
    # The spec's own constraint on both values.
    assert max_seq_len % (ttnn.TILE_SIZE * sp) == 0, "max_seq_len must be a multiple of TILE_SIZE*sp"
    assert chunk_size % (ttnn.TILE_SIZE * sp) == 0, "chunk_size must be a multiple of TILE_SIZE*sp"

    # The spec's max_seq_len is NOT a multiple of its chunk_size (262144 / 5120 = 51.2), but the
    # block-cyclic period must divide the allocated capacity or the last partial period aliases onto
    # the first. The allocator therefore rounds capacity UP to a whole number of chunks; check that
    # the rounded value still satisfies the tile/SP alignment.
    from models.demos.mistral_medium_3_5_128b.tt.attention.kv_cache import round_cache_capacity

    capacity = round_cache_capacity(max_seq_len, chunk_size)
    assert capacity == 266240, capacity
    assert capacity >= max_seq_len
    assert capacity % chunk_size == 0, "the block-cyclic period must divide the cache capacity"
    assert capacity % (ttnn.TILE_SIZE * sp) == 0, "rounded capacity must stay tile/SP aligned"
