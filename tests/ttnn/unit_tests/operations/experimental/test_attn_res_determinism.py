# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Determinism of the three AttnRes ops: many back-to-back runs, no sync between them, bit-identical out.

The ops' accuracy is gated in their own test files; this one only asks whether a run repeats. Each
test builds its inputs, persistent buffers and semaphores once, the way the model holds them, keeps
the first run's outputs on device as the reference, and folds every later run's exact mismatch into
one device marker read back at the end (`device_determinism.py`). Without a sync the chips drift
apart, which is what exposes a buffer or a semaphore reused before a slow peer is done with it.

Shapes are K3's: `d = 7168` over TP4, so 1792 columns per chip, and the 5120-token chunk over the SP
axis, 640 rows per chip on 8x4 and 1280 on a 4x4 stage. The sealed set is the boundary block's,
six inherited snapshots and the sealed live stream, with 25 read sites, the first one closed.

The 4x4 arm needs the Galaxy carved to a subtorus:
    TT_VISIBLE_DEVICES=2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31
    TT_MESH_GRAPH_DESC_PATH=models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_subtorus_xy4_graph_descriptor.textproto
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.attn_res.device_determinism import (
    deterministic_input,
    device_mismatch_marker_is_set,
    fold_mismatches,
    mismatch_report,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import STATS_FIDELITY
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_num_links, per_axis_topology

pytestmark = run_for_blackhole()

HIDDEN_SIZE = 7168
GLOBAL_TOKENS = 5120
EPS = 1e-5
TP_AXIS = 1
SP_AXIS = 0
# The boundary block: six inherited snapshots and the sealed live stream, then one more snapshot a
# block later. 24 read sites and the read owed across the rank boundary, which comes back closed.
PLANES = (6, 1, 1)
NUM_SITES = 25
NUM_CLOSED = 1
ITERATIONS = 100

EXACT_BOX = {"require_exact_physical_num_devices": True}
TORUS_XY = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    **EXACT_BOX,
}
MESH_ARMS = [
    pytest.param((8, 4), TORUS_XY, id="torusxy-mesh-8x4"),
    pytest.param((4, 4), TORUS_XY, id="torusxy-mesh-4x4"),
]


def _layout(mesh_device):
    mesh_shape = tuple(mesh_device.shape)
    return mesh_shape, mesh_shape[TP_AXIS], mesh_shape[SP_AXIS]


def _mapper(mesh_device, sp_dim, tp_dim):
    mesh_shape = tuple(mesh_device.shape)
    dims = [None, None]
    dims[SP_AXIS], dims[TP_AXIS] = sp_dim, tp_dim
    return ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=mesh_shape)


def _to_device(mesh_device, tensor, mapper, dtype=ttnn.bfloat16):
    return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)


def _sealed(mesh_device):
    """The snapshots, `[1, S, N, d]` each, sharded tokens over SP and `d` over TP."""
    return [
        _to_device(
            mesh_device, deterministic_input(1, planes, GLOBAL_TOKENS, HIDDEN_SIZE, seed=i), _mapper(mesh_device, 2, 3)
        )
        for i, planes in enumerate(PLANES)
    ]


def _score_columns(mesh_device):
    """`[1, 1, d, 32]` with a query per site in columns 1..NUM_SITES and column 0 zero, `d` over TP."""
    queries = torch.zeros(1, 1, HIDDEN_SIZE, 32)
    queries[..., 1 : 1 + NUM_SITES] = deterministic_input(
        HIDDEN_SIZE, NUM_SITES, seed=len(PLANES), scale=0.02, dtype=torch.float32
    )
    return _to_device(mesh_device, queries, _mapper(mesh_device, None, 2))


def _num_links(mesh_device):
    """The TP axis's live fabric links, as the model takes them: 2 on a Galaxy, 1 on a 4x4 subtorus carved from one."""
    return get_num_links(mesh_device, TP_AXIS)


def _block_stats_semaphore(mesh_device):
    """Over the gather cores: the last `num_links` cores of the grid's last row."""
    grid = mesh_device.compute_with_storage_grid_size()
    num_links = _num_links(mesh_device)
    cores = ttnn.CoreRange(ttnn.CoreCoord(grid.x - num_links, grid.y - 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1))
    return ttnn.create_global_semaphore(mesh_device, ttnn.CoreRangeSet([cores]), 0)


def _gathered(mesh_device, num_candidates, tp_factor):
    """One persistent gather target per sealed depth, as the model's `_gathered_stats` keeps them."""
    zeros = torch.zeros(1, num_candidates, GLOBAL_TOKENS, 32 * tp_factor)
    return _to_device(mesh_device, zeros, _mapper(mesh_device, 2, None), ttnn.float32)


def _block_stats(mesh_device, sealed, queries, gathered, semaphore):
    return ttnn.experimental.deepseek_prefill.attn_res_block_stats(
        sealed,
        queries,
        gathered=gathered,
        cluster_axis=TP_AXIS,
        semaphore=semaphore,
        topology=per_axis_topology()[TP_AXIS],
        num_links=_num_links(mesh_device),
        compute_kernel_config=STATS_FIDELITY,
    )


def _assert_repeats(marker, names, references, outputs, what):
    if device_mismatch_marker_is_set(marker):
        raise AssertionError(f"{what} is not deterministic:\n{mismatch_report(names, references, outputs)}")
    logger.info(f"{what}: {ITERATIONS} runs bit-identical")


@pytest.mark.parametrize("mesh_device, device_params", MESH_ARMS, indirect=True)
def test_attn_res_block_stats_repeats(mesh_device, device_params):
    """The gathered statistics, alternating two sealed depths as consecutive blocks do.

    Each depth has its own persistent `gathered` buffer and its own program, and both depths share
    one arrival semaphore, so every run reuses a buffer its peers wrote into two runs ago and a
    semaphore the other depth's program last left. The gathered buffer is the output, so the
    reference is a copy of it.
    """
    _, tp_factor, _ = _layout(mesh_device)
    sealed = _sealed(mesh_device)
    queries = _score_columns(mesh_device)
    semaphore = _block_stats_semaphore(mesh_device)
    depths = [sealed[:2], sealed]
    gathered = [_gathered(mesh_device, sum(PLANES[: len(depth)]), tp_factor) for depth in depths]

    references = [
        ttnn.clone(_block_stats(mesh_device, depth, queries, out, semaphore)) for depth, out in zip(depths, gathered)
    ]
    marker = None
    for iteration in range(len(depths), ITERATIONS):
        which = iteration % len(depths)
        out = _block_stats(mesh_device, depths[which], queries, gathered[which], semaphore)
        marker = fold_mismatches(marker, [references[which]], [out])

    names = [f"gathered C={sum(PLANES[: len(depth)])}" for depth in depths]
    _assert_repeats(marker, names, references, gathered, "attn_res_block_stats")


@pytest.mark.parametrize("mesh_device, device_params", MESH_ARMS, indirect=True)
def test_attn_res_block_mix_repeats(mesh_device, device_params):
    """The boundary block's scoring and mixture, from one gathered statistics tensor held for every run.

    The op is rank-local, so each chip repeats on its own; the mesh is here so every chip's shard is
    a different one, at the shape it has in the model.
    """
    _, tp_factor, _ = _layout(mesh_device)
    sealed = _sealed(mesh_device)[:2]
    gathered = _gathered(mesh_device, sum(PLANES[:2]), tp_factor)
    stats = _block_stats(
        mesh_device, sealed, _score_columns(mesh_device), gathered, _block_stats_semaphore(mesh_device)
    )

    def mix():
        return list(
            ttnn.experimental.deepseek_prefill.attn_res_block_mix(
                sealed, stats, num_sites=NUM_SITES, eps=EPS, num_closed_sites=NUM_CLOSED
            )
        )

    references = mix()
    marker = None
    for iteration in range(1, ITERATIONS):
        outputs = mix()
        marker = fold_mismatches(marker, references, outputs)
        # The last run's outputs stay for the failure report.
        if iteration < ITERATIONS - 1:
            for output in outputs:
                ttnn.deallocate(output)

    _assert_repeats(marker, ["partials", "shifts", "masses", "closed"], references, outputs, "attn_res_block_mix")


@pytest.mark.parametrize("mesh_device, device_params", MESH_ARMS, indirect=True)
def test_attn_res_gather_softmax_repeats(mesh_device, device_params):
    """An ordinary layer's read: the live stream with a deferred write folded in and settled, at a
    few sites of one batch.

    Every run reuses the one exchange scratch and the one arrival semaphore the model keeps for the
    whole walk, and the site reaches the cached program as a runtime arg, so the sites rotate.
    """
    _, tp_factor, _ = _layout(mesh_device)
    num_sites = NUM_SITES - NUM_CLOSED
    sites = (0, num_sites // 2, num_sites - 1)
    stream = lambda seed: _to_device(
        mesh_device, deterministic_input(1, 1, GLOBAL_TOKENS, HIDDEN_SIZE, seed=seed), _mapper(mesh_device, 2, 3)
    )
    scalars = lambda seed, scale, offset: _to_device(
        mesh_device,
        deterministic_input(num_sites, 1, GLOBAL_TOKENS, 1, seed=seed, scale=scale, offset=offset, dtype=torch.float32),
        _mapper(mesh_device, 2, None),
        ttnn.float32,
    )
    partial = _to_device(
        mesh_device, deterministic_input(num_sites, 1, GLOBAL_TOKENS, HIDDEN_SIZE, seed=0), _mapper(mesh_device, 2, 3)
    )
    running_sum, pending = stream(1), stream(2)
    # A mass is a sum of exponentials against the running maximum, so at least one.
    shift, mass = scalars(3, 2.0, 0.0), scalars(4, 0.5, 4.0)
    query = _to_device(
        mesh_device, deterministic_input(1, 1, 1, HIDDEN_SIZE, seed=5, scale=0.05), _mapper(mesh_device, None, 3)
    )
    exchange = _to_device(
        mesh_device, torch.zeros(1, 2 * tp_factor, GLOBAL_TOKENS, 1), _mapper(mesh_device, 2, None), ttnn.float32
    )
    grid = mesh_device.compute_with_storage_grid_size()
    semaphore = ttnn.create_global_semaphore(
        mesh_device,
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]),
        0,
    )

    def read(site):
        return list(
            ttnn.experimental.deepseek_prefill.attn_res_gather_softmax(
                partial,
                running_sum,
                shift,
                mass,
                query,
                exchange,
                semaphore,
                site=site,
                cluster_axis=TP_AXIS,
                topology=per_axis_topology()[TP_AXIS],
                inv_hidden_size=1.0 / HIDDEN_SIZE,
                eps=EPS,
                pending=pending,
            )
        )

    references = [read(site) for site in sites]
    marker = None
    for iteration in range(len(sites), ITERATIONS):
        which = iteration % len(sites)
        outputs = read(sites[which])
        marker = fold_mismatches(marker, references[which], outputs)
        if iteration < ITERATIONS - 1:
            for output in outputs:
                ttnn.deallocate(output)

    names = [f"site {sites[which]} read", f"site {sites[which]} settled stream"]
    _assert_repeats(marker, names, references[which], outputs, "attn_res_gather_softmax")
