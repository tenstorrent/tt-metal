# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_num_links, per_axis_topology

pytestmark = run_for_blackhole()

SHARD_WIDTH = 1792  # 7168 over a 4-way tensor-parallel axis
PER_CHIP_TOKENS = 640
TP_AXIS = 1
SP_AXIS = 0

# Each arm names the whole box it runs on, so the ones that do not match this machine drop
# out rather than open a submesh, which fabric does not come up on.
EXACT_BOX = {"require_exact_physical_num_devices": True}
FABRIC_2D = {"fabric_config": ttnn.FabricConfig.FABRIC_2D, **EXACT_BOX}
TORUS_XY = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    **EXACT_BOX,
}
MESH_ARMS = [
    pytest.param((2, 2), FABRIC_2D, id="fabric2d-mesh-2x2"),
    pytest.param((2, 4), FABRIC_2D, id="fabric2d-mesh-2x4"),
    pytest.param((8, 4), TORUS_XY, id="torusxy-mesh-8x4"),
    # A 4x4 pipeline stage: a Galaxy carved to 16 chips, see `SUBTORUS_4X4_ENV` in the AttnRes
    # model harness.
    pytest.param((4, 4), TORUS_XY, id="torusxy-mesh-4x4"),
]


def reference(v, queries, num_sites):
    """Column 0 the sum of squares, columns 1..num_sites the dots, the rest zero."""
    out = torch.zeros(*v.shape[:-1], 32)
    out[..., 0] = (v * v).sum(dim=-1)
    out[..., 1 : 1 + num_sites] = v @ queries[0, 0, :, 1 : 1 + num_sites]
    return out


def check(out, want, num_sites):
    # The squares are packed through bf16 before they are summed; the dots accumulate in
    # fp32 dest at HiFi4.
    torch.testing.assert_close(out[..., 0], want[..., 0], rtol=5e-3, atol=0)
    passed, pcc = comp_pcc(want[..., 1 : 1 + num_sites], out[..., 1 : 1 + num_sites], 0.99999)
    assert passed, f"dots PCC {pcc}"
    torch.testing.assert_close(out[..., 1 : 1 + num_sites], want[..., 1 : 1 + num_sites], rtol=2e-2, atol=2e-2)
    assert out[..., 1 + num_sites :].abs().max() == 0, "columns past the last site must stay zero"


@pytest.mark.parametrize("planes", [[1], [6], [6, 1], [3, 1, 1, 2]], ids=lambda p: "+".join(map(str, p)))
@pytest.mark.parametrize("num_sites", [1, 24])
@pytest.mark.parametrize("num_tokens", [64, 640])
def test_attn_res_block_stats(device, planes, num_sites, num_tokens):
    torch.manual_seed(0)
    snapshots = [torch.randn(1, s, num_tokens, SHARD_WIDTH).bfloat16() for s in planes]
    queries = torch.zeros(1, 1, SHARD_WIDTH, 32)
    queries[..., 1 : 1 + num_sites] = torch.randn(SHARD_WIDTH, num_sites) * 0.02
    queries = queries.bfloat16()

    to_device = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.experimental.deepseek_prefill.attn_res_block_stats([to_device(s) for s in snapshots], to_device(queries))
    assert tuple(out.shape) == (1, sum(planes), num_tokens, 32)
    assert out.dtype == ttnn.float32
    out = ttnn.to_torch(out)

    check(out, reference(torch.cat(snapshots, dim=1).float(), queries.float(), num_sites), num_sites)


# The K3 shape, then a long sequence on a narrow shard: 160 row tiles a chip is more rows than
# the gather's per-row counts fit at full width, so they count in steps of candidates.
@pytest.mark.parametrize(
    "per_chip_tokens, shard_width", [(PER_CHIP_TOKENS, SHARD_WIDTH), (5120, 64)], ids=["k3", "long-narrow"]
)
@pytest.mark.parametrize("mesh_device, device_params", MESH_ARMS, indirect=True)
@pytest.mark.parametrize("num_sites", [1, 25])
def test_attn_res_block_stats_gathered(mesh_device, device_params, num_sites, per_chip_tokens, shard_width):
    """Each rank's statistics over its shard of `d`, gathered across the tensor-parallel axis:
    every chip of a row ends up with every rank's, rank p's in tile column p."""
    torch.manual_seed(0)
    mesh_shape = tuple(mesh_device.shape)
    tp_factor, sp_factor = mesh_shape[TP_AXIS], mesh_shape[SP_AXIS]
    planes = [6, 1]
    num_tokens = per_chip_tokens * sp_factor
    width = shard_width * tp_factor

    snapshots = [torch.randn(1, s, num_tokens, width).bfloat16() for s in planes]
    queries = torch.zeros(1, 1, width, 32)
    queries[..., 1 : 1 + num_sites] = torch.randn(width, num_sites) * 0.02
    queries = queries.bfloat16()

    def mapper(sp_dim, tp_dim):
        dims = [None, None]
        dims[SP_AXIS], dims[TP_AXIS] = sp_dim, tp_dim
        return ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=mesh_shape)

    to_device = lambda t, m, dtype=ttnn.bfloat16: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=m
    )
    gathered = to_device(torch.zeros(1, sum(planes), num_tokens, 32 * tp_factor), mapper(2, None), ttnn.float32)
    # The peers' arrivals, counted on the gather cores: the last `num_links` of the grid's last row.
    num_links = get_num_links(mesh_device, TP_AXIS)
    grid = mesh_device.compute_with_storage_grid_size()
    gather_cores = ttnn.CoreRange(
        ttnn.CoreCoord(grid.x - num_links, grid.y - 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1)
    )
    semaphore = ttnn.create_global_semaphore(mesh_device, ttnn.CoreRangeSet([gather_cores]), 0)
    out = ttnn.experimental.deepseek_prefill.attn_res_block_stats(
        [to_device(s, mapper(2, 3)) for s in snapshots],
        to_device(queries, mapper(None, 2)),
        gathered=gathered,
        cluster_axis=TP_AXIS,
        semaphore=semaphore,
        topology=per_axis_topology()[TP_AXIS],
        num_links=num_links,
    )

    # Tokens concatenate over the sequence axis and the tensor-parallel replicas stack on
    # dim 0, so every chip's copy of the gather is checked.
    dims = [None, None]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 0
    out = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=mesh_shape)
    )
    assert tuple(out.shape) == (tp_factor, sum(planes), num_tokens, 32 * tp_factor)

    v = torch.cat(snapshots, dim=1).float()
    for p in range(tp_factor):
        want = reference(
            v[..., p * shard_width : (p + 1) * shard_width],
            queries.float()[..., p * shard_width : (p + 1) * shard_width, :],
            num_sites,
        )
        for replica in range(tp_factor):
            check(out[replica : replica + 1, ..., p * 32 : (p + 1) * 32], want, num_sites)
