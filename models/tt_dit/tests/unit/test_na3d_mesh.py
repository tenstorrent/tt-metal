# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""3D neighborhood attention over a volume split across a mesh, halos crossing the fabric.

Everything else in the NA3D suite either runs the whole volume on one device or hands a shard
its halo by slicing on host. This is the first test where the halo is a real
``neighbor_pad_async`` exchange between neighbouring chips, the gather indices differ per
device, and the answer is assembled from all 32 of them.

Two properties are load-bearing and only observable here:

* The exchange's ``padding_mode`` fill at the volume border never reaches the result. Window
  bounds are global, so a query near the border attends inside the volume; the pad rows sit in
  the buffer unread. ``replicate`` and ``zeros`` must therefore agree.
* Every device runs one program. ``build_mesh_device_plan`` asserts the plans agree on shape,
  which holds only because ``uniform_spans`` equalises the key spans an edge shard would
  otherwise shorten.
"""

import pytest
import torch

import ttnn

from ...layers.na3d import (
    DEFAULT_SCORE_BUDGET,
    AxisShard,
    build_mesh_device_plan,
    na3d_torch,
    neighborhood_attention_3d,
    plan_na3d_mesh,
    uniform_halo,
)
from ...parallel.manager import CCLManager
from ...utils.check import assert_quality


def _axis_shards(length: int, parts: int) -> list[AxisShard]:
    edges = [round(i * length / parts) for i in range(parts + 1)]
    return [AxisShard(length=length, start=a, stop=b) for a, b in zip(edges, edges[1:])]


def _mesh_plans(dims, kernel, mesh, halo, budget=DEFAULT_SCORE_BUDGET):
    """One plan per device, row-major over the mesh: H on axis 0, W on axis 1."""
    grid = [
        (AxisShard(dims[0], 0, dims[0]), h_shard, w_shard)
        for h_shard in _axis_shards(dims[1], mesh[0])
        for w_shard in _axis_shards(dims[2], mesh[1])
    ]
    return plan_na3d_mesh(grid, kernel, halo=halo, budget=budget)


# neighbor_pad_async routes over fabric, so the mesh has to come up with a fabric config —
# without one the op fails at get_fabric_topology with an uninitialised context. Every geometry
# is the full 4x8: smaller meshes have a known-flaky fabric handshake here (ltx_mesh_params.py
# skips 1x1 and 1x4 for the VAE decoder over it), and `require_exact_physical_num_devices` makes
# anything short of the whole galaxy skip rather than run on a slice of it.
_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}


@pytest.mark.parametrize(
    "mesh_device, device_params, dims, kernel, budget",
    [
        ((4, 8), _FABRIC, (4, 32, 64), (3, 7, 7), DEFAULT_SCORE_BUDGET),  # det-stage kernel
        ((4, 8), _FABRIC, (4, 32, 64), (11, 11, 11), DEFAULT_SCORE_BUDGET),  # stage-5 kernel, halo 5 a side
        ((4, 8), _FABRIC, (4, 48, 96), (11, 11, 11), DEFAULT_SCORE_BUDGET),  # wider shards, same halo
        # A budget small enough to force several tiles per shard, which is where the plans stop
        # agreeing on their own and the canonical padded group set is what makes them agree.
        ((4, 8), _FABRIC, (4, 32, 64), (3, 7, 7), 2**13),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("padding_mode", ["replicate", "zeros"])
@pytest.mark.parametrize("heads, head_dim", [(2, 64)])
def test_na3d_across_mesh_matches_host(*, mesh_device, dims, kernel, budget, padding_mode, heads, head_dim):
    mesh = tuple(mesh_device.shape)
    channels = heads * head_dim
    halo = (0, uniform_halo(dims[1], mesh[0], kernel[1]), uniform_halo(dims[2], mesh[1], kernel[2]))

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, *dims, heads, head_dim, dtype=torch.float32) for _ in range(3))
    expected = na3d_torch(q, k, v, kernel, scale=1.0).reshape(1, *dims, channels)

    ccl = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    # (T, H, W, C): the layout neighbor_pad_async wants, with T as its untouched outer dim.
    padded = []
    for tensor in (q, k, v):
        flat = tensor.reshape(*dims, channels)
        sharded = ttnn.from_torch(
            flat,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh, dims=[1, 2]),
        )
        padded.append(
            ccl.neighbor_pad(
                sharded,
                dims=[1, 2],
                pad_left=[halo[1], halo[2]],
                pad_right=[halo[1], halo[2]],
                padding_mode=padding_mode,
                axes=[0, 1],
                neighbor_sems=[ccl.get_np_ping_pong_semaphore(0), ccl.get_np_ping_pong_semaphore(1)],
                num_links=[1, 1],
            )
        )

    plans = _mesh_plans(dims, kernel, mesh, halo, budget=budget)
    device_plan = build_mesh_device_plan(plans, mesh_device=mesh_device)
    buffer_dims = plans[0].dims
    reshaped = [ttnn.reshape(x, (1, *buffer_dims, heads, head_dim)) for x in padded]

    actual = neighborhood_attention_3d(*reshaped, kernel_size=kernel, scale=1.0, device_plan=device_plan)

    gathered = ttnn.to_torch(actual, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[2, 3], mesh_shape=mesh))
    assert tuple(gathered.shape) == tuple(expected.shape), f"{tuple(gathered.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, gathered, pcc=0.999)
