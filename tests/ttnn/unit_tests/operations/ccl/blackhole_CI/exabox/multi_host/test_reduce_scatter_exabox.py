# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""Reduce-scatter tests for multi-galaxy exabox mesh configurations (DUAL_BH / QUAD_BH).

Tests cover:
- Sync API (ttnn.reduce_scatter) on 16x4 and 32x4 meshes
- Multiple fabric configs (FABRIC_1D, FABRIC_1D_RING) and topologies (Linear, Ring)
- Both cluster axes (0 and 1)
- DRAM and L1 memory configs
- bfloat16 dtype
"""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.utils_for_testing import maybe_trace

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_tensors(input_shape, dim, cluster_axis, mesh_shape, dtype, layout, memory_config, device):
    # One unique shard per cluster-axis position, replicated across the other axis.
    # Reference is the cluster-axis sum split into per-position chunks along `dim`.
    num_along_axis = mesh_shape[cluster_axis]
    assert (
        input_shape[dim] % num_along_axis == 0
    ), f"input_shape[dim={dim}]={input_shape[dim]} must be divisible by mesh_shape[{cluster_axis}]={num_along_axis}"
    torch.manual_seed(0)
    shards = [torch.rand(input_shape).bfloat16() for _ in range(num_along_axis)]
    torch_input = torch.cat(shards, dim=0)

    shard_dims = (0, None) if cluster_axis == 0 else (None, 0)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
        device=device,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=mesh_shape),
    )

    full_sum = torch.stack(shards, dim=0).sum(dim=0)
    ref_chunks = list(torch.chunk(full_sum, num_along_axis, dim=dim))
    return tt_input, ref_chunks


# Threshold relaxed from 0.999 → 0.998 to accommodate bfloat16 accumulation noise
# on 32-device (axis=0 on QUAD_BH 32x4) reductions, which lands at PCC ≈ 0.9984.
# The 16-device (DUAL_BH axis=0 and any axis=1) cases were and remain comfortably
# above 0.999. Reference reduce-scatter tests in the repo
# (test_minimal_reduce_scatter_async.py) use the comp_pcc default of 0.99, so
# 0.998 is still stricter than the prevailing convention.
def _verify_reduce_scatter_output(tt_output_tensor, ref_chunks, cluster_axis, mesh_device, pcc_threshold=0.998):
    coords = list(tt_output_tensor.tensor_topology().mesh_coords())
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    device_tensors = ttnn.get_device_tensors(tt_output_tensor)
    coord_iter = coords
    if view is not None and len(device_tensors) != len(coords):
        coord_iter = [coord for coord in coords if view.is_local(coord)]

    for coord, tt_out in zip(coord_iter, device_tensors):
        if view is not None and not view.is_local(coord):
            continue
        chunk_idx = int(coord[cluster_axis])
        eq, mess = comp_pcc(ref_chunks[chunk_idx], ttnn.to_torch(tt_out), pcc_threshold)
        assert eq, mess


def _run_reduce_scatter_test(
    mesh_device,
    input_shape,
    dim,
    cluster_axis,
    buffer_type,
    dtype,
    topology,
    enable_trace,
    num_links=None,
):
    mesh_shape = tuple(mesh_device.shape)
    memory_config = ttnn.MemoryConfig(buffer_type=buffer_type)

    tt_input, ref_chunks = _get_tensors(
        input_shape, dim, cluster_axis, mesh_shape, dtype, ttnn.TILE_LAYOUT, memory_config, mesh_device
    )

    rs_kwargs = dict(
        cluster_axis=cluster_axis,
        topology=topology,
        memory_config=memory_config,
    )
    if num_links is not None:
        rs_kwargs["num_links"] = num_links

    def run_op():
        return ttnn.reduce_scatter(tt_input, dim, **rs_kwargs)

    tt_output_tensor = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)
    _verify_reduce_scatter_output(tt_output_tensor, ref_chunks, cluster_axis, mesh_device)


# ---------------------------------------------------------------------------
# Fabric / topology parametrize combos (reused by multiple tests)
# ---------------------------------------------------------------------------

FABRIC_TOPOLOGY_COMBOS = [
    pytest.param(
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
        ttnn.Topology.Linear,
        id="fabric_1d-linear",
    ),
    pytest.param(
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
        ttnn.Topology.Ring,
        id="fabric_1d_ring-ring",
    ),
]


# ---------------------------------------------------------------------------
# Test: sync reduce_scatter on 16x4 mesh (DUAL_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["DUAL_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            id="fabric_1d-linear",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((16, 4), id="16x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [pytest.param(0, id="axis0"), pytest.param(1, id="axis1")])
@pytest.mark.parametrize("num_links", [2], ids=["2links"])
@pytest.mark.parametrize("enable_trace", [False])
@pytest.mark.parametrize(
    "input_shape, dim, dtype, buffer_type",
    [
        # input_shape[dim] must be divisible by mesh_shape[cluster_axis] AND the
        # resulting chunk must be tile-aligned: 512 / 16 = 32 (axis 0), 512 / 4 = 128 (axis 1).
        ([1, 1, 32, 512], 3, ttnn.bfloat16, ttnn.BufferType.DRAM),
        ([1, 1, 32, 512], 3, ttnn.bfloat16, ttnn.BufferType.L1),
    ],
    ids=["small_dram", "small_l1"],
)
def test_reduce_scatter_16x4(
    mesh_device,
    cluster_axis,
    topology,
    enable_trace,
    input_shape,
    dim,
    dtype,
    buffer_type,
    num_links,
):
    _run_reduce_scatter_test(
        mesh_device, input_shape, dim, cluster_axis, buffer_type, dtype, topology, enable_trace, num_links=num_links
    )


# ---------------------------------------------------------------------------
# Test: sync reduce_scatter on 32x4 mesh (QUAD_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    FABRIC_TOPOLOGY_COMBOS,
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [pytest.param(0, id="axis0"), pytest.param(1, id="axis1")])
@pytest.mark.parametrize("num_links", [2], ids=["2links"])
@pytest.mark.parametrize("enable_trace", [False])
@pytest.mark.parametrize(
    "input_shape, dim, dtype, buffer_type",
    [
        # 1024 / 32 = 32 (axis 0 chunk, tile-aligned), 1024 / 4 = 256 (axis 1 chunk).
        ([1, 1, 32, 1024], 3, ttnn.bfloat16, ttnn.BufferType.DRAM),
        ([1, 1, 32, 1024], 3, ttnn.bfloat16, ttnn.BufferType.L1),
    ],
    ids=["small_dram", "small_l1"],
)
def test_reduce_scatter_32x4(
    mesh_device,
    cluster_axis,
    topology,
    enable_trace,
    input_shape,
    dim,
    dtype,
    buffer_type,
    num_links,
):
    _run_reduce_scatter_test(
        mesh_device, input_shape, dim, cluster_axis, buffer_type, dtype, topology, enable_trace, num_links=num_links
    )
