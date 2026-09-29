# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness for high_bw_all_reduce: run under `run_safe_pytest.sh --profile` and read
DEVICE KERNEL DURATION from the ops CSV. Correctness is still asserted (PCC)."""

import pytest
import torch
import ttnn

from ttnn.operations.high_bw_all_reduce import high_bw_all_reduce


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("num_links", [1, 2], ids=["links1", "links2"])
@pytest.mark.parametrize(
    "cluster_axis,topology",
    [(0, ttnn.Topology.Linear), (1, ttnn.Topology.Linear), (None, ttnn.Topology.Linear), (None, ttnn.Topology.Ring)],
    ids=["axis0-linear", "axis1-linear", "none-linear", "none-ring"],
)
@pytest.mark.parametrize("shape", [(1, 1, 2048, 2048), (1, 1, 4096, 4096)], ids=lambda s: "x".join(map(str, s)))
def test_perf(mesh_device, shape, cluster_axis, topology, num_links):
    torch.manual_seed(0)
    rows, cols = tuple(mesh_device.shape)
    stacked = torch.randn((rows, cols, *shape), dtype=torch.bfloat16)
    dims = (0, 1) if cluster_axis is None else (cluster_axis,)
    expected = stacked.float().sum(dim=dims, keepdim=True).expand_as(stacked).bfloat16()
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    t = ttnn.from_torch(
        glob,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    for _ in range(2):
        out = high_bw_all_reduce(t, cluster_axis=cluster_axis, topology=topology, num_links=num_links)
    ttnn.synchronize_device(mesh_device)
    for idx, dt in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        assert _pcc(ttnn.to_torch(dt), expected[r, c]) > 0.995


# Refinement 3 no-regression guard set (one representative per config the credit batch touches):
# fp32 (packet_tiles = 1), a ragged-tail shape and the single-tile shape, on the axis-0 line.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("num_links", [1, 2], ids=["links1", "links2"])
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 1, 2048, 2048), ttnn.float32),
        ((1, 1, 4001, 2048), ttnn.bfloat16),
        ((1, 1, 32, 32), ttnn.bfloat16),
    ],
    ids=["fp32-2048x2048", "bf16-ragged-4001x2048", "bf16-single-tile"],
)
def test_perf_guard(mesh_device, shape, dtype, num_links):
    torch.manual_seed(0)
    rows, cols = tuple(mesh_device.shape)
    stacked = torch.randn((rows, cols, *shape), dtype=torch.float32)
    expected = stacked.sum(dim=0, keepdim=True).expand_as(stacked)
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    t = ttnn.from_torch(
        glob,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    for _ in range(2):
        out = high_bw_all_reduce(t, cluster_axis=0, topology=ttnn.Topology.Linear, num_links=num_links)
    ttnn.synchronize_device(mesh_device)
    for idx, dt in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        assert _pcc(ttnn.to_torch(dt), expected[r, c]) > 0.995
