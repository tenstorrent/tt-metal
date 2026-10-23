# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


# Same-device point_to_point (issue #28945): sender_coord == receiver_coord must
# succeed and behave as a local on-device copy of the shard into the output tensor.
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("shape", [(30, 60), (10, 10, 30, 60), (16, 64)])
@pytest.mark.parametrize("preallocated_output", [True, False])
def test_point_to_point_same_device(mesh_device, shape, preallocated_output):
    coord = ttnn.MeshCoordinate(0, 0)

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    output_tensor = None
    if preallocated_output:
        output_tensor = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    result = ttnn.point_to_point(
        input_tensor,
        coord,
        coord,
        topology=ttnn.Topology.Linear,
        output_tensor=output_tensor,
    )

    result_torch = ttnn.to_torch(result, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert torch.equal(result_torch, torch_input), "same-device point_to_point output != input"


# Regression for the alias-first program-cache poisoning found while auditing #28945: a
# same-device call whose output aliases the input must not corrupt the cache entry used by a
# later same-shape call with a distinct preallocated output.
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_point_to_point_same_device_alias_then_distinct(mesh_device):
    mesh_device.enable_program_cache()
    coord = ttnn.MeshCoordinate(0, 0)
    shape = (64, 64)
    common = dict(
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    # 1) output aliases input: must short-circuit -> NO new program-cache entry.
    a = torch.randn(shape, dtype=torch.bfloat16)
    ta = ttnn.from_torch(a, **common)
    before = mesh_device.num_program_cache_entries()
    ttnn.point_to_point(ta, coord, coord, topology=ttnn.Topology.Linear, output_tensor=ta)
    assert (
        mesh_device.num_program_cache_entries() == before
    ), "aliased same-device output must not create a program-cache entry (should short-circuit)"

    # 2) same shape, DISTINCT preallocated output: DOES create an entry (positive control that the
    #    counter is live) AND must receive the input data (cache not poisoned by the alias).
    b = torch.randn(shape, dtype=torch.bfloat16)
    tb = ttnn.from_torch(b, **common)
    out = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), **common)
    before = mesh_device.num_program_cache_entries()
    result = ttnn.point_to_point(tb, coord, coord, topology=ttnn.Topology.Linear, output_tensor=out)
    assert (
        mesh_device.num_program_cache_entries() > before
    ), "distinct same-device output should create a program-cache entry (counter sanity check)"
    assert torch.equal(
        ttnn.to_torch(result, mesh_composer=comp), b
    ), "distinct output stale after alias-first same-device call (cache poisoning)"


# Reverse ordering: distinct output first, then an alias call, then distinct again. Confirms
# the alias no-op neither alters its own tensor nor poisons the cache for the following
# distinct-output call.
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_point_to_point_same_device_distinct_then_alias(mesh_device):
    mesh_device.enable_program_cache()
    coord = ttnn.MeshCoordinate(0, 0)
    shape = (48, 48)
    common = dict(
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    # 1) distinct preallocated output (cache miss -> correct entry)
    a = torch.randn(shape, dtype=torch.bfloat16)
    out_a = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), **common)
    r1 = ttnn.point_to_point(
        ttnn.from_torch(a, **common), coord, coord, topology=ttnn.Topology.Linear, output_tensor=out_a
    )
    assert torch.equal(ttnn.to_torch(r1, mesh_composer=comp), a), "distinct output wrong"

    # 2) alias output (output == input): no-op, input unchanged, and NO new program-cache entry
    b = torch.randn(shape, dtype=torch.bfloat16)
    tb = ttnn.from_torch(b, **common)
    before = mesh_device.num_program_cache_entries()
    r2 = ttnn.point_to_point(tb, coord, coord, topology=ttnn.Topology.Linear, output_tensor=tb)
    assert torch.equal(ttnn.to_torch(r2, mesh_composer=comp), b), "alias no-op altered the tensor"
    assert (
        mesh_device.num_program_cache_entries() == before
    ), "aliased same-device output must not create a program-cache entry (should short-circuit)"

    # 3) distinct output again: cache must still be correct (not poisoned by the alias)
    c = torch.randn(shape, dtype=torch.bfloat16)
    out_c = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), **common)
    r3 = ttnn.point_to_point(
        ttnn.from_torch(c, **common), coord, coord, topology=ttnn.Topology.Linear, output_tensor=out_c
    )
    assert torch.equal(ttnn.to_torch(r3, mesh_composer=comp), c), "distinct output stale after alias call"
