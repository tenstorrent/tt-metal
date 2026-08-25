# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Probe: why didn't the P*page_size > 512 KiB accumulator gate fire for a
(1,1,1024,512) f32 shard? Print the buffer quantities validate() sees."""
import os
from math import prod

import pytest
import torch
import ttnn

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _hw_mesh_shape(default=(1, 4)):
    raw = os.environ.get("CCL_HW_MESH_SHAPE")
    return tuple(int(x) for x in raw.split(",")) if raw else default


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [_hw_mesh_shape()], indirect=True)
def test_probe_budget_quantities(mesh_device, topology):
    num_devices = prod(tuple(mesh_device.shape))
    shard_shape = (1, 1, 1024, 512)
    full = torch.randn((num_devices, 1, 1024, 512), dtype=torch.float32)
    t = ttnn.from_torch(
        full,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    print(
        f"\nPROBE shape={list(t.shape)} dtype={t.dtype} layout={t.layout} "
        f"buffer_num_pages={t.buffer_num_pages()} buffer_page_size={t.buffer_page_size()} "
        f"product={t.buffer_num_pages() * t.buffer_page_size()}"
    )


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [_hw_mesh_shape()], indirect=True)
def test_probe_validate_gate(mesh_device, topology):
    from ttnn.operations.all_reduce.all_reduce import validate, _MAX_ACCUMULATOR_BYTES

    num_devices = prod(tuple(mesh_device.shape))
    full = torch.randn((num_devices, 1, 1024, 512), dtype=torch.float32)
    t = ttnn.from_torch(
        full,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    print(f"\nPROBE gate: budget={_MAX_ACCUMULATOR_BYTES} product={t.buffer_num_pages() * t.buffer_page_size()}")
    try:
        validate(t, topology=topology, output_tensor=None)
        print("PROBE validate: NO RAISE")
    except Exception as e:
        print(f"PROBE validate raised {type(e).__name__}: {e}")
