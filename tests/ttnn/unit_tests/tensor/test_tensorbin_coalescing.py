# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch

import ttnn


def _make_distributed_bf4(mesh_device, values, width=32):
    device_count = mesh_device.get_num_devices()
    shards = [
        torch.full((32, 1, 32, width), values(device_index), dtype=torch.bfloat16)
        for device_index in range(device_count)
    ]
    torch_tensor = torch.cat(shards, dim=0)
    tt_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    return torch_tensor, tt_tensor


def test_coalesce_tensorbins_scales_k3_width_shard_height(tmp_path):
    shape = (1, 1, 3584, 32)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(grid, (3584, 32), ttnn.ShardOrientation.ROW_MAJOR)
    spec = ttnn.TensorSpec(
        shape,
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        shard_spec,
        ttnn.BufferType.DRAM,
    )
    input_paths = [tmp_path / "gate0.tensorbin", tmp_path / "gate1.tensorbin"]
    for value, input_path in enumerate(input_paths):
        tensor = ttnn.from_torch(torch.full(shape, value, dtype=torch.bfloat16), spec=spec)
        ttnn.dump_tensor(input_path, tensor, mode=ttnn.DumpTensorMode.LOCAL)

    coalesced = ttnn.coalesce_tensorbins(input_paths)

    assert tuple(coalesced.shape) == (2, 1, 3584, 32)
    assert coalesced.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert coalesced.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert coalesced.memory_config().shard_spec.grid == grid
    assert tuple(coalesced.memory_config().shard_spec.shape) == (7168, 32)
    assert coalesced.memory_config().shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    expected = torch.cat([torch.full(shape, value, dtype=torch.bfloat16) for value in range(len(input_paths))], dim=0)
    assert torch.equal(ttnn.to_torch(coalesced), expected)


def test_coalesce_tensorbins_distributed_bf4_to_mesh(tmp_path, mesh_device):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("This distributed ownership regression requires a 32-device Galaxy")

    input_paths = [tmp_path / "expert0.tensorbin", tmp_path / "expert1.tensorbin"]

    torch_inputs = []
    input_tensors = []
    for input_index, input_path in enumerate(input_paths):
        torch_tensor, tt_tensor = _make_distributed_bf4(
            mesh_device, lambda device_index, input_index=input_index: input_index * 4 + device_index % 4
        )
        torch_inputs.append(torch_tensor)
        input_tensors.append(tt_tensor)
        ttnn.dump_tensor(input_path, tt_tensor)

    coalesced = ttnn.coalesce_tensorbins(input_paths)
    assert coalesced.storage_type() == ttnn.StorageType.HOST
    assert tuple(coalesced.shape) == (64, 1, 32, 32)
    assert coalesced.dtype == ttnn.bfloat4_b
    assert coalesced.layout == ttnn.TILE_LAYOUT
    assert coalesced.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert coalesced.tensor_topology() == input_tensors[0].tensor_topology()
    template_host = ttnn.load_tensor(input_paths[0])

    # The returned tensor owns its coalesced payload; source mappings/files are not needed after the call.
    for input_path in input_paths:
        input_path.unlink()

    coalesced_device = ttnn.to_device(coalesced, mesh_device)
    actual = ttnn.to_torch(
        coalesced_device,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    per_device_rows = torch_inputs[0].shape[0] // mesh_device.get_num_devices()
    expected = torch.cat(
        [
            torch.cat(
                [
                    tensor[device_index * per_device_rows : (device_index + 1) * per_device_rows]
                    for tensor in torch_inputs
                ],
                dim=0,
            )
            for device_index in range(mesh_device.get_num_devices())
        ],
        dim=0,
    )
    assert torch.equal(actual, expected)

    packed_addresses = [tensor.buffer_address() for tensor in ttnn.get_device_tensors(coalesced_device)]
    alias = ttnn.alias_coalesced_tensor(coalesced_device, template_host)
    alias_addresses = [tensor.buffer_address() for tensor in ttnn.get_device_tensors(alias)]

    assert tuple(alias.shape) == tuple(template_host.shape)
    assert alias.dtype == template_host.dtype
    assert alias.layout == template_host.layout
    assert alias.memory_config() == template_host.memory_config()
    assert alias.tensor_topology() == template_host.tensor_topology()
    assert alias_addresses == packed_addresses

    del coalesced_device
    gc.collect()
    assert alias.is_allocated()
    alias_actual = ttnn.to_torch(
        alias,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    assert torch.equal(alias_actual, torch_inputs[0])

    # Explicitly releasing the surviving alias after its packed owner is gone must not double-free.
    alias.deallocate()
    assert not alias.is_allocated()


def test_coalesce_tensorbins_rejects_shape_mismatch_and_truncation(tmp_path, mesh_device):
    first_path = tmp_path / "first.tensorbin"
    mismatch_path = tmp_path / "mismatch.tensorbin"
    truncated_path = tmp_path / "truncated.tensorbin"

    _, first = _make_distributed_bf4(mesh_device, lambda device_index: device_index % 4)
    _, mismatch = _make_distributed_bf4(mesh_device, lambda device_index: device_index % 4, width=64)
    ttnn.dump_tensor(first_path, first)
    ttnn.dump_tensor(mismatch_path, mismatch)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        ttnn.coalesce_tensorbins([first_path, mismatch_path])

    data = first_path.read_bytes()
    truncated_path.write_bytes(data[:-1])
    with pytest.raises(RuntimeError, match="truncated"):
        ttnn.coalesce_tensorbins([first_path, truncated_path])
