# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib
import struct
import subprocess
import sys

import torch

import ttnn


def test_load_tensor_without_device_leaves_devices_alone(tmp_path, device):
    # The device fixture holds the chips open. A load that initialized MetalContext would open
    # the cluster again in the child, which logs the UMD line below or blocks on the chip lock.
    shards = torch.cat([torch.full((32, 64), value, dtype=torch.bfloat16) for value in (11, 29)], dim=1)
    mapper = ttnn.create_mesh_mapper(
        ttnn.MeshShape(1, 2),
        ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(1)]),
    )
    tensor_path = tmp_path / "two_shards.tensorbin"
    torch_path = tmp_path / "loaded.pt"
    tensor = ttnn.from_torch(shards, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
    ttnn.dump_tensor(tensor_path, tensor, mode=ttnn.DumpTensorMode.LOCAL)

    child = (
        "import sys, torch, ttnn\n"
        "loaded = ttnn.load_tensor(sys.argv[1])\n"
        "torch.save([ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(loaded)], sys.argv[2])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", child, str(tensor_path), str(torch_path)],
        env={**os.environ, "TT_LOGGER_LEVEL": "Info"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Opening user mode device driver" not in output, output
    loaded = torch.load(torch_path)
    assert len(loaded) == 2
    torch.testing.assert_close(torch.cat(loaded, dim=1), shards, rtol=0, atol=0)


@pytest.mark.parametrize("malformation", ["header-size", "shard-buffer"])
def test_load_malformed_tensor_raises(tmp_path, expect_error, malformation):
    file_name = tmp_path / "malformed.tensorbin"
    expected = torch.full((32, 32), 11, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(expected, layout=ttnn.TILE_LAYOUT)
    ttnn.dump_tensor(file_name, tensor, mode=ttnn.DumpTensorMode.LOCAL)
    original = file_name.read_bytes()
    data = bytearray(original)
    if malformation == "header-size":
        data[:8] = len(data).to_bytes(8, byteorder="little")
    else:
        # tensor.fbs: Tensor.shards is field 2; TensorShard.buffer_type is field 0.
        # NONE is structurally valid FlatBuffers data but cannot represent a tensor shard.
        def field(table, index):
            vtable = table - struct.unpack_from("<i", data, table)[0]
            return table + struct.unpack_from("<H", data, vtable + 4 + index * 2)[0]

        root = 8 + struct.unpack_from("<I", data, 8)[0]
        shards_field = field(root, 2)
        shards = shards_field + struct.unpack_from("<I", data, shards_field)[0]
        first_shard = shards + 4 + struct.unpack_from("<I", data, shards + 4)[0]
        buffer_type = field(first_shard, 0)
        assert data[buffer_type] == 1
        data[buffer_type] = 0
    file_name.write_bytes(data)

    diagnostic = "truncated or corrupt" if malformation == "header-size" else "Only InlineFileStorage"
    with expect_error(RuntimeError, diagnostic):
        ttnn.load_tensor(file_name)

    file_name.write_bytes(original)
    torch.testing.assert_close(ttnn.to_torch(ttnn.load_tensor(file_name)), expected, rtol=0, atol=0)


@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_dump_and_load(tmp_path, height, width, layout):
    file_name = tmp_path / pathlib.Path("tensor.tensorbin")

    torch_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_tensor, layout=layout)
    ttnn.dump_tensor(file_name, tensor)

    loaded_tensor = ttnn.load_tensor(file_name)
    loaded_torch_tensor = ttnn.to_torch(loaded_tensor)
    assert torch.allclose(torch_tensor, loaded_torch_tensor)


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_dump_and_load_int8(tmp_path, height, width, layout):
    file_name = tmp_path / pathlib.Path("tensor.tensorbin")

    torch_tensor = torch.randint(-128, 128, (height, width), dtype=torch.int8)
    tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.int8, layout=layout)
    ttnn.dump_tensor(file_name, tensor)

    loaded_tensor = ttnn.load_tensor(file_name)
    assert loaded_tensor.dtype == ttnn.int8
    loaded_torch_tensor = ttnn.to_torch(loaded_tensor)
    assert torch.equal(torch_tensor, loaded_torch_tensor)


@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_dump_and_load_local_mode(tmp_path, height, width, layout):
    file_name = tmp_path / pathlib.Path("tensor.tensorbin")

    torch_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_tensor, layout=layout)
    ttnn.dump_tensor(file_name, tensor, mode=ttnn.DumpTensorMode.LOCAL)

    loaded_tensor = ttnn.load_tensor(file_name)
    loaded_torch_tensor = ttnn.to_torch(loaded_tensor)
    assert torch.allclose(torch_tensor, loaded_torch_tensor)


@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [None, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_dump_from_device_and_load_to_device(tmp_path, device, height, width, layout, memory_config):
    file_name = tmp_path / pathlib.Path("tensor.tensorbin")

    torch_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device, memory_config=memory_config)
    ttnn.dump_tensor(file_name, tensor)

    loaded_tensor = ttnn.load_tensor(file_name, device=device)
    if memory_config is not None:
        assert ttnn.get_memory_config(loaded_tensor) == memory_config
    else:
        assert ttnn.get_memory_config(loaded_tensor) == ttnn.DRAM_MEMORY_CONFIG

    loaded_torch_tensor = ttnn.to_torch(loaded_tensor)
    assert torch.allclose(torch_tensor, loaded_torch_tensor)
