# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import pathlib
import struct

import torch

import ttnn


@pytest.mark.parametrize("malformation", ["header-size", "shard-buffer"])
def test_load_malformed_tensor_raises_typed_error(tmp_path, expect_error, malformation):
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
    with expect_error(ttnn.MalformedTensorError, diagnostic):
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
