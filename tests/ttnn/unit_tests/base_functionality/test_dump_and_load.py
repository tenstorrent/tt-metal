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


def _mislabelled_replicate_tensor():
    """A 1x2 host tensor sharded along dim 1 whose topology was relabelled as fully replicated.

    The two shards hold different values (11s and 29s), so a serializer that trusts the Replicate label
    writes the first shard for both coordinates and the second one is lost. Returns the tensor, the
    label that describes the data, and the torch source.
    """
    shards = torch.cat([torch.full((32, 32), value, dtype=torch.bfloat16) for value in (11, 29)], dim=1)
    mapper = ttnn.create_mesh_mapper(
        ttnn.MeshShape(1, 2),
        ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(1)]),
    )
    tensor = ttnn.from_torch(shards, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
    # Spelled out rather than read back from the tensor: `tensor_topology()` aliases the tensor's own
    # topology, which the update below overwrites.
    mesh_coords = [ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1)]
    true_label = ttnn.TensorTopology(
        ttnn.MeshShape(1, 2), [ttnn.PlacementReplicate(), ttnn.PlacementShard(1)], mesh_coords
    )
    assert tensor.tensor_topology() == true_label, "test bug: the mapper's label is not the one spelled out here"
    tensor.update_tensor_topology(
        ttnn.TensorTopology(ttnn.MeshShape(1, 2), [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], mesh_coords)
    )
    return tensor, true_label, shards


def _load_shards(file_name):
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(ttnn.load_tensor(file_name))]


def test_dump_mislabelled_replicate_raises(tmp_path, expect_error):
    """Negative control for the write-side replica check: before it, this dump succeeded and the loaded
    tensor's second shard read back as a copy of the first. The tensor dumps once relabelled correctly.
    """
    tensor, true_label, shards = _mislabelled_replicate_tensor()
    file_name = tmp_path / "mislabelled.tensorbin"

    with expect_error(RuntimeError, "contents differ"):
        ttnn.dump_tensor(file_name, tensor, mode=ttnn.DumpTensorMode.LOCAL)
    assert not file_name.exists(), "a rejected dump must not leave a file behind"

    tensor.update_tensor_topology(true_label)
    ttnn.dump_tensor(file_name, tensor, mode=ttnn.DumpTensorMode.LOCAL)
    torch.testing.assert_close(torch.cat(_load_shards(file_name), dim=1), shards, rtol=0, atol=0)


def test_dump_mislabelled_replicate_config_opt_out(tmp_path):
    """With `verify_replicated_shards_on_dump` off, the label is trusted as it was before the check: one
    copy, taken from the first replica, stands for the whole group."""
    # The property is generated from `Config::attributes_t` by `reflect::for_each` in ttnn-nanobind/core.cpp.
    assert hasattr(ttnn.CONFIG, "verify_replicated_shards_on_dump")
    tensor, _, shards = _mislabelled_replicate_tensor()
    file_name = tmp_path / "opted_out.tensorbin"

    previous = ttnn.CONFIG.verify_replicated_shards_on_dump
    ttnn.CONFIG.verify_replicated_shards_on_dump = False
    try:
        ttnn.dump_tensor(file_name, tensor, mode=ttnn.DumpTensorMode.LOCAL)
    finally:
        ttnn.CONFIG.verify_replicated_shards_on_dump = previous

    loaded = _load_shards(file_name)
    assert len(loaded) == 2
    first_replica = shards[:, :32]
    torch.testing.assert_close(loaded[0], first_replica, rtol=0, atol=0)
    torch.testing.assert_close(loaded[1], first_replica, rtol=0, atol=0)


# Field indices from tensor.fbs (root `Tensor` table) and tensor_topology.fbs (`TensorTopology` table).
_TENSOR_TOPOLOGY_FIELD = 3
_SCHEMA_VERSION_FIELD = 4
_TOPOLOGY_PLACEMENTS_FIELD = 1
_TOPOLOGY_MESH_COORDS_FIELD = 2
_FLATBUFFER_START = 8  # The uint64 header size precedes the flatbuffer.


class _TensorFile:
    """Byte-level view of a `.tensorbin` file for the schema tests.

    Fields are resolved through the table's vtable, which is how a reader finds them, so a patch here changes
    exactly what the reader sees. Clearing a vtable slot makes the field absent, and the reader gets its default.
    """

    def __init__(self, path):
        self.path = path
        self.data = bytearray(path.read_bytes())

    def u32(self, pos):
        return struct.unpack_from("<I", self.data, pos)[0]

    def set_u32(self, pos, value):
        struct.pack_into("<I", self.data, pos, value)

    def follow(self, pos):
        """Position that the uoffset stored at `pos` points to."""
        return pos + self.u32(pos)

    def root(self):
        return self.follow(_FLATBUFFER_START)

    def topology(self):
        return self.follow(self.field(self.root(), _TENSOR_TOPOLOGY_FIELD))

    def _vtable_slot(self, table, index):
        vtable = table - struct.unpack_from("<i", self.data, table)[0]
        return vtable + 4 + index * 2

    def field(self, table, index):
        offset = struct.unpack_from("<H", self.data, self._vtable_slot(table, index))[0]
        assert offset != 0, f"field {index} is absent"
        return table + offset

    def drop_field(self, table, index):
        struct.pack_into("<H", self.data, self._vtable_slot(table, index), 0)

    def table_element(self, vector_field, i):
        """Position of element `i` of the vector of tables whose uoffset is stored at `vector_field`."""
        vector = self.follow(vector_field)
        return self.follow(vector + 4 + i * 4)

    def write(self):
        self.path.write_bytes(self.data)


def _dump_two_shard_file(file_name):
    """Dumps a 1x2 tensor whose shards hold 11s at (0,0) and 29s at (0,1) under a `[Replicate, Shard(1)]` label and
    returns the torch source."""
    shards = torch.cat([torch.full((32, 32), value, dtype=torch.bfloat16) for value in (11, 29)], dim=1)
    mapper = ttnn.create_mesh_mapper(
        ttnn.MeshShape(1, 2),
        ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(1)]),
    )
    tensor = ttnn.from_torch(shards, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
    ttnn.dump_tensor(file_name, tensor, mode=ttnn.DumpTensorMode.LOCAL)
    return shards


def _fully_replicated_label(rows, cols):
    """The label a file without a topology loads with: a 1-D distribution shape over every coordinate of the storage
    mesh, as `TensorTopology::create_fully_replicated_tensor_topology` builds it."""
    coords = [ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)]
    return ttnn.TensorTopology(ttnn.MeshShape([rows * cols]), [ttnn.PlacementReplicate()], coords)


def test_file_has_identifier_and_schema_version(tmp_path):
    """Files open with the "TTNB" identifier at bytes [12, 16) and record schema version 1 in the root table.
    Negative control: before this change bytes [12, 16) were table data and the version field did not exist."""
    file_name = tmp_path / "versioned.tensorbin"
    _dump_two_shard_file(file_name)
    file = _TensorFile(file_name)
    assert bytes(file.data[12:16]) == b"TTNB"
    assert file.u32(file.field(file.root(), _SCHEMA_VERSION_FIELD)) == 1


@pytest.mark.parametrize("num_shards", [1, 2])
def test_load_legacy_topology_less_file(tmp_path, num_shards):
    """A file from before the topology field existed (both the `tensor_topology` and the `schema_version` vtable
    slots cleared) still loads, labelled fully replicated as it always was, with every shard intact. With one shard
    that label is correct; with two distinct shards the placement is unknowable and the loader logs a warning
    (through tt-logger, so this test asserts the label and the data rather than the log line). Negative control:
    the refusal the plan first proposed would have raised on the two-shard file, and before this change the same
    file loaded with the same label and no warning."""
    file_name = tmp_path / "legacy.tensorbin"
    if num_shards == 1:
        expected = torch.full((32, 32), 11, dtype=torch.bfloat16)
        tensor = ttnn.from_torch(expected, layout=ttnn.TILE_LAYOUT)
        ttnn.dump_tensor(file_name, tensor, mode=ttnn.DumpTensorMode.LOCAL)
        mesh_shape = (1, 1)
    else:
        expected = _dump_two_shard_file(file_name)
        mesh_shape = (1, 2)
    file = _TensorFile(file_name)
    file.drop_field(file.root(), _TENSOR_TOPOLOGY_FIELD)
    file.drop_field(file.root(), _SCHEMA_VERSION_FIELD)
    file.write()

    loaded = ttnn.load_tensor(file_name)

    assert loaded.tensor_topology() == _fully_replicated_label(*mesh_shape)
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(loaded)]
    assert len(shards) == num_shards
    torch.testing.assert_close(torch.cat(shards, dim=1), expected, rtol=0, atol=0)


def test_load_versioned_file_without_topology_raises(tmp_path, expect_error):
    """A file that records schema version 1 but no topology is corrupt: every versioned writer records one.
    Negative control: before this change the version field did not exist, so the same patch produced a legacy file
    that loaded silently as fully replicated."""
    file_name = tmp_path / "versioned_without_topology.tensorbin"
    _dump_two_shard_file(file_name)
    file = _TensorFile(file_name)
    file.drop_field(file.root(), _TENSOR_TOPOLOGY_FIELD)
    file.write()

    with expect_error(RuntimeError, "no tensor topology"):
        ttnn.load_tensor(file_name)


def test_load_newer_schema_version_raises(tmp_path, expect_error):
    """A file written by a newer tt-metal (schema version 2) is refused rather than misread; with the version
    restored it loads again. Negative control: before this change there was no version to refuse on."""
    file_name = tmp_path / "newer_version.tensorbin"
    shards = _dump_two_shard_file(file_name)
    file = _TensorFile(file_name)
    version = file.field(file.root(), _SCHEMA_VERSION_FIELD)
    file.set_u32(version, 2)
    file.write()

    with expect_error(RuntimeError, "schema version 2"):
        ttnn.load_tensor(file_name)

    file.set_u32(version, 1)
    file.write()
    torch.testing.assert_close(torch.cat(_load_shards(file_name), dim=1), shards, rtol=0, atol=0)


def test_load_topology_mesh_coords_mismatch_raises(tmp_path, expect_error):
    """The topology's coordinate list is cut to one entry for a two-position distribution shape. Negative control:
    `TensorTopology` accepts that, so before this change the file loaded and the first label-driven lookup indexed
    past the list."""
    file_name = tmp_path / "mesh_coords_mismatch.tensorbin"
    _dump_two_shard_file(file_name)
    file = _TensorFile(file_name)
    coords_vector = file.follow(file.field(file.topology(), _TOPOLOGY_MESH_COORDS_FIELD))
    assert file.u32(coords_vector) == 2
    file.set_u32(coords_vector, 1)
    file.write()

    with expect_error(RuntimeError, "lists 1 mesh coordinate"):
        ttnn.load_tensor(file_name)


def test_load_inconsistent_replica_group_raises(tmp_path, expect_error):
    """The Shard(1) placement is flipped to Replicate, so the label calls (0,0) and (0,1) replicas while their
    records point at different data. Negative control: before this change the file loaded under that label, and
    re-dumping it failed the replica byte check (or, before that check existed, dropped the (0,1) shard)."""
    file_name = tmp_path / "inconsistent_group.tensorbin"
    _dump_two_shard_file(file_name)
    file = _TensorFile(file_name)
    shard_placement = file.table_element(file.field(file.topology(), _TOPOLOGY_PLACEMENTS_FIELD), 1)
    placement_type = file.field(shard_placement, 0)
    assert file.data[placement_type] == 1  # MeshMapperPlacementType.Shard
    file.data[placement_type] = 0  # MeshMapperPlacementType.Replicate
    file.write()

    with expect_error(RuntimeError, "different data"):
        ttnn.load_tensor(file_name)


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
