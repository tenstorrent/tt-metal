# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks of the Python bindings for strided KV chunk table authoring (no device needed)."""

import pytest

import ttnn

D = ttnn.experimental.disaggregation


def _cfg(num_layers=2, max_seq=1024, num_slots=1):
    c = D.KvChunkAddressTableConfig()
    c.num_layers = num_layers
    c.max_sequence_length = max_seq
    c.num_slots = num_slots
    c.chunk_n_tokens = 32
    c.chunk_size_bytes = 64
    return c


def _strided_table(names=("a", "b"), strided=("a",)):
    return D.KvChunkAddressTable({n: _cfg() for n in names}, {n: D.ChunkCompression.STRIDED_ROWS for n in strided})


def _group(table, chip=0):
    return table.add_device_group([ttnn.FabricNodeId(ttnn.MeshId(0), chip)])


@pytest.mark.parametrize("as_objects", [True, False], ids=["DeviceGroupIndex", "int"])
def test_set_strided_row_lookup(as_objects):
    t = _strided_table()
    g0, g1 = _group(t, 0), _group(t, 1)
    groups = [g0, g1] if as_objects else [int(g0), int(g1)]
    t.set_strided_row(1, 0, 64, [0x1000, 0x2000], [0x100, 0x200], groups, t.config_id_of("a"))
    assert t.compression(t.config_id_of("a")) == D.ChunkCompression.STRIDED_ROWS
    assert t.compression(t.config_id_of("b")) == D.ChunkCompression.UNROLLED
    for c in range(8):
        loc = t.lookup(1, c * 32, 0, t.config_id_of("a"))
        assert loc.noc_addr == (0x2000 + (c // 2) * 0x200 if c % 2 else 0x1000 + (c // 2) * 0x100)
        assert loc.size_bytes == 64
        assert int(loc.device_group_index) == int(g1 if c % 2 else g0)
    assert t.lookup(0, 0, 0, t.config_id_of("a")).size_bytes == 0  # other rows unset


def test_set_strided_row_rejects_bad_input(expect_error):
    t = _strided_table()
    g = int(_group(t))
    a = t.config_id_of("a")
    with expect_error(RuntimeError, "size_bytes must be > 0"):
        t.set_strided_row(0, 0, 0, [0x1000], [0x100], [g], a)  # zero-size chunks
    with expect_error(RuntimeError, "wraps"):
        t.set_strided_row(0, 0, 64, [0x100], [-0x100], [g], a)  # last address wraps below 0
    with expect_error(RuntimeError, "on an UNROLLED config"):
        t.set_strided_row(0, 0, 64, [0x1000], [0x100], [g], t.config_id_of("b"))  # UNROLLED config
    with expect_error(RuntimeError, "compression given for unknown config"):
        D.KvChunkAddressTable({"a": _cfg()}, {"nope": D.ChunkCompression.STRIDED_ROWS})  # unknown config name


def test_strided_tables_round_trip_including_an_empty_strided_config(tmp_path):
    t = _strided_table(strided=("a", "b"))  # "b" stays empty
    g = _group(t)
    t.set_strided_row(0, 0, 64, [0x1000, 0x2000], [0x100, 0x200], [g, g], t.config_id_of("a"))
    path = tmp_path / "t.pb"
    D.export_to_protobuf_file(t, str(path))
    r = D.import_from_protobuf_file(str(path))
    assert r.compression(r.config_id_of("a")) == D.ChunkCompression.STRIDED_ROWS
    assert r.compression(r.config_id_of("b")) == D.ChunkCompression.UNROLLED
    for c in range(32):
        x, y = t.lookup(0, c * 32, 0, t.config_id_of("a")), r.lookup(0, c * 32, 0, r.config_id_of("a"))
        assert (x.noc_addr, x.size_bytes, int(x.device_group_index)) == (
            y.noc_addr,
            y.size_bytes,
            int(y.device_group_index),
        )
