# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import ttnn

pytestmark = pytest.mark.skipif(os.environ.get("ARCH_NAME") != "blackhole", reason="Blackhole-only")

disagg = ttnn.experimental.disaggregation
LAYERS, SEQ, SLOTS, CHUNK_TOKENS, CHUNK_BYTES = 2, 256, 2, 32, 1088


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_METAL_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache" / "tt-metal-cache" / "kv-chunk-tables"


def make_table(base):
    cfg = disagg.KvChunkAddressTableConfig()
    cfg.num_layers, cfg.max_sequence_length, cfg.num_slots = LAYERS, SEQ, SLOTS
    cfg.chunk_n_tokens, cfg.chunk_size_bytes = CHUNK_TOKENS, CHUNK_BYTES
    table = disagg.KvChunkAddressTable(cfg)
    group = table.add_device_group([ttnn.FabricNodeId(ttnn.MeshId(1), 0), ttnn.FabricNodeId(ttnn.MeshId(1), 1)])
    for slot in range(SLOTS):
        for layer in range(LAYERS):
            for pos in range(0, SEQ, CHUNK_TOKENS):
                loc = disagg.KvCacheLocation()
                loc.noc_addr = (3 << 32) | (base + ((slot * LAYERS + layer) * SEQ + pos) // CHUNK_TOKENS * CHUNK_BYTES)
                loc.size_bytes = CHUNK_BYTES
                loc.device_group_index = group
                table.set(layer, pos, slot, loc)
    return table


def locations(path):
    table = disagg.import_from_protobuf_file(str(path))
    return [
        (loc.noc_addr, loc.size_bytes, int(loc.device_group_index))
        for slot in range(SLOTS)
        for layer in range(LAYERS)
        for pos in range(0, SEQ, CHUNK_TOKENS)
        for loc in [table.lookup(layer, pos, slot)]
    ]


class Builder:
    def __init__(self, base):
        self.base, self.calls = base, 0

    def __call__(self):
        self.calls += 1
        return make_table(self.base)


def test_relaunch_hits_and_matches_a_fresh_table(cache_dir, tmp_path):
    first, second = Builder(0x1000), Builder(0x9000)

    assert disagg.get_or_build_kv_chunk_table("abc1234", {"b": 1, "a": [1, 2]}, first, str(tmp_path / "l0.pb")) is False
    assert disagg.get_or_build_kv_chunk_table("abc1234", {"a": [1, 2], "b": 1}, second, str(tmp_path / "l1.pb")) is True

    assert (first.calls, second.calls) == (1, 0)
    assert (tmp_path / "l1.pb").read_bytes() == (tmp_path / "l0.pb").read_bytes()
    disagg.export_to_protobuf_file(make_table(0x1000), str(tmp_path / "fresh.pb"))
    assert locations(tmp_path / "l1.pb") == locations(tmp_path / "fresh.pb")
    assert len(list(cache_dir.glob("*.pb"))) == 1


def test_changed_key_or_seed_misses(cache_dir, tmp_path):
    build = Builder(0x1000)
    disagg.get_or_build_kv_chunk_table("abc1234", {"base": 1}, build, str(tmp_path / "a.pb"))

    assert disagg.get_or_build_kv_chunk_table("abc1234", {"base": 2}, build, str(tmp_path / "b.pb")) is False
    assert disagg.get_or_build_kv_chunk_table("abc1235", {"base": 1}, build, str(tmp_path / "c.pb")) is False
    assert build.calls == 3
    assert len(list(cache_dir.glob("*.pb"))) == 3


@pytest.mark.parametrize("key", [None, {"base": 1}])
def test_empty_seed_always_builds_and_never_caches(cache_dir, tmp_path, key):
    build = Builder(0x1000)

    assert disagg.get_or_build_kv_chunk_table("", key, build, str(tmp_path / "a.pb")) is False
    assert disagg.get_or_build_kv_chunk_table("", key, build, str(tmp_path / "b.pb")) is False

    assert build.calls == 2
    assert locations(tmp_path / "b.pb") == locations(tmp_path / "a.pb")
    assert not cache_dir.exists()


def test_failing_build_raises_and_leaves_no_files(cache_dir, tmp_path, expect_error):
    def boom():
        raise ValueError("build failed")

    with expect_error(ValueError, "build failed"):
        disagg.get_or_build_kv_chunk_table("abc1234", {"base": 1}, boom, str(tmp_path / "a.pb"))

    assert list(tmp_path.glob("a.pb*")) == []
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_unserializable_key_is_rejected_before_building(cache_dir, tmp_path, expect_error):
    build = Builder(0x1000)

    with expect_error(TypeError, "not JSON serializable"):
        disagg.get_or_build_kv_chunk_table("abc1234", {"x": object()}, build, str(tmp_path / "a.pb"))

    assert build.calls == 0
