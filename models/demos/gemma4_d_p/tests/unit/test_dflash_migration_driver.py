# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The published drafter rows must be requested, acknowledged and byte-checked."""

import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import ttnn
from models.demos.common.prefill.runners import migration_driver, prefill_producer
from models.demos.gemma4_d_p.tests.unit.test_dflash import config
from models.demos.gemma4_d_p.tt.attention.ring_prefill import GlobalRingKVCache, SlidingRingKVCache
from models.demos.gemma4_d_p.tt.dflash import DFlashKVCache
from models.demos.gemma4_d_p.tt.runners.adapters import gemma4
from models.demos.gemma4_d_p.tt.runners.kv_caches import Gemma4KvCaches
from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table


@pytest.fixture
def migration_tables(monkeypatch):
    def tensor(address):
        return SimpleNamespace(dtype=ttnn.bfloat8_b, buffer_address=lambda: address)

    caches = Gemma4KvCaches(
        layers=[
            (
                GlobalRingKVCache(tensor(0x100000 + layer * 0x10000))
                if layer % 6 == 5
                else SlidingRingKVCache(tensor(0x100000 + layer * 0x10000), tensor(0x800000 + layer * 0x10000))
            )
            for layer in range(60)
        ],
        layer_types=tuple("full_attention" if layer % 6 == 5 else "sliding_attention" for layer in range(60)),
        num_users=2,
        max_seq_len=256,
        cp=8,
        tp=4,
        dflash=DFlashKVCache(tensor(0x1000000), tensor(0x2000000), config(), 2, 256),
    )
    mesh = SimpleNamespace(
        shape=(8, 4),
        dram_grid_size=lambda: SimpleNamespace(x=8),
        get_fabric_node_id=lambda xy: ttnn.FabricNodeId(ttnn.MeshId(0), xy[0] * 4 + xy[1]),
    )
    monkeypatch.setattr(gemma4, "load_dflash_config", lambda _: config())
    monkeypatch.setattr(prefill_producer, "ADAPTER", gemma4.Gemma4PrefillAdapter())
    monkeypatch.setattr(prefill_producer, "NUM_LAYERS", 60)
    monkeypatch.setattr(prefill_producer, "NUM_ACK_LAYERS", 60)
    monkeypatch.setattr(prefill_producer, "MTP_DECODE_LEVELS", 0)
    return {
        enabled: build_kv_chunk_address_table(
            mesh_device=mesh, kv_caches=caches if enabled else replace(caches, dflash=None), chunk_size=256
        )
        for enabled in (False, True)
    }


@pytest.fixture
def driver(monkeypatch, tmp_path):
    for key in os.environ:
        if key.startswith("PREFILL_MIGRATION_") or key == "PREFILL_DFLASH":
            monkeypatch.delenv(key)
    monkeypatch.setenv("MIGRATION_DONE_FILE", str(tmp_path / "done"))
    result = migration_driver.MigrationDriver(chunk_size=256, num_layers=60, default_dst_slot_offset=1)
    result.client = Mock()
    return result


@pytest.mark.parametrize("enabled", [False, True])
def test_driver_requests_every_published_layer_and_preserves_target_only(driver, migration_tables, enabled):
    table = migration_tables[enabled]
    stats = SimpleNamespace(resident={0: SimpleNamespace(real_len=224)})
    assert driver.run(stats, num_slots=2, kv_table=table) == [(0, 1, 224)]
    request = driver.client.migrate.call_args.kwargs
    assert (request["layer_start"], request["layer_end_exclusive"]) == (0, 65 if enabled else 60)
    assert (request["src_slot"], request["dst_slot"], request["pos_end_exclusive"]) == (0, 1, 224)
    driver.client.wait_complete.assert_called_once()
    assert prefill_producer._completion_layers_per_chunk(table) == (65 if enabled else 60)


def test_explicit_draft_layer_subset_remains_explicit(driver, migration_tables):
    driver.layers = [60, 64]
    driver.run(
        SimpleNamespace(resident={0: SimpleNamespace(real_len=256)}), num_slots=2, kv_table=migration_tables[True]
    )
    assert [
        (call.kwargs["layer_start"], call.kwargs["layer_end_exclusive"])
        for call in driver.client.migrate.call_args_list
    ] == [(60, 61), (64, 65)]


@pytest.mark.parametrize("table_present", [False, True])
def test_enabled_drafter_cannot_silently_migrate_target_only(
    driver, migration_tables, monkeypatch, table_present, expect_error
):
    monkeypatch.setenv("PREFILL_DFLASH", "1")
    with expect_error(RuntimeError, "drafter configs"):
        driver.run(SimpleNamespace(), kv_table=migration_tables[False] if table_present else None)
    driver.client.migrate.assert_not_called()


def test_no_table_keeps_legacy_target_migration(driver):
    driver.run(SimpleNamespace(resident={0: SimpleNamespace(real_len=256)}), num_slots=2)
    assert driver.client.migrate.call_args.kwargs["layer_end_exclusive"] == 60


def test_drafter_byte_verification_covers_every_head_and_detects_corruption(migration_tables, monkeypatch):
    table = migration_tables[True]
    memory = {}
    monkeypatch.setattr(prefill_producer, "_resolve_unique_id", lambda nodes, _: int(nodes[0].chip_id))
    for index in range(36, 52):
        for layer in range(60, 65):
            for position in range(0, 256, 32):
                for slot in (0, 1):
                    loc = table.lookup(layer, position, slot, index)
                    node = table.get_device_group(loc.device_group_index).fabric_node_ids[0]
                    memory[int(node.chip_id), loc.noc_addr] = bytes([index, layer, position // 32])
    reads = []

    def read(chip, address, size):
        reads.append((chip, address))
        return memory[chip, address]

    monkeypatch.setattr(ttnn.experimental.disaggregation, "read_dram_umd", read)
    assert migration_driver._verify_dst_vs_src_bytes(table, {}, [(0, 1, 256)], list(range(60, 65)))
    assert len(set(reads)) == 16 * 5 * 8 * 2
    last = table.lookup(64, 224, 1, "dflash_v_h07")
    chip = int(table.get_device_group(last.device_group_index).fabric_node_ids[0].chip_id)
    memory[chip, last.noc_addr] = b"corrupted"
    assert not migration_driver._verify_dst_vs_src_bytes(table, {}, [(0, 1, 256)], list(range(60, 65)))
