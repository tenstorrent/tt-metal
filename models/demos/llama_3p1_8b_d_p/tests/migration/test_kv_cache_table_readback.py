# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Independent live-tensor oracle for the Llama prefill KV address table.

The packed-copy migration gate intentionally uses the table for both source and
destination.  This test closes the remaining address-oracle gap: it writes the
real production cache, slices the live TTNN tensor by logical SP/TP ownership,
and compares that independent slice with bytes read through the exported table.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.llama_3p1_8b_d_p.tests.migration.kv_table_oracle import (
    assert_independent_page,
    device_major_positions,
    independent_tensor_location,
    logical_tag,
    tagged_page,
    tagged_values,
)
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache, write_kv_chunk
from models.demos.llama_3p1_8b_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table

MESH_SHAPE = (4, 8)
SP, TP = MESH_SHAPE
NUM_SLOTS = 2
NUM_LAYERS = 32
NUM_KV_HEADS = 8
MAX_SEQ_LEN = 2048
GLOBAL_CHUNK = 1024
PAGE_TOKENS = 32
HEAD_DIM = 128
PAGE_BYTES = 4 * 1088
CONFIG_NAMES = tuple(f"{kind}_h{head}" for kind in ("k", "v") for head in range(NUM_KV_HEADS))
EXPECTED_PAGES = len(CONFIG_NAMES) * NUM_SLOTS * NUM_LAYERS * (MAX_SEQ_LEN // PAGE_TOKENS)
BOUNDARY_PAGES = {0, 224, 256, 992, 1024, 2016}


def _to_chunk(mesh_device, values):
    return ttnn.from_torch(
        values.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1)),
    )


def _read_table_page(table, *, layer, position, slot, config_id, decode):
    raw = table.read_device_chunk(layer=layer, position=position, slot=slot, config_id=config_id)
    byte_count = len(raw)
    assert byte_count == PAGE_BYTES
    return decode(raw), byte_count


def _noc_bank_id(noc_addr):
    packed = int(noc_addr)
    assert packed >= 0
    return packed >> 32


def _write_tagged_cache(mesh_device, cache):
    for slot in range(NUM_SLOTS):
        for layer in range(NUM_LAYERS):
            for start in range(0, MAX_SEQ_LEN, GLOBAL_CHUNK):
                positions = device_major_positions(start)
                tt_k = _to_chunk(mesh_device, tagged_values("k", slot, layer, positions))
                tt_v = _to_chunk(mesh_device, tagged_values("v", slot, layer, positions))
                try:
                    write_kv_chunk(
                        cache,
                        tt_k,
                        tt_v,
                        slot_idx=slot,
                        layer_idx=layer,
                        actual_start=start,
                        actual_end=start + GLOBAL_CHUNK,
                    )
                finally:
                    tt_k.deallocate(True)
                    tt_v.deallocate(True)
    ttnn.synchronize_device(mesh_device)


# The tag packs slot/layer/head/position without collisions and emits only BF16/BFP8-exact values.
def test_host_tagged_fixture_is_unique_and_bfp8_exact():
    tags = {
        logical_tag(slot, layer, head, position)
        for slot in range(NUM_SLOTS)
        for layer in range(NUM_LAYERS)
        for head in range(NUM_KV_HEADS)
        for position in range(MAX_SEQ_LEN)
    }
    assert len(tags) == NUM_SLOTS * NUM_LAYERS * NUM_KV_HEADS * MAX_SEQ_LEN

    for kind in ("k", "v"):
        for slot in range(NUM_SLOTS):
            for layer in (0, NUM_LAYERS - 1):
                for head in range(NUM_KV_HEADS):
                    for position in BOUNDARY_PAGES:
                        page = tagged_page(kind, slot, layer, head, position)
                        assert set(torch.unique(page).tolist()) <= ({32.0, 64.0} if kind == "k" else {-64.0, -32.0})
                        assert torch.equal(page, page.to(torch.bfloat16).float())


@pytest.mark.parametrize(
    "label,expected_key,wrong_key",
    [
        ("slot", ("k", 1, 31, 7, 2016), ("k", 0, 31, 7, 2016)),
        ("head", ("v", 0, 17, 6, 992), ("v", 0, 17, 7, 992)),
        ("page", ("k", 1, 3, 2, 256), ("k", 1, 3, 2, 288)),
    ],
)
# Independent CPU pages must reject each table-index corruption that M03 is meant to expose.
def test_host_wrong_table_mapping_is_detected(label, expected_key, wrong_key, expect_error):
    expected = tagged_page(*expected_key)
    wrong = tagged_page(*wrong_key)
    with expect_error(AssertionError, "independent tensor oracle"):
        assert_independent_page(wrong, expected, f"wrong {label} mapping")


# Raw-page byte accounting is completed inside the helper, before its temporary bytes leave scope.
def test_host_read_table_page_preserves_size_and_decode_order():
    payload = bytes((index % 251 for index in range(PAGE_BYTES)))

    class FakeTable:
        def read_device_chunk(self, **address):
            assert address == {"layer": 31, "position": 2016, "slot": 1, "config_id": 15}
            return payload

    decode_calls = []

    def decode(raw):
        decode_calls.append(raw is payload)
        return "decoded-page"

    decoded, byte_count = _read_table_page(FakeTable(), layer=31, position=2016, slot=1, config_id=15, decode=decode)
    assert decoded == "decoded-page"
    assert byte_count == PAGE_BYTES
    assert decode_calls == [True]


# The accepted native table preserves noc_addr as an integer with the DRAM bank in its high 32 bits.
def test_host_native_table_noc_addr_preserves_bank_bits():
    api = ttnn.experimental.disaggregation
    config = api.KvChunkAddressTableConfig()
    config.num_layers = 1
    config.max_sequence_length = PAGE_TOKENS
    config.num_slots = 1
    config.chunk_n_tokens = PAGE_TOKENS
    config.chunk_size_bytes = PAGE_BYTES
    table = api.KvChunkAddressTable({"k_h0": config})
    location = api.KvCacheLocation()
    location.noc_addr = (7 << 32) | 0x12345678
    location.size_bytes = PAGE_BYTES
    location.device_group_index = api.DeviceGroupIndex(0)
    table.set(0, 0, 0, location, 0)

    restored = table.lookup(0, 0, 0, config_id=0)
    assert type(restored.noc_addr) is int
    assert int(restored.noc_addr) == (7 << 32) | 0x12345678
    assert _noc_bank_id(restored.noc_addr) == 7


@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="Llama migration KV table targets a Blackhole Galaxy")
@pytest.mark.timeout(0)
# Every exported address is checked against both its independent live-tensor slice and its CPU tag.
def test_llama_kv_table_readback_matches_live_tensor(mesh_device, device_params):
    cache = allocate_kv_cache(
        mesh_device,
        MeshConfig(MESH_SHAPE, TP),
        num_users=NUM_SLOTS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        cache_dtype=ttnn.bfloat8_b,
    )
    try:
        _write_tagged_cache(mesh_device, cache)
        table = build_kv_chunk_address_table(mesh_device=mesh_device, kv_cache=cache, chunk_size=GLOBAL_CHUNK)

        assert table.num_configs() == len(CONFIG_NAMES)
        assert tuple(table.config_name(index) for index in range(table.num_configs())) == CONFIG_NAMES
        assert table.total_entries() == EXPECTED_PAGES
        for config_id in range(table.num_configs()):
            config = table.config(config_id)
            assert config.num_layers == NUM_LAYERS
            assert config.max_sequence_length == MAX_SEQ_LEN
            assert config.num_slots == NUM_SLOTS
            assert config.chunk_n_tokens == PAGE_TOKENS
            assert config.chunk_size_bytes == PAGE_BYTES

        comparisons = 0
        raw_bytes_read = 0
        boundaries_seen = set()
        banks_seen = set()
        for config_id, config_name in enumerate(CONFIG_NAMES):
            kind, head_text = config_name.split("_h")
            head = int(head_text)
            cache_tensor = cache.k if kind == "k" else cache.v
            shards = ttnn.get_device_tensors(cache_tensor)
            assert len(shards) == SP * TP
            for sp_row in range(SP):
                live = ttnn.to_torch(shards[sp_row * TP + head]).to(torch.bfloat16)
                for slot in range(NUM_SLOTS):
                    for layer in range(NUM_LAYERS):
                        batch = slot * NUM_LAYERS + layer
                        for position in range(0, MAX_SEQ_LEN, PAGE_TOKENS):
                            owner, local_position = independent_tensor_location(position)
                            if owner != sp_row:
                                continue
                            tensor_page = live[
                                batch : batch + 1,
                                :1,
                                local_position : local_position + PAGE_TOKENS,
                                :HEAD_DIM,
                            ].reshape(1, 1, PAGE_TOKENS, HEAD_DIM)
                            cpu_page = tagged_page(kind, slot, layer, head, position).to(torch.bfloat16)
                            assert_independent_page(
                                tensor_page, cpu_page, f"live tensor {config_name}/{slot}/{layer}/{position}"
                            )

                            location = table.lookup(layer, position, slot, config_id=config_id)
                            assert location.size_bytes == PAGE_BYTES
                            assert location.noc_addr != 0
                            banks_seen.add(_noc_bank_id(location.noc_addr))
                            table_page, page_bytes = _read_table_page(
                                table,
                                layer=layer,
                                position=position,
                                slot=slot,
                                config_id=config_id,
                                decode=lambda raw: ttnn.to_torch(
                                    ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(
                                        raw, [1, 1, PAGE_TOKENS, HEAD_DIM]
                                    )
                                ).to(torch.bfloat16),
                            )
                            assert_independent_page(
                                table_page,
                                tensor_page,
                                f"table address {config_name}/{slot}/{layer}/{position}",
                            )
                            comparisons += 1
                            raw_bytes_read += page_bytes
                            del table_page
                            if position in BOUNDARY_PAGES:
                                boundaries_seen.add((config_id, sp_row, position))
                del live

        assert comparisons == EXPECTED_PAGES
        assert raw_bytes_read == EXPECTED_PAGES * PAGE_BYTES
        assert banks_seen == set(range(mesh_device.dram_grid_size().x))
        assert boundaries_seen == {
            (config_id, independent_tensor_location(position)[0], position)
            for config_id in range(len(CONFIG_NAMES))
            for position in BOUNDARY_PAGES
        }
        logger.info(
            "Llama KV table independent readback OK: comparisons={}, raw_bytes={}, configs={}, slots={}, "
            "layers={}, sequence={}, banks={}, boundary_checks={}",
            comparisons,
            raw_bytes_read,
            len(CONFIG_NAMES),
            NUM_SLOTS,
            NUM_LAYERS,
            MAX_SEQ_LEN,
            sorted(banks_seen),
            len(boundaries_seen),
        )
    finally:
        cache.k.deallocate(True)
        cache.v.deallocate(True)
