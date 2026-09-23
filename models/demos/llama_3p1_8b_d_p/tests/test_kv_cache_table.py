# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama SP4/TP8 KV-cache address-table integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from ttnn.device import is_blackhole

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tests.kv_table_test_utils import (
    GLOBAL_CHUNK,
    HEAD_DIM,
    MAX_SEQ_LEN,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_SLOTS,
    PAGE_TOKENS,
    assert_page_equal,
    device_major_positions,
    independent_tensor_location,
    logical_tag,
    tagged_page,
    tagged_values,
)
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache, write_kv_chunk
from models.demos.llama_3p1_8b_d_p.tt.model_config import resolve_weights_path
from models.demos.llama_3p1_8b_d_p.tt.qkv import QKVProjection
from models.demos.llama_3p1_8b_d_p.tt.rope import apply_indexed_rope, build_indexed_rope, build_transformation_mat
from models.demos.llama_3p1_8b_d_p.tt.runners.kv_chunk_table import (
    build_and_serialize_kv_chunk_table,
    build_kv_chunk_address_table,
)

MESH_SHAPE = (4, 8)
TP = MESH_SHAPE[1]
PAGE_BYTES = 4 * 1088
CONFIG_NAMES = tuple(f"{kind}_h{head}" for kind in ("k", "v") for head in range(NUM_KV_HEADS))
EXPECTED_PAGES = len(CONFIG_NAMES) * NUM_SLOTS * NUM_LAYERS * (MAX_SEQ_LEN // PAGE_TOKENS)
BOUNDARY_POSITIONS = (0, 224, 256, 992, 1024, 2016)
HF_MODEL = Path(resolve_weights_path())
QKV_WEIGHT_NAMES = {
    "q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
    "k_proj.weight": "model.layers.0.self_attn.k_proj.weight",
    "v_proj.weight": "model.layers.0.self_attn.v_proj.weight",
}


def _to_chunk(mesh_device, values):
    return ttnn.from_torch(
        values.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1)),
    )


def _decode_page(raw):
    assert len(raw) == PAGE_BYTES
    return ttnn.to_torch(
        ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw, [1, 1, PAGE_TOKENS, HEAD_DIM])
    ).to(torch.bfloat16)


def _snapshot_shards(cache_tensor):
    """Copy each cache shard once; page loops must not issue repeated full-shard reads."""
    return [ttnn.to_torch(shard).to(torch.bfloat16) for shard in ttnn.get_device_tensors(cache_tensor)]


def _snapshot_page(shards, *, head, slot, layer, position):
    sp_row, local_position = independent_tensor_location(position)
    live = shards[sp_row * TP + head]
    batch = slot * NUM_LAYERS + layer
    return live[batch : batch + 1, :1, local_position : local_position + PAGE_TOKENS, :HEAD_DIM]


def _write_synthetic_cache(mesh_device, cache):
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


def _assert_table_geometry(table):
    assert table.num_configs() == len(CONFIG_NAMES)
    assert tuple(table.config_name(config_id) for config_id in range(table.num_configs())) == CONFIG_NAMES
    assert table.total_entries() == EXPECTED_PAGES
    for config_id in range(table.num_configs()):
        config = table.config(config_id)
        assert (
            config.num_layers,
            config.max_sequence_length,
            config.num_slots,
            config.chunk_n_tokens,
            config.chunk_size_bytes,
        ) == (NUM_LAYERS, MAX_SEQ_LEN, NUM_SLOTS, PAGE_TOKENS, PAGE_BYTES)


def _assert_tables_equivalent(expected, actual):
    _assert_table_geometry(expected)
    _assert_table_geometry(actual)
    assert actual.num_device_groups() == expected.num_device_groups() == 32
    for group_id in range(expected.num_device_groups()):
        expected_group = expected.get_device_group(ttnn.experimental.disaggregation.DeviceGroupIndex(group_id))
        actual_group = actual.get_device_group(ttnn.experimental.disaggregation.DeviceGroupIndex(group_id))
        assert actual_group.fabric_node_ids == expected_group.fabric_node_ids
        for node in expected_group.fabric_node_ids:
            assert expected.has_host(node)
            assert actual.has_host(node)
            assert actual.get_host(node) == expected.get_host(node)

    for config_id in range(len(CONFIG_NAMES)):
        for slot in range(NUM_SLOTS):
            for layer in range(NUM_LAYERS):
                for position in range(0, MAX_SEQ_LEN, PAGE_TOKENS):
                    left = expected.lookup(layer, position, slot, config_id=config_id)
                    right = actual.lookup(layer, position, slot, config_id=config_id)
                    assert (right.noc_addr, right.size_bytes, int(right.device_group_index)) == (
                        left.noc_addr,
                        left.size_bytes,
                        int(left.device_group_index),
                    )


def _load_layer_zero_qkv_weights():
    with (HF_MODEL / "model.safetensors.index.json").open() as index_file:
        weight_map = json.load(index_file)["weight_map"]
    weights = {}
    for local_name, checkpoint_name in QKV_WEIGHT_NAMES.items():
        with safe_open(HF_MODEL / weight_map[checkpoint_name], framework="pt", device="cpu") as checkpoint:
            weights[local_name] = checkpoint.get_tensor(checkpoint_name)
    return weights


def _producer_input():
    positions = torch.tensor(device_major_positions(0), dtype=torch.float32)
    columns = torch.arange(Llama31_8BConfig.EMB_SIZE, dtype=torch.float32)
    return (torch.sin(positions[:, None] / 23 + columns[None, :] / 127) * 0.125).reshape(
        1, 1, GLOBAL_CHUNK, Llama31_8BConfig.EMB_SIZE
    )


# The pure oracle must distinguish every logical field and both SP chunk boundaries before it is
# trusted against hardware; exact powers of two ensure BF16/BFP8 conversion cannot hide an error.
def test_kv_table_oracle_coordinates_are_unique_and_exact():
    keys = [
        (slot, layer, head, position)
        for slot in range(NUM_SLOTS)
        for layer in (0, NUM_LAYERS - 1)
        for head in range(NUM_KV_HEADS)
        for position in BOUNDARY_POSITIONS
    ]
    assert len({logical_tag(*key) for key in keys}) == len(keys)
    for kind in ("k", "v"):
        for key in keys:
            page = tagged_page(kind, *key)
            allowed = {32.0, 64.0} if kind == "k" else {-64.0, -32.0}
            assert set(torch.unique(page).tolist()) <= allowed
            assert torch.equal(page, page.to(torch.bfloat16).float())
    assert [independent_tensor_location(position) for position in BOUNDARY_POSITIONS] == [
        (0, 0),
        (0, 224),
        (1, 0),
        (3, 224),
        (0, 256),
        (3, 480),
    ]


@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-sp4-tp8")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], ids=["line"], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="Llama KV-table readback requires a Blackhole Galaxy")
@pytest.mark.timeout(1800)
# Fill every production cache coordinate with an independent exact tag, then require all 65,536
# table pages to match the live tensor and CPU tag; the same allocation also proves protobuf lookup
# and ownership roundtrip, avoiding a second Galaxy open and cache allocation for metadata alone.
def test_llama_kv_table_reads_all_synthetic_cache_pages(mesh_device, device_params, tmp_path):
    cache = allocate_kv_cache(
        mesh_device,
        MeshConfig(MESH_SHAPE, TP),
        num_users=NUM_SLOTS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        cache_dtype=ttnn.bfloat8_b,
    )
    try:
        _write_synthetic_cache(mesh_device, cache)
        table = build_kv_chunk_address_table(mesh_device=mesh_device, kv_cache=cache, chunk_size=GLOBAL_CHUNK)
        _assert_table_geometry(table)
        snapshots = {"k": _snapshot_shards(cache.k), "v": _snapshot_shards(cache.v)}
        comparisons = 0
        for config_id, config_name in enumerate(CONFIG_NAMES):
            kind, head_text = config_name.split("_h")
            head = int(head_text)
            for slot in range(NUM_SLOTS):
                for layer in range(NUM_LAYERS):
                    for position in range(0, MAX_SEQ_LEN, PAGE_TOKENS):
                        live = _snapshot_page(snapshots[kind], head=head, slot=slot, layer=layer, position=position)
                        expected = tagged_page(kind, slot, layer, head, position).to(torch.bfloat16)
                        assert_page_equal(live, expected, f"live {config_name}/{slot}/{layer}/{position}")
                        table_page = _decode_page(table.read_device_chunk(layer, position, slot, config_id=config_id))
                        assert_page_equal(table_page, live, f"table {config_name}/{slot}/{layer}/{position}")
                        comparisons += 1
        assert comparisons == EXPECTED_PAGES
        table_path = tmp_path / "llama-kv-table.pb"
        build_and_serialize_kv_chunk_table(
            mesh_device=mesh_device,
            kv_cache=cache,
            chunk_size=GLOBAL_CHUNK,
            path=str(table_path),
        )
        actual = ttnn.experimental.disaggregation.import_from_protobuf_file(str(table_path))
        _assert_tables_equivalent(table, actual)
    finally:
        cache.k.deallocate(True)
        cache.v.deallocate(True)


@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-sp4-tp8")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], ids=["line"], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="Llama GQA production requires a Blackhole Galaxy")
@pytest.mark.timeout(1200)
# Run real layer-0 GQA projection and indexed RoPE, write its K/V through the production writer,
# then compare every produced page through the table with the independently read live cache tensor.
def test_llama_gqa_producer_pages_match_kv_table(mesh_device, device_params):
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    projection = QKVProjection(mesh_device, mesh_config, _load_layer_zero_qkv_weights())
    rope_tables = build_indexed_rope(mesh_device, max_seq_len=MAX_SEQ_LEN, chunk_size=GLOBAL_CHUNK, sp_axis=0)
    transformation = build_transformation_mat(mesh_device)
    cache = allocate_kv_cache(
        mesh_device,
        mesh_config,
        num_users=NUM_SLOTS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        cache_dtype=ttnn.bfloat8_b,
    )
    tt_input = None
    produced = []
    try:
        tt_input = ttnn.from_torch(
            _producer_input().to(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, None)),
        )
        tt_q, tt_k, tt_v = projection(tt_input)
        tt_k_rot = apply_indexed_rope(tt_k, rope_tables, transformation, kv_actual_global=0, sp_axis=0)
        produced.extend((tt_q, tt_k, tt_v, tt_k_rot))
        write_kv_chunk(cache, tt_k_rot, tt_v, slot_idx=0, layer_idx=0, actual_start=0, actual_end=GLOBAL_CHUNK)
        ttnn.synchronize_device(mesh_device)

        table = build_kv_chunk_address_table(mesh_device=mesh_device, kv_cache=cache, chunk_size=GLOBAL_CHUNK)
        snapshots = {"k": _snapshot_shards(cache.k), "v": _snapshot_shards(cache.v)}
        comparisons = 0
        nonzero_values = 0
        for config_id, config_name in enumerate(CONFIG_NAMES):
            kind, head_text = config_name.split("_h")
            head = int(head_text)
            for position in range(0, GLOBAL_CHUNK, PAGE_TOKENS):
                live = _snapshot_page(snapshots[kind], head=head, slot=0, layer=0, position=position)
                nonzero_values += torch.count_nonzero(live).item()
                table_page = _decode_page(table.read_device_chunk(0, position, 0, config_id=config_id))
                assert_page_equal(table_page, live, f"producer {config_name}/0/0/{position}")
                comparisons += 1
        assert comparisons == len(CONFIG_NAMES) * (GLOBAL_CHUNK // PAGE_TOKENS)
        assert nonzero_values > 0
    finally:
        for tensor in reversed(produced):
            tensor.deallocate(True)
        if tt_input is not None:
            tt_input.deallocate(True)
        cache.k.deallocate(True)
        cache.v.deallocate(True)
        for tensor in (*rope_tables, transformation):
            tensor.deallocate(True)
