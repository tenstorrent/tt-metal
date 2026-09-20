# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import ttnn
from models.demos.common.prefill.runners import migration_driver, prefill_producer
from models.demos.gemma4_d_p.tt.attention.ring_prefill import GlobalRingKVCache, SlidingRingKVCache
from models.demos.gemma4_d_p.tt.runners.adapter import Gemma4PrefillAdapter, Gemma4ServiceConfig
from models.demos.gemma4_d_p.tt.runners.kv_caches import Gemma4KvCaches
from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table
from models.demos.gemma4_d_p.tt.runners.kv_validation import (
    PREPARED_GPU_TRACE_LAYOUT,
    cache_pcc,
    load_gpu_cache_heads,
    read_cache_head,
    read_slot_kv_and_check_pcc,
)
from models.demos.gemma4_d_p.tt.runners.prepare_gpu_reference import prepare_layer
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


@pytest.fixture
def caches_and_mesh():
    address = 0

    def tensor():
        nonlocal address
        address += 0x100000
        base = address
        return SimpleNamespace(dtype=ttnn.bfloat8_b, buffer_address=lambda: base)

    caches = Gemma4KvCaches(
        layers=[SlidingRingKVCache(tensor(), tensor()) for _ in range(5)] + [GlobalRingKVCache(tensor())],
        layer_types=("sliding_attention",) * 5 + ("full_attention",),
        num_users=6,
        max_seq_len=256,
        cp=8,
        tp=4,
    )
    mesh = SimpleNamespace(
        shape=(8, 4),
        dram_grid_size=lambda: SimpleNamespace(x=8),
        get_fabric_node_id=lambda coordinates: ttnn.FabricNodeId(ttnn.MeshId(0), coordinates[0] * 4 + coordinates[1]),
    )
    return caches, mesh


def test_migration_stages_preserve_each_tensor_and_semantic_layer(caches_and_mesh, monkeypatch, expect_error):
    caches, mesh = caches_and_mesh
    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = SimpleNamespace(num_layers=6, chunk_size=256)
    runtime.mesh_device = mesh
    monkeypatch.setattr(runtime, "_check_cache", lambda supplied: None)
    stages = runtime.kv_migration_stages(caches, 0, 6)
    assert [(stage.first_layer, stage.count) for stage in stages] == [
        (layer, 1) for layer in range(5) for _ in range(2)
    ] + [(5, 1)]
    assert len({stage.base_addr for stage in stages}) == 11
    with expect_error(ValueError, "all layers on one rank"):
        runtime.kv_migration_stages(caches, 1, 5)
    with expect_error(ValueError, "one stage per cache tensor"):
        runtime.build_kv_chunk_table(caches, "unused.pb", stage_layouts=[])
    layouts = [[dict(rank=0, base_addr=stage.base_addr, first_layer=stage.first_layer, count=1)] for stage in stages]
    layouts[0][0]["base_addr"] += 32
    with expect_error(ValueError, "does not match"):
        runtime.build_kv_chunk_table(caches, "unused.pb", stage_layouts=layouts)


def test_loopback_byte_check_covers_all_configs_and_detects_corruption(caches_and_mesh, monkeypatch, tmp_path):
    caches, mesh = caches_and_mesh
    table = build_kv_chunk_address_table(mesh_device=mesh, kv_caches=caches, chunk_size=256)
    table_path = str(tmp_path / "table.pb")
    ttnn.experimental.disaggregation.export_to_protobuf_file(table, table_path)
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(table_path)
    monkeypatch.setattr(prefill_producer, "ADAPTER", Gemma4PrefillAdapter())
    monkeypatch.setattr(prefill_producer, "NUM_LAYERS", 6)
    monkeypatch.setattr(prefill_producer, "_resolve_unique_id", lambda nodes, mapping: int(nodes[0].chip_id))
    memory = {}
    pairs = [(0, 5, 256), (1, 3, 256), (2, 4, 256)]
    adapter = Gemma4PrefillAdapter()
    plan = migration_driver._cache_plan(table, None)
    assert len(plan) == 36
    assert [len(entry["rows"]) for entry in plan] == [1] * 4 + [5] * 32
    for source, destination, length in pairs:
        for config in range(36):
            for layer in adapter.cache_layer_rows(config, 6):
                for position in range(0, length, 32):
                    for slot in (source, destination):
                        location = table.lookup(layer, position, slot, config)
                        node = table.get_device_group(location.device_group_index).fabric_node_ids[0]
                        memory[int(node.chip_id), location.noc_addr] = bytes(
                            [source + 1, layer, config, position // 32]
                        )
    monkeypatch.setattr(
        ttnn.experimental.disaggregation, "read_dram_umd", lambda uid, address, size: memory[uid, address]
    )
    assert migration_driver._verify_dst_vs_src_bytes(table, {}, pairs, None)
    location = table.lookup(4, 224, 4, 35)
    node = table.get_device_group(location.device_group_index).fabric_node_ids[0]
    memory[int(node.chip_id), location.noc_addr] = b"corrupted"
    assert not migration_driver._verify_dst_vs_src_bytes(table, {}, pairs, None)


def encode_integer_bfp8(rows):
    tiles = rows.numpy().reshape(32, -1, 32).transpose(1, 0, 2)
    faces = tiles.reshape(-1, 2, 16, 2, 16).transpose(0, 1, 3, 2, 4).reshape(-1, 1024)
    raw = np.empty((len(tiles), 1088), dtype=np.uint8)
    raw[:, :64] = 133
    raw[:, 64:] = np.abs(faces).astype(np.uint8) | ((faces < 0).astype(np.uint8) << 7)
    return raw.tobytes()


def test_shared_producer_checks_packed_global_and_sliding_kv(caches_and_mesh, monkeypatch, tmp_path):
    caches, mesh = caches_and_mesh
    table = build_kv_chunk_address_table(mesh_device=mesh, kv_caches=caches, chunk_size=256)
    monkeypatch.setattr(Gemma4ServiceConfig, "NUM_LAYERS", 6)
    monkeypatch.setattr(prefill_producer, "ADAPTER", Gemma4PrefillAdapter())
    monkeypatch.setattr(prefill_producer, "_resolve_unique_id", lambda nodes, mapping: int(nodes[0].chip_id))
    monkeypatch.delenv("PREFILL_PCC_GOLDEN_LEN", raising=False)
    monkeypatch.setenv("PREFILL_PCC_SUMMARY_DIR", str(tmp_path))
    (tmp_path / "kv_cache").mkdir()
    memory = {}
    for layer in range(6):
        heads, width = (4, 512) if layer == 5 else (16, 256)
        key = (torch.arange(heads * 32 * width).reshape(1, heads, 32, width) % 113 - 56).float()
        value = (key * 3 + 7).remainder(107) - 53
        directory = tmp_path / "kv_cache" / f"layer_{layer}"
        directory.mkdir()
        rows = torch.cat((key[0].permute(1, 0, 2).reshape(32, -1), value[0].permute(1, 0, 2).reshape(32, -1)), dim=-1)
        for start, end in ((0, 16), (16, 32)):
            save_file(
                {f"kv_post_transform_layer_{layer}": rows[start:end].contiguous()},
                str(directory / f"rows_{start:08d}_{end:08d}.safetensors"),
            )
        if layer == 5:
            rotary = torch.stack((torch.arange(64), torch.arange(256, 320)), dim=1).flatten()
            values = torch.cat(
                (torch.arange(64, 256), torch.arange(320, 512), torch.arange(64), torch.arange(256, 320))
            )
            expected = torch.cat((key[0, ..., rotary], value[0, ..., values]), dim=-1)
            entries = {head: expected[head] for head in range(4)}
        else:
            order = torch.stack((torch.arange(128), torch.arange(128, 256)), dim=1).flatten()
            entries = {
                **{4 + head: key[0, head, :, order] for head in range(16)},
                **{20 + head: value[0, head] for head in range(16)},
            }
        for config, rows in entries.items():
            location = table.lookup(layer, 0, 0, config)
            node = table.get_device_group(location.device_group_index).fabric_node_ids[0]
            memory[int(node.chip_id), location.noc_addr] = encode_integer_bfp8(rows)
    monkeypatch.setattr(
        ttnn.experimental.disaggregation, "read_dram_umd", lambda uid, address, size: memory[uid, address]
    )
    scores = prefill_producer._read_slot_kv_and_check_pcc(table, {}, 0, 32, tmp_path)
    assert scores == {"global_k_rotary": 1.0, "global_v": 1.0, "sliding_k": 1.0, "sliding_v": 1.0}
    report = json.loads((tmp_path / "gemma4_slot0.json").read_text())
    assert report["tokens"] == 32 and report["slot"] == 0
    assert report["minima"] == scores
    assert len(report["measurements"]) == 164
    location = table.lookup(5, 0, 0, 3)
    node = table.get_device_group(location.device_group_index).fabric_node_ids[0]
    memory[int(node.chip_id), location.noc_addr] = bytes(location.size_bytes)
    assert read_slot_kv_and_check_pcc(table, {}, 0, 32, tmp_path)["global_v"] == 0.0


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("operand", ["expected", "actual", "both"])
def test_pcc_rejects_nonfinite_values(expect_error, nonfinite, operand):
    expected = torch.ones(4097, 256)
    actual = expected.clone()
    if operand in ("expected", "both"):
        expected[-1, -1] = nonfinite
    if operand in ("actual", "both"):
        actual[-1, -1] = nonfinite
    with expect_error(ValueError, "finite"):
        cache_pcc(expected, actual)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pcc_matches_reference_across_blocks(dtype):
    values = (torch.arange(5003 * 640).reshape(5003, 640) % 257 - 128).float()
    offsets = torch.arange(5003).reshape(-1, 1).remainder(997)
    expected = values.to(dtype)[:, 128:]
    actual = (values * 0.75 + offsets).to(dtype)[:, 128:]
    reference = torch.corrcoef(torch.stack((expected, actual)).float().reshape(2, -1))[0, 1]
    assert cache_pcc(expected, actual) == pytest.approx(float(reference), rel=0, abs=1e-4)


def test_pcc_constant_and_identical_inputs():
    values = torch.ones(4097, 256)
    assert cache_pcc(values, values) == 1.0
    assert cache_pcc(values, values * 2) == 0.0
    varying = torch.arange(4097).reshape(-1, 1).float()
    assert cache_pcc(varying, -varying) == pytest.approx(-1.0)
    assert cache_pcc(varying, torch.ones_like(varying)) == 0.0


def test_gpu_trace_requires_complete_contiguous_rows(tmp_path, expect_error):
    directory = tmp_path / "kv_cache" / "layer_0"
    directory.mkdir(parents=True)
    rows = torch.zeros(32, 8192)
    save_file({"kv_post_transform_layer_0": rows}, str(directory / "rows_00000000_00000032.safetensors"))
    with expect_error(ValueError, "covers 32/64 tokens"):
        load_gpu_cache_heads(tmp_path, 0, 64)
    save_file({"kv_post_transform_layer_0": rows}, str(directory / "rows_00000064_00000096.safetensors"))
    with expect_error(ValueError, "noncontiguous GPU trace"):
        load_gpu_cache_heads(tmp_path, 0, 64)
    save_file(
        {"kv_post_transform_layer_0": rows[:, :256].contiguous()}, str(directory / "rows_00000000_00000032.safetensors")
    )
    with expect_error(ValueError, "invalid GPU KV shape"):
        load_gpu_cache_heads(tmp_path, 0, 32)


def test_bank_reads_restore_token_order(monkeypatch, expect_error):
    width, blocks = 64, 8
    chunk_bytes = width // 32 * 1088
    expected = (torch.arange(blocks * 32 * width).reshape(blocks, 32, width) % 113 - 56).float()
    locations = {
        block
        * 32: SimpleNamespace(
            noc_addr=((block % 2) << 32) | (4096 + block // 2 * chunk_bytes),
            size_bytes=chunk_bytes,
            device_group_index=ttnn.experimental.disaggregation.DeviceGroupIndex(0),
        )
        for block in range(blocks)
    }
    memory = {
        (bank << 32) | 4096: b"".join(encode_integer_bfp8(expected[block]) for block in range(bank, blocks, 2))
        for bank in range(2)
    }
    config = SimpleNamespace(chunk_n_tokens=32, num_layers=60, chunk_size_bytes=chunk_bytes)
    table = SimpleNamespace(
        config=lambda _: config,
        lookup=lambda layer, position, slot, config_id: locations[position],
        get_device_group=lambda _: SimpleNamespace(fabric_node_ids=[]),
    )
    calls = []

    def read(unique_id, address, size):
        calls.append((address, size))
        return memory[address][:size]

    monkeypatch.setattr(prefill_producer, "_resolve_unique_id", lambda nodes, mapping: 1)
    monkeypatch.setattr(ttnn.experimental.disaggregation, "read_dram_umd", read)
    actual = read_cache_head(table, {}, 0, 0, 4, blocks * 32, width)
    torch.testing.assert_close(actual, expected.reshape(-1, width), rtol=0, atol=0)
    assert len(calls) == 2
    assert all(size == 4 * chunk_bytes for _, size in calls)
    locations[32].size_bytes -= 1
    with expect_error(ValueError, "Missing or invalid KV chunk"):
        read_cache_head(table, {}, 0, 0, 4, blocks * 32, width)
    locations[32].size_bytes += 1
    locations[64].noc_addr += chunk_bytes
    with expect_error(ValueError, "Noncontiguous Gemma4 KV bank"):
        read_cache_head(table, {}, 0, 0, 4, blocks * 32, width)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_command_queue_read_restores_chunk_order(monkeypatch, dtype):
    from models.demos.gemma4_d_p.tt.runners import kv_validation

    heads, width, tokens, chunk_size = 16, 32, 512, 256
    expected = (torch.arange(heads * tokens * width).reshape(heads, tokens, width) % 113).to(dtype)
    positions = [
        chunk + rank * 32 + row for rank in range(8) for chunk in range(0, tokens, chunk_size) for row in range(32)
    ]
    gathered = expected[:, positions].unsqueeze(0)
    cache = SimpleNamespace(shape=(6, 4, 32768, width))
    selected = object()
    row_major = object()
    released = []

    def select(tensor, starts, ends, *, memory_config):
        assert tensor is cache
        assert starts == (3, 0, 0, 0)
        assert ends == (4, 4, tokens // 8, width)
        assert memory_config == ttnn.DRAM_MEMORY_CONFIG
        return selected

    monkeypatch.setattr(Gemma4ServiceConfig, "CHUNK_SIZE", chunk_size)
    monkeypatch.setattr(ttnn, "slice", select)
    monkeypatch.setattr(ttnn, "untilize", lambda tensor, memory_config: row_major)
    monkeypatch.setattr(ttnn, "from_device", lambda tensor, blocking: gathered)
    monkeypatch.setattr(ttnn, "deallocate", released.append)
    shards = [
        gathered[:, column * 4 : (column + 1) * 4, row * (tokens // 8) : (row + 1) * (tokens // 8)]
        for row in range(8)
        for column in range(4)
    ]
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda tensor: shards)
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor: tensor)
    actual = kv_validation.read_cache_tensor(cache, 3, tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert released == [selected, row_major]


@pytest.mark.parametrize("layer", [0, 5])
def test_prepared_gpu_reference_preserves_heads_and_prefixes(tmp_path, layer):
    source, prepared = tmp_path / "source", tmp_path / "prepared"
    directory = source / "kv_cache" / f"layer_{layer}"
    directory.mkdir(parents=True)
    (prepared / "kv_cache").mkdir(parents=True)
    width = 4096 if layer == 5 else 8192
    rows = (torch.arange(32 * width).reshape(32, width) % 251 - 125).bfloat16()
    for start in (0, 16):
        save_file(
            {f"kv_post_transform_layer_{layer}": rows[start : start + 16].contiguous()},
            str(directory / f"rows_{start:08d}_{start + 16:08d}.safetensors"),
        )
    prepare_layer(source, prepared, layer, 32)
    for tokens in (16, 23, 32):
        expected = load_gpu_cache_heads(source, layer, tokens)
        actual = load_gpu_cache_heads(prepared, layer, tokens)
        assert actual.keys() == expected.keys()
        for config in expected:
            assert actual[config].is_contiguous()
            assert torch.equal(actual[config], expected[config])


@pytest.mark.parametrize(
    "head_count,tokens,dtype,message",
    [(3, 32, torch.bfloat16, "head names"), (4, 16, torch.bfloat16, "shape"), (4, 32, torch.float32, "BF16")],
)
def test_prepared_gpu_reference_rejects_invalid_tensors(tmp_path, expect_error, head_count, tokens, dtype, message):
    from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import CONFIG_NAMES

    directory = tmp_path / "kv_cache"
    directory.mkdir()
    save_file(
        {CONFIG_NAMES[head]: torch.zeros(tokens, 640, dtype=dtype) for head in range(head_count)},
        str(directory / "layer_5.safetensors"),
        metadata={"layout": PREPARED_GPU_TRACE_LAYOUT},
    )
    with expect_error(ValueError, message):
        load_gpu_cache_heads(tmp_path, 5, 32)
