# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Contract and focused device tests for Gemma 4 mixed KV migration."""

import inspect
import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

try:
    from ttnn.device import is_blackhole

    import ttnn
    from models.demos.gemma4.tt.attention.global_migration import HEAD_DIM as GLOBAL_HEAD_DIM
    from models.demos.gemma4.tt.attention.global_migration import (
        ROTARY_DIM,
        ROW_DIM,
        allocate_global_migration_cache,
        global_layer_indices,
        interleave_perm,
        merged_kv_perms,
        pack_global_kv_reference,
        write_global_kv_chunk,
    )
    from models.demos.gemma4.tt.attention.sliding_migration import HEAD_DIM as SLIDING_HEAD_DIM
    from models.demos.gemma4.tt.attention.sliding_migration import (
        allocate_sliding_migration_cache,
        pack_sliding_k_reference,
        sliding_k_perm,
        sliding_layer_indices,
        write_sliding_kv_chunk,
    )
    from models.demos.gemma4.tt.model import Gemma4Model
    from models.demos.gemma4.tt.runners.adapters.gemma4 import migration_runtime_paged_caches
    from models.demos.gemma4.tt.runners.kv_chunk_table import (
        CONFIG_NAMES,
        GLOBAL_CHUNK_SIZE_BYTES,
        SLIDING_CHUNK_SIZE_BYTES,
        build_kv_chunk_address_table,
        iter_source_chunk_locations,
        worker_host_name,
    )
    from models.demos.gemma4.tt.tt_prefill_runtime import TtPrefillRuntime
    from tests.ttnn.utils_for_testing import assert_with_pcc
except Exception as exc:  # pragma: no cover - depends on a built ttnn package
    pytest.skip(f"Gemma 4 migration tests require built ttnn: {exc}", allow_module_level=True)


GLOBAL_LAYERS = tuple(range(5, 60, 6))
SLIDING_LAYERS = tuple(layer for layer in range(60) if layer not in GLOBAL_LAYERS)


def test_migration_manifest_temporarily_caps_advertised_context_at_64k():
    manifest_path = Path(__file__).resolve().parents[2] / "tt/runners/manifests/gemma4_31b.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["env"]["PREFILL_MAX_SEQ_LEN"] == "65536"
    assert manifest["env"]["PREFILL_GEMMA4_SLIDING_CACHE_LEN"] == "65536"
    comment = " ".join(manifest["_comment"])
    assert "TEMPORARY" in comment and "memory" in comment


def test_loopback_manifests_are_a_matching_golden_free_full_cache_pair():
    gemma4_root = Path(__file__).resolve().parents[2]
    manifests = gemma4_root / "tt/runners/manifests"
    model = json.loads((manifests / "gemma4_31b.json").read_text())["env"]
    binding = yaml.safe_load((manifests / "gemma4_binding_loopback_migration_1rank.yaml").read_text())
    producer = yaml.safe_load((manifests / "gemma4_producer_loopback_migration.yaml").read_text())
    runner = binding["global_env"]

    assert producer["model"] == {
        "variant": model["PREFILL_MODEL"],
        "num_layers": int(model["PREFILL_NUM_LAYERS"]),
        "max_seq_len": int(model["PREFILL_MAX_SEQ_LEN"]),
        "chunk_size": int(model["PREFILL_CHUNK_SIZE"]),
    }
    assert producer["transport"]["sp"] == int(runner["PREFILL_SP"])
    assert producer["transport"]["tp"] == int(runner["PREFILL_TP"])
    assert producer["transport"]["h2d_service_id"] == runner["PREFILL_H2D_SERVICE_ID"]

    migration = producer["migration"]
    assert migration["table_path"] == runner["PREFILL_MIGRATION_TABLE_PATH"]
    assert migration["device_map_path"] == runner["PREFILL_MIGRATION_DEVICE_MAP_PATH"]
    assert migration["cmd_queue"] == runner["PREFILL_MIGRATION_CMD_QUEUE"]
    assert migration["table_queue"] == runner["PREFILL_MIGRATION_TABLE_QUEUE"]
    assert migration["resp_queue"] == runner["PREFILL_MIGRATION_RESP_QUEUE"]

    source_slots = producer["workload"]["num_users"]
    assert int(runner["PREFILL_NUM_USERS"]) == 2
    assert source_slots == 1
    assert migration["dest_endpoint_id"] == 1
    assert migration["dst_slot_offset"] == source_slots
    assert producer["workload"]["chunks"] == str(producer["model"]["max_seq_len"] // producer["model"]["chunk_size"])
    assert producer["workload"]["check_pcc"] is False
    assert producer["env"]["PREFILL_VERIFY_MIGRATION"] == "dst-bytes"
    assert producer["env"]["PREFILL_SEND_SHUTDOWN"] == "1"

    token_trace = gemma4_root / "tests/test_data/loopback_tokens"
    assert Path(producer["workload"]["trace_dir"]) == token_trace.relative_to(Path(__file__).resolve().parents[5])
    metadata = json.loads((token_trace / "metadata.json").read_text())
    assert metadata["token_ids"] and all(isinstance(token, int) and token >= 0 for token in metadata["token_ids"])
    assert not (token_trace / "kv_cache").exists()


def test_source_table_uses_legacy_worker_crc_host_key():
    hostname = "gemma-prefill-host"
    expected = f"host-{zlib.crc32(hostname.encode()) & 0x7FFFFFFF:08x}"
    assert worker_host_name(hostname) == expected


def test_migration_runtime_has_no_paged_kv_and_explicit_cache_fill_mode():
    paged = migration_runtime_paged_caches(60)
    assert len(paged) == 60 and all(cache is None for cache in paged)

    call_param = inspect.signature(Gemma4Model.__call__).parameters["skip_lm_head"]
    prefill_param = inspect.signature(Gemma4Model.ttnn_prefill_forward).parameters["skip_lm_head"]
    assert call_param.default is False
    assert prefill_param.default is False


def test_runtime_dispatches_ring_and_packed_caches_without_page_tables():
    captured = {}

    class FakeInput:
        def deallocate(self, _force):
            captured["input_deallocated"] = True

    class FakeModel:
        def transform_and_embed_prefill_inputs_device(self, input_tensor, page_table, chunk_page_table, start):
            assert page_table is None and chunk_page_table is None and start is None
            return object(), None, None, None

        def ttnn_prefill_forward(self, **kwargs):
            captured.update(kwargs)
            return None

    packed_global = object()
    packed_sliding = SimpleNamespace(max_seq_len=8192)
    caches = SimpleNamespace(
        paged=migration_runtime_paged_caches(60),
        migration=packed_global,
        sliding_migration=packed_sliding,
    )
    runtime = TtPrefillRuntime.__new__(TtPrefillRuntime)
    runtime.config = SimpleNamespace(num_users=1, chunk_size=8192, max_seq_len=8192)
    runtime.model = FakeModel()
    runtime._on_layer_complete = None
    runtime._layer_completion_sink = None
    runtime._resolve_kv = lambda _caches: caches
    runtime._stage_metadata = lambda **_kwargs: None

    runtime.prefill_chunk(
        FakeInput(),
        caches,
        slot_id=0,
        actual_start=0,
        actual_end=8192,
    )

    assert captured["input_deallocated"]
    assert captured["page_table"] is None and captured["chunk_page_table"] is None
    assert captured["skip_lm_head"] is True
    assert all(cache is None for cache in captured["kv_cache"])
    assert captured["global_migration_cache"] is packed_global
    assert captured["sliding_migration_cache"] is packed_sliding


def test_layer_ids_and_36_config_order_are_locked():
    pattern = tuple(["sliding_attention"] * 5 + ["full_attention"]) * 10
    assert global_layer_indices(pattern) == GLOBAL_LAYERS
    assert sliding_layer_indices(pattern) == SLIDING_LAYERS
    assert CONFIG_NAMES[:4] == tuple(f"{head:02d}_kv_h{head}" for head in range(4))
    assert CONFIG_NAMES[4:20] == tuple(f"{4 + head:02d}_k_h{head}" for head in range(16))
    assert CONFIG_NAMES[20:] == tuple(f"{20 + head:02d}_v_h{head}" for head in range(16))
    assert GLOBAL_CHUNK_SIZE_BYTES == 20 * 1088 == 21760
    assert SLIDING_CHUNK_SIZE_BYTES == 8 * 1088 == 8704


def test_global_and_sliding_k_permutations_match_decode_contract():
    global_perm = interleave_perm()
    k_cols, v_cols = merged_kv_perms()
    assert global_perm[:8].tolist() == [0, 256, 1, 257, 2, 258, 3, 259]
    assert k_cols.tolist() == global_perm[:ROTARY_DIM].tolist()
    assert v_cols[:8].tolist() == list(range(64, 72))
    assert v_cols[-8:].tolist() == list(range(312, 320))
    assert sorted(v_cols.tolist()) == list(range(GLOBAL_HEAD_DIM))

    local_perm = sliding_k_perm()
    assert local_perm[:8].tolist() == [0, 128, 1, 129, 2, 130, 3, 131]
    assert sorted(local_perm.tolist()) == list(range(SLIDING_HEAD_DIM))


def test_reference_packs_select_exact_channels():
    global_k = torch.arange(GLOBAL_HEAD_DIM, dtype=torch.float32).reshape(1, 1, 1, GLOBAL_HEAD_DIM)
    global_v = (10_000 + torch.arange(GLOBAL_HEAD_DIM, dtype=torch.float32)).reshape(1, 1, 1, GLOBAL_HEAD_DIM)
    packed = pack_global_kv_reference(global_k, global_v)
    k_cols, v_cols = merged_kv_perms()
    assert packed.shape[-1] == ROW_DIM
    torch.testing.assert_close(packed[..., :ROTARY_DIM], global_k[..., k_cols])
    torch.testing.assert_close(packed[..., ROTARY_DIM:], global_v[..., v_cols])

    local_k = torch.arange(SLIDING_HEAD_DIM, dtype=torch.float32)
    torch.testing.assert_close(pack_sliding_k_reference(local_k), local_k[sliding_k_perm()])


def test_source_bank_walk_accounts_for_compact_layers_and_local_heads():
    global_entries = list(
        iter_source_chunk_locations(
            seq_len=8192,
            chunk_size=8192,
            sp=8,
            num_users=2,
            semantic_layers=GLOBAL_LAYERS,
            num_banks=12,
            chunk_size_bytes=GLOBAL_CHUNK_SIZE_BYTES,
        )
    )
    assert len(global_entries) == 8 * 2 * 10 * 32
    cp0 = [entry for entry in global_entries if entry[0] == 0]
    assert cp0[0][:5] == (0, 0, 5, 0, 0)
    assert cp0[31][:5] == (0, 0, 5, 992, 7)
    assert cp0[32][:4] == (0, 0, 11, 0)
    assert cp0[320][:4] == (0, 1, 5, 0)

    head0 = next(
        iter_source_chunk_locations(
            seq_len=8192,
            chunk_size=8192,
            sp=8,
            num_users=1,
            semantic_layers=SLIDING_LAYERS,
            num_banks=8,
            chunk_size_bytes=SLIDING_CHUNK_SIZE_BYTES,
            heads_per_device=4,
            local_head=0,
        )
    )
    head3 = next(
        iter_source_chunk_locations(
            seq_len=8192,
            chunk_size=8192,
            sp=8,
            num_users=1,
            semantic_layers=SLIDING_LAYERS,
            num_banks=8,
            chunk_size_bytes=SLIDING_CHUNK_SIZE_BYTES,
            heads_per_device=4,
            local_head=3,
        )
    )
    assert head0[4:] == (0, 0)
    assert head3[4:] == (0, 12 * SLIDING_CHUNK_SIZE_BYTES)


def _to_cp_tp(mesh_device, tensor):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 1)),
    )


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="Gemma 4 CP8/TP4 migration staging targets Blackhole")
@pytest.mark.timeout(0)
def test_device_global_and_sliding_packs_with_ragged_tail(mesh_device, device_params):
    del device_params
    seq_len = 256
    valid_global = seq_len - 11
    torch.manual_seed(0)

    global_k = torch.randn(1, 4, seq_len, GLOBAL_HEAD_DIM)
    global_v = torch.randn(1, 4, seq_len, GLOBAL_HEAD_DIM)
    global_cache = allocate_global_migration_cache(mesh_device, num_users=1, num_layers=1, max_seq_len=seq_len)
    write_global_kv_chunk(
        global_cache,
        _to_cp_tp(mesh_device, global_k),
        _to_cp_tp(mesh_device, global_v),
        slot_idx=0,
        layer_idx=0,
        kv_actual=0,
        valid_global=valid_global,
    )

    sliding_k = torch.randn(1, 16, seq_len, SLIDING_HEAD_DIM)
    sliding_v = torch.randn(1, 16, seq_len, SLIDING_HEAD_DIM)
    sliding_cache = allocate_sliding_migration_cache(mesh_device, num_users=1, num_layers=1, max_seq_len=seq_len)
    write_sliding_kv_chunk(
        sliding_cache,
        _to_cp_tp(mesh_device, sliding_k),
        _to_cp_tp(mesh_device, sliding_v),
        slot_idx=0,
        layer_idx=0,
        kv_actual=0,
        valid_global=valid_global,
    )
    ttnn.synchronize_device(mesh_device)

    expected_global = pack_global_kv_reference(global_k, global_v)
    expected_global[:, :, valid_global:] = 0
    expected_sliding_k = pack_sliding_k_reference(sliding_k)
    expected_sliding_v = sliding_v.clone()
    expected_sliding_k[:, :, valid_global:] = 0
    expected_sliding_v[:, :, valid_global:] = 0

    cp = int(mesh_device.shape[0])
    tokens_per_cp = seq_len // cp
    global_shards = ttnn.get_device_tensors(global_cache.kv)
    sliding_k_shards = ttnn.get_device_tensors(sliding_cache.k)
    sliding_v_shards = ttnn.get_device_tensors(sliding_cache.v)
    for cp_row in range(cp):
        token_slice = slice(cp_row * tokens_per_cp, (cp_row + 1) * tokens_per_cp)
        for tp_col in range(4):
            shard_idx = cp_row * 4 + tp_col
            actual_global = ttnn.to_torch(global_shards[shard_idx]).float()[0, 0]
            assert_with_pcc(expected_global[0, tp_col, token_slice], actual_global, 0.999)
            for local_head in range(4):
                global_head = tp_col * 4 + local_head
                actual_k = ttnn.to_torch(sliding_k_shards[shard_idx]).float()[0, local_head]
                actual_v = ttnn.to_torch(sliding_v_shards[shard_idx]).float()[0, local_head]
                assert_with_pcc(expected_sliding_k[0, global_head, token_slice], actual_k, 0.999)
                assert_with_pcc(expected_sliding_v[0, global_head, token_slice], actual_v, 0.999)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="Gemma 4 CP8/TP4 migration staging targets Blackhole")
@pytest.mark.timeout(0)
def test_source_table_authors_the_correct_layer_family(mesh_device, device_params):
    del device_params
    seq_len = 256
    global_cache = allocate_global_migration_cache(
        mesh_device, num_users=1, num_layers=len(GLOBAL_LAYERS), max_seq_len=seq_len
    )
    sliding_cache = allocate_sliding_migration_cache(
        mesh_device, num_users=1, num_layers=len(SLIDING_LAYERS), max_seq_len=seq_len
    )
    table = build_kv_chunk_address_table(
        mesh_device=mesh_device,
        global_cache=global_cache,
        sliding_cache=sliding_cache,
        seq_len=seq_len,
        sliding_seq_len=seq_len,
        mesh_shape=(2, 4),
        sp_axis=0,
        num_users=1,
        chunk_size=seq_len,
        global_layers=GLOBAL_LAYERS,
    )
    assert table.num_configs() == 36
    for config_id in range(4):
        assert table.lookup(5, 0, 0, config_id).size_bytes == GLOBAL_CHUNK_SIZE_BYTES
        assert table.lookup(0, 0, 0, config_id).size_bytes == 0
    for config_id in range(4, 36):
        assert table.lookup(0, 0, 0, config_id).size_bytes == SLIDING_CHUNK_SIZE_BYTES
        assert table.lookup(5, 0, 0, config_id).size_bytes == 0

    for config_id, layer in ((0, 5), (4, 0), (20, 0)):
        location = table.lookup(layer, 0, 0, config_id)
        group = table.get_device_group(location.device_group_index)
        assert all(table.get_host(fnid) == worker_host_name() for fnid in group.fabric_node_ids)
