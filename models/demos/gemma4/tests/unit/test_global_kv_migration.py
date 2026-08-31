# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Contract and focused device tests for Gemma 4 global KV migration."""

import pytest
import torch

try:
    from ttnn.device import is_blackhole

    import ttnn
    from models.demos.gemma4.tt.attention.global_migration import (
        HEAD_DIM,
        ROTARY_DIM,
        ROW_DIM,
        allocate_global_migration_cache,
        global_layer_indices,
        interleave_perm,
        merged_kv_perms,
        pack_global_kv_reference,
        write_global_kv_chunk,
    )
    from models.demos.gemma4.tt.runners.kv_chunk_table import (
        CHUNK_SIZE_BYTES,
        CONFIG_NAMES,
        build_kv_chunk_address_table,
        iter_source_chunk_locations,
    )
    from tests.ttnn.utils_for_testing import assert_with_pcc
except Exception as exc:  # pragma: no cover - depends on a built ttnn package
    pytest.skip(f"Gemma 4 migration tests require built ttnn: {exc}", allow_module_level=True)


def test_global_layer_ids_and_config_order_are_locked():
    pattern = tuple(["sliding_attention"] * 5 + ["full_attention"]) * 10
    assert global_layer_indices(pattern) == tuple(range(5, 60, 6))
    assert CONFIG_NAMES == ("kv_h0", "kv_h1", "kv_h2", "kv_h3")
    assert CHUNK_SIZE_BYTES == 20 * 1088 == 21760


def test_512_128_column_permutations_match_decode_contract():
    perm = interleave_perm()
    k_cols, v_cols = merged_kv_perms()

    assert perm[:8].tolist() == [0, 256, 1, 257, 2, 258, 3, 259]
    assert k_cols.tolist() == perm[:ROTARY_DIM].tolist()
    assert v_cols[:8].tolist() == list(range(64, 72))
    assert v_cols[184:200].tolist() == list(range(248, 256)) + list(range(320, 328))
    assert v_cols[-8:].tolist() == list(range(312, 320))
    assert sorted(v_cols.tolist()) == list(range(HEAD_DIM))


def test_reference_pack_selects_exact_k_and_v_channels():
    k = torch.arange(HEAD_DIM, dtype=torch.float32).reshape(1, 1, 1, HEAD_DIM)
    v = (10_000 + torch.arange(HEAD_DIM, dtype=torch.float32)).reshape(1, 1, 1, HEAD_DIM)
    packed = pack_global_kv_reference(k, v)
    k_cols, v_cols = merged_kv_perms()

    assert packed.shape[-1] == ROW_DIM
    torch.testing.assert_close(packed[..., :ROTARY_DIM], k[..., k_cols])
    torch.testing.assert_close(packed[..., ROTARY_DIM:], v[..., v_cols])


def test_source_bank_walk_compacts_global_layers_but_keeps_semantic_ids():
    entries = list(
        iter_source_chunk_locations(
            seq_len=8192,
            chunk_size=8192,
            sp=8,
            num_users=2,
            global_layers=tuple(range(5, 60, 6)),
            num_banks=12,
        )
    )
    # 8 CP rows * 2 users * 10 global layers * (1024 local tokens / 32).
    assert len(entries) == 8 * 2 * 10 * 32
    cp0 = [entry for entry in entries if entry[0] == 0]
    assert cp0[0][:5] == (0, 0, 5, 0, 0)
    assert cp0[31][:5] == (0, 0, 5, 992, 7)
    assert cp0[32][:4] == (0, 0, 11, 0)
    assert cp0[320][:4] == (0, 1, 5, 0)
    assert {entry[2] for entry in entries} == set(range(5, 60, 6))
    assert not ({entry[2] for entry in entries} & {0, 1, 2, 3, 4, 6})


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="Gemma 4 CP8/TP4 migration staging targets Blackhole")
@pytest.mark.timeout(0)
def test_device_pack_and_compact_cache_write(mesh_device, device_params):
    del device_params
    seq_len = 256
    torch.manual_seed(0)
    k = torch.randn(1, 4, seq_len, HEAD_DIM)
    v = torch.randn(1, 4, seq_len, HEAD_DIM)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(8, 4), dims=(2, 1))
    tt_k = ttnn.from_torch(
        k,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    tt_v = ttnn.from_torch(
        v,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    cache = allocate_global_migration_cache(mesh_device, num_users=1, num_layers=1, max_seq_len=seq_len)
    valid_global = seq_len - 11
    write_global_kv_chunk(
        cache,
        tt_k,
        tt_v,
        slot_idx=0,
        layer_idx=0,
        kv_actual=0,
        valid_global=valid_global,
    )
    ttnn.synchronize_device(mesh_device)

    expected = pack_global_kv_reference(k, v)
    expected[:, :, valid_global:] = 0
    shards = ttnn.get_device_tensors(cache.kv)
    for cp_row in range(8):
        for head in range(4):
            actual = ttnn.to_torch(shards[cp_row * 4 + head]).float()[0, 0]
            wanted = expected[0, head, cp_row * 32 : (cp_row + 1) * 32]
            assert_with_pcc(wanted, actual, 0.999)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="Gemma 4 CP8/TP4 migration staging targets Blackhole")
@pytest.mark.timeout(0)
def test_source_table_authors_only_semantic_global_rows(mesh_device, device_params):
    del device_params
    globals_ = tuple(range(5, 60, 6))
    cache = allocate_global_migration_cache(mesh_device, num_users=1, num_layers=len(globals_), max_seq_len=256)
    table = build_kv_chunk_address_table(
        mesh_device=mesh_device,
        cache=cache,
        seq_len=256,
        mesh_shape=(8, 4),
        sp_axis=0,
        num_users=1,
        chunk_size=256,
        global_layers=globals_,
    )
    assert table.num_configs() == 4
    for config_id in range(4):
        assert table.lookup(5, 0, 0, config_id).size_bytes == CHUNK_SIZE_BYTES
        assert table.lookup(59, 224, 0, config_id).size_bytes == CHUNK_SIZE_BYTES
        assert table.lookup(0, 0, 0, config_id).size_bytes == 0
        assert table.lookup(6, 0, 0, config_id).size_bytes == 0
