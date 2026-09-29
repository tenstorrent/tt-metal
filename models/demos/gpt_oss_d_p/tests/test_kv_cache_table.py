# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the GPT-OSS prefill KV chunk address table.

Validates ``build_kv_chunk_address_table`` against caches from
``tt/attention/kv_cache.allocate_kv_cache``:

  * multi-config layout: ``k_h0..N-1``, ``v_h0..N-1`` (``N == TP cols``)
  * block-cyclic SP positions + ROUND_ROBIN_1D DRAM bank walk
  * ``read_device_chunk`` bytes match the live device cache after ``write_kv_chunk``
  * protobuf round-trip preserves lookups

These tests do **not** run the model or fabric migrate — they isolate address-table correctness.

Run (Blackhole galaxy)::

    pytest models/demos/gpt_oss_d_p/tests/test_kv_cache_table.py -k smoke
    pytest models/demos/gpt_oss_d_p/tests/test_kv_cache_table.py -k readback
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.gpt_oss_d_p.tt.attention.kv_cache import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    allocate_kv_cache,
    write_kv_chunk,
)
from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import (
    build_and_serialize_kv_chunk_table,
    build_kv_chunk_address_table,
)
from tests.ttnn.utils_for_testing import assert_equal

HEAD_DIM = 64
SP_AXIS = 0
TP_AXIS = 1

# Small mesh: TP cols == num_kv_heads (table maps head h -> column h).
KV_TABLE_MESH_PARAMS = [
    pytest.param((2, 4), id="2x4"),
]


def _chunk_bytes_bf8(head_dim: int = HEAD_DIM) -> int:
    return (head_dim // 32) * 1088


def _write_random_cache(mesh_device, kv_cache, *, num_users, num_layers, seq_len, seed=0):
    """Fill every (user, layer) slot with random natural-order K/V via write_kv_chunk (one-shot)."""
    rows, cols = tuple(mesh_device.shape)
    sp, tp = rows, cols
    nkv = tp
    assert seq_len % (32 * sp) == 0

    torch.manual_seed(seed)
    sent_k = torch.randn(num_users, num_layers, nkv, seq_len, HEAD_DIM)
    sent_v = torch.randn(num_users, num_layers, nkv, seq_len, HEAD_DIM)

    in_dims = [None, None]
    in_dims[SP_AXIS] = 2
    in_dims[TP_AXIS] = 1

    def to_chunk(nat):
        return ttnn.from_torch(
            nat.reshape(1, nkv, seq_len, HEAD_DIM),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
        )

    for u in range(num_users):
        for layer in range(num_layers):
            tt_k = to_chunk(sent_k[u, layer])
            tt_v = to_chunk(sent_v[u, layer])
            write_kv_chunk(kv_cache, tt_k, tt_v, slot_idx=u, layer_idx=layer, kv_actual=0, sp_axis=SP_AXIS)
            tt_k.deallocate(True)
            tt_v.deallocate(True)
    ttnn.synchronize_device(mesh_device)


def _expected_chunk_from_device(
    cache,
    mesh_device,
    *,
    slot: int,
    layer: int,
    head: int,
    position: int,
    num_layers: int,
    chunk_size: int,
):
    """Slice the live device cache at the SP/TP chip that owns ``(head, position)``.

    Local seq layout matches the table builder's block-cyclic placement (one contiguous
    ``tokens_per_chunk_local`` block per ``chunk_size`` period on each SP row).
    """
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    tokens_per_chunk_local = chunk_size // sp
    seq_chunk = position // chunk_size
    offset_in_chunk = position % chunk_size
    sp_row = offset_in_chunk // tokens_per_chunk_local
    local_in_chunk = offset_in_chunk % tokens_per_chunk_local
    local_pos = seq_chunk * tokens_per_chunk_local + local_in_chunk

    batch_idx = slot * num_layers + layer
    dt = ttnn.get_device_tensors(cache)[sp_row * cols + head]
    torch_cache = ttnn.to_torch(dt).to(torch.bfloat16)
    return torch_cache[
        batch_idx : batch_idx + 1, :, local_pos : local_pos + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, :
    ].reshape(1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, HEAD_DIM)


@pytest.mark.parametrize("mesh_device", KV_TABLE_MESH_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("num_users, num_layers", [(1, 2), (2, 2)], ids=["u1xl2", "u2xl2"])
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
@pytest.mark.skipif(not is_blackhole(), reason="GPT-OSS KV NdShardSpec table tests target Blackhole")
@pytest.mark.timeout(0)
def test_gpt_oss_kv_chunk_table_smoke(mesh_device, num_users, num_layers, seq_len, device_params):
    """Tier 1: build multi-config table; every lookup has sane metadata."""
    rows, cols = tuple(mesh_device.shape)
    nkv = cols
    chunk_size = seq_len  # one-shot period (block-cyclic period == full seq)

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=seq_len,
        sp_axis=SP_AXIS,
        num_users=num_users,
        head_dim=HEAD_DIM,
    )
    table = build_kv_chunk_address_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=seq_len,
        num_layers=num_layers,
        mesh_shape=(rows, cols),
        sp_axis=SP_AXIS,
        num_users=num_users,
        chunk_size=chunk_size,
        num_kv_heads=nkv,
        head_dim=HEAD_DIM,
    )

    chunk_bytes = _chunk_bytes_bf8()
    chunks_per_seq = seq_len // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    assert table.num_configs() == 2 * nkv
    assert table.total_entries() == 2 * nkv * num_layers * chunks_per_seq * num_users
    assert table.config(0).chunk_size_bytes == chunk_bytes

    for config_id in range(table.num_configs()):
        for slot in range(num_users):
            for layer in range(num_layers):
                for position in range(0, seq_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                    loc = table.lookup(layer, position, slot, config_id=config_id)
                    assert loc.size_bytes == chunk_bytes
                    assert loc.noc_addr != 0


@pytest.mark.parametrize("mesh_device", KV_TABLE_MESH_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("num_users, num_layers", [(2, 2)], ids=["u2xl2"])
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
@pytest.mark.skipif(not is_blackhole(), reason="GPT-OSS KV NdShardSpec table tests target Blackhole")
@pytest.mark.timeout(0)
def test_gpt_oss_kv_chunk_table_protobuf_roundtrip(
    mesh_device, num_users, num_layers, seq_len, device_params, tmp_path
):
    """Export multi-config table to .pb and assert import restores identical lookups."""
    rows, cols = tuple(mesh_device.shape)
    nkv = cols
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=seq_len,
        sp_axis=SP_AXIS,
        num_users=num_users,
        head_dim=HEAD_DIM,
    )
    pb_path = tmp_path / "gpt_oss_kv_chunk_table.pb"
    build_and_serialize_kv_chunk_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=seq_len,
        num_layers=num_layers,
        mesh_shape=(rows, cols),
        sp_axis=SP_AXIS,
        num_users=num_users,
        chunk_size=seq_len,
        num_kv_heads=nkv,
        head_dim=HEAD_DIM,
        path=str(pb_path),
    )
    assert pb_path.is_file() and pb_path.stat().st_size > 0

    original = build_kv_chunk_address_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=seq_len,
        num_layers=num_layers,
        mesh_shape=(rows, cols),
        sp_axis=SP_AXIS,
        num_users=num_users,
        chunk_size=seq_len,
        num_kv_heads=nkv,
        head_dim=HEAD_DIM,
    )
    restored = ttnn.experimental.disaggregation.import_from_protobuf_file(str(pb_path))

    assert restored.num_configs() == original.num_configs()
    assert restored.total_entries() == original.total_entries()
    for config_id in range(original.num_configs()):
        assert restored.config(config_id).chunk_size_bytes == original.config(config_id).chunk_size_bytes
        for slot in range(num_users):
            for layer in range(num_layers):
                for position in range(0, seq_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                    o = original.lookup(layer, position, slot, config_id=config_id)
                    r = restored.lookup(layer, position, slot, config_id=config_id)
                    assert r.noc_addr == o.noc_addr
                    assert r.size_bytes == o.size_bytes
                    assert int(r.device_group_index) == int(o.device_group_index)


@pytest.mark.parametrize("mesh_device", KV_TABLE_MESH_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("num_users, num_layers", [(2, 2)], ids=["u2xl2"])
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
@pytest.mark.skipif(not is_blackhole(), reason="GPT-OSS KV NdShardSpec table tests target Blackhole")
@pytest.mark.timeout(0)
def test_gpt_oss_kv_chunk_table_readback(mesh_device, num_users, num_layers, seq_len, device_params):
    """Tier 2: write random KV, read every table chunk, assert bytes match the live device cache."""
    rows, cols = tuple(mesh_device.shape)
    nkv = cols
    chunk_size = seq_len

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=seq_len,
        sp_axis=SP_AXIS,
        num_users=num_users,
        head_dim=HEAD_DIM,
    )
    _write_random_cache(mesh_device, kv_cache, num_users=num_users, num_layers=num_layers, seq_len=seq_len)

    table = build_kv_chunk_address_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=seq_len,
        num_layers=num_layers,
        mesh_shape=(rows, cols),
        sp_axis=SP_AXIS,
        num_users=num_users,
        chunk_size=chunk_size,
        num_kv_heads=nkv,
        head_dim=HEAD_DIM,
    )

    chunk_shape = [1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, HEAD_DIM]
    comparisons = 0
    expected_comparisons = 2 * nkv * num_layers * (seq_len // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK) * num_users

    for config_id in range(table.num_configs()):
        is_k = config_id < nkv
        head = config_id if is_k else config_id - nkv
        cache = kv_cache.k if is_k else kv_cache.v
        for slot in range(num_users):
            for layer in range(num_layers):
                for position in range(0, seq_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                    raw_bytes = table.read_device_chunk(layer=layer, position=position, slot=slot, config_id=config_id)
                    assert len(raw_bytes) == table.config(config_id).chunk_size_bytes
                    chunk_tt = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw_bytes, chunk_shape)
                    chunk_torch = ttnn.to_torch(chunk_tt).to(torch.bfloat16)
                    expected = _expected_chunk_from_device(
                        cache,
                        mesh_device,
                        slot=slot,
                        layer=layer,
                        head=head,
                        position=position,
                        num_layers=num_layers,
                        chunk_size=chunk_size,
                    )
                    assert_equal(expected, chunk_torch)
                    comparisons += 1

    assert comparisons == expected_comparisons
    logger.info(
        f"GPT-OSS KV table readback OK: {comparisons} chunks "
        f"(configs={table.num_configs()}, users={num_users}, layers={num_layers}, seq={seq_len})"
    )
