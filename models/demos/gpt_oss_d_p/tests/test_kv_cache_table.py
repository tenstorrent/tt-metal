# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for GPT-OSS d_p KV chunk address table builder.

``KvChunkAddressTableMulti`` is a mock: ``bundle.unified_table`` aliases
``groups[0].table`` (K, global head 0). Tier 1 smoke asserts lookups on every
group; Tier 2 readback asserts bytes only for K / global-head-0.
"""

from __future__ import annotations

import pytest
import torch
from ttnn.device import is_blackhole

import ttnn
from models.demos.gpt_oss_d_p.tt.attention.config import AttentionConfig
from models.demos.gpt_oss_d_p.tt.attention.kv_cache_prefill_only import init_kv_cache
from models.demos.gpt_oss_d_p.utils.kv_cache_table import (
    CHUNK_N_TOKENS,
    KvKind,
    compute_kv_chunk_size_bytes,
    create_kv_chunk_address_table_gpt_oss_prefill,
    global_position_to_sp_local,
)
from tests.ttnn.utils_for_testing import assert_equal

SP_AXIS = 0
TP_AXIS = 1
HEAD_DIM = 64
NUM_KV_HEADS = 2
NUM_LAYERS = 1
MAX_SEQ_LEN = 128


def _attention_config(max_seq_len: int = MAX_SEQ_LEN) -> AttentionConfig:
    return AttentionConfig(
        hidden_size=512,
        num_heads=8,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=max_seq_len,
        max_local_batch_size=1,
    )


def _alloc_kv_caches(mesh_device, num_layers: int = NUM_LAYERS, max_seq_len: int = MAX_SEQ_LEN):
    cfg = _attention_config(max_seq_len)
    return [[*init_kv_cache(mesh_device, cfg)] for _ in range(num_layers)]


def _build_table(mesh_device, kv_caches, max_seq_len: int = MAX_SEQ_LEN):
    return create_kv_chunk_address_table_gpt_oss_prefill(
        mesh_device=mesh_device,
        mesh_shape=mesh_device.shape,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        kv_caches=kv_caches,
        num_transformer_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=max_seq_len,
    )


def _is_k_head_zero_group(group) -> bool:
    return group.kv_kind == KvKind.K and group.global_head == 0


def _fill_cache_chunks(cache, mesh_device, max_seq_len: int, value_fn):
    """Write one 32-token tile per chunk via ``fill_cache`` (prefill layout)."""
    for position in range(0, max_seq_len, CHUNK_N_TOKENS):
        torch_chunk = torch.full(
            (1, 1, CHUNK_N_TOKENS, HEAD_DIM),
            value_fn(position),
            dtype=torch.bfloat16,
        )
        tt_chunk = ttnn.from_torch(
            torch_chunk,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
        )
        ttnn.fill_cache(cache, tt_chunk, batch_idx=0, update_idx=position)
        tt_chunk.deallocate(True)


def _fill_all_kv_caches(kv_caches, mesh_device, max_seq_len: int):
    """Fill every layer's K and V caches with distinct per-chunk patterns."""
    for layer_idx, (k_cache, v_cache) in enumerate(kv_caches):
        k_base = layer_idx * 1_000_000
        v_base = k_base + 100_000
        _fill_cache_chunks(
            k_cache,
            mesh_device,
            max_seq_len,
            value_fn=lambda pos, base=k_base: float(pos) + base,
        )
        _fill_cache_chunks(
            v_cache,
            mesh_device,
            max_seq_len,
            value_fn=lambda pos, base=v_base: float(pos) + base,
        )


def _expected_chunk_from_device(cache, mesh_device, group, layer: int, position: int, max_seq_len: int):
    sp_len = mesh_device.shape[SP_AXIS]
    tp_len = mesh_device.shape[TP_AXIS]
    sp_row, local_position = global_position_to_sp_local(position, max_seq_len, sp_len)
    device_tensors = ttnn.get_device_tensors(cache)
    dt = device_tensors[sp_row * tp_len + group.tp_col]
    torch_cache = ttnn.to_torch(dt).to(torch.bfloat16)
    return torch_cache[:, :, local_position : local_position + CHUNK_N_TOKENS, :]


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],
    ids=["1x2"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="GPT-OSS prefill KV NdShardSpec table tests target Blackhole")
def test_gpt_oss_prefill_kv_table_smoke(mesh_device):
    """Tier 1: build multi-group bundle and assert lookups on every group table."""
    kv_caches = _alloc_kv_caches(mesh_device)
    bundle = _build_table(mesh_device, kv_caches)
    chunk_bytes = compute_kv_chunk_size_bytes(HEAD_DIM)

    assert len(bundle.configs) == 2 * NUM_KV_HEADS
    assert len(bundle.groups) == 2 * NUM_KV_HEADS

    for group in bundle.groups:
        assert group.config.num_slots == 1
        assert group.config.num_layers == NUM_LAYERS
        assert group.config.max_sequence_length == MAX_SEQ_LEN
        assert group.config.chunk_size_bytes == chunk_bytes
        assert group.table.total_entries() == NUM_LAYERS * (MAX_SEQ_LEN // CHUNK_N_TOKENS)

        for layer in range(NUM_LAYERS):
            for position in range(0, MAX_SEQ_LEN, CHUNK_N_TOKENS):
                loc = group.table.lookup(layer, position, slot=0)
                assert loc.size_bytes == chunk_bytes
                assert loc.noc_addr != 0

    k0 = bundle.groups[0]
    assert bundle.unified_table is k0.table
    unified_loc = bundle.unified_table.lookup(0, 0, 0)
    k0_loc = k0.table.lookup(0, 0, 0)
    assert unified_loc.noc_addr == k0_loc.noc_addr
    assert unified_loc.size_bytes == chunk_bytes


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],
    ids=["1x2"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="GPT-OSS prefill KV NdShardSpec table tests target Blackhole")
def test_gpt_oss_prefill_kv_table_readback(mesh_device):
    """Tier 2: read all groups; assert bytes only for K global-head-0."""
    kv_caches = _alloc_kv_caches(mesh_device)
    bundle = _build_table(mesh_device, kv_caches)

    _fill_all_kv_caches(kv_caches, mesh_device, MAX_SEQ_LEN)

    chunk_shape = [1, 1, CHUNK_N_TOKENS, HEAD_DIM]
    for group in bundle.groups:
        is_k_head_zero = _is_k_head_zero_group(group)

        for layer in range(NUM_LAYERS):
            layer_cache = kv_caches[layer][int(group.kv_kind)]
            for position in range(0, MAX_SEQ_LEN, CHUNK_N_TOKENS):
                raw_bytes = group.table.read_device_chunk(layer=layer, position=position, slot=0)
                if not is_k_head_zero:
                    continue
                chunk_tt = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw_bytes, chunk_shape)
                chunk_torch = ttnn.to_torch(chunk_tt).to(torch.bfloat16)
                expected = _expected_chunk_from_device(
                    layer_cache,
                    mesh_device,
                    group,
                    layer=layer,
                    position=position,
                    max_seq_len=MAX_SEQ_LEN,
                )
                assert_equal(expected, chunk_torch)
