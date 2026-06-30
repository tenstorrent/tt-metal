# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the GPT-OSS prefill KV chunk address table (``kv_cache_table.py``).

These tests validate **migration address tables only** — that
``create_kv_chunk_address_table_gpt_oss_prefill`` records the correct NOC
addresses for each 32-token chunk in SP-sharded, NdShardSpec K/V caches. They
do **not** run the model, ``fill_cache``, or fabric send/recv.

**Tier 1 (smoke):** build the multi-group table and check every
``(layer, position, slot)`` lookup has sane metadata (non-zero NOC addr, expected
entry counts for SP × TP × slots).

**Tier 2 (readback):** upload **random** SP-sharded cache data, then for every
(K|V, head) group, slot, layer, and global position call ``read_device_chunk``
and compare decoded bytes to the matching slice of the live device tensor.
Random data is enough because the test is about **addressing**, not KV semantics:
a wrong SP row, TP column, slot, or bank-walk offset reads unrelated bytes and
``assert_equal`` fails. Structured patterns (zeros, constants) could mask systematic
offsets; arbitrary per-chunk values catch those bugs.
"""

from __future__ import annotations

import pytest
import torch
from ttnn.device import is_blackhole

import ttnn
from models.demos.gpt_oss_d_p.tt.attention.config import AttentionConfig
from models.demos.gpt_oss_d_p.tt.attention.kv_cache_prefill_only import (
    init_kv_cache,
    make_gpt_oss_prefill_kv_memory_config,
)
from models.demos.gpt_oss_d_p.utils.kv_cache_table import (
    CHUNK_N_TOKENS,
    compute_kv_chunk_size_bytes,
    create_kv_chunk_address_table_gpt_oss_prefill,
    global_position_to_sp_local,
)
from tests.ttnn.utils_for_testing import assert_equal

SP_AXIS = 0
TP_AXIS = 1
HEAD_DIM = 64
NUM_LAYERS = 2
MAX_SEQ_LEN = 128

KV_TABLE_MESH_PARAMS = [
    pytest.param((1, 2), 2, id="1x2-kv2"),
    pytest.param((2, 4), 4, id="2x4-kv4"),
    pytest.param((4, 8), 8, id="4x8-kv8"),
]

NUM_SLOTS_PARAMS = [
    pytest.param(1, id="1slot"),
    pytest.param(2, id="2slots"),
]


def _attention_config(
    num_kv_heads: int,
    max_seq_len: int = MAX_SEQ_LEN,
    num_slots: int = 1,
) -> AttentionConfig:
    return AttentionConfig(
        hidden_size=512,
        num_heads=max(8, num_kv_heads * 2),
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        max_seq_len=max_seq_len,
        max_local_batch_size=num_slots,
    )


def _sp_shard_mesh_mapper(mesh_device):
    """Mesh mapper matching ``kv_cache_prefill_only.init_kv_cache`` (SP rows, TP replicate)."""
    return ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(SP_AXIS, None),
    )


def _sp_sharded_host_shape(
    mesh_device,
    num_kv_heads: int,
    max_seq_len: int,
    num_slots: int,
) -> tuple[int, ...]:
    """Host tensor shape for ``ShardTensor2dMesh(dims=(0, None))`` upload.

    Leading ``sp_len`` dim is split across mesh rows; each device receives
    ``[num_slots, num_local_heads, seq_local, head_dim]`` — same layout as prefill KV init.
    """
    sp_len = mesh_device.shape[SP_AXIS]
    tp_len = mesh_device.shape[TP_AXIS]
    seq_local = max_seq_len // sp_len
    num_local_heads = num_kv_heads // tp_len
    return (sp_len, num_slots, num_local_heads, seq_local, HEAD_DIM)


def _upload_sp_sharded_cache(mesh_device, host: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=_sp_shard_mesh_mapper(mesh_device),
        memory_config=make_gpt_oss_prefill_kv_memory_config(HEAD_DIM),
    )


def _alloc_kv_caches(
    mesh_device,
    num_kv_heads: int,
    num_layers: int = NUM_LAYERS,
    max_seq_len: int = MAX_SEQ_LEN,
    num_slots: int = 1,
):
    cfg = _attention_config(num_kv_heads, max_seq_len, num_slots)
    return [[*init_kv_cache(mesh_device, cfg)] for _ in range(num_layers)]


def _alloc_kv_caches_random(
    mesh_device,
    num_kv_heads: int,
    num_layers: int = NUM_LAYERS,
    max_seq_len: int = MAX_SEQ_LEN,
    num_slots: int = 1,
    seed: int = 42,
):
    """Allocate SP-sharded K/V caches with random host data (DeepSeek-style table readback)."""
    host_shape = _sp_sharded_host_shape(mesh_device, num_kv_heads, max_seq_len, num_slots)
    kv_caches = []
    for layer_idx in range(num_layers):
        k_gen = torch.Generator().manual_seed(seed + layer_idx * 2)
        v_gen = torch.Generator().manual_seed(seed + layer_idx * 2 + 1)
        k_host = torch.randn(host_shape, generator=k_gen, dtype=torch.bfloat16)
        v_host = torch.randn(host_shape, generator=v_gen, dtype=torch.bfloat16)
        kv_caches.append([_upload_sp_sharded_cache(mesh_device, k_host), _upload_sp_sharded_cache(mesh_device, v_host)])
    return kv_caches


def _build_table(
    mesh_device,
    kv_caches,
    num_kv_heads: int,
    max_seq_len: int = MAX_SEQ_LEN,
    num_slots: int = 1,
):
    return create_kv_chunk_address_table_gpt_oss_prefill(
        mesh_device=mesh_device,
        mesh_shape=mesh_device.shape,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        kv_caches=kv_caches,
        num_transformer_layers=NUM_LAYERS,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        max_seq_len=max_seq_len,
        num_slots=num_slots,
    )


def _expected_chunk_from_device(
    cache,
    mesh_device,
    group,
    *,
    slot: int,
    num_kv_heads: int,
    num_slots: int,
    layer: int,
    position: int,
    max_seq_len: int,
):
    sp_len = mesh_device.shape[SP_AXIS]
    tp_len = mesh_device.shape[TP_AXIS]
    seq_local = max_seq_len // sp_len
    num_local_heads = num_kv_heads // tp_len
    sp_row, local_position = global_position_to_sp_local(position, max_seq_len, sp_len)
    device_tensors = ttnn.get_device_tensors(cache)
    dt = device_tensors[sp_row * tp_len + group.tp_col]
    torch_cache = ttnn.to_torch(dt).to(torch.bfloat16)
    # to_torch rank/order varies (e.g. [1, B, 1, S, D] vs [B, 1, S, D]); flatten to canonical
    # [num_slots, H_local, seq_local, head_dim] before selecting the migration slot (batch row).
    torch_cache = torch_cache.reshape(num_slots, num_local_heads, seq_local, HEAD_DIM)
    chunk = torch_cache[slot : slot + 1, :, local_position : local_position + CHUNK_N_TOKENS, :]
    return chunk.reshape(1, 1, CHUNK_N_TOKENS, HEAD_DIM)


# sp x tp; num_kv_heads is global GQA head count (must be divisible by TP)
@pytest.mark.parametrize("num_slots", NUM_SLOTS_PARAMS)
@pytest.mark.parametrize(
    "mesh_device, num_kv_heads",
    KV_TABLE_MESH_PARAMS,
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="GPT-OSS prefill KV NdShardSpec table tests target Blackhole")
def test_gpt_oss_prefill_kv_table_smoke(mesh_device, num_kv_heads, num_slots):
    """Tier 1: build multi-group bundle and assert lookups on every group table."""
    kv_caches = _alloc_kv_caches(mesh_device, num_kv_heads, num_slots=num_slots)
    bundle = _build_table(mesh_device, kv_caches, num_kv_heads, num_slots=num_slots)
    chunk_bytes = compute_kv_chunk_size_bytes(HEAD_DIM)
    chunks_per_seq = MAX_SEQ_LEN // CHUNK_N_TOKENS

    assert len(bundle.configs) == 2 * num_kv_heads
    assert len(bundle.groups) == 2 * num_kv_heads

    for group in bundle.groups:
        assert group.config.num_slots == num_slots
        assert group.config.num_layers == NUM_LAYERS
        assert group.config.max_sequence_length == MAX_SEQ_LEN
        assert group.config.chunk_size_bytes == chunk_bytes
        assert group.table.total_entries() == NUM_LAYERS * chunks_per_seq * num_slots

        for slot in range(num_slots):
            for layer in range(NUM_LAYERS):
                for position in range(0, MAX_SEQ_LEN, CHUNK_N_TOKENS):
                    loc = group.table.lookup(layer, position, slot=slot)
                    assert loc.size_bytes == chunk_bytes
                    assert loc.noc_addr != 0


# sp x tp; num_kv_heads is global GQA head count (must be divisible by TP), not the TP degree
@pytest.mark.parametrize("num_slots", NUM_SLOTS_PARAMS)
@pytest.mark.parametrize(
    "mesh_device, num_kv_heads",
    KV_TABLE_MESH_PARAMS,
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="GPT-OSS prefill KV NdShardSpec table tests target Blackhole")
def test_gpt_oss_prefill_kv_table_readback(mesh_device, num_kv_heads, num_slots):
    """Tier 2: read every group table and assert bytes match device cache."""
    kv_caches = _alloc_kv_caches_random(mesh_device, num_kv_heads, num_slots=num_slots)
    bundle = _build_table(mesh_device, kv_caches, num_kv_heads, num_slots=num_slots)

    chunks_per_seq = MAX_SEQ_LEN // CHUNK_N_TOKENS
    expected_comparisons = 2 * num_kv_heads * NUM_LAYERS * chunks_per_seq * num_slots

    chunk_shape = [1, 1, CHUNK_N_TOKENS, HEAD_DIM]
    comparisons = 0
    for group in bundle.groups:
        for slot in range(num_slots):
            for layer in range(NUM_LAYERS):
                layer_cache = kv_caches[layer][int(group.kv_kind)]
                for position in range(0, MAX_SEQ_LEN, CHUNK_N_TOKENS):
                    raw_bytes = group.table.read_device_chunk(layer=layer, position=position, slot=slot)
                    assert len(raw_bytes) == group.config.chunk_size_bytes
                    chunk_tt = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw_bytes, chunk_shape)
                    chunk_torch = ttnn.to_torch(chunk_tt).to(torch.bfloat16)
                    expected = _expected_chunk_from_device(
                        layer_cache,
                        mesh_device,
                        group,
                        slot=slot,
                        num_kv_heads=num_kv_heads,
                        num_slots=num_slots,
                        layer=layer,
                        position=position,
                        max_seq_len=MAX_SEQ_LEN,
                    )
                    assert_equal(expected, chunk_torch)
                    comparisons += 1

    assert comparisons == expected_comparisons
