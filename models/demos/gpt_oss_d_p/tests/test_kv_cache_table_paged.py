# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the GPT-OSS **paged** prefill KV chunk address table (``kv_cache_table_paged.py``).

These tests validate **migration address tables only** — that
``create_kv_chunk_address_table_gpt_oss_prefill_paged`` records the correct NOC
addresses for each 32-token chunk in SP-sharded, flat-DRAM paged K/V caches
(``kv_cache_prefill_only_paged.py``). They do **not** run the model,
``paged_fill_cache``, or fabric send/recv.

**Tier 1 (smoke):** build the multi-group table and check every
``(layer, position, slot)`` lookup has sane metadata (non-zero NOC addr, expected
entry counts for SP × TP × slots).

**Tier 2 (readback):** allocate via ``init_kv_cache`` (INTERLEAVED DRAM), fill with random
host data (``ttnn.copy`` into the init buffers), then for every (K|V, head) group, slot,
layer, and global position call ``read_paged_device_chunk`` and compare decoded bytes
to the matching slice of the live device tensor.
"""

from __future__ import annotations

import pytest
import torch
from ttnn.device import is_blackhole

import ttnn
from models.demos.gpt_oss_d_p.tt.attention.config import AttentionConfig
from models.demos.gpt_oss_d_p.tt.attention.kv_cache_prefill_only_paged import (
    DEFAULT_PREFILL_PAGE_BLOCK_SIZE,
    init_kv_cache,
    make_paged_attention_config_for_prefill,
    make_paged_kv_memory_config,
)
from models.demos.gpt_oss_d_p.utils.kv_cache_table import (
    CHUNK_N_TOKENS,
    compute_kv_chunk_size_bytes,
    global_head_to_tp_shard,
    global_position_to_sp_local,
)
from models.demos.gpt_oss_d_p.utils.kv_cache_table_paged import (
    canonical_paged_cache_shape,
    create_kv_chunk_address_table_gpt_oss_prefill_paged,
    read_paged_device_chunk,
)
from tests.ttnn.utils_for_testing import assert_equal

SP_AXIS = 0
TP_AXIS = 1
HEAD_DIM = 64
BLOCK_SIZE = DEFAULT_PREFILL_PAGE_BLOCK_SIZE
NUM_LAYERS = 2
MAX_SEQ_LEN = 128

KV_TABLE_MESH_PARAMS = [
    # pytest.param((1, 2), 2, id="1x2-kv2"),
    pytest.param((2, 4), 4, id="2x4-kv4"),
    # pytest.param((4, 8), 8, id="4x8-kv8"),
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


def _paged_attention_config(mesh_device, max_seq_len: int, num_slots: int):
    sp_len = mesh_device.shape[SP_AXIS]
    return make_paged_attention_config_for_prefill(
        max_seq_len=max_seq_len,
        max_local_batch_size=num_slots,
        sp=sp_len,
        block_size=BLOCK_SIZE,
    )


def _sp_shard_mesh_mapper(mesh_device):
    """Mesh mapper matching ``kv_cache_prefill_only_paged.init_kv_cache`` (SP rows, TP replicate)."""
    return ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(SP_AXIS, None),
    )


def _sp_sharded_paged_host_shape(
    mesh_device,
    num_kv_heads: int,
    max_seq_len: int,
    num_slots: int,
) -> tuple[int, ...]:
    """Host tensor shape for ``ShardTensor2dMesh(dims=(0, None))`` paged upload.

    Leading ``sp_len`` dim is split across mesh rows; each device receives
    ``[num_blocks_local, num_local_heads, block_size, head_dim]``.
    """
    sp_len = mesh_device.shape[SP_AXIS]
    tp_len = mesh_device.shape[TP_AXIS]
    num_local_heads = num_kv_heads // tp_len
    paged_cfg = _paged_attention_config(mesh_device, max_seq_len, num_slots)
    return (sp_len, paged_cfg.max_num_blocks, num_local_heads, BLOCK_SIZE, HEAD_DIM)


def _upload_sp_sharded_paged_cache(mesh_device, host: torch.Tensor) -> ttnn.Tensor:
    """Stage host data for ``ttnn.copy`` into an ``init_kv_cache`` paged buffer."""
    return ttnn.from_torch(
        host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=_sp_shard_mesh_mapper(mesh_device),
        memory_config=make_paged_kv_memory_config(),
    )


def _fill_sp_sharded_paged_cache(mesh_device, cache: ttnn.Tensor, host: torch.Tensor) -> None:
    """Overwrite paged ``cache`` (from ``init_kv_cache``) with random ``host`` data."""
    staging = _upload_sp_sharded_paged_cache(mesh_device, host)
    ttnn.copy(staging, cache)
    staging.deallocate(True)


def _alloc_kv_caches(
    mesh_device,
    num_kv_heads: int,
    num_layers: int = NUM_LAYERS,
    max_seq_len: int = MAX_SEQ_LEN,
    num_slots: int = 1,
):
    cfg = _attention_config(num_kv_heads, max_seq_len, num_slots)
    paged_cfg = _paged_attention_config(mesh_device, max_seq_len, num_slots)
    return [
        [
            *init_kv_cache(
                mesh_device,
                cfg,
                paged_cfg,
                sp_axis=SP_AXIS,
                tp_axis=TP_AXIS,
                memory_config=make_paged_kv_memory_config(),
            )
        ]
        for _ in range(num_layers)
    ]


def _alloc_kv_caches_random(
    mesh_device,
    num_kv_heads: int,
    num_layers: int = NUM_LAYERS,
    max_seq_len: int = MAX_SEQ_LEN,
    num_slots: int = 1,
    seed: int = 42,
):
    """Allocate via ``init_kv_cache``, then fill with random host data."""
    kv_caches = _alloc_kv_caches(mesh_device, num_kv_heads, num_layers, max_seq_len, num_slots)
    host_shape = _sp_sharded_paged_host_shape(mesh_device, num_kv_heads, max_seq_len, num_slots)
    for layer_idx in range(num_layers):
        k_gen = torch.Generator().manual_seed(seed + layer_idx * 2)
        v_gen = torch.Generator().manual_seed(seed + layer_idx * 2 + 1)
        k_host = torch.randn(host_shape, generator=k_gen, dtype=torch.bfloat16)
        v_host = torch.randn(host_shape, generator=v_gen, dtype=torch.bfloat16)
        _fill_sp_sharded_paged_cache(mesh_device, kv_caches[layer_idx][0], k_host)
        _fill_sp_sharded_paged_cache(mesh_device, kv_caches[layer_idx][1], v_host)
    return kv_caches


def _build_table(
    mesh_device,
    kv_caches,
    num_kv_heads: int,
    max_seq_len: int = MAX_SEQ_LEN,
    num_slots: int = 1,
):
    return create_kv_chunk_address_table_gpt_oss_prefill_paged(
        mesh_device=mesh_device,
        mesh_shape=mesh_device.shape,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        kv_caches=kv_caches,
        num_transformer_layers=NUM_LAYERS,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        max_seq_len=max_seq_len,
        block_size=BLOCK_SIZE,
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
    num_local_heads = num_kv_heads // tp_len
    _, head_idx_local = global_head_to_tp_shard(group.global_head, num_local_heads)

    sp_row, local_position = global_position_to_sp_local(position, max_seq_len, sp_len)
    device_tensors = ttnn.get_device_tensors(cache)
    dt = device_tensors[sp_row * tp_len + group.tp_col]
    torch_cache = ttnn.to_torch(dt).to(torch.bfloat16)

    num_blocks_local, num_local_heads, _, _ = canonical_paged_cache_shape(tuple(torch_cache.shape))
    torch_cache = torch_cache.reshape(num_blocks_local, num_local_heads, BLOCK_SIZE, HEAD_DIM)

    blocks_per_slot = num_blocks_local // num_slots
    block_in_slot = local_position // BLOCK_SIZE
    token_start = local_position % BLOCK_SIZE
    block_idx = slot * blocks_per_slot + block_in_slot

    chunk = torch_cache[
        block_idx : block_idx + 1,
        head_idx_local : head_idx_local + 1,
        token_start : token_start + CHUNK_N_TOKENS,
        :,
    ]
    return chunk.reshape(1, 1, CHUNK_N_TOKENS, HEAD_DIM)


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
@pytest.mark.skipif(not is_blackhole(), reason="GPT-OSS paged prefill KV table tests target Blackhole")
def test_gpt_oss_prefill_kv_table_paged_smoke(mesh_device, num_kv_heads, num_slots):
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
@pytest.mark.skipif(not is_blackhole(), reason="GPT-OSS paged prefill KV table tests target Blackhole")
def test_gpt_oss_prefill_kv_table_paged_readback(mesh_device, num_kv_heads, num_slots):
    """Tier 2: read every group table and assert bytes match device paged cache."""
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
                    raw_bytes = read_paged_device_chunk(group.table, layer=layer, position=position, slot=slot)
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
