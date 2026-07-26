# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused real-device coverage for the GLM-5.2 paged-prefill primitives.

Run on an exclusively assigned four-device Blackhole QuietBox:

    TT_RUN_PAGED_PREFILL_QB_TESTS=1 \
      scripts/run_safe_pytest.sh \
      models/demos/deepseek_v3_d_p/tests/paged_prefill/test_paged_ops_qb.py -s

The pools use the production folded geometries: 78 BF16 KVPE layers and 21
BFP8 index-key layers.  Slot 1's only logical bundle maps to physical bundle 2,
so every check exercises compact-table translation rather than identity
addressing.
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import golden

GLM_BUNDLE_TOKENS = 5120
GLM_PRIMARY_LAYERS = 78
GLM_INDEX_LAYERS = 21
KVPE_HEAD_DIM = 576
INDEX_HEAD_DIM = 128
PHYSICAL_BUNDLES = 3
LOGICAL_SLOT = 1
PHYSICAL_BUNDLE = 2

QB_CASES = [
    pytest.param(
        (4, 1),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_tensix_config": ttnn.FabricTensixConfig.UDM,
            "fabric_udm_mode": ttnn.FabricUDMMode.ENABLED,
        },
        id="sp4xtp1-ring-udm",
    ),
]

QB_OPT_IN = os.environ.get("TT_RUN_PAGED_PREFILL_QB_TESTS") == "1"

pytestmark = [
    pytest.mark.skipif(not QB_OPT_IN, reason="set TT_RUN_PAGED_PREFILL_QB_TESTS=1 on an exclusively assigned QB"),
    pytest.mark.skipif(not is_blackhole(), reason="GLM-5.2 sparse prefill is Blackhole-only"),
    pytest.mark.timeout(0),
]


def _page_table(mesh_device):
    # Row 0 deliberately differs from the tested row.  One entry represents a
    # complete 5120-token bundle, not 160 individual 32-token subpages.
    table = torch.tensor([[0], [PHYSICAL_BUNDLE]], dtype=torch.int32)
    return ttnn.from_torch(
        table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _two_bundle_page_table(mesh_device):
    table = torch.tensor([[0, 2], [1, 2]], dtype=torch.int32)
    return ttnn.from_torch(
        table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _pool(mesh_device, width, layers, dtype, layout):
    sp, tp = tuple(mesh_device.shape)
    return init_kvpe_cache(
        kvpe_cache_head_dim=width,
        mesh_device=mesh_device,
        seq_len=GLM_BUNDLE_TOKENS,
        mesh_shape=[sp, tp],
        sp_axis=0,
        num_kvpe_cache_layers=layers,
        num_users=PHYSICAL_BUNDLES,
        dtype=dtype,
        layout=layout,
    )


def _sp_mapper(mesh_device):
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, None))


def _sp_composer(mesh_device):
    return ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 1))


def _upload_sp_sharded(source, mesh_device, dtype, layout):
    return ttnn.from_torch(
        source,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_sp_mapper(mesh_device),
    )


def _write_bundle(pool, source, table, layer_idx, num_layers, kv_actual_global=0):
    ttnn.experimental.deepseek_prefill.paged_update_padded_kv_cache(
        pool,
        source,
        table,
        slot_idx=LOGICAL_SLOT,
        layer_idx=layer_idx,
        num_layers=num_layers,
        kv_actual_global=kv_actual_global,
        cluster_axis=0,
    )


def _physical_layer(pool, layer_idx, num_layers, physical_bundle=PHYSICAL_BUNDLE):
    batch = physical_bundle * num_layers + layer_idx
    return ttnn.slice(
        pool,
        [batch, 0, 0, 0],
        [batch + 1, 1, pool.shape[2], pool.shape[3]],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _concat_seq_shards(tensor, mesh_device):
    """Reassemble tensors whose sequence was split over SP, then TP."""
    sp, tp = tuple(mesh_device.shape)
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor.cpu())]
    assert len(shards) == sp * tp
    per_sp = [torch.cat(shards[row * tp : (row + 1) * tp], dim=2) for row in range(sp)]
    return torch.cat(per_sp, dim=2)


def _shard_seq_over_qb(source, mesh_device, dtype, layout):
    tensor = _upload_sp_sharded(source, mesh_device, dtype, layout)
    if mesh_device.shape[1] > 1:
        partitioned = ttnn.mesh_partition(tensor, dim=2, cluster_axis=1)
        ttnn.deallocate(tensor)
        tensor = partitioned
    return tensor


def _pcc(actual, expected):
    return torch.corrcoef(torch.stack([actual.flatten().float(), expected.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize(
    "mesh_device,device_params",
    QB_CASES,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "width,layers,layer_idx,dtype,layout",
    [
        pytest.param(
            KVPE_HEAD_DIM,
            GLM_PRIMARY_LAYERS,
            GLM_PRIMARY_LAYERS - 1,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            id="primary-bf16-rm-78-layers",
        ),
        pytest.param(
            INDEX_HEAD_DIM,
            GLM_INDEX_LAYERS,
            GLM_INDEX_LAYERS - 1,
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            id="index-bfp8-tile-21-layers",
        ),
    ],
)
def test_paged_writer_production_folded_pool(
    mesh_device,
    device_params,
    width,
    layers,
    layer_idx,
    dtype,
    layout,
):
    del device_params
    pool = _pool(mesh_device, width, layers, dtype, layout)
    table = _page_table(mesh_device)
    generator = torch.Generator().manual_seed(2000 + width)
    source_host = torch.randn(
        1,
        1,
        GLM_BUNDLE_TOKENS,
        width,
        generator=generator,
        dtype=torch.bfloat16,
    )
    source = _upload_sp_sharded(source_host, mesh_device, dtype, layout)
    # Read back the encoded source, so the BFP8 case tests addressing/copying
    # independently of the expected block-float quantization loss.
    expected = ttnn.to_torch(source, mesh_composer=_sp_composer(mesh_device)).to(torch.bfloat16)

    _write_bundle(pool, source, table, layer_idx, layers)
    selected = _physical_layer(pool, layer_idx, layers)
    actual = ttnn.to_torch(selected, mesh_composer=_sp_composer(mesh_device)).to(torch.bfloat16)

    assert torch.equal(actual, expected), (
        f"paged writer did not place slot {LOGICAL_SLOT} in physical bundle {PHYSICAL_BUNDLE}; "
        f"max diff={(actual.float() - expected.float()).abs().max().item()}"
    )

    ttnn.deallocate(selected)
    ttnn.deallocate(source)
    ttnn.deallocate(table)
    ttnn.deallocate(pool)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    QB_CASES,
    indirect=["mesh_device", "device_params"],
)
def test_paged_indexer_streaming_topk_merges_two_bundles(mesh_device, device_params):
    del device_params
    layer_idx = GLM_INDEX_LAYERS - 1
    pool = _pool(mesh_device, INDEX_HEAD_DIM, GLM_INDEX_LAYERS, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    table = _two_bundle_page_table(mesh_device)
    generator = torch.Generator().manual_seed(3311)
    key_parts = [
        torch.randn(1, 1, GLM_BUNDLE_TOKENS, INDEX_HEAD_DIM, generator=generator, dtype=torch.bfloat16)
        for _ in range(2)
    ]
    encoded_parts = []
    for bundle, key_host in enumerate(key_parts):
        key = _upload_sp_sharded(key_host, mesh_device, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        encoded_parts.append(ttnn.to_torch(key, mesh_composer=_sp_composer(mesh_device)).to(torch.bfloat16))
        _write_bundle(
            pool,
            key,
            table,
            layer_idx,
            GLM_INDEX_LAYERS,
            kv_actual_global=bundle * GLM_BUNDLE_TOKENS,
        )
        ttnn.deallocate(key)
    for physical_bundle, expected in zip((1, 2), encoded_parts):
        selected = _physical_layer(pool, layer_idx, GLM_INDEX_LAYERS, physical_bundle)
        actual = ttnn.to_torch(selected, mesh_composer=_sp_composer(mesh_device)).to(torch.bfloat16)
        assert torch.equal(actual, expected)
        ttnn.deallocate(selected)

    fixed_keys = [
        ttnn.from_torch(
            key_part,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for key_part in key_parts
    ]
    query_tokens = GLM_BUNDLE_TOKENS
    q_host = torch.randn(1, 8, query_tokens, INDEX_HEAD_DIM, generator=generator, dtype=torch.bfloat16)
    weights_host = torch.randn(1, 8, query_tokens, 1, generator=generator, dtype=torch.bfloat16)
    q = _shard_seq_over_qb(q_host, mesh_device, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    weights = _shard_seq_over_qb(weights_host, mesh_device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    seq_axes = [0, 1] if mesh_device.shape[1] > 1 else [0]
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=512, head_group_size=0)

    dense_score_parts = [
        ttnn.experimental.indexer_score_dsa(
            q,
            fixed_key,
            weights,
            chunk_start_idx=None if bundle == 0 else 0,
            kv_len=GLM_BUNDLE_TOKENS,
            seq_shard_axes=seq_axes,
            block_cyclic_sp_axis=0,
            block_cyclic_chunk_local=GLM_BUNDLE_TOKENS // mesh_device.shape[0],
            program_config=cfg,
        )
        for bundle, fixed_key in enumerate(fixed_keys)
    ]
    dense_score_hosts = [_concat_seq_shards(score, mesh_device).float() for score in dense_score_parts]
    paged_score_hosts = []
    for stripe_start in (0, GLM_BUNDLE_TOKENS):
        paged_score = ttnn.experimental.indexer_score_dsa_paged(
            q,
            pool,
            table,
            weights,
            layer_idx=layer_idx,
            num_layers=GLM_INDEX_LAYERS,
            kv_len=stripe_start + GLM_BUNDLE_TOKENS,
            kv_start=stripe_start,
            chunk_start_idx=GLM_BUNDLE_TOKENS,
            paged_sp_axis=0,
            seq_shard_axes=seq_axes,
            cache_slot=LOGICAL_SLOT,
            program_config=cfg,
        )
        paged_host = _concat_seq_shards(paged_score, mesh_device).float()
        paged_score_hosts.append(paged_host)
        dense_stripe = dense_score_hosts[stripe_start // GLM_BUNDLE_TOKENS]
        masked = dense_stripe <= torch.finfo(torch.bfloat16).min
        if stripe_start:
            assert torch.equal(paged_host <= torch.finfo(torch.bfloat16).min, masked)
        score_pcc = _pcc(paged_host[~masked], dense_stripe[~masked])
        assert score_pcc >= 0.999, f"paged bundle {stripe_start // GLM_BUNDLE_TOKENS} PCC {score_pcc:.6f}"
        ttnn.deallocate(paged_score)
    for tensor in (*dense_score_parts, weights, q, *fixed_keys, table, pool):
        ttnn.deallocate(tensor)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    QB_CASES,
    indirect=["mesh_device", "device_params"],
)
def test_paged_indexer_reads_nonidentity_production_pool(mesh_device, device_params):
    del device_params
    layer_idx = GLM_INDEX_LAYERS - 1
    pool = _pool(mesh_device, INDEX_HEAD_DIM, GLM_INDEX_LAYERS, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    table = _page_table(mesh_device)

    generator = torch.Generator().manual_seed(3301)
    key_host = torch.randn(1, 1, GLM_BUNDLE_TOKENS, INDEX_HEAD_DIM, generator=generator, dtype=torch.bfloat16)
    key = _upload_sp_sharded(key_host, mesh_device, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    _write_bundle(pool, key, table, layer_idx, GLM_INDEX_LAYERS)
    ttnn.deallocate(key)
    fixed_key = ttnn.from_torch(
        key_host,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # GLM prefill always computes a 5120-token chunk.  The production model
    # divides its query rows over SP and TP while keeping all 8 index heads.
    q_host = torch.randn(1, 8, GLM_BUNDLE_TOKENS, INDEX_HEAD_DIM, generator=generator, dtype=torch.bfloat16)
    weights_host = torch.randn(1, 8, GLM_BUNDLE_TOKENS, 1, generator=generator, dtype=torch.bfloat16)
    q = _shard_seq_over_qb(q_host, mesh_device, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    weights = _shard_seq_over_qb(weights_host, mesh_device, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    score = ttnn.experimental.indexer_score_dsa_paged(
        q,
        pool,
        table,
        weights,
        layer_idx=layer_idx,
        num_layers=GLM_INDEX_LAYERS,
        kv_len=GLM_BUNDLE_TOKENS,
        chunk_start_idx=0,
        paged_sp_axis=0,
        seq_shard_axes=[0, 1] if mesh_device.shape[1] > 1 else [0],
        program_config=ttnn.IndexerScoreProgramConfig(
            q_chunk_size=64,
            k_chunk_size=512,
            head_group_size=0,
        ),
        cache_slot=LOGICAL_SLOT,
    )
    actual = _concat_seq_shards(score, mesh_device)

    # The established dense reader is the oracle for the unchanged scoring
    # compute.  Both paths consume the same BFP8-encoded logical K and the same
    # q/weights; this comparison isolates compact-table + remote-read mapping.
    fixed_score = ttnn.experimental.indexer_score_dsa(
        q,
        fixed_key,
        weights,
        chunk_start_idx=0,
        kv_len=GLM_BUNDLE_TOKENS,
        seq_shard_axes=[0, 1] if mesh_device.shape[1] > 1 else [0],
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=GLM_BUNDLE_TOKENS // mesh_device.shape[0],
        program_config=ttnn.IndexerScoreProgramConfig(
            q_chunk_size=64,
            k_chunk_size=512,
            head_group_size=0,
        ),
    )
    reference = _concat_seq_shards(fixed_score, mesh_device)
    masked = reference <= torch.finfo(torch.bfloat16).min
    assert torch.equal(actual <= torch.finfo(torch.bfloat16).min, masked)
    score_pcc = _pcc(actual[~masked], reference[~masked])
    assert score_pcc >= 0.999, f"paged indexer PCC {score_pcc:.6f} < 0.999"

    # A nonzero logical stripe must preserve absolute causal positions while
    # materializing only its compact width. This is the bounded-memory bridge
    # used by streaming top-k validation.
    stripe_start = GLM_BUNDLE_TOKENS // 2
    striped = ttnn.experimental.indexer_score_dsa_paged(
        q,
        pool,
        table,
        weights,
        layer_idx=layer_idx,
        num_layers=GLM_INDEX_LAYERS,
        kv_len=GLM_BUNDLE_TOKENS,
        kv_start=stripe_start,
        chunk_start_idx=0,
        paged_sp_axis=0,
        seq_shard_axes=[0, 1] if mesh_device.shape[1] > 1 else [0],
        program_config=ttnn.IndexerScoreProgramConfig(
            q_chunk_size=64,
            k_chunk_size=512,
            head_group_size=0,
        ),
        cache_slot=LOGICAL_SLOT,
    )
    striped_actual = _concat_seq_shards(striped, mesh_device)
    striped_reference = reference[..., stripe_start:]
    striped_mask = striped_reference <= torch.finfo(torch.bfloat16).min
    assert striped_actual.shape == striped_reference.shape
    assert torch.equal(striped_actual <= torch.finfo(torch.bfloat16).min, striped_mask)
    stripe_pcc = _pcc(striped_actual[~striped_mask], striped_reference[~striped_mask])
    assert stripe_pcc >= 0.999, f"paged stripe PCC {stripe_pcc:.6f} < 0.999"

    ttnn.deallocate(striped)
    ttnn.deallocate(fixed_score)
    ttnn.deallocate(fixed_key)
    ttnn.deallocate(score)
    ttnn.deallocate(weights)
    ttnn.deallocate(q)
    ttnn.deallocate(table)
    ttnn.deallocate(pool)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    QB_CASES,
    indirect=["mesh_device", "device_params"],
)
def test_fixed_indexer_production_shape_timeout_baseline(mesh_device, device_params):
    """Prove the unchanged 5120x5120 scorer completes inside the safe wrapper's device timeout."""
    del device_params
    generator = torch.Generator().manual_seed(3302)
    key = ttnn.from_torch(
        torch.randn(1, 1, GLM_BUNDLE_TOKENS, INDEX_HEAD_DIM, generator=generator, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    q = _shard_seq_over_qb(
        torch.randn(1, 8, GLM_BUNDLE_TOKENS, INDEX_HEAD_DIM, generator=generator, dtype=torch.bfloat16),
        mesh_device,
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
    )
    weights = _shard_seq_over_qb(
        torch.randn(1, 8, GLM_BUNDLE_TOKENS, 1, generator=generator, dtype=torch.bfloat16),
        mesh_device,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
    )
    score = ttnn.experimental.indexer_score_dsa(
        q,
        key,
        weights,
        chunk_start_idx=0,
        kv_len=GLM_BUNDLE_TOKENS,
        seq_shard_axes=[0, 1] if mesh_device.shape[1] > 1 else [0],
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=GLM_BUNDLE_TOKENS // mesh_device.shape[0],
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=512, head_group_size=0),
    )
    actual = _concat_seq_shards(score, mesh_device)
    assert actual.shape == (1, 1, GLM_BUNDLE_TOKENS, GLM_BUNDLE_TOKENS)
    ttnn.deallocate(score)
    ttnn.deallocate(weights)
    ttnn.deallocate(q)
    ttnn.deallocate(key)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    QB_CASES,
    indirect=["mesh_device", "device_params"],
)
def test_paged_indexer_remote_read_smoke(mesh_device, device_params):
    """Short actual KV prefix with the unchanged 5120-token GLM compute chunk."""
    del device_params
    kv_len = 512
    layer_idx = GLM_INDEX_LAYERS - 1
    pool = _pool(mesh_device, INDEX_HEAD_DIM, GLM_INDEX_LAYERS, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    table = _page_table(mesh_device)
    generator = torch.Generator().manual_seed(3303)
    full_key_host = torch.randn(1, 1, GLM_BUNDLE_TOKENS, INDEX_HEAD_DIM, generator=generator, dtype=torch.bfloat16)
    full_key = _upload_sp_sharded(full_key_host, mesh_device, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    _write_bundle(pool, full_key, table, layer_idx, GLM_INDEX_LAYERS)
    ttnn.deallocate(full_key)
    q = _shard_seq_over_qb(
        torch.randn(1, 8, GLM_BUNDLE_TOKENS, INDEX_HEAD_DIM, generator=generator, dtype=torch.bfloat16),
        mesh_device,
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
    )
    weights = _shard_seq_over_qb(
        torch.randn(1, 8, GLM_BUNDLE_TOKENS, 1, generator=generator, dtype=torch.bfloat16),
        mesh_device,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
    )
    config = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=256, head_group_size=0)
    paged = ttnn.experimental.indexer_score_dsa_paged(
        q,
        pool,
        table,
        weights,
        layer_idx=layer_idx,
        num_layers=GLM_INDEX_LAYERS,
        kv_len=kv_len,
        chunk_start_idx=0,
        paged_sp_axis=0,
        seq_shard_axes=[0, 1] if tuple(mesh_device.shape)[1] > 1 else [0],
        program_config=config,
        cache_slot=LOGICAL_SLOT,
    )
    actual = _concat_seq_shards(paged, mesh_device)
    assert actual.shape == (1, 1, GLM_BUNDLE_TOKENS, kv_len)
    active_scores = actual[0, 0, :kv_len]
    # Indexer output is causal: the diagonal/lower triangle is computed and
    # the strict upper triangle is intentionally -inf.
    assert torch.isfinite(torch.diagonal(active_scores)).all()
    assert torch.isfinite(active_scores[-1]).all()
    assert torch.isneginf(active_scores[0, 1:]).all()
    for tensor in (paged, weights, q, table, pool):
        ttnn.deallocate(tensor)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    QB_CASES,
    indirect=["mesh_device", "device_params"],
)
def test_paged_sparse_sdpa_reads_nonidentity_production_pool(mesh_device, device_params):
    del device_params
    layer_idx = GLM_PRIMARY_LAYERS - 1
    pool = _pool(mesh_device, KVPE_HEAD_DIM, GLM_PRIMARY_LAYERS, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    table = _page_table(mesh_device)

    generator = torch.Generator().manual_seed(4401)
    kv_host = torch.randn(1, 1, GLM_BUNDLE_TOKENS, KVPE_HEAD_DIM, generator=generator, dtype=torch.bfloat16)
    kv = _upload_sp_sharded(kv_host, mesh_device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    _write_bundle(pool, kv, table, layer_idx, GLM_PRIMARY_LAYERS)
    ttnn.deallocate(kv)

    # Keep the attention test small while selecting keys across every SP owner.
    # Query rows are still tiled and distributed the same way as the model.
    heads, seq, topk, v_dim = 32, 128, 32, 512
    q_host = torch.randn(1, heads, seq, KVPE_HEAD_DIM, generator=generator, dtype=torch.bfloat16)
    indices_host = torch.empty(1, 1, seq, topk, dtype=torch.int64)
    for row in range(seq):
        # Non-contiguous, deterministic positions spanning the full logical
        # bundle, including all position-dependent SP owner regions.
        indices_host[0, 0, row] = (torch.arange(topk, dtype=torch.int64) * 157 + row * 29) % GLM_BUNDLE_TOKENS

    q = _shard_seq_over_qb(q_host, mesh_device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    indices = _shard_seq_over_qb(indices_host.to(torch.int32), mesh_device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.transformer.sparse_sdpa(
        q,
        pool,
        indices,
        v_dim,
        kv_format=ttnn.transformer.SparseKVFormat.BF16,
        scale=KVPE_HEAD_DIM**-0.5,
        k_chunk_size=32,
        cache_batch_idx=LOGICAL_SLOT,
        page_table=table,
        paged_layer_idx=layer_idx,
        paged_sp_axis=0,
    )
    actual = _concat_seq_shards(output, mesh_device)
    reference = golden(
        q_host.float(),
        kv_host.float(),
        indices_host,
        KVPE_HEAD_DIM**-0.5,
        v_dim,
    )
    output_pcc = _pcc(actual, reference)
    assert output_pcc >= 0.99, f"paged sparse SDPA PCC {output_pcc:.6f} < 0.99"

    ttnn.deallocate(output)
    ttnn.deallocate(indices)
    ttnn.deallocate(q)
    ttnn.deallocate(table)
    ttnn.deallocate(pool)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    QB_CASES,
    indirect=["mesh_device", "device_params"],
)
def test_paged_sparse_sdpa_reads_fragmented_two_bundle_pool(mesh_device, device_params):
    del device_params
    layer_idx = GLM_PRIMARY_LAYERS - 1
    pool = _pool(mesh_device, KVPE_HEAD_DIM, GLM_PRIMARY_LAYERS, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    table = _two_bundle_page_table(mesh_device)

    generator = torch.Generator().manual_seed(4402)
    kv_hosts = []
    for logical_bundle in range(2):
        kv_host = torch.randn(
            1,
            1,
            GLM_BUNDLE_TOKENS,
            KVPE_HEAD_DIM,
            generator=generator,
            dtype=torch.bfloat16,
        )
        kv = _upload_sp_sharded(kv_host, mesh_device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        _write_bundle(
            pool,
            kv,
            table,
            layer_idx,
            GLM_PRIMARY_LAYERS,
            kv_actual_global=logical_bundle * GLM_BUNDLE_TOKENS,
        )
        ttnn.deallocate(kv)
        kv_hosts.append(kv_host)

    heads, seq, topk, v_dim = 32, 128, 32, 512
    q_host = torch.randn(1, heads, seq, KVPE_HEAD_DIM, generator=generator, dtype=torch.bfloat16)
    indices_host = torch.empty(1, 1, seq, topk, dtype=torch.int64)
    for row in range(seq):
        # Alternate bundles within each 32-key chunk while covering every SP
        # owner.  Slot 1 maps logical bundles [0,1] to physical bundles [1,2].
        base = (torch.arange(topk, dtype=torch.int64) * 317 + row * 43) % GLM_BUNDLE_TOKENS
        bundle = torch.arange(topk, dtype=torch.int64) % 2
        indices_host[0, 0, row] = base + bundle * GLM_BUNDLE_TOKENS

    q = _shard_seq_over_qb(q_host, mesh_device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    indices = _shard_seq_over_qb(indices_host.to(torch.int32), mesh_device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.transformer.sparse_sdpa(
        q,
        pool,
        indices,
        v_dim,
        kv_format=ttnn.transformer.SparseKVFormat.BF16,
        scale=KVPE_HEAD_DIM**-0.5,
        k_chunk_size=32,
        cache_batch_idx=LOGICAL_SLOT,
        page_table=table,
        paged_layer_idx=layer_idx,
        paged_sp_axis=0,
    )
    actual = _concat_seq_shards(output, mesh_device)
    logical_kv = torch.cat(kv_hosts, dim=2)
    reference = golden(
        q_host.float(),
        logical_kv.float(),
        indices_host,
        KVPE_HEAD_DIM**-0.5,
        v_dim,
    )
    output_pcc = _pcc(actual, reference)
    assert output_pcc >= 0.99, f"two-bundle paged sparse SDPA PCC {output_pcc:.6f} < 0.99"

    ttnn.deallocate(output)
    ttnn.deallocate(indices)
    ttnn.deallocate(q)
    ttnn.deallocate(table)
    ttnn.deallocate(pool)
