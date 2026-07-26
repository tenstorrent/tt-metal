# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in production-geometry GLM-5.2 paged sparse-SDPA validation.

This deliberately bypasses the indexer scorer: deterministic, unique causal
logical IDs stand in for its top-k output, then the production sparse reader
must resolve them through a fragmented two-bundle page table.

Run on an exclusively assigned four-device Blackhole QuietBox:

    TT_SAFE_PYTEST_DISPATCH_TIMEOUT=30 \
      TT_RUN_GLM52_PAGED_SPARSE_SDPA_PROD=1 scripts/run_safe_pytest.sh \
      models/demos/deepseek_v3_d_p/tests/paged_prefill/test_glm52_paged_sparse_sdpa_prod_qb.py -s

The SP4 proxy assigns eight times as many query rows to each chip as the
SP8xTP4 production mesh, so its full-geometry dispatch exceeds the safe
wrapper's five-second default watchdog.
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

BUNDLE_TOKENS = 5120
NUM_HEADS = 64
TOPK = 2048
KVPE_WIDTH = 576
V_DIM = 512
PRIMARY_LAYERS = 78
PHYSICAL_BUNDLES = 3
CACHE_SLOT = 1
LAYER_IDX = PRIMARY_LAYERS - 1
OPT_IN = os.environ.get("TT_RUN_GLM52_PAGED_SPARSE_SDPA_PROD") == "1"

QB_SP4_TP1 = pytest.param(
    (4, 1),
    {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "fabric_tensix_config": ttnn.FabricTensixConfig.UDM,
        "fabric_udm_mode": ttnn.FabricUDMMode.ENABLED,
    },
    id="sp4xtp1-udm",
)


def _sp_mapper(mesh_device):
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, None))


def _upload_sp_sharded(source, mesh_device, dtype):
    return ttnn.from_torch(
        source,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_sp_mapper(mesh_device),
    )


def _shard_query_geometry(source, mesh_device, dtype):
    """Match ttMLA's SP4xTP1 sparse-attention distribution."""

    return _upload_sp_sharded(source, mesh_device, dtype)


def _concat_query_shards(tensor, mesh_device):
    sp, tp = tuple(mesh_device.shape)
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor.cpu())]
    assert len(shards) == sp * tp
    return torch.cat(
        [torch.cat(shards[sp_rank * tp : (sp_rank + 1) * tp], dim=2) for sp_rank in range(sp)],
        dim=2,
    )


def _make_bundle(logical_bundle):
    """Low-rank but width-varying BF16 KVPE makes the full reference bounded."""

    token = torch.arange(BUNDLE_TOKENS, dtype=torch.float32) + logical_bundle * BUNDLE_TOKENS
    token_feature = torch.sin(token * 0.0031) + 0.5 * torch.cos(token * 0.0017)
    width = torch.arange(KVPE_WIDTH, dtype=torch.float32)
    width_feature = torch.sin(width * 0.037) * 0.75 + 0.25
    width_bias = torch.cos(width * 0.019) * 0.125
    return (token_feature[:, None] * width_feature[None, :] + width_bias[None, :]).to(torch.bfloat16)


def _causal_global_ids():
    """Unique IDs spread across the complete causal prefix for every query."""

    query_position = torch.arange(BUNDLE_TOKENS, 2 * BUNDLE_TOKENS, dtype=torch.int64)[:, None]
    rank = torch.arange(TOPK, dtype=torch.int64)[None, :]
    indices = torch.div(rank * query_position, TOPK - 1, rounding_mode="floor")
    assert torch.all(indices[:, 1:] > indices[:, :-1])
    assert torch.all(indices <= query_position)
    assert indices.min().item() == 0
    assert indices.max().item() == 2 * BUNDLE_TOKENS - 1
    return indices.reshape(1, 1, BUNDLE_TOKENS, TOPK)


def _gathered_uniform_reference(logical_v, indices, query_chunk=8):
    """Reference zero-Q attention without materializing [S, topk, V] globally."""

    reference = torch.empty((1, 1, BUNDLE_TOKENS, V_DIM), dtype=torch.float32)
    flat_indices = indices.reshape(BUNDLE_TOKENS, TOPK)
    for start in range(0, BUNDLE_TOKENS, query_chunk):
        end = min(start + query_chunk, BUNDLE_TOKENS)
        gathered = torch.index_select(logical_v, 0, flat_indices[start:end].reshape(-1))
        reference[0, 0, start:end] = gathered.reshape(end - start, TOPK, V_DIM).float().mean(dim=1)
    return reference


def _broadcast_reference_pcc(actual, reference, query_chunk=32):
    """Pearson correlation against a head-broadcast reference with bounded temporaries."""

    count = 0
    sum_x = sum_y = sum_xx = sum_yy = sum_xy = 0.0
    for start in range(0, BUNDLE_TOKENS, query_chunk):
        end = min(start + query_chunk, BUNDLE_TOKENS)
        x = actual[:, :, start:end].float()
        y = reference[:, :, start:end].expand_as(x)
        count += x.numel()
        sum_x += x.sum(dtype=torch.float64).item()
        sum_y += y.sum(dtype=torch.float64).item()
        sum_xx += (x * x).sum(dtype=torch.float64).item()
        sum_yy += (y * y).sum(dtype=torch.float64).item()
        sum_xy += (x * y).sum(dtype=torch.float64).item()
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = sum_xx - sum_x * sum_x / count
    variance_y = sum_yy - sum_y * sum_y / count
    return covariance / (variance_x * variance_y) ** 0.5


@pytest.mark.parametrize("mesh_device,device_params", [QB_SP4_TP1], indirect=["mesh_device", "device_params"])
@pytest.mark.skipif(not OPT_IN, reason="set TT_RUN_GLM52_PAGED_SPARSE_SDPA_PROD=1 on an exclusively assigned QB")
@pytest.mark.skipif(not is_blackhole(), reason="GLM-5.2 sparse prefill is Blackhole-only")
@pytest.mark.timeout(0)
def test_glm52_paged_sparse_sdpa_production_geometry(mesh_device, device_params):
    del device_params
    assert tuple(mesh_device.shape) == (4, 1)
    print("GLM52 sparse-SDPA: allocating production folded pool", flush=True)

    pool = init_kvpe_cache(
        kvpe_cache_head_dim=KVPE_WIDTH,
        mesh_device=mesh_device,
        seq_len=BUNDLE_TOKENS,
        mesh_shape=[4, 1],
        sp_axis=0,
        num_kvpe_cache_layers=PRIMARY_LAYERS,
        num_users=PHYSICAL_BUNDLES,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    assert tuple(pool.shape) == (
        PHYSICAL_BUNDLES * PRIMARY_LAYERS,
        1,
        BUNDLE_TOKENS // 4,
        KVPE_WIDTH,
    )
    page_table_host = torch.tensor([[0, 2], [1, 2]], dtype=torch.int32)
    page_table = ttnn.from_torch(
        page_table_host,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logical_v_parts = []
    for logical_bundle in range(2):
        print(f"GLM52 sparse-SDPA: writing logical bundle {logical_bundle}", flush=True)
        kv_host = _make_bundle(logical_bundle)
        logical_v_parts.append(kv_host[:, :V_DIM])
        kv = _upload_sp_sharded(kv_host.reshape(1, 1, BUNDLE_TOKENS, KVPE_WIDTH), mesh_device, ttnn.bfloat16)
        ttnn.experimental.deepseek_prefill.paged_update_padded_kv_cache(
            pool,
            kv,
            page_table,
            slot_idx=CACHE_SLOT,
            layer_idx=LAYER_IDX,
            num_layers=PRIMARY_LAYERS,
            kv_actual_global=logical_bundle * BUNDLE_TOKENS,
            cluster_axis=0,
        )
        ttnn.deallocate(kv)

    print("GLM52 sparse-SDPA: preparing second-chunk q and causal global IDs", flush=True)
    indices_host = _causal_global_ids()
    q_host = torch.zeros((1, NUM_HEADS, BUNDLE_TOKENS, KVPE_WIDTH), dtype=torch.bfloat16)
    q = _shard_query_geometry(q_host, mesh_device, ttnn.bfloat16)
    indices = _shard_query_geometry(indices_host.to(torch.int32), mesh_device, ttnn.uint32)

    print("GLM52 sparse-SDPA: dispatching production attention", flush=True)
    output = ttnn.transformer.sparse_sdpa(
        q,
        pool,
        indices,
        V_DIM,
        kv_format=ttnn.transformer.SparseKVFormat.BF16,
        scale=KVPE_WIDTH**-0.5,
        k_chunk_size=128,
        cache_batch_idx=CACHE_SLOT,
        page_table=page_table,
        paged_layer_idx=LAYER_IDX,
        paged_sp_axis=0,
    )
    ttnn.synchronize_device(mesh_device)
    print("GLM52 sparse-SDPA: attention complete; gathering output", flush=True)
    actual = _concat_query_shards(output, mesh_device)
    assert actual.shape == (1, NUM_HEADS, BUNDLE_TOKENS, V_DIM)

    logical_v = torch.cat(logical_v_parts, dim=0)
    print("GLM52 sparse-SDPA: computing bounded gathered reference", flush=True)
    reference = _gathered_uniform_reference(logical_v, indices_host)
    output_pcc = _broadcast_reference_pcc(actual, reference)
    print(f"GLM52 sparse-SDPA: PCC={output_pcc:.6f}", flush=True)
    assert output_pcc >= 0.99, f"production paged sparse SDPA PCC {output_pcc:.6f} < 0.99"

    for tensor in (output, indices, q, page_table, pool):
        ttnn.deallocate(tensor)
