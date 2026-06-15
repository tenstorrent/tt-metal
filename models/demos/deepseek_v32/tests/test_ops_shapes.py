# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shape tests for the missing-op APIs (status.md "Missing op APIs", agreement 10).

These pin the agreed signatures and output shapes only; numerics vs
reference_cpu come after. Bodies may be composed/CPU-fallback/stub — shapes are
the contract a future fused op must match. Tensors are replicated across the
mesh (indexer is replicated under TP; SP sharding tested at integration).
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v32.tt import ops

pytestmark = pytest.mark.dev  # fast op-contract tests — inner loop

H_IDX, D_IDX = 64, 128
KVPE_DIM, KV_RANK = 576, 512


def _dev(t, mesh_device):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("sq,skv", [(128, 128), (128, 512)], ids=["s128", "s128kv512"])
def test_indexer_logits_shape(mesh_device, sq, skv):
    q = _dev(torch.randn(1, H_IDX, sq, D_IDX), mesh_device)
    k = _dev(torch.randn(1, 1, skv, D_IDX), mesh_device)
    w = _dev(torch.randn(1, 1, sq, H_IDX), mesh_device)
    logits = ops.indexer_logits(q, k, w)
    assert list(logits.shape) == [1, 1, sq, skv]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("sq,skv,k", [(128, 512, 64), (128, 4096, 2048)], ids=["k64", "k2048"])
def test_topk_indices_shape(mesh_device, sq, skv, k):
    # topk_large_indices is ROW_MAJOR bf16 in (chains off indexer_score's row-major out).
    logits = ttnn.from_torch(
        torch.randn(1, 1, sq, skv),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    indices = ops.topk_indices(logits, k)
    assert list(indices.shape) == [1, 1, sq, k]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("h,sq,skv,k", [(128, 128, 512, 64)], ids=["h128"])
def test_sparse_mla_shape(mesh_device, h, sq, skv, k):
    # q is head-sharded across TP (mesh axis 1); kvpe/indices replicated; out head-sharded.
    q = ttnn.from_torch(
        torch.randn(1, h, sq, KVPE_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 1)),
    )
    kvpe_host = torch.randn(skv, KVPE_DIM, dtype=torch.bfloat16)  # full-T latent on host
    idx = ttnn.from_torch(
        torch.randint(0, skv, (1, 1, sq, k), dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,  # sparse_sdpa requires uint32 indices (topk_indices emits uint32)
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = ops.sparse_mla(q, kvpe_host, idx, scale=KVPE_DIM**-0.5)
    out_t = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=mesh_device.shape)
    )[:1]
    assert list(out_t.shape) == [1, h, sq, KV_RANK]
