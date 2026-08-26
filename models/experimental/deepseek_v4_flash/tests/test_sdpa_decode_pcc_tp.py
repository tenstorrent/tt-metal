# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel PCC for fused SDPA decode on a 1x4 submesh.

Q and the learned attention sink are head-sharded (16 of the model's 64 heads
per rank); the shared MQA KV cache and decode bounds are replicated. SDPA has no
cross-head reduction, so its output remains head-sharded and needs no CCL.

Requires an 8x4 system mesh.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tests.test_sdpa_decode_pcc import (
    PCC_THRESHOLD,
    _make_attention,
)

TP_SIZE = 4
SUBMESH_SHAPE = (1, TP_SIZE)
PARENT_MESH = (8, 4)
NUM_HEADS = 64
HEAD_DIM = 256


def _torch_reference(
    q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor, sinks: torch.Tensor, scaling: float
) -> torch.Tensor:
    """Reference for q [1,B,H,D], kv [B,1,S,D], returning [1,B,H,D]."""
    q_heads = q.squeeze(0).transpose(0, 1).unsqueeze(2)  # [H,B,1,D]
    k = kv.transpose(0, 1).expand(NUM_HEADS, -1, -1, -1)  # [H,B,S,D]
    scores = torch.matmul(q_heads, k.transpose(-2, -1)) * scaling + mask
    sink = sinks.reshape(NUM_HEADS, 1, 1, 1)
    maximum = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
    numerator = torch.exp(scores - maximum)
    denominator = numerator.sum(dim=-1, keepdim=True) + torch.exp(sink - maximum)
    output = torch.matmul(numerator / denominator, k)  # [H,B,1,D]
    return output.squeeze(2).transpose(0, 1).unsqueeze(0)


def _to_tt(tensor: torch.Tensor, device, *, shard_dim: int | None = None) -> ttnn.Tensor:
    mapper = (
        ttnn.ShardTensorToMesh(device, dim=shard_dim) if shard_dim is not None else ttnn.ReplicateTensorToMesh(device)
    )
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=mapper,
    )


@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("skv", [128, 512], ids=["skv128", "skv512"])
@pytest.mark.parametrize("bounds", ["mask", "causal"])
def test_sdpa_decode_pcc_tp4(mesh_device, reset_seeds, skv: int, bounds: str) -> None:
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"need an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")

    submesh = mesh_device.create_submesh(ttnn.MeshShape(*SUBMESH_SHAPE), ttnn.MeshCoordinate(0, 0))
    torch.manual_seed(1234)

    batch = 1
    q = torch.randn(1, batch, NUM_HEADS, HEAD_DIM) * 0.1
    kv = torch.randn(batch, 1, skv, HEAD_DIM) * 0.1
    sinks = torch.randn(NUM_HEADS) * 0.5
    mask = torch.zeros(1, 1, 1, skv)
    cur_pos = None
    if bounds == "mask":
        mask[..., skv // 2 :] = -1.0e9
    else:
        cur_pos_value = skv // 2 - 1
        mask[..., cur_pos_value + 1 :] = -1.0e9
        cur_pos = ttnn.from_torch(
            torch.full((batch,), cur_pos_value, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )

    attn = _make_attention(submesh, NUM_HEADS, HEAD_DIM, sinks, tp_size=TP_SIZE)
    output = attn._sdpa_decode(
        _to_tt(q, submesh, shard_dim=2),
        _to_tt(kv, submesh),
        None if cur_pos is not None else _to_tt(mask, submesh),
        cur_pos=cur_pos,
    )
    output_torch = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2)).float()
    reference = _torch_reference(q, kv, mask, attn.sinks_torch, attn.scaling)

    passing, pcc_message = comp_pcc(reference, output_torch, pcc=PCC_THRESHOLD)
    logger.info(comp_allclose(reference, output_torch))
    logger.info(f"[sdpa decode tp{TP_SIZE}, {bounds}] PCC: {pcc_message}")
    assert passing, f"SDPA decode TP{TP_SIZE} PCC < {PCC_THRESHOLD}: {pcc_message}"
