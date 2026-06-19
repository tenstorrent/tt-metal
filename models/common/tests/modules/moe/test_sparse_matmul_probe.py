# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Scratch diagnostic: a single TP-sharded ttnn.sparse_matmul on a (1,8) mesh.

Mirrors MoE1D's first decode sparse_matmul (col-parallel gate weight, replicated activation, top-k
sparsity). Bisects WHY the (1,8) MoE forward hangs while a bare all_reduce passes. Three cases, ordered
so the expected-pass controls report before the suspect:

  1. replicate / full-n cfg   — weight full 256-wide, program-config n=256  -> control, should pass
  2. shard / per-device-n cfg — weight 32-wide/device, program-config n=32  -> reference-style, should pass
  3. shard / full-n cfg       — weight 32-wide/device, program-config n=256 -> MoE1D's ACTUAL call (suspect)

If only #3 hangs, the root cause is MoE1D building the sparse_matmul program config from the FULL
intermediate (cfg.intermediate_size) instead of the per-device shard width. Delete after.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.modules.moe.moe_1d import _build_sparse_matmul_config

E, H, I, TOP_K, SEQ = 8, 256, 256, 2, 1  # I=256 -> 32/device on a 1x8 mesh (tile-aligned)


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 8)], ids=["1x8"], indirect=True)
@pytest.mark.parametrize(
    "shard_mode,n_mode",
    [("replicate", "full"), ("shard", "per_device"), ("shard", "full")],
    ids=["replicate-fulln", "shard-perdevn", "shard-fulln"],
)
def test_sharded_sparse_matmul_probe(ttnn_mesh_device: ttnn.MeshDevice, shard_mode, n_mode):
    md = ttnn_mesh_device
    nd = md.get_num_devices()
    per_dev_I = I // nd  # 32

    act = ttnn.from_torch(
        torch.randn(1, 1, SEQ, H),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    gate_mapper = ttnn.ShardTensorToMesh(md, dim=-1) if shard_mode == "shard" else ttnn.ReplicateTensorToMesh(md)
    gate = ttnn.from_torch(
        torch.randn(1, E, H, I) * 0.1,
        device=md,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=gate_mapper,
    )
    # top-k=2 dense routing sparsity, ROW_MAJOR bf16, replicated (matches MoE1D decode)
    dense = torch.zeros(1, 1, SEQ, E)
    dense[..., 0] = 0.5
    dense[..., 1] = 0.5
    sparsity = ttnn.from_torch(
        dense,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )

    n = I if n_mode == "full" else per_dev_I
    prg = _build_sparse_matmul_config(SEQ, n, 1, H)
    logger.info(f"sparse_matmul probe: nd={nd} shard={shard_mode} n_mode={n_mode} (n={n}, per_dev_I={per_dev_I})")
    out = ttnn.sparse_matmul(
        act,
        gate,
        sparsity=sparsity,
        nnz=None,  # matches MoE1D decode_forward
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=ttnn.Tile([32, 32]),
        program_config=prg,
        dtype=ttnn.bfloat16,
    )
    ttnn.synchronize_device(md)
    logger.info(f"DONE [{shard_mode}/{n_mode}] out shape={tuple(out.shape)}")
    assert out is not None
