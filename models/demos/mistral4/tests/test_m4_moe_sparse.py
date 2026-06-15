# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-dispatch MoE vs dense: TtMistral4MoE._forward_sparse (sharded tokens) must match the dense
forward (replicated tokens). Both select the same top-4 experts + renormalize, so the all-to-all
sparse path (dispatch tokens to their experts' devices, compute only those, combine) reproduces the
dense-local result. B=8 tokens, 1/device on the native 1x8 mesh (cluster_axis=1 spans all 8 — validated
in test_m4_moe_roundtrip at PCC 0.9998). Gates the sparse path before wiring into the decode loop."""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import load_m4_weights
from models.demos.mistral4.tt.mistral4_text import TtMistral4MoE

BATCH = 8


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}],
    indirect=True,
)
def test_m4_moe_sparse(mesh_device, reset_seeds):
    pcc_required = 0.98
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    tsd = load_m4_weights(ckpt, 1)
    sd = {k[len("model.layers.0.") :]: v for k, v in tsd.items() if k.startswith("model.layers.0.mlp.")}
    moe = TtMistral4MoE(mesh_device, sd, cfg, shard_experts=True, expert_dtype=ttnn.bfloat16)

    torch.manual_seed(0)
    x = torch.randn(BATCH, 1, cfg.hidden_size) * 0.1

    # dense reference: replicated tokens, dense-local experts (the verified default path)
    x_repl = ttnn.from_torch(
        x,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    dense = ttnn.to_torch(moe.forward(x_repl), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[
        :BATCH
    ]

    # sparse path: tokens SHARDED 1/device across the 8, all-to-all dispatch/combine. Needs a sub-device.
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sdm = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(sdm)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    x_shard = ttnn.from_torch(
        x,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )  # [1,1,H] per device (token b on device b)
    sparse = ttnn.to_torch(
        moe._forward_sparse(x_shard, sub_device_id=ttnn.SubDeviceId(0)),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    ).float()[:BATCH]

    passing, msg = comp_pcc(dense, sparse, pcc_required)
    logger.info(f"Mistral-Small-4 sparse-dispatch MoE (sharded) vs dense (B={BATCH}): {msg}")
    assert passing, f"sparse MoE PCC below {pcc_required}: {msg}"
