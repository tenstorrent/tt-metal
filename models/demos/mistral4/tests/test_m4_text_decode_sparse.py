# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""A10 model-level: full 2-layer decode with SPARSE MoE (forward_decode use_sparse=True) must match the
dense decode logits. Sparse routes each token to only its top-4 experts (all_to_all_dispatch/combine via
mesh_partition/all_gather) instead of computing all 16 local experts dense — the production throughput
path. B=8 (1 token/device) on the native 1x8 mesh, replicated tokens (the model's normal decode input)."""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import _repl
from models.demos.mistral4.tt.mistral4_text import TtMistral4TextModel

N_LAYERS = 2
B = 8


def _pos(p, mesh):
    return ttnn.from_torch(
        torch.tensor([p] * B, dtype=torch.int32), device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}], indirect=True
)
def test_m4_text_decode_sparse(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    rope = cfg.qk_rope_head_dim
    tsd = load_m4_weights(ckpt, N_LAYERS)
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=True, expert_dtype=ttnn.bfloat8_b
    )

    torch.manual_seed(0)
    h = torch.randn(B, 1, cfg.hidden_size) * 0.02
    c = torch.randn(1, 1, 1, rope) * 0.0 + 1.0  # pos-0 rope ~ identity (cos=1, sin=0) for a clean compare
    s = torch.zeros(1, 1, 1, rope)
    hd = _repl(h, mesh_device)
    cd = _repl(c.repeat(B, 1, 1, 1), mesh_device)
    sd_ = _repl(s.repeat(B, 1, 1, 1), mesh_device)

    # dense decode (verified default)
    kv_d = tt.init_kv_caches(B, 64)
    dense = ttnn.to_torch(
        tt.forward_decode(hd, _pos(0, mesh_device), cd, sd_, kv_d),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    ).float()[:B]

    # sparse decode (production throughput path): needs a sub-device for the all-to-all CCL ops
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sdm = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(sdm)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    kv_s = tt.init_kv_caches(B, 64)
    sparse = ttnn.to_torch(
        tt.forward_decode(hd, _pos(0, mesh_device), cd, sd_, kv_s, use_sparse=True, sub_device_id=ttnn.SubDeviceId(0)),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    ).float()[:B]

    passing, msg = comp_pcc(dense, sparse, 0.98)
    logger.info(f"Mistral-Small-4 2-layer decode SPARSE-MoE vs dense logits (B={B}): {msg}")
    assert passing, f"sparse-MoE full-model decode PCC below 0.98: {msg}"
