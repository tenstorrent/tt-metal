# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-dispatch MoE vs dense: TtMistral4MoE._forward_sparse must match the dense forward.

Both select the same top-4 experts and renormalize, so the all-to-all sparse-dispatch path (dispatch
tokens to their experts' devices, compute only those, combine) must reproduce the dense-local result.
Batched decode shape (B=8 users, 1 token each) on the 1x8 mesh. Gates the sparse path before wiring
it into the decode loop.
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import _repl
from models.demos.mistral4.tt.mistral4_text import TtMistral4MoE

BATCH = 8


@pytest.mark.xfail(
    reason="A10 sparse dispatch needs a 2D DP×EP mesh (tokens batch-sharded on a data-parallel axis, "
    "experts sharded on an expert-parallel axis via ShardTensor2dMesh). This test runs on the flat 1x8 "
    "mesh where experts are sharded across all 8 (expert-parallel) with tokens replicated — no free DP "
    "axis to dispatch over. DE-RISKED (2026-06-15): all_to_all_dispatch + all_to_all_combine are PROVEN "
    "on a (2,4) mesh for the mistral4 MoE config (test_m4_a2a_probe_2x4 PASSES; the upstream "
    "test_moe_ccl_end_to_end::test_integration[2x4_grid] validates the full dispatch→experts→combine "
    "round-trip), and the dispatch/combine shape contracts are known. The remaining work is the model-side "
    "2x4 refactor: lay out the expert weights with ShardTensor2dMesh (32 experts/EP-device), migrate the "
    "whole model's collective axes (MLA/norm/lm_head/all_reduce/all_gather) to 2x4, and wire _forward_sparse. "
    "The dense expert-parallel MoE is the verified default on 1x8 meanwhile. See MISTRAL4_DESIGN.md (A10).",
    strict=False,
)
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
    x = _repl(torch.randn(BATCH, 1, cfg.hidden_size) * 0.1, mesh_device)
    dense = ttnn.to_torch(moe.forward(x), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:BATCH]
    sparse = ttnn.to_torch(moe._forward_sparse(x), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[
        :BATCH
    ]

    passing, msg = comp_pcc(dense, sparse, pcc_required)
    logger.info(f"Mistral-Small-4 sparse-dispatch MoE vs dense (B={BATCH}): {msg}")
    assert passing, f"sparse MoE PCC below {pcc_required}: {msg}"
