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
    reason="WIP: all_to_all_dispatch is proven on this mesh (test_m4_a2a_probe) and TtMistral4MoE."
    "_forward_sparse implements the full dispatch->expert->combine pipeline, but consuming the "
    "multi-device dispatch/combine output requires the exact ttnn sharded-tensor reshape contract "
    "(the per-device vs global logical shape of the dispatch output) which is not yet matched. The "
    "dense + multi-user-batched MoE is the verified default path; sparse dispatch is the remaining "
    "throughput lever (helps low-batch weight-streaming + high-batch compute). See MISTRAL4_DESIGN.md.",
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
