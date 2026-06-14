# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bottom-up PCC for the Mistral-Small-4 MLA attention block (model-local, generic ttnn ops).

Step 1 (this file): the projection chains, PCC'd against reference goldens:
  q_b_out  = q_b_proj(q_a_layernorm(q_a_proj(x)))
  kv_a_out = kv_a_proj_with_mqa(x)
  kv_b_out = kv_b_proj(kv_a_layernorm(kv_a_out[..., :kv_lora_rank]))
Weights are bf16 (fp8-dequantized), replicated across the (1,8) mesh — correctness first;
sharding / flash-MLA / paged come later. Reference goldens from m4_text_reference.
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tests.multimodal.mistral_24b.m4_text_reference import capture_golden, load_m4_text_reference


def _lin_w(hf_weight, mesh):
    # HF nn.Linear weight is [out, in]; ttnn.linear computes x @ W with W = [in, out].
    return ttnn.as_tensor(
        hf_weight.transpose(0, 1).contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _norm_w(hf_weight, mesh):
    return ttnn.as_tensor(
        hf_weight.reshape(1, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _to_host(tt, mesh, batch):
    # replicated -> concat on dim 0 gives `mesh` identical copies; take the first `batch`
    return ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()[:batch]


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_mla_projections(mesh_device, reset_seeds):
    pcc_required = 0.99
    ckpt = os.environ["HF_MODEL"]
    eps = AutoConfig.from_pretrained(ckpt).text_config.rms_norm_eps
    kv_lora = AutoConfig.from_pretrained(ckpt).text_config.kv_lora_rank

    # reference + goldens
    model, cfg, _ = load_m4_text_reference(ckpt, n_layers=1)
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 32))
    g = capture_golden(model, ids)
    sa = model.model.layers[0].self_attn
    B = g["mla_in"].shape[0]

    # weights -> ttnn (replicated)
    w_qa = _lin_w(sa.q_a_proj.weight, mesh_device)
    w_qan = _norm_w(sa.q_a_layernorm.weight, mesh_device)
    w_qb = _lin_w(sa.q_b_proj.weight, mesh_device)
    w_kva = _lin_w(sa.kv_a_proj_with_mqa.weight, mesh_device)
    w_kvan = _norm_w(sa.kv_a_layernorm.weight, mesh_device)
    w_kvb = _lin_w(sa.kv_b_proj.weight, mesh_device)

    x = ttnn.from_torch(
        g["mla_in"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # q chain
    q_a = ttnn.linear(x, w_qa)
    q_a = ttnn.rms_norm(q_a, epsilon=eps, weight=w_qan)
    q_b = ttnn.linear(q_a, w_qb)

    # kv chain
    kv_a = ttnn.linear(x, w_kva)
    kv_pass = ttnn.slice(kv_a, [0, 0, 0], [B, kv_a.shape[1], kv_lora])
    kv_pass = ttnn.rms_norm(kv_pass, epsilon=eps, weight=w_kvan)
    kv_b = ttnn.linear(kv_pass, w_kvb)

    results = {
        "q_b_out": (_to_host(q_b, mesh_device, B), g["q_b_out"]),
        "kv_a_out": (_to_host(kv_a, mesh_device, B), g["kv_a_out"]),
        "kv_b_out": (_to_host(kv_b, mesh_device, B), g["kv_b_out"]),
    }
    all_pass = True
    for name, (tt, ref) in results.items():
        passing, msg = comp_pcc(ref, tt, pcc_required)
        logger.info(f"MLA projection {name}: {msg}")
        all_pass = all_pass and passing
    assert all_pass, "MLA projection-chain PCC below threshold"
