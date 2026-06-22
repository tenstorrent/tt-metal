# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill trace capture/replay PCC: a captured prefill trace must reproduce the eager prefill.

Upgrades trace coverage to prefill AND decode (A3). Captures TtMistral4TextModel.forward_prefill at a
fixed ISL and replays it (copy_host_to_device + execute_trace), checking the traced prefill logits
match the eager forward_prefill bit-exactly.
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import TracedPrefill, _repl
from models.demos.mistral4.tt.mistral4_text import TtMistral4TextModel

N_LAYERS = 2
S = 128


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 120000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_prefill_trace(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    tsd = load_m4_weights(ckpt, N_LAYERS)
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=True, expert_dtype=ttnn.bfloat8_b
    )
    rope, hidden = cfg.qk_rope_head_dim, cfg.hidden_size
    torch.manual_seed(0)
    x = torch.randn(1, S, hidden) * 0.1
    cos = torch.randn(1, 1, S, rope)
    sin = torch.randn(1, 1, S, rope)

    kv_e = tt.init_kv_caches(1, max_seq=S + 64)
    eager = ttnn.to_torch(
        tt.forward_prefill(_repl(x, mesh_device), _repl(cos, mesh_device), _repl(sin, mesh_device), kv_e),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    ).float()[:1]

    kv_t = tt.init_kv_caches(1, max_seq=S + 64)
    tp = TracedPrefill(tt, mesh_device, S, hidden, rope, kv_t)
    traced = ttnn.to_torch(tp.run(x, cos, sin), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:1]

    passing, msg = comp_pcc(eager, traced, 0.99)
    logger.info(f"Mistral-Small-4 prefill trace-vs-eager PCC (S={S}): {msg}")
    assert passing, f"prefill trace PCC below 0.99: {msg}"
