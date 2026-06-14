# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode trace capture/replay PCC: a captured decode trace must reproduce the eager decode.

Runs N eager decode steps (collecting logits), then captures TtMistral4TextModel.forward_decode as
a trace and replays it for the same N steps (copy_host_to_device into persistent buffers +
execute_trace). The traced logits must match the eager logits — proving the trace runs the full
decode graph (paged_update_cache + SDPA decode + MoE + LM head) with no per-op host dispatch.
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import get_cached_golden, load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import TracedDecode, _repl
from models.demos.mistral4.tt.mistral4_text import TtMistral4TextModel

N_LAYERS = 2
N_STEPS = 8


def _pos(p, B, mesh):
    return ttnn.from_torch(
        torch.tensor([p] * B, dtype=torch.int32), device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_trace(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    g = get_cached_golden(ckpt, N_LAYERS, 0, 32)
    ids, cos, sin = g["input_ids"], g["rope_cos"], g["rope_sin"]
    B, rope, hidden = ids.shape[0], cfg.qk_rope_head_dim, cfg.hidden_size
    embed = load_m4_weights(ckpt, N_LAYERS)

    tsd = embed
    # shard_experts=False (replicated experts, no all_reduce) isolates the trace mechanism from
    # CCL-in-trace; the sharded production path is traced separately (CCL needs persistent buffers).
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=False, expert_dtype=ttnn.bfloat8_b
    )
    emb = tsd["model.embed_tokens.weight"][ids]  # [B,S,hidden]

    def emb_i(i):
        return emb[:, i : i + 1, :]

    def cos_i(i):
        return cos[:, i : i + 1, :].reshape(B, 1, 1, rope)

    def sin_i(i):
        return sin[:, i : i + 1, :].reshape(B, 1, 1, rope)

    # Eager decode reference
    kv_e = tt.init_kv_caches(B, max_seq=64)
    eager = []
    for i in range(N_STEPS):
        o = tt.forward_decode(
            _repl(emb_i(i), mesh_device),
            _pos(i, B, mesh_device),
            _repl(cos_i(i), mesh_device),
            _repl(sin_i(i), mesh_device),
            kv_e,
        )
        eager.append(ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B])
    eager = torch.cat(eager, dim=1)

    # Traced decode (fresh cache) — capture once, replay per step
    kv_t = tt.init_kv_caches(B, max_seq=64)
    td = TracedDecode(tt, mesh_device, B, hidden, rope, kv_t)
    traced = []
    for i in range(N_STEPS):
        out = td.step(emb_i(i), cos_i(i), sin_i(i), [i] * B)
        traced.append(ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B])
    traced = torch.cat(traced, dim=1)

    passing, msg = comp_pcc(eager, traced, 0.99)
    logger.info(f"Mistral-Small-4 traced-vs-eager decode PCC: {msg}")
    assert passing, f"traced decode PCC below 0.99: {msg}"
