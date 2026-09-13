# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Router / routed experts / shared expert / dense MLP vs the fp32 torch oracle, real weights (layers 1 and 0)."""
from __future__ import annotations

import pytest
import torch

from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.moe_ref import (
    dense_mlp_reference,
    moe_reference,
    router_reference,
)
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tests.utils import assert_pcc, first_shard, replicated
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.moe.moe import KimiDenseMLP, KimiMoE
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.moe.router import KimiRouter

MOE_LAYER = 1
DENSE_LAYER = 0


@pytest.fixture(scope="module")
def moe_sd(checkpoint):
    return checkpoint.moe_state_dict(MOE_LAYER)


def _x(S, hidden, seed):
    torch.manual_seed(seed)
    return (torch.randn(1, 1, S, hidden) * 0.5).to(torch.bfloat16)


@pytest.mark.parametrize("S", [32, 128])
def test_router_topk_agreement(mesh_device, hf_config, moe_sd, cache_path, S):
    x = _x(S, hf_config.hidden_size, 10)
    idx, w, dense_ref = router_reference(
        x, moe_sd["moe.gate.weight"], moe_sd["moe.gate.e_score_correction_bias"], hf_config
    )
    router = KimiRouter(
        mesh_device,
        hf_config,
        moe_sd["moe.gate.weight"],
        moe_sd["moe.gate.e_score_correction_bias"],
        name="t.router",
        cache_path=None,
    )
    dense = first_shard(router(replicated(mesh_device, x))).float()[0, 0]
    chosen_ref = dense_ref > 0
    chosen_tt = dense > 0
    agree = (chosen_ref & chosen_tt).sum().item() / chosen_ref.sum().item()
    print(
        f"[router] top-{hf_config.num_experts_per_token} set agreement: {agree:.4f}; tt picks/token {chosen_tt.sum(-1).float().mean():.2f}"
    )
    assert agree >= 0.98, agree
    assert_pcc(dense_ref, dense, 0.99, "dense routing weights")


@pytest.mark.parametrize("S,mode", [(32, "decode"), (8, "decode"), (128, "prefill")])
def test_moe_block(mesh_device, ccl, hf_config, moe_sd, cache_path, S, mode):
    x = _x(S, hf_config.hidden_size, 11)
    ref = moe_reference(x, moe_sd, hf_config)[0, 0]
    moe = KimiMoE(mesh_device, hf_config, moe_sd, layer_idx=MOE_LAYER, ccl=ccl, cache_path=cache_path)
    out = moe.forward(replicated(mesh_device, x), mode)
    out_t = first_shard(out).float()[0, 0, :S]
    assert_pcc(ref, out_t, 0.99, f"moe {mode} S={S}")


def test_dense_mlp(mesh_device, ccl, hf_config, checkpoint, cache_path):
    sd = checkpoint.dense_mlp_state_dict(DENSE_LAYER)
    x = _x(32, hf_config.hidden_size, 12)
    ref = dense_mlp_reference(x, sd)[0, 0]
    mlp = KimiDenseMLP(mesh_device, hf_config, sd, layer_idx=DENSE_LAYER, ccl=ccl, cache_path=cache_path)
    out = first_shard(mlp.forward(replicated(mesh_device, x))).float()[0, 0]
    assert_pcc(ref, out, 0.99, "dense mlp layer 0")
