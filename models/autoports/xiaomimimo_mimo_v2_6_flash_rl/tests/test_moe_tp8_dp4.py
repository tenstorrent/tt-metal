# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo routed MoE on TP=8 x DP=4, split into routing agreement and expert math.

Inputs are the real MoE inputs of the golden prompts at this layer (computed on the host from the
golden hidden states), spread over the four Galaxy rows, plus normalized random tokens. For every
token we check:

* routing: whether the device picked the same top-8 expert set as the fp32 reference;
* expert math: the device output against the reference evaluated with the device's own routing
  (so a near-tie flip in top-8 does not hide or fake an error in the TP-sharded experts);
* end to end: the device output against the plain reference.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.reference import model as ref
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.common import (
    DP,
    GALAXY_DEVICE_PARAMS,
    cache_dir,
    dp_submeshes,
    load_config_and_ckpt,
    pcc,
)
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.common import TILE, MeshPlan, pad_to
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.moe import TtMiMoMoE

GOLDEN = os.environ.get("MIMO_GOLDEN", "/home/zni/models/mimo-golden/golden_48l.pt")


def moe_inputs_from_golden(cfg, weights, layer_idx):
    """Post-attention-norm activations of the golden prompts at ``layer_idx`` (fp32, one row per token)."""
    golden = torch.load(GOLDEN)
    dims = cfg.attention_dims(layer_idx)
    rows = []
    with torch.no_grad():
        for hidden in golden["hidden"]:
            x = hidden[layer_idx]
            pos = torch.arange(x.shape[0])
            h = ref.rms_norm(x, weights["input_layernorm"], cfg.layernorm_epsilon)
            x = x + ref.attention(h, weights["attn"], dims, cfg.attention_value_scale, pos)
            rows.append(ref.rms_norm(x, weights["post_attention_layernorm"], cfg.layernorm_epsilon))
    return torch.cat(rows)


def reference_with_routing(x, experts, topk_idx, topk_w):
    out = torch.zeros_like(x)
    for e in torch.unique(topk_idx).tolist():
        tok, slot = torch.where(topk_idx == e)
        xe = x[tok]
        y = (
            torch.nn.functional.silu(xe @ experts["gate_proj"][e].float().T) * (xe @ experts["up_proj"][e].float().T)
        ) @ experts["down_proj"][e].float().T
        out.index_add_(0, tok, y * topk_w[tok, slot, None])
    return out


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("layer_idx", [1, 5])
def test_moe_tp8_dp4(mesh_device, layer_idx):
    if not os.path.exists(GOLDEN):
        pytest.skip(f"golden file {GOLDEN} not found")
    torch.manual_seed(7)
    cfg, ckpt = load_config_and_ckpt()
    weights = ref.load_layer_weights(ckpt, cfg, layer_idx)
    real = moe_inputs_from_golden(cfg, weights, layer_idx)
    rand = ref.rms_norm(torch.randn(128, cfg.hidden_size), weights["post_attention_layernorm"], cfg.layernorm_epsilon)
    sources = {"real": real, "random": rand}

    moes, plans = [], []
    for r, sm in enumerate(dp_submeshes(mesh_device)):
        plan = MeshPlan(sm)
        plans.append(plan)
        moes.append(
            TtMiMoMoE(
                plan,
                cfg,
                layer_idx,
                weights["router"] if r == 0 else None,
                weights["experts"] if r == 0 else None,
                cache_dir=cache_dir("layer_bfp8"),
            )
        )

    results = {}
    for name, x_all in sources.items():
        x_all = x_all.bfloat16().float()  # the device sees exactly this input
        chunks = torch.chunk(x_all, DP)  # each row gets different tokens
        pending = []
        for r, (x, plan) in enumerate(zip(chunks, plans)):
            n = x.shape[0]
            padded = pad_to(n, TILE)
            xp = torch.nn.functional.pad(x, (0, 0, 0, padded - n))
            mask = (torch.arange(padded) < n).bfloat16().reshape(1, 1, -1, 1)
            to_dev = lambda t, layout: ttnn.from_torch(
                t, device=plan.mesh_device, dtype=ttnn.bfloat16, layout=layout, mesh_mapper=plan.replicate()
            )
            x_tt = to_dev(xp.reshape(1, 1, padded, -1).bfloat16(), ttnn.TILE_LAYOUT)
            m_tt = to_dev(mask, ttnn.TILE_LAYOUT)
            dense = moes[r].route(x_tt, m_tt)
            out = moes[r].forward(x_tt, m_tt)
            pending.append((x, n, dense, out))

        match, got_all, ref_all, forced_all = [], [], [], []
        for r, (x, n, dense, out) in enumerate(pending):
            plan = plans[r]
            dense = plan.to_torch_replicated(dense).float()[0, 0, :n]
            got = plan.to_torch_replicated(out).float()[0, 0, :n]
            ref_idx, ref_w = ref.route(x, weights["router"], cfg)
            tt_w, tt_idx = dense.topk(cfg.num_experts_per_tok, dim=-1)
            match += [set(a.tolist()) == set(b.tolist()) for a, b in zip(ref_idx, tt_idx)]
            with torch.no_grad():
                ref_all.append(ref.moe(x, weights["router"], weights["experts"], cfg))
                forced_all.append(reference_with_routing(x, weights["experts"], tt_idx, tt_w))
            got_all.append(got)
        got, plain, forced = torch.cat(got_all), torch.cat(ref_all), torch.cat(forced_all)
        rate = sum(match) / len(match)
        p_plain, p_forced = pcc(got, plain), pcc(got, forced)
        ok = torch.tensor(match)
        p_matched = pcc(got[ok], plain[ok]) if ok.any() else float("nan")
        results[name] = (rate, p_forced, p_plain, p_matched)
        logger.info(
            f"layer {layer_idx} {name}: {len(match)} tokens over {DP} rows; top-8 set match {rate:.3f}; "
            f"PCC vs reference with device routing {p_forced:.5f}; vs plain reference {p_plain:.5f} "
            f"(tokens with matching routing {p_matched:.5f})"
        )

    for name, (rate, p_forced, p_plain, p_matched) in results.items():
        assert p_forced > 0.99, (name, p_forced)
        assert p_matched > 0.99, (name, p_matched)
    assert results["real"][0] > 0.9, results["real"]
