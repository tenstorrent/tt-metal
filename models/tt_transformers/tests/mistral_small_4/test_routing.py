# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.routing import (
    route_tokens_to_experts_reference_torch,
    router_softmax_then_route_bf16,
    router_softmax_then_route_device_bf16,
)


def _tiny_moe_config():
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    return Mistral4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        n_routed_experts=8,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=2,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )


def test_route_tokens_reference_matches_hf_moe():
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

    torch.manual_seed(42)
    cfg = _tiny_moe_config()
    moe = Mistral4MoE(cfg).eval()
    logits = torch.randn(5, cfg.n_routed_experts, dtype=torch.bfloat16)

    with torch.no_grad():
        exp_idx, exp_w = moe.route_tokens_to_experts(logits)
    got_idx, got_w = route_tokens_to_experts_reference_torch(
        logits,
        n_group=cfg.n_group,
        n_routed_experts=cfg.n_routed_experts,
        topk_group=cfg.topk_group,
        top_k=cfg.num_experts_per_tok,
        norm_topk_prob=cfg.norm_topk_prob,
        routed_scaling_factor=cfg.routed_scaling_factor,
    )

    assert torch.equal(exp_idx, got_idx)
    assert torch.equal(exp_w, got_w)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_route_tokens_device_softmax_then_routing_matches_hf(mesh_device, reset_seeds):
    """Device bf16 softmax on logits, then host routing; compare to full HF path."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

    torch.manual_seed(7)
    cfg = _tiny_moe_config()
    moe = Mistral4MoE(cfg).eval()
    logits = torch.randn(6, cfg.n_routed_experts, dtype=torch.bfloat16)

    with torch.no_grad():
        exp_idx, exp_w = moe.route_tokens_to_experts(logits)

    got_idx, got_w = router_softmax_then_route_bf16(
        mesh_device,
        logits,
        n_group=cfg.n_group,
        n_routed_experts=cfg.n_routed_experts,
        topk_group=cfg.topk_group,
        top_k=cfg.num_experts_per_tok,
        norm_topk_prob=cfg.norm_topk_prob,
        routed_scaling_factor=cfg.routed_scaling_factor,
    )

    assert torch.equal(exp_idx, got_idx), f"indices mismatch:\n{exp_idx}\n{got_idx}"

    ok, msg = comp_pcc(exp_w, got_w, pcc=0.999)
    assert ok, msg
    close, amsg = comp_allclose(exp_w, got_w, rtol=0.02, atol=0.02)
    assert close, amsg


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_route_tokens_full_device_routing_matches_hf(mesh_device, reset_seeds):
    """Softmax + grouped routing computed on device; compare to HF path."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

    torch.manual_seed(9)
    cfg = _tiny_moe_config()
    moe = Mistral4MoE(cfg).eval()
    logits = torch.randn(6, cfg.n_routed_experts, dtype=torch.bfloat16)

    with torch.no_grad():
        exp_idx, exp_w = moe.route_tokens_to_experts(logits)

    got_idx, got_w = router_softmax_then_route_device_bf16(
        mesh_device,
        logits,
        n_group=cfg.n_group,
        n_routed_experts=cfg.n_routed_experts,
        topk_group=cfg.topk_group,
        top_k=cfg.num_experts_per_tok,
        norm_topk_prob=cfg.norm_topk_prob,
        routed_scaling_factor=cfg.routed_scaling_factor,
    )

    assert torch.equal(exp_idx, got_idx), f"indices mismatch:\n{exp_idx}\n{got_idx}"
    ok, msg = comp_pcc(exp_w, got_w, pcc=0.999)
    assert ok, msg
    close, amsg = comp_allclose(exp_w, got_w, rtol=0.02, atol=0.02)
    assert close, amsg
