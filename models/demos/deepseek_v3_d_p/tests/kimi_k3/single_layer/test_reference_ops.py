# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the functional single-layer references match the module references they stand in for.

Reduced dims, fp32 both sides, so agreement is to fp32 rounding (sampled rows bit-for-bit where the
math is token-wise). Pooled routed experts are checked with more experts than pools.
"""

import types

import pytest
import torch
from transformers.cache_utils import DynamicCache

from models.common.utility_functions import hf_cache_layer_kv
from models.demos.deepseek_v3_d_p.reference.kda import KDAConfig, kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3.configuration_kimi_k3 import KimiLinearConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiMLP, KimiSparseMoeBlock
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.reference_ops import (
    MlaDims,
    PooledExperts,
    dense_mlp,
    kda_forward,
    mla_forward,
    moe_forward,
)

ATOL = 1e-4


def _close(actual, expected, name, rtol=1e-4):
    scale = expected.abs().max().item()
    err = (actual - expected).abs().max().item()
    assert err <= ATOL + rtol * scale, f"{name}: max abs err {err:.3e} (ref max {scale:.3e})"


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(7)


def _situ_config(hidden, intermediate, **extra):
    return KimiLinearConfig(
        hidden_size=hidden,
        intermediate_size=intermediate,
        hidden_act="situ",
        activation_situ_beta=KimiK3Config.ACTIVATION_SITU_BETA,
        activation_situ_linear_beta=KimiK3Config.ACTIVATION_SITU_LINEAR_BETA,
        rms_norm_eps=KimiK3Config.RMS_NORM_EPS,
        **extra,
    )


def test_dense_mlp_matches_kimi_mlp(generator):
    module = KimiMLP(_situ_config(64, 160)).eval()
    for p in module.parameters():
        torch.nn.init.normal_(p, std=0.3, generator=generator)
    x = torch.randn(40, 64, generator=generator) * 3
    rows = torch.tensor([0, 7, 39])
    w = {n: getattr(module, n).weight.detach() for n in ("gate_proj", "up_proj", "down_proj")}
    with torch.no_grad():
        expected = module(x)
    _close(dense_mlp(x, w), expected, "dense")
    _close(dense_mlp(x, w, rows=rows), expected[rows], "dense sampled")


@pytest.mark.parametrize("full_rank_gate", [True, False], ids=["full_rank_gate", "low_rank_gate"])
def test_kda_matches_layer_reference(generator, full_rank_gate):
    config = KDAConfig(
        hidden_size=64,
        num_heads=4,
        head_k_dim=16,
        head_v_dim=16,
        conv_kernel_size=4,
        norm_eps=1e-5,
        use_full_rank_gate=full_rank_gate,
        gate_lower_bound=-5.0,
    )
    from models.demos.deepseek_v3_d_p.tests.kda.utils import random_weights

    weights = random_weights(config)
    x = torch.randn(1, 48, 64, generator=generator)
    expected, expected_state = kda_forward_reference(x, weights, config)
    actual, state = kda_forward(x[0], weights, config)
    _close(actual, expected[0], "kda out")
    _close(state.recurrent, expected_state.recurrent, "kda recurrent")

    # Split at an unaligned point with carried state, sampled rows on the second half.
    first, carry = kda_forward(x[0, :19], weights, config)
    rows = torch.tensor([0, 5, 28])
    second, final = kda_forward(x[0, 19:], weights, config, state=carry, rows=rows)
    _close(first, expected[0, :19], "kda chunk1")
    _close(second, expected[0, 19:][rows], "kda chunk2 sampled")
    for name in ("recurrent", "q_convolution", "k_convolution", "v_convolution"):
        _close(getattr(final, name), getattr(expected_state, name), f"kda {name}")


def _mla_small_config():
    cfg = kimi_k3_hf_config(max_seq=256)
    return types.SimpleNamespace(
        **{
            **vars(cfg),
            "hidden_size": 96,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "q_lora_rank": 48,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
        }
    )


def test_mla_matches_mla_reference(generator):
    cfg = _mla_small_config()
    heads, h = cfg.num_attention_heads, cfg.hidden_size

    def normal(*shape, std=0.1):
        return torch.randn(*shape, generator=generator) * std

    w = {
        "q_a_proj.weight": normal(cfg.q_lora_rank, h),
        "q_a_layernorm.weight": 1 + normal(cfg.q_lora_rank),
        "q_b_proj.weight": normal(heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), cfg.q_lora_rank),
        "kv_a_proj_with_mqa.weight": normal(cfg.kv_lora_rank + cfg.qk_rope_head_dim, h),
        "kv_a_layernorm.weight": 1 + normal(cfg.kv_lora_rank),
        "kv_b_proj.weight": normal(heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank),
        "o_proj.weight": normal(h, heads * cfg.v_head_dim),
        "g_proj.weight": normal(heads * cfg.v_head_dim, h),
    }
    module = create_mla_reference(
        cfg, {"model.layers.0.self_attn." + k: v for k, v in w.items()}, 0, "model.layers.0.self_attn"
    ).eval()
    x = torch.randn(70, h, generator=generator) * 2
    with torch.no_grad():
        expected, _, cache = module(
            hidden_states=x.unsqueeze(0),
            position_ids=torch.arange(70).unsqueeze(0),
            past_key_value=DynamicCache(),
            use_cache=True,
        )
    dims = MlaDims.from_hf(cfg)
    actual, kvpe = mla_forward(x, w, dims, query_block=16)
    _close(actual, expected[0], "mla out")
    # The cache row layout the device KV is compared against: normed latent ‖ unrotated k_pe.
    _close(kvpe, hf_cache_layer_kv(cache, 0)[0][0, 0], "mla kvpe vs module cache")

    # Continuation: chunk 2 attends to chunk 1's kvpe; sampled, unsorted query rows split across
    # several query blocks so the per-block causal limit is exercised with a prefix.
    _, prefix = mla_forward(x[:33], w, dims)
    rows = torch.tensor([20, 0, 36, 1, 7, 29])
    second, kvpe2 = mla_forward(x[33:], w, dims, prefix_kvpe=prefix, rows=rows, query_block=2)
    _close(second, expected[0, 33:][rows], "mla continuation sampled")
    _close(torch.cat([prefix, kvpe2]), kvpe, "mla kvpe")


def test_pooled_moe_matches_sparse_block(generator, expect_error):
    hidden, latent, inter, experts, pools, top_k = 64, 32, 24, 64, 8, 4
    cfg = _situ_config(
        hidden,
        inter * 4,
        routed_expert_hidden_size=latent,
        moe_intermediate_size=inter,
        num_experts=experts,
        num_experts_per_token=top_k,
        num_shared_experts=2,
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=1.0,
        latent_moe_use_norm=True,  # K3; moe_forward always applies the latent norm
    )
    pooled = PooledExperts(
        gate=torch.randn(pools, inter, latent, generator=generator) * 0.3,
        up=torch.randn(pools, inter, latent, generator=generator) * 0.3,
        down=torch.randn(pools, latent, inter, generator=generator) * 0.3,
        pool_of=torch.arange(experts) % pools,
        sign=torch.where(torch.rand(experts, inter, generator=generator) < 0.5, -1.0, 1.0),
    )
    block = KimiSparseMoeBlock(cfg).eval()
    with torch.no_grad():
        torch.nn.init.normal_(block.gate.weight, std=0.3, generator=generator)
        torch.nn.init.normal_(block.gate.e_score_correction_bias, std=0.05, generator=generator)
        for e, module in enumerate(block.experts):
            expert = pooled.expert(e)
            module.w1.weight.copy_(expert["gate_proj"])
            module.w3.weight.copy_(expert["up_proj"])
            module.w2.weight.copy_(expert["down_proj"])
        for p in (
            *block.shared_experts.parameters(),
            block.routed_expert_down_proj.weight,
            block.routed_expert_up_proj.weight,
        ):
            torch.nn.init.normal_(p, std=0.3, generator=generator)
        block.routed_expert_norm.weight.copy_(1 + 0.1 * torch.randn(latent, generator=generator))
    w = {
        "gate_weights": {
            "weight": block.gate.weight.detach(),
            "e_score_correction_bias": block.gate.e_score_correction_bias.detach(),
        },
        "latent_weights": {
            "down_proj": block.routed_expert_down_proj.weight.detach(),
            "up_proj": block.routed_expert_up_proj.weight.detach(),
            "norm": block.routed_expert_norm.weight.detach(),
        },
        "shared_expert_weights": {
            n: getattr(block.shared_experts, n).weight.detach() for n in ("gate_proj", "up_proj", "down_proj")
        },
    }
    x = torch.randn(1, 50, hidden, generator=generator)
    with torch.no_grad():
        expected = block(x)[0]
        expected_idx, expected_weight = block.gate(x)
    actual, (idx, weight) = moe_forward(x[0], w, pooled, top_k=top_k)
    # Gate weights checked directly: the latent RMSNorm would largely cancel a per-token scale
    # error (e.g. a dropped renormalization) in the block output alone.
    order, expected_order = idx.argsort(-1), expected_idx.argsort(-1)
    assert torch.equal(idx.gather(1, order), expected_idx.gather(1, expected_order)), "routing differs"
    _close(weight.gather(1, order), expected_weight.gather(1, expected_order), "moe gate weights")
    _close(actual, expected, "moe")
    rows = torch.tensor([3, 17, 49])
    sampled, _ = moe_forward(x[0], w, pooled, top_k=top_k, rows=rows)
    _close(sampled, expected[rows], "moe sampled")

    # The device-routed path: routing supplied for the sampled rows reproduces the block exactly...
    routed, _ = moe_forward(x[0], w, pooled, top_k=top_k, rows=rows, routing=(idx[rows], weight[rows]))
    _close(routed, expected[rows], "moe device-routed")
    # ...and a wrong expert in one slot moves that row and only that row.
    bad_idx = idx[rows].clone()
    bad_idx[1, 0] = (bad_idx[1, 0] + 1) % experts
    perturbed, _ = moe_forward(x[0], w, pooled, top_k=top_k, rows=rows, routing=(bad_idx, weight[rows]))
    moved = (perturbed - expected[rows]).abs().amax(-1) > 1e-3
    assert moved.tolist() == [False, True, False], f"expert swap moved rows {moved.tolist()}"
    # Supplied WEIGHTS are used as given too: reweighting one row against an independent
    # per-expert sum, so an implementation that took the ids but recomputed the weights fails.
    new_weight = weight[rows].clone()
    new_weight[2] = torch.linspace(0.1, 0.9, top_k)
    reweighted, _ = moe_forward(x[0], w, pooled, top_k=top_k, rows=rows, routing=(idx[rows], new_weight))
    t = int(rows[2])
    with torch.no_grad():
        latent_t = block.routed_expert_down_proj(x[0, t])
        routed_t = sum(new_weight[2, k] * block.experts[int(idx[t, k])](latent_t) for k in range(top_k))
        expected_t = block.routed_expert_up_proj(block.routed_expert_norm(routed_t)) + block.shared_experts(x[0, t])
    _close(reweighted[2], expected_t, "moe reweighted row")
    with expect_error(ValueError, "routing must be"):
        moe_forward(x[0], w, pooled, top_k=top_k, rows=rows, routing=(idx, weight))


@pytest.mark.parametrize("top_k, scaling", [(1, 1.0), (4, 2.5)])
def test_moe_route_matches_kimi_gate(generator, top_k, scaling):
    """top_k == 1 skips renormalization and the routed scale is applied, as in KimiMoEGate."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiMoEGate
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.reference_ops import moe_route

    cfg = _situ_config(
        32,
        64,
        num_experts=12,
        num_experts_per_token=top_k,
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=scaling,
    )
    gate = KimiMoEGate(cfg).eval()
    with torch.no_grad():
        torch.nn.init.normal_(gate.weight, std=0.5, generator=generator)
        torch.nn.init.normal_(gate.e_score_correction_bias, std=0.05, generator=generator)
    x = torch.randn(1, 30, 32, generator=generator)
    with torch.no_grad():
        expected_idx, expected_weight = gate(x)
    idx, weight = moe_route(x[0], gate.weight.detach(), gate.e_score_correction_bias.detach(), top_k, scaling)
    order, expected_order = idx.argsort(-1), expected_idx.argsort(-1)
    assert torch.equal(idx.gather(1, order), expected_idx.gather(1, expected_order))
    _close(weight.gather(1, order), expected_weight.gather(1, expected_order), "gate weights")


@pytest.mark.parametrize("gate_floor", [-5.0, -40.0], ids=["k3_gate_floor", "past_factoring_limit"])
@pytest.mark.parametrize("tokens", [1, 17, 70])
def test_chunked_kda_recurrence_matches_token_loop(generator, gate_floor, tokens):
    """The suite's chunked recurrence against the reference token loop: blocks that end mid-way,
    a carried-in state, and decays strong enough to take the pairwise (non-factored) path."""
    from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.reference_ops import kda_recurrence_chunked

    heads, dim = 3, 16
    q, k, v = (torch.randn(1, tokens, heads, dim, generator=generator) for _ in range(3))
    gate = gate_floor * torch.rand(1, tokens, heads, dim, generator=generator)
    beta = torch.rand(1, tokens, heads, generator=generator)
    state = torch.randn(1, heads, dim, dim, generator=generator) * 0.3
    want_out, want_state = kda_recurrent_reference(q, k, v, gate, beta, state)
    got_out, got_state = kda_recurrence_chunked(q, k, v, gate, beta, state)
    _close(got_out, want_out, "chunked output")
    _close(got_state, want_state, "chunked state")


def test_chunked_kda_recurrence_long_sequence_real_dims(generator):
    """K3's head dim over an LB chunk (80 blocks of carried state), K3's gate floor: the regime the
    suite runs in, where accumulated block-to-block error would show."""
    from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.reference_ops import kda_recurrence_chunked

    tokens, heads, dim = 1280, 2, KimiK3Config.KDA_HEAD_DIM
    q, k, v = (torch.randn(1, tokens, heads, dim, generator=generator) for _ in range(3))
    gate = -5.0 * torch.rand(1, tokens, heads, dim, generator=generator) ** 4  # mostly weak decay: long memory
    beta = torch.rand(1, tokens, heads, generator=generator)
    want_out, want_state = kda_recurrent_reference(q, k, v, gate, beta)
    got_out, got_state = kda_recurrence_chunked(q, k, v, gate, beta)
    _close(got_out, want_out, "chunked output")
    _close(got_state, want_state, "chunked state")


def test_chunked_kda_recurrence_rejects_mismatched_shapes(generator, expect_error):
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.reference_ops import kda_recurrence_chunked

    q = k = v = torch.randn(1, 4, 2, 8, generator=generator)
    with expect_error(ValueError, "gate shape"):
        kda_recurrence_chunked(q, k, v, torch.zeros(1, 2, 4, 8), torch.zeros(1, 4, 2))
