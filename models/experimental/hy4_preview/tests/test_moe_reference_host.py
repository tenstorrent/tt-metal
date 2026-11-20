# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Anchor: the standalone feed-forward reference must track HF transformers' HYV4MoE and HYV4MLP.

reference/moe.py is what the device PCC tests validate against; this pins it to the HF modules,
router picks included. Pure CPU (torch vs torch), no device. Skipped when the installed
transformers predates ``hy_v4``.
"""

import pytest
import torch

from models.experimental.hy4_preview.reference.moe import Hy4FFN, Hy4MoE, MoEConfig

hy_v4 = pytest.importorskip("transformers.models.hy_v4.modeling_hy_v4")
from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config  # noqa: E402

SMALL = MoEConfig(
    hidden_size=96,
    num_experts=32,
    top_k=4,
    intermediate_size=48,
    dense_intermediate_size=160,
)


def _hf_config(cfg: MoEConfig) -> HYV4Config:
    return HYV4Config(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.dense_intermediate_size,
        moe_intermediate_size=cfg.intermediate_size,
        n_routed_experts=cfg.num_experts,
        num_experts_per_tok=cfg.top_k,
        n_shared_experts=cfg.n_shared_experts,
        routed_scaling_factor=cfg.routed_scaling_factor,
        swiglu_limit=cfg.swiglu_limit,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        indexer_types=["full", "shared"],
        mlp_layer_types=["dense", "sparse"],
    )


def _randomize(ref, seed, scale=1.0):
    """Linear outputs of order ``scale``; a correction bias large enough to change which experts win."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if p.dim() == 1:  # the router's correction bias
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(scale * torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
    return ref


def _moe_pair(cfg, seed, scale=1.0):
    ref = _randomize(Hy4MoE(cfg).eval(), seed, scale)
    hf = hy_v4.HYV4MoE(_hf_config(cfg)).eval()
    hf_state = hf.state_dict()
    for name, value in ref.weights().items():
        assert hf_state[name].shape == value.shape, (name, hf_state[name].shape, value.shape)
        hf_state[name] = value.clone()
    hf.load_state_dict(hf_state)
    return hf, ref


@pytest.mark.parametrize("tokens", [1, 7, 64], ids=["T1", "T7", "T64"])
def test_moe_matches_hf(tokens):
    cfg = SMALL
    hf, ref = _moe_pair(cfg, seed=3)
    x = torch.randn(tokens, cfg.hidden_size, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        _, hf_weights, hf_indices = hf.gate(x)
        weights, indices = ref.route(x)
        hf_out = hf(x)
        out = ref(x)
    assert torch.equal(indices.sort(dim=-1).values, hf_indices.sort(dim=-1).values)
    assert torch.equal(ref.dense_weights(weights, indices), ref.dense_weights(hf_weights, hf_indices))
    max_diff = (out - hf_out).abs().max().item()
    assert max_diff <= 1e-5 * hf_out.abs().max().item(), f"max|Δ|={max_diff:.3e}"


def test_bias_steers_the_choice_but_not_the_weights():
    cfg = SMALL
    _, ref = _moe_pair(cfg, seed=5)
    x = torch.randn(16, cfg.hidden_size, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        _, base = ref.route(x)
        ref.router_bias[0] = 10.0  # expert 0 now always wins a slot
        weights, indices = ref.route(x)
        scores = ref.scores(x)
    assert not (base == 0).any(dim=-1).all() and (indices == 0).any(dim=-1).all()
    assert torch.allclose(weights.sum(dim=-1), torch.full((16,), cfg.routed_scaling_factor))
    picked = scores.gather(1, indices)
    assert torch.allclose(weights, picked / picked.sum(dim=-1, keepdim=True) * cfg.routed_scaling_factor)


def test_clamp_bites_and_matches_hf():
    """With activations well past the limit the clamp changes the output; HF must agree on by how much."""
    cfg = SMALL
    hf, ref = _moe_pair(cfg, seed=7, scale=8.0)
    x = torch.randn(32, cfg.hidden_size, generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        gate_up = torch.nn.functional.linear(x, ref.gate_up_proj[0])
        assert (gate_up.abs() > cfg.swiglu_limit).float().mean() > 0.05
        out, hf_out = ref(x), hf(x)
        ref.cfg = MoEConfig(**{**cfg.__dict__, "swiglu_limit": 1e9})
        unclamped = ref(x)
    assert (out - hf_out).abs().max() <= 1e-5 * hf_out.abs().max()
    assert (out - unclamped).abs().max() > 0.1 * out.abs().max()


@pytest.mark.parametrize("which", ["dense", "shared"])
def test_ffn_matches_hf(which):
    cfg = SMALL
    intermediate = cfg.dense_intermediate_size if which == "dense" else cfg.shared_intermediate_size
    ref = _randomize(Hy4FFN(cfg.hidden_size, intermediate).eval(), seed=9)
    hf = hy_v4.HYV4MLP(_hf_config(cfg), intermediate_size=intermediate).eval()
    hf.load_state_dict({name: value.clone() for name, value in ref.weights().items()})
    x = torch.randn(20, cfg.hidden_size, generator=torch.Generator().manual_seed(10))
    with torch.no_grad():
        out, hf_out = ref(x), hf(x)
    assert (out - hf_out).abs().max() <= 1e-5 * hf_out.abs().max()
