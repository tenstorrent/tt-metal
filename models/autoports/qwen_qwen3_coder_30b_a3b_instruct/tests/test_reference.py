# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validate the layer-only reference before any TTNN work is built against it.

The reference does one thing the checkpoint does not: it fuses each expert's
``gate_proj``/``up_proj`` into a single ``gate_up_proj`` and stacks all experts.
If that fusion were reversed, the layer would still run and produce
plausible-looking numbers -- it would just be wrong. Everything downstream is
compared against this reference, so an error here is invisible forever after.

``test_moe_matches_unfused_reimplementation`` therefore recomputes the MoE block
from the raw per-expert checkpoint tensors, following
``Qwen3MoeSparseMoeBlock.forward`` literally, and requires the two to agree.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc

from .reference import build_reference_layer, layer_state_dict, rotary_embeddings

LAYER_IDX = 0
SEQ_LEN = 32


@pytest.fixture(scope="module")
def reference():
    layer, config = build_reference_layer(LAYER_IDX)
    return layer, config


def _hidden(config, seq_len=SEQ_LEN, seed=0):
    """Activations roughly matching what reaches a decoder layer post-embedding."""
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, config.hidden_size, dtype=torch.float32) * 0.02


def test_layer_forward_is_finite_and_non_degenerate(reference):
    layer, config = reference
    hidden = _hidden(config)
    cos, sin = rotary_embeddings(config, SEQ_LEN)

    with torch.no_grad():
        out = layer(hidden, position_embeddings=(cos, sin), attention_mask=None)
    out = out[0] if isinstance(out, tuple) else out

    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()
    assert out.std() > 1e-6, "output is constant -- layer is not doing anything"
    assert not torch.allclose(out, hidden), "output equals input -- residual-only path"


def test_layer_forward_is_deterministic(reference):
    layer, config = reference
    hidden = _hidden(config)
    cos, sin = rotary_embeddings(config, SEQ_LEN)

    with torch.no_grad():
        a = layer(hidden, position_embeddings=(cos, sin), attention_mask=None)
        b = layer(hidden, position_embeddings=(cos, sin), attention_mask=None)
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b

    assert torch.equal(a, b), "reference is not deterministic; PCC comparisons would be unstable"


def test_moe_matches_unfused_reimplementation(reference):
    """Recompute the MoE block from raw per-expert tensors and require agreement.

    This is the guard on the gate/up fusion order and on the router's
    softmax -> top-k -> renormalise ordering.
    """
    layer, config = reference
    # build_reference_layer upcasts the bf16 checkpoint into the fp32 module, so
    # cast here too -- this test is about the fusion order, not about dtype.
    sd = {k: v.float() for k, v in layer_state_dict(LAYER_IDX).items()}
    hidden = _hidden(config)
    flat = hidden.view(-1, config.hidden_size)

    with torch.no_grad():
        fused_out = layer.mlp(hidden).view(-1, config.hidden_size)

    # --- independent implementation, straight from Qwen3MoeTopKRouter.forward ---
    logits = torch.nn.functional.linear(flat, sd["mlp.gate.weight"])
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)  # over ALL experts, fp32
    top_w, top_i = torch.topk(probs, config.num_experts_per_tok, dim=-1)
    if config.norm_topk_prob:
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    top_w = top_w.to(logits.dtype)

    # --- and from Qwen3MoeExperts.forward, but with unfused checkpoint tensors ---
    manual = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for slot in range(config.num_experts_per_tok):
            e = int(top_i[token, slot])
            x = flat[token]
            gate = torch.nn.functional.linear(x, sd[f"mlp.experts.{e}.gate_proj.weight"])
            up = torch.nn.functional.linear(x, sd[f"mlp.experts.{e}.up_proj.weight"])
            h = torch.nn.functional.silu(gate) * up
            manual[token] += torch.nn.functional.linear(h, sd[f"mlp.experts.{e}.down_proj.weight"]) * top_w[token, slot]

    passing, message = comp_pcc(manual, fused_out, pcc=0.9999)
    assert passing, f"fused MoE disagrees with unfused checkpoint math: {message}"


def test_router_selects_expected_expert_count(reference):
    """top_k experts per token, weights renormalised to 1."""
    layer, config = reference
    hidden = _hidden(config)
    flat = hidden.view(-1, config.hidden_size)

    with torch.no_grad():
        _, scores, indices = layer.mlp.gate(flat)

    assert indices.shape == (flat.shape[0], config.num_experts_per_tok)
    assert indices.min() >= 0 and indices.max() < config.num_experts
    if config.norm_topk_prob:
        sums = scores.float().sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"router weights not normalised: {sums[:4]}"
