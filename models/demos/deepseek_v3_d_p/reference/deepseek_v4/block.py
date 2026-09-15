# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One DeepSeek-V4 decoder block on CPU, composed from the reference modules, plus the same weights in
the layout TtV4Block wants.

Composed rather than driven through DeepseekV4DecoderLayer because there is no V4 checkpoint: the
weights are random either way, so a whole-model module buys nothing and costs the MoE's runtime.
"""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4HyperConnection,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4SparseMoeBlock,
)


class _RefCache:
    """Minimal ``past_key_values``: attention calls ``.update(k, v, layer_idx)`` on the container and
    the compressor reaches into ``.layers[layer_idx]``."""

    def __init__(self, layer):
        self.layers = [layer]

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return self.layers[layer_idx].update(key_states, value_states)


def build_v4_block_reference(config, layer_idx: int, seed: int = 0):
    """A randomised V4 decoder block: the two norms, the layer's attention, and its MoE.

    Random, and deliberately not near-identity: the norm gains are drawn away from 1.0 and the sinks
    from N(0,1), so a PCC pass cannot come from weights that make the block an identity map.
    """
    torch.manual_seed(seed)
    attn = DeepseekV4Attention(config, layer_idx=layer_idx).eval()
    with torch.no_grad():
        attn.q_a_norm.weight.uniform_(0.5, 1.5)
        attn.kv_norm.weight.uniform_(0.5, 1.5)
        attn.sinks.normal_(0.0, 1.0)
        if attn.compressor is not None:
            attn.compressor.position_bias.normal_(0.0, 0.02)
            attn.compressor.kv_norm.weight.uniform_(0.5, 1.5)
        else:
            # A sliding-only layer carries no rotary_emb of its own; the config determines it.
            attn.rotary_emb = DeepseekV4RotaryEmbedding(config)

    mlp = DeepseekV4SparseMoeBlock(config, layer_idx=layer_idx).eval()
    hidden, inter, n_exp = config.hidden_size, config.intermediate_size, config.num_local_experts
    hs, ds = hidden**-0.5, inter**-0.5
    with torch.no_grad():
        mlp.gate.weight.normal_(0.0, hs)
        mlp.experts.gate_up_proj.normal_(0.0, hs)
        mlp.experts.down_proj.normal_(0.0, ds)
        for p in (mlp.shared_experts.gate_proj, mlp.shared_experts.up_proj):
            p.weight.normal_(0.0, hs)
        mlp.shared_experts.down_proj.weight.normal_(0.0, ds)
        if mlp.is_hash:
            # A frozen token-id -> expert-id table. Random ids, so routing is not a function of
            # anything the device could recompute from the hidden state.
            mlp.gate.tid2eid.copy_(torch.randint(0, n_exp, mlp.gate.tid2eid.shape))
        else:
            mlp.gate.e_score_correction_bias.normal_(0.0, 0.01)

    # Two hyper-connections, one per site, with independent parameters, initialised the way the model
    # prescribes (DeepseekV4PreTrainedModel._init_weights): fn ~ N(0, initializer_range), base zero,
    # scale one. A non-zero base with a small scale -- which tt/mhc's own op test uses deliberately,
    # to cover the bias add -- would make the projection ~1% of each sigmoid's argument and the rest
    # bias, leaving this test blind to whether the projection is computed correctly at all.
    hcs = {}
    for site in ("attn", "ffn"):
        hc = DeepseekV4HyperConnection(config).eval()
        with torch.no_grad():
            hc.fn.normal_(0.0, config.initializer_range)
            hc.base.zero_()
            hc.scale.fill_(1.0)
        hcs[site] = hc

    attn_norm = DeepseekV4RMSNorm(hidden, eps=config.rms_norm_eps).eval()
    ffn_norm = DeepseekV4RMSNorm(hidden, eps=config.rms_norm_eps).eval()
    with torch.no_grad():
        attn_norm.weight.uniform_(0.5, 1.5)
        ffn_norm.weight.uniform_(0.5, 1.5)
    return {
        "attn": attn,
        "mlp": mlp,
        "attn_norm": attn_norm,
        "ffn_norm": ffn_norm,
        "attn_hc": hcs["attn"],
        "ffn_hc": hcs["ffn"],
    }


def v4_mhc_weights(ref):
    """``ref`` -> the ``(fn, base, scale)`` triples TtV4Block's mhc_weights wants, per site."""
    return {
        site: (
            ref[f"{site}_hc"].fn.detach(),
            ref[f"{site}_hc"].base.detach(),
            ref[f"{site}_hc"].scale.detach(),
        )
        for site in ("attn", "ffn")
    }


def v4_block_state_dict(ref, config):
    """``ref`` -> the state-dict keys TtV4Block reads.

    The experts are stored packed: ``gate_up_proj`` is ``[experts, 2*inter, hidden]`` whose first
    ``inter`` rows are the gate half (``_apply_gate`` chunks the [T, 2*inter] product, so the split is
    by output row), and ``down_proj`` is ``[experts, hidden, inter]`` -- already the per-expert layout.
    """
    mlp, inter = ref["mlp"], config.intermediate_size
    gate_up, down = mlp.experts.gate_up_proj.detach(), mlp.experts.down_proj.detach()
    experts = [
        {
            "gate_proj": gate_up[e, :inter].to(torch.bfloat16),
            "up_proj": gate_up[e, inter:].to(torch.bfloat16),
            "down_proj": down[e].to(torch.bfloat16),
        }
        for e in range(config.num_local_experts)
    ]
    shared = {
        "gate_proj": mlp.shared_experts.gate_proj.weight.detach().to(torch.bfloat16),
        "up_proj": mlp.shared_experts.up_proj.weight.detach().to(torch.bfloat16),
        "down_proj": mlp.shared_experts.down_proj.weight.detach().to(torch.bfloat16),
    }
    # e_score_correction_bias biases selection only, and hash routing does not select; the device gate
    # still wants the tensor, so a hash layer hands it zeros.
    bias = getattr(mlp.gate, "e_score_correction_bias", None)
    state = {
        "attn_norm_weight": ref["attn_norm"].weight.detach(),
        "ffn_norm_weight": ref["ffn_norm"].weight.detach(),
        "gate_weights": {
            "weight": mlp.gate.weight.detach().to(torch.bfloat16),
            "e_score_correction_bias": (
                bias.detach().to(torch.float32)
                if bias is not None
                else torch.zeros(config.num_local_experts, dtype=torch.float32)
            ),
        },
        "routed_expert_weights": experts,
        "shared_expert_weights": shared,
    }
    if mlp.is_hash:
        state["hash_table"] = mlp.gate.tid2eid.detach()
    return state


def v4_block_forward(ref, config, hidden_states, input_ids):
    """One unchunked pass, which the device's chunked path has to reproduce.

    ``hidden_states`` is [1, seq, hc_mult, hidden] in and out: V4's residual is hc_mult parallel
    streams mixed by two hyper-connections, not a sum. ``input_ids`` drives hash routing and a top-k
    layer ignores it.
    """
    attn, mlp = ref["attn"], ref["mlp"]
    seq = hidden_states.shape[1]
    position_ids = torch.arange(seq).unsqueeze(0).expand(1, -1)
    sw = config.sliding_window

    i = torch.arange(seq).view(seq, 1)
    j = torch.arange(seq).view(1, seq)
    mask = torch.zeros(seq, seq).masked_fill(~((j <= i) & (i - j < sw)), float("-inf")).view(1, 1, seq, seq)
    rope_src = attn.compressor.rotary_emb if attn.compressor is not None else attn.rotary_emb

    def _attend(collapsed):
        cos, sin = rope_src(collapsed, position_ids=position_ids, layer_type=attn.rope_layer_type)
        out, _ = attn(
            ref["attn_norm"](collapsed), {attn.rope_layer_type: (cos, sin)}, position_ids, mask, past_key_values=None
        )
        return out

    with torch.no_grad():
        # comb is consumed transposed: sum_j comb[j, k] * residual[j, d]. Sinkhorn yields a
        # doubly-stochastic but non-symmetric matrix, so the direction matters.
        h, dtype = hidden_states, hidden_states.dtype
        post, comb, collapsed = ref["attn_hc"](h)
        h = post.to(dtype).unsqueeze(-1) * _attend(collapsed).unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), h
        )
        post, comb, collapsed = ref["ffn_hc"](h)
        mlp_out = mlp(ref["ffn_norm"](collapsed), input_ids=input_ids)
        return post.to(dtype).unsqueeze(-1) * mlp_out.unsqueeze(-2) + torch.matmul(comb.to(dtype).transpose(-1, -2), h)
