# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def _to_tt(t: torch.Tensor, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh_device)


def _split_in_proj(in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor):
    """nn.MultiheadAttention packs Q/K/V into a single in_proj_weight of shape [3*dim, dim]."""
    dim = in_proj_weight.shape[1]
    qw, kw, vw = in_proj_weight.split(dim, dim=0)
    qb, kb, vb = (in_proj_bias.split(dim, dim=0) if in_proj_bias is not None else (None, None, None))
    return qw, kw, vw, qb, kb, vb


def _linear_weight(w: torch.Tensor) -> torch.Tensor:
    """ttnn.linear expects weight in [in, out] order; PyTorch stores [out, in]."""
    return w.transpose(0, 1).contiguous()


def _attention(query, key, value, qw, kw, vw, qb, kb, vb, ow, ob, num_heads):
    """Multi-head attention. query/key/value are ttnn tensors of shape [B, S, D]."""
    q = ttnn.linear(query, qw, bias=qb)
    k = ttnn.linear(key, kw, bias=kb)
    v = ttnn.linear(value, vw, bias=vb)

    batch, sq, dim = q.shape
    sk = k.shape[1]
    head_dim = dim // num_heads

    q = ttnn.reshape(q, (batch, sq, num_heads, head_dim))
    k = ttnn.reshape(k, (batch, sk, num_heads, head_dim))
    v = ttnn.reshape(v, (batch, sk, num_heads, head_dim))

    q = ttnn.transpose(q, 1, 2)
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)

    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = ttnn.transpose(out, 1, 2)
    out = ttnn.reshape(out, (batch, sq, dim))
    return ttnn.linear(out, ow, bias=ob)


def _transformer_encoder_layer(x, layer_weights, num_heads):
    """A single nn.TransformerEncoderLayer with norm_first=True, gelu activation."""
    norm_x = ttnn.layer_norm(x, weight=layer_weights["ln1_w"], bias=layer_weights["ln1_b"])
    attn_out = _attention(
        norm_x, norm_x, norm_x,
        layer_weights["q_w"], layer_weights["k_w"], layer_weights["v_w"],
        layer_weights["q_b"], layer_weights["k_b"], layer_weights["v_b"],
        layer_weights["o_w"], layer_weights["o_b"],
        num_heads,
    )
    x = ttnn.add(x, attn_out)

    norm_x = ttnn.layer_norm(x, weight=layer_weights["ln2_w"], bias=layer_weights["ln2_b"])
    ff = ttnn.linear(norm_x, layer_weights["ff1_w"], bias=layer_weights["ff1_b"])
    ff = ttnn.gelu(ff)
    ff = ttnn.linear(ff, layer_weights["ff2_w"], bias=layer_weights["ff2_b"])
    return ttnn.add(x, ff)


class TtMAFBlock:
    """TTNN port of AudioX Multimodal Adaptive Fusion block."""

    def __init__(self, mesh_device, state_dict: dict, dim: int, num_heads: int,
                 num_experts_per_modality: int, num_fusion_layers: int, mlp_ratio: float = 4.0):
        self.mesh_device = mesh_device
        self.dim = dim
        self.num_heads = num_heads
        self.num_fusion_layers = num_fusion_layers

        sd = state_dict
        ff_dim = int(dim * mlp_ratio)

        self.gating_w1 = _to_tt(_linear_weight(sd["gating_network.0.weight"]), mesh_device)
        self.gating_b1 = _to_tt(sd["gating_network.0.bias"], mesh_device)
        self.gating_w2 = _to_tt(_linear_weight(sd["gating_network.2.weight"]), mesh_device)
        self.gating_b2 = _to_tt(sd["gating_network.2.bias"], mesh_device)

        self.unified_experts = _to_tt(sd["unified_experts"], mesh_device)

        cqw, ckw, cvw, cqb, ckb, cvb = _split_in_proj(sd["cross_attn.in_proj_weight"], sd["cross_attn.in_proj_bias"])
        self.ca_qw = _to_tt(_linear_weight(cqw), mesh_device)
        self.ca_kw = _to_tt(_linear_weight(ckw), mesh_device)
        self.ca_vw = _to_tt(_linear_weight(cvw), mesh_device)
        self.ca_qb = _to_tt(cqb, mesh_device)
        self.ca_kb = _to_tt(ckb, mesh_device)
        self.ca_vb = _to_tt(cvb, mesh_device)
        self.ca_ow = _to_tt(_linear_weight(sd["cross_attn.out_proj.weight"]), mesh_device)
        self.ca_ob = _to_tt(sd["cross_attn.out_proj.bias"], mesh_device)

        self.norm1_w = _to_tt(sd["norm1.weight"], mesh_device)
        self.norm1_b = _to_tt(sd["norm1.bias"], mesh_device)

        self.fusion_layers = []
        for i in range(num_fusion_layers):
            p = f"fusion_transformer.layers.{i}"
            qw, kw, vw, qb, kb, vb = _split_in_proj(sd[f"{p}.self_attn.in_proj_weight"], sd[f"{p}.self_attn.in_proj_bias"])
            self.fusion_layers.append({
                "q_w": _to_tt(_linear_weight(qw), mesh_device),
                "k_w": _to_tt(_linear_weight(kw), mesh_device),
                "v_w": _to_tt(_linear_weight(vw), mesh_device),
                "q_b": _to_tt(qb, mesh_device),
                "k_b": _to_tt(kb, mesh_device),
                "v_b": _to_tt(vb, mesh_device),
                "o_w": _to_tt(_linear_weight(sd[f"{p}.self_attn.out_proj.weight"]), mesh_device),
                "o_b": _to_tt(sd[f"{p}.self_attn.out_proj.bias"], mesh_device),
                "ff1_w": _to_tt(_linear_weight(sd[f"{p}.linear1.weight"]), mesh_device),
                "ff1_b": _to_tt(sd[f"{p}.linear1.bias"], mesh_device),
                "ff2_w": _to_tt(_linear_weight(sd[f"{p}.linear2.weight"]), mesh_device),
                "ff2_b": _to_tt(sd[f"{p}.linear2.bias"], mesh_device),
                "ln1_w": _to_tt(sd[f"{p}.norm1.weight"], mesh_device),
                "ln1_b": _to_tt(sd[f"{p}.norm1.bias"], mesh_device),
                "ln2_w": _to_tt(sd[f"{p}.norm2.weight"], mesh_device),
                "ln2_b": _to_tt(sd[f"{p}.norm2.bias"], mesh_device),
            })

        self.norm_v2_w = _to_tt(sd["norm_v2.weight"], mesh_device)
        self.norm_v2_b = _to_tt(sd["norm_v2.bias"], mesh_device)
        self.norm_t2_w = _to_tt(sd["norm_t2.weight"], mesh_device)
        self.norm_t2_b = _to_tt(sd["norm_t2.bias"], mesh_device)
        self.norm_a2_w = _to_tt(sd["norm_a2.weight"], mesh_device)
        self.norm_a2_b = _to_tt(sd["norm_a2.bias"], mesh_device)

        self.bypass_v = float(torch.sigmoid(sd["bypass_gate_v"]).item())
        self.bypass_t = float(torch.sigmoid(sd["bypass_gate_t"]).item())
        self.bypass_a = float(torch.sigmoid(sd["bypass_gate_a"]).item())

    def __call__(self, video_tokens: ttnn.Tensor, text_tokens: ttnn.Tensor, audio_tokens: ttnn.Tensor) -> dict:
        batch = video_tokens.shape[0]

        v_global = ttnn.mean(video_tokens, dim=1)
        t_global = ttnn.mean(text_tokens, dim=1)
        a_global = ttnn.mean(audio_tokens, dim=1)

        all_global = ttnn.concat([v_global, t_global, a_global], dim=-1)
        h = ttnn.linear(all_global, self.gating_w1, bias=self.gating_b1)
        h = ttnn.gelu(h)
        h = ttnn.linear(h, self.gating_w2, bias=self.gating_b2)
        gates = ttnn.sigmoid(h)
        w_v, w_t, w_a = ttnn.chunk(gates, 3, dim=-1)

        w_v = ttnn.unsqueeze(w_v, -1)
        w_t = ttnn.unsqueeze(w_t, -1)
        w_a = ttnn.unsqueeze(w_a, -1)

        gated_v = ttnn.multiply(video_tokens, w_v)
        gated_t = ttnn.multiply(text_tokens, w_t)
        gated_a = ttnn.multiply(audio_tokens, w_a)
        full_context = ttnn.concat([gated_v, gated_t, gated_a], dim=1)

        experts = ttnn.unsqueeze(self.unified_experts, 0)
        experts = ttnn.repeat(experts, ttnn.Shape([batch, 1, 1]))

        info = _attention(
            experts, full_context, full_context,
            self.ca_qw, self.ca_kw, self.ca_vw, self.ca_qb, self.ca_kb, self.ca_vb,
            self.ca_ow, self.ca_ob, self.num_heads,
        )
        updated_experts = ttnn.layer_norm(ttnn.add(experts, info), weight=self.norm1_w, bias=self.norm1_b)

        fused = updated_experts
        for layer in self.fusion_layers:
            fused = _transformer_encoder_layer(fused, layer, self.num_heads)

        fused_v, fused_t, fused_a = ttnn.chunk(fused, 3, dim=1)
        refinement_v = ttnn.mean(fused_v, dim=1)
        refinement_t = ttnn.mean(fused_t, dim=1)
        refinement_a = ttnn.mean(fused_a, dim=1)

        ref_v = ttnn.unsqueeze(ttnn.layer_norm(refinement_v, weight=self.norm_v2_w, bias=self.norm_v2_b), 1)
        ref_t = ttnn.unsqueeze(ttnn.layer_norm(refinement_t, weight=self.norm_t2_w, bias=self.norm_t2_b), 1)
        ref_a = ttnn.unsqueeze(ttnn.layer_norm(refinement_a, weight=self.norm_a2_w, bias=self.norm_a2_b), 1)

        final_v = ttnn.add(video_tokens, ttnn.multiply(ref_v, self.bypass_v))
        final_t = ttnn.add(text_tokens, ttnn.multiply(ref_t, self.bypass_t))
        final_a = ttnn.add(audio_tokens, ttnn.multiply(ref_a, self.bypass_a))

        return {"video": final_v, "text": final_t, "audio": final_a}
