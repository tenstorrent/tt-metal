# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch reference for the Qwen3.8 MTP drafter head.

Mirrors vLLM's Qwen3_5MultiTokenPredictor (HF transformers ignores mtp.* weights,
so vLLM is the reference): fc(cat[norm(embed), norm(hidden)]) -> one full-attention
decoder layer -> final norm -> LM head. Same position convention as tt/mtp.py:
the pair (target_hidden[i], token[i+1]) sits at drafter position i.
"""

import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen36.tt.rope import compute_rope_freqs
from models.experimental.gated_attention_gated_deltanet.torch_functional.gated_attention import (
    gated_attention_forward,
)


def rms_norm_zero_centered(x, weight, eps=1e-6):
    """Qwen3.5 zero-centered RMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + weight)."""
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * (1.0 + weight.float())).to(x.dtype)


class MTPTorchReference:
    """Stateful torch twin of Qwen36MTPHead (per-step KV cache kept internally).

    Args:
        mtp_state_dict: mtp.*-stripped weights (remap_mtp_state_dict).
        embed_weight: [vocab, dim] embedding table (shared with the target).
        lm_head_weight: [out, dim] LM head in HF [out, in] convention.
        num_heads / num_kv_heads / head_dim / rope_head_dim / rope_theta /
        norm_eps / max_seq_len: attention geometry from the model args.
    """

    def __init__(
        self,
        mtp_state_dict,
        embed_weight,
        lm_head_weight,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_head_dim,
        rope_theta,
        norm_eps=1e-6,
        max_seq_len=2048,
    ):
        self.sd = {k: v.float() for k, v in mtp_state_dict.items()}
        self.embed_weight = embed_weight.float()
        self.lm_head_weight = lm_head_weight.float()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.cos_cpu, self.sin_cpu = compute_rope_freqs(rope_head_dim, max_seq_len, theta=rope_theta)
        self.past_k = None
        self.past_v = None

    def reset(self):
        self.past_k = None
        self.past_v = None

    def step(self, token_id, hidden_row, position):
        """One drafter step. Returns (logits [out], hidden [dim]); appends KV.

        The KV cache is append-only (this reference exists for step/chain PCC,
        not for rollback), so callers must feed strictly increasing positions.
        """
        sd = self.sd
        emb = self.embed_weight[token_id].reshape(1, 1, -1)
        e_n = rms_norm_zero_centered(emb, sd["pre_fc_norm_embedding.weight"], eps=self.norm_eps)
        h_n = rms_norm_zero_centered(
            hidden_row.float().reshape(1, 1, -1), sd["pre_fc_norm_hidden.weight"], eps=self.norm_eps
        )
        x = F.linear(torch.cat([e_n, h_n], dim=-1), sd["fc.weight"])

        attn_in = rms_norm_zero_centered(x, sd["layers.0.input_layernorm.weight"], eps=self.norm_eps)
        cos = self.cos_cpu[position : position + 1].unsqueeze(0)  # [1, 1, rope_head_dim]
        sin = self.sin_cpu[position : position + 1].unsqueeze(0)
        # Explicit all-visible mask: torch SDPA's is_causal aligns query row 0 with
        # key 0, which would mask the whole cache on a T=1 decode step.
        s_total = 1 + (self.past_k.shape[2] if self.past_k is not None else 0)
        mask = torch.zeros(1, 1, 1, s_total)
        attn_out, self.past_k, self.past_v = gated_attention_forward(
            hidden_states=attn_in,
            q_proj_weight=sd["layers.0.self_attn.q_proj.weight"],
            k_proj_weight=sd["layers.0.self_attn.k_proj.weight"],
            v_proj_weight=sd["layers.0.self_attn.v_proj.weight"],
            o_proj_weight=sd["layers.0.self_attn.o_proj.weight"],
            q_norm_weight=sd["layers.0.self_attn.q_norm.weight"],
            k_norm_weight=sd["layers.0.self_attn.k_norm.weight"],
            cos=cos,
            sin=sin,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            attention_mask=mask,
            norm_eps=self.norm_eps,
            past_key_value_k=self.past_k,
            past_key_value_v=self.past_v,
            output_kv_cache=True,
        )
        h1 = x + attn_out

        ff_in = rms_norm_zero_centered(h1, sd["layers.0.post_attention_layernorm.weight"], eps=self.norm_eps)
        gate = F.silu(F.linear(ff_in, sd["layers.0.mlp.gate_proj.weight"]))
        up = F.linear(ff_in, sd["layers.0.mlp.up_proj.weight"])
        ff_out = F.linear(gate * up, sd["layers.0.mlp.down_proj.weight"])
        out = h1 + ff_out

        normed = rms_norm_zero_centered(out, sd["norm.weight"], eps=self.norm_eps)
        logits = F.linear(normed, self.lm_head_weight)
        return logits.reshape(-1), normed.reshape(-1)
