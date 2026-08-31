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
from models.experimental.gated_attention_gated_deltanet.torch_functional.gated_attention import gated_attention_forward


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
        self._kv = {}  # position -> (k [1,H_kv,1,D], v [1,H_kv,1,D])

    def reset(self):
        self._kv = {}

    def step(self, token_id, hidden_row, position):
        """One drafter step. Returns (logits [out], hidden [dim]).

        KV is position-addressed (paged-cache semantics): re-stepping a slot
        overwrites it, so rejected-draft replays behave like the TT head.
        Slots 0..position-1 must all be populated.
        """
        sd = self.sd
        missing = [p for p in range(position) if p not in self._kv]
        assert not missing, f"drafter KV slots {missing} unwritten below position {position}"
        if position > 0:
            past_k = torch.cat([self._kv[p][0] for p in range(position)], dim=2)
            past_v = torch.cat([self._kv[p][1] for p in range(position)], dim=2)
        else:
            past_k = past_v = None
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
        mask = torch.zeros(1, 1, 1, position + 1)
        attn_out, k_all, v_all = gated_attention_forward(
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
            past_key_value_k=past_k,
            past_key_value_v=past_v,
            output_kv_cache=True,
        )
        self._kv[position] = (k_all[:, :, -1:], v_all[:, :, -1:])
        h1 = x + attn_out

        ff_in = rms_norm_zero_centered(h1, sd["layers.0.post_attention_layernorm.weight"], eps=self.norm_eps)
        gate = F.silu(F.linear(ff_in, sd["layers.0.mlp.gate_proj.weight"]))
        up = F.linear(ff_in, sd["layers.0.mlp.up_proj.weight"])
        ff_out = F.linear(gate * up, sd["layers.0.mlp.down_proj.weight"])
        out = h1 + ff_out

        normed = rms_norm_zero_centered(out, sd["norm.weight"], eps=self.norm_eps)
        logits = F.linear(normed, self.lm_head_weight)
        return logits.reshape(-1), normed.reshape(-1)


def build_mtp_reference(model_path=None, max_seq_len=2048) -> "MTPTorchReference":
    """Wire up ``MTPTorchReference`` against a real checkpoint's actual weights.

    Loads the checkpoint's real ``mtp.*`` head weights plus its real shared
    ``embed_tokens`` / ``lm_head`` (not the small random stand-ins
    ``tests/unit/test_mtp.py`` uses for a portable single-device PCC gate) —
    this is the reference the TT head's *end-to-end* numerics should be
    checked against, not just its component-level math.

    Args:
        model_path: checkpoint dir; defaults to ``Qwen36ModelArgs``' resolved
            ``CKPT_DIR`` (``HF_MODEL`` env var, snapshot-downloaded if it's a
            hub id). CPU-only (``mesh_device=None``) — no TT device required.
        max_seq_len: RoPE table length; must cover the longest drafter
            position you intend to step to.
    """
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.demos.blackhole.qwen36.tt.weight_mapping import (
        checkpoint_has_mtp,
        load_qwen36_mtp_state_dict,
        load_qwen36_shared_head_weights,
    )

    args = Qwen36ModelArgs(mesh_device=None)
    ckpt_dir = model_path or args.CKPT_DIR
    assert checkpoint_has_mtp(ckpt_dir), f"{ckpt_dir} carries no mtp.* weights"

    mtp_sd = load_qwen36_mtp_state_dict(ckpt_dir)
    heads = load_qwen36_shared_head_weights(ckpt_dir)

    return MTPTorchReference(
        mtp_sd,
        embed_weight=heads["embed_weight"],
        lm_head_weight=heads["lm_head_weight"],
        num_heads=args.n_heads,
        num_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim,
        rope_theta=args.rope_theta,
        norm_eps=args.norm_eps,
        max_seq_len=max_seq_len,
    )


if __name__ == "__main__":
    """Smoke-run build_mtp_reference() against a real checkpoint's real weights.

        HF_MODEL=/path/to/checkpoint python models/demos/blackhole/qwen36/reference/mtp_torch.py

    HF_MODEL defaults to Qwen/Qwen3.6-27B (see Qwen36ModelArgs) if unset. Feeds
    a synthetic (random) hidden state at position 0 — this only proves the
    real-weight wiring runs and produces finite output; it is not yet fed a
    real target hidden state from an actual backbone forward pass.
    """
    torch.manual_seed(0)
    ref = build_mtp_reference()
    print(f"embed_weight  {tuple(ref.embed_weight.shape)} {ref.embed_weight.dtype}")
    print(f"lm_head_weight{tuple(ref.lm_head_weight.shape)} {ref.lm_head_weight.dtype}")
    print(f"num_heads={ref.num_heads} num_kv_heads={ref.num_kv_heads} head_dim={ref.head_dim}")

    hidden = torch.randn(ref.embed_weight.shape[1]) * 0.02
    logits, hidden_out = ref.step(token_id=100, hidden_row=hidden, position=0)
    print(f"logits    {tuple(logits.shape)} finite={torch.isfinite(logits).all().item()}")
    print(f"hidden_out{tuple(hidden_out.shape)} finite={torch.isfinite(hidden_out).all().item()}")
    print(f"top5 token ids: {torch.topk(logits, 5).indices.tolist()}")
