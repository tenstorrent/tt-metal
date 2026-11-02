# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Torch reference of the z-lab/gpt-oss-20b-DFlash drafter and of DFlash greedy speculative decoding.

The drafter (0.8B parameters, the checkpoint's dflash.py DFlashDraftModel) is 8 Qwen3-style layers (hidden 2880, 64
query / 8 KV heads of 64, q / k RMSNorm, biased projections, YaRN RoPE, SiLU MLP of 7680) that draft a block of 8
tokens at once:
  - context: the target's hidden states (outputs of target layers 1, 6, 11, 16, 21, concatenated: 5 x 2880) of every
    accepted token, projected by fc (14400 -> 2880) and RMSNorm'ed (hidden_norm), give each draft layer's context K / V;
  - noise: the target's embeddings of the block [last accepted token, mask x 7] (positions start .. start + 7) give the
    queries and the block's own K / V;
  - attention is not causal (every noise row sees all context and all noise rows); the final RMSNorm'ed rows 1..7 go
    through the target's LM head and their argmax are the 7 draft tokens.
The target then scores the 8-token block in one pass; the drafts are accepted up to the first mismatch with the
target's own greedy choice, plus the target's token after them (the checkpoint's spec_generate). Greedy DFlash output
therefore equals the target's greedy output.

This module recomputes the drafter over the whole context every round (no draft KV cache); for a causal target the
context hidden states do not depend on later tokens, so that equals the cached spec_generate.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from models.demos.gpt_oss.tt.dflash.draft import draft_snapshot, load_draft_config


def yarn_cos_sin(positions, path=None, dtype=torch.float32):
    """cos / sin [len(positions), head_dim] of the drafter's YaRN RoPE (transformers' own yarn init, the attention
    factor folded in, as Qwen3RotaryEmbedding)."""
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    config = AutoConfig.from_pretrained(str(path or draft_snapshot()))
    inv_freq, attention_factor = ROPE_INIT_FUNCTIONS["yarn"](config, "cpu")
    freqs = torch.outer(torch.as_tensor(positions, dtype=torch.float32), inv_freq.float())
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * attention_factor).to(dtype), (emb.sin() * attention_factor).to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rms_norm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


class DraftReference:
    """state: the checkpoint tensors (model.safetensors keys); forward(ctx_hidden [n, 5 * hidden], noise [8, hidden],
    ctx_positions, noise_positions) -> the final normed noise rows [8, hidden]."""

    def __init__(self, path=None, dtype=torch.float32):
        from safetensors.torch import load_file

        self.path = Path(path or draft_snapshot())
        self.cfg = load_draft_config(self.path)
        self.dtype = dtype
        self.w = {k: v.to(dtype) for k, v in load_file(str(self.path / "model.safetensors")).items()}
        c = self.cfg
        self.eps = c["rms_norm_eps"]
        self.hd = c["head_dim"]
        self.nq, self.nkv = c["num_attention_heads"], c["num_key_value_heads"]
        self.layers = c["num_hidden_layers"]
        self.target_layer_ids = c["dflash_config"]["target_layer_ids"]
        self.mask_token_id = c["dflash_config"]["mask_token_id"]
        self.block_size = c["block_size"]

    def forward(self, ctx_hidden, noise, ctx_positions, noise_positions):
        w, hd, eps = self.w, self.hd, self.eps
        ctx = rms_norm(ctx_hidden.to(self.dtype) @ w["fc.weight"].T, w["hidden_norm.weight"], eps)
        h = noise.to(self.dtype)
        cos, sin = yarn_cos_sin(list(ctx_positions) + list(noise_positions), self.path, self.dtype)
        n_ctx, n_q = ctx.shape[0], h.shape[0]
        for i in range(self.layers):
            p = f"layers.{i}."
            res = h
            x = rms_norm(h, w[p + "input_layernorm.weight"], eps)
            a = p + "self_attn."
            q = (x @ w[a + "q_proj.weight"].T + w[a + "q_proj.bias"]).view(n_q, self.nq, hd)
            kv_in = torch.cat([ctx, x])
            k = (kv_in @ w[a + "k_proj.weight"].T + w[a + "k_proj.bias"]).view(n_ctx + n_q, self.nkv, hd)
            v = (kv_in @ w[a + "v_proj.weight"].T + w[a + "v_proj.bias"]).view(n_ctx + n_q, self.nkv, hd)
            q = rms_norm(q, w[a + "q_norm.weight"], eps)
            k = rms_norm(k, w[a + "k_norm.weight"], eps)
            q = q * cos[-n_q:, None] + rotate_half(q) * sin[-n_q:, None]
            k = k * cos[:, None] + rotate_half(k) * sin[:, None]
            g = self.nq // self.nkv
            kq = k.repeat_interleave(g, dim=1)  # [kv_len, nq, hd]
            vq = v.repeat_interleave(g, dim=1)
            s = torch.einsum("qhd,khd->hqk", q.float(), kq.float()) * hd**-0.5
            o = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), vq.float()).to(self.dtype).reshape(n_q, -1)
            h = res + (o @ w[a + "o_proj.weight"].T + w[a + "o_proj.bias"])
            res = h
            x = rms_norm(h, w[p + "post_attention_layernorm.weight"], eps)
            m = p + "mlp."
            x = F.silu(x @ w[m + "gate_proj.weight"].T) * (x @ w[m + "up_proj.weight"].T)
            h = res + x @ w[m + "down_proj.weight"].T
        return rms_norm(h, w["norm.weight"], eps)


def target_context(hidden_states, layer_ids):
    """The drafter's context features from a HF output's hidden_states (index L + 1 = output of target layer L)."""
    return torch.cat([hidden_states[i + 1] for i in layer_ids], dim=-1)


@torch.no_grad()
def dflash_generate(target, draft, prompt_ids, max_new_tokens, eos_ids=(), log=None):
    """Greedy DFlash (spec_generate semantics) with full recomputation. Returns (tokens after the prompt: at most
    max_new_tokens, up to and including the first EOS; acceptance lengths per round, [(start, draft tokens)] per
    round)."""
    embed = target.get_input_embeddings()
    lm_head = target.get_output_embeddings()
    ids = list(prompt_ids)
    out = target(torch.tensor([ids]), output_hidden_states=True)
    ids.append(int(out.logits[0, -1].argmax()))
    ctx = target_context(out.hidden_states, draft.target_layer_ids)[0]  # [len(prompt), 5 * hidden]
    accept, drafts = [], []
    B = draft.block_size
    while len(ids) - len(prompt_ids) < max_new_tokens:
        start = len(ids) - 1  # position of the last (already chosen, not yet verified-into-cache) token
        block = [ids[-1]] + [draft.mask_token_id] * (B - 1)
        noise = embed(torch.tensor(block)).to(draft.dtype)
        hid = draft.forward(ctx[:start], noise, range(start), range(start, start + B))
        d = lm_head(hid[1:].to(lm_head.weight.dtype)).argmax(-1).tolist()
        drafts.append((start, d))
        full = ids[:-1] + [ids[-1]] + d
        o = target(torch.tensor([full]), output_hidden_states=True)
        post = o.logits[0, start:].argmax(-1).tolist()  # target's choice after each block row
        a = 0
        while a < B - 1 and d[a] == post[a]:
            a += 1
        ids = ids[:-1] + [ids[-1]] + d[:a] + [post[a]]
        ctx = target_context(o.hidden_states, draft.target_layer_ids)[0][: len(ids) - 1]
        accept.append(a + 1)
        if log:
            log(f"round {len(accept)}: start {start} accepted {a + 1}")
        if any(t in eos_ids for t in ids[len(prompt_ids) :]):
            break
    toks = ids[len(prompt_ids) :][: max(max_new_tokens, 0)]  # a round appends its whole accepted block
    end = next((i + 1 for i, t in enumerate(toks) if t in eos_ids), len(toks))
    return toks[:end], accept, drafts
