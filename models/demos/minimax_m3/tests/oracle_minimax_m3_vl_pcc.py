# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Numerical ORACLE PCC — our self-authored MiniMax-M3 torch references vs the upstream
`transformers` (>=5.12) `minimax_m3_vl` implementation, on identical random weights.

WHY: our per-module / full-model PCC tests (tests/unit/test_*_vs_ref.py) compare TTNN against
self-authored torch refs — we wrote both sides, so a shared misreading of the arch would pass
silently. This script closes that gap by checking our refs against the REAL impl. Together they
give: TTNN ==(device PCC)== our refs ==(this, exact)== real minimax_m3_vl.

This is a STANDALONE script, NOT a pytest test — the filename has no `test_` prefix on purpose, so
the normal pytest suite (which runs in the ttnn python_env, pinned to an older transformers WITHOUT
minimax_m3_vl) does NOT collect it. It needs transformers>=5.12, which we install into a throwaway
env (we do NOT bump the repo's pinned transformers). CPU only, no checkpoint, no ttnn. Exits 0 on
all-pass.

HOW TO RUN — option A (uv, recommended; fetches deps into a cached ephemeral env):

    uv run --no-project --with "transformers>=5.12" --with "torch" --python 3.10 \
        python models/demos/minimax_m3/tests/oracle_minimax_m3_vl_pcc.py

HOW TO RUN — option B (no uv; a one-off venv):

    python3.10 -m venv /tmp/m3_oracle && /tmp/m3_oracle/bin/pip install -q "transformers>=5.12" torch
    /tmp/m3_oracle/bin/python models/demos/minimax_m3/tests/oracle_minimax_m3_vl_pcc.py

First run downloads transformers + torch (~GB, then cached). Covers RMSNorm (gemma), DenseMLP +
SparseMoeBlock (clamped swigluoai, router, shared, routed_scaling), Attention (per-head QK-norm +
partial RoPE + GQA). Pure torch (HF convention, no Meta-RoPE swizzle — that is TTNN-only).
Expect PCC == 1.0 (both fp32, same weights). Pin an exact transformers version if you want a frozen
golden (a future release could change minimax_m3_vl — in which case this test surfaces the drift).
"""

import sys

import torch
from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as M
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig

torch.manual_seed(0)

# Dims chosen so H, 2*INTER, 2*DENSE_INTER, 2*SHARED_INTER are all distinct (disambiguates the
# fused gate_up weight orientation).
H, NQ, NKV, HD, ROT = 256, 4, 2, 64, 32
INTER, DENSE_INTER, SHARED_INTER, E, TOPK = 96, 160, 192, 6, 2
ALPHA, LIMIT, EPS, SCALE = 1.702, 7.0, 1e-6, 2.0
S = 16

cfg = MiniMaxM3VLTextConfig(
    vocab_size=512,
    hidden_size=H,
    intermediate_size=INTER,
    num_hidden_layers=2,
    num_attention_heads=NQ,
    num_key_value_heads=NKV,
    head_dim=HD,
    rotary_dim=ROT,
    partial_rotary_factor=0.5,
    num_local_experts=E,
    num_experts_per_tok=TOPK,
    dense_intermediate_size=DENSE_INTER,
    shared_intermediate_size=SHARED_INTER,
    routed_scaling_factor=SCALE,
    swiglu_alpha=ALPHA,
    swiglu_limit=LIMIT,
    rms_norm_eps=EPS,
)


def pcc(a, b):
    a, b = a.detach().flatten().float(), b.detach().flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ----- our self-authored M3 torch references (HF convention) -----
def gemma_norm(x, w):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(v + EPS)) * (1.0 + w.float())


def clamped_swiglu(gate, up):
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    return (up + 1.0) * (gate * torch.sigmoid(ALPHA * gate))


def fused_ffn(x, gu, down, inter):
    """Orientation-robust fused-gate_up FFN: gate=first chunk of (x @ gate_up), clamped swigluoai."""
    gu = gu if gu.shape[-1] == 2 * inter else gu.t()  # -> [H, 2*inter]
    gate_up = x @ gu
    gate, up = gate_up[..., :inter], gate_up[..., inter:]
    down = down if down.shape[0] == inter else down.t()  # -> [inter, H]
    return clamped_swiglu(gate, up) @ down


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def partial_rope(t, cos, sin):
    rd = cos.shape[-1]  # rotary_dim = cos width (matches HF apply_rotary_pos_emb)
    r, p = t[..., :rd], t[..., rd:]
    return torch.cat([r * cos + rotate_half(r) * sin, p], dim=-1)


def main():
    results = {}

    # 1. RMSNorm (gemma 1+w)
    m = M.MiniMaxM3VLRMSNorm(H).eval()
    m.weight.data.normal_(0, 0.2)
    x = torch.randn(1, S, H)
    results["RMSNorm"] = pcc(m(x), gemma_norm(x, m.weight))

    # 2. DenseMLP (clamped swigluoai)
    m = M.MiniMaxM3VLDenseMLP(cfg).eval()
    sd = m.state_dict()
    x = torch.randn(1, S, H) * 0.5
    results["DenseMLP"] = pcc(m(x), fused_ffn(x, sd["gate_up_proj.weight"], sd["down_proj.weight"], DENSE_INTER))

    # 3. SparseMoeBlock (router + routed experts + shared + routed_scaling)
    m = M.MiniMaxM3VLSparseMoeBlock(cfg).eval()
    sd = m.state_dict()
    gate_w, bias = sd["gate.weight"], sd["gate.e_score_correction_bias"]
    exp_gu, exp_down = sd["experts.gate_up_proj"], sd["experts.down_proj"]
    sh_gu, sh_down = sd["shared_experts.gate_up_proj.weight"], sd["shared_experts.down_proj.weight"]

    def moe_ref(x):
        scores = torch.sigmoid(x @ gate_w.t())
        _, idx = torch.topk(scores + bias, TOPK, dim=-1)
        tw = torch.gather(scores, 1, idx)
        tw = (tw / tw.sum(-1, keepdim=True)) * SCALE
        routed = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(TOPK):
                e = idx[t, j].item()
                routed[t] += tw[t, j] * fused_ffn(x[t : t + 1], exp_gu[e], exp_down[e], INTER).squeeze(0)
        return routed + fused_ffn(x, sh_gu, sh_down, SHARED_INTER)

    x = torch.randn(1, S, H) * 0.5
    results["SparseMoeBlock"] = pcc(m(x), moe_ref(x.view(-1, H)).view(1, S, H))

    # 4. Attention (per-head QK-norm before RoPE + partial RoPE + GQA)
    m = M.MiniMaxM3VLAttention(cfg, layer_idx=0).eval()
    sd = m.state_dict()
    rotary = M.MiniMaxM3VLRotaryEmbedding(cfg)
    x = torch.randn(1, S, H)
    cos, sin = rotary(x, torch.arange(S).unsqueeze(0))
    mask = torch.triu(torch.full((1, 1, S, S), float("-inf")), diagonal=1)
    with torch.no_grad():
        hf_out, _ = m(x, position_embeddings=(cos, sin), attention_mask=mask)

    def attn_ref(x):
        B = x.shape[0]
        q = gemma_norm((x @ sd["q_proj.weight"].t()).view(B, S, NQ, HD), sd["q_norm.weight"]).transpose(1, 2)
        k = gemma_norm((x @ sd["k_proj.weight"].t()).view(B, S, NKV, HD), sd["k_norm.weight"]).transpose(1, 2)
        v = (x @ sd["v_proj.weight"].t()).view(B, S, NKV, HD).transpose(1, 2)
        c, s = cos[:, None], sin[:, None]
        q, k = partial_rope(q, c, s), partial_rope(k, c, s)
        rep = NQ // NKV
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        sc = (q @ k.transpose(-1, -2)) * (HD**-0.5) + mask
        o = (torch.softmax(sc, -1) @ v).transpose(1, 2).reshape(B, S, NQ * HD)
        return o @ sd["o_proj.weight"].t()

    results["Attention"] = pcc(hf_out, attn_ref(x))

    print("=" * 52)
    for k, v in results.items():
        print(f"  {k:16s} PCC = {v:.6f}  [{'PASS' if v >= 0.99 else 'FAIL'}]")
    print("=" * 52)
    allpass = all(v >= 0.99 for v in results.values())
    print("ORACLE PCC:", "ALL PASS" if allpass else "SOME FAILED")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
