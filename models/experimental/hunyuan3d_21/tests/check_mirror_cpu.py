# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU check of the port's host-side rewrites against the upstream blocks, with real weights, fp32:
  - fix_qkv_layout + a standard per-head split == upstream interleaved attention (block 0)
  - skip_linear split into two halves == the concat form (block 11)
  - top-2 by threshold, dense expert sum == upstream moe_infer (block 20)
  - timestep token at the end of the sequence == at the front (blocks 0, 11, 20 chained)
usage: python -I check_mirror_cpu.py HY3DSHAPE_REPO CKPT CACHE TT_METAL_ROOT
"""
import sys

import torch
import torch.nn.functional as F

repo, ckpt, cache, root = sys.argv[1:5]
sys.path.insert(0, repo)
sys.path.insert(0, root)
from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTBlock  # noqa: E402
from models.experimental.hunyuan3d_21.reference import dit_host as host  # noqa: E402

torch.manual_seed(0)
state = host.load_dit_state(ckpt, cache)
fixed = host.fix_qkv_layout(state)


def upstream_block(i):
    b = HunYuanDiTBlock(
        hidden_size=2048,
        c_emb_size=2048,
        num_heads=16,
        text_states_dim=1024,
        qk_norm=True,
        norm_layer=torch.nn.LayerNorm,
        qk_norm_layer=torch.nn.RMSNorm,
        skip_connection=host.has_skip(i),
        qkv_bias=False,
        use_moe=host.is_moe_layer(i),
        num_experts=8,
        moe_top_k=2,
    )
    pre = f"blocks.{i}."
    b.load_state_dict({k[len(pre) :]: v for k, v in state.items() if k.startswith(pre)})
    return b.eval()


def ln(x, p):
    return F.layer_norm(x, (x.shape[-1],), fixed[p + ".weight"], fixed[p + ".bias"], host.NORM_EPS)


def attn(x, ctx, p):
    q = F.linear(x, fixed[p + ".to_q.weight"])
    src = x if ctx is None else ctx
    k = F.linear(src, fixed[p + ".to_k.weight"])
    v = F.linear(src, fixed[p + ".to_v.weight"])
    h = lambda t: t.reshape(t.shape[0], t.shape[1], 16, 128).transpose(1, 2)
    q, k, v = h(q), h(k), h(v)
    q = F.rms_norm(q, (128,), fixed[p + ".q_norm.weight"], host.QK_NORM_EPS)
    k = F.rms_norm(k, (128,), fixed[p + ".k_norm.weight"], host.QK_NORM_EPS)
    o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
    return F.linear(o, fixed[p + ".out_proj.weight"], fixed[p + ".out_proj.bias"])


def ff(x, p):
    return F.linear(
        F.gelu(F.linear(x, fixed[p + "1.weight"], fixed[p + "1.bias"])), fixed[p + "2.weight"], fixed[p + "2.bias"]
    )


def moe(x, p):
    s = F.linear(x, fixed[p + ".gate.weight"]).softmax(-1)
    m1 = s.max(-1, keepdim=True).values
    m2 = torch.where(s >= m1, -1.0, s).max(-1, keepdim=True).values
    w = s * (s >= m2)
    y = ff(x, p + ".shared_experts.net.0.proj.weight"[:0] + p + ".shared_experts.fc") if False else None
    e = lambda q, t: F.linear(
        F.gelu(F.linear(t, fixed[q + ".net.0.proj.weight"], fixed[q + ".net.0.proj.bias"])),
        fixed[q + ".net.2.weight"],
        fixed[q + ".net.2.bias"],
    )
    y = e(p + ".shared_experts", x)
    for j in range(8):
        y = y + e(f"{p}.experts.{j}", x) * w[..., j : j + 1]
    return y


def mirror_block(i, x, ctx, skip=None):
    p = f"blocks.{i}"
    if host.has_skip(i):
        w = fixed[p + ".skip_linear.weight"]
        x = F.linear(skip, w[:, :2048], fixed[p + ".skip_linear.bias"]) + F.linear(x, w[:, 2048:])
        x = ln(x, p + ".skip_norm")
    x = x + attn(ln(x, p + ".norm1"), None, p + ".attn1")
    x = x + attn(ln(x, p + ".norm2"), ctx, p + ".attn2")
    h = ln(x, p + ".norm3")
    return x + (moe(h, p + ".moe") if host.is_moe_layer(i) else ff(h, p + ".mlp.fc"))


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1])


n, L = 513, 77  # short sequences: the checks are about layout, not length
x = torch.randn(1, n, 2048)
ctx = torch.randn(1, L, 1024)
skip = torch.randn(1, n, 2048)
with torch.no_grad():
    for i in (0, 11, 20):
        b = upstream_block(i)
        ref = b(x, None, ctx, skip_value=skip if host.has_skip(i) else None)
        # mirror with the timestep token (row 0 upstream) moved to the end
        perm = torch.cat([torch.arange(1, n), torch.tensor([0])])
        out = mirror_block(i, x[:, perm], ctx, skip[:, perm] if host.has_skip(i) else None)[:, perm.argsort()]
        print(
            f"block {i:2d} skip={host.has_skip(i)} moe={host.is_moe_layer(i)}  pcc={pcc(out, ref):.8f}  max_abs={float((out - ref).abs().max()):.3e}"
        )
