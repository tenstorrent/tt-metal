# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU validation of the FULL chunked-prefill pipeline math:
causal conv1d -> silu -> per-head L2norm(+q scale) -> head-expand(x3) ->
gated delta-rule recurrence -> gated RMSNorm.  Sequential reference vs the
factored chunked form the kernel implements.  This de-risks all host-precompute
+ kernel math before any device/C++ work.
"""
import torch

torch.manual_seed(1)

# Qwen3.6-27B DeltaNet dims
Hk, Hv, Dk, Dv, ck = 16, 48, 128, 128, 4
key_dim, value_dim = Hk * Dk, Hv * Dv
conv_dim = key_dim * 2 + value_dim
scale = Dk ** -0.5
C = 32


def l2n(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def conv_silu_l2(qkv_proj, conv_w):
    """qkv_proj [S,conv_dim] -> post-conv silu l2-normed q,k [S,Hv,Dk], v [S,Hv,Dv]."""
    S = qkv_proj.shape[0]
    xpad = torch.cat([torch.zeros(ck - 1, conv_dim), qkv_proj], 0)        # causal left-pad
    conv = sum(conv_w[:, j] * xpad[j:j + S] for j in range(ck))          # [S,conv_dim]
    conv = torch.nn.functional.silu(conv)
    q, k, v = torch.split(conv, [key_dim, key_dim, value_dim], dim=-1)
    q = (l2n(q.reshape(S, Hk, Dk)) * scale)
    k = l2n(k.reshape(S, Hk, Dk))
    v = v.reshape(S, Hv, Dv)
    q = q.repeat_interleave(Hv // Hk, dim=1)                              # expand k-heads ->v-heads
    k = k.repeat_interleave(Hv // Hk, dim=1)
    return q, k, v


def gated_rmsnorm(o, z, w, eps=1e-6):
    # o,z [..,Dv]; w [Dv]
    n = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + eps) * w
    return n * torch.nn.functional.silu(z)


def seq_ref(q, k, v, z, beta, decay, normw):
    """Ground truth: per-token gated delta recurrence + gated RMSNorm, per v-head."""
    S = q.shape[0]
    out = torch.zeros(S, Hv, Dv)
    for h in range(Hv):
        Smat = torch.zeros(Dk, Dv)
        for t in range(S):
            Smid = decay[t, h] * Smat
            mem = k[t, h] @ Smid
            delta = beta[t, h] * (v[t, h] - mem)
            Smat = Smid + torch.outer(k[t, h], delta)
            out[t, h] = q[t, h] @ Smat
    return gated_rmsnorm(out, z, normw)


def chunk_factored(q, k, v, z, beta, decay, normw, bf16=False):
    """Factored chunked form (kernel-faithful), per v-head."""
    def b16(x):
        return x.bfloat16().float() if bf16 else x
    S = q.shape[0]
    out = torch.zeros(S, Hv, Dv)
    for h in range(Hv):
        Smat = torch.zeros(Dk, Dv)
        for c0 in range(0, S, C):
            c1 = min(c0 + C, S)
            n = c1 - c0
            qc, kc, vc = q[c0:c1, h], k[c0:c1, h], v[c0:c1, h]
            bc, gc = beta[c0:c1, h], decay[c0:c1, h]
            d = torch.cumprod(gc, 0)
            dinv = 1.0 / d
            Kdec = b16(bc[:, None] * d[:, None] * kc)
            Kinv = b16(dinv[:, None] * kc)
            Qd = b16(d[:, None] * qc)
            A = torch.tril(b16(Kdec @ Kinv.T), -1)
            kS0 = kc @ Smat
            RHS = bc[:, None] * (vc - d[:, None] * kS0)
            U = torch.linalg.solve(torch.eye(n) + A, RHS)
            M = torch.tril(b16(Qd @ Kinv.T), 0)
            out[c0:c1, h] = d[:, None] * (qc @ Smat) + M @ U
            Smat = d[-1] * Smat + d[-1] * (Kinv.T @ U)
    return gated_rmsnorm(out, z, normw)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


if __name__ == "__main__":
    conv_w = torch.randn(conv_dim, ck) * 0.3
    normw = torch.randn(Dv) * 0.1 + 1.0
    for S in [64, 251, 1024]:
        qkv_proj = torch.randn(S, conv_dim) * 0.5
        z = torch.randn(S, Hv, Dv)
        beta = torch.sigmoid(torch.randn(S, Hv))
        decay = torch.exp(-torch.exp(torch.randn(S, Hv) * 0.4)
                          * torch.nn.functional.softplus(torch.randn(S, Hv)))
        q, k, v = conv_silu_l2(qkv_proj, conv_w)
        ref = seq_ref(q, k, v, z, beta, decay, normw)
        cf = chunk_factored(q, k, v, z, beta, decay, normw, bf16=False)
        cb = chunk_factored(q, k, v, z, beta, decay, normw, bf16=True)
        print(f"S={S:5d}: full-pipeline PCC fp32={pcc(ref, cf):.8f}  bf16={pcc(ref, cb):.8f}  "
              f"maxabs(bf16)={(ref - cb).abs().max():.2e}")
