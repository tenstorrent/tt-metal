# The fused chunk preprocessing on real inputs (layer 0 of a prefill, dumped to /dev/shm/jw_prep_in.pt by a debug run), each output against an fp64 torch reference.
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.gdn_prefill.chunk_prep import ChunkPrep


def reference(q, k, v, beta, g, alpha, scale):
    BH, T, K = q.shape
    C, NC = 128, T // 128
    q, k, v, beta, g = (t.double() for t in (q * scale, k, v, beta.reshape(BH, T, 1), g))
    qc, kc, vc = (t.reshape(BH, NC, C, -1) for t in (q, k, v))
    bc, gc = beta.reshape(BH, NC, C, 1), g.reshape(BH, NC, C)
    d = gc.cumsum(-1)
    dexp = d.clamp(-20, 0).exp().unsqueeze(-1)
    tril = torch.tril(torch.ones(C, C, dtype=torch.float64))
    LM = ((d.unsqueeze(-1) - d.unsqueeze(-2)) * tril).clamp(-20, 0).exp() * tril
    kk = (kc * bc) @ kc.transpose(-1, -2)
    M = kk * LM
    s = torch.diagonal(M, dim1=-2, dim2=-1).unsqueeze(-1)
    dinv = 1.0 / (1 + alpha * s)
    N = dinv * (M * torch.tril(torch.ones(C, C, dtype=torch.float64), -1))
    L_unit = torch.eye(C, dtype=torch.float64) + N
    blocks = [torch.linalg.inv(L_unit[..., 32 * b : 32 * b + 32, 32 * b : 32 * b + 32]) for b in range(4)]
    L_inv = torch.cat(blocks, -2)
    S = d[..., -1:]
    kdf = (S - d).clamp(-20, 0).exp().unsqueeze(-1)
    return {
        "L_unit": L_unit, "v_beta_sc": vc * bc * dinv, "k_bd_sc": kc * bc * dexp * dinv,
        "intra": (qc @ kc.transpose(-1, -2)) * LM, "q_decay": qc * dexp, "k_decay_t": (kc * kdf).transpose(-1, -2),
        "dl_exp": S.clamp(-20, 0).exp().unsqueeze(-1), "L_inv": L_inv,
    }  # fmt: skip


def test_chunk_prep_real(device):
    import os

    import pytest

    if not os.path.exists("/dev/shm/jw_prep_in.pt"):
        pytest.skip("no dumped inputs")
    x = torch.load("/dev/shm/jw_prep_in.pt")
    q, k, v, beta, g = x["q"], x["k"], x["v"], x["beta"], x["g"]
    BH, T, K = q.shape
    print(
        f"REAL shapes q {tuple(q.shape)} beta {tuple(beta.shape)} g {tuple(g.shape)} g range [{g.min():.3g}, {g.max():.3g}] beta range [{beta.min():.3g}, {beta.max():.3g}] |k| [{k.norm(dim=-1).min():.3g}, {k.norm(dim=-1).max():.3g}] finite {[torch.isfinite(t).all().item() for t in (q, k, v, beta, g)]}"
    )
    F = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    outs = ChunkPrep(device)(ttnn.multiply(F(q), K**-0.5), F(k), F(v), F(beta.reshape(BH, T, 1)), F(g), 0.25)
    ref = reference(q, k, v, beta, g, 0.25, K**-0.5)
    for (name, r), t in zip(ref.items(), outs):
        y = ttnn.to_torch(t).double().reshape(r.shape)
        fin = torch.isfinite(y)
        bad = (~fin).nonzero()[:3].tolist()
        rel = ((y[fin] - r[fin]).norm() / r[fin].norm()).item() if fin.any() else float("nan")
        print(
            f"REAL {name}: finite {fin.all().item()} first bad {bad} rel {rel:.4g} ref max {r.abs().max().item():.3g}"
        )
