# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FLA-naive GDN reference. Device op L2-norms q/k internally, so normalize here and pass beta in (0, 1).
g is log-decay (g<0). Falls back to a vendored naive copy when the FLA checkout is not importable."""
import os
import sys

import torch


def _load_fla_naive():
    """Import FLA naive_recurrent_gated_delta_rule, else a vendored fallback."""
    try:
        from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

        return naive_recurrent_gated_delta_rule
    except Exception:
        pass
    candidates = [
        os.environ.get("FLA_REPO"),
        "/home/ttuser/atupe/flash-linear-attention",
        os.path.expanduser("~/flash-linear-attention"),
    ]
    for p in candidates:
        if p and os.path.isdir(os.path.join(p, "fla")):
            if p not in sys.path:
                sys.path.insert(0, p)
            try:
                from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

                return naive_recurrent_gated_delta_rule
            except Exception:
                break
    return _vendored_naive_recurrent_gated_delta_rule


def _vendored_naive_recurrent_gated_delta_rule(
    q, k, v, beta, g, scale=None, initial_state=None, output_final_state=False
):
    """Vendored copy of FLA naive_recurrent_gated_delta_rule, used only when FLA is not importable."""
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


naive_recurrent_gated_delta_rule = _load_fla_naive()


def naive_recurrent_per_token_state(q, k, v, beta, g, scale=None, initial_state=None):
    """FLA-naive recurrence returning per-token states; q/k must already be L2-normalized."""
    qt, kt, vt, bt, gt = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    B, H, T, K, V = *kt.shape, vt.shape[-1]
    if scale is None:
        scale = 1 / (K**0.5)
    qt = qt * scale
    h = torch.zeros(B, H, K, V, dtype=torch.float32)
    if initial_state is not None:
        h = initial_state.to(torch.float32).clone()
    o = torch.zeros(B, H, T, V, dtype=torch.float32)
    states = torch.zeros(B, T, H, K, V, dtype=torch.float32)
    for i in range(T):
        b_q, b_k, b_v, b_beta = qt[:, :, i], kt[:, :, i], vt[:, :, i].clone(), bt[:, :, i]
        h = h * gt[:, :, i].exp()[..., None, None]
        b_v = b_v - (h * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
        states[:, i] = h
    return o.transpose(1, 2).contiguous(), states


def l2norm_fla(x, eps=1e-6):
    """FLA L2-norm over the last dim; matches the device op's rms_norm form."""
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def make_gdn_inputs(T, H=32, Dk=128, Dv=128, B=1, seed=0, g_scale=2.0):
    """Post-conv GDN inputs at Qwen3.6-27B dims: beta in (0, 1), g negative log-decay."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, Dk, generator=gen, dtype=torch.float32)
    k = torch.randn(B, T, H, Dk, generator=gen, dtype=torch.float32)
    v = torch.randn(B, T, H, Dv, generator=gen, dtype=torch.float32)
    beta = torch.rand(B, T, H, generator=gen, dtype=torch.float32)
    g = -torch.rand(B, T, H, generator=gen, dtype=torch.float32) * g_scale
    return q, k, v, beta, g


def pcc(a, b):
    """Pearson correlation of two tensors (flattened, double). Matches the repo's compute_pcc."""
    a = a.detach().reshape(-1).double()
    b = b.detach().reshape(-1).double()
    if torch.allclose(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if a.norm() == b.norm() else 0.0
    return (a @ b / denom).item()
