# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared FLA-naive ground-truth reference + input/PCC helpers for the fused GDN kernels.

The PCC oracle for both the single-token decode kernel and the multi-token verify kernel
is FLA's ``naive_recurrent_gated_delta_rule`` (the recurrent gated-delta-rule math, the exact
form the vLLM ``fused_sigmoid_gating_delta_rule_update`` implements). We import the real
function from the flash-linear-attention checkout when available (set ``FLA_REPO`` or drop it
at the default path) and otherwise fall back to a vendored byte-for-byte copy so the tests run
anywhere.

Contract note (must match the device kernels):
  * FLA naive does NOT L2-normalize q/k and does NOT sigmoid beta — those are the layer's job
    (in FLA, ``use_qk_l2norm_in_kernel`` / ``use_beta_sigmoid_in_kernel``). Our device op does the
    L2-norm + query scale internally, so to compare against naive we L2-normalize q/k here
    (``l2norm_fla``) and pass beta already in (0,1) and g already as log-decay (g<0, decay=exp(g)).
  * scale defaults to Dk**-0.5, applied to q AFTER the L2-norm (matches gdn/tp.py and FLA).
"""
import os
import sys

import torch


def _load_fla_naive():
    """Import fla.ops.gated_delta_rule.naive.naive_recurrent_gated_delta_rule (real FLA), trying
    an installed package, then FLA_REPO / known checkout paths, then a vendored fallback."""
    try:
        from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

        return naive_recurrent_gated_delta_rule
    except Exception:
        pass
    candidates = [
        os.environ.get("FLA_REPO"),
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
    """Vendored byte-for-byte copy of FLA's naive_recurrent_gated_delta_rule
    (flash-linear-attention/fla/ops/gated_delta_rule/naive.py, MIT license, © Songlin Yang et al.).
    Used only when the real FLA checkout is not importable."""
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
    """FLA-naive recurrence that ALSO returns the state after every token — the ground truth for
    the multi-token verify kernel's per-token state fan-out (mirrors FLA gdn2's per-token store).

    Returns o [B,T,H,V], states [B,T,H,K,V] (state AFTER absorbing token t), final_state == states[:,-1].
    Same math as naive_recurrent_gated_delta_rule; q/k are expected already L2-normalized (see module note).
    """
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
    """FLA in-kernel L2-norm: x / sqrt(sum(x^2) + eps) over the last dim. Matches l2_norm_ttnn
    (rms_norm(x, eps/K) * K**-0.5) used by recurrent_gated_delta_rule_decode_ttnn."""
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def make_gdn_inputs(T, H=32, Dk=128, Dv=128, B=1, seed=0, g_scale=2.0):
    """Post-conv, post-GQA-expand GDN recurrence inputs at real Qwen3.6-27B dims.

    q/k/v [B,T,H,D]; beta [B,T,H] in (0,1) (sigmoid range); g [B,T,H] negative log-decay
    (decay = exp(g) in (0,1)). Convention matches test_gdn_chunk_recurrent_parity.py.
    """
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
