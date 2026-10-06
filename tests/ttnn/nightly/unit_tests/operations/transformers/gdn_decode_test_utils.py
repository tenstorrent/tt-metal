# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test utilities for the fused GDN decode recurrence (ttnn.transformer.fused_recurrent_gated_delta_rule).

The reference semantics live in the op's registered golden (ttnn.get_golden_function(op), implemented in
ttnn/ttnn/operations/transformer_golden.py); this module holds what is not reference semantics: the model shapes, input
generation, the chained single-token helper, metrics with the accuracy gates, output digests, and FLA's naive
recurrence as the external oracle the golden is checked against."""

import hashlib
import os
import sys
from dataclasses import dataclass

import torch

from ttnn.operations.transformer_golden import l2_norm

# Accuracy gates: PCC, and max |delta| relative to max |reference| as a gross-error guard. The FPU reads fp32 operands
# truncated to tf32-class precision (9-10 mantissa bits), so the current kernel sits at up to 5e-3 relative error at
# every token count (1 to 12) while its PCC stays above 0.99999.
PCC_MIN = 0.9999
MAX_ABS_REL = 1e-2


@dataclass(frozen=True)
class Shape:
    name: str
    num_key_heads: int
    num_value_heads: int
    head_dim: int
    served_as: str
    supported_today: bool  # the current op: rank-3 g only, B * HV <= 110 cores


# Per-device head counts of the served configurations. K = V = head_dim.
SHAPES = {
    "qwen27b_tp4": Shape("qwen27b_tp4", 4, 12, 128, "Qwen3.6-27B TP 4", True),
    "qwen27b_tp8": Shape("qwen27b_tp8", 2, 6, 128, "Qwen3.6-27B TP 8", True),
    "qwen35b_a3b_tp4": Shape("qwen35b_a3b_tp4", 4, 8, 128, "Qwen3.6-35B-A3B TP 4", True),
    "qwen9b_tp1": Shape("qwen9b_tp1", 16, 32, 128, "Qwen3.5-9B single chip", True),
    # Kimi-K3's KDA has H == HV and a per-key decay vector (rank-4 g); the current op takes rank-3 g only, so this
    # entry is the GDN-shaped stand-in at the served head count until the KDA path lands.
    "kimi_k3_sp8tp4": Shape("kimi_k3_sp8tp4", 24, 24, 128, "Kimi-K3 SP 8 x TP 4 (GDN-shaped stand-in)", True),
}


def make_inputs(B, T, H, HV, K, V, *, seed, g_scale=2.0, normalize_qk=True, with_state=True, dtype=torch.float32):
    """Recurrence inputs in the layer's post-conv convention: q, k, v ~ N(0, 1), beta ~ U(0, 1) (a sigmoid output),
    g = -U(0, 1) * g_scale (log decay), state ~ 0.05 N(0, 1). With normalize_qk the q, k rows are L2-normalised
    (the golden module's l2_norm) as the current op expects them. Returns a dict of torch tensors, token-major [B, T, heads, D].
    """
    gen = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32)
    q, k = r(B, T, H, K), r(B, T, H, K)
    if normalize_qk:
        q, k = l2_norm(q), l2_norm(k)
    v = r(B, T, HV, V)
    beta = torch.rand(B, T, HV, generator=gen, dtype=torch.float32)
    g = -torch.rand(B, T, HV, generator=gen, dtype=torch.float32) * g_scale
    s0 = 0.05 * r(B, HV, K, V) if with_state else None
    out = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "initial_state": s0}
    return {n: (t.to(dtype) if t is not None else None) for n, t in out.items()}


def chained_decode(fn, q, k, v, g, beta, initial_state=None):
    """Run fn (op or golden, (q, k, v, g, beta, *, initial_state, output_final_state) -> (o, state)) one token at a
    time and stitch: o [B, T, HV, V], states [B, T, HV, K, V] (the state after each token)."""
    T = q.shape[1]
    state = initial_state
    outs, states = [], []
    for t in range(T):
        sl = slice(t, t + 1)
        o_t, state = fn(
            q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], initial_state=state, output_final_state=True
        )
        outs.append(o_t)
        states.append(state)
    return torch.cat(outs, dim=1), torch.stack(states, dim=1)


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    if torch.equal(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b / denom).item()) if denom > 0 else float("nan")


def metrics(ref, got):
    """PCC, max |delta| and max |delta| relative to max |ref| between a reference and a result."""
    ref64, got64 = ref.detach().to(torch.float64), got.detach().to(torch.float64)
    max_abs = (ref64 - got64).abs().max().item()
    scale = ref64.abs().max().item()
    return {"pcc": pcc(ref64, got64), "max_abs": max_abs, "max_abs_rel": max_abs / scale if scale > 0 else max_abs}


def assert_accuracy(ref, got, label, *, pcc_min=PCC_MIN, max_abs_rel=MAX_ABS_REL):
    m = metrics(ref, got)
    assert m["pcc"] >= pcc_min and m["max_abs_rel"] <= max_abs_rel, f"{label}: {m}"
    return m


def golden_digest(tensor):
    """sha256 of the fp32 bytes of a tensor: the compatibility golden of an output."""
    return hashlib.sha256(tensor.detach().to(torch.float32).contiguous().numpy().tobytes()).hexdigest()


def _vendored_fla_naive_recurrent_gated_delta_rule(
    q, k, v, beta, g, scale=None, initial_state=None, output_final_state=False
):
    """Vendored copy of FLA's naive_recurrent_gated_delta_rule (flash-linear-attention,
    fla/ops/gated_delta_rule/naive.py, MIT license, (c) Songlin Yang et al.). The external oracle of the golden."""
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


def fla_naive_recurrent_gated_delta_rule():
    """(function, source): FLA's naive_recurrent_gated_delta_rule from an installed `fla` package or the FLA_REPO
    checkout when importable ("fla"), else the vendored copy ("vendored"). FLA order (q, k, v, beta, g)."""
    try:
        from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

        return naive_recurrent_gated_delta_rule, "fla"
    except Exception:
        pass
    repo = os.environ.get("FLA_REPO")
    if repo and os.path.isdir(os.path.join(repo, "fla")):
        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

            return naive_recurrent_gated_delta_rule, "fla"
        except Exception:
            pass
    return _vendored_fla_naive_recurrent_gated_delta_rule, "vendored"
