# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent FP32 CPU reference of one Qwen3.5 / 3.6 / 3.8 Gated DeltaNet layer (prefill, streaming state).

Pure torch, no ttnn. This is the one GDN oracle: the GDN baseline harness (``qwen36/tests/gdn_baseline``) and the
GDN-on-KDA tests both import it. Math follows transformers ``Qwen3_5GatedDeltaNet`` (qwen3_5 and qwen3_5_moe share it):
fused ``in_proj_qkv`` -> depthwise causal conv (kernel 4, no bias) + SiLU -> q | k | v; ``beta = sigmoid(b)``;
``g = -exp(A_log) * softplus(a + dt_bias)`` per V head; q/k L2-normalized (eps 1e-6), q scaled by ``K^-0.5``,
V head ``j`` reads K head ``j // (Nv / Nk)``; token-by-token delta rule with an FP32 ``[Nv, K, V]`` state;
gated RMSNorm (raw weight, norm before gate, ``silu(z)``, or ``sigmoid(z)`` for ``output_gate_activation`` sigmoid as
transformers ``qwen4_exp``); ``out_proj``. Everything is computed in FP32 from the
stored (bf16) weights, without the intermediate bf16 roundings of the HF module. The recurrence is the naive
per-token form, not the chunked algorithm under test.

Weights use the canonical layer-local schema of ``weights.py`` (checkpoint ``linear_attn.*`` names without the
prefix). ``GDNReferenceState`` carries what a following chunk needs: the last ``K - 1`` raw (pre-conv) qkv rows in
HF channel order and the recurrent state. Running chunks one after another with the returned state equals one pass
over their concatenation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.reference.gdn.weights import GDN_WEIGHT_NAMES, validate_gdn_weights


@dataclass
class GDNReferenceState:
    conv: torch.Tensor  # [K - 1, conv_dim] fp32, raw qkv rows preceding the next chunk (HF channel order)
    recurrent: torch.Tensor  # [Nv, K, V] fp32

    @classmethod
    def zeros(cls, config: GDNConfig) -> "GDNReferenceState":
        return cls(
            conv=torch.zeros(config.conv_kernel_size - 1, config.conv_dim),
            recurrent=torch.zeros(config.num_value_heads, config.head_k_dim, config.head_v_dim),
        )


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def delta_rule_recurrence(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive gated delta rule. q, k [T, H, K] (normalized, q scaled); v [T, H, V]; g, beta [T, H]; state [H, K, V]."""
    s = state.clone()
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], dtype=torch.float32)
    decay = g.exp()
    for t in range(q.shape[0]):
        s.mul_(decay[t, :, None, None])
        kv_mem = torch.bmm(k[t].unsqueeze(1), s).squeeze(1)  # [H, V] = k_t^T S
        delta = (v[t] - kv_mem) * beta[t, :, None]
        s.baddbmm_(k[t].unsqueeze(2), delta.unsqueeze(1))  # S += k_t delta^T
        out[t] = torch.bmm(q[t].unsqueeze(1), s).squeeze(1)
    return out, s


def gdn_forward_reference(
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    config: GDNConfig,
    state: GDNReferenceState | None = None,
) -> tuple[torch.Tensor, GDNReferenceState]:
    """One chunk of the GDN layer. hidden_states [T, hidden] -> (out [T, hidden] fp32, state after the chunk)."""
    if hidden_states.ndim != 2 or hidden_states.shape[-1] != config.hidden_size:
        raise ValueError(f"hidden_states shape {tuple(hidden_states.shape)} must be [T, {config.hidden_size}]")
    validate_gdn_weights(weights, config)
    if state is None:
        state = GDNReferenceState.zeros(config)
    w = {name: weights[name].float() for name in GDN_WEIGHT_NAMES}
    xf = hidden_states.float()
    T = xf.shape[0]
    Nk, Nv, Dk, Dv = config.num_key_heads, config.num_value_heads, config.head_k_dim, config.head_v_dim
    kc = config.conv_kernel_size

    qkv = xf @ w["in_proj_qkv.weight"].T  # [T, C]
    z = xf @ w["in_proj_z.weight"].T  # [T, Nv*Dv]
    b = xf @ w["in_proj_b.weight"].T  # [T, Nv]
    a = xf @ w["in_proj_a.weight"].T  # [T, Nv]

    padded = torch.cat([state.conv, qkv], dim=0)  # [K-1+T, C]
    taps = w["conv1d.weight"][:, 0, :]  # [C, K]; tap j multiplies row t - (K-1) + j
    conv = sum(padded[j : j + T] * taps[:, j] for j in range(kc))
    conv = F.silu(conv)
    q = conv[:, : config.q_dim].reshape(T, Nk, Dk)
    k = conv[:, config.q_dim : config.q_dim + config.k_dim].reshape(T, Nk, Dk)
    v = conv[:, config.q_dim + config.k_dim :].reshape(T, Nv, Dv)

    beta = torch.sigmoid(b)
    g = -w["A_log"].exp() * F.softplus(a + w["dt_bias"])
    q = _l2norm(q) * Dk**-0.5
    k = _l2norm(k)
    q = q.repeat_interleave(config.group, dim=1)
    k = k.repeat_interleave(config.group, dim=1)

    o, recurrent = delta_rule_recurrence(q, k, v, g, beta, state.recurrent)

    normed = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + config.norm_eps) * w["norm.weight"]
    activation = F.silu if config.output_gate_activation == "silu" else torch.sigmoid
    gated = (normed * activation(z.reshape(T, Nv, Dv))).reshape(T, Nv * Dv)
    out = gated @ w["out_proj.weight"].T
    new_conv = padded[-(kc - 1) :].clone()
    return out, GDNReferenceState(conv=new_conv, recurrent=recurrent)
