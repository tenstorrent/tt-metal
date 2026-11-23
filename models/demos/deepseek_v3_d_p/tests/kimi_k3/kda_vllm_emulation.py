# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU emulation of vLLM's Kimi-K3 KDA prefill precision (FlashKDA), for the FP64 chain's KDA-precision arms.

vLLM's K3 fork that recorded the K3.1 trace is not available here; the rounding points below follow its public
documentation and upstream vLLM's KDA layer (``vllm/model_executor/layers/kda.py``, Kimi-Linear):

* vLLM blog 2026-07-27 (Kimi K3): prefill runs FlashKDA.
* FlashKDA README / v1 deep dive: ``q``, ``k``, ``v``, the gate ``g`` (before activation) and ``beta`` (logits) are
  bf16; ``A_log`` and ``dt_bias`` fp32 with ``lower_bound``; CHUNK = 16; bf16 HMMA with fp32 accumulation; the
  intra-chunk inverse seeded and solved in fp32; q/k L2 normalization and the gate activation in the kernel.
* Upstream vLLM: projections in the model dtype (bf16 out); ``causal_conv1d_fn`` with fp32 weights and bf16 out;
  the recurrent state cached in fp32 between prefill steps (``kda_state_dtype``); ``FusedRMSNormGated`` on the
  bf16 core output.

Emulated, per KDA layer (``low`` = round to bf16; every product accumulates in fp32):

    x = low(input) ; q, k, v, g, beta logits, output gate = low(x W^T) (f_a then f_b for g)
    q, k, v = low(silu(conv(.)))           ; gate = lower_bound * sigmoid(exp(A_log) (g + dt_bias)), fp32
    beta = sigmoid(beta logits), fp32      ; q, k = low(l2norm(q)), low(l2norm(k))
    per chunk of 16 (WY form, decay G = in-chunk cumsum of gate, fp32 state S at the chunk start):
        T = low((I + beta_i low(k_i e^G_i) . low(k_j e^-G_j) [j < i])^-1)
        w = low(T low(beta k e^G)) ; u = low(T low(beta v)) ; v_new = low(u - w low(S))
        o = low(scale (low(q e^G) low(S) + low(Aqk) v_new)),  Aqk_ij = low(q_i e^G_i) . low(k_j e^-G_j), j <= i
        S = e^G_last S + low(k e^(G_last - G))^T v_new                       (fp32)
    out = low(o_proj(low(rmsnorm(o) w sigmoid(output gate))))

This is a model of the documented rounding points, not FlashKDA's exact instruction sequence (its block-merge
order inside the inverse, ``ex2.approx`` and the tensor-parallel shard reductions are not reproduced). With
``low`` the identity and ``dtype=float64`` the same code is the exact KDA layer, which the contract test checks
against ``kda_forward_reference``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState

CHUNK = 16  # FlashKDA v1 chunk length


def bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 (nearest even) and keep the working dtype."""
    return x.to(torch.bfloat16).to(x.dtype)


def kda_flashkda_emulation(
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    config: KDAConfig,
    state: KDAReferenceState | None = None,
    *,
    low: Callable[[torch.Tensor], torch.Tensor] = bf16,
    dtype: torch.dtype = torch.float32,
    chunk: int = CHUNK,
) -> tuple[torch.Tensor, KDAReferenceState]:
    """One KDA layer over ``hidden_states`` ``[1, T, hidden]`` with vLLM/FlashKDA rounding (module docstring),
    computing in ``dtype``. The chunk grid starts at this call's first row (vLLM's prefill steps and the chain's
    segments are multiples of ``chunk``). Returns the output ``[1, T, hidden]`` and the carried state (recurrent in
    ``dtype``; convolution histories hold the rounded projections)."""
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(f"hidden_states shape {tuple(hidden_states.shape)} must be [1, T, hidden]")
    if config.gate_lower_bound is None or not config.use_full_rank_gate:
        raise ValueError("the emulation covers Kimi-K3's bounded gate and full-rank output gate only")
    H, K, V = config.num_heads, config.head_k_dim, config.head_v_dim
    w = {n: t.to(dtype) for n, t in weights.items()}
    x = low(hidden_states[0].to(dtype))
    T = x.shape[0]

    def lin(a: torch.Tensor, name: str) -> torch.Tensor:
        return low(a @ w[name].T)

    hist = config.conv_kernel_size - 1
    if state is None:
        zeros = lambda d: x.new_zeros(hist, d)
        state = KDAReferenceState(
            x.new_zeros(1, H, K, V), zeros(config.q_dim)[None], zeros(config.k_dim)[None], zeros(config.v_dim)[None]
        )

    def conv(raw: torch.Tensor, name: str, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.cat([history[0].to(dtype), raw])
        out = F.conv1d(window.T[None], w[name], groups=raw.shape[1])[0].T
        return low(F.silu(out)), window[-hist:][None].clone()

    q, q_state = conv(lin(x, "q_proj.weight"), "q_conv1d.weight", state.q_convolution)
    k, k_state = conv(lin(x, "k_proj.weight"), "k_conv1d.weight", state.k_convolution)
    v, v_state = conv(lin(x, "v_proj.weight"), "v_conv1d.weight", state.v_convolution)
    g_raw = low(lin(x, "f_a_proj.weight") @ w["f_b_proj.weight"].T).reshape(T, H, K)
    gate = config.gate_lower_bound * torch.sigmoid(
        w["A_log"].reshape(1, H, 1).exp() * (g_raw + w["dt_bias"].reshape(1, H, K))
    )
    beta = torch.sigmoid(lin(x, "b_proj.weight"))  # [T, H]
    output_gate = lin(x, "g_proj.weight").reshape(T, H, V)

    def l2(t: torch.Tensor) -> torch.Tensor:
        return t * torch.rsqrt(t.square().sum(-1, keepdim=True) + 1e-6)

    q, k = low(l2(q.reshape(T, H, K))), low(l2(k.reshape(T, H, K)))
    v = v.reshape(T, H, V)
    scale = K**-0.5

    S = state.recurrent[0].to(dtype).clone()  # [H, K, V]
    out = x.new_empty(T, H, V)
    eye = torch.eye(chunk, dtype=dtype)
    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        n = hi - lo
        # per chunk, heads first: [H, n, .]
        qc, kc, vc = (t[lo:hi].transpose(0, 1) for t in (q, k, v))
        bc = beta[lo:hi].T.unsqueeze(-1)  # [H, n, 1]
        G = gate[lo:hi].transpose(0, 1).cumsum(1)  # [H, n, K], fp32 in-chunk log decay
        eg, egi = G.exp(), (-G).exp()
        kd, ki, qd = low(kc * eg), low(kc * egi), low(qc * eg)
        strict = torch.tril(torch.ones(n, n, dtype=torch.bool), -1)
        akk = (bc * (kd @ ki.transpose(1, 2))).masked_fill(~strict, 0)
        tinv = low(torch.linalg.solve_triangular(eye[:n, :n] + akk, eye[:n, :n].expand(H, n, n), upper=False))
        wc = low(tinv @ low(bc * kd))
        uc = low(tinv @ low(bc * vc))
        s_op = low(S)
        v_new = low(uc - wc @ s_op)
        aqk = low((qd @ ki.transpose(1, 2)).masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool)), 0))
        out[lo:hi] = low(scale * (qd @ s_op + aqk @ v_new)).transpose(0, 1)
        g_last = G[:, -1:]  # [H, 1, K]
        S = g_last.transpose(1, 2).exp() * S + low(kc * (g_last - G).exp()).transpose(1, 2) @ v_new

    o = out.reshape(T, H, V)
    o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + config.norm_eps)
    o = low(o * w["o_norm.weight"].reshape(-1) * torch.sigmoid(output_gate)).reshape(T, config.v_dim)
    result = lin(o, "o_proj.weight")
    return result[None], KDAReferenceState(S[None], q_state, k_state, v_state)
